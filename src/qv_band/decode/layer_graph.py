"""CUDA-graph replay of WHOLE linear-attention decoder layers at verify / decode.

WHY: the V100 host (Xeon E5-2673 v4) issues ~3,700 kernel launches per speculative verify step of the 27B; the GPUs
were measured busy ~17% of the step (nsys, 2026-11-07: 63 ms of kernels in a ~360 ms step). moe_graph (the MLPs) and
proj_graph (the four input projections) cover a few hundred of them; the rest -- RMSNorms, the conv window, the gated
delta rule, the gated norm, out_proj, residual adds, dtype casts -- are ~50 small launches per Gated-DeltaNet layer,
x 48 layers. Here each of those decoder layers (input norm, token mixer, residual, post norm, MLP, residual) replays
as ONE graph.

Why it is safe: a linear-attention layer's cache state lives at static addresses that the forward updates IN PLACE
(transformers' LinearAttentionLayer.update_conv_state / update_recurrent_state copy_ "to keep the static address for
cudagraphs"). The graph is keyed on those addresses: if a state tensor is replaced (prefix reuse restores a snapshot by
assignment) the layer's graphs are dropped and re-captured. The one Python side effect of the forward -- the rollback
capture stashed on the cache layer (ring_snapshot: `_qv_ring_modern`, `_qv_ring_*`) -- is recorded at capture time and
re-assigned after every replay; its tensors are the graph's static buffers, refreshed by the replay itself.

Scope: batch 1, 2..MAX_ROWS new tokens (a verify step), a cache whose layer already has state. Anything else (prefill,
a first step, batched engine steps, record_past caches) runs eagerly. OPT-IN, QV_LAYER_GRAPH=1: correct (output identical)
but measured no gain on the dual V100 (0.227 vs 0.208 s per verify step) -- the launches it removes cost ~20 ms; the
rest of the step is Python work elsewhere.
"""
from __future__ import annotations

import os

import torch

MAX_ROWS = int(os.environ.get("QV_LAYER_GRAPH_MAX_ROWS", "16"))
_SIDE_PREFIX = "_qv_ring"          # cache-layer attributes the forward may set (ring_snapshot)
_CHECK = os.environ.get("QV_LAYER_GRAPH_CHECK", "0") == "1"


def _side_attrs(obj) -> dict:
    return {k: v for k, v in vars(obj).items() if k.startswith(_SIDE_PREFIX)}


class _LayerGraph:
    def __init__(self, layer, prev_forward):
        self.layer = layer
        self.prev = prev_forward
        self.idx = layer.linear_attn.layer_idx
        self.dev = next(layer.parameters()).device
        self.graphs: dict = {}
        self.seen: set = set()
        self.state_ptrs = None
        self.disabled = False

    def _eligible(self, x, past):
        if self.disabled or not torch.is_tensor(x) or not x.is_cuda or x.dim() != 3 or x.shape[0] != 1:
            return None
        if not (2 <= x.shape[1] <= MAX_ROWS) or torch.cuda.is_current_stream_capturing():
            return None
        if not getattr(self, "_told", False) and os.environ.get("QV_LAYER_GRAPH_DEBUG"):
            self._told = True
            print(f"[layer-graph debug] layer {self.idx}: x on {x.device}, layer param on {self.dev}", flush=True)
        if x.device != self.dev:
            # the first layer of each card gets its input from the other card / the host: the cross-card copy is
            # not ordered with the replay (measured: graph state off by 10-50), so those 2 of 48 layers stay eager
            return None
        from qv_band.kv.step_ctx import current as _step_ctx
        _ctx = _step_ctx()
        if past is None or getattr(past, "_ring", None) is not None or (_ctx is not None and _ctx.segs is not None):
            return None                                  # batched engine steps: per-slot hooks, not this path
        try:
            cl = past.layers[self.idx]
        except Exception:
            return None
        if getattr(cl, "record_past", False) or not past.has_previous_state(self.idx):
            return None
        cs, rs = getattr(cl, "conv_states", {}).get(0), getattr(cl, "recurrent_states", {}).get(0)
        if cs is None or rs is None:
            return None
        return cl, (cs.data_ptr(), rs.data_ptr())

    def __call__(self, hidden_states, *args, **kwargs):
        past = kwargs.get("past_key_values")
        el = self._eligible(hidden_states, past)
        if el is None:
            return self.prev(hidden_states, *args, **kwargs)
        cl, ptrs = el
        if ptrs != self.state_ptrs:                      # state tensors replaced: old graphs point at dead memory
            self.graphs.clear()
            self.seen.clear()
            self.state_ptrs = ptrs
        key = (tuple(hidden_states.shape), hidden_states.dtype, bool(getattr(past, "_qv_ring_capture", False)))
        ent = self.graphs.get(key)
        if ent is None:
            if key not in self.seen:                     # first sighting: eager (warms Triton / cuBLAS for the shape)
                self.seen.add(key)
                return self.prev(hidden_states, *args, **kwargs)
            ent = self._capture(hidden_states, args, kwargs, cl, key)
            if ent is None:
                return self.prev(hidden_states, *args, **kwargs)
        g, sin, sout, side = ent
        if _CHECK:
            return self._check(g, sin, sout, side, cl, hidden_states, args, kwargs, key)
        sin.copy_(hidden_states)
        g.replay()
        for k, v in side.items():                        # the forward's own side effects, pointing at static buffers
            setattr(cl, k, dict(v) if isinstance(v, dict) else v)
        return sout.clone()

    def _check(self, g, sin, sout, side, cl, x, args, kwargs, key):
        """QV_LAYER_GRAPH_CHECK=1: replay, then the eager forward from the same state; report differences, keep eager."""
        cs, rs = cl.conv_states[0], cl.recurrent_states[0]
        cs0, rs0 = cs.clone(), rs.clone()
        sin.copy_(x)
        g.replay()
        og, csg, rsg = sout.clone(), cs.clone(), rs.clone()
        cs.copy_(cs0)
        rs.copy_(rs0)
        oe = self.prev(x, *args, **kwargs)
        d = [(og.float() - oe.float()).abs().max().item(), (csg.float() - cs.float()).abs().max().item(),
             (rsg.float() - rs.float()).abs().max().item()]
        self._nchk = getattr(self, "_nchk", 0) + 1
        if max(d) > 1e-2 and self._nchk <= 3:
            print(f"[layer-graph check] layer {self.idx} {key[0]} call {self._nchk}: |out| {d[0]:.3g} |conv| {d[1]:.3g} "
                  f"|rec| {d[2]:.3g}  (|out| {oe.float().abs().max().item():.3g})", flush=True)
        return oe

    def _capture(self, x, args, kwargs, cl, key):
        from qv_band.formats.qvb8.moe_graph import capture_stream_for
        dev = x.device
        sin = torch.empty_like(x)
        sin.copy_(x)
        before = _side_attrs(cl)
        try:
            with torch.cuda.device(dev):
                s = capture_stream_for(dev)
                s.wait_stream(torch.cuda.current_stream(dev))
                g = torch.cuda.CUDAGraph()
                # PRIVATE pool (pool=None), not the shared per-card one: the rollback capture this forward leaves on
                # the cache layer stays alive until the step ends, and in a shared pool it can sit in memory that a
                # later replay of another graph (an attention layer's MLP graph) uses as scratch -- the rollback
                # then restores garbage state and the model degenerates into repetition.
                with torch.cuda.graph(g, pool=None, stream=s, capture_error_mode="thread_local"):
                    sout = self.prev(sin, *args, **kwargs)
                torch.cuda.current_stream(dev).wait_stream(s)
        except Exception as e:                           # not capturable here: eager for this layer from now on
            print(f"[layer-graph] layer {self.idx}: capture failed ({type(e).__name__}: {str(e)[:160]}); eager",
                  flush=True)
            torch.cuda.synchronize(dev)
            for k in list(_side_attrs(cl)):
                if k not in before:
                    delattr(cl, k)
            for k, v in before.items():
                setattr(cl, k, v)
            self.disabled = True
            return None
        if isinstance(sout, tuple):                      # some transformers versions return (hidden,) tuples
            self.disabled = True
            print(f"[layer-graph] layer {self.idx}: tuple output, eager", flush=True)
            return None
        side = _side_attrs(cl)
        ent = (g, sin, sout, side)
        self.graphs[key] = ent
        return ent


def install(model) -> int:
    """Wrap every linear-attention decoder layer of `model` (Qwen3.5 / 3.6 / 3.8 hybrids). Returns how many."""
    if os.environ.get("QV_LAYER_GRAPH", "0") != "1" or not torch.cuda.is_available():     # opt-in: measured no gain on V100 (2026-11-07)
        return 0
    layers = getattr(getattr(model, "model", None), "layers", None) or []
    n = 0
    for layer in layers:
        if getattr(layer, "linear_attn", None) is None or getattr(layer, "_qv_layer_graph", False):
            continue
        layer.forward = _LayerGraph(layer, layer.forward)
        layer._qv_layer_graph = True
        n += 1
    return n
