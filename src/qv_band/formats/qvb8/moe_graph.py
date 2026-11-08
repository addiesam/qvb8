"""CUDA-graph replay for MoE blocks at decode.

WHY. Once the routed experts run as one fused kernel (QVBandExpertsV8's fused
path) a decode step on the 35B is no longer bound by the GPU: kernels take
~23 ms per step but the step takes ~34 ms, because the CPU spends the rest
issuing ~4,500 tiny kernels through Python. A MoE block -- router, shared
expert, shared-expert gate, the fused experts, and the final add -- is ~25
launches and ~0.24 ms of CPU time per layer, 40 layers per step. It is a pure
tensor program with no host synchronisation and static weight addresses, which
is exactly what a CUDA graph replays at a fraction of the cost.

HOW. `install_moe_graphs(model)` wraps each MoE block's forward. For a block
call with M <= QVBandExpertsV8.FUSED_MAX_TOKENS rows the wrapper keeps one
captured graph per (M, dtype): on first use it warms up eagerly (which also
builds the experts' stacked layout, an allocation that must not happen inside
capture), then captures; afterwards a call is `static_in.copy_(x); graph.replay();
out = static_out.clone()`. Larger M (prefill), a non-CUDA device, or ANY failure
during warm-up/capture falls back to the eager forward permanently for that
block, so this can only ever make a step faster or leave it as it was.

Replay is numerically identical to eager execution (same kernels, same
arguments). QVB8_MOE_GRAPH=0 disables it.
"""
from __future__ import annotations

import os

import torch


_POOLS: dict = {}


def pool_for(device) -> tuple:
    """One CUDA-graph memory pool PER CARD. A pool handle belongs to the card it is first used on: sharing one
    across cards (layers split over two GPUs) invalidates the other card's captures or leaves them empty."""
    d = torch.device(device)
    idx = d.index if d.index is not None else torch.cuda.current_device()
    if idx not in _POOLS:
        with torch.cuda.device(idx):
            _POOLS[idx] = torch.cuda.graph_pool_handle()
    return _POOLS[idx]


_CAP_STREAMS: dict = {}


def capture_stream_for(device) -> "torch.cuda.Stream":
    """A capture stream PER CARD. torch.cuda.graph() without stream= reuses ONE class-level stream made on whichever
    card captured first; a later capture on the other card then records nothing (an empty graph)."""
    d = torch.device(device)
    idx = d.index if d.index is not None else torch.cuda.current_device()
    if idx not in _CAP_STREAMS:
        _CAP_STREAMS[idx] = torch.cuda.Stream(device=idx)
    return _CAP_STREAMS[idx]


class _BlockGraphs:
    def __init__(self, block, max_tokens: int, pool=None):
        self.block = block
        self.orig = block.forward
        self.max_tokens = max_tokens
        self.pool = pool                # shared capture pool (None: one per graph)
        self.graphs: dict = {}
        self.disabled = False

    def __call__(self, hidden_states: torch.Tensor):
        if (self.disabled or not hidden_states.is_cuda
                or hidden_states.dim() != 3):
            return self.orig(hidden_states)
        b, s, h = hidden_states.shape
        if b * s > self.max_tokens or torch.cuda.is_current_stream_capturing():
            return self.orig(hidden_states)
        key = (b, s, hidden_states.dtype)
        ent = self.graphs.get(key)
        if ent is None:
            ent = self._capture(hidden_states, key)
            if ent is None:
                return self.orig(hidden_states)
        graph, static_in, static_out = ent
        static_in.copy_(hidden_states)
        graph.replay()
        return static_out.clone()

    def _capture(self, x: torch.Tensor, key):
        # Capture on the card that holds x, not the current device: with layers split over two cards
        # (QV_PIPELINE_SPLIT) the current device is cuda:0 while half the MLPs live on cuda:1, and a capture begun
        # on cuda:0 records nothing -- an EMPTY graph whose replay returns the warm-up output for every later call.
        with torch.cuda.device(x.device):
            return self._capture_here(x, key)

    def _capture_here(self, x: torch.Tensor, key):
        try:
            static_in = torch.empty_like(x)
            static_in.copy_(x)
            # Eager warm-up on a side stream: builds the stacked expert layout,
            # primes the allocator and any lazily-created state.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.orig(static_in)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            pool = pool_for(x.device) if self.pool == "per-device" else self.pool
            # thread_local: the two-card prefill runs one worker thread per card; a global-mode capture forbids the OTHER
            # thread from synchronizing (cudaErrorStreamCaptureUnsupported in pipe_prefill).
            with torch.cuda.graph(graph, pool=pool, stream=capture_stream_for(x.device), capture_error_mode="thread_local"):
                static_out = self.orig(static_in)
            ent = (graph, static_in, static_out)
            self.graphs[key] = ent
            return ent
        except Exception as exc:                      # unsupported op, OOM, ...
            self.disabled = True
            self.graphs.clear()
            if os.environ.get("QVB8_MOE_GRAPH_VERBOSE"):
                print(f"[moe_graph] capture failed, using eager: "
                      f"{type(exc).__name__}: {exc}", flush=True)
            torch.cuda.synchronize(x.device)
            return None


def install_moe_graphs(model) -> int:
    """Wrap every MoE block of `model` (blocks whose `.experts` is a
    QVBandExpertsV8). Returns how many were wrapped; 0 for dense models, so a
    dense model such as the 27B is left untouched."""
    if os.environ.get("QVB8_MOE_GRAPH", "1") == "0" or not torch.cuda.is_available():
        return 0
    from qv_band.formats.qvb8.backends_experts import QVBandExpertsV8
    n = 0
    for mod in model.modules():
        ex = getattr(mod, "experts", None)
        if not isinstance(ex, QVBandExpertsV8) or getattr(mod, "_qvb8_moe_graph", False):
            continue
        g = _BlockGraphs(mod, QVBandExpertsV8.FUSED_MAX_TOKENS)
        mod.forward = g
        mod._qvb8_moe_graph = True
        n += 1
    return n


MLP_GRAPH_MAX_TOKENS = 16      # decode (1) and a speculative verify (8 drafts + 1, two requests 16)


def install_mlp_graphs(model) -> int:
    """Dense models: replay each decoder layer's MLP as a CUDA graph at decode.

    The same CPU-bound decode the MoE path fixes, on a dense 27B: a decode
    step is ~500 packed-linear calls plus (tensor parallel) their all-gathers,
    and the GPU idles while Python issues them (measured 2x V100: each rank at
    100% of one core, GPUs 53-75% busy). An MLP -- gate/up, SwiGLU, down and
    their collectives -- is stateless with static weight addresses, so it
    captures cleanly; attention and Gated-DeltaNet carry Python-managed cache
    state and stay eager. Under tensor parallel every rank sees the same row
    counts in the same order, so both capture the same collectives together.
    Opt-in, QVB8_MLP_GRAPH=1: +17% decode on 2x V100 but it pins ~2 GB of local
    memory per card (and blocks the rest-trim), and on the dev GPU it measured ~10%
    SLOWER (DFlash decode, 27B). Returns how many MLPs were wrapped."""
    if os.environ.get("QVB8_MLP_GRAPH", "0") != "1" or not torch.cuda.is_available():
        return 0
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return 0
    from qv_band.formats.qvb8.backends_experts import QVBandExpertsV8
    # One pool for every MLP graph: 64 layers x up to 16 row counts with a
    # private pool each would hold gigabytes a 16 GB card cannot spare. Sharing
    # is safe here -- graphs replay one at a time and each output is cloned
    # straight after its replay.
    pool = "per-device"       # one shared pool per card (see pool_for): layers may be split over two GPUs
    n = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if (mlp is None or getattr(mlp, "_qvb8_moe_graph", False)
                or isinstance(getattr(mlp, "experts", None), QVBandExpertsV8)):
            continue
        mlp.forward = _BlockGraphs(mlp, MLP_GRAPH_MAX_TOKENS, pool=pool)
        mlp._qvb8_moe_graph = True
        n += 1
    return n


def install_lean_norms(model) -> int:
    """MoE-only: replace Qwen3_5MoeRMSNorm.forward (~9 eager launches: float,
    pow, mean, add, rsqrt, mul, weight.float, add, mul, cast) by fp32
    F.rms_norm with a cached (1 + weight) -- 3 launches. Same math up to fp32
    rounding order. QVB8_LEAN_NORM=1 enables (off by default: ~1% faster, shifts outputs); dense models are untouched."""
    if os.environ.get("QVB8_LEAN_NORM", "0") != "1":
        return 0
    import torch.nn.functional as F
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm
    except Exception:
        return 0

    def fwd(self, x):
        w = self.__dict__.get("_w1")
        if w is None or w.device != x.device:
            w = (1.0 + self.weight.float()).detach()
            self.__dict__["_w1"] = w
        return F.rms_norm(x.float(), (x.shape[-1],), w, self.eps).to(x.dtype)

    n = 0
    for mod in model.modules():
        if type(mod) is Qwen3_5MoeRMSNorm:
            mod.forward = fwd.__get__(mod)
            n += 1
    return n
