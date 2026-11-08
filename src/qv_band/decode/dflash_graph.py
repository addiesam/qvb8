"""CUDA-graph replay of one DFlash2 draft block (the dev GPU speed-up #1).

The eager draft block (DFlashDrafter._block_hidden + the greedy selector walk) is ~1000 small kernel launches
(5 layers x dynamic convs / norms / rope / attention, then 15 sequential selector steps) plus host syncs (the
candidate gather from the host codebooks, the packed embedding's per-token escape check). Measured ~115 ms per
block against ~26 ms to read the drafter's weights once: the gap is launch latency and synchronisation.

Here everything after the embedding is captured once and replayed with ONE launch:

  * positions are device scalars (ctx_start, off, n, start, last_token) written before each replay, so the
    graph never bakes in a step-dependent value;
  * the draft block's K/V go into the scratch room of the context buffers with index_copy_ at a device index,
    and attention runs over the WHOLE buffer with a mask that hides everything outside [off, off+n+bs) -- the
    same keys the eager path slices out, so drafts are the same up to fp reduction order (the target verifies
    every draft, so output tokens are unaffected either way);
  * the selector codebooks are gathered on the device (a shared device copy, ~2 x 51 MB on Ripple);
  * the mask-token rows of the block embedding are constant and cached; only last_token is embedded per step.

A graph is tied to the context buffers' addresses: when DFlashDrafter._reserve reallocates them (geometric
growth, so O(log context) times per request) the graph is captured again.

Greedy (temperature 0) DFlash v2 only; everything else keeps the eager path. Enabled on sm_80+ without tensor
parallelism, opt-in with QV_DRAFT_GRAPH=1 (measured on the dev GPU, Ripple 3e-3 drafter: 63.4 -> 59.8 ms at a 3k context, 71.2 -> 72.4 ms at 30k -- launch overhead is not where the draft time goes); the V100 / multi-GPU paths are untouched.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_POOL = None


def graph_enabled(d) -> bool:
    if os.environ.get("QV_DRAFT_GRAPH", "0") != "1" or d.version != 2 or d.device.type != "cuda":
        return False
    import torch.distributed as dist
    if dist.is_initialized() and dist.get_world_size() > 1:
        return False
    return torch.cuda.get_device_capability(d.device)[0] >= 8


class DraftGraph:
    def __init__(self, d):
        self.d = d
        self.graph = None
        self.key = None
        dev, bs = d.device, d.block_size
        self.scal = torch.zeros(5, dtype=torch.long, device=dev)     # ctx_start, off, n, start, last_token
        self.h_in = torch.zeros(bs, d.H, dtype=d.dtype, device=dev)
        self.out = None

    # -- shared, per-weights state (a dict copy.copy shares across forks) --------------------------------------
    def _shared(self):
        sh = self.d._graph_shared
        if "succ" not in sh:
            sh["succ"] = self.d.sel_succ.to(self.d.device)
            sh["pred"] = self.d.sel_pred.to(self.d.device)
        return sh

    def _mask_rows(self):
        sh = self.d._graph_shared
        if "mask_emb" not in sh:
            ids = torch.full((1,), self.d.mask_token_id, dtype=torch.long, device=self.d.device)
            e = self.d.embed(ids)
            self.d._target_dtype = e.dtype
            sh["mask_emb"] = (e * getattr(self.d, "embed_mult", 1.0)).to(self.d.dtype)[0]
        return sh["mask_emb"]

    # -- the captured body -----------------------------------------------------------------------------------
    def _body(self):
        from qv_band.decode.dflash_draft import _lin, _rms
        d, sh = self.d, self._shared()
        bs, dev = d.block_size, d.device
        ctx_start, off, n, start, tok = self.scal.unbind(0)
        cap = d._kb[0].shape[2]
        ar = torch.arange(bs, device=dev)
        q_pos = start + ar
        cos, sin = d._cos_sin(q_pos)
        j = torch.arange(cap, device=dev)
        k_pos = ctx_start + j - off
        valid = (j >= off) & (j < off + n + bs)
        dd = q_pos[:, None] - k_pos[None, :]
        masks = {}
        for causal, win in set(d.layer_attn):
            m = valid[None, :].expand(bs, cap)
            if causal:
                m = m & (dd >= 0)
            if win is not None:
                m = m & (dd < win)
                if not causal:
                    m = m & (-dd < win)
            masks[(causal, win)] = m[None, None]
        ones, zeros = torch.ones_like(cos), torch.zeros_like(cos)
        widx = off + n + ar
        h = self.h_in
        for i, L in enumerate(d.layers):
            res = h
            x = _rms(h, L["input_layernorm"], d.eps)
            dyn = None
            if L["attn_conv"] is not None:
                x, dyn = L["attn_conv"].prepare(x[None])
                x = x[0]
            q = _lin(x, L["self_attn.q_proj"]).view(bs, d.n_heads, d.hd)
            qc, qs = (ones, zeros) if L.get("nope") else (cos, sin)      # Ripple: NoPE draft layer (no rotary)
            q = d._rope(_rms(q, L["self_attn.q_norm"], d.eps).transpose(0, 1)[None], qc, qs)
            k, v = d._kv(L, x, cos, sin)
            d._kb[i].index_copy_(2, widx, k)
            d._vb[i].index_copy_(2, widx, v)
            a = d._attend(q, d._kb[i], d._vb[i], masks[d.layer_attn[i]], i)
            a = _lin(a[0].transpose(0, 1).reshape(bs, -1), L["self_attn.o_proj"])
            if dyn is not None:
                a = L["attn_conv"].finish(a[None], dyn)[0]
            h = res + a
            res = h
            x = _rms(h, L["post_attention_layernorm"], d.eps)
            dyn = None
            if L["mlp_conv"] is not None:
                x, dyn = L["mlp_conv"].prepare(x[None])
                x = x[0]
            x = _lin(F.silu(_lin(x, L["mlp.gate_proj"])) * _lin(x, L["mlp.up_proj"]), L["mlp.down_proj"])
            if dyn is not None:
                x = L["mlp_conv"].finish(x[None], dyn)[0]
            h = res + x
        hid = _rms(h, d.norm, d.eps)[1:]
        logits = d.lm_head(hid.to(d._target_dtype))
        unary, cand = torch.topk(logits.float(), d.top_k, dim=-1, sorted=False)
        hp = _lin(hid, d.sel_proj).float()
        succ = sh["succ"][cand].float()
        psub = sh["pred"][cand]
        prow = sh["pred"].index_select(0, tok.view(1))[0]
        path = []
        for r in range(hid.shape[0]):
            scores = unary[r] + succ[r] @ (prow.float() * hp[r])
            idx = scores.argmax().view(1)
            prow = psub[r].index_select(0, idx)[0]
            path.append(cand[r].index_select(0, idx))
        return torch.cat(path)

    # -- driver ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def propose(self, last_token: int, start: int) -> list[int] | None:
        global _POOL
        d = self.d
        if not d._n:
            return None
        d._reserve(0)
        mask_emb = self._mask_rows()
        self.h_in[1:].copy_(mask_emb.expand(d.block_size - 1, -1))
        e = d.embed(torch.tensor([last_token], dtype=torch.long, device=d.device))
        d._target_dtype = e.dtype
        self.h_in[0].copy_((e[0] * getattr(d, "embed_mult", 1.0)).to(d.dtype))
        self.scal.copy_(torch.tensor([d._ctx_start, d._off, d._n, start, last_token], dtype=torch.long),
                        non_blocking=False)
        key = (d._kb[0].data_ptr(), d._kb[0].shape[2], d._vb[-1].data_ptr())
        if key != self.key:
            self.graph = self.out = None
            s = torch.cuda.Stream(device=d.device)
            s.wait_stream(torch.cuda.current_stream(d.device))
            with torch.cuda.stream(s):
                self._body()                                  # warm-up: lazy inits / workspaces outside capture
            torch.cuda.current_stream(d.device).wait_stream(s)
            if _POOL is None:
                _POOL = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g, pool=_POOL):
                    out = self._body()
            except Exception as ex:                           # an uncapturable op: stay eager from now on
                print(f"DFlash2 graph: capture failed ({type(ex).__name__}: {ex}); eager drafting", flush=True)
                d._graph = False
                return None
            self.graph, self.out, self.key = g, out, key
        self.graph.replay()
        return self.out.tolist()
