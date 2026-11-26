"""Long-context prefill for TP, layer by layer (framework level).

Chunk-major prefill (every chunk through all layers) must re-read all earlier KV for every chunk; with KV offloaded to
host RAM that is O(n^2) PCIe traffic (108k tokens: ~650 s on the V100 pair). Layer-major prefill runs ONE layer over the
whole prompt, then the next:

  * attention layer: this card's share of that layer's KV for the whole sequence lives in VRAM only while the layer
    runs (108k x 2 heads x 256 x fp16 x (k, v) = 220 MB), so attention is GPU-local; then the KV goes to the decoder's
    hot window (newest positions) and to pinned host RAM (older ones) ONCE -- O(n) traffic;
  * recurrent layer (Gated DeltaNet, Mamba-like): the family's chunked kernel carries its state across chunks;
  * hidden states of the prompt stay in VRAM between layers (n x hidden fp16).

Everything model-specific is behind a small FAMILY ADAPTER (which modules make a layer, how q/k/v/rope are formed);
the loop, the attention, the KV placement, the prefix checkpoints and the drafter hand-off are generic. An adapter
for another family (Llama-like, Ripple, MoE such as Qwen3.6-35B-A3B: its MoE block is just `mlp`) only implements
the methods below.

The TP split is untouched: each card runs the same loop on its own share (heads / MLP columns); the row-parallel
outputs all-reduce inside the modules, so the cards stay in step.
"""
from __future__ import annotations

import os
import time

import torch


# ---------------------------------------------------------------------------------------------- family adapters
class FamilyAdapter:
    """What the layer-major prefill needs from a model family. One instance per loaded model."""

    def __init__(self, model):
        self.model = model

    # structure
    def layers(self):                          # list of decoder layers
        raise NotImplementedError

    def kind(self, i) -> str:                  # "attn" or "recurrent"
        raise NotImplementedError

    # pieces of one layer (h: [1, T, hidden])
    def norm_in(self, i, h):
        raise NotImplementedError

    def recurrent(self, i, x, cache):          # recurrent mixer, state carried in `cache` across chunks
        raise NotImplementedError

    def qkv(self, i, x, pos):                  # -> q [1, Hq, T, hd] (roped), k, v [1, nkv, T, hd], extra (e.g. gate)
        raise NotImplementedError

    def attn_out(self, i, o, extra):           # o [1, T, Hq * hd] -> mixer output [1, T, hidden]
        raise NotImplementedError

    def scale(self, i) -> float:
        raise NotImplementedError

    def ffn(self, i, h):                       # h + post-norm MLP / MoE (residual included)
        raise NotImplementedError

    # ends
    def embed(self, ids):                      # list[int] -> [1, T, hidden] on the model's device
        raise NotImplementedError

    def logits(self, h_last):                  # [1, 1, hidden] -> [vocab]
        raise NotImplementedError

    def new_cache(self):                       # state container for the recurrent layers
        return None


class Qwen35Adapter(FamilyAdapter):
    """Qwen3.5 / 3.6 / 3.8 hybrid: Gated DeltaNet + gated full attention (M-RoPE, text positions)."""

    def __init__(self, model, device):
        super().__init__(model)
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        self.M, self.dev = M, device
        self._layers = list(model.model.layers)

    def layers(self):
        return self._layers

    def kind(self, i):
        return "recurrent" if getattr(self._layers[i], "linear_attn", None) is not None else "attn"

    def norm_in(self, i, h):
        return self._layers[i].input_layernorm(h)

    def recurrent(self, i, x, cache):
        return self._layers[i].linear_attn(hidden_states=x, cache_params=cache, attention_mask=None)

    def qkv(self, i, x, pos):
        m = self._layers[i].self_attn
        T, hd = x.shape[1], m.head_dim
        q, gate = torch.chunk(m.q_proj(x).view(1, T, -1, hd * 2), 2, dim=-1)
        q = m.q_norm(q).transpose(1, 2)
        k = m.k_norm(m.k_proj(x).view(1, T, -1, hd)).transpose(1, 2)
        v = m.v_proj(x).view(1, T, -1, hd).transpose(1, 2)
        cos, sin = self.model.model.rotary_emb(x, pos.view(1, 1, T).expand(3, 1, T))
        q, k = self.M.apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v, gate.reshape(1, T, -1)

    def attn_out(self, i, o, gate):
        return self._layers[i].self_attn.o_proj(o * torch.sigmoid(gate))

    def scale(self, i):
        return self._layers[i].self_attn.scaling

    def ffn(self, i, h):
        L = self._layers[i]
        return h + L.mlp(L.post_attention_layernorm(h))

    def embed(self, ids):
        return self.model.model.embed_tokens(torch.tensor([ids], device=self.dev))

    def logits(self, h_last):
        return self.model.lm_head(self.model.model.norm(h_last))[0, -1]

    def new_cache(self):
        from transformers import DynamicCache
        return DynamicCache(config=self.model.config)

    # recurrent state <-> host checkpoints / decoder buffers
    @staticmethod
    def rec_state(cache, i):
        L = cache.layers[i]
        return L.conv_states[0], L.recurrent_states[0]

    @staticmethod
    def seed_rec_state(cache, i, conv, rec):
        L = cache.layers[i]
        L.lazy_initialization(conv_states=conv, recurrent_states=rec, conv_kernel_size=conv.shape[-1])
        L.conv_states[0].copy_(conv)
        L.recurrent_states[0].copy_(rec)
        L.has_previous_state[0] = True


def adapter_for(model, device) -> FamilyAdapter:
    mt = getattr(model.config, "model_type", "")
    if mt.startswith("qwen3_5") or mt.startswith("qwen3_6") or hasattr(model.model.layers[0], "linear_attn") or \
            any(hasattr(L, "linear_attn") for L in model.model.layers):
        return Qwen35Adapter(model, device)
    raise NotImplementedError(f"no TP long-context adapter for model_type {mt!r}")


# ---------------------------------------------------------------------------------------------- generic pieces
def _attn_lse(q, k, v, scale, mask=None):
    """q [1, Hq, T, hd] over k, v [1, nkv, L, hd] (GQA, query heads grouped per kv head); fp32 softmax.
    Returns normalised o [1, Hq, T, hd] fp32 and lse [1, Hq, T]."""
    _, hq, T, hd = q.shape
    nkv = k.shape[1]
    g = hq // nkv
    s = torch.matmul(q[0].reshape(nkv, g * T, hd), k[0].transpose(-1, -2)).float() * scale
    if mask is not None:
        s = s.view(nkv, g, T, -1).masked_fill(~mask, float("-inf")).view(nkv, g * T, -1)
    mx = s.amax(-1, keepdim=True).clamp(min=-1e30)
    p = torch.exp(s - mx)
    l = p.sum(-1, keepdim=True)
    o = torch.matmul(p, v[0].float()) / l
    return o.view(1, hq, T, hd), (mx + torch.log(l)).squeeze(-1).view(1, hq, T)


def _merge(o1, l1, o2, l2):
    l = torch.logaddexp(l1, l2)
    return o1 * torch.exp(l1 - l)[..., None] + o2 * torch.exp(l2 - l)[..., None], l


def causal_attention(q, K, V, p0, scale, kv_block=4096):
    """Queries at absolute positions p0 .. p0+T-1 over K, V [1, nkv, >= p0+T, hd] (all earlier positions in VRAM)."""
    T = q.shape[2]
    end = p0 + T
    qpos = torch.arange(p0, end, device=q.device)
    o = l = None
    for a in range(0, end, kv_block):
        b = min(end, a + kv_block)
        mask = None
        if b > p0:                                   # the block reaches the queries: causal mask
            mask = torch.arange(a, b, device=q.device)[None, :] <= qpos[:, None]
        o_, l_ = _attn_lse(q, K[:, :, a:b], V[:, :, a:b], scale, mask)
        o, l = (o_, l_) if o is None else _merge(o, l, o_, l_)
    return o


# ---------------------------------------------------------------------------------------------- the prefill
@torch.no_grad()
def prefill_layer_major(dec, ids, drafter=None, chunk=int(os.environ.get("QV_TP_LM_CHUNK", "1024"))):
    """Prefill `ids` into a StaticDecoder `dec` (tp_graph) with KV offload, one layer at a time.

    Prefix cache (dec.prefix): KV of [0, start) is already in the decoder (hot window + host RAM); recurrent layers
    restart from a checkpoint at `start`. Checkpoints of every recurrent layer are taken at grid points and at the
    prompt end (per layer, as each layer passes them). Returns the first generated token (greedy)."""
    ad = getattr(dec, "adapter", None)
    if ad is None:
        ad = dec.adapter = adapter_for(dec.model, dec.dev)
    caps, pc, dev = list(dec.caps), getattr(dec, "prefix", None), dec.dev
    n = len(ids)
    start = pc.best(ids) if pc is not None else 0
    cache = ad.new_cache()
    if start > 0:
        ck = pc.ckpt[start]
        for i in range(len(ad.layers())):
            if ad.kind(i) == "recurrent":
                ad.seed_rec_state(cache, i, ck["conv"][i].to(dev), ck["rec"][i].to(dev))
        for k in [k for k in pc.ckpt if k > start]:
            del pc.ckpt[k]
        if drafter is not None:
            if ck.get("draft") is not None:
                drafter.import_state(ck["draft"])
            elif hasattr(drafter, "_reset_state"):
                drafter._reset_state()
        dec.truncate(start)
    else:
        dec.reset()
        if pc is not None:
            pc.ckpt.clear()
        if drafter is not None and hasattr(drafter, "_reset_state"):
            drafter._reset_state()
    dec.state_wide.zero_()
    old_base = dec.base
    N = n - start
    # hidden states of the new tokens, in VRAM between layers
    h_all = torch.empty(1, N, dec.h_in.shape[-1], dtype=dec.h_in.dtype, device=dev)
    for a in range(0, N, chunk):
        h_all[:, a:a + chunk].copy_(ad.embed(ids[start + a:start + min(N, a + chunk)]))
    pos_all = torch.arange(start, n, device=dev)
    # the drafter only keeps a sliding window of context: its features are kept for the last `fw` positions only
    fw = min(N, int(os.environ.get("QV_TP_DRAFT_FEATS", "4096")))
    feats = torch.empty(fw, len(caps) * h_all.shape[-1], dtype=h_all.dtype, device=dev) if (drafter and caps) else None
    grid = pc.grid if pc is not None else 0
    ck_points = sorted({p for p in range(start + 1, n + 1) if grid and p % grid == 0} | {n} |
                       ({n - 1} if n - 1 > start else set()) |
                       {m for m in pc.marks if start < m < n}) if pc is not None else []
    W = dec.W
    new_base = max(0, n - W // 2) // 256 * 256 if dec.offload else 0     # whole qvb8 blocks
    if not dec.offload:
        assert n <= W, f"prompt of {n} tokens exceeds the KV buffer {W}"

    tk = {"attn": 0.0, "recurrent": 0.0}
    prof = os.environ.get("QV_TP_LM_PROF") == "1"
    for i in range(len(ad.layers())):
        if prof:
            torch.cuda.synchronize(dev)
            t0 = time.time()
        if ad.kind(i) == "attn":
            st = dec.state[i]
            kv = st[1]                                     # qvb8 hot window (engine/static_decode.StaticQKV)
            nkv, hd = kv.kt.shape[1], kv.kt.shape[3]
            K = torch.empty(1, nkv, n, hd, dtype=kv.kt.dtype, device=dev)
            V = torch.empty_like(K)
            if start > 0:                                  # earlier positions: host [0, old_base) + hot window
                cold = min(start, old_base)
                if cold > 0:
                    dec.cold_store().read_into(i, K, V, cold)
                if start > old_base:
                    hk, hv = dec.hot_read(i, start - old_base)
                    K[:, :, old_base:start].copy_(hk)
                    V[:, :, old_base:start].copy_(hv)
                    del hk, hv
            for a in range(0, N, chunk):
                b = min(N, a + chunk)
                h = h_all[:, a:b]
                x = ad.norm_in(i, h)
                q, k, v, extra = ad.qkv(i, x, pos_all[a:b])
                K[:, :, start + a:start + b].copy_(k)
                V[:, :, start + a:start + b].copy_(v)
                L_ = start + b
                mask = torch.arange(L_, device=dev)[None, :] <= pos_all[a:b, None]
                o = torch.nn.functional.scaled_dot_product_attention(
                    q, K[:, :, :L_], V[:, :, :L_], attn_mask=mask, scale=ad.scale(i), enable_gqa=True)
                o = o.transpose(1, 2).reshape(1, b - a, -1)
                h = h + ad.attn_out(i, o, extra)
                h_all[:, a:b].copy_(ad.ffn(i, h))
            # place the KV: [new_base, n) in the hot window, [lo, new_base) to host RAM (once)
            if dec.offload:
                lo = min(start, old_base)
                if new_base > lo:
                    dec.cold_store().append(i, K[0, :, lo:new_base], V[0, :, lo:new_base])
            # same base: the hot blocks below `start` hold the same tokens -- kept, not rounded twice
            dec.hot_load(i, K[:, :, new_base:n], V[:, :, new_base:n],
                         keep=max(0, start - new_base) if new_base == old_base else 0)
            del K, V
        else:
            a = 0
            while a < N:
                b = min(N, a + chunk)
                for p in ck_points:                         # chunks end exactly on checkpoint positions
                    if start + a < p < start + b:
                        b = p - start
                        break
                h = h_all[:, a:b]
                h = h + ad.recurrent(i, ad.norm_in(i, h), cache)
                h_all[:, a:b].copy_(ad.ffn(i, h))
                if (start + b) in ck_points:
                    conv, rec = ad.rec_state(cache, i)
                    _ckpt(pc, start + b, i, conv, rec)
                a = b
        if prof:
            torch.cuda.synchronize(dev)
            tk[ad.kind(i)] += time.time() - t0
        if feats is not None and i in caps:
            c = caps.index(i)
            H = h_all.shape[-1]
            feats[:, c * H:(c + 1) * H].copy_(h_all[0, N - fw:])
    from qv_band.run.tp_graph import _first
    tok = _first(dec, ad.logits(h_all[:, -1:]))
    if prof:
        print(f"[tp-longctx] {N} tokens: attention layers {tk['attn']:.1f}s, recurrent layers "
              f"{tk['recurrent']:.1f}s", flush=True)
    if feats is not None:
        drafter.observe_features(n - fw, feats)
    # recurrent states into the decoder's graph buffers
    for i in range(len(ad.layers())):
        if ad.kind(i) == "recurrent":
            conv, rec = ad.rec_state(cache, i)
            st = dec.state[i]
            cv, _, hn, hist = st[1:]
            k = cv.shape[-1]
            hn[:, :, 1:1 + k].copy_(conv)
            hist[0].copy_(rec[0].float())
    del cache, h_all
    if pc is not None and n in pc.ckpt and drafter is not None:
        pc.ckpt[n]["draft"] = drafter.export_state()
    if pc is not None:
        while len(pc.ckpt) > pc.max_ckpt:
            del pc.ckpt[min(k for k in pc.ckpt if k != max(pc.ckpt))]
        pc.ids = list(ids)
    dec.base = new_base if dec.offload else 0
    dec.base_t.fill_(dec.base)
    dec.pos.fill_(n - 1)
    dec.commit(1)
    dec.prefill_start = start
    torch.cuda.synchronize(dev)
    return tok


def _ckpt(pc, p, i, conv, rec):
    if pc is None:
        return
    c = pc.ckpt.setdefault(p, {"conv": {}, "rec": {}, "draft": None})
    for name, t in (("conv", conv), ("rec", rec.float())):
        hbuf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        hbuf.copy_(t)
        c[name][i] = hbuf
