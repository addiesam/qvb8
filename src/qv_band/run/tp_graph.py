"""TP stage 4: Qwen3.5 / Qwen3.8 decode as CUDA graphs per card, plain or with DFlash.

    torchrun --nproc-per-node 2 -m qv_band.run.tp_graph --model PACK --new 128 [--draft DFLASH_PACK]

Why: the plain HF forward launches ~5,300 kernels per token from Python (220 ms/token on the V100
pair; the GPUs do ~31 ms of it). Captured once and replayed, the step is GPU-bound, and NCCL inside a
graph costs 29 us per all-reduce instead of 107 us.

Graphs per card (each holds only this card's heads):
  step[T]  T tokens at positions pos..pos+T-1 (T = 1 decode, T = block verify / prefill chunk):
           64 layers (2 all-reduces each), norm, vocab-split lm_head (gathered), argmax per position,
           the target-layer features the drafter reads. KV is written at its positions (later
           positions are masked, so rejected tokens need no undo); every DeltaNet layer keeps its
           recurrent state AFTER EACH token and its conv window, in static buffers.
  commit   given c (tokens kept, a GPU tensor), restores each DeltaNet state and conv window to
           "after token c" and advances pos by c. Exact rollback, no recompute.
Every card computes the same full logits, so the target tokens agree on every card; the drafter
(on every card: its lm_head is the gathered one, a collective) proposes, and card 0's drafts are
broadcast so the verify block is the same everywhere. Embedding: host RAM, outside the graphs.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn.functional as F

from qv_band.engine.static_decode import StaticQKV

from qv_band.formats.qvb8.tp import TPGroup
from qv_band.run.tp_qwen35 import build
import qv_band.formats.qvb8.wmma_sm70  # noqa: F401  (QVB8_WMMA=1: V100 tensor-core GEMM for 5..16 rows)


STATE_STOCHASTIC = os.environ.get("QV_STATE_STOCHASTIC", "0") == "1"


def round_state(S, tau, stochastic=None):
    """Certified rounding of a DeltaNet state (heads, d_k, d_v): per head and d_v column, the d_k values go on a grid
    min + j * step. Nearest: step 2*tau, |err| <= tau. Stochastic (unbiased: E[rounded] = S, so updates smaller
    than a step survive on average instead of being erased every token): step tau, |err| < tau."""
    stochastic = STATE_STOCHASTIC if stochastic is None else stochastic
    mn = S.amin(1, keepdim=True)
    if stochastic:
        u = (S - mn) / tau
        return mn + torch.floor(u + torch.rand_like(u)) * tau
    step = 2.0 * tau
    return mn + torch.round((S - mn) / step) * step


def state_bits(S, tau, stochastic=None):
    """Bits per value that the rounded state needs (per head and column: ceil(log2(span / step + 1)))."""
    stochastic = STATE_STOCHASTIC if stochastic is None else stochastic
    span = (S.amax(1) - S.amin(1)).float()
    return torch.ceil(torch.log2(span / (tau if stochastic else 2.0 * tau) + 1)).clamp(min=1)


PACK_TAU = float(os.environ.get("QV_STATE_TAU", "0") or 0) or None   # e.g. 1e-5: rounded + packed live DeltaNet state
QKV_BLOCK = 256                    # qvb8 KV: tokens per (block, channel) min / step (engine/static_decode.StaticQKV)
GDN_FUSED = os.environ.get("QV_GDN_FUSED", "1") == "1"   # one Triton kernel per layer for the T-token recurrence
STATE_ESC = int(os.environ.get("QV_STATE_ESC", "64"))   # per layer: the widest columns kept exact (fp32)


def _l2(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class StaticDecoder:
    def __init__(self, model, tcfg, device, max_len=4096, t_max=8, capture_layers=(), kv_window=None):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        self.M, self.model, self.cfg, self.dev = M, model, tcfg, device
        self.max_len, self.t_max = max_len, t_max
        # KV offload (as the original engine): the newest W positions of each attention layer stay in VRAM (hot); older
        # ones go to pinned host RAM (cold, this card's heads only) and are streamed back and merged exactly (LSE).
        self.W = min(kv_window or max_len, max_len)
        self.offload = self.W < max_len
        self.W = -(-self.W // QKV_BLOCK) * QKV_BLOCK                        # qvb8 KV: whole blocks
        self.base = 0                                                       # absolute position of hot slot 0
        self.base_t = torch.zeros(1, dtype=torch.long, device=device)
        self.host = {}                                                      # layer -> (k, v) pinned [nkv, max_len, hd]
        self.caps = list(capture_layers)
        dt = torch.float16
        # pipeline stage (tp_pipe): only this rank's layers; the others are nn.Identity placeholders
        self.layers = [l for l in model.model.layers if not isinstance(l, torch.nn.Identity)]
        self.pp_last = not isinstance(model.lm_head, torch.nn.Identity)    # last stage: norm + lm_head here
        self.state = []
        for layer in self.layers:
            g = getattr(layer, "linear_attn", None)
            if g is not None:
                n = g.conv_kernel_size
                conv = torch.zeros(1, g.conv_dim, n, dtype=dt, device=device)
                hn = torch.zeros(1, g.conv_dim, n + t_max, dtype=dt, device=device)   # window history
                shp = (1, g.num_v_heads, g.head_k_dim, g.head_v_dim)
                if PACK_TAU:      # live state as certified 16-bit ids + per (head, column) min and step: 0.5x of fp32
                    E = min(STATE_ESC, shp[1] * shp[3])
                    rec = (torch.zeros(shp, dtype=torch.int16, device=device),
                           torch.zeros(1, shp[1], 1, shp[3], dtype=torch.float32, device=device),
                           torch.full((1, shp[1], 1, shp[3]), 2 * PACK_TAU, dtype=torch.float32, device=device),
                           torch.arange(E, dtype=torch.long, device=device),            # escaped columns (head*dv + col)
                           torch.zeros(E, shp[2], dtype=torch.float32, device=device))   # their exact values (d_k)
                else:
                    rec = torch.zeros(shp, dtype=torch.float32, device=device)
                hist = torch.zeros(t_max, *shp[1:], dtype=torch.float32, device=device)  # state after each token
                self.state.append(("gdn", conv, rec, hn, hist))
            else:
                a = layer.self_attn
                nkv = a.k_proj.out_features // a.head_dim
                # qvb8 KV (engine/static_decode.StaticQKV): 8-bit blocks + a dense tail with room for one draft block
                self.state.append(("attn", StaticQKV(1, nkv, self.W, a.head_dim, dt, device, QKV_BLOCK, t_max)))
        H = tcfg.hidden_size
        self.h_in = torch.zeros(1, t_max, H, dtype=dt, device=device)
        self.h_out = torch.zeros(1, t_max, H, dtype=dt, device=device)    # a non-last stage's output
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        # an image prompt: Qwen-VL M-RoPE puts every token after a picture at (KV position + delta); 0 for text
        self.rope_delta = torch.zeros(1, dtype=torch.long, device=device)
        self.keep = torch.ones(1, dtype=torch.long, device=device)          # c for commit
        self.tok_out = torch.zeros(t_max, dtype=torch.long, device=device)
        self.logits_out = None                    # [t_max, vocab] of the last run (sampling): _keep_logits
        self.first_logits = None                  # the prompt's last logits after a prefill (sampling the first token)
        self.feats = torch.zeros(t_max, len(self.caps) * H, dtype=dt, device=device)
        self.A = [i for i, st in enumerate(self.state) if st[0] == "attn"]
        if self.offload and any(getattr(l, "_mhc_settings", None) for l in self.layers):
            raise ValueError("mHC layers (SenseNova) run in _step / the HF prefill only, not in the KV-offload paths "
                             "(_seg, prefill_off, layer-major prefill): --kv-window must be >= --max-len")
        if self.A:                                                          # a stage may hold no attention layer
            a0 = self.layers[self.A[0]].self_attn
            self.hq = a0.q_proj.out_features // (2 * a0.head_dim)
            self.hb = torch.zeros(1, t_max, H, dtype=dt, device=device)    # segmented decode: residual at the split
            self.qb = torch.zeros(1, self.hq, t_max, a0.head_dim, dtype=dt, device=device)
            self.gb = torch.zeros(1, t_max, self.hq * a0.head_dim, dtype=dt, device=device)
            self.ob = torch.zeros(1, self.hq, t_max, a0.head_dim, dtype=torch.float32, device=device)
            self.lb = torch.zeros(1, self.hq, t_max, dtype=torch.float32, device=device)
        self.seg_graphs = {}
        self.cold_block = int(os.environ.get("QV_TP_COLD_BLOCK", "8192"))
        self.copy_stream = torch.cuda.Stream(device)
        self.kvb = None                                                     # 2 GPU staging buffers for cold blocks
        self.graphs, self.commit_graph = {}, None
        self.tar = torch.arange(t_max, device=device)
        if os.environ.get("QV_ROW_DETERMINISTIC", "1") == "1":     # bit-identical rows at any T (spec == plain)
            from qv_band.run.tp_rownorm import install
            install(model)
        self.state_tau = PACK_TAU          # certified rounding of the DeltaNet state after every token (None: exact fp32)
        self.state_wide = torch.zeros(1, dtype=torch.long, device=device)   # columns that needed a step > 2 tau
        self.bits_log = None               # a list to collect state_bits() means (evaluation only, eager)
        self.rec = None                    # diagnostics: {(layer, name): tensor [T, ...]} (eager only)

    def reset(self):
        for st in self.state:
            for t in st[1:]:
                if isinstance(t, StaticQKV):
                    t.reset()
                elif isinstance(t, tuple):                                     # packed state: decodes to 0
                    t[0].fill_(-32768); t[1].zero_(); t[2].fill_(2 * PACK_TAU); t[4].zero_()
                else:
                    t.zero_()
        self.pos.zero_()
        self.state_wide.zero_()
        self.base = 0
        self.base_t.zero_()
        if getattr(self, "cold", None) is not None:
            self.cold.clear()

    def state_bytes(self) -> int:
        """Bytes of the live DeltaNet state on this card (one sequence)."""
        n = 0
        for st in self.state:
            if st[0] == "gdn":
                r = st[2]
                n += sum(x.numel() * x.element_size() for x in (r if isinstance(r, tuple) else (r,)))
        return n

    # ---------------------------------------------------------------- T tokens, all layers
    def _gdn(self, m, x, T, conv, rec, hn, hist):
        mixed = m.in_proj_qkv(x).transpose(1, 2)                            # [1, C, T]
        z = m.in_proj_z(x).reshape(1, T, -1, m.head_v_dim)
        b, a = m.in_proj_b(x), m.in_proj_a(x)
        n = conv.shape[-1]
        w = torch.cat([conv, mixed.to(conv.dtype)], dim=-1)                # [1, C, n + T]
        hn[:, :, :n + T].copy_(w)
        out = F.conv1d(w, m.conv1d.weight, m.conv1d.bias, groups=w.shape[1])[:, :, -T:]
        mixed = F.silu(out).to(x.dtype).transpose(1, 2)
        self._r("conv", mixed[0])
        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(1, T, -1, m.head_k_dim)
        k = k.reshape(1, T, -1, m.head_k_dim)
        v = v.reshape(1, T, -1, m.head_v_dim)
        beta = b.sigmoid()
        g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
        r = m.num_v_heads // m.num_k_heads
        if r > 1:
            q, k = q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2)
        # gated delta rule, one token at a time (HF torch_recurrent_gated_delta_rule), keeping each state
        q, k = q.float(), k.float()                       # l2-normalised per token below: same shapes at any T
        v, beta, g = v.float(), beta.float(), g.float()
        scale = q.shape[-1] ** -0.5
        if isinstance(rec, tuple):                                          # packed: decode (exact grid values)
            ids, mn, step, eidx, evals = rec
            S = (mn + (ids.float() + 32768.0) * step)[0]                    # [heads, dk, dv]
            h_, dk_, dv_ = S.shape
            St = S.permute(0, 2, 1).reshape(h_ * dv_, dk_)                  # one row per (head, column)
            St.index_copy_(0, eidx, evals)                                  # the widest columns, exact
            S = St.view(h_, dv_, dk_).permute(0, 2, 1)
        else:
            S = rec[0]
        if GDN_FUSED and self.bits_log is None:
            from qv_band.run.tp_gdn_fused import gdn_fused
            o = gdn_fused(q[0].contiguous(), k[0].contiguous(), v[0].contiguous(), g[0].contiguous(),
                          beta[0].contiguous(), S.contiguous(), hist, self.state_tau).to(x.dtype)
            self._r("delta", o)
            o = m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(1, T, -1)
            return m.out_proj(o)
        outs = []
        for t in range(T):
            qt, kt = _l2(q[0, t]) * scale, _l2(k[0, t])
            S = S * g[0, t].exp()[:, None, None]
            kv = (S * kt[:, :, None]).sum(-2)
            d = (v[0, t] - kv) * beta[0, t][:, None]
            S = S + kt[:, :, None] * d[:, None, :]
            if self.state_tau:
                if self.bits_log is not None:
                    self.bits_log.append(state_bits(S, self.state_tau).mean().item())
                S = round_state(S, self.state_tau)
            hist[t].copy_(S)
            outs.append((S * qt[:, :, None]).sum(-2))
        o = torch.stack(outs, 0).to(x.dtype)                                # [T, heads, dv]
        self._r("delta", o)
        o = m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(1, T, -1)
        self._r("gnorm", o[0])
        return m.out_proj(o)

    def _proj(self, m, x, T, cos, sin):
        """q, k (roped), v [1, heads, T, hd] and the output gate."""
        hd = m.head_dim
        q, gate = torch.chunk(m.q_proj(x).view(1, T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(1, T, -1)
        q = m.q_norm(q).transpose(1, 2)
        k = m.k_norm(m.k_proj(x).view(1, T, -1, hd)).transpose(1, 2)
        v = m.v_proj(x).view(1, T, -1, hd).transpose(1, 2)
        q, k = self.M.apply_rotary_pos_emb(q, k, cos, sin)
        self._r("q", q[0].transpose(0, 1))
        self._r("k", k[0].transpose(0, 1))
        return q, gate, k, v

    def _qkv(self, m, x, T, kv, cos, sin):
        """q (roped), the output gate; k, v written into the hot window at (pos - base) .. + T - 1 (qvb8 tail)."""
        q, gate, k, v = self._proj(m, x, T, cos, sin)
        kv.write(k, v, self.pos - self.base_t)
        return q, gate

    def _hot(self, kv, q, T, scale):
        """Attention of T new queries over the hot window (qvb8 blocks + tail, causal): o fp32, lse."""
        vis = (self.pos - self.base_t - kv.lc).view(1, 1) + self.tar[None, :T] + 1
        return kv.attn(q, vis, scale)

    def _attn(self, m, x, T, kv, cos, sin):
        q, gate = self._qkv(m, x, T, kv, cos, sin)
        o, _ = self._hot(kv, q, T, m.scaling)
        o = o.to(x.dtype)
        self._r("sdpa", o[0].transpose(0, 1))
        o = o.transpose(1, 2).reshape(1, T, -1) * torch.sigmoid(gate)
        return m.o_proj(o)

    def _mhc_layer(self, layer, st, h, T, mh, cos, sin):
        """models/sensenova/adapter.MHCDecoderLayer.forward on the static decode state: enter the streams at the first layer of
        the range, read -> token mixer -> merge, read -> MLP -> merge, average the streams at its last layer."""
        from qv_band.models.sensenova.adapter import MHCRouter
        S, phys = int(mh["streams"]), layer._mhc_physical
        R = h.unsqueeze(-2).expand(*h.shape[:-1], S, h.shape[-1]).contiguous() if phys == mh["start"] else h.unflatten(-1, (S, -1))
        branch, post, mix = layer.mhc_attn(R)
        x = layer.input_layernorm(branch)
        if st[0] == "gdn":
            o = self._gdn(layer.linear_attn, x, T, *st[1:])
        else:
            o = self._attn(layer.self_attn, x, T, st[1], cos, sin)
        R = MHCRouter.merge(R, o, post, mix)
        branch, post, mix = layer.mhc_ffn(R)
        R = MHCRouter.merge(R, layer.mlp(layer.post_attention_layernorm(branch)), post, mix)
        return R.mean(-2) if phys == int(mh["end"]) - 1 else R.flatten(-2)

    def _step(self, T):
        h = self.h_in[:, :T]
        posT = self.pos + torch.arange(T, device=self.dev)
        posR = posT + self.rope_delta                                       # an image prompt shifts rope positions
        cos, sin = self.model.model.rotary_emb(h, posR.view(1, 1, T).expand(3, 1, T))
        caps = {}
        for i, (layer, st) in enumerate(zip(self.layers, self.state)):
            self._li = i
            mh = getattr(layer, "_mhc_settings", None)
            if mh is not None:                      # SenseNova mHC layer: the reference forward over the residual streams
                h = self._mhc_layer(layer, st, h, T, mh, cos, sin)
                if i in self.caps:
                    caps[i] = h[0]
                continue
            self._r("h_in", h[0])
            x = layer.input_layernorm(h)
            self._r("in_norm", x[0])
            if st[0] == "gdn":
                o = self._gdn(layer.linear_attn, x, T, *st[1:])
            else:
                o = self._attn(layer.self_attn, x, T, st[1], cos, sin)
            self._r("mixer_out", o[0])
            h = h + o
            self._r("h_mid", h[0])
            pn = layer.post_attention_layernorm(h)
            self._r("post_norm", pn[0])
            mo = layer.mlp(pn)
            self._r("mlp_out", mo[0])
            h = h + mo
            if i in self.caps:
                caps[i] = h[0]
        hn = self.model.model.norm(h) if self.pp_last else None
        if len(self.layers) in self.caps:                    # drafter source "final_postnorm"
            caps[len(self.layers)] = hn[0]
        if self.caps:
            self.feats[:T].copy_(torch.cat([caps[i] for i in self.caps], dim=-1))
        if not self.pp_last:
            self.h_out[:, :T].copy_(h)
            return None
        logits = self.model.lm_head(hn)
        self.tok_out[:T].copy_(logits[0].argmax(-1))
        self._keep_logits(logits, T)
        return logits

    # ---------------------------------------------------------------- KV offload
    @staticmethod
    def _attn_lse(q, k, v, scale, mask=None):
        """Attention of q [1, Hq, T, hd] over k, v [1, nkv, L, hd] (GQA), softmax in fp32.
        mask [T, L] bool (True = visible). Returns o [1, Hq, T, hd] fp32 (normalised) and lse [1, Hq, T]."""
        _, hq, T, hd = q.shape
        nkv = k.shape[1]
        g = hq // nkv
        s = torch.matmul(q[0].reshape(nkv, g * T, hd), k[0].transpose(-1, -2)).float() * scale   # [nkv, gT, L]
        if mask is not None:
            s = s.view(nkv, g, T, -1).masked_fill(~mask, float("-inf")).view(nkv, g * T, -1)
        mx = s.amax(-1, keepdim=True).clamp(min=-1e30)
        p = torch.exp(s - mx)
        l = p.sum(-1, keepdim=True)
        o = torch.matmul(p, v[0].float()) / l
        lse = (mx + torch.log(l)).squeeze(-1)
        return o.view(1, hq, T, hd), lse.view(1, hq, T)

    @staticmethod
    def _merge(o1, l1, o2, l2):
        l = torch.logaddexp(l1, l2)
        return o1 * torch.exp(l1 - l)[..., None] + o2 * torch.exp(l2 - l)[..., None], l

    def cold_store(self):
        if getattr(self, "cold", None) is None:
            from qv_band.run.tp_coldkv import ColdStore
            self.cold = ColdStore(self.A, self.dev, seg=self.cold_block)
        return self.cold

    def _cold(self, li, q, o, lse, scale):
        """Merge attention over the cold positions [0, base) (pinned host RAM, packed, streamed segment by segment)."""
        if self.base == 0:
            return o, lse
        for k, v in self.cold_store().blocks(li):
            oc, lc = self._attn_lse(q, k, v, scale)
            o, lse = self._merge(o, lse, oc, lc)
        return o, lse

    def spill(self, T):
        """Make room for T new tokens in the hot window: move the oldest hot positions to host RAM (eager)."""
        p = int(self.pos)
        used = p - self.base
        if used + T <= self.W:
            return
        assert self.offload, f"context {p + T} exceeds the KV window {self.W} (no offload)"
        B = QKV_BLOCK
        s = -(-max(used - self.W // 2, used + T - self.W) // B) * B        # whole qvb8 blocks
        assert self.base + s <= self.max_len - T, "context exceeds --max-len"
        cs = self.cold_store()
        for li in self.A:
            kv = self.state[li][1]
            assert s <= int(kv.lc[0]), "spill beyond the closed qvb8 blocks"
            k, v = kv.drop_front(s // B)
            cs.append(li, k[0], v[0])
        self.base += s
        self.base_t.fill_(self.base)

    def truncate(self, p):
        """Keep positions [0, p) only (prefix-cache restore)."""
        if p < self.base:            # the cold part holds [0, base): keep [0, p), the hot window starts empty at p
            self.cold_store().truncate(p)
            self.base = p
            self.base_t.fill_(p)
            for li in self.A:
                self.state[li][1].clear(0)
        else:
            for li in self.A:
                self.state[li][1].truncate(0, p - self.base)

    def hot_read(self, li, n):
        """Dense k, v [1, nkv, n, hd] of the first n hot positions of attention layer li."""
        k, v = self.state[li][1].dense(0)
        return k[None, :, :n], v[None, :, :n]

    def hot_load(self, li, k, v, keep=0):
        """The hot window of layer li := k, v [1, nkv, n, hd]; blocks wholly below `keep` are unchanged (not re-rounded)."""
        self.state[li][1].load(0, k, v, keep)

    def _layer(self, i, h, T, cos, sin, mask, posT):
        layer, st = self.layers[i], self.state[i]
        x = layer.input_layernorm(h)
        if st[0] == "gdn":
            o = self._gdn(layer.linear_attn, x, T, *st[1:])
        else:
            o = self._attn(layer.self_attn, x, T, st[1], cos, sin)
        h = h + o
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def _cap(self, i, h, T):
        if i in self.caps:
            H = h.shape[-1]
            k = self.caps.index(i)
            self.feats[:T, k * H:(k + 1) * H].copy_(h[0])

    def _seg(self, T, j):
        """Segment j of a step split at the attention layers: finish attention layer A[j-1] (its cold part merged
        eagerly in between), the DeltaNet layers after it, then the hot part of attention layer A[j]."""
        A = self.A
        posT = self.pos + torch.arange(T, device=self.dev)
        if j == 0:
            h, lo = self.h_in[:, :T], 0
        else:
            li = A[j - 1]
            layer = self.layers[li]
            o = self.ob[:, :, :T].to(self.hb.dtype).transpose(1, 2).reshape(1, T, -1) * torch.sigmoid(self.gb[:, :T])
            h = self.hb[:, :T] + layer.self_attn.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            self._cap(li, h, T)
            lo = li + 1
        hi = A[j] if j < len(A) else len(self.layers)
        for i in range(lo, hi):
            h = self._layer(i, h, T, None, None, None, posT)
            self._cap(i, h, T)
        if j < len(A):
            li = A[j]
            layer, st = self.layers[li], self.state[li]
            posR = posT + self.rope_delta
            cos, sin = self.model.model.rotary_emb(h, posR.view(1, 1, T).expand(3, 1, T))
            self.hb[:, :T].copy_(h)
            q, gate = self._qkv(layer.self_attn, layer.input_layernorm(h), T, st[1], cos, sin)
            self.qb[:, :, :T].copy_(q)
            self.gb[:, :T].copy_(gate)
            o, lse = self._hot(st[1], q, T, layer.self_attn.scaling)
            self.ob[:, :, :T].copy_(o)
            self.lb[:, :, :T].copy_(lse)
        elif not self.pp_last:
            self.h_out[:, :T].copy_(h)
        else:
            logits = self.model.lm_head(self.model.model.norm(h))
            self.tok_out[:T].copy_(logits[0].argmax(-1))
            self._keep_logits(logits, T)

    def _seg_cold(self, T, j):
        li = self.A[j]
        o, l = self._cold(li, self.qb[:, :, :T], self.ob[:, :, :T], self.lb[:, :, :T],
                          self.layers[li].self_attn.scaling)
        self.ob[:, :, :T].copy_(o)
        self.lb[:, :, :T].copy_(l)

    def _r(self, name, t):
        if self.rec is not None:
            self.rec[(self._li, name)] = t.detach().clone()

    def _commit(self):
        c = self.keep                                                       # tokens kept (1..t_max)
        for st in self.state:
            if st[0] == "gdn":
                conv, rec, hn, hist = st[1:]
                n = conv.shape[-1]
                Sc = hist.index_select(0, c - 1)
                if isinstance(rec, tuple):                                  # encode: min + j * step, j < 2^16
                    ids, mn, step, eidx, evals = rec
                    lo = Sc.amin(2, keepdim=True)
                    span = Sc.amax(2, keepdim=True) - lo
                    st_ = torch.clamp(span / 65535.0, min=2 * PACK_TAU)
                    # the E widest columns are stored exactly (fixed shape: topk); wider-than-16-bit columns beyond
                    # them are counted (each loses the tau bound)
                    flat = span.reshape(-1)
                    top = flat.topk(eidx.numel()).indices
                    eidx.copy_(top)
                    evals.copy_(Sc[0].permute(0, 2, 1).reshape(flat.numel(), -1).index_select(0, top))
                    over = (flat > 2 * PACK_TAU * 65535.0).sum() - eidx.numel()
                    self.state_wide.add_(over.clamp(min=0))
                    mn.copy_(lo)
                    step.copy_(st_)
                    ids.copy_((torch.round((Sc - lo) / st_).clamp(0, 65535) - 32768.0).to(torch.int16))
                else:
                    rec.copy_(Sc)
                idx = (c + torch.arange(n, device=self.dev)).view(1, 1, n).expand(1, hn.shape[1], n)
                conv.copy_(torch.gather(hn, 2, idx))
        self.pos.add_(c)
        for li in self.A:                                                   # qvb8 KV: c keys kept, a full tail sealed
            self.state[li][1].set_len(self.pos - self.base_t)

    # ---------------------------------------------------------------- capture / run
    @torch.no_grad()
    def capture(self, sizes=(1,), warm=2):
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        pool = None
        for T in sorted(set(sizes)):
            with torch.cuda.stream(s):
                for _ in range(warm):
                    self._step(T)
                    self._commit()
            s.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s, pool=pool):
                self._step(T)
            pool = g.pool()
            self.graphs[T] = g
        if self.offload:
            for T in sorted(set(sizes)):
                with torch.cuda.stream(s):
                    for _ in range(warm):
                        for j in range(len(self.A) + 1):
                            self._seg(T, j)
                s.synchronize()
                for j in range(len(self.A) + 1):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=s, pool=pool):
                        self._seg(T, j)
                    self.seg_graphs[(T, j)] = g
        with torch.cuda.stream(s):
            self._commit()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s, pool=pool):
            self._commit()
        self.commit_graph = g
        torch.cuda.synchronize(self.dev)
        self.reset()

    def _embed(self, tokens):
        e = self.model.model.embed_tokens(torch.tensor([tokens], device=self.dev))
        self.h_in[:, :len(tokens)].copy_(e.to(self.dev, non_blocking=True))

    def _keep_logits(self, logits, T):
        """The step's logits for sampling (logits_out; allocated by the warm-up before any graph is captured)."""
        if self.logits_out is None:
            self.logits_out = torch.empty(self.t_max, logits.shape[-1], dtype=torch.float32, device=self.dev)
        self.logits_out[:T].copy_(logits[0])

    def logits(self, T: int):
        """[T, vocab] fp32 logits of the last run (sampling)."""
        return self.logits_out[:T]

    @torch.no_grad()
    def run(self, tokens: list[int]) -> list[int]:
        """Put tokens in (pos .. pos+T-1), return the argmax after each. Does NOT commit."""
        T = len(tokens)
        if self.offload:
            self.spill(T)
        self._embed(tokens)
        self._forward(T)
        return self.tok_out[:T].tolist()

    @torch.no_grad()
    def _forward(self, T: int):
        """Layers on h_in[:, :T] (filled by the caller; spill already done): tok_out, or h_out on a non-last stage."""
        g = self.graphs.get(T)
        if self.base > 0:                                   # cold KV in host RAM: segmented graphs + streamed merge
            for j in range(len(self.A) + 1):
                sg = self.seg_graphs.get((T, j))
                if sg is not None:
                    sg.replay()
                else:
                    self._seg(T, j)
                if j < len(self.A):
                    self._seg_cold(T, j)
        elif g is not None:
            g.replay()
        else:
            self._step(T)

    @torch.no_grad()
    def commit(self, c: int):
        self.keep.fill_(c)
        if self.commit_graph is not None:
            self.commit_graph.replay()
        else:
            self._commit()

    @torch.no_grad()
    def prefill(self, ids: list[int], drafter=None) -> int:
        """All prompt tokens, in chunks of t_max; returns the first generated token."""
        p0 = 0
        while p0 < len(ids):
            part = ids[p0:p0 + self.t_max]
            if len(part) not in self.graphs:
                part = part[:1]                                              # a size without a graph: one at a time
            out = self.run(part)
            if drafter is not None:
                drafter.observe_features(p0, self.feats[:len(part)])
            self.commit(len(part))
            p0 += len(part)
        self.first_logits = self.logits_out[len(part) - 1].clone()
        return out[-1]


def _first(dec, lg) -> int:
    """A prefill's last logits: kept on the decoder for sampling the first token; their argmax is returned."""
    dec.first_logits = lg.float()
    return int(lg.argmax())


def _lcp(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefixCache:
    """Saved prefixes of THIS card's share (TP: each card keeps only its own heads' state).

    KV needs no copy: the static KV buffers keep positions [0, p) of the last conversation and new tokens are written
    after them. The Gated-DeltaNet state cannot be rewound, so it is CHECKPOINTED: at every multiple of `grid` tokens
    during a prefill and at the end of every prompt, each DeltaNet layer's conv window and recurrent state (fp32) go to
    pinned host RAM (as the original engine's --prefix-store host), with the drafter's context. A new request restores
    the latest checkpoint at or before its longest common prefix with the cached sequence and prefills only the rest.
    Every card decides from the token ids alone, so the cards stay in step."""

    def __init__(self, grid: int, max_ckpt: int):
        self.grid, self.max_ckpt = grid, max_ckpt
        self.ids: list[int] = []          # tokens whose KV is in the static buffers, in order
        self.ckpt: dict[int, dict] = {}   # position -> {"conv": [...], "rec": [...], "draft": ...}
        self.marks: set[int] = set()      # extra checkpoint positions of the next prefill (the server: end of last turn)

    def stop(self, p: int, e: int) -> int:
        """End of a prefill chunk [p, e): earlier if a mark lies inside, so the checkpoint lands exactly on it."""
        inside = [m for m in self.marks if p < m < e]
        return min(inside) if inside else e

    def point(self, p: int, n: int) -> bool:
        return p % self.grid == 0 or p == n or p in self.marks

    def save(self, p: int, cache, drafter):
        def host(t):
            h = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            h.copy_(t, non_blocking=True)
            return h
        conv, rec = {}, {}
        for li, L in enumerate(cache.layers):
            if getattr(L, "conv_states", None) is not None and L.conv_states.get(0) is not None:
                conv[li] = host(L.conv_states[0])
                rec[li] = host(L.recurrent_states[0].float())
        self.ckpt[p] = {"conv": conv, "rec": rec,
                        "draft": drafter.export_state() if drafter is not None else None}
        while len(self.ckpt) > self.max_ckpt:                 # drop the oldest grid points, keep the newest
            del self.ckpt[min(k for k in self.ckpt if k != max(self.ckpt))]
        torch.cuda.synchronize()

    def best(self, ids) -> int:
        common = min(_lcp(ids, self.ids), len(ids) - 1)       # at least one token must be prefilled (logits)
        good = [p for p in self.ckpt if p <= common]
        return max(good) if good else 0

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for c in self.ckpt.values()
                   for part in ("conv", "rec") for t in c[part].values())


def tap_features(out, caps):
    """Drafter features [T, len(caps) * H] from an HF forward run with output_hidden_states. HF records each layer's
    INPUT, then the final norm's output: hidden_states[i + 1] = output of layer i (i < NL - 1), hidden_states[NL] =
    last_hidden_state = tap NL ("final_postnorm"). The raw output of the last layer is not recorded."""
    hs = out.hidden_states
    NL = len(hs) - 1
    if NL - 1 in caps:
        raise ValueError(f"drafter tap {NL - 1} (the last layer's raw output) is not in the HF hidden_states")
    return torch.cat([hs[min(i + 1, NL)][0] for i in caps], dim=-1)


@torch.no_grad()
def prefill_hf(dec, ids, drafter=None, chunk=int(os.environ.get("QV_TP_PREFILL_CHUNK", "512"))):
    """The prompt through the (TP-split) HF forward in chunks -- the chunked gated-delta rule, big matmuls -- then
    its KV, conv windows and DeltaNet states copied into the graph buffers (packed if QV_STATE_TAU). With dec.prefix
    (a PrefixCache) only the tokens after the best saved prefix are prefilled.
    Returns the first generated token (greedy; the same on every card)."""
    from transformers import DynamicCache
    model, caps, pc = dec.model, dec.caps, getattr(dec, "prefix", None)
    start = pc.best(ids) if pc is not None else 0
    cache = DynamicCache(config=model.config)
    if start > 0:                                            # seed the HF cache from the static KV + the checkpoint
        ck = pc.ckpt[start]
        for li, st in enumerate(dec.state):
            L = cache.layers[li]
            if st[0] == "gdn":
                cv = ck["conv"][li].to(dec.dev, non_blocking=True)
                rc = ck["rec"][li].to(dec.dev, non_blocking=True)
                L.lazy_initialization(conv_states=cv, recurrent_states=rc, conv_kernel_size=cv.shape[-1])
                L.conv_states[0].copy_(cv)
                L.recurrent_states[0].copy_(rc)
                L.has_previous_state[0] = True
            else:
                L.update(*dec.hot_read(li, start))
        if drafter is not None and ck["draft"] is not None:
            drafter.import_state(ck["draft"])
        for k in [k for k in pc.ckpt if k > start]:          # they belong to the old continuation
            del pc.ckpt[k]
    else:
        dec.reset()
        if pc is not None:
            pc.ckpt.clear()
        if drafter is not None and hasattr(drafter, "_reset_state"):
            drafter._reset_state()
    dec.state_wide.zero_()
    t = torch.tensor([ids], device=dec.dev)
    last, p = None, start
    while p < len(ids):
        e = min(len(ids), (p // chunk + 1) * chunk)         # chunk ends on absolute multiples: grid points are hit
        if pc is not None:
            e = pc.stop(p, e)
        out = model.model(input_ids=t[:, p:e], past_key_values=cache, use_cache=True, output_hidden_states=bool(caps))
        last = out.last_hidden_state[:, -1:]
        if drafter is not None and caps:
            drafter.observe_features(p, tap_features(out, caps))
        p = e
        if pc is not None and pc.point(p, len(ids)):
            pc.save(p, cache, drafter)
    tok = _first(dec, model.lm_head(last)[0, -1])
    n = len(ids)
    for li, st in enumerate(dec.state):
        L = cache.layers[li]
        if st[0] == "gdn":
            conv, rec, hn, hist = st[1:]
            k = conv.shape[-1]
            hn[:, :, 1:1 + k].copy_(L.conv_states[0])
            hist[0].copy_(L.recurrent_states[0][0].float())
        else:
            dec.hot_load(li, L.keys[:, :, :n], L.values[:, :, :n], keep=start)
    del cache
    dec.pos.fill_(n - 1)
    dec.commit(1)                                        # state "after token 1 of hist" = the prompt's; pos = n
    dec.prefill_start = start
    if pc is not None:
        pc.ids = list(ids)
    return tok


@torch.no_grad()
@torch.no_grad()
def prefill_vis(dec, ids, emb, pos, delta, drafter=None, chunk=int(os.environ.get("QV_TP_PREFILL_CHUNK", "512"))):
    """An IMAGE prompt: `emb` [n, H] (the embeddings with every picture's vision features in place, on this card), `pos`
    [3, n] its M-RoPE positions, `delta` the rope shift of everything after it. Through the TP HF forward in chunks,
    then the KV / conv windows / DeltaNet states into the graph buffers as prefill_hf. No prefix cache: the image pad
    ids are the same for every picture, so an id-keyed prefix would match a different image."""
    from transformers import DynamicCache
    model, caps, pc = dec.model, dec.caps, getattr(dec, "prefix", None)
    n = len(ids)
    if n > dec.W:
        raise ValueError(f"an image prompt of {n} tokens is longer than the VRAM KV window ({dec.W})")
    dec.reset()
    if pc is not None:
        pc.ckpt.clear()
        pc.ids = []
    if drafter is not None and hasattr(drafter, "_reset_state"):
        drafter._reset_state()
    dec.state_wide.zero_()
    torch.cuda.empty_cache()                 # a full card (27B on one V100): the cached blocks back before the prompt
    cache = DynamicCache(config=model.config)
    last, p = None, 0
    while p < n:
        e = min(n, p + chunk)
        out = model.model(inputs_embeds=emb[None, p:e].to(dec.dev), position_ids=pos[:, None, p:e].to(dec.dev),
                          past_key_values=cache, use_cache=True, output_hidden_states=bool(caps))
        last = out.last_hidden_state[:, -1:]
        if drafter is not None and caps:
            drafter.observe_features(p, tap_features(out, caps))
        p = e
    tok = _first(dec, model.lm_head(last)[0, -1])
    for li, st in enumerate(dec.state):
        L = cache.layers[li]
        if st[0] == "gdn":
            conv, rec, hn, hist = st[1:]
            k = conv.shape[-1]
            hn[:, :, 1:1 + k].copy_(L.conv_states[0])
            hist[0].copy_(L.recurrent_states[0][0].float())
        else:
            dec.hot_load(li, L.keys[:, :, :n], L.values[:, :, :n])
    del cache
    dec.pos.fill_(n - 1)
    dec.commit(1)
    dec.rope_delta.fill_(int(delta))
    dec.prefill_start = 0
    return tok


def prefill_off(dec, ids, drafter=None, chunk=int(os.environ.get("QV_TP_PREFILL_CHUNK", "512"))):
    """Long-context prefill with KV offload: DeltaNet layers through the HF chunked kernel (state in a DynamicCache),
    attention layers through our own hot (VRAM) + cold (host RAM, streamed) attention, merged exactly by LSE. The KV
    goes straight into the decoder's hot window; older positions spill to host RAM. Prefix cache as prefill_hf."""
    from transformers import DynamicCache
    model, caps, pc = dec.model, dec.caps, getattr(dec, "prefix", None)
    start = pc.best(ids) if pc is not None else 0
    cache = DynamicCache(config=model.config)
    if start > 0:
        ck = pc.ckpt[start]
        for li, st in enumerate(dec.state):
            if st[0] == "gdn":
                L = cache.layers[li]
                cv = ck["conv"][li].to(dec.dev, non_blocking=True)
                rc = ck["rec"][li].to(dec.dev, non_blocking=True)
                L.lazy_initialization(conv_states=cv, recurrent_states=rc, conv_kernel_size=cv.shape[-1])
                L.conv_states[0].copy_(cv)
                L.recurrent_states[0].copy_(rc)
                L.has_previous_state[0] = True
        if drafter is not None and ck["draft"] is not None:
            drafter.import_state(ck["draft"])
        for k in [k for k in pc.ckpt if k > start]:
            del pc.ckpt[k]
        dec.truncate(start)
    else:
        dec.reset()
        if pc is not None:
            pc.ckpt.clear()
        if drafter is not None and hasattr(drafter, "_reset_state"):
            drafter._reset_state()
    dec.state_wide.zero_()
    n = len(ids)
    last, p = None, start
    while p < n:
        e = min(n, (p // chunk + 1) * chunk)
        if pc is not None:
            e = pc.stop(p, e)
        T = e - p
        dec.pos.fill_(p)
        dec.spill(T)
        h = model.model.embed_tokens(torch.tensor([ids[p:e]], device=dec.dev))
        posT = torch.arange(p, e, device=dec.dev)
        cos, sin = model.model.rotary_emb(h, posT.view(1, 1, T).expand(3, 1, T))
        feats = []
        for i, (layer, st) in enumerate(zip(dec.layers, dec.state)):
            x = layer.input_layernorm(h)
            if st[0] == "gdn":
                o = layer.linear_attn(hidden_states=x, cache_params=cache, attention_mask=None)
            else:
                m = layer.self_attn
                q, gate, k, v = dec._proj(m, x, T, cos, sin)
                kv = st[1]
                ob, lb = dec._attn_lse(q, k, v, m.scaling, torch.ones(T, T, dtype=torch.bool, device=dec.dev).tril())
                if int(kv.n[0]):                             # the hot window before this chunk: all of it visible
                    o_, l_ = kv.attn(q, kv.nt.view(1, 1).expand(1, T), m.scaling)
                    ob, lb = dec._merge(ob, lb, o_, l_)
                ob, lb = dec._cold(i, q, ob, lb, m.scaling)
                kv.append(0, k, v)
                o = ob.to(h.dtype).transpose(1, 2).reshape(1, T, -1) * torch.sigmoid(gate)
                o = m.o_proj(o)
            h = h + o
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            if i in caps:
                feats.append(h[0])
        last = h[:, -1:]
        if drafter is not None and caps:
            drafter.observe_features(p, torch.cat(feats, dim=-1))
        p = e
        if pc is not None and pc.point(p, n):
            pc.save(p, cache, drafter)
    tok = _first(dec, model.lm_head(model.model.norm(last))[0, -1])
    for li, st in enumerate(dec.state):
        if st[0] == "gdn":
            L = cache.layers[li]
            conv, rec, hn, hist = st[1:]
            k = conv.shape[-1]
            hn[:, :, 1:1 + k].copy_(L.conv_states[0])
            hist[0].copy_(L.recurrent_states[0][0].float())
    del cache
    dec.pos.fill_(n - 1)
    dec.commit(1)
    dec.prefill_start = start
    if pc is not None:
        pc.ids = list(ids)
    return tok


def _group_first(tp) -> bool:
    return tp is None or tp.world == 1 or torch.distributed.get_rank() == (
        torch.distributed.get_global_rank(tp.group, 0) if tp.group is not None else 0)


def _agree(tp, vals, dev):
    """The group's first card's values on every card (a sampled decision: random on one card only)."""
    if tp is None or tp.world == 1:
        return vals
    buf = torch.tensor(vals, dtype=torch.long, device=dev)
    src = torch.distributed.get_global_rank(tp.group, 0) if tp.group is not None else 0
    torch.distributed.broadcast(buf, src, group=tp.group)
    return buf.tolist()


def generate(dec, ids, new, drafter=None, tp=None, eos=(), on_tokens=None, prefill="graph", vis=None,
             temperature=0.0, top_k=0, top_p=1.0, stop=None):
    """stop(): read on the group's first card every few steps (the client left) and agreed by every card, so all
    cards end the request together. Greedy, or sampled (temperature > 0, top_k, top_p); with a drafter speculative and exact either way (greedy:
    the target's own greedy text; sampled: speculative sampling, the target's own distribution -- engine/spec.py).
    A sampled decision is made on the group's first card and broadcast. on_tokens(list): called with each step's new
    tokens (streaming). prefill: "graph" (8 tokens per replay) or "hf" (chunked HF forward, for long prompts)."""
    from qv_band.engine.spec import accept, sample_token, verify_row
    sampled = temperature > 0
    pc = getattr(dec, "prefix", None)
    if drafter is not None and hasattr(drafter, "_reset_state") and not (prefill == "hf" and pc is not None):
        drafter._reset_state()                 # with a prefix cache, prefill_hf restores or resets the drafter itself
    t_pre = time.time()
    dec.rope_delta.zero_()                     # text; an image prompt sets its own in prefill_vis
    if vis is not None:                        # (emb [n, H], pos [3, n], delta): an image prompt
        tok = prefill_vis(dec, ids, *vis, drafter=drafter)
    elif prefill == "hf" and getattr(dec, "offload", False):
        if os.environ.get("QV_TP_LAYER_MAJOR", "1") == "1":
            from qv_band.run.tp_longctx import prefill_layer_major
            tok = prefill_layer_major(dec, ids, drafter)
        else:
            tok = prefill_off(dec, ids, drafter)
    elif prefill == "hf":
        tok = prefill_hf(dec, ids, drafter)
    else:
        dec.reset()
        tok = dec.prefill(ids, drafter)
    torch.cuda.synchronize(dec.dev)
    t_pre = time.time() - t_pre
    if sampled:
        tok = _agree(tp, [sample_token(dec.first_logits, temperature, top_k, top_p) if _group_first(tp) else 0],
                     dec.dev)[0]
    if on_tokens is not None:
        on_tokens([tok])
    out, steps, proposed, accepted = [tok], 0, 0, 0
    t_draft = t_verify = 0.0
    if drafter is not None:
        drafter.reset(ids + [tok])
        drafter.pos = len(ids) + 1
    k = dec.t_max - 1
    # Draft or not (engine/spec.SpecPlanner: this request's measured acceptance against the measured cost of a
    # 1-token and a (k+1)-token step; the graphs verify all k drafts or none). Card 0 decides: its drafts (or none)
    # reach every card each step, so the cards stay in step. Output is unchanged either way (exact verification).
    from qv_band.engine.spec import SpecPlanner
    plan = getattr(dec, "spec_planner", None)              # one per decoder: the costs are this card's
    if plan is None:
        plan = dec.spec_planner = SpecPlanner()
    plan.forget(0)                                         # a new request
    multi = tp is not None and tp.world > 1
    while len(out) < new and out[-1] not in eos:
        drafts, dl = [], None
        torch.cuda.synchronize(dec.dev)
        t0 = time.time()
        drafting = drafter is not None and plan.plan_k([(0, k)], whole=True)[0] > 0
        want = bool(stop()) if (stop is not None and steps % 4 == 0 and _group_first(tp)) else False
        if multi:
            # card 0's decisions on every card: stop (the client left) and draft-or-not. The planner times each card
            # on its own clock, so the cards may disagree, and a split drafter's propose() is a collective: one card
            # drafting alone hung both (seen 2026-11-28, V100 pair, 100% at 60 W).
            want, drafting = (bool(v) for v in _agree(tp, [int(want), int(drafting)], dec.dev))
        if want:
            break
        if drafter is not None:
            if drafting:
                drafts, dl = drafter.propose(out[-1], k, temperature, top_k, top_p)
                drafts = list(drafts)[:k]
            if tp is not None and tp.world > 1:                              # card 0's block on every card
                buf = torch.full((k + 1,), -1, dtype=torch.long, device=dec.dev)
                buf[0] = len(drafts)
                if drafts:
                    buf[1:1 + len(drafts)] = torch.tensor(drafts, device=dec.dev)
                src = torch.distributed.get_global_rank(tp.group, 0) if tp.group is not None else 0
                torch.distributed.broadcast(buf, src, group=tp.group)   # the group's first card
                n = int(buf[0])
                drafts = buf[1:1 + n].tolist()
            if len(drafts) not in (0, k):                                   # graphs exist for 1 and k+1 tokens
                drafts = []
        torch.cuda.synchronize(dec.dev)
        t1 = time.time()
        t_draft += t1 - t0
        tgt = dec.run([out[-1]] + drafts)
        if sampled:
            n_acc, nxt = verify_row(drafts, dl if drafts else None, dec.logits(len(drafts) + 1), temperature, top_k,
                                    top_p, getattr(drafter, "returns_probs", False)) if _group_first(tp) else (0, 0)
            n_acc, nxt = _agree(tp, [n_acc, nxt], dec.dev)
        else:
            n_acc = accept(drafts, tgt)
            nxt = tgt[n_acc]
        dec.commit(n_acc + 1)
        if drafter is not None:
            drafter.observe_features(drafter._ctx_end, dec.feats[:n_acc + 1])
            drafter.pos = drafter._ctx_end + 1
        torch.cuda.synchronize(dec.dev)
        t_verify += time.time() - t1
        if drafts:
            plan.observe(plan.pcost, 1, t1 - t0)
            plan.feedback(0, n_acc, len(drafts))
        plan.observe(plan.vcost, 1 + len(drafts), time.time() - t1)
        new_toks = drafts[:n_acc] + [nxt]
        cut = next((j for j, t_ in enumerate(new_toks) if t_ in eos), None)
        if cut is not None:
            new_toks = new_toks[:cut + 1]
        new_toks = new_toks[:max(0, new - len(out))]
        if on_tokens is not None and new_toks:
            on_tokens(new_toks)
        out.extend(new_toks)
        steps += 1
        proposed += len(drafts)
        accepted += n_acc
    pc = getattr(dec, "prefix", None)
    if pc is not None and prefill == "hf" and vis is None:
        pc.ids = list(ids) + out[:-1]          # the last token was produced but never put into the KV
    return out[:new], dict(steps=steps, proposed=proposed, accepted=accepted, t_draft=t_draft, t_verify=t_verify,
                           t_prefill=t_pre, cached=getattr(dec, "prefill_start", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=None, help="DFlash drafter (dir or qvb8 pack)")
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--block", type=int, default=8, help="verify / prefill graph size (drafts = block - 1)")
    ap.add_argument("--tp-weights", default=None)
    ap.add_argument("--draft-replicated", action="store_true", help="whole drafter on every card (no split)")
    ap.add_argument("--verify-k", type=int, default=0, help="verify only the first k drafts (0: the whole block)")
    ap.add_argument("--eager", action="store_true", help="no graphs (to compare)")
    ap.add_argument("--prefill", default="graph", choices=("graph", "hf"))
    ap.add_argument("--prompt-file", default=None, help="read the prompt from a file (overrides --prompt)")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--kv-window", type=int, default=None, help="hot KV slots in VRAM; older KV in host RAM")
    a = ap.parse_args()

    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local}"
    torch.cuda.set_device(device)
    from qv_band.formats.qvb8.tp import init_distributed
    tp = init_distributed(device)                     # one machine, or several (QV_TP_XNODE)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    weights = [float(x) for x in a.tp_weights.split(",")] if a.tp_weights else None
    model, tcfg = build(a.model, tp, device, weights=weights, verbose=(tp.rank == 0))
    drafter = None
    if a.draft:
        from qv_band.decode.dflash_draft import build_dflash_drafter
        drafter = build_dflash_drafter(model, a.draft, device=device)
        a.block = drafter.block_size if a.verify_k <= 0 else min(drafter.block_size, a.verify_k + 1)
        if not a.draft_replicated:
            from qv_band.run.tp_dflash import split_drafter
            split_drafter(drafter, tp, weights)
    dec = StaticDecoder(model, tcfg, device, a.max_len, t_max=a.block,
                        capture_layers=drafter.capture_layers if drafter else (), kv_window=a.kv_window)
    t0 = time.time()
    if not a.eager and tp.graphable:
        dec.capture(sizes=(1, a.block))
    t_cap = time.time() - t0

    if a.prompt_file:
        a.prompt = open(a.prompt_file).read()
    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text).input_ids
    eos = set(x for x in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")] if x is not None)
    if drafter is not None:                                                 # warm the drafter's own graph
        generate(dec, ids, 12, drafter, tp, eos, prefill=a.prefill)
        dec.reset()
        drafter._reset_state()
    torch.cuda.synchronize(device)
    t0 = time.time()
    out, st = generate(dec, ids, a.new, drafter, tp, eos, prefill=a.prefill)
    torch.cuda.synchronize(device)
    dt = time.time() - t0
    if tp.rank == 0:
        acc = (f", accepted {st['accepted']}/{st['proposed']} drafts in {st['steps']} steps" if drafter else "")
        acc += (f"; per step: draft {st['t_draft'] / max(1, st['steps']) * 1e3:.1f} ms, "
                f"verify {st['t_verify'] / max(1, st['steps']) * 1e3:.1f} ms")
        print(f"[tp-graph] prefill {a.prefill}: {st.get('t_prefill', 0):.2f}s for {len(ids)} tokens", flush=True)
        print(f"[tp-graph] world {tp.world}: capture {t_cap:.1f}s, {len(ids)} prompt + {len(out)} new tokens in "
              f"{dt:.2f}s = {len(out) / dt:.2f} tok/s{acc}, peak VRAM "
              f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB", flush=True)
        print("[tp-graph] text:", repr(tok.decode(out)), flush=True)
        if a.dump:
            torch.save({"tokens": out}, a.dump)
    if tp.world > 1:
        # destroy_process_group() can deadlock while captured graphs still hold the NCCL communicator: every card
        # reaches the barrier, then the process ends without tearing NCCL down.
        torch.distributed.barrier()
        torch.cuda.synchronize(device)
        import sys
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
