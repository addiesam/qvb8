"""Inkling-2-Mini (inkling2_loop) forward for the engine: no HF modeling, no HF generate.

Weights come straight from the pack (checkpoint tensor names):
  big linears      qv_band QVBandLinearV8 (qvb8, certified)
  routed experts   the "qvbq" format's experts class (NVFP4 / MXFP8 bit-exact, on-the-fly matmul)
  the rest         the pack's dense bundle, or a qvb8 tensor decoded once (certified restore)
Attention: attention.inkling_attention (compact relative bias + sinks + window, O(L) memory).

RAGGED BATCH. forward() takes several requests at once (rows of different lengths, each with its own
state): their tokens are concatenated, so every linear, the router and the experts read their weights once
for the whole batch. Only the two things that cross tokens run per request: attention over that request's
K/V, and the short convolutions over that request's last K-1 inputs.

State of one request ("slot"), per (layer, loop pass): K, V (sliding layers keep the last window-1 tokens),
k0 (position of the first kept key), and the last K-1 inputs of the k / v / post-attention / post-MLP
convolutions; plus "pos" (tokens consumed). Block states, attention residuals, message gates and routing
are per token. The maths follows the reference implementation op for op (fp32 where it is fp32).

CARDS.
  dtype        bf16 (sm >= 80) or fp16 (V100, sm_70: no bf16 tensor cores). Matmul inputs are normed / clipped /
               softcapped values; with fp16 the residual stream and block states are kept fp32, so nothing that can
               grow is ever held in fp16.
  tp           one card of a Tensor Parallel group (run/tp_dit.ThreadTP over a Hub: one thread per card): attention
               split by heads (q|gate|rel|sink rows and K/V rows by whole KV-head groups, formats/qvb8/tp.slice_rows;
               wo by its input columns, slice_cols), routed experts split by ranges (expert parallel); one all-reduce
               after wo and one after the experts. Everything else (embedding, norms, block / message residuals,
               router, short convs over the hidden size, lm_head) is replicated, so every card holds the same hidden
               states.
  experts      "resident" (the stack on the card) or "host" (formats/qvbq QVBQHostExperts: the stack in page-locked
               host RAM, the experts in use cached on the card; one 16 GB V100).
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

from qv_band.layers.fused_rownorm import block_residual, gate_in, gate_norm, l2clip, rms
from qv_band.layers.fused_short_conv import short_conv1, short_conv_r

from .attention import inkling_attention


def default_dtype(device) -> torch.dtype:
    """bf16 where the card has bf16 tensor cores (sm >= 80), else fp16 (V100)."""
    return torch.bfloat16 if torch.cuda.get_device_capability(torch.device(device))[0] >= 8 else torch.float16


# ----------------------------------------------------------------------------------------- qvb8 KV
class QVB8KV:
    """Growing K or V of a full-attention layer, [1, Hk, L, D]: whole blocks of BLOCK tokens are stored in the
    engine's certified qvb8 KV codec (kv/codec.py, per-channel min_s + id*tau_s, |err| <= tau); the newest
    < 2*BLOCK tokens stay bf16. Attention reads the decoded bf16 tensor; the saving is what stays resident.
    Sealed blocks are never changed, so a clone (prefix snapshot) shares them and copies only the bf16 tail."""

    BLOCK = 1024

    def __init__(self, tau, blocks=(), tail=None):
        self.tau, self.blocks, self.tail = tau, list(blocks), tail

    def append(self, x):
        self.tail = x if self.tail is None else torch.cat([self.tail, x], 2)
        n = self.tail.shape[2]
        if n >= 2 * self.BLOCK:
            from qv_band.kv.codec import band_kv_heads
            cut = (n - self.BLOCK) // self.BLOCK * self.BLOCK
            for b in range(0, cut, self.BLOCK):
                self.blocks.append(band_kv_heads(self.tail[0, :, b:b + self.BLOCK], self.tail.dtype, 6, self.tau))
            self.tail = self.tail[:, :, cut:].clone()

    def dense(self):
        if not self.blocks:
            return self.tail
        dev = self.tail.device
        parts = [torch.stack([h.decode_all_torch(dev, self.tail.dtype) for h in blk]).unsqueeze(0)
                 for blk in self.blocks]
        return torch.cat(parts + [self.tail], 2)

    def clone(self):
        return QVB8KV(self.tau, self.blocks, None if self.tail is None else self.tail.clone())

    def drop(self, n):
        """Remove the newest n tokens (a rejected speculative tail: always in the bf16 tail, >= BLOCK tokens)."""
        if n:
            self.tail = self.tail[:, :, :self.tail.shape[2] - n].clone()

    def nbytes(self):
        n = sum(h.nbytes() for blk in self.blocks for h in blk)
        return n + (0 if self.tail is None else self.tail.numel() * self.tail.element_size())


# ----------------------------------------------------------------------------------------- small ops
def softcap(x, c):
    return torch.tanh(x / c) * c


class SoftcappedSwiGLU(torch.nn.Module):
    """act(gate, up) = softcap(gate, 4) * sigmoid(gate) * softcap(up, 25), in fp32."""

    def __init__(self, gate_cap, up_cap):
        super().__init__()
        self.gc, self.uc = gate_cap, up_cap

    def forward(self, gate, up):
        dt = gate.dtype
        g, u = gate.float(), up.float()
        return (softcap(g, self.gc) * torch.sigmoid(g) * softcap(u, self.uc)).to(dt)


def short_conv(x, state, w, K, full=False):
    """(conv + x) / sqrt(K + 1), depthwise causal. x [L, C], state [K-1, C] fp32 -> (y in x.dtype, state); full: the
    whole history [K-1+L, C] as the state (a speculative verify: InklingModel.trim keeps the accepted part)."""
    dt = x.dtype
    xf = x.float()
    ctx = torch.cat([state, xf], dim=0)
    win = ctx.unfold(0, K, 1)                                         # [L, C, K]; last = current
    y = (win * w.float().unsqueeze(0)).sum(-1)
    return ((y + xf) / math.sqrt(K + 1)).to(dt), ctx if full else ctx[-(K - 1):].clone()   # clone: a view keeps ctx


def dyn_conv(x, dw, state, static_w, n_heads, K, full=False):
    """Dynamic short conv (full-attention K/V): taps = static_w + per-token dw [L, n_heads*K]."""
    dt = x.dtype
    L, C = x.shape
    w = static_w.float().T.reshape(n_heads, C // n_heads, K) + dw.float().view(L, n_heads, 1, K)
    xf = x.float()
    ctx = torch.cat([state, xf], dim=0)
    win = ctx.unfold(0, K, 1).flip(-1).reshape(L, n_heads, C // n_heads, K)     # lag 0 = current
    y = (win * w).sum(-1).reshape(L, C)
    return ((y + xf) / math.sqrt(2 * K + 1)).to(dt), ctx if full else ctx[-(K - 1):].clone()


def dyn_conv1(x, dw, state, static_w, n_heads, K, active):
    """dyn_conv for ONE token per slot (taps static_w + per-token dw [S, n_heads*K])."""
    from qv_band.engine.static_decode import conv_push
    dt = x.dtype
    S, C = x.shape
    w = static_w.float().T.reshape(n_heads, C // n_heads, K).unsqueeze(0) + dw.float().view(S, n_heads, 1, K)
    xf = x.float()
    ctx = torch.cat([state, xf.unsqueeze(1)], 1)                      # [S, K, C]
    win = ctx.transpose(1, 2).flip(-1).reshape(S, n_heads, C // n_heads, K)     # lag 0 = current
    y = (win * w).sum(-1).reshape(S, C)
    conv_push(state, xf, active)
    return ((y + xf) / math.sqrt(2 * K + 1)).to(dt)


def dyn_conv_r(x, dw, state, static_w, n_heads, K, scratch, slot, j):
    """dyn_conv for the ragged rows of a speculative verify (row r = token j[r] of slot slot[r]): x [R, C],
    dw [R, n_heads*K], state [S, K-1, C] fp32 (NOT changed; x fp32 goes to scratch [S, Ts, C] at (slot, j) for
    engine/static_decode.conv_commit) -> [R, C]."""
    dt = x.dtype
    R, C = x.shape
    xf = x.float()
    scratch[slot, j] = xf
    ctx = torch.cat([state, scratch], 1)                              # [S, K-1+Ts, C]
    win = ctx[slot[:, None], j[:, None] + torch.arange(K, device=x.device)[None]]    # [R, K, C], oldest first
    win = win.flip(1).transpose(1, 2).reshape(R, n_heads, C // n_heads, K)           # lag 0 = current
    w = static_w.float().T.reshape(n_heads, C // n_heads, K) + dw.float().view(R, n_heads, 1, K)
    y = (win * w).sum(-1).reshape(R, C)
    return ((y + xf) / math.sqrt(2 * K + 1)).to(dt)


class StaticCtx:
    """The slot buffers of a static decode step: kv[(layer, pass)] StaticKV / StaticQKV, conv[(layer, pass)] {kc vc ac mc},
    active [S] (slots in the step). A speculative verify (verify = True) runs RAGGED rows: row r = token j[r] of slot
    slot[r] (a slot's rows contiguous from start[s]), row_active [R]; scratch[(layer, pass)] {kc vc ac mc} [S, Tmax+1, C]
    holds the conv inputs for the commit (row Tmax: the padding rows')."""

    def __init__(self):
        self.kv, self.conv, self.scratch, self.active, self.row_active = {}, {}, {}, None, None
        self.verify, self.slot, self.j, self.start, self.Tmax = False, None, None, None, 0


# ----------------------------------------------------------------------------------------- weights
class Weights:
    """Pack reader by checkpoint name: dense bundle first, else the certified qvb8 restore."""

    def __init__(self, pack, device, dtype=torch.bfloat16):
        self.pack, self.device, self.dt = pack, device, dtype
        self.mani = json.load(open(os.path.join(pack, "manifest.json")))["tensors"]
        self.bundle = safe_open(os.path.join(pack, "model.safetensors"), "pt", device=str(device))
        self.bkeys = set(self.bundle.keys())

    def dense(self, k, dtype=None, rows=None, cols=None):
        """The tensor by name in `dtype` (default: the model's); rows / cols = (a, b): only that range of dim 0 / 1."""
        dtype = dtype or self.dt
        if k in self.bkeys:
            w = self.bundle.get_tensor(k)
        else:
            from qv_band.formats.qvb8.module import QVBandLinearV8, _load_ext
            lin = QVBandLinearV8.from_packfile(os.path.join(self.pack, self.mani[k]["pack"]), device=self.device)
            if dtype == torch.bfloat16:      # certified bf16 restore: the fp32 band value rounded once
                w = _load_ext().decode_bf16(lin.ids, lin.min_s, lin.nbits, lin.bit_off, lin.in_features,
                                            lin.tau_s, lin.esc_idx_bf16, lin.esc_vals_bf16)
            else:                            # fp32 band value (fp16: rounded once below)
                from qv_band.common.formats import get_format
                w = get_format("qvb8").decoder(lin).float()
        if rows is not None:
            w = w[rows[0]:rows[1]]
        if cols is not None:
            w = w[:, cols[0]:cols[1]]
        return w.to(dtype).contiguous()

    def linear(self, k, rows=None, cols=None):
        """Matmul module: the qvb8 linear itself (never densified); rows / cols = (a, b): a TP share of it
        (formats/qvb8/tp.slice_rows / slice_cols: the bits of those rows / columns only)."""
        from qv_band.formats.qvb8.module import QVBandLinearV8
        if k in self.mani and self.mani[k].get("format") != "qvbq":
            lin = QVBandLinearV8.from_packfile(os.path.join(self.pack, self.mani[k]["pack"]), device=self.device)
            if rows is not None or cols is not None:
                from qv_band.formats.qvb8.tp import slice_cols, slice_rows
                full = lin
                lin = slice_rows(full, rows, device=self.device) if rows is not None else \
                    slice_cols(full, cols[0], cols[1], device=self.device)
                del full
            return lin
        w = self.dense(k, rows=rows, cols=cols)
        return lambda x: F.linear(x.to(w.dtype), w)


def _lin(mod, x):
    return mod(x).to(x.dtype)


# ----------------------------------------------------------------------------------------- layer
class Layer:
    def __init__(self, W: Weights, c, i, experts, tp=None):
        p, a = f"model.llm.layers.{i}.", f"model.llm.layers.{i}.attn."
        self.i, self.c = i, c
        self.dt, self.rd = c["dtype"], c["res_dtype"]
        self.tp = tp
        self.sliding = i in c["local_layer_ids"]
        self.loop = c["loop_start"] <= i < c["loop_end"]
        self.boundary = i % c["n_layers_per_attn_res_block"] == 0
        Hf, D, K = c["heads"], c["head_dim"], c["conv"]
        # this card's heads: [h0, h1) of the q heads and [g0, g1) of the KV heads (whole GQA groups)
        (h0, h1), (g0, g1) = c["head_range"], c["kv_range"]
        self.H, self.Hk, self.D, self.K = h1 - h0, g1 - g0, D, K
        self.W = c["window"] if self.sliding else None
        self.kv_tau = c.get("kv_tau")                      # not None: full layers keep K/V in qvb8
        split = tp is not None and tp.world > 1
        qr, kr = ((h0 * D, h1 * D), (g0 * D, g1 * D)) if split else (None, None)
        hr = (h0, h1) if split else None
        self.wq, self.gate = W.linear(a + "wq_du.weight", rows=qr), W.linear(a + "gate_attn.weight", rows=qr)
        self.wk, self.wv = W.linear(a + "wk_dv.weight", rows=kr), W.linear(a + "wv_dv.weight", rows=kr)
        self.wo = W.linear(a + "wo_ud.weight", cols=qr)
        self.sink = W.linear(a + "attn_sink_proj.weight", rows=hr)
        self.rel_scale = W.dense(a + "rel_l2_clip.scale", rows=hr)                       # [H, R]
        R = self.rel_scale.shape[-1]
        self.wr = W.linear(a + "wr_du.weight", rows=(h0 * R, h1 * R) if split else None)
        qs = W.dense(a + "q_l2_clip.scale")
        self.q_scale = qs[h0:h1].contiguous() if (split and qs.dim() >= 2 and qs.shape[0] == Hf) else qs
        self.rel_proj = W.dense(a + "rel_logits_proj.proj")                              # shared by the heads
        if self.sliding:
            self.k_conv = W.dense(a + "k_sconv.weight", torch.float32).reshape(-1, K)[g0 * D:g1 * D].contiguous()
            self.v_conv = W.dense(a + "v_sconv.weight", torch.float32).reshape(-1, K)[g0 * D:g1 * D].contiguous()
            self.in_norm = W.dense(p + "attn_norm.weight")
        else:
            self.k_static = W.dense(a + "k_sconv.static_w", cols=(g0 * D, g1 * D))
            self.v_static = W.dense(a + "v_sconv.static_w", cols=(g0 * D, g1 * D))
            wr_ = (g0 * K, g1 * K) if split else None                                       # K taps per KV head
            self.k_wproj = W.linear(a + "k_sconv.weight_proj", rows=wr_)
            self.v_wproj = W.linear(a + "v_sconv.weight_proj", rows=wr_)
            self.q_norm, self.k_norm = W.dense(p + "q_input_norm.weight"), W.dense(p + "k_input_norm.weight")
            self.v_norm = W.dense(p + "v_input_norm.weight")
        self.attn_conv = W.dense(p + "attn_sconv.weight", torch.float32).reshape(-1, self.K)
        self.mlp_conv = W.dense(p + "mlp_sconv.weight", torch.float32).reshape(-1, self.K)
        self.mlp_norm = W.dense(p + "mlp_norm.weight")
        self.router = W.dense(p + "mlp.gate.weight", torch.float32)
        self.router_bias = W.dense(p + "mlp.gate.bias", torch.float32)
        self.experts = experts
        self.attn_q = (W.dense(f"model.llm.attn_res.attn_queries.{i - 1}", torch.float32).reshape(-1, c["hidden"])
                       if i > 0 else None)
        self.mlp_q = W.dense(f"model.llm.attn_res.mlp_queries.{i}", torch.float32).reshape(-1, c["hidden"])
        if self.loop:
            g = p + "controllers."
            self.gates = {n: (W.dense(g + ck + ".down_weight"),
                              torch.cat([W.dense(g + ck + ".forget_weight", torch.float32),
                                         W.dense(g + ck + ".gain_weight", torch.float32)]))     # forget | gain rows
                          for n, ck in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp"))}

    def reduce(self, y):
        """Sum of the cards' partial outputs (TP), the output itself on one card."""
        return y if self.tp is None or self.tp.world == 1 else self.tp.all_reduce(y)

    def msg_gate_norm(self, name, h, m, w):
        """rms(message gate(h, m)) * w in the matmul dtype: gate = (1 - tanh(forget)) * h + tanh(gain) * m."""
        down, fg = self.gates[name]
        g = F.linear(gate_in(h, m, self.c["eps"], down.dtype), down).float()
        return gate_norm(F.linear(g, fg), h, m, w, self.c["eps"], self.dt)

    def _clip(self, x, scale, thr, pos):
        if self.sliding:
            return l2clip(x, thr, scale)
        return l2clip(x, thr, scale, pos, self.c["alpha"], self.c["n_floor"])

    def attention(self, hq, hk, hv, pos, segs, p):
        """hq/hk/hv [T, hidden] bf16 (all rows); segs: [(start, end, slot state)]; p: loop pass."""
        T = hq.shape[0]
        H, Hk, D, K = self.H, self.Hk, self.D, self.K
        q = self._clip(_lin(self.wq, hq).view(T, H, D), self.q_scale, self.c["qk_thr"], pos)
        rs = self._clip(_lin(self.wr, hq).view(T, H, -1), self.rel_scale, self.c["rel_thr"], pos)
        rel = rs @ self.rel_proj                                                  # [T, H, E]
        sinks = _lin(self.sink, hq)                                               # [T, H]
        k_raw, v_raw = _lin(self.wk, hk), _lin(self.wv, hv)
        if not self.sliding:
            kdw, vdw = _lin(self.k_wproj, hk), _lin(self.v_wproj, hv)
        out = torch.empty(T, H * D, dtype=self.dt, device=hq.device)
        for s, e, slot in segs:
            st = slot[(self.i, p)]
            spec = "spec" in slot
            if self.sliding:
                k, st["kc"] = short_conv(k_raw[s:e], st["kc"], self.k_conv, K, spec)
                v, st["vc"] = short_conv(v_raw[s:e], st["vc"], self.v_conv, K, spec)
            else:
                k, st["kc"] = dyn_conv(k_raw[s:e], kdw[s:e], st["kc"], self.k_static, Hk, K, spec)
                v, st["vc"] = dyn_conv(v_raw[s:e], vdw[s:e], st["vc"], self.v_static, Hk, K, spec)
            k = l2clip(k.view(e - s, Hk, D), self.c["qk_thr"]).transpose(0, 1).unsqueeze(0)   # [1, Hk, L, D]
            v = rms(v.view(e - s, Hk, D), eps=self.c["eps"]).transpose(0, 1).unsqueeze(0)
            p0 = slot["pos"]
            if self.kv_tau is not None and not self.sliding:        # qvb8 KV (full layers: the growing cache)
                if st["k"] is None:
                    st["k"], st["v"], st["k0"] = QVB8KV(self.kv_tau), QVB8KV(self.kv_tau), p0
                st["k"].append(k)
                st["v"].append(v)
                o = inkling_attention(q[s:e].transpose(0, 1).unsqueeze(0), st["k"].dense(), st["v"].dense(),
                                      rel[s:e].transpose(0, 1).unsqueeze(0), sinks[s:e].T.unsqueeze(0),
                                      p0 - st["k0"], D ** -0.5, None)
                out[s:e] = o.reshape(e - s, H * D)
                continue
            if st["k"] is None:
                kk, vv, k0 = k, v, p0
            else:
                kk, vv, k0 = torch.cat([st["k"], k], 2), torch.cat([st["v"], v], 2), st["k0"]
            o = inkling_attention(q[s:e].transpose(0, 1).unsqueeze(0), kk, vv,
                                  rel[s:e].transpose(0, 1).unsqueeze(0), sinks[s:e].T.unsqueeze(0),
                                  p0 - k0, D ** -0.5, self.W)
            if self.W is not None and kk.shape[2] > self.W - 1 and not spec:     # keep the last W-1 keys
                cut = kk.shape[2] - (self.W - 1)
                kk, vv, k0 = kk[:, :, cut:].clone(), vv[:, :, cut:].clone(), k0 + cut
            st["k"], st["v"], st["k0"] = kk, vv, k0
            out[s:e] = o.reshape(e - s, H * D)
        g = torch.sigmoid(_lin(self.gate, hq).float())
        return self.reduce(_lin(self.wo, (out.float() * g).to(self.dt)))

    # --- static decode (one token per slot, fixed buffers; graph-safe) -------------------------------------
    def attention_static(self, hq, hk, hv, pos, sc, p):
        """attention() for S slots x 1 token on the slot buffers (sc: StaticCtx); sc.verify: its ragged rows."""
        if sc.verify:
            return self._attention_verify(hq, hk, hv, pos, sc, p)
        S = hq.shape[0]
        H, Hk, D, K = self.H, self.Hk, self.D, self.K
        q = self._clip(_lin(self.wq, hq).view(S, H, D), self.q_scale, self.c["qk_thr"], pos)
        rs = self._clip(_lin(self.wr, hq).view(S, H, -1), self.rel_scale, self.c["rel_thr"], pos)
        rel = rs @ self.rel_proj
        sinks = _lin(self.sink, hq)
        k_raw, v_raw = _lin(self.wk, hk), _lin(self.wv, hv)
        cs = sc.conv[(self.i, p)]
        if self.sliding:
            k = short_conv1(k_raw, cs["kc"], self.k_conv, K, sc.active)
            v = short_conv1(v_raw, cs["vc"], self.v_conv, K, sc.active)
        else:
            k = dyn_conv1(k_raw, _lin(self.k_wproj, hk), cs["kc"], self.k_static, Hk, K, sc.active)
            v = dyn_conv1(v_raw, _lin(self.v_wproj, hv), cs["vc"], self.v_static, Hk, K, sc.active)
        k = l2clip(k.view(S, Hk, D), self.c["qk_thr"]).unsqueeze(2)
        v = rms(v.view(S, Hk, D), eps=self.c["eps"]).unsqueeze(2)
        kv = sc.kv[(self.i, p)]
        kv.push(k, v, sc.active)
        n = kv.n
        if hasattr(kv, "kid"):                    # qvb8 KV (StaticQKV): the kernel reads the ids + the dense tail
            o = inkling_attention(q.unsqueeze(2), kv.kt, kv.vt, rel.unsqueeze(2), sinks.unsqueeze(-1), n - 1,
                                  D ** -0.5, self.W, lk=n, qkv=kv)
        else:
            o = inkling_attention(q.unsqueeze(2), kv.k, kv.v, rel.unsqueeze(2), sinks.unsqueeze(-1), n - 1,
                                  D ** -0.5, self.W, lk=n)
        g = torch.sigmoid(_lin(self.gate, hq).float())
        return self.reduce(_lin(self.wo, (o.reshape(S, H * D).float() * g).to(self.dt)))

    def _attention_verify(self, hq, hk, hv, pos, sc, p):
        """Ragged rows of a speculative verify: K / V of every row written after its slot's keys but not counted (the
        commit keeps the accepted ones); row r attends its slot's keys up to itself (causal inside the slot's rows)."""
        R = hq.shape[0]
        H, Hk, D, K = self.H, self.Hk, self.D, self.K
        q = self._clip(_lin(self.wq, hq).view(R, H, D), self.q_scale, self.c["qk_thr"], pos)
        rs = self._clip(_lin(self.wr, hq).view(R, H, -1), self.rel_scale, self.c["rel_thr"], pos)
        rel = rs @ self.rel_proj
        sinks = _lin(self.sink, hq)
        k_raw, v_raw = _lin(self.wk, hk), _lin(self.wv, hv)
        cs, sx = sc.conv[(self.i, p)], sc.scratch[(self.i, p)]
        if self.sliding:
            k = short_conv_r(k_raw, cs["kc"], self.k_conv, K, sx["kc"], sc.slot, sc.j, sc.start)
            v = short_conv_r(v_raw, cs["vc"], self.v_conv, K, sx["vc"], sc.slot, sc.j, sc.start)
        else:
            k = dyn_conv_r(k_raw, _lin(self.k_wproj, hk), cs["kc"], self.k_static, Hk, K, sx["kc"], sc.slot, sc.j)
            v = dyn_conv_r(v_raw, _lin(self.v_wproj, hv), cs["vc"], self.v_static, Hk, K, sx["vc"], sc.slot, sc.j)
        k = l2clip(k.view(R, Hk, D), self.c["qk_thr"])
        v = rms(v.view(R, Hk, D), eps=self.c["eps"])
        kv = sc.kv[(self.i, p)]
        if hasattr(kv, "kid"):                    # qvb8 KV (StaticQKV): ids + the dense tail, read in the kernel
            kv.write_rows(k, v, sc.slot, sc.j)
            kt, vt, qkv = kv.kt, kv.vt, kv
        else:
            kv.write_rows(k, v, sc.slot, sc.j, sc.row_active, sc.active, sc.Tmax + 1)   # + the padding row
            kt, vt, qkv = kv.k, kv.v, None
        d0 = kv.n[sc.slot] + sc.j
        o = inkling_attention(q.unsqueeze(2), kt, vt, rel.unsqueeze(2), sinks.unsqueeze(-1), d0, D ** -0.5, self.W,
                              lk=d0 + 1, qkv=qkv, slot=sc.slot)
        g = torch.sigmoid(_lin(self.gate, hq).float())
        return self.reduce(_lin(self.wo, (o.reshape(R, H * D).float() * g).to(self.dt)))

    def _conv_static(self, x, sc, p, key, w):
        if sc.verify:
            return short_conv_r(x, sc.conv[(self.i, p)][key], w, self.K, sc.scratch[(self.i, p)][key],
                                sc.slot, sc.j, sc.start)
        return short_conv1(x, sc.conv[(self.i, p)][key], w, self.K, sc.active)

    def _seg_conv(self, x, segs, p, key, w):
        y = torch.empty_like(x)
        for s, e, slot in segs:
            st = slot[(self.i, p)]
            y[s:e], st[key] = short_conv(x[s:e], st[key], w, self.K, "spec" in slot)
        return y

    def moe(self, x, active=None):
        scores = torch.sigmoid(F.linear(x.float(), self.router))
        idx = torch.topk(scores + self.router_bias, self.c["topk"], dim=-1, sorted=False).indices
        if active is not None:                    # static decode: idle slots route nowhere (the expert kernel skips them)
            idx = torch.where(active[:, None], idx, torch.full_like(idx, -1))
        w = scores.gather(1, idx.clamp(min=0))
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.c["route_scale"]
        return self.reduce(self.experts(x, idx, w))

    def forward(self, h, blocks, pos, segs, p, messages=None):
        """h [T, hidden] bf16, blocks [T, nb, hidden]."""
        eps = self.c["eps"]
        block_res = h
        if self.sliding:
            x = block_residual(self.attn_q, blocks, h)[:, 0] if self.attn_q is not None else h
            if messages is None:
                hq = hk = hv = rms(x, self.in_norm, eps, self.dt)
            else:
                hq = self.msg_gate_norm("q", x, messages[:, 0], self.in_norm)
                hk = self.msg_gate_norm("k", x, messages[:, 1], self.in_norm)
                hv = self.msg_gate_norm("v", x, messages[:, 2], self.in_norm)
        else:
            if self.attn_q is None:
                xq = xk = xv = h
            else:
                r = block_residual(self.attn_q, blocks, h)
                xq, xk, xv = r[:, 0], r[:, 1], r[:, 2]
            if messages is None:
                hq, hk, hv = rms(xq, self.q_norm, eps, self.dt), rms(xk, self.k_norm, eps, self.dt), rms(xv, self.v_norm, eps, self.dt)
            else:
                hq = self.msg_gate_norm("q", xq, messages[:, 0], self.q_norm)
                hk = self.msg_gate_norm("k", xk, messages[:, 1], self.k_norm)
                hv = self.msg_gate_norm("v", xv, messages[:, 2], self.v_norm)
        if isinstance(segs, StaticCtx):           # static decode: fixed slot buffers (engine/static_decode)
            a = self._conv_static(self.attention_static(hq, hk, hv, pos, segs, p), segs, p, "ac", self.attn_conv)
        else:
            a = self._seg_conv(self.attention(hq, hk, hv, pos, segs, p), segs, p, "ac", self.attn_conv)
        hid = a.float()
        if self.boundary:
            blocks = torch.cat([blocks, block_res.unsqueeze(-2)], dim=-2)
            res = hid
        else:
            res = hid = block_res.float() + hid
        hid = block_residual(self.mlp_q, blocks, hid)[:, 0]
        hn = (rms(hid, self.mlp_norm, eps, self.dt) if messages is None
              else self.msg_gate_norm("mlp", hid, messages[:, 3], self.mlp_norm))
        mo = self.moe(hn, segs.row_active if isinstance(segs, StaticCtx) else None)
        m = (self._conv_static(mo, segs, p, "mc", self.mlp_conv) if isinstance(segs, StaticCtx)
             else self._seg_conv(mo, segs, p, "mc", self.mlp_conv))
        return (res + m.float()).to(self.rd)


# ----------------------------------------------------------------------------------------- model
class InklingModel:
    def __init__(self, pack, device="cuda", kv_cache="bf16", kv_tau=0.1, dtype=None, tp=None,
                 experts="resident", expert_cache_gb=8.0):
        """dtype None: default_dtype(device). tp: run/tp_dit.ThreadTP (this card's rank of a group) or None.
        experts "resident" | "host" (stack in host RAM, `expert_cache_gb` of it cached on the card)."""
        dtype = dtype or default_dtype(device)
        self.dspark = None                                 # enable_dspark(): the checkpoint's own drafter
        self.spec_width = 0                                # enable_spec(): speculative verify
        rank, world = (tp.rank, tp.world) if tp is not None else (0, 1)
        cfg = json.load(open(os.path.join(pack, "config.json")))["text_config"]
        lc = cfg["loop_config"]
        self.cfg = c = {
            "hidden": cfg["hidden_size"], "layers": cfg["num_hidden_layers"], "heads": cfg["num_attention_heads"],
            "kv_heads": cfg["num_key_value_heads"], "head_dim": cfg["head_dim"], "eps": cfg["rms_norm_eps"],
            "local_layer_ids": set(cfg["local_layer_ids"]), "window": cfg["sliding_window_size"],
            "conv": cfg["sconv_kernel_size"], "n_layers_per_attn_res_block": cfg["n_layers_per_attn_res_block"],
            "loop_start": lc["core_start"], "loop_end": lc["core_end"], "passes": lc["num_passes"],
            "qk_thr": cfg["qk_l2_clip_threshold"], "rel_thr": cfg["rel_l2_clip_threshold"],
            "alpha": cfg["log_scaling_alpha"], "n_floor": cfg["log_scaling_n_floor"],
            "topk": cfg["num_experts_per_tok"], "route_scale": cfg["route_scale"],
            "vocab": cfg["unpadded_vocab_size"], "logit_cap": cfg["final_logit_softcapping"],
            "gate_cap": cfg["mlp_gate_softcap"], "up_cap": cfg["mlp_up_softcap"], "n_experts": cfg["n_routed_experts"],
            "max_pos": cfg.get("model_max_length", 1 << 20),
            "kv_tau": float(kv_tau) if kv_cache == "qvb8" else None,
            "dtype": dtype,
            # fp16: the residual stream / block states in fp32 (they grow); bf16 has the range to hold them
            "res_dtype": torch.float32 if dtype == torch.float16 else dtype,
        }
        from qv_band.formats.qvb8.tp import split_even
        c["head_range"] = split_even(c["heads"], world, rank)
        c["kv_range"] = split_even(c["kv_heads"], world, rank)
        if (c["head_range"][1] - c["head_range"][0]) * c["kv_heads"] != \
                (c["kv_range"][1] - c["kv_range"][0]) * c["heads"]:
            raise ValueError(f"inkling TP: {world} cards do not split {c['heads']} q / {c['kv_heads']} KV heads "
                             "into whole GQA groups")
        from qv_band.run.expert_parallel import expert_range
        ep_w = [float(x) for x in os.environ["QV_EP_WEIGHTS"].split(",")] if os.environ.get("QV_EP_WEIGHTS") else None
        e0, e1 = expert_range(c["n_experts"], rank, world, ep_w)        # uneven cards: QV_EP_WEIGHTS=3,1
        self.device, self.dt, self.rd, self.tp = device, dtype, c["res_dtype"], tp
        W = Weights(pack, device, dtype)
        from qv_band.formats.qvbq import QVBQExperts, QVBQHostExperts
        act = SoftcappedSwiGLU(c["gate_cap"], c["up_cap"])
        stacks = []
        for i in range(c["layers"]):
            files = {}
            for proj in ("gate_up_proj", "down_proj"):
                blk = f"model.language_model.layers.{i}.mlp.experts.{proj}#expert"
                files[proj] = {int(k.rsplit("#expert", 1)[1]): os.path.join(pack, v["pack"])
                               for k, v in W.mani.items() if k.startswith(blk)}
            stacks.append(files)
        if experts == "host":
            # the same share of every layer's experts cached on the card (this card's range: [e0, e1))
            paths = {p for f in stacks for d in f.values() for p in d.values()}
            total = sum(os.path.getsize(p) for p in paths) * (e1 - e0) / c["n_experts"]
            frac = min(1.0, float(expert_cache_gb) * 2**30 / max(total, 1))
            slots = max(2 * c["topk"], int(frac * (e1 - e0)))
        elif experts != "resident":
            raise ValueError(f"experts must be resident or host (got {experts!r})")
        self.layers = []
        for i in range(c["layers"]):
            f = stacks[i]
            if experts == "host":
                ex = QVBQHostExperts.from_packfiles(f["gate_up_proj"], f["down_proj"], num_experts=c["n_experts"],
                                                    slots=slots, act_fn=act, device=device, experts=(e0, e1),
                                                    dtype=dtype)
            else:
                ex = QVBQExperts.from_packfiles(f["gate_up_proj"], f["down_proj"], num_experts=c["n_experts"],
                                                act_fn=act, device=device,
                                                experts=None if world == 1 else (e0, e1), dtype=dtype)
            self.layers.append(Layer(W, c, i, ex, tp))
        self.expert_mode = experts if experts != "host" else f"host ({slots} of {e1 - e0} experts per layer cached)"
        self.embed = W.dense("model.llm.embed.weight")
        self.embed_norm = W.dense("model.llm.embed_norm.weight")
        self.norm = W.dense("model.llm.norm.weight")
        self.out_q = W.dense("model.llm.attn_res.output_query", torch.float32).reshape(1, -1)
        qs = W.dense("model.llm.middle_cycle_cross_pass_queries", torch.float32)      # q, mlp per loop layer
        kvs = W.dense("model.llm.middle_cycle_cross_pass_kv_queries", torch.float32)  # k, v per loop layer
        n = c["loop_end"] - c["loop_start"]
        self.msg_q = torch.stack([qs[0::2], kvs[0::2], kvs[1::2], qs[1::2]], 1).reshape(n, 4, -1)   # q, k, v, mlp
        self.lm_head = W.linear("model.llm.unembed.weight")
        torch.cuda.empty_cache()

    # --- slot state
    def new_state(self):
        c = self.cfg
        C = (c["kv_range"][1] - c["kv_range"][0]) * c["head_dim"]        # this card's KV channels
        z = lambda n: torch.zeros(c["conv"] - 1, n, dtype=torch.float32, device=self.device)
        def entry():
            return {"k": None, "v": None, "k0": 0, "kc": z(C), "vc": z(C), "ac": z(c["hidden"]), "mc": z(c["hidden"])}
        st = {(i, 0): entry() for i in range(c["layers"])}
        for p in range(1, c["passes"]):
            for i in range(c["loop_start"], c["loop_end"]):
                st[(i, p)] = entry()
        st["pos"] = 0
        if self.dspark is not None:
            st["draft"] = self.dspark.new_state()
        return st

    # --- speculative decoding with the checkpoint's own DSpark drafter (models/inkling/dspark.py) ------------------
    def enable_spec(self, width):
        """Speculative verify on this card: up to `width` tokens per request per step (eager: verify rows + trim;
        static slots: the ragged verify buffers). Cards that only verify (Tensor Parallel followers) need no drafter."""
        self.spec_width = int(width)

    def enable_dspark(self, pack, max_k=7):
        """The checkpoint's own DSpark drafter (dspark.py): speculative decoding by itself (engine/spec.py protocol)."""
        from .dspark import InklingDSpark
        self.dspark = InklingDSpark(self, pack)
        self.enable_spec(self.dspark.block)
        self.spec_max_draft = max(0, min(int(max_k), self.dspark.block - 1))
        print(f"[inkling] DSpark drafter on: up to {self.spec_max_draft} drafts per step", flush=True)

    # --- engine/spec.py protocol (eager) ------------------------------------------------------------------------------
    def spec_propose(self, rows):
        """[(state, last token, (temperature, top_k, top_p))] -> [(drafts, each one's chance, distributions or None)]
        from each state's drafter context (sampled rows: the drafter's filtered distributions)."""
        return [self.dspark.propose(st["draft"], last, sp) for st, last, sp in rows]

    def spec_verify(self, rows):
        """[(state, tokens)] -> logits of every token; each state holds them all (and their taps) until spec_keep."""
        for st, toks in rows:
            st["spec"] = len(toks)
        return self.forward(rows)

    def spec_keep(self, rows):
        """[(state, keep)]: the first `keep` tokens of each state's last verify stay (the drafter gets their taps)."""
        for st, keep in rows:
            taps = st.pop("spec_taps", None)
            self.trim(st, st["spec"] - keep)
            if self.dspark is not None:
                self.dspark.extend(st["draft"], taps[:keep])

    def trim(self, st, drop):
        """After verify(): drop the last `drop` of the T tokens just run (conv histories, K / V, position)."""
        T, K, W = st.pop("spec"), self.cfg["conv"], self.cfg["window"]
        keep = T - drop
        for key, e in st.items():
            if not isinstance(key, tuple):
                continue
            for n in ("kc", "vc", "ac", "mc"):
                e[n] = e[n][keep:keep + K - 1].clone()            # (a view would keep the whole history)
            if e["k"] is None or not drop and isinstance(e["k"], QVB8KV):
                continue
            if isinstance(e["k"], QVB8KV):
                e["k"].drop(drop)
                e["v"].drop(drop)
                continue
            kk, vv = e["k"][:, :, :e["k"].shape[2] - drop], e["v"][:, :, :e["v"].shape[2] - drop]
            if self.layers[key[0]].sliding and kk.shape[2] > W - 1:
                cut = kk.shape[2] - (W - 1)
                kk, vv, e["k0"] = kk[:, :, cut:], vv[:, :, cut:], e["k0"] + cut
            e["k"], e["v"] = kk.clone(), vv.clone()
        st["pos"] -= drop

    @staticmethod
    def clone_state(st):
        out = {}
        for k, e in st.items():
            if k == "draft":
                out[k] = {"k": list(e["k"]), "v": list(e["v"]), "n": e["n"], "f_rnn": e["f_rnn"]}
            else:
                out[k] = e if k == "pos" else {n: (v.clone() if torch.is_tensor(v) or isinstance(v, QVB8KV) else v)
                                                for n, v in e.items()}
        return out

    @staticmethod
    def export_state(st):
        """-> a plain tree for the engine's shared prefix pool (QVB8KV -> {"qvb8kv": tau, "blocks", "tail"})."""
        def ex(v):
            return {"qvb8kv": v.tau, "blocks": [list(b) for b in v.blocks], "tail": v.tail} if isinstance(v, QVB8KV) else v
        return {k: (e if not isinstance(k, tuple) else {n: ex(v) for n, v in e.items()}) for k, e in st.items()}

    @staticmethod
    def import_state(tree, device):
        """export_state's tree (tensors may be views of the pool) -> a live state of this process's own tensors."""
        from qv_band.kv.codec import BandedKV

        def t(x):
            return x.to(device, non_blocking=False).clone() if torch.is_tensor(x) else x

        def bk(b):
            return BandedKV(packed=t(b.packed), min_s=t(b.min_s), tau_s=t(b.tau_s), bits=b.bits,
                            orig_shape=b.orig_shape, orig_dtype=b.orig_dtype, tau=b.tau)

        def im(v):
            if isinstance(v, dict) and "qvb8kv" in v:
                return QVB8KV(v["qvb8kv"], [[bk(h) for h in blk] for blk in v["blocks"]], t(v["tail"]))
            return t(v)
        def dr(e):
            return {"k": [t(x) for x in e["k"]], "v": [t(x) for x in e["v"]], "n": e["n"], "f_rnn": t(e["f_rnn"])}
        return {k: (dr(e) if k == "draft" else e if k == "pos" else {n: im(v) for n, v in e.items()})
                for k, e in tree.items()}

    @staticmethod
    def state_nbytes(st):
        n = 0
        for k, e in st.items():
            if k == "draft":
                n += sum(x.numel() * x.element_size() for x in e["k"] + e["v"] if x is not None)
            elif k != "pos":
                n += sum(v.numel() * v.element_size() for v in e.values() if torch.is_tensor(v))
                n += sum(v.nbytes() for v in e.values() if isinstance(v, QVB8KV))
        return n

    # --- forward
    @torch.no_grad()
    def forward(self, rows, all_logits: bool = False):
        """rows: [(state, token ids list)] or [(state, ids, (positions, embeds))] -> logits [len(rows), vocab] fp32
        of each row's LAST token. Every state is advanced by its row's length. (positions [k] within the row,
        embeds [k, hidden]): those tokens' input embeddings are replaced -- image features, which come after the
        embedding RMSNorm (the reference's masked_scatter into the normed embeddings).
        Rows of a speculative verify (spec_verify) return every position's logits. With the DSpark drafter on
        (enable_dspark) every forward extends each row's drafter context with its taps (a verify: at spec_keep)."""
        c = self.cfg
        dev = self.device
        ids, pos, segs, s, swaps = [], [], [], 0, []
        for row in rows:
            st, toks = row[0], row[1]
            if len(row) > 2 and row[2] is not None:
                p_, e_ = row[2]
                swaps.append((torch.as_tensor(p_, dtype=torch.long) + s, e_))
            n = len(toks)
            ids.extend(toks)
            pos.append(torch.arange(st["pos"], st["pos"] + n))
            segs.append((s, s + n, st))
            s += n
        ids = torch.tensor(ids, device=dev)
        pos = torch.cat(pos).to(dev)
        h = rms(F.embedding(ids, self.embed).to(self.rd), self.embed_norm, c["eps"])
        for p_, e_ in swaps:
            h[p_.to(dev)] = e_.to(dev, h.dtype)
        if swaps and os.environ.get("QV_DEBUG_VISION"):
            print(f"[dbg-vision] dev={dev} rows={len(rows)} n={h.shape[0]} swaps={[(len(p_), tuple(e_.shape), str(e_.device), str(e_.dtype), float(e_.float().abs().max()), bool(torch.isnan(e_.float()).any())) for p_, e_ in swaps]} h.absmax={float(h.abs().max()):.2f} h.nan={bool(torch.isnan(h).any())}", flush=True)
        dsp = self.dspark
        spec = any("spec" in st for _, _, st in segs)
        if spec and not all("spec" in st for _, _, st in segs):
            raise ValueError("Inkling forward: a speculative verify (spec_verify) cannot share a batch with other rows")
        cap = None if dsp is None else []
        h, blocks = self._stack(h, pos, segs, cap, dsp.taps if dsp is not None else ())
        # all_logits / a verify: every position's logits [T, vocab] instead of each row's last
        last = (torch.arange(segs[-1][1], device=dev) if all_logits or spec
                else torch.tensor([e - 1 for _, e, _ in segs], device=dev))
        if dsp is not None:                       # the final tap: the normed output of every position
            hn = rms(block_residual(self.out_q, blocks, h)[:, 0], self.norm, c["eps"], self.dt)
            taps = torch.stack(cap + [hn], 1)
            h = hn[last]
        else:
            h = rms(block_residual(self.out_q, blocks[last], h[last])[:, 0], self.norm, c["eps"], self.dt)
        for (s0, e0, st) in segs:
            st["pos"] += e0 - s0
            if dsp is not None:
                if spec:
                    st["spec_taps"] = taps[s0:e0]          # spec_keep hands the kept ones to the drafter
                else:
                    dsp.extend(st["draft"], taps[s0:e0])
        return softcap(_lin(self.lm_head, h)[:, :c["vocab"]], c["logit_cap"]).float()

    def _stack(self, h, pos, segs, cap=None, taps=()):
        """Every layer (the loop range `passes` times, with cross-pass messages) -> (h, blocks). segs: the rows' eager
        states, or a StaticCtx (static decode). cap: a list that gets the input of every layer visit in `taps`."""
        c = self.cfg
        blocks = h.new_zeros(h.shape[0], 0, c["hidden"])
        a, b = c["loop_start"], c["loop_end"]
        visit = [0]

        def run(layer, h, blocks, p, msgs=None):
            if cap is not None and visit[0] in taps:
                cap.append(h.to(self.dt))
            visit[0] += 1
            out = layer.forward(h, blocks, pos, segs, p, msgs)
            if layer.boundary:
                blocks = torch.cat([blocks, h.unsqueeze(-2)], dim=-2)
            return out, blocks

        for layer in self.layers[:a]:
            h, blocks = run(layer, h, blocks, 0)
        pre_h, pre_b = h, blocks
        n_pre = pre_b.shape[-2]
        for layer in self.layers[a:b]:
            h, blocks = run(layer, h, blocks, 0)
        for p in range(1, c["passes"]):
            prev_b, prev_h = blocks[:, n_pre + 1:], h
            h, blocks = pre_h, pre_b
            for j, layer in enumerate(self.layers[a:b]):
                h, blocks = run(layer, h, blocks, p, block_residual(self.msg_q[j], prev_b, prev_h))
        for layer in self.layers[b:]:
            h, blocks = run(layer, h, blocks, 0)
        return h, blocks

    # --- static decode (engine/static_decode.StaticStepper drives these) ----------------------------------
    def static_init(self, slots, max_len):
        from qv_band.engine.static_decode import StaticKV, StaticQKV
        c, dev = self.cfg, self.device
        Hk, D, K = c["kv_range"][1] - c["kv_range"][0], c["head_dim"], c["conv"]
        C = Hk * D
        sc = self.sc = StaticCtx()
        z = lambda n: torch.zeros(slots, K - 1, n, dtype=torch.float32, device=dev)
        TV = self.spec_width                                              # verify width (0: no speculative)
        for key in (k for k in self.new_state() if isinstance(k, tuple)):
            L = self.layers[key[0]]
            if not L.sliding and c["kv_tau"] is not None:     # --kv-cache qvb8: the growing layers as 8-bit ids
                sc.kv[key] = StaticQKV(slots, Hk, max_len, D, self.dt, dev, extra=TV + 1 if TV else 0)
            else:                                            # sliding window (bounded) / --kv-cache bf16: dense
                sc.kv[key] = StaticKV(slots, Hk, c["window"] if L.sliding else max_len, D, self.dt, dev, window=L.sliding)
            sc.conv[key] = {"kc": z(C), "vc": z(C), "ac": z(c["hidden"]), "mc": z(c["hidden"])}
            if TV:
                zt = lambda n: torch.zeros(slots, TV + 1, n, dtype=torch.float32, device=dev)
                sc.scratch[key] = {"kc": zt(C), "vc": zt(C), "ac": zt(c["hidden"]), "mc": zt(c["hidden"])}
        if TV:                                    # speculative verify: ragged rows, <= TV per slot
            Rm = slots * TV
            self.st_vtok = torch.zeros(Rm, dtype=torch.long, device=dev)
            self.st_vslot = torch.zeros(Rm, dtype=torch.long, device=dev)
            self.st_vj = torch.zeros(Rm, dtype=torch.long, device=dev)
            self.st_vrow = torch.zeros(Rm, dtype=torch.bool, device=dev)
            self.st_vstart = torch.zeros(slots, dtype=torch.long, device=dev)
            self.st_acc = torch.zeros(slots, dtype=torch.long, device=dev)
            self.st_vlogits = torch.zeros(Rm, c["vocab"], dtype=torch.float32, device=dev)
            sc.Tmax = TV
            if self.dspark is not None:
                self.st_taps = torch.zeros(Rm, len(self.dspark.taps), c["hidden"], dtype=self.dt, device=dev)
                self.dspark.static_init(slots, max_len)
        self.st_tok = torch.zeros(slots, dtype=torch.long, device=dev)
        self.st_active = torch.zeros(slots, dtype=torch.bool, device=dev)
        self.st_pos = torch.zeros(slots, dtype=torch.long, device=dev)
        self.st_logits = torch.zeros(slots, c["vocab"], dtype=torch.float32, device=dev)
        self.static_capacity = int(max_len)
        nb = sum(kv.nbytes() for kv in sc.kv.values()) + sum(t.numel() * 4 for e in sc.conv.values() for t in e.values())
        print(f"[inkling] static decode: {slots} slots x {max_len} tokens, {nb / 2**30:.2f} GiB of slot buffers", flush=True)

    def static_bytes(self, slots, max_len):
        """What static_init(slots, max_len) allocates (engine/static_decode.StaticStepper sizes the slots with it)."""
        from qv_band.engine.static_decode import StaticKV, StaticQKV
        c = self.cfg
        Hk, D, K, Hd = c["kv_range"][1] - c["kv_range"][0], c["head_dim"], c["conv"], c["hidden"]
        TV = self.spec_width
        n = 0
        for key in (k for k in self.new_state() if isinstance(k, tuple)):
            L = self.layers[key[0]]
            if not L.sliding and c["kv_tau"] is not None:
                n += StaticQKV.bytes_for(slots, Hk, max_len, D, self.dt, extra=TV + 1 if TV else 0)
            else:
                n += StaticKV.bytes_for(slots, Hk, c["window"] if L.sliding else max_len, D, self.dt, window=L.sliding)
            n += 4 * slots * (K - 1 + (TV + 1 if TV else 0)) * 2 * (Hk * D + Hd)      # conv states (+ verify scratch)
        n += 4 * slots * c["vocab"]                                                    # st_logits
        if TV:
            n += 4 * slots * TV * c["vocab"]                                           # st_vlogits
        if self.dspark is not None:
            n += self.dspark.static_bytes(slots, max_len, slots * TV)
        return n

    def static_free(self):
        for n in ("sc", "st_tok", "st_active", "st_pos", "st_logits", "st_vtok", "st_vslot", "st_vj", "st_vrow",
                  "st_vstart", "st_acc", "st_vlogits", "st_taps"):
            self.__dict__.pop(n, None)
        if self.dspark is not None and hasattr(self.dspark, "skv"):
            self.dspark.static_free()

    def static_load(self, slot, st):
        sc = self.sc
        for key, kv in sc.kv.items():
            e = st[key]
            for n in ("kc", "vc", "ac", "mc"):
                sc.conv[key][n][slot].copy_(e[n])
            if e["k"] is None:
                kv.clear(slot)
            else:
                k, v = e["k"], e["v"]
                if isinstance(k, QVB8KV):
                    k, v = k.dense(), v.dense()
                kv.load(slot, k, v)
        self.st_pos[slot] = st["pos"]
        if self.dspark is not None:
            self.dspark.static_load(slot, st["draft"])

    def static_step(self):
        c, sc = self.cfg, self.sc
        sc.active = sc.row_active = self.st_active
        sc.verify = False
        pos = self.st_pos
        h = rms(F.embedding(self.st_tok, self.embed).to(self.rd), self.embed_norm, c["eps"])
        h, blocks = self._stack(h, pos, sc)
        h = block_residual(self.out_q, blocks, h)[:, 0]
        h = rms(h, self.norm, c["eps"], self.dt)
        self.st_logits.copy_(softcap(_lin(self.lm_head, h)[:, :c["vocab"]], c["logit_cap"]).float())
        self.st_pos.copy_(torch.where(self.st_active, pos + 1, pos))

    # --- speculative verify on the slot buffers (engine/static_decode.StaticStepper.spec captures these) --------------
    def static_verify(self, R):
        """Ragged rows [:R] (st_vtok, st_vslot, st_vj, st_vrow; st_vstart per slot) after the active slots' keys ->
        st_vlogits[:R], st_taps[:R]; the states are NOT advanced (static_commit keeps st_acc[s] tokens per slot)."""
        c, sc = self.cfg, self.sc
        sc.verify, sc.active = True, self.st_active
        sc.slot, sc.j, sc.row_active, sc.start = self.st_vslot[:R], self.st_vj[:R], self.st_vrow[:R], self.st_vstart
        pos = self.st_pos[sc.slot] + sc.j
        h = rms(F.embedding(self.st_vtok[:R], self.embed).to(self.rd), self.embed_norm, c["eps"])
        cap = None if self.dspark is None else []
        h, blocks = self._stack(h, pos, sc, cap, self.dspark.taps if cap is not None else ())
        hn = rms(block_residual(self.out_q, blocks, h)[:, 0], self.norm, c["eps"], self.dt)
        if cap is not None:
            self.st_taps[:R].copy_(torch.stack(cap + [hn], 1))
        self.st_vlogits[:R].copy_(softcap(_lin(self.lm_head, hn)[:, :c["vocab"]], c["logit_cap"]).float())
        sc.verify = False

    def static_commit(self):
        """Keep the first st_acc[s] verified tokens of every slot (0: unchanged): K / V counts, conv histories, the
        drafter's context (their taps), position."""
        from qv_band.engine.static_decode import conv_commit
        keep = self.st_acc
        sl, j = self.st_vslot, self.st_vj
        if self.dspark is not None:
            self.dspark.static_extend(self.st_taps, sl, j, self.st_vrow & (j < keep[sl]),
                                      (self.st_vstart + keep - 1).clamp(min=0), keep, self.st_active, self.st_pos)
        for key, kv in self.sc.kv.items():
            kv.commit(keep)
            for n, st in self.sc.conv[key].items():
                conv_commit(st, self.sc.scratch[key][n], keep)
        self.st_pos.add_(keep)

    def static_propose(self, n, sp=None):
        """The drafter on the slot buffers: slots st_pslot[:n] after st_panchor[:n] -> st_draft[:n], st_dprob[:n]
        (sp: per row sampling settings; sampled rows also st_ddist[:n])."""
        self.dspark.static_propose(n, self.st_pos, sp)
