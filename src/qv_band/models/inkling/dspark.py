"""Inkling-2-Mini's own DSpark drafter (the checkpoint's bundled "mtp_config" head), for models/inkling/model.py.

Port of SGLang's reference (python/sglang/srt/models/inkling2_mtp.py + models/dflash.py + speculative/dspark_components/
dspark_draft.py + kernels/ops/speculative/inkling2_mtp.py). Weights from the pack (qvb8 linears, the rest dense bf16):
  3 DFlash layers (Qwen3-style: pre-norm, q/k head norms, RoPE neox theta 1e6, GQA 24/8 x 128, SwiGLU 9216),
  local_layer_ids [0, 2]: 1024-token window anchored at the block start; the block attends NON-causally.
  context K / V of layer i from the target's taps: rms(sum_t softmax(fuse_weight[i])_t * tap_t) * dflare_norm_i ->
  k_proj_ctx -> k_norm_ctx -> RoPE (position of that token), v_proj_ctx. Taps = the target's layer-visit inputs at
  target_layer_ids[:-1] and its final normed hidden (the last id = number of visits).
  block of 8 at positions L..L+7: [embed(anchor), mask_embed x 7]; hidden = norm(out)[1:] -> 7 drafts.
  Markov head (rnn_hidden, rank 256): state 0, f_rnn = rms(mix(taps of the last committed token), fuse row -1) *
  markov dflare_norm; per draft j: [state, w1[prev], hidden_j, f_rnn] -> rnn_proj -> gate = sigmoid, state' =
  gate * state + (1 - gate) * tanh(cand), latent = tanh(out); logits_j = softcap(lm_head(hidden_j) +
  lm_head(hidden_out(latent)), 30) over the unpadded vocab; greedy draft = argmax (prev = that token).
The lm_head . hidden_out product is folded once ([vocab, 256] + [vocab]), as the reference does.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from qv_band.engine.spec import sampling_probs
from qv_band.layers.fold_logits import fold_logits
from qv_band.layers.fused_rownorm import rms

from .model import Weights, _lin


def _rope(x, pos, theta, D):
    """neox RoPE: x [T, h, D], pos [T]."""
    inv = theta ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    f = pos.float()[:, None] * inv[None]
    cos, sin = f.cos()[:, None], f.sin()[:, None]
    x1, x2 = x[..., : D // 2].float(), x[..., D // 2:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)


class InklingDSpark:
    def __init__(self, target, pack):
        """target: models/inkling/model.InklingModel (shares its embedding and lm_head)."""
        cfg = json.load(open(os.path.join(pack, "config.json")))
        mc = cfg["text_config"].get("mtp_config") or cfg["mtp_config"]
        self.t, self.c = target, mc
        dev, dt = target.device, target.dt
        self.dt = dt
        W = Weights(pack, dev, dt)
        self.H, self.nh, self.nk, self.D = mc["hidden_size"], mc["num_attention_heads"], mc["num_key_value_heads"], mc["head_dim"]
        self.eps, self.theta, self.W = mc["rms_norm_eps"], float(mc["rope_theta"]), mc["sliding_window_size"]
        self.block = mc["block_size"]
        self.taps = tuple(mc["target_layer_ids"])
        self.local = set(mc["local_layer_ids"])
        self.L = mc["num_hidden_layers"]
        self.alpha = torch.softmax(W.dense("fuse_weight", torch.float32), -1)          # [L + 1, taps]
        self.mask_embed = W.dense("mask_embed")
        self.norm = W.dense("norm.weight")
        self.layers = []
        for i in range(self.L):
            p = f"layers.{i}."
            a = p + "self_attn."
            self.layers.append({n: W.linear(a + n + ".weight") for n in ("q_proj", "k_proj", "v_proj", "o_proj",
                                                                       "k_proj_ctx", "v_proj_ctx")} |
                               {n: W.linear(p + "mlp." + n + ".weight") for n in ("gate_proj", "up_proj", "down_proj")} |
                               {n: W.dense(p + n + ".weight") for n in ("input_layernorm", "post_attention_layernorm",
                                                                       "dflare_norm")} |
                               {n: W.dense(a + n + ".weight") for n in ("q_norm", "k_norm", "k_norm_ctx")})
        mh = "markov_head."
        self.w1 = W.dense(mh + "markov_w1.weight")                                       # [vocab, R]
        self.rnn_proj = W.linear(mh + "rnn_proj.weight")
        self.rnn_bias = W.dense(mh + "rnn_proj.bias", torch.float32)
        self.m_norm = W.dense(mh + "dflare_norm.weight")
        self.R = self.w1.shape[1]
        ho_w = W.dense(mh + "hidden_out.weight", torch.float32)                           # [H, R]
        ho_b = W.dense(mh + "hidden_out.bias", torch.float32)
        V = target.cfg["vocab"]
        fw, fb = torch.empty(V, self.R, device=dev, dtype=dt), torch.empty(V, device=dev, dtype=torch.float32)
        head = W.dense("model.llm.unembed.weight", torch.bfloat16)                       # the target's lm_head, decoded once
        for r0 in range(0, V, 16384):                       # lm_head . hidden_out, folded
            r1 = min(r0 + 16384, V)
            hr = head[r0:r1].float()
            fw[r0:r1] = (hr @ ho_w).to(dt)
            fb[r0:r1] = hr @ ho_b
        del head
        self.fold_w, self.fold_b = fw, fb
        self.cap = target.cfg["logit_cap"]

    # --- per request ------------------------------------------------------------------------------------------
    def new_state(self):
        return {"k": [None] * self.L, "v": [None] * self.L, "n": 0, "f_rnn": None}

    def clone_state(self, st):
        return {"k": list(st["k"]), "v": list(st["v"]), "n": st["n"], "f_rnn": st["f_rnn"]}

    def _mix(self, taps, i, w):
        """taps [T, n, H] -> rms(sum alpha[i] * tap) * w, [T, H] in the matmul dtype."""
        return rms((taps.float() * self.alpha[i][None, :, None]).sum(1), w, self.eps, self.dt)

    @torch.no_grad()
    def extend(self, st, taps):
        """The target's taps [T, n, H] of the next T committed tokens: context K / V of every layer + f_rnn."""
        T = taps.shape[0]
        pos = torch.arange(st["n"], st["n"] + T, device=taps.device)
        for i, Lw in enumerate(self.layers):
            ctx = self._mix(taps, i, Lw["dflare_norm"])
            k = rms(_lin(Lw["k_proj_ctx"], ctx).view(T, self.nk, self.D), Lw["k_norm_ctx"], self.eps)
            k = _rope(k, pos, self.theta, self.D)
            v = _lin(Lw["v_proj_ctx"], ctx).view(T, self.nk, self.D)
            k = k if st["k"][i] is None else torch.cat([st["k"][i], k])
            v = v if st["v"][i] is None else torch.cat([st["v"][i], v])
            if i in self.local and k.shape[0] > self.W:
                k, v = k[-self.W:], v[-self.W:]
            st["k"][i], st["v"][i] = k, v
        st["n"] += T
        st["f_rnn"] = self._mix(taps[-1:], -1, self.m_norm)

    @torch.no_grad()
    def propose(self, st, anchor: int, sp=None):
        """7 drafts after `anchor` (the token at position st['n']) -> (tokens, each one's chance, distributions).
        sp (temperature, top_k, top_p) with temperature > 0: drafts drawn from the drafter's filtered distribution,
        returned for speculative sampling; else greedy (distributions None)."""
        B, dev = self.block, self.mask_embed.device
        L0 = st["n"]
        pos = torch.arange(L0, L0 + B, device=dev)
        x = self.mask_embed.expand(B, -1).clone()
        x[0] = self.t.embed[anchor].to(x.dtype)
        x = x.float()
        for i, Lw in enumerate(self.layers):
            hn = rms(x, Lw["input_layernorm"], self.eps, self.dt)
            q = _rope(rms(_lin(Lw["q_proj"], hn).view(B, self.nh, self.D), Lw["q_norm"], self.eps), pos, self.theta, self.D)
            k = _rope(rms(_lin(Lw["k_proj"], hn).view(B, self.nk, self.D), Lw["k_norm"], self.eps), pos, self.theta, self.D)
            v = _lin(Lw["v_proj"], hn).view(B, self.nk, self.D)
            kc, vc = st["k"][i], st["v"][i]
            if i in self.local and kc is not None and kc.shape[0] > self.W - 1:   # window anchored at the block start
                kc, vc = kc[-(self.W - 1):], vc[-(self.W - 1):]
            kk = k if kc is None else torch.cat([kc, k])
            vv = v if vc is None else torch.cat([vc, v])
            o = F.scaled_dot_product_attention(q.transpose(0, 1), kk.transpose(0, 1), vv.transpose(0, 1),
                                               enable_gqa=True)                  # non-causal block
            x = x + _lin(Lw["o_proj"], o.transpose(0, 1).reshape(B, -1)).float()
            h2 = rms(x, Lw["post_attention_layernorm"], self.eps, self.dt)
            x = x + _lin(Lw["down_proj"], F.silu(_lin(Lw["gate_proj"], h2)) * _lin(Lw["up_proj"], h2)).float()
        hid = rms(x[1:], self.norm, self.eps, self.dt)                              # [7, H]
        V = self.t.cfg["vocab"]
        base = _lin(self.t.lm_head, hid)[:, :V].float()
        state = torch.zeros(1, self.R, device=dev, dtype=self.dt)
        f = st["f_rnn"]
        prev = anchor
        out, dists = [], []
        sampled = sp is not None and sp[0] > 0
        for j in range(B - 1):
            inp = torch.cat([state, self.w1[prev][None].to(self.dt), hid[j:j + 1], f], -1)
            pr = (_lin(self.rnn_proj, inp).float() + self.rnn_bias)
            g, cand, o = pr.chunk(3, -1)
            g = torch.sigmoid(g)
            state = (g * state.float() + (1 - g) * torch.tanh(cand)).to(self.dt)
            lat = torch.tanh(o)
            lg = fold_logits(lat.to(self.dt), self.fold_w, self.fold_b, base[j:j + 1], self.cap)[0]
            if sampled:
                q = sampling_probs(lg, *sp)
                prev = int(torch.multinomial(q, 1))
                out.append((prev, float(q[prev])))
                dists.append(q)
            else:
                pj, tj = torch.softmax(lg, -1).max(-1)
                prev = int(tj)
                out.append((prev, float(pj)))
        return [t for t, _ in out], [p for _, p in out], (dists if sampled else None)

    # --- static slots (engine/static_decode.StaticStepper.spec): the context in slot buffers, graph-safe steps -------
    def static_init(self, slots, max_len):
        """Per slot: every layer's context K / V (a window buffer on the local layers), f_rnn; the proposal's I/O."""
        from qv_band.engine.static_decode import StaticKV
        dev, B = self.mask_embed.device, self.block
        self.skv = [StaticKV(slots, self.nk, self.W if i in self.local else max_len, self.D, self.dt, dev,
                             window=i in self.local) for i in range(self.L)]
        self.sf = torch.zeros(slots, self.H, dtype=self.dt, device=dev)
        self.st_pslot = torch.zeros(slots, dtype=torch.long, device=dev)      # the proposing slots, first n
        self.st_panchor = torch.zeros(slots, dtype=torch.long, device=dev)    # their last tokens
        self.st_draft = torch.zeros(slots, B - 1, dtype=torch.long, device=dev)
        self.st_dprob = torch.zeros(slots, B - 1, dtype=torch.float32, device=dev)
        self.st_ddist = torch.zeros(slots, B - 1, self.t.cfg["vocab"], dtype=torch.float32, device=dev)   # sampled rows

    def static_bytes(self, slots, max_len, rows):
        """What static_init(slots, max_len) allocates + the target's tap rows for it (rows verify rows)."""
        from qv_band.engine.static_decode import StaticKV
        es = torch.empty(0, dtype=self.dt).element_size()
        n = sum(StaticKV.bytes_for(slots, self.nk, self.W if i in self.local else max_len, self.D, self.dt,
                                   window=i in self.local) for i in range(self.L))
        return (n + slots * self.H * es + rows * len(self.taps) * self.H * es + 16 * slots * self.block
                + 4 * slots * (self.block - 1) * self.t.cfg["vocab"])

    def static_free(self):
        for n in ("skv", "sf", "st_pslot", "st_panchor", "st_draft", "st_dprob", "st_ddist"):
            self.__dict__.pop(n, None)

    def static_load(self, slot, ds):
        """The eager drafter state `ds` (new_state / extend) into slot `slot`."""
        for i, kv in enumerate(self.skv):
            if ds["k"][i] is None:
                kv.clear(slot)
            else:
                kv.load(slot, ds["k"][i].transpose(0, 1), ds["v"][i].transpose(0, 1))
        if ds["f_rnn"] is not None:
            self.sf[slot].copy_(ds["f_rnn"][0])

    def static_propose(self, n, pos, sp=None):
        """Slots st_pslot[:n] after tokens st_panchor[:n] (at positions pos[slot]) -> st_draft[:n], st_dprob[:n]
        (7 drafts + each one's chance). Greedy: fixed shapes, CUDA-graph safe. sp (per row (temperature, top_k,
        top_p) or None): the sampled rows draw from the drafter's filtered distribution, kept in st_ddist[:n] for
        speculative sampling (eager: no graph)."""
        B, H, dev = self.block, self.H, self.mask_embed.device
        R = n * B
        slots, anchors = self.st_pslot[:n], self.st_panchor[:n]
        slot_r = slots.repeat_interleave(B)
        j_r = torch.arange(B, device=dev).repeat(n)
        pos_r = pos[slot_r] + j_r
        act = torch.zeros(pos.shape[0], dtype=torch.bool, device=dev).scatter_(0, slots, True)
        allr = torch.ones(R, dtype=torch.bool, device=dev)
        x = self.mask_embed.expand(R, H).clone().view(n, B, H)
        x[:, 0] = self.t.embed[anchors].to(x.dtype)
        x = x.view(R, H).float()
        rel = torch.zeros(R, self.nh, 1, 1, dtype=self.dt, device=dev)
        sinks = torch.full((R, self.nh, 1), float("-inf"), dtype=self.dt, device=dev)
        from .attention import inkling_attention
        for i, Lw in enumerate(self.layers):
            hn = rms(x, Lw["input_layernorm"], self.eps, self.dt)
            q = _rope(rms(_lin(Lw["q_proj"], hn).view(R, self.nh, self.D), Lw["q_norm"], self.eps), pos_r, self.theta, self.D)
            k = _rope(rms(_lin(Lw["k_proj"], hn).view(R, self.nk, self.D), Lw["k_norm"], self.eps), pos_r, self.theta, self.D)
            v = _lin(Lw["v_proj"], hn).view(R, self.nk, self.D)
            kv = self.skv[i]
            kv.write_rows(k, v, slot_r, j_r, allr, act, B + 1)          # the block after the context (not counted)
            d0 = kv.n[slot_r] + B - 1                                  # every row sees the whole block: non-causal
            o = inkling_attention(q.unsqueeze(2), kv.k, kv.v, rel, sinks, d0, self.D ** -0.5,
                                  self.W + B - 1 if i in self.local else None, lk=d0 + 1, slot=slot_r)
            x = x + _lin(Lw["o_proj"], o.reshape(R, -1)).float()          # (window anchored at the block start)
            h2 = rms(x, Lw["post_attention_layernorm"], self.eps, self.dt)
            x = x + _lin(Lw["down_proj"], F.silu(_lin(Lw["gate_proj"], h2)) * _lin(Lw["up_proj"], h2)).float()
        hid = rms(x, self.norm, self.eps, self.dt).view(n, B, H)[:, 1:]  # [n, 7, H]
        V = self.t.cfg["vocab"]
        base = _lin(self.t.lm_head, hid.reshape(-1, H))[:, :V].float().view(n, B - 1, V)
        state = torch.zeros(n, self.R, device=dev, dtype=self.dt)
        f, prev = self.sf[slots], anchors
        for j in range(B - 1):
            inp = torch.cat([state, self.w1[prev].to(self.dt), hid[:, j], f], -1)
            g, cand, o = (_lin(self.rnn_proj, inp).float() + self.rnn_bias).chunk(3, -1)
            g = torch.sigmoid(g)
            state = (g * state.float() + (1 - g) * torch.tanh(cand)).to(self.dt)
            lg = fold_logits(torch.tanh(o).to(self.dt), self.fold_w, self.fold_b, base[:, j], self.cap)
            pj, prev = torch.softmax(lg, -1).max(-1)
            if sp is not None:
                prev = prev.clone()
                for i, spi in enumerate(sp):
                    if spi is not None and spi[0] > 0:
                        q = sampling_probs(lg[i], *spi)
                        t_ = torch.multinomial(q, 1)[0]
                        prev[i], pj[i] = t_, q[t_]
                        self.st_ddist[i, j].copy_(q)
            self.st_draft[:n, j].copy_(prev)
            self.st_dprob[:n, j].copy_(pj)

    def static_extend(self, taps, slot_r, j_r, keep_r, last_r, keep, act, pos):
        """The commit of a speculative step: rows keep_r of taps [Rm, n, H] (row r = token j_r of slot slot_r, at
        position pos[slot] + j) into every layer's context, keep[s] counted; f_rnn <- the taps of row last_r[s]."""
        for i, (Lw, kv) in enumerate(zip(self.layers, self.skv)):
            ctx = self._mix(taps, i, Lw["dflare_norm"])
            R = ctx.shape[0]
            k = rms(_lin(Lw["k_proj_ctx"], ctx).view(R, self.nk, self.D), Lw["k_norm_ctx"], self.eps)
            k = _rope(k, pos[slot_r] + j_r, self.theta, self.D)
            v = _lin(Lw["v_proj_ctx"], ctx).view(R, self.nk, self.D)
            kv.write_rows(k, v, slot_r, j_r, keep_r, act, self.block + 1)
            kv.commit(keep)
        f = self._mix(taps[last_r], -1, self.m_norm)
        self.sf.copy_(torch.where((keep > 0)[:, None], f, self.sf))
