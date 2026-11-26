"""DSpine parallel drafter (Zhang et al. 2026, "Draft in Parallel, Condition Through Depth", arXiv 2609.36173;
tencent/Qwen3.8-27B-DSpine-b16) for the speculative harness in spec_decode.py.

The checkpoint is a DFlash (v1) backbone -- fc over five captured target layers, five causal sliding-window layers,
no dynamic convs, no candidate selector -- plus the DSpine *code bridge* (`selected_prefix_bridge.*`):

  * after EVERY draft layer each block row is read into a unit code (C = 1024); row t then receives a gated message
    from row t-1's code (the anchor row uses the codebook entry of the anchor token), written back into the residual
    stream scaled by the row's RMS. Causal dependence inside the block travels through depth this way while the block
    is still drafted in ONE pass.
  * the LAST write is redone exactly ("serial last-write refinement"): the top-K (16) candidates of every row come
    from the target head; for each (sender candidate j at row t-1, candidate c at row t) the last write is recomputed
    with the sender's TRUE code and scored, and one path is walked left to right.

Port of github.com/Tencent/DSpine (dspine_mlx/{model,bridge,propose}.py, sglang_dspine/models/dspine.py). Written as a
SEPARATE module so the Qwen DFlash implementation (dflash_draft.py) stays untouched; build_dflash_drafter selects it
from the drafter config (`dflash_config.selected_prefix_code_bridge`).

Two things differ from the reference, neither changes what is drafted beyond bf16 rounding:
  * the target head here is a packed (qvb8) module with no dense rows to gather, so everything the tail needs from
    the head goes THROUGH it: the table TG = (W diag(norm) Wwrite) is one head call at load, and the unrefined score of
    a candidate is recovered from the pass-1 logits (base = rms_out * logit - rms * message . TG[c]) instead of a
    second head call;
  * fp32 statistics / bf16 weights as dflash_draft.py; the reference's kernel-exact rounding points are not kept.

Correctness is the harness's job: greedy verification is exact-match. At temperature > 0 propose() returns the
distribution the path was sampled from (mass on the K candidates of each row), as DFlash 2 does.
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

from qv_band.decode.dflash_draft import DFlashDrafter, _lin, _rms

_BRIDGE = "selected_prefix_bridge."
_F32_TINY = 1.1754943508222875e-38


class DSpineDrafter(DFlashDrafter):
    version = 4
    returns_probs = True

    def __init__(self, draft_dir: str, embed, lm_head, *, device="cuda", dtype=torch.bfloat16,
                 pack_dir: str | None = None):
        super().__init__(draft_dir, embed, lm_head, device=device, dtype=dtype, pack_dir=pack_dir)
        cfg = json.load(open(os.path.join(draft_dir, "config.json")))
        dc = cfg["dflash_config"]
        br = dc["selected_prefix_code_bridge"]
        if dc.get("selected_prefix_terminal", True) or dc.get("selected_prefix_feature_write_mode") != "independent" \
                or not dc.get("draft_block_causal") or dc.get("selected_prefix_feature_next_token_rows") \
                or dc.get("selected_prefix_terminal_residual_rank"):
            raise NotImplementedError(f"{draft_dir}: only terminal-disabled, independent-write, causal DSpine drafts")
        active = sorted(int(i) - 1 for i in dc["selected_prefix_feature_layers"])
        if active != list(range(self.n_layers)):
            raise NotImplementedError(f"DSpine code writes on layers {active}: expected every layer")
        self.block_size = int(cfg.get("block_size", dc.get("block_size", 16)))
        self.trained_block = self.block_size
        self.head_topk = int((dc.get("selected_prefix_last_write_refine") or cfg["code_bridge_second_pass"])["head_topk"])
        self.C, self.R, self.b_eps = int(br["code_size"]), int(br["rank"]), float(br["eps"])
        self.sqrt_c = math.sqrt(self.C)
        self.version = 4                          # the base read "DFlashDraftModel" (v1) from the config
        self._graph = False                       # dflash_graph captures the DFlash block; this one runs eager

        from safetensors import safe_open
        with safe_open(os.path.join(draft_dir, "model.safetensors"), "pt", device=str(self.device)) as f:
            man = json.load(open(os.path.join(draft_dir, "manifest.json"))) if os.path.isfile(
                os.path.join(draft_dir, "manifest.json")) else {"tensors": {}}
            keys = set(f.keys())

            def g(n):
                name = _BRIDGE + n
                if name in keys:
                    return f.get_tensor(name)
                info = man["tensors"][name]       # a pack used as its own checkpoint: decode the packed tensor by
                from qv_band.common.formats import get_format      # pushing an identity through it (every output
                path = os.path.join(draft_dir, info["pack"])        # is one weight times 1.0, as in _swap_packed)
                lin = get_format(os.path.splitext(path)[1].lstrip(".")).linear.from_packfile(path, device=str(self.device))
                eye = torch.eye(info["P"], dtype=torch.float16, device=self.device)
                out = lin(eye).t().contiguous()
                del lin, eye
                return out
            for n in ("independent_write_version", "terminal_disabled_version"):
                if int(g(n)) != 1:
                    raise NotImplementedError(f"DSpine {n} != 1")
            ow = g("out.weight")      # all zero in the checkpoint; a pack decodes it to +-tau at most (certified error)
            tol = float(man.get("tau", 0.0)) * 1.01 if _BRIDGE + "out.weight" not in keys else 0.0
            if int((ow.abs() > tol).sum()):
                raise NotImplementedError("terminal-disabled DSpine checkpoint with a nonzero terminal projection")
            self.codebook = g("codebook").to(dtype)                                   # (V, C)
            self.read_w = list(g("early_read_weights").to(dtype)) + [g("read_weight").to(dtype)]   # per layer (C, H)
            self.write_w = g("feature_write_weights").to(dtype)                       # (L, H, R)
            gate, pgate, prev = g("gate.weight"), g("predecessor_gate.weight"), g("prev.weight")
            self.packed_w = torch.cat((gate, pgate, prev), 0).to(dtype).contiguous()   # (3R, C): [G_q; G_p; A]
            self.gate_b = g("gate.bias").float()
        self.param_bytes += sum(t.numel() * t.element_size() for t in
                                (self.codebook, self.write_w, self.packed_w, *self.read_w))
        self._tg = self._tsv = None               # vocabulary-wide tables, built at the first propose (needs the head)
        dyn = os.environ.get("QV_DSPINE_DYNAMIC", "7:1.22").strip()
        self.dyn_short, self.dyn_ratio = (0, 0.0) if dyn in ("0", "") else (int(dyn.split(":")[0]), float(dyn.split(":")[1]))
        self.dyn_full = self.dyn_cut = 0          # how often the whole draft / the short one was sent

    def fork(self):
        d = super().fork()
        d._graph = False
        return d

    # -- vocabulary-wide tables of the serial refinement ---------------------------------------------------------------
    @torch.no_grad()
    def _tables(self):
        if self._tg is None:
            nw = self.norm.float()
            x = (self.write_w[-1].float() * nw[:, None]).t().contiguous()             # (R, H): rows norm * Wwrite[:, r]
            tdt = self._target_dtype or next(self.lm_head.parameters(), x).dtype
            tg = self.lm_head(x.to(tdt)).float().t().contiguous().to(self.dtype)       # (V, R) = W diag(norm) Wwrite
            pv = self.packed_w[self.R:]                                                # (2R, C) = [G_p; A]
            tsv = torch.empty(self.codebook.shape[0], 2 * self.R, dtype=self.dtype, device=self.device)
            for s in range(0, tsv.shape[0], 32768):
                tsv[s:s + 32768] = F.linear((self.sqrt_c * self.codebook[s:s + 32768].float()).to(self.dtype), pv)
            self._tg, self._tsv = tg, tsv
            # forks share the tables (copy.copy shares attributes set before the fork; later forks read the parent's)
            self._graph_shared["dspine_tables"] = (tg, tsv)
        return self._tg, self._tsv

    # -- one code write after layer `i` (rows of ONE block) ------------------------------------------------------------
    def _write(self, h: torch.Tensor, anchor: int, i: int, stash: dict | None):
        f = h.float()
        rms = f.pow(2).mean(-1).clamp_min(_F32_TINY).sqrt()                            # (T,)
        normalized = (f / (rms[:, None] + self.b_eps)).to(self.dtype)
        proj = F.linear(normalized, self.read_w[i]).float()
        codes = self.sqrt_c * proj / (proj.norm(dim=-1, keepdim=True) + self.b_eps)
        codes[0] = self.sqrt_c * self.codebook[anchor].float()                         # the anchor's true code
        packed = F.linear(codes.to(self.dtype), self.packed_w).float()                 # (T, 3R)
        q, s, v = packed.split(self.R, dim=-1)
        msg = torch.sigmoid(q[1:] + s[:-1] + self.gate_b) * v[:-1]                     # (T-1, R): row t from row t-1
        delta = F.linear(msg.to(self.dtype), self.write_w[i]).float()                  # (T-1, H)
        out = f.clone()
        out[1:] += rms[1:, None] * delta
        if stash is not None:
            stash.update(raw=f[1:], rms=rms[1:], q=q[1:], msg=msg, out=out[1:])
        return out.to(h.dtype)

    @torch.no_grad()
    def _block_hidden(self, last_token: int, start: int, bs: int | None = None):
        """DFlashDrafter._block_hidden (v1: no convs) with a code write after every layer. Returns the final normed
        rows 1..bs-1 and the last write's pre-write state."""
        bs = bs or self.block_size
        ids = torch.full((bs,), self.mask_token_id, dtype=torch.long, device=self.device)
        ids[0] = last_token
        h = self.embed(ids)
        self._target_dtype = h.dtype
        h = h.to(self.dtype)
        q_pos = torch.arange(start, start + bs, device=self.device)
        cos, sin = self._cos_sin(q_pos)
        k_pos = torch.cat([torch.arange(self._ctx_start, self._ctx_end, device=self.device), q_pos])
        d = q_pos[:, None] - k_pos[None, :]
        masks = {}
        for causal, win in set(self.layer_attn):
            m = torch.ones_like(d, dtype=torch.bool)
            if causal:
                m &= d >= 0
            if win is not None:
                m &= d < win
                if not causal:
                    m &= -d < win
            masks[(causal, win)] = None if bool(m.all()) else m[None, None]
        stash = {}
        last = self.n_layers - 1
        for i, L in enumerate(self.layers):
            mask = masks[self.layer_attn[i]]
            res = h
            x = _rms(h, L["input_layernorm"], self.eps)
            q = _lin(x, L["self_attn.q_proj"]).view(bs, self.n_heads, self.hd)
            q = self._rope(_rms(q, L["self_attn.q_norm"], self.eps).transpose(0, 1)[None], cos, sin)
            k, v = self._kv(L, x, cos, sin)
            if self._n:
                self._reserve(0)
                a0 = self._off + self._n
                self._kb[i][:, :, a0:a0 + bs] = k
                self._vb[i][:, :, a0:a0 + bs] = v
                k = self._kb[i][:, :, self._off:a0 + bs]
                v = self._vb[i][:, :, self._off:a0 + bs]
            a = self._attend(q, k, v, mask, i)
            a = _lin(a[0].transpose(0, 1).reshape(bs, -1), L["self_attn.o_proj"])
            h = res + a
            x = _rms(h, L["post_attention_layernorm"], self.eps)
            h = h + _lin(F.silu(_lin(x, L["mlp.gate_proj"])) * _lin(x, L["mlp.up_proj"]), L["mlp.down_proj"])
            h = self._write(h, last_token, i, stash if i == last else None)
        return _rms(h, self.norm, self.eps)[1:], stash

    @torch.no_grad()
    def propose(self, last_token: int, k: int, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0):
        """Draft min(k, block-1) tokens. The block is causal, so a shorter block drafts the same leading tokens as a
        full one (the reference serves block sizes 2..trained)."""
        if k <= 0:
            return [], None
        start = self.pos - 1
        if self._ctx_end != start:
            return [], None
        bs = min(self.trained_block, k + 1)
        if bs < 2:
            return [], None
        rows, st = self._block_hidden(last_token, start, bs)                           # (n, H) normed
        if self._tg is None and "dspine_tables" in self._graph_shared:
            self._tg, self._tsv = self._graph_shared["dspine_tables"]
        tg, tsv = self._tables()
        n, K, R = rows.shape[0], self.head_topk, self.R
        logits = self.lm_head(rows.to(self._target_dtype)).float()                     # (n, V) pass-1
        top, cand = torch.topk(logits, K, dim=-1, sorted=False)                        # (n, K)
        # unrefined score of candidate c at row t: (raw * norm) . W_c, from the pass-1 logit of the normed written row
        rms_out = (st["out"].pow(2).mean(-1) + self.eps).sqrt()                        # (n,)
        tg_c = tg[cand].float()                                                        # (n, K, R)
        base = rms_out[:, None] * top - st["rms"][:, None] * torch.einsum("nr,nkr->nk", st["msg"], tg_c)
        # redo the last write with each sender candidate's true code
        anchor = torch.full((1, K), int(last_token), dtype=torch.long, device=self.device)
        sid = torch.cat([anchor, cand[:-1]], 0) if n > 1 else anchor                   # (n, K) senders of row t
        tv = tsv[sid].float()                                                          # (n, K, 2R)
        m = torch.sigmoid(st["q"][:, None, :] + tv[..., :R] + self.gate_b) * tv[..., R:]   # (n, K, R)
        score = base[:, None, :] + st["rms"][:, None, None] * torch.einsum("njr,ncr->njc", m, tg_c)   # (n, K_sender, K)
        self.blocks += 1
        dynamic = bool(self.dyn_short) and n > self.dyn_short
        table = None
        if temperature > 0 or dynamic:
            # the true logits: the refined row is normalised by ITS rms before the head
            refined = st["raw"][:, None, :] + st["rms"][:, None, None] * F.linear(m.to(self.dtype), self.write_w[-1]).float()
            table = score / (refined.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()  # (n, K, K)
        if temperature <= 0:
            idx = score.argmax(-1)                                                     # (n, K)
            j, path, conf = 0, [], []
            idx_l, cand_l = idx.tolist(), cand.tolist()
            p_l = torch.softmax(table, dim=-1).amax(-1).tolist() if dynamic else None  # (n, K): p of the choice per sender
            for t in range(n):
                if p_l is not None:
                    conf.append(p_l[t][j])
                j = idx_l[t][j]
                path.append(cand_l[t][j])
            return path[:self._send_len(conf, n)], None
        j, path, probs, conf = 0, [], [], []
        V = logits.shape[-1]
        for t in range(n):
            qd = torch.softmax(table[t, j] / temperature, dim=-1)
            j = int(torch.multinomial(qd, 1))
            conf.append(float(qd[j]))
            path.append(int(cand[t, j]))
            probs.append(torch.zeros(V, device=self.device).scatter_(0, cand[t], qd))
        keep = self._send_len(conf if dynamic else [], n)
        return path[:keep], probs[:keep]

    def _send_len(self, conf: list, n: int) -> int:
        """How many of the n drafted tokens to send for verification (see QV_DSPINE_DYNAMIC in __init__)."""
        short = self.dyn_short
        if not conf or n <= short:
            return n
        s, head, total = 1.0, 1.0, 1.0
        for t, p in enumerate(conf):
            s *= p
            total += s
            if t < short:
                head += s
        if total / head > self.dyn_ratio:
            self.dyn_full += 1
            return n
        self.dyn_cut += 1
        return short
