"""Drafter kind: DFlash2 with NoPE layers (drafter config `no_rope_layers`). build_dflash_drafter selects it from
the drafter's config. Used by: ricursiveAI/Ripple-65B-Preview-DFlash2.

Written as a SEPARATE module so the Qwen DFlash2 implementation (dflash_draft.py) stays untouched.
Differences from the Qwen drafter, all handled here:
  * NoPE layers: config `no_rope_layers` (draft layer 4, the global-attention layer) skip rotary
    embeddings for queries AND keys (context K and block K/Q).
  * Optional embedding multiplier (QV_DRAFT_EMBED_SCALE=1 -> target config embed_scale): Ripple applies
    embed_scale in its model forward, not inside embed_tokens; whether the drafter was trained on scaled
    block embeddings is unconfirmed (one short prompt could not tell: 95/150 vs 91/150 accepted).
Everything else (kernel-4 dynamic convs, rank-256 selector, top-k 32, block 16) is config-driven in the base.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from qv_band.decode.dflash_draft import DFlashDrafter, _lin, _rms


class NoPEDFlashDrafter(DFlashDrafter):
    def __init__(self, draft_dir, embed, lm_head, **kw):
        super().__init__(draft_dir, embed, lm_head, **kw)
        cfg = json.load(open(os.path.join(draft_dir, "config.json")))
        self.no_rope = set(cfg.get("no_rope_layers") or ())
        for i, L in enumerate(self.layers):
            L["nope"] = i in self.no_rope
        self.embed_mult = 1.0
        # The global (NoPE) draft layer was trained on 16Ki-token rows (reference runtime refuses a smaller
        # draft window); without a cap the base class keeps the whole context for it.
        if self._ctx_cap is None:
            self._ctx_cap = int(os.environ.get("RIPPLE_DRAFT_CTX", "16384"))

    @torch.no_grad()
    def _attend(self, q, k, v, mask, layer_idx):
        # Same GQA folding as the base class, but .reshape on the way out: with a mask present (context beyond
        # the 4096 draft window) SDPA returns a non-contiguous tensor and the base class's .view fails.
        b, H, bs, d = q.shape
        kvh = k.shape[1]
        if kvh == H:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        g = H // kvh
        if mask is not None:
            mask = mask.expand(b, 1, bs, mask.shape[-1]).repeat(1, 1, g, 1)
        o = F.scaled_dot_product_attention(q.reshape(b, kvh, g * bs, d), k, v, attn_mask=mask)
        return o.reshape(b, H, bs, d)

    def _kv(self, L, x, cos, sin):
        if L["nope"]:
            cos, sin = torch.ones_like(cos), torch.zeros_like(cos)
        return super()._kv(L, x, cos, sin)

    @torch.no_grad()
    def _block_hidden(self, last_token: int, start: int) -> torch.Tensor:
        bs = self.block_size
        ids = torch.full((bs,), self.mask_token_id, dtype=torch.long, device=self.device)
        ids[0] = last_token
        h = self.embed(ids)
        self._target_dtype = h.dtype      # what the target's lm_head consumes
        if self.embed_mult != 1.0:
            h = h * self.embed_mult
        h = h.to(self.dtype)                                     # (bs, H)
        q_pos = torch.arange(start, start + bs, device=self.device)
        cos, sin = self._cos_sin(q_pos)
        k_pos = torch.cat([torch.arange(self._ctx_start, self._ctx_end, device=self.device), q_pos])
        d = q_pos[:, None] - k_pos[None, :]     # query minus key position
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
        for i, L in enumerate(self.layers):
            mask = masks[self.layer_attn[i]]
            res = h
            x = _rms(h, L["input_layernorm"], self.eps)
            dyn = None
            if L["attn_conv"] is not None:
                x, dyn = L["attn_conv"].prepare(x[None])
                x = x[0]
            q = _lin(x, L["self_attn.q_proj"]).view(bs, self.n_heads, self.hd)
            qc, qs = (torch.ones_like(cos), torch.zeros_like(cos)) if L["nope"] else (cos, sin)
            q = self._rope(_rms(q, L["self_attn.q_norm"], self.eps).transpose(0, 1)[None], qc, qs)
            k, v = self._kv(L, x, cos, sin)
            if self._n:                            # block keys go in the scratch room
                self._reserve(0)
                a0 = self._off + self._n
                self._kb[i][:, :, a0:a0 + bs] = k
                self._vb[i][:, :, a0:a0 + bs] = v
                k = self._kb[i][:, :, self._off:a0 + bs]
                v = self._vb[i][:, :, self._off:a0 + bs]
            a = self._attend(q, k, v, mask, i)
            a = _lin(a[0].transpose(0, 1).reshape(bs, -1), L["self_attn.o_proj"])
            if dyn is not None:
                a = L["attn_conv"].finish(a[None], dyn)[0]
            h = res + a
            res = h
            x = _rms(h, L["post_attention_layernorm"], self.eps)
            dyn = None
            if L["mlp_conv"] is not None:
                x, dyn = L["mlp_conv"].prepare(x[None])
                x = x[0]
            x = _lin(F.silu(_lin(x, L["mlp.gate_proj"])) * _lin(x, L["mlp.up_proj"]),
                         L["mlp.down_proj"])
            if dyn is not None:
                x = L["mlp_conv"].finish(x[None], dyn)[0]
            h = res + x
        return _rms(h, self.norm, self.eps)[1:]                  # (bs-1, H)


    # -- tree-verification feasibility probe (QV_TREE_SIM=<path.jsonl>; off by default, no effect on output) -----
    # Each greedy step also builds ALTERNATIVE chains: at each of the first TREE_SIM_J positions, the selector's
    # 2nd-best candidate is forced and the walk continues greedily (cheap: candidates and their codebook rows
    # are already gathered; no extra draft-model pass). Main chain, alternatives and selector confidences are
    # appended to the jsonl file; scratchpad tree_eval.py scores them against the final greedy output.
    TREE_SIM_J = int(os.environ.get("QV_TREE_SIM_J", "6"))

    @torch.no_grad()
    def propose(self, last_token, k, temperature=0.0, top_k=0, top_p=1.0):
        sim = os.environ.get("QV_TREE_SIM")
        if not sim or temperature > 0 or self.version != 2 or k <= 0:
            return super().propose(last_token, k, temperature, top_k, top_p)
        start = self.pos - 1
        if self._ctx_end != start:
            return [], None
        hid = self._block_hidden(last_token, start)
        logits = self.lm_head(hid.to(self._target_dtype))
        unary, cand = torch.topk(logits.float(), self.top_k, dim=-1, sorted=False)
        hp = _lin(hid, self.sel_proj).float()
        cc = cand.cpu()
        succ = self.sel_succ[cc].to(self.device, non_blocking=True).float()
        psub = self.sel_pred[cc].to(self.device, non_blocking=True)
        prow0 = self.sel_pred[last_token].to(self.device, non_blocking=True)

        def walk(j0, prow, forced=None):
            """Greedy walk from position j0; at j0 take the `forced`-th best (0 = best)."""
            path, conf = [], []
            for j in range(j0, hid.shape[0]):
                scores = unary[j] + succ[j] @ (prow.float() * hp[j])
                if j == j0 and forced:
                    top = torch.topk(scores, forced + 1).indices
                    idx = top[forced]
                else:
                    idx = scores.argmax()
                pr = torch.softmax(scores, -1)
                conf.append(pr[idx])
                prow = psub[j, idx]
                path.append(cand[j, idx])
            return path, conf, prow

        main, mconf, _ = walk(0, prow0)
        alts = []
        prow = prow0
        for j in range(min(self.TREE_SIM_J, hid.shape[0])):
            alt, aconf, _ = walk(j, prow, forced=1)
            alts.append({"j": j, "tok": torch.stack(main[:j] + alt).tolist() if j else torch.stack(alt).tolist()})
            prow = psub[j, (cand[j] == main[j]).nonzero()[0, 0]]
        self.blocks += 1
        toks = torch.stack(main).tolist()
        rec = {"d": id(self), "start": start, "last": int(last_token), "main": toks,
               "conf": torch.stack(mconf).float().tolist(), "alts": alts, "k": k}
        with open(sim, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return toks[:min(k, len(toks))], None



def build_nope_drafter(model, draft_dir, device="cuda", pack_dir=None):
    """Bind a NoPE DFlash2 checkpoint to a loaded target (its embedding and lm_head)."""
    cfg = json.load(open(os.path.join(draft_dir, "config.json")))
    n_target = model.config.num_hidden_layers
    d = NoPEDFlashDrafter(draft_dir, model.get_input_embeddings(), model.get_output_embeddings(),
                            device=device, pack_dir=pack_dir)
    if cfg.get("num_target_layers", n_target) != n_target or max(d.capture_layers) >= n_target:
        raise ValueError(f"{draft_dir} was trained for a {cfg.get('num_target_layers')}-layer target, "
                         f"this model has {n_target}")
    if d.fc_in != len(d.capture_layers) * model.config.hidden_size:
        raise ValueError("draft/target hidden size mismatch")
    if os.environ.get("QV_DRAFT_EMBED_SCALE", "0") == "1":
        d.embed_mult = float(getattr(model.config, "embed_scale", None) or 1.0)
        print(f"drafter: block embeddings x {d.embed_mult:.4f} (target embed_scale)", flush=True)
    print(f"NoPE DFlash2 drafter: {d.param_bytes/1e9:.2f} GB on {device}, block {d.block_size}, "
          f"target layers {d.capture_layers}, NoPE layers {sorted(d.no_rope)}", flush=True)
    return d
