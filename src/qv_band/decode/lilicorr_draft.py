"""LiLiCorr parallel drafter (Rusanovsky et al. 2026, arXiv 2608.20530) for Nemotron-3.5-Bolt-74B-A9B.

The checkpoint (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr) is a DFlash-2 backbone -- fc over six
captured target layers, five sliding-window dense layers with grouped dynamic convs -- plus the LiLiCorr head:
a two-layer bidirectional transformer over the (B-1) x K lattice of the drafter's own top-K candidates, whose
output is two unit vectors per candidate (in / out); adjacent slots are coupled by sigma * cos(out_s, in_s+1)
and the block is committed left to right by an argmax walk.

The backbone, context handling and rollback bookkeeping are DFlashDrafter's; what is new here is
  * NVFP4 (modelopt W4A16) dequantisation -- the draft ships 4-bit linears, run here as dense bf16;
  * attention sinks (a learned per-head logit that takes softmax mass, value 0) and a NON-causal window;
  * the head.
The paper and the checkpoint leave a few conventions open (sign of the signed-slot-distance bias, nibble order,
placement of the feature norm); they are instance attributes so they can be settled by measured acceptance. A
wrong choice only lowers acceptance: verification is exact, so the text never changes.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from qv_band.decode.dflash_draft import DFlashDrafter, _DynConv, _lin, _rms

_E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def dequant_nvfp4(w, scale, scale2, low_first: bool = True):
    """modelopt NVFP4 -> float: two E2M1 nibbles per byte, one E4M3 scale per 16 input values, one fp32 tensor scale."""
    lo, hi = (w & 15).long(), (w >> 4).long()
    a, b = (lo, hi) if low_first else (hi, lo)
    lut = _E2M1.to(w.device)
    v = torch.stack([lut[a], lut[b]], -1).reshape(w.shape[0], -1)
    return (v.view(w.shape[0], -1, 16) * scale.float()[..., None] * scale2.float()).reshape(w.shape[0], -1)


def load_lilicorr_weights(path: str, device, dtype, low_first: bool = True) -> dict:
    from safetensors import safe_open
    out = {}
    with safe_open(path, "pt", device=str(device)) as f:
        keys = set(f.keys())
        for k in keys:
            if k.endswith(".weight_scale") or k.endswith(".weight_scale_2"):
                continue
            t = f.get_tensor(k)
            if t.dtype == torch.uint8:
                base = k[: -len(".weight")]
                t = dequant_nvfp4(t, f.get_tensor(base + ".weight_scale"), f.get_tensor(base + ".weight_scale_2"), low_first)
            out[k] = t.to(dtype)
    return out


class LiLiCorrDrafter(DFlashDrafter):
    version = 3
    returns_probs = False        # the draft is a deterministic path (point-mass proposal)

    def __init__(self, draft_dir: str, embed, lm_head, *, device="cuda", dtype=torch.bfloat16,
                 low_first: bool = True, rel_sign: int = 1, **_):
        cfg = json.load(open(os.path.join(draft_dir, "config.json")))
        if "LiLiCorrDraftModel" not in (cfg.get("architectures") or []):
            raise ValueError(f"{draft_dir} is not a LiLiCorr draft model")
        dc = cfg["dflash_config"]
        self.embed, self.lm_head = embed, lm_head
        self.device, self.dtype = torch.device(device), dtype
        self.H, self.n_heads, self.n_kv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd, self.eps, self.n_layers = cfg["head_dim"], cfg["rms_norm_eps"], cfg["num_hidden_layers"]
        win = int(dc["swa_window_size"])
        self.layer_attn = [(False, win)] * self.n_layers          # bidirectional inside the window
        self.ctx_keep = win
        self.block_size = int(cfg["block_size"])
        self.mask_token_id = int(dc["mask_token_id"])
        self.capture_layers = list(dc["target_layer_ids"])
        self.top_k = int(dc["lilicorr_candidate_topk"])
        self.sigma = float(dc["lilicorr_logit_scale"])
        self.vec_eps = float(dc["lilicorr_vector_eps"])
        self.low_first, self.rel_sign = low_first, rel_sign
        theta = float(cfg.get("rope_theta", 10000.0))
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32, device=self.device) / self.hd))
        w = load_lilicorr_weights(os.path.join(draft_dir, "model.safetensors"), self.device, dtype, low_first)
        self.param_bytes = sum(v.numel() * v.element_size() for v in w.values())
        self.fc, self.hidden_norm, self.norm = w["fc.weight"], w["hidden_norm.weight"], w["norm.weight"]
        self.fc_in = self.fc.shape[-1]
        ks, gs = int(dc["conv_kernel_size"]), int(dc["conv_group_size"])
        self.layers = []
        for i in range(self.n_layers):
            p = f"layers.{i}."
            L = {n: w[p + n + ".weight"] for n in (
                "input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj",
                "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
            L["sink"] = w[p + "self_attn.attention_sink_bias"]
            L["attn_conv"] = _DynConv(w[p + "attention_conv.base_kernel"], w[p + "attention_conv.kernel_projection.weight"], ks, gs)
            L["mlp_conv"] = _DynConv(w[p + "mlp_conv.base_kernel"], w[p + "mlp_conv.kernel_projection.weight"], ks, gs)
            self.layers.append(L)
        self.lw = {k[len("lilicorr."):]: v for k, v in w.items() if k.startswith("lilicorr.")}
        self.n_head_layers = int(dc["lilicorr_num_layers"])
        self.head_heads = int(dc["lilicorr_num_heads"])
        del w
        self.var = {"tgt_norm": True}   # convention switches for the open points (see module docstring)
        self._target_dtype = None
        self._last_th = None
        self._reset_state()

    # -- sink attention: softmax over [keys..., sink] with the sink value 0 ----------------------
    def _attend(self, q, k, v, mask, i):
        g = self.n_heads // self.n_kv
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
        s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (self.hd ** -0.5)       # (1, nh, bs, n)
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        sink = self.layers[i]["sink"].float().view(1, -1, 1, 1).expand(1, -1, s.shape[2], 1)
        p = torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)[..., :-1]
        return torch.matmul(p.to(v.dtype), v)

    # -- the LiLiCorr head ---------------------------------------------------------------------
    def _rms1(self, x, name):
        return _rms(x, self.lw[name], 1e-6)

    @torch.no_grad()
    def _lattice(self, hid, logits):
        """-> candidates (S, K) long, anchor scores z0 (K,), transition scores (S-1, K, K)."""
        lw, K, S = self.lw, self.top_k, hid.shape[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        topv, topi = torch.topk(lp, K, dim=-1)                                   # sorted descending
        rank = torch.arange(K, device=hid.device, dtype=torch.float32)
        feats = torch.stack([topv, topv.exp(), topv - topv[:, :1], (rank / (K - 1)).expand(S, K),
                             (rank == 0).float().expand(S, K)], dim=-1)          # (S, K, 5)
        if self.var.get("no_fnorm"):
            f = feats.to(self.dtype)
        else:
            f = F.layer_norm(feats, (5,), lw["feature_norm.weight"].float(), lw["feature_norm.bias"].float()).to(self.dtype)
        f = F.linear(F.silu(F.linear(f, lw["feature_mlp.up_proj.weight"], lw["feature_mlp.up_proj.bias"])),
                     lw["feature_mlp.down_proj.weight"], lw["feature_mlp.down_proj.bias"])
        tok = F.linear(self.embed(topi.reshape(-1)).view(S, K, -1).to(self.dtype), lw["token_proj.weight"], lw["token_proj.bias"])
        hs = F.linear(hid.to(self.dtype), lw["pass_hidden_proj.weight"], lw["pass_hidden_proj.bias"])[:, None]
        x = (tok + hs + f + lw["slot_embedding"][0, 0] + lw["rank_embedding"][0, 0, 0]).reshape(S * K, -1)
        slot = torch.arange(S, device=hid.device).repeat_interleave(K)
        dist = (slot[None, :] - slot[:, None]) * self.rel_sign                  # signed slot distance j - i
        rel = lw["relative_slot_bias"].float()[:, (dist + (self.block_size - 1)).clamp(0, 2 * self.block_size - 2)]
        bias = rel + lw["same_slot_bias"].float().view(-1, 1, 1) * (dist == 0).float()          # (heads, N, N)
        nh, N = self.head_heads, S * K
        for j in range(self.n_head_layers):
            a = f"layers.{j}."
            h = self._rms1(x, a + "attn_norm.weight")
            qkv = F.linear(h, lw[a + "attn.in_proj_weight"], lw[a + "attn.in_proj_bias"]).view(N, 3, nh, -1)
            q, k, v = (qkv[:, t].transpose(0, 1) for t in range(3))              # (nh, N, hd)
            sc = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5) + bias
            o = torch.matmul(torch.softmax(sc, -1).to(v.dtype), v).transpose(0, 1).reshape(N, -1)
            x = x + F.linear(o, lw[a + "attn.out_proj.weight"], lw[a + "attn.out_proj.bias"])
            h = self._rms1(x, a + "mlp_norm.weight")
            x = x + F.linear(F.silu(F.linear(h, lw[a + "mlp.up_proj.weight"], lw[a + "mlp.up_proj.bias"])),
                             lw[a + "mlp.down_proj.weight"], lw[a + "mlp.down_proj.bias"])
        x = self._rms1(x, "output_norm.weight")
        th = self._last_th.to(self.dtype)
        if self.var.get("raw_th"):                                              # skip hidden_norm: fc output as is
            th = F.linear(self._last_feats.to(self.dtype), self.fc)
        tgt = F.linear(th, lw["context_proj.weight"], lw["context_proj.bias"])  # (1024,)
        tg = self._rms1(tgt, "anchor_norm.weight") if self.var.get("tgt_norm") else tgt
        cat = {"xtp": [x, tg.expand_as(x), x * tg], "txp": [tg.expand_as(x), x, x * tg]}[self.var.get("order", "xtp")]
        fused = F.silu(F.linear(torch.cat(cat, dim=-1), lw["factor_input_proj.weight"], lw["factor_input_proj.bias"]))
        nrm = lambda t: F.normalize(t.float(), dim=-1, eps=self.vec_eps)
        hn_o, hn_i = ("in_head", "out_head") if self.var.get("swap") else ("out_head", "in_head")
        out = nrm(F.linear(fused, lw[hn_o + ".weight"], lw[hn_o + ".bias"])).view(S, K, -1)
        inn = nrm(F.linear(fused, lw[hn_i + ".weight"], lw[hn_i + ".bias"])).view(S, K, -1)
        o0 = nrm(F.linear(self._rms1(tgt, "anchor_norm.weight"), lw["anchor_out_head.weight"], lw["anchor_out_head.bias"]))
        if self.var.get("anchor_raw"):
            o0 = nrm(F.linear(tgt, lw["anchor_out_head.weight"], lw["anchor_out_head.bias"]))
        z0 = self.sigma * (inn[0] @ o0)                                          # (K,)
        zt = self.sigma * torch.einsum("skd,sjd->skj", out[:-1], inn[1:])        # (S-1, K, K): out_s[k] . in_{s+1}[j]
        return topi, z0, zt

    @torch.no_grad()
    def propose(self, last_token: int, k: int, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0):
        if k <= 0:
            return [], None
        start = self.pos - 1
        if self._ctx_end != start or self._last_th is None:
            return [], None
        hid = self._block_hidden(last_token, start)
        logits = self.lm_head(hid.to(self._target_dtype))
        cand, z0, zt = self._lattice(hid, logits)
        cand, z0, zt = cand.cpu(), z0.cpu(), zt.cpu()
        j = int(z0.argmax())
        path = [int(cand[0, j])]
        for s in range(1, cand.shape[0]):
            j = int(zt[s - 1, j].argmax())
            path.append(int(cand[s, j]))
        self.blocks += 1
        return path[:k], None
