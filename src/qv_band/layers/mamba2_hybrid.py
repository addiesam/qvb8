"""Layer kind: Mamba-2 / attention / latent-MoE hybrid stack (NVIDIA Nemotron-H) -- skeleton, forwards, fast paths.

The checkpoint (NemotronH_Omni_Reasoning_V3, e.g. Nemotron-3.5-Bolt-74B-A9B) is a vision-language wrapper
around a heterogeneous "puzzle" text model: every layer has its own block type and, for MoE layers, its own
expert count and width. transformers has no puzzle class and AutoModel cannot resolve the sub-config's
remote code (it carries no repo path), so the model's OWN modeling files are imported from the checkpoint
folder as a package.

Two deviations from the shipped code, both semantics-preserving:
  * `mamba_ssm` is not installed; the shipped file only needs `rmsnorm_fn` from it at import time and has a
    pure-torch Mamba-2 path for everything else. A torch `rmsnorm_fn` stands in while the file is imported.
  * NemotronHMoE.moe() runs EVERY expert for every token (an empty expert is fed a zero row and its output,
    exactly 0 for a bias-free relu2 MLP, is added). Only experts that received a token are run here.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

import torch
import torch.nn.functional as F

_PKG = "qv_nemotron_h_remote"


_NORM_C = {}


def _rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, **_):
    """Grouped gated RMSNorm. Large CUDA inputs (prefill) take a torch.compile'd fusion of the eight elementwise
    passes (10.3 -> 1.8 ms for 4096 x 8192); small ones (decode, inside CUDA graphs) stay eager."""
    if x.is_cuda and x.numel() // x.shape[-1] > 64 and bias is None and _NORM_C.get("ok", True):
        try:
            f = _NORM_C.get("f")
            if f is None:
                import torch as _t
                f = _NORM_C["f"] = _t.compile(_rmsnorm_eager, dynamic=True)
            return f(x, weight, bias, z, eps, group_size, norm_before_gate)
        except Exception:
            _NORM_C["ok"] = False
    return _rmsnorm_eager(x, weight, bias, z, eps, group_size, norm_before_gate)


def _rmsnorm_eager(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    """torch twin of mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn (grouped, optionally gated)."""
    dt = x.dtype
    x = x.float()
    if z is not None and not norm_before_gate:
        x = x * F.silu(z.float())
    shp = x.shape
    g = group_size or shp[-1]
    xg = x.reshape(*shp[:-1], shp[-1] // g, g)
    xg = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
    y = xg.reshape(shp) * weight.float()
    if bias is not None:
        y = y + bias.float()
    if z is not None and norm_before_gate:
        y = y * F.silu(z.float())
    return y.to(dt)


def _import_remote(model_dir: str):
    """Import <model_dir> as a package and return its modeling_nemotron_h_puzzle module."""
    if _PKG + ".modeling_nemotron_h_puzzle" in sys.modules:
        return sys.modules[_PKG + ".modeling_nemotron_h_puzzle"]
    import transformers.utils.import_utils as iu

    fake = {}
    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton", "mamba_ssm.ops.triton.layernorm_gated"):
        fake[name] = types.ModuleType(name)
    fake["mamba_ssm.ops.triton.layernorm_gated"].rmsnorm_fn = _rmsnorm_fn
    real_avail = iu.is_mamba_2_ssm_available
    saved = {k: sys.modules.get(k) for k in fake}
    try:
        sys.modules.update(fake)
        iu.is_mamba_2_ssm_available = lambda *a, **k: False      # force the torch Mamba-2 path
        spec = importlib.util.spec_from_file_location(
            _PKG, f"{model_dir}/__init__.py", submodule_search_locations=[model_dir])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[_PKG] = pkg
        spec.loader.exec_module(pkg)
        mod = importlib.import_module(_PKG + ".modeling_nemotron_h_puzzle")
    finally:
        iu.is_mamba_2_ssm_available = real_avail
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def _sparse_moe(self, hidden_states, topk_indices, topk_weights):
    """NemotronHMoE.moe() visiting only the experts that got a token: ONE routing sort and ONE host sync per layer
    (the old loop scanned the whole routing table once per expert), tokens gathered once, results index_add-ed.
    Prefill (rows > _TRACE_MAX) takes the grouped bmm path instead, when the layer has an expert stack."""
    if (hidden_states.shape[0] * hidden_states.shape[1] > _TRACE_MAX
            and os.environ.get("QV_NEMO_GEMM", "1") != "0"
            and getattr(self, "_qv_stk", None) is not None
            and getattr(self, "_qv_grp_ok", True)):
        try:
            return _grouped_moe(self, hidden_states, topk_indices, topk_weights)
        except Exception as exc:
            self._qv_grp_ok = False
            print(f"[nemotron] grouped moe disabled: {type(exc).__name__}: {exc}", flush=True)
    T, k = topk_indices.shape
    out = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    flat = topk_indices.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=len(self.experts)).tolist()
    tok = order // k                                         # token of each expert-sorted (token, slot) pair
    w = topk_weights.reshape(-1)[order]
    xs = hidden_states.index_select(0, tok)
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        end = start + c
        y = self.experts[e](xs[start:end]) * w[start:end, None]
        out.index_add_(0, tok[start:end], y.to(out.dtype))
        start = end
    return out.type(hidden_states.dtype)


def _grouped_moe(self, hidden_states, topk_indices, topk_weights):
    """Prefill twin of _sparse_moe: ONE stacked decode per projection (every expert's weights in a single
    kernel over the _qv_stk buffer) and ONE bmm per projection over the expert-sorted, padded tokens,
    instead of 2*E per-expert decode+GEMM pairs. Same maths; the differences are the padded GEMM shapes
    (rows up to the busiest expert) and the fp16 bmm operands/store (the certified fp16 decode is the
    weight operand, exactly as in the per-expert path). QV_NEMO_GEMM=0 keeps the per-expert loop."""
    from qv_band.formats.qvb8.module import _load_ext
    T, k = topk_indices.shape
    E = len(self.experts)
    u, d = self._qv_stk
    dev = hidden_states.device
    N = T * k
    flat = topk_indices.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    Mmax = int(counts.max())
    if Mmax == 0:
        return torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    starts = torch.cumsum(counts, 0) - counts
    e_sorted = flat[order]
    rank = torch.arange(N, device=dev) - starts[e_sorted]      # position of each pair inside its expert
    pos = torch.full((E, Mmax), -1, dtype=torch.long, device=dev)
    pos[e_sorted, rank] = torch.arange(N, device=dev)
    tok = order // k
    xs = hidden_states.index_select(0, tok)
    xpad = xs[pos.clamp_(min=0)].to(torch.float16)             # fp16 bmm operand (the per-expert path narrows too)
    ext = _load_ext()
    _ei = torch.empty(0, dtype=torch.int64, device=dev)       # _qv_stk exists => no escape entries
    _ev = torch.empty(0, dtype=torch.float32, device=dev)
    W = ext.decode(u["ids"], u["min_s"].reshape(-1), u["nbits"].reshape(-1),
                   u["bit_off"].reshape(-1), u["P"], u["tau_s"].reshape(-1),
                   _ei, _ev).view(E, u["Q"], u["P"])
    h = torch.bmm(xpad, W.transpose(1, 2))
    h.clamp_min_(0)
    h.mul_(h)                                                  # relu^2, in place
    W = ext.decode(d["ids"], d["min_s"].reshape(-1), d["nbits"].reshape(-1),
                   d["bit_off"].reshape(-1), d["P"], d["tau_s"].reshape(-1),
                   _ei, _ev).view(E, d["Q"], d["P"])
    h = torch.bmm(h, W.transpose(1, 2))
    y = h[e_sorted, rank] * topk_weights.reshape(-1)[order, None]
    out = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    out.index_add_(0, tok, y.to(out.dtype))
    return out.type(hidden_states.dtype)


_TRACE_MAX = 8
_FLA = []


def _fla_gla():
    """fla.ops.simple_gla.chunk_simple_gla, or None (QV_NEMO_SSD=torch, or fla missing/unusable)."""
    if not _FLA:
        import os
        fn = None
        if os.environ.get("QV_NEMO_SSD", "fla") != "torch":
            try:
                from fla.ops.simple_gla import chunk_simple_gla as fn
            except Exception:
                fn = None
        _FLA.append(fn)
    return _FLA[0]


def _mamba2_forward(self, hidden_states, cache_params=None, attention_mask=None):
    """Mamba-2 mixer for ANY chunk length with carried conv / SSM state (the shipped torch path treats every
    call after the first as a single-token step, so chunked prefill and multi-token verify are wrong there,
    and its einsum-free SSD materialises b*c*l*s*h*n floats). Batch 1; padding masks are not supported.

    State layout is the shipped cache's: conv_states[i] = last K pre-conv inputs [b, conv_dim, K],
    ssm_states[i] = [b, heads, head_dim, state]. Zero state == start of sequence."""
    b, L, _ = hidden_states.shape
    dt_in = hidden_states.dtype
    H, P, N, G = self.num_heads, self.head_dim, self.ssm_state_size, self.n_groups
    proj = self.in_proj(hidden_states)
    gate, xBC, dt = proj.split([self.intermediate_size, self.conv_dim, H], dim=-1)

    K = self.conv_kernel_size
    if cache_params is not None:
        prev = cache_params.conv_states[self.layer_idx]                    # [b, conv_dim, K]
        if prev is None or prev.shape[0] != b:
            # Hybrid banded cache (kv.banded_cache.HybridBandedCache): the mamba state is
            # allocated lazily at the first forward's batch size and re-allocated when the
            # batch axis changes under continuous batching. The model's own
            # NemotronHHybridDynamicCache always holds a batch-1 tensor, so this branch is
            # never taken there.
            cache_params._qv_ensure_mamba_batch(b, dt_in, xBC.device)
            prev = cache_params.conv_states[self.layer_idx]
        h0 = cache_params.ssm_states[self.layer_idx].float()
    else:
        prev = xBC.new_zeros(b, self.conv_dim, K)
        h0 = xBC.new_zeros(b, H, P, N, dtype=torch.float32)
    xt = torch.cat([prev[:, :, 1:].to(xBC.dtype), xBC.transpose(1, 2)], dim=-1)   # [b, conv_dim, K-1+L]
    if cache_params is not None:
        cache_params.conv_states[self.layer_idx].copy_(xt[:, :, -K:] if xt.shape[-1] >= K else
                                                       F.pad(xt, (K - xt.shape[-1], 0)))
    _cc = _causal_conv1d()
    if _cc is not None:
        # one fused kernel for the depthwise causal conv + SiLU (self.act is SiLU: config.mamba_hidden_act)
        # instead of F.conv1d + a separate activation pass over [b, conv_dim, L]. The kernel treats the
        # input as starting at position 0 (zero before it), so the K-1 warm-up positions of xt are dropped.
        xBC = _cc(xt, self.conv1d.weight.squeeze(1), self.conv1d.bias,
                  activation="silu")[:, :, K - 1:].transpose(1, 2)
    else:
        conv = F.conv1d(xt, self.conv1d.weight, self.conv1d.bias, groups=self.conv_dim)   # [b, conv_dim, L]
        xBC = self.act(conv).transpose(1, 2)
    x, Bm, Cm = xBC.split([self.intermediate_size, G * N, G * N], dim=-1)

    A = -torch.exp(self.A_log.float())                                       # [H]
    dtv = torch.clamp(F.softplus(dt.float() + self.dt_bias.float()), self.time_step_min)   # [b, L, H]
    x = x.reshape(b, L, H, P).float()
    Bm = Bm.reshape(b, L, G, N).float()
    Cm = Cm.reshape(b, L, G, N).float()
    rep = H // G

    if L == 1:                                                               # recurrent step
        dA = torch.exp(dtv[:, 0] * A)                                        # [b, H]
        Bh = Bm[:, 0].repeat_interleave(rep, dim=1)                          # [b, H, N]
        Ch = Cm[:, 0].repeat_interleave(rep, dim=1)
        h = h0 * dA[..., None, None] + (dtv[:, 0, :, None] * x[:, 0])[..., None] * Bh[:, :, None, :]
        y = torch.einsum("bhpn,bhn->bhp", h, Ch)[:, None]
    elif getattr(cache_params, "trace", None) is not None and L <= _TRACE_MAX:   # verify block, closed form
        if getattr(self, "_tr_x", None) is None:
            T = _TRACE_MAX
            dev = x.device
            self._tr_x = torch.empty(b, self.conv_dim, K - 1 + T, dtype=xt.dtype, device=dev)
            self._tr_cum = torch.empty(b, T, H, dtype=torch.float32, device=dev)
            self._tr_u = torch.empty(b, T, H, P, dtype=torch.float32, device=dev)
            self._tr_B = torch.empty(b, T, H, N, dtype=torch.float32, device=dev)
            self._tr_h0 = torch.empty(b, H, P, N, dtype=torch.float32, device=dev)
        Bh = Bm.repeat_interleave(rep, dim=2)                                # (b, L, H, N)
        Ch = Cm.repeat_interleave(rep, dim=2)
        cum = torch.cumsum(dtv * A, dim=1)                                   # (b, L, H) inclusive log-decay
        u = dtv[..., None] * x                                               # (b, L, H, P)
        tri = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        seg = (cum[:, :, None, :] - cum[:, None, :, :]).masked_fill(~tri[None, :, :, None], float("-inf"))   # (b,t,s,H)
        Mm = torch.exp(seg) * torch.einsum("bthn,bshn->btsh", Ch, Bh)
        y = torch.einsum("btsh,bshp->bthp", Mm, u) \
            + torch.exp(cum)[..., None] * torch.einsum("bhpn,bthn->bthp", h0, Ch)
        h = torch.exp(cum[:, -1])[..., None, None] * h0 + torch.einsum(
            "bsh,bshp,bshn->bhpn", torch.exp(cum[:, -1:] - cum), u, Bh)
        self._tr_x[:, :, :xt.shape[-1]].copy_(xt)
        self._tr_cum[:, :L].copy_(cum)
        self._tr_u[:, :L].copy_(u)
        self._tr_B[:, :L].copy_(Bh)
        self._tr_h0.copy_(h0)
    elif x.is_cuda and _fla_gla() is not None and L > _TRACE_MAX:                          # prefill chunk: fla's fused scan
        # Mamba-2 SSD == scalar-decay linear attention: h_t = exp(a_t) h_{t-1} + B_t (x) (dt_t x_t), y_t = C_t h_t.
        # fp32 inputs (state error 4e-4 vs 1.9e-3 for bf16: the final state is carried into decode).
        o, ht = _fla_gla()(
            q=Cm.repeat_interleave(rep, dim=2), k=Bm.repeat_interleave(rep, dim=2), v=dtv[..., None] * x,
            g=dtv * A, scale=1.0, initial_state=h0.transpose(-1, -2).contiguous(), output_final_state=True)
        y = o.float()
        h = ht.transpose(-1, -2).float()
    else:                                                                    # chunked SSD with initial state
        cs = self.chunk_size
        pad = (-L) % cs
        if pad:
            x, Bm, Cm = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (x, Bm, Cm))
            dtv = F.pad(dtv, (0, 0, 0, pad))                                 # dt = 0 -> identity step
        C_ = x.shape[1] // cs
        xb = (x * dtv[..., None]).reshape(b, C_, cs, H, P)
        a = (dtv * A).reshape(b, C_, cs, H).permute(0, 3, 1, 2)             # [b, H, C, cs]
        acs = torch.cumsum(a, dim=-1)
        Bc = Bm.reshape(b, C_, cs, G, N)
        Cc = Cm.reshape(b, C_, cs, G, N)
        CB = torch.einsum("bclgn,bcsgn->bcgls", Cc, Bc).repeat_interleave(rep, dim=2)   # [b, C, H, cs, cs]
        seg = acs[..., :, None] - acs[..., None, :]                          # [b, H, C, l, s]
        tri = torch.tril(torch.ones(cs, cs, dtype=torch.bool, device=x.device))
        Lm = torch.exp(seg.masked_fill(~tri, float("-inf"))).permute(0, 2, 1, 3, 4)       # [b, C, H, l, s]
        y_diag = torch.einsum("bchls,bcshp->bclhp", CB * Lm, xb)
        decay_s = torch.exp(acs[..., -1:] - acs).permute(0, 2, 3, 1)         # [b, C, cs, H]
        Bh = Bc.repeat_interleave(rep, dim=3)                                # [b, C, cs, H, N]
        st = torch.einsum("bcshn,bcshp->bchpn", Bh * decay_s[..., None], xb)  # chunk-local end states
        chunk_decay = torch.exp(acs[..., -1]).permute(0, 2, 1)               # [b, C, H]
        hs, h = [], h0
        for c in range(C_):
            hs.append(h)
            h = h * chunk_decay[:, c, :, None, None] + st[:, c]
        h_in = torch.stack(hs, dim=1)                                        # [b, C, H, P, N] state entering chunk
        Ch = Cc.repeat_interleave(rep, dim=3)
        y_off = torch.einsum("bclhn,bchpn->bclhp", Ch, h_in) * torch.exp(acs).permute(0, 2, 3, 1)[..., None]
        y = (y_diag + y_off).reshape(b, C_ * cs, H, P)[:, :L]
    y = y + self.D.float()[None, None, :, None] * x[:, :L]
    if cache_params is not None:
        cache_params.ssm_states[self.layer_idx].copy_(h.to(cache_params.ssm_states[self.layer_idx].dtype))
    y = self.norm(y.reshape(b, L, -1).to(dt_in), gate)
    return self.out_proj(y.to(dt_in))


def _flash_ok() -> bool:
    """flash-attn imports AND launches on this GPU (an sm mismatch only shows at the first call)."""
    try:
        from flash_attn import flash_attn_func
        q = torch.zeros(1, 8, 2, 64, device="cuda", dtype=torch.bfloat16)
        flash_attn_func(q, q, q, causal=True)
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


_CC = {}


def _causal_conv1d():
    """causal_conv1d_fn if it imports AND launches on this GPU (an sm mismatch only shows at the first call);
    QV_NEMO_CONV=0 keeps the F.conv1d + act path. The test uses the real shape class (dim 10240, kernel 4)
    so the first real call is never the first launch."""
    if os.environ.get("QV_NEMO_CONV", "1") == "0":
        return None
    f = _CC.get("f")
    if f is None:
        try:
            from causal_conv1d import causal_conv1d_fn
            x = torch.zeros(1, 10240, 4099, device="cuda", dtype=torch.bfloat16)
            w = torch.zeros(10240, 4, device="cuda", dtype=torch.bfloat16)
            causal_conv1d_fn(x, w, None, activation="silu")
            torch.cuda.synchronize()
            f = _CC["f"] = causal_conv1d_fn
        except Exception:
            f = _CC["f"] = False
    return f or None


def _flash_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **_):
    """transformers attention-interface function: FlashAttention-2, native GQA (32 q / 2 kv heads), causal with
    the query block aligned to the END of the keys (chunked prefill and 1-token decode both). The model's mask
    is not needed and not built (see _no_mask)."""
    from flash_attn import flash_attn_func
    o = flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                        dropout_p=0.0, softmax_scale=scaling, causal=True)
    return o, None


def rollback(model, cache, keep: int, attn_len_before: int) -> None:
    """Undo a traced verify pass down to its first `keep` tokens. Every Mamba layer rebuilds the state it had after
    token `keep` from the block inputs its mixer kept (_tr_*): h_keep = exp(cum_k) h0 + sum_{s<=k} exp(cum_k - cum_s)
    u_s (x) B_s; the conv window comes from _tr_x; the attention KV cache is cropped."""
    K = model.config.conv_kernel
    k = keep - 1
    for i, blk in enumerate(model.model.layers):
        if blk.block_type == "mamba":
            mx = blk.mixer
            cum = mx._tr_cum[:, :keep]
            w = torch.exp(cum[:, k:k + 1] - cum)                                          # (b, keep, H)
            h = torch.exp(cum[:, k])[..., None, None] * mx._tr_h0 + torch.einsum(
                "bsh,bshp,bshn->bhpn", w, mx._tr_u[:, :keep], mx._tr_B[:, :keep])
            cache.ssm_states[i].copy_(h)
            cache.conv_states[i].copy_(mx._tr_x[:, :, k: k + K])
        elif blk.block_type == "attention":
            n = attn_len_before + keep
            cache.key_cache[i] = cache.key_cache[i][:, :, :n]
            cache.value_cache[i] = cache.value_cache[i][:, :, :n]


# ---------------------------------------------------------------------------------------------------------------
# Fast paths: fused routed experts + CUDA-graph replay (decode / verify blocks have no host sync left)
# ---------------------------------------------------------------------------------------------------------------
def _stack_experts(mods):
    """Concatenate the packed ids of every expert (see QVBandExpertsV8.build_stack); None when not eligible
    (escape entries, mixed shapes, odd P)."""
    if not mods or not mods[0].ids.is_cuda or any(m.esc_idx.numel() for m in mods):
        return None
    if any(m.out_features != mods[0].out_features or m.in_features != mods[0].in_features for m in mods):
        return None
    if mods[0].in_features % 8:
        return None
    dev = mods[0].ids.device
    offs, total = [], 0
    for m in mods:
        offs.append(total)
        total += (m.ids.numel() + 15) // 16 * 16
    ids_all = torch.zeros(total + 64, dtype=torch.uint8, device=dev)
    for off, m in zip(offs, mods):
        n = m.ids.numel()
        ids_all[off:off + n].copy_(m.ids)
        m.ids = ids_all[off:off + n]                      # view; frees the original
    return dict(ids=ids_all,
                min_s=torch.stack([m.min_s for m in mods]).contiguous(),
                tau_s=torch.stack([m.tau_s for m in mods]).contiguous(),
                nbits=torch.stack([m.nbits for m in mods]).contiguous(),
                bit_off=torch.stack([m.bit_off + o * 8 for m, o in zip(mods, offs)]).contiguous(),
                Q=int(mods[0].out_features), P=int(mods[0].in_features))


def _fused_moe_forward(self, hidden_states):
    """NemotronHMoE.forward on the fused relu^2 expert kernel: router, latent in, experts, latent out, shared."""
    from qv_band.formats.qvb8.module import _load_ext
    shp = hidden_states.shape
    if shp[0] * shp[1] > _TRACE_MAX:                 # prefill: grouped per-expert GEMMs beat the per-pair kernel
        return type(self).forward(self, hidden_states)
    residuals = hidden_states
    idx, w = self.gate(hidden_states)
    x = self.fc1_latent_proj(hidden_states.view(-1, shp[-1]))
    u, d = self._qv_stk
    y = _load_ext().moe_experts_relu2(
        x.contiguous(), idx.contiguous(), w.float(),
        u["ids"], u["min_s"], u["tau_s"], u["nbits"], u["bit_off"], u["Q"], u["P"],
        d["ids"], d["min_s"], d["tau_s"], d["nbits"], d["bit_off"], d["Q"], d["P"])
    return self.fc2_latent_proj(y).view(shp) + self.shared_experts(residuals)


class _Graphed:
    """CUDA-graph replay of a module call for small row counts. `kind` 'moe': no side effects. 'mamba': only the
    1-token step with an untraced persistent cache; the layer's states are snapshotted around warm-up (the warm-up
    runs update them in place)."""

    def __init__(self, mod, kind: str, max_rows: int = 8):
        self.mod, self.orig, self.kind, self.max_rows = mod, mod.forward, kind, max_rows
        self.graphs, self.disabled = {}, False

    def _capture(self, key, x, cache):
        try:
            static_in = x.clone()
            snap = None
            if self.kind == "mamba":
                i = self.mod.layer_idx
                # A lazy-state cache (kv.banded_cache.HybridBandedCache) has no state
                # tensors until the first forward; allocate them before the snapshot clone.
                ensure = getattr(cache, "_qv_ensure_mamba_batch", None)
                if ensure is not None:
                    ensure(x.shape[0], x.dtype, x.device)
                # Drop graphs captured against a now-replaced state tensor (same
                # (cache, rows, dtype), different state identity -- the engine's
                # batch grow/drop re-allocates the tensors).
                for k in [k for k in self.graphs if k[:3] == key[:3] and k != key]:
                    del self.graphs[k]
                snap = (cache.conv_states[i].clone(), cache.ssm_states[i].clone())
            call = (lambda t: self.orig(t)) if self.kind == "moe" else (lambda t: self.orig(t, cache_params=cache))
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    call(static_in)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out = call(static_in)
            if snap is not None:
                cache.conv_states[i].copy_(snap[0])
                cache.ssm_states[i].copy_(snap[1])
            self.graphs[key] = (g, static_in, static_out)
            return self.graphs[key]
        except Exception as exc:
            self.disabled = True
            self.graphs.clear()
            torch.cuda.synchronize()
            print(f"[nemotron] graph capture failed ({self.kind}): {type(exc).__name__}: {exc}", flush=True)
            return None

    def __call__(self, hidden_states, *args, cache_params=None, attention_mask=None, **kw):
        eager = lambda: (self.orig(hidden_states) if self.kind == "moe" else
                         self.orig(hidden_states, cache_params=cache_params, attention_mask=attention_mask))
        if self.disabled or not hidden_states.is_cuda or torch.cuda.is_current_stream_capturing():
            return eager()
        rows = hidden_states.shape[0] * hidden_states.shape[1]
        if rows > self.max_rows:
            return eager()
        if self.kind == "mamba":
            traced = getattr(cache_params, "trace", None) is not None
            if cache_params is None or (rows == 1 and traced) or (rows > 1 and not traced):
                return eager()
            i = self.mod.layer_idx
            # The state tensors' identities are part of the key: the engine's batch
            # grow/drop replaces them (torch.cat), and a graph captured against the
            # old addresses would write into dead tensors. Ensure the state exists
            # FIRST -- a lazy-state cache (kv.banded_cache.HybridBandedCache) has
            # None tensors before the first forward, and id(None) would key the
            # graph against nothing.
            ensure = getattr(cache_params, "_qv_ensure_mamba_batch", None)
            if ensure is not None:
                ensure(hidden_states.shape[0], hidden_states.dtype, hidden_states.device)
            key = (id(cache_params), rows, hidden_states.dtype,
                   id(cache_params.conv_states[i]), id(cache_params.ssm_states[i]))
        else:
            key = (tuple(hidden_states.shape), hidden_states.dtype)
        ent = self.graphs.get(key) or self._capture(key, hidden_states, cache_params)
        if ent is None:
            return eager()
        g, sin, sout = ent
        sin.copy_(hidden_states)
        g.replay()
        return sout.clone()


def install_fast_paths(model) -> dict:
    """Fused experts + graph replay for a loaded Nemotron-H skeleton. Anything not eligible stays on the eager path
    (QV_NEMOTRON_FAST=0 disables all of it)."""
    import os
    st = {"moe_fused": 0, "moe_eager": 0, "mamba_graphed": 0}
    if os.environ.get("QV_NEMOTRON_FAST", "1") == "0" or not torch.cuda.is_available():
        return st
    for blk in model.model.layers:
        if blk.block_type == "moe":
            m = blk.mixer
            ups = [e.up_proj for e in m.experts]
            dns = [e.down_proj for e in m.experts]
            u, d = _stack_experts(ups), _stack_experts(dns)
            if u is None or d is None or u["Q"] != d["P"]:
                st["moe_eager"] += 1
                continue
            m._qv_stk = (u, d)
            m.forward = _fused_moe_forward.__get__(m)
            m.forward = _Graphed(m, "moe")
            st["moe_fused"] += 1
        elif blk.block_type == "mamba":
            blk.mixer.forward = _Graphed(blk.mixer, "mamba", _TRACE_MAX)
            st["mamba_graphed"] += 1
    torch.cuda.empty_cache()
    return st


def _patch_logits_to_keep(mh):
    """The shipped CausalLM forward ignores logits_to_keep and runs lm_head over EVERY position -- a
    (seq, vocab) fp32 tensor, ~6 GiB for a 12288-token prefill tail on this 131k-vocab model. Slice to the
    tail before lm_head, exactly as transformers' own CausalLM wrapper does with logits_to_keep."""
    if getattr(mh.NemotronHForCausalLM, "_qv_logits_keep", False):
        return
    orig = mh.NemotronHForCausalLM.forward

    def forward(self, *args, **kwargs):
        k = kwargs.pop("logits_to_keep", None)
        if k is None:
            return orig(self, *args, **kwargs)
        out = self.model(*args, **kwargs)
        if isinstance(out, tuple):
            hidden, past = out[0], out[1]
            hs = out[2] if len(out) > 2 else None
            at = out[3] if len(out) > 3 else None
        else:
            hidden, past = out.last_hidden_state, out.past_key_values
            hs, at = out.hidden_states, out.attentions
        logits = self.lm_head(hidden[:, -k:].to(self.lm_head.weight.dtype)).float()
        return mh.NemotronHCausalLMOutput(loss=None, logits=logits,
                                          past_key_values=past, hidden_states=hs, attentions=at)

    mh.NemotronHForCausalLM.forward = forward
    mh.NemotronHForCausalLM._qv_logits_keep = True


def build_text_skeleton(model_dir: str, config, attn_impl: str):
    """The text-only NemotronHPuzzleForCausalLM of an Omni checkpoint. Call under init_empty_weights()."""
    mod = _import_remote(model_dir)
    text_cfg = config.llm_config
    # The shipped class refuses transformers' own flash_attention_2 dispatch, so FA2 is registered as a custom
    # attention function. QV_ATTN_IMPL=sdpa (or no usable flash-attn) -> torch SDPA.
    impl = "sdpa"
    if attn_impl == "flash_attention_2" and _flash_ok():
        from transformers import AttentionInterface
        AttentionInterface.register("qv_fa2", _flash_attention)
        impl = "qv_fa2"
        base = mod.modeling_nemotron_h.NemotronHModel
        if not getattr(base, "_qv_nomask", False):
            orig = base._update_causal_mask
            base._update_causal_mask = lambda self, am, it, cp: None if self.config._attn_implementation == "qv_fa2" else orig(self, am, it, cp)
            base._qv_nomask = True
    text_cfg._attn_implementation = impl
    print(f">>> nemotron attention: {impl}", flush=True)
    mod.modeling_nemotron_h.NemotronHMoE.moe = _sparse_moe
    mod.modeling_nemotron_h.NemotronHMamba2Mixer.forward = _mamba2_forward
    _patch_logits_to_keep(mod.modeling_nemotron_h)
    return mod.NemotronHPuzzleForCausalLM(text_cfg), text_cfg
