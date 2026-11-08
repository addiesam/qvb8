"""Attention kernels for GPUs the stock fast kernels do not cover.

v100_attention(q, k, v, scale, causal): FlashAttention-style forward on Volta tensor cores (fp16, head dim 128 or 256; optional
causal mask, bottom-right aligned; k / v may have fewer heads (GQA)), q / k / v (B, H, S, D). See v100_attn.cu. `available(q)` says whether a call can take it.
"""
from __future__ import annotations

import os

import torch


def _ext():
    from qv_band.common.jit_ext import load_cuda_ext
    here = os.path.dirname(__file__)
    return load_cuda_ext("qv_v100_attn", [os.path.join(here, "v100_attn.cu")], cuda_cflags=["-O3"])


def available(q: torch.Tensor, k: torch.Tensor | None = None) -> bool:
    if os.environ.get("QV_V100_ATTENTION", "1") == "0":
        return False
    return (q.is_cuda and q.shape[-1] in (128, 256) and torch.cuda.get_device_capability(q.device) == (7, 0)
            and (k is None or k.shape[-1] == q.shape[-1]))


def v100_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None = None,
                   causal: bool = False) -> torch.Tensor:
    """(B, H, S, D) fp16, D in {128, 256} -> (B, H, Sq, D) fp16; softmax(q k^T * scale) v. k / v: (B, Hkv, Sk, D), Hq % Hkv == 0.
    causal: query row i sees keys <= i + (Sk - Sq) (prefill, also with a cache before it)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    with torch.cuda.device(q.device):
        fix = lambda t: t.half() if t.stride(-1) == 1 else t.half().contiguous()      # any layout with a contiguous last dim
        return _ext().forward(fix(q), fix(k), fix(v), float(scale), bool(causal))


MIN_KEYS = int(os.environ.get("QV_V100_ATTN_MIN_KEYS", "2048"))   # below ~1k keys the stock memory-efficient kernel is faster (V100, measured)


def try_bshd(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """diffusers' dispatch_attention_fn layout (B, S, H, D): the V100 kernel's result in the query's dtype when this call
    can take it (V100, head dim 128 / 256, no mask / dropout / causal, >= MIN_KEYS keys), else None (the caller's own path)."""
    if attn_mask is not None or dropout_p or is_causal or key.shape[1] < MIN_KEYS or not available(query, key):
        return None
    try:
        out = v100_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale)   # views: no copy
    except RuntimeError:          # an input the kernel refuses (host-side checks: alignment, grid limit, empty keys)
        return None
    if _NAN_GUARD and not bool(torch.isfinite(out).all()):      # one reduction pass + one sync, ~0.05 ms at Z-Image size
        # a non-finite output (one unexplained black-image run): recompute this call on the stock path, warn once
        global _guard_hits
        _guard_hits += 1
        if _guard_hits == 1:
            import warnings
            warnings.warn("qv_band v100_attention produced non-finite output; falling back to SDPA for that call")
        return None
    return out.transpose(1, 2).to(query.dtype)


_NAN_GUARD = os.environ.get("QV_V100_ATTN_NAN_GUARD", "1") != "0"
_guard_hits = 0


PREFILL_MIN_ROWS = int(os.environ.get("QV_V100_PREFILL_MIN_ROWS", "512"))


def try_causal_bhsd(query, key, value, scale=None):
    """LLM prefill, transformers' layout (B, H, S, D), causal, k / v possibly GQA: the V100 kernel's result (B, H, Sq, D) in
    the query's dtype, or None (not a V100 / short prompt / refused / non-finite) so the caller keeps its own path.
    bf16 inputs are narrowed to fp16 (sm_70 has no bf16 tensor cores); q is pre-scaled in fp32 and saturated so an
    outlier channel cannot overflow fp16."""
    if (query.shape[2] < PREFILL_MIN_ROWS or not available(query, key) or query.shape[1] % key.shape[1]
            or query.shape[2] > key.shape[2]):
        return None
    if scale is None:
        scale = query.shape[-1] ** -0.5
    q = (query.float() * scale).clamp_(-65504.0, 65504.0).half() if query.dtype != torch.float16 else query
    s = 1.0 if q is not query else scale
    k = key.clamp(-65504.0, 65504.0).half() if key.dtype != torch.float16 else key
    v = value.clamp(-65504.0, 65504.0).half() if value.dtype != torch.float16 else value
    try:
        out = v100_attention(q, k, v, s, causal=True)
    except RuntimeError:
        return None
    if _NAN_GUARD and not bool(torch.isfinite(out).all()):
        return None
    return out.to(query.dtype)


_LLM_INSTALLED = False


def install_llm_prefill() -> bool:
    """Route transformers' "sdpa" causal prefill calls (no mask, >= PREFILL_MIN_ROWS rows, head dim 128 / 256, V100) through the
    V100 kernel; everything else (decode, masks, sliding window, position bias) keeps the stock function. Call BEFORE
    kv.fused_attn.install so the fused-KV wrapper stays outermost and still sees its own calls first. Idempotent."""
    global _LLM_INSTALLED
    if _LLM_INSTALLED or os.environ.get("QV_V100_PREFILL", "1") == "0":
        return _LLM_INSTALLED
    if not torch.cuda.is_available() or (7, 0) not in {torch.cuda.get_device_capability(i)
                                                       for i in range(torch.cuda.device_count())}:
        return False
    from transformers.modeling_utils import AttentionInterface
    base = AttentionInterface._global_mapping.get("sdpa")
    if base is None:
        return False

    def sdpa_v100(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
        causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        if (causal and attention_mask is None and not dropout and kw.get("position_bias") is None
                and not kw.get("output_attentions") and not kw.get("sliding_window")
                and query.shape[2] >= PREFILL_MIN_ROWS):
            k, v = key, value
            if k.shape[2] > query.shape[2]:            # transformers: no mask + causal = top-left aligned (empty static cache)
                k, v = k[:, :, :query.shape[2]], v[:, :, :query.shape[2]]
            out = try_causal_bhsd(query, k, v, scaling)
            if out is not None:
                return out.transpose(1, 2).contiguous(), None
        return base(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kw)

    AttentionInterface.register("sdpa", sdpa_v100)
    _LLM_INSTALLED = True
    return True
