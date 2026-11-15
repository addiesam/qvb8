"""Gated DeltaNet prefill kernels for sm_70 (V100).

fla's chunk_gated_delta_rule_fwd_h / chunk_fwd_o are Triton kernels whose
tl.dot has no tensor-core path on Volta (and spill heavily): measured 48% of
a 7k-token prefill on 2x V100. cuda_attn/qvattn_cuda.cu holds WMMA versions
with the same math and stored dtypes; install() patches them into fla's
gated_delta_rule forward for the shapes they cover (K = V = 128, chunk 64,
scalar gate, no varlen / gk / state_v_first) and leaves every other call on
fla. QV_GDN_CUDA=0 disables.
"""
from __future__ import annotations

import os

import torch

_DONE = False


def _ext():
    from .fused_attn import _cuda_ext
    return _cuda_ext(torch.cuda.current_device())


def install() -> bool:
    global _DONE
    if _DONE or os.environ.get("QV_GDN_CUDA", "1") == "0" or not torch.cuda.is_available():
        return _DONE
    if torch.cuda.get_device_capability()[0] >= 8 and os.environ.get("QV_GDN_CUDA") != "1":
        return False
    try:
        import fla.ops.gated_delta_rule.chunk as C
    except ImportError:
        return False
    ext = _ext()
    if ext is None or not hasattr(ext, "gdn_h"):
        return False
    orig_h, orig_o = C.chunk_gated_delta_rule_fwd_h, C.chunk_fwd_o

    def fwd_h(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False,
              chunk_size=64, save_new_value=True, state_v_first=False, cu_seqlens=None,
              cu_seqlens_cpu=None, chunk_indices=None):
        ok = (g is not None and gk is None and cu_seqlens is None and not state_v_first
              and save_new_value and chunk_size == 64 and k.shape[-1] == 128
              and u.shape[-1] == 128 and k.dtype in (torch.float16, torch.bfloat16)
              and w.dtype == k.dtype and u.dtype == k.dtype and g.dtype == torch.float32
              and (initial_state is None or initial_state.dtype == torch.float32))
        if not ok:
            return orig_h(k, w, u, g=g, gk=gk, initial_state=initial_state,
                          output_final_state=output_final_state, chunk_size=chunk_size,
                          save_new_value=save_new_value, state_v_first=state_v_first,
                          cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
                          chunk_indices=chunk_indices)
        B, T, H, K = k.shape
        HV, V = u.shape[2], u.shape[3]
        NT = (T + 63) // 64
        h = k.new_empty(B, NT, HV, K, V)
        v_new = torch.empty_like(u)
        final = k.new_zeros(B, HV, K, V, dtype=torch.float32) if output_final_state else None
        h0 = initial_state.contiguous() if initial_state is not None else None
        _ext().gdn_h(k.contiguous(), w.contiguous(), u.contiguous(), g.contiguous(),
                     h, v_new, h0, final)
        return h, v_new, final

    def fwd_o(q, k, v, h, g=None, g_gamma=None, scale=None, state_v_first=False,
              cu_seqlens=None, chunk_size=64, chunk_indices=None):
        ok = (g is not None and g_gamma is None and cu_seqlens is None and not state_v_first
              and chunk_size == 64 and q.shape[-1] == 128 and v.shape[-1] == 128
              and q.dtype in (torch.float16, torch.bfloat16) and k.dtype == q.dtype
              and v.dtype == q.dtype and h.dtype == q.dtype and g.dtype == torch.float32)
        if not ok:
            return orig_o(q, k, v, h, g=g, g_gamma=g_gamma, scale=scale,
                          state_v_first=state_v_first, cu_seqlens=cu_seqlens,
                          chunk_size=chunk_size, chunk_indices=chunk_indices)
        if scale is None:
            scale = k.shape[-1] ** -0.5
        o = torch.empty_like(v)
        _ext().gdn_o(q.contiguous(), k.contiguous(), v.contiguous(), h.contiguous(),
                     g.contiguous(), o, float(scale))
        return o

    C.chunk_gated_delta_rule_fwd_h = fwd_h
    C.chunk_fwd_o = fwd_o
    C._qv_orig = (orig_h, orig_o)
    _DONE = True
    print("[gdn] Gated DeltaNet prefill: WMMA kernels for sm_70 installed", flush=True)
    return True
