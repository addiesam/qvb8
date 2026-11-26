"""qvbq on cards without FP8 / FP4 hardware and without a fast Triton dot (sm_70, V100).

The general kernel (kernel.py) keeps a 16 x BLOCK_N x 128 tl.dot and a per-element scale gather: on a V100 one launch for
one token took ~10 ms (500x the memory time). Two paths here instead:

  small  (few token-expert pairs: decode)   one program per (pair, BLOCK_N rows): decode the weight tile, multiply by the
         pair's x row, reduce over k with fp32 sums. No dot, one scale load per GK elements (not per element).
  dequant (many pairs: prefill)             decode ONE expert to a dense fp16 [N, K] (memory-bound elementwise kernel),
         then torch.mm on the tensor cores for that expert's rows.

Same numbers as kernel.py: the decoded weight is rounded once to x's dtype (fp16), sums are fp32.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .kernel import _e2m1, _e4m3, _e5m2


@triton.jit
def _decode(b, s, CODE: tl.constexpr, SCALE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GK: tl.constexpr):
    """Codes b (uint8 tile: [BN, BK//2] for E2M1 else [BN, BK]) times the group scales s ([BN, BK//GK]) -> [BN, BK] fp32."""
    if CODE == 0:
        v = tl.reshape(tl.join(_e2m1(b & 0xF), _e2m1(b >> 4)), (BLOCK_N, BLOCK_K))
    elif CODE == 1:
        v = _e4m3(b)
    else:
        v = _e5m2(b)
    if SCALE == 2:
        sc = s
    elif SCALE == 0:
        sc = _e4m3(s)
    else:
        sc = tl.exp2(s.to(tl.float32) - 127.0)
    NG: tl.constexpr = BLOCK_K // GK
    v = tl.reshape(v, (BLOCK_N, NG, GK)) * sc[:, :, None]
    return tl.reshape(v, (BLOCK_N, BLOCK_K))


@triton.jit
def _small_kernel(x_ptr, q_ptr, s_ptr, f_ptr, y_ptr, w_ptr, pe_ptr, N, K, SN, SK, stride_x, stride_y,
                  TOPK: tl.constexpr, CODE: tl.constexpr, SCALE: tl.constexpr, GN: tl.constexpr, GK: tl.constexpr,
                  HAS_F: tl.constexpr, MUL_W: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    p = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(pe_ptr + p).to(tl.int64)
    if e < 0:
        return
    tok = (p // TOPK).to(tl.int64)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    NG: tl.constexpr = BLOCK_K // GK
    CB: tl.constexpr = BLOCK_K // 2 if CODE == 0 else BLOCK_K
    s_row = s_ptr + e * SN * SK + (rn // GN).to(tl.int64) * SK
    q_row = q_ptr + (e * N + rn.to(tl.int64)) * (K // 2 if CODE == 0 else K)
    for k0 in range(0, K, BLOCK_K):
        ck = (k0 // 2 if CODE == 0 else k0) + tl.arange(0, CB)
        b = tl.load(q_row[:, None] + ck[None, :], mask=nmask[:, None], other=0)
        gk = k0 // GK + tl.arange(0, NG)
        s = tl.load(s_row[:, None] + gk[None, :], mask=nmask[:, None] & (gk < SK)[None, :], other=0)
        v = _decode(b, s, CODE, SCALE, BLOCK_N, BLOCK_K, GK)
        rk = k0 + tl.arange(0, BLOCK_K)
        xv = tl.load(x_ptr + tok * stride_x + rk, mask=rk < K, other=0.0)
        v = v.to(x_ptr.dtype.element_ty).to(tl.float32)          # the weight rounded once to x's dtype (as the dot path)
        acc += tl.sum(v * xv.to(tl.float32)[None, :], axis=1)
    if HAS_F:
        acc = acc * tl.load(f_ptr + e * N + rn, mask=nmask, other=0.0)
    if MUL_W:
        acc = acc * tl.load(w_ptr + p)
    tl.store(y_ptr + p.to(tl.int64) * stride_y + rn, acc.to(y_ptr.dtype.element_ty), mask=nmask)


@triton.jit
def _dequant_kernel(q_ptr, s_ptr, f_ptr, o_ptr, e, N, K, SN, SK,
                    CODE: tl.constexpr, SCALE: tl.constexpr, GN: tl.constexpr, GK: tl.constexpr, HAS_F: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    k0 = pk * BLOCK_K
    NG: tl.constexpr = BLOCK_K // GK
    CB: tl.constexpr = BLOCK_K // 2 if CODE == 0 else BLOCK_K
    ck = (k0 // 2 if CODE == 0 else k0) + tl.arange(0, CB)
    q_row = q_ptr + (e * N + rn.to(tl.int64)) * (K // 2 if CODE == 0 else K)
    b = tl.load(q_row[:, None] + ck[None, :], mask=nmask[:, None], other=0)
    gk = k0 // GK + tl.arange(0, NG)
    s_row = s_ptr + e * SN * SK + (rn // GN).to(tl.int64) * SK
    s = tl.load(s_row[:, None] + gk[None, :], mask=nmask[:, None] & (gk < SK)[None, :], other=0)
    v = _decode(b, s, CODE, SCALE, BLOCK_N, BLOCK_K, GK)
    if HAS_F:
        v = v * tl.load(f_ptr + e * N + rn, mask=nmask, other=0.0)[:, None]
    rk = k0 + tl.arange(0, BLOCK_K)
    tl.store(o_ptr + rn.to(tl.int64)[:, None] * K + rk[None, :], v.to(o_ptr.dtype.element_ty),
             mask=nmask[:, None] & (rk < K)[None, :])


import os
_BN = int(os.environ.get("QV_V100_BN", "32"))
_BKM = int(os.environ.get("QV_V100_BK", "128"))
_NW = int(os.environ.get("QV_V100_NW", "4"))


def _bk(w) -> int:
    return max(_BKM, w.GK)


def small_matmul(x, w, pair_expert, n_pairs, topk, *, weights=None, x_is_pairs=False, out_dtype=torch.float32):
    """y[pair, N] = x[row(pair)] @ W[pair_expert[pair]].T  (pair_expert < 0: the pair's row stays 0)."""
    y = torch.zeros(n_pairs, w.N, dtype=out_dtype, device=x.device)
    BN = _BN
    grid = (n_pairs, triton.cdiv(w.N, BN))
    _small_kernel[grid](x, w.q, w.s, w.f if w.f is not None else w.s, y, weights if weights is not None else y, pair_expert,
                        w.N, w.K, w.SN, w.SK, x.stride(0), y.stride(0),
                        TOPK=1 if x_is_pairs else topk, CODE=w.code, SCALE=w.scale_kind, GN=w.GN, GK=w.GK,
                        HAS_F=w.f is not None, MUL_W=weights is not None, BLOCK_N=BN, BLOCK_K=_bk(w), num_warps=_NW)
    return y


def dequant_expert(w, e: int, out: torch.Tensor) -> torch.Tensor:
    """Expert e of w as a dense [N, K] in out's dtype (out: preallocated)."""
    BN = 32
    grid = (triton.cdiv(w.N, BN), triton.cdiv(w.K, _bk(w)))
    _dequant_kernel[grid](w.q, w.s, w.f if w.f is not None else w.s, out, e, w.N, w.K, w.SN, w.SK,
                          CODE=w.code, SCALE=w.scale_kind, GN=w.GN, GK=w.GK, HAS_F=w.f is not None,
                          BLOCK_N=BN, BLOCK_K=_bk(w), num_warps=4)
    return out


def pair_experts(sorted_ids, blk_expert, n_pairs: int, block_m: int = 16) -> torch.Tensor:
    """Expert of every pair from moe_align's layout (no host sync): -1 where the pair is not routed here."""
    pe = torch.full((n_pairs + 1,), -1, dtype=torch.int32, device=sorted_ids.device)
    pe.scatter_(0, sorted_ids.long().clamp(max=n_pairs), blk_expert.repeat_interleave(block_m))
    return pe[:n_pairs].contiguous()
