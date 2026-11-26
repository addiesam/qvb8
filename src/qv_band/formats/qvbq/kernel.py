"""On-the-fly matmul over low-precision weight codes (NVFP4 / FP8 families).

Each weight is decoded inside the matmul from its own formula -- no
min_s + idx*tau_s band, no materialised dense copy:

    w[e, n, k] = CODE(q[e, n, k]) * SCALE(s[e, n // GN, k // GK]) * f[e, n]

    CODE   0  E2M1, two codes per byte, even k in the low nibble (NVFP4)
           1  E4M3 byte                                          (FP8)
           2  E5M2 byte                                          (FP8)
    SCALE  0  e4m3 byte        (NVFP4 per-16 block scale)
           1  E8M0 byte, 2^(s-127)   (MX per-32 scale)
           2  fp32             (FP8 block 128x128 scale_inv, per-channel, per-tensor)
    f      optional fp32 per-row factor (NVFP4 global scale, per output row so
           concatenated gate/up rows with different globals stay exact)

Rows of the matmul are token-expert pairs sorted by expert and padded per
expert to BLOCK_M (moe_align). A plain linear is the E=1 case.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

CODE_E2M1, CODE_E4M3, CODE_E5M2 = 0, 1, 2
SCALE_E4M3, SCALE_E8M0, SCALE_F32 = 0, 1, 2


@triton.jit
def _e2m1(c):
    e = (c >> 1) & 3
    m = (c & 1).to(tl.float32)
    v = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where((c & 8) != 0, -v, v)


@triton.jit
def _fp8(b, M: tl.constexpr, EMASK: tl.constexpr, MMASK: tl.constexpr, MSCALE: tl.constexpr,
         BIAS: tl.constexpr, SUB: tl.constexpr):
    """An FP8 byte (sign | exponent | M mantissa bits) by its formula: the cards without FP8 conversion (sm < 89).
    normal (1 + m * MSCALE) * 2^(e - BIAS), subnormal m * MSCALE * SUB (SUB = 2^(1 - BIAS))."""
    b = b.to(tl.int32)
    e = (b >> M) & EMASK
    m = (b & MMASK).to(tl.float32) * MSCALE
    v = tl.where(e == 0, m * SUB, (1.0 + m) * tl.exp2((e - BIAS).to(tl.float32)))
    return tl.where((b & 0x80) != 0, -v, v)


@triton.jit
def _e4m3(b):
    return _fp8(b, 3, 15, 7, 0.125, 7, 0.015625)


@triton.jit
def _e5m2(b):
    return _fp8(b, 2, 31, 3, 0.25, 15, 0.00006103515625)


_FP8_HW: dict = {}


def _fp8_hw(device) -> bool:
    i = device.index if device.index is not None else torch.cuda.current_device()
    if i not in _FP8_HW:
        _FP8_HW[i] = torch.cuda.get_device_capability(i) >= (8, 9)
    return _FP8_HW[i]


@triton.jit
def _qvbq_kernel(x_ptr, q_ptr, s_ptr, f_ptr, y_ptr, w_ptr,
                sorted_ptr, expert_ptr, n_valid,
                N, K, SN, SK, stride_x, stride_y,
                TOPK: tl.constexpr, CODE: tl.constexpr, SCALE: tl.constexpr,
                GN: tl.constexpr, GK: tl.constexpr,
                HAS_F: tl.constexpr, MUL_W: tl.constexpr, FP8_HW: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(expert_ptr + pid_m)
    if e < 0:
        return
    e = e.to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair = tl.load(sorted_ptr + rm)
    valid = pair < n_valid
    tok = pair // TOPK
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    s_base = s_ptr + e * SN * SK + (rn // GN)[:, None].to(tl.int64) * SK
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        kmask = rk < K
        a = tl.load(x_ptr + tok[:, None].to(tl.int64) * stride_x + rk[None, :],
                    mask=valid[:, None] & kmask[None, :], other=0.0)
        if CODE == 0:
            rkh = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            b = tl.load(q_ptr + e * N * (K // 2) + rn[:, None].to(tl.int64) * (K // 2) + rkh[None, :],
                        mask=nmask[:, None] & (rkh < K // 2)[None, :], other=0)
            v = tl.reshape(tl.join(_e2m1(b & 0xF), _e2m1(b >> 4)), (BLOCK_N, BLOCK_K))
        else:
            b = tl.load(q_ptr + e * N * K + rn[:, None].to(tl.int64) * K + rk[None, :],
                        mask=nmask[:, None] & kmask[None, :], other=0)
            if FP8_HW:
                if CODE == 1:
                    v = b.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                else:
                    v = b.to(tl.float8e5, bitcast=True).to(tl.float32)
            else:
                if CODE == 1:
                    v = _e4m3(b)
                else:
                    v = _e5m2(b)
        # scale gathered per element (L1-resident: GK consecutive k share one value)
        sm = nmask[:, None] & kmask[None, :]
        if SCALE == 2:
            sc = tl.load(s_base + (rk // GK)[None, :], mask=sm, other=0.0)
        else:
            sb = tl.load(s_base + (rk // GK)[None, :], mask=sm, other=0)
            if SCALE == 0:
                if FP8_HW:
                    sc = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                else:
                    sc = _e4m3(sb)
            else:
                sc = tl.exp2(sb.to(tl.float32) - 127.0)
        v = v * sc
        if HAS_F:
            fr = tl.load(f_ptr + e * N + rn, mask=nmask, other=0.0)
            v = v * fr[:, None]
        acc += tl.dot(a, tl.trans(v.to(x_ptr.dtype.element_ty)))      # bf16, or fp16 on sm_70
    if MUL_W:
        wt = tl.load(w_ptr + pair, mask=valid, other=0.0)
        acc = acc * wt[:, None]
    tl.store(y_ptr + pair[:, None].to(tl.int64) * stride_y + rn[None, :],
             acc.to(y_ptr.dtype.element_ty), mask=valid[:, None] & nmask[None, :])


def moe_align(topk_ids: torch.Tensor, num_experts: int, block_m: int):
    """Pairs sorted by expert, each expert's run padded to block_m. Host-sync free:
    the block count is the static upper bound; unused blocks have expert -1."""
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    dev = flat.device
    order = torch.argsort(flat, stable=True)
    # scatter_add, not bincount: CUDA bincount reads the max on the host (a sync, not CUDA-graph capturable)
    cnt = torch.zeros(num_experts, dtype=torch.long, device=dev).scatter_add_(0, flat.long(), torch.ones_like(flat, dtype=torch.long))
    padded = (cnt + block_m - 1) // block_m * block_m
    max_blocks = (n + num_experts * (block_m - 1)) // block_m
    starts = torch.cumsum(padded, 0) - padded
    fe = flat[order]
    rank = torch.arange(n, device=dev) - (torch.cumsum(cnt, 0) - cnt)[fe]
    sorted_ids = torch.full((max_blocks * block_m,), n, dtype=torch.int32, device=dev)
    sorted_ids[starts[fe] + rank] = order.to(torch.int32)
    # expert of each block: block b belongs to expert e iff starts[e] <= b*block_m < starts[e]+padded[e]
    bstart = torch.arange(max_blocks, device=dev) * block_m
    ends = starts + padded
    be = torch.searchsorted(ends, bstart, right=True)
    ok = (be < num_experts)
    be = torch.where(ok, be, torch.zeros_like(be))
    ok = ok & (bstart >= starts[be]) & (padded[be] > 0)
    blk_expert = torch.where(ok, be, torch.full_like(be, -1)).to(torch.int32)
    return sorted_ids, blk_expert


def qvbq_matmul(x, w, sorted_ids, blk_expert, n_pairs, topk, *, weights=None,
               x_is_pairs=False, block_m=16, out_dtype=torch.bfloat16):
    """y[pair, N] = x[row] @ W[e].T with W decoded on the fly; `w` is an QVBQWeight. x bf16 or fp16 (sm_70: no bf16
    tensor cores): the decoded weight is rounded once to x's dtype, the sums are fp32."""
    if x.device.type == "cuda" and torch.cuda.get_device_capability(x.device) < (8, 0):
        from .kernel_v100 import pair_experts, small_matmul       # V100: the dot path is ~500x off the memory time
        return small_matmul(x, w, pair_experts(sorted_ids, blk_expert, n_pairs, block_m), n_pairs, topk,
                            weights=weights, x_is_pairs=x_is_pairs, out_dtype=out_dtype)
    y = torch.empty(n_pairs, w.N, dtype=out_dtype, device=x.device)
    BN, BK = 64, 128
    grid = (blk_expert.numel(), triton.cdiv(w.N, BN))
    _qvbq_kernel[grid](x, w.q, w.s, w.f if w.f is not None else w.s, y,
                      weights if weights is not None else y,
                      sorted_ids, blk_expert, n_pairs, w.N, w.K, w.SN, w.SK,
                      x.stride(0), y.stride(0),
                      TOPK=1 if x_is_pairs else topk, CODE=w.code, SCALE=w.scale_kind,
                      GN=w.GN, GK=w.GK, HAS_F=w.f is not None, MUL_W=weights is not None,
                      FP8_HW=_fp8_hw(x.device),
                      BLOCK_M=block_m, BLOCK_N=BN, BLOCK_K=BK, num_warps=4)
    return y
