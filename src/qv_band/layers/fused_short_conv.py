"""Layer kind: fused one-token causal short conv (Triton) -- depthwise conv over the last K-1 inputs + the current one,
(conv + x) / sqrt(K + 1), AND the history push (drop the oldest row, append x where the slot is active): ONE launch.
Written for Inkling-2 static decode (models/inkling/model.py short_conv1: 176 calls per step).

Why: the torch version is 10 kernels per call (cat, transpose-mul, sum, add, div, cast + conv_push's cat / where /
copy); inside the CUDA graph each costs ~1.5 us of GPU time whatever its size. Measured on the dev GPU [4, 2304] bf16:
13.8 us per call.

Numerics: bit-identical to the torch ops (bf16 and fp16): fp32 products and the K-term sum in torch's order
((p0 + p1) + p2) + ..., rounded ops (libdevice mul_rn / add_rn: no FMA contraction), and the division as torch does a
division by a scalar: multiply by the fp32 reciprocal.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _short_conv1_kernel(x_ptr, st_ptr, w_ptr, act_ptr, out_ptr, C, inv, K: tl.constexpr, BC: tl.constexpr):
    s = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < C
    xs = tl.load(x_ptr + s * C + c, mask=m).to(tl.float32)
    row = st_ptr + s * (K - 1) * C + c
    acc = libdevice.mul_rn(tl.load(row, mask=m), tl.load(w_ptr + c * K, mask=m))
    for j in tl.static_range(1, K - 1):
        acc = libdevice.add_rn(acc, libdevice.mul_rn(tl.load(row + j * C, mask=m), tl.load(w_ptr + c * K + j, mask=m)))
    acc = libdevice.add_rn(acc, libdevice.mul_rn(xs, tl.load(w_ptr + c * K + K - 1, mask=m)))
    tl.store(out_ptr + s * C + c, libdevice.mul_rn(libdevice.add_rn(acc, xs), inv).to(out_ptr.dtype.element_ty), mask=m)
    if tl.load(act_ptr + s) != 0:                           # history: rows 1.. move up one, x becomes the newest
        for j in tl.static_range(0, K - 2):
            tl.store(row + j * C, tl.load(row + (j + 1) * C, mask=m), mask=m)
        tl.store(row + (K - 2) * C, xs, mask=m)


def short_conv1(x: torch.Tensor, state: torch.Tensor, w: torch.Tensor, K: int, active: torch.Tensor) -> torch.Tensor:
    """x [S, C] (bf16 / fp16), state [S, K-1, C] fp32 (oldest first; updated in place where active), w [C, K] fp32,
    active [S] bool -> (conv + x) / sqrt(K + 1) [S, C] in x.dtype."""
    S, C = x.shape
    out = torch.empty_like(x)
    inv = float(np.float32(1.0) / np.float32(math.sqrt(K + 1)))
    with torch.cuda.device(x.device):
        _short_conv1_kernel[(S, triton.cdiv(C, 256))](x.contiguous(), state, w, active, out, C, inv, K=K, BC=256,
                                                      num_warps=2)
    return out



@triton.jit
def _short_conv_r_kernel(x_ptr, st_ptr, w_ptr, out_ptr, sc_ptr, slot_ptr, j_ptr, start_ptr, R, TS, C, inv,
                         K: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0)
    s = tl.load(slot_ptr + r)
    j = tl.load(j_ptr + r)
    r0 = tl.load(start_ptr + s)                       # the slot's first row
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < C
    acc = tl.zeros((BC,), tl.float32)
    for i in tl.static_range(K):                      # input j+i of [history (K-1) | the slot's rows]
        jj = j + i
        if jj < K - 1:
            xv = tl.load(st_ptr + (s * (K - 1) + jj) * C + c, mask=m, other=0.0)
        else:
            row = tl.minimum(r0 + jj - (K - 1), R - 1)
            xv = tl.load(x_ptr + row * C + c, mask=m, other=0.0).to(tl.float32)
        acc += xv * tl.load(w_ptr + c * K + i, mask=m, other=0.0)
    xs = tl.load(x_ptr + r * C + c, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + r * C + c, ((acc + xs) * inv).to(out_ptr.dtype.element_ty), mask=m)
    tl.store(sc_ptr + (s * TS + j) * C + c, xs, mask=m)


def short_conv_r(x, state, w, K: int, scratch, slot, j, start) -> torch.Tensor:
    """Ragged rows of a speculative verify: row r = token j[r] of slot slot[r] (a slot's rows contiguous from start[s]).
    x [R, C], state [S, K-1, C] fp32 (NOT changed), w [C, K] fp32 -> (conv + x) / sqrt(K + 1) [R, C] in x.dtype;
    x (fp32) goes to scratch [S, Ts, C] at (slot, j) for engine/static_decode.conv_commit."""
    R, C = x.shape
    out = torch.empty_like(x)
    inv = float(np.float32(1.0) / np.float32(math.sqrt(K + 1)))
    with torch.cuda.device(x.device):
        _short_conv_r_kernel[(R, triton.cdiv(C, 256))](x.contiguous(), state, w, out, scratch, slot, j, start, R,
                                                       scratch.shape[1], C, inv, K=K, BC=256, num_warps=2)
    return out
