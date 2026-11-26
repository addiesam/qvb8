"""Fused Gated-DeltaNet recurrence for the graph decoder (DFlash verify and decode), framework level.

The eager version runs, per token and layer, ~10 small kernels (l2 norms, decay, read-out of k, delta, rank-1
update, certified rounding, history copy, read-out of q) and moves the fp32 state (heads x d_k x d_v) through HBM
several times per token. Here ONE kernel does all T tokens of a layer: each program owns one head and a block of
d_v columns, keeps that slice of the state in registers for the whole block, and writes only what is needed:
the state after every token (for exact speculative rollback) and the outputs.

Same math, same order per token as the eager loop, and row-deterministic (a token's result does not depend on how
many tokens are in the block), so speculative output == plain output still holds:
    q, k <- l2(q) * d_k^-0.5, l2(k)            (per token, fp32)
    S    <- S * exp(g)
    d    <- (v - k S) * beta
    S    <- S + k (x) d
    S    <- min_col + round((S - min_col) / (2 tau)) * 2 tau      (certified rounding, when tau > 0)
    o    <- q S
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

GDN_BV = int(os.environ.get("QV_GDN_BV", "16"))      # d_v columns per program (more programs = more SMs busy)
GDN_WARPS = int(os.environ.get("QV_GDN_WARPS", "4"))


@triton.jit
def _round_half_even(x):
    # round half to even, as torch.round (ties are measure-zero in practice; kept for bit-compatibility)
    f = tl.floor(x)
    r = x - f
    up = (r > 0.5) | ((r == 0.5) & ((f - 2.0 * tl.floor(f * 0.5)) != 0.0))
    return tl.where(up, f + 1.0, f)


@triton.jit(do_not_specialize=["T"])       # T=1 must run the same code as T=8 (row determinism)
def _gdn_fused(Q, K, V, G, B, S0, HIST, O, T, H,
               scale, step, eps,
               DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, ROUND: tl.constexpr):
    h = tl.program_id(0)
    vb = tl.program_id(1)
    dk = tl.arange(0, DK)
    dv = vb * BV + tl.arange(0, BV)
    sl = h * DK * DV + dk[:, None] * DV + dv[None, :]
    S = tl.load(S0 + sl)
    for t in range(T):
        row = t * H + h
        q = tl.load(Q + row * DK + dk)
        k = tl.load(K + row * DK + dk)
        q = q * (1.0 / tl.sqrt(tl.sum(q * q, axis=0) + eps)) * scale
        k = k * (1.0 / tl.sqrt(tl.sum(k * k, axis=0) + eps))
        v = tl.load(V + row * DV + dv)
        g = tl.load(G + row)
        b = tl.load(B + row)
        S = S * tl.exp(g)
        kv = tl.sum(S * k[:, None], axis=0)
        d = (v - kv) * b
        S = S + k[:, None] * d[None, :]
        if ROUND:
            mn = tl.min(S, axis=0)
            S = mn[None, :] + _round_half_even((S - mn[None, :]) / step) * step
        tl.store(HIST + t * H * DK * DV + sl, S)
        o = tl.sum(S * q[:, None], axis=0)
        tl.store(O + row * DV + dv, o)


def gdn_fused(q, k, v, g, beta, S, hist, tau=None, eps=1e-6):
    """q, k [T, H, dk]; v [T, H, dv]; g, beta [T, H] (fp32, contiguous); S [H, dk, dv] fp32 state before the block;
    hist [>= T, H, dk, dv] fp32 receives the state after each token. Returns o [T, H, dv] fp32."""
    T, H, DK = q.shape
    DV = v.shape[-1]
    BV = GDN_BV if DV % GDN_BV == 0 else DV
    o = torch.empty(T, H, DV, dtype=torch.float32, device=q.device)
    _gdn_fused[(H, DV // BV)](q, k, v, g, beta, S, hist, o, T, H,
                              DK ** -0.5, 2.0 * tau if tau else 1.0, eps,
                              DK=DK, DV=DV, BV=BV, ROUND=bool(tau), num_warps=GDN_WARPS,
                              enable_fp_fusion=False)   # min + j*step as mul then add: equals the torch decode of
    #                                                     the stored state, so a token after a commit sees the same state
    return o
