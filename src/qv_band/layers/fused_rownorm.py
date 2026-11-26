"""Layer kind: fused per-row norms of the small-vector paths (Triton), ONE launch per call, fp32 inside:
  rms(x, w, eps, out)                RMS norm of each row (optional weight)
  l2clip(x, thr, scale, pos, ...)    x * scale, clipped to L2 norm thr; optional log position scaling
  block_residual(q, blocks, part)    attention over the block axis (unweighted RMS-normed states, softmax, weighted sum)
  gate_in(h, m, eps, out)            [rms(h) | rms(m)] (the message gate's input)
  gate_norm(fg, h, m, w, eps, out)   rms((1 - tanh(f)) * h + tanh(g) * m) * w  (the gate's output + the next norm)
Written for Inkling-2 (models/inkling/model.py). Why: each torch version is 8-24 kernels on rows of 2304 (or 128); in a
CUDA-graph decode step every kernel costs ~1.5 us of GPU time, ~13 ms of a 66 ms the dev GPU step.
Numerics: same maths, own reduction order (not bit-identical to torch; quality checked by teacher-forced KL).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _blk(n):
    return triton.next_power_of_2(n)


@triton.jit
def _rms_kernel(x_ptr, w_ptr, o_ptr, N, eps, HAS_W: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, B)
    m = c < N
    x = tl.load(x_ptr + r * N + c, mask=m, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / N + eps)
    if HAS_W:
        y = y * tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + r * N + c, y.to(o_ptr.dtype.element_ty), mask=m)


def rms(x, w=None, eps=1e-6, out=None):
    """RMS norm over the last dim; out dtype = `out` (default x.dtype)."""
    N = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    o = torch.empty(x2.shape, dtype=out or x.dtype, device=x.device)
    if x2.shape[0]:
        with torch.cuda.device(x.device):
            _rms_kernel[(x2.shape[0],)](x2, w if w is not None else x2, o, N, eps, HAS_W=w is not None, B=_blk(N),
                                        num_warps=4 if N > 512 else 1)
    return o.view(x.shape)


@triton.jit
def _l2clip_kernel(x_ptr, s_ptr, pos_ptr, o_ptr, N, RS, PR, thr, alpha, n_floor,
                   HAS_S: tl.constexpr, LOG: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, B)
    m = c < N
    x = tl.load(x_ptr + r * N + c, mask=m, other=0.0).to(tl.float32)
    if HAS_S:                                             # scale rows repeat every RS rows ([H, d] or [d])
        x = x * tl.load(s_ptr + (r % RS) * N + c, mask=m, other=0.0).to(tl.float32)
    n = tl.sqrt(tl.sum(x * x, 0))
    y = x * tl.minimum(thr / (n + 1e-6), 1.0)
    if LOG:                                               # 1 + alpha * log(max((pos + 1) / n_floor, 1)), pos per token
        p = tl.load(pos_ptr + r // PR).to(tl.float32)
        y = y * (1.0 + alpha * tl.log(tl.maximum((p + 1.0) / n_floor, 1.0)))
    tl.store(o_ptr + r * N + c, y.to(o_ptr.dtype.element_ty), mask=m)


def l2clip(x, thr, scale=None, pos=None, alpha=0.0, n_floor=1.0):
    """x [..., d] (rows), scale broadcast over the leading dims or None; pos [T] with x [T, h, d]: log scaling."""
    N = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    o = torch.empty_like(x2)
    s = None if scale is None else scale.reshape(-1, N).contiguous()
    pr = x2.shape[0] // x.shape[0] if pos is not None else 1
    if x2.shape[0]:
        with torch.cuda.device(x.device):
            _l2clip_kernel[(x2.shape[0],)](x2, s if s is not None else x2, pos if pos is not None else x2, o, N,
                                           s.shape[0] if s is not None else 1, pr, float(thr), float(alpha),
                                           float(n_floor), HAS_S=s is not None, LOG=pos is not None, B=_blk(N),
                                           num_warps=1)
    return o.view(x.shape)


@triton.jit
def _block_res_kernel(q_ptr, b_ptr, p_ptr, o_ptr, NB, NQ, N, scale, eps, NBP: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    qi = tl.program_id(1)
    c = tl.arange(0, B)
    m = c < N
    k = tl.arange(0, NBP)
    q = tl.load(q_ptr + qi * N + c, mask=m, other=0.0).to(tl.float32)
    # pass 1: per state 1/rms and q . x
    inv = tl.zeros((NBP,), tl.float32)
    dot = tl.zeros((NBP,), tl.float32)
    for j in range(0, NB + 1):
        if j < NB:
            x = tl.load(b_ptr + (t * NB + j) * N + c, mask=m, other=0.0).to(tl.float32)
        else:
            x = tl.load(p_ptr + t * N + c, mask=m, other=0.0).to(tl.float32)
        rj = tl.rsqrt(tl.sum(x * x, 0) / N + eps)
        dj = tl.sum(q * x, 0) * rj
        inv = tl.where(k == j, rj, inv)
        dot = tl.where(k == j, dj, dot)
    s = tl.where(k <= NB, dot * scale, float("-inf"))
    e = tl.exp(s - tl.max(s, 0))
    w = e / tl.sum(e, 0) * inv                             # p_j / rms_j
    acc = tl.zeros((B,), tl.float32)
    for j in range(0, NB + 1):
        wj = tl.sum(tl.where(k == j, w, 0.0), 0)
        if j < NB:
            x = tl.load(b_ptr + (t * NB + j) * N + c, mask=m, other=0.0).to(tl.float32)
        else:
            x = tl.load(p_ptr + t * N + c, mask=m, other=0.0).to(tl.float32)
        acc += wj * x
    tl.store(o_ptr + (t * NQ + qi) * N + c, acc, mask=m)


def block_residual(query, blocks, partial, eps=1e-6):
    """query [nq, H], blocks [T, nb, H], partial [T, H] -> [T, nq, H] fp32: softmax over the nb + 1 RMS-normed states."""
    T, NB, N = blocks.shape
    NQ = query.shape[0]
    o = torch.empty(T, NQ, N, dtype=torch.float32, device=partial.device)
    if T:
        with torch.cuda.device(partial.device):
            _block_res_kernel[(T, NQ)](query.contiguous(), blocks.contiguous(), partial.contiguous(), o, NB, NQ, N,
                                       N ** -0.5, eps, NBP=_blk(NB + 1), B=_blk(N), num_warps=8)
    return o


@triton.jit
def _gate_in_kernel(h_ptr, m_ptr, o_ptr, N, eps, B: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, B)
    msk = c < N
    h = tl.load(h_ptr + r * N + c, mask=msk, other=0.0).to(tl.float32)
    x = tl.load(m_ptr + r * N + c, mask=msk, other=0.0).to(tl.float32)
    tl.store(o_ptr + r * 2 * N + c, (h * tl.rsqrt(tl.sum(h * h, 0) / N + eps)).to(o_ptr.dtype.element_ty), mask=msk)
    tl.store(o_ptr + r * 2 * N + N + c, (x * tl.rsqrt(tl.sum(x * x, 0) / N + eps)).to(o_ptr.dtype.element_ty), mask=msk)


def gate_in(h, m, eps, out):
    """[rms(h) | rms(m)] [T, 2H] in dtype `out`."""
    T, N = h.shape
    o = torch.empty(T, 2 * N, dtype=out, device=h.device)
    if T:
        with torch.cuda.device(h.device):
            _gate_in_kernel[(T,)](h.contiguous(), m.contiguous(), o, N, eps, B=_blk(N), num_warps=4)
    return o


@triton.jit
def _tanh(x):
    e = tl.exp(-2.0 * tl.abs(x))
    t = (1.0 - e) / (1.0 + e)
    return tl.where(x < 0, -t, t)


@triton.jit
def _gate_norm_kernel(fg_ptr, h_ptr, m_ptr, w_ptr, o_ptr, N, eps, B: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, B)
    msk = c < N
    f = tl.load(fg_ptr + r * 2 * N + c, mask=msk, other=0.0)
    g = tl.load(fg_ptr + r * 2 * N + N + c, mask=msk, other=0.0)
    h = tl.load(h_ptr + r * N + c, mask=msk, other=0.0).to(tl.float32)
    x = tl.load(m_ptr + r * N + c, mask=msk, other=0.0).to(tl.float32)
    y = (1.0 - _tanh(f)) * h + _tanh(g) * x
    y = y * tl.rsqrt(tl.sum(y * y, 0) / N + eps) * tl.load(w_ptr + c, mask=msk, other=0.0).to(tl.float32)
    tl.store(o_ptr + r * N + c, y.to(o_ptr.dtype.element_ty), mask=msk)


def gate_norm(fg, h, m, w, eps, out):
    """fg [T, 2H] fp32 (forget | gain logits) -> rms((1 - tanh f) * h + tanh(g) * m) * w [T, H] in dtype `out`."""
    T, N = h.shape
    o = torch.empty(T, N, dtype=out, device=h.device)
    if T:
        with torch.cuda.device(h.device):
            _gate_norm_kernel[(T,)](fg.contiguous(), h.contiguous(), m.contiguous(), w, o, N, eps, B=_blk(N),
                                    num_warps=4)
    return o
