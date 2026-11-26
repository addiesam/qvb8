"""Layer kind: low-rank logit correction (Triton) -- logits = softcap(base + b + x @ W^T, cap), x [n, K] (small K, e.g.
the 256-wide latent of a DSpark Markov head), W [V, K] (a folded lm_head . projection), base [n, V] fp32: ONE pass over W.
Written for Inkling-2's own DSpark drafter (models/inkling/dspark.py: 7 calls per proposal). Why: cuBLAS takes the
[1, 256] x [256, 200k] bf16 product as a transposed GEMV at ~115 GB/s (0.9 ms each on the dev GPU); here each program reads
a [BV, K] tile of W once for all n rows and adds base / bias / softcap in the same pass.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fold_logits_kernel(x_ptr, w_ptr, b_ptr, base_ptr, out_ptr, V, cap, N: tl.constexpr, K: tl.constexpr,
                        BV: tl.constexpr):
    v = tl.program_id(0) * BV + tl.arange(0, BV)
    mv = v < V
    kk = tl.arange(0, K)
    w = tl.load(w_ptr + v[:, None].to(tl.int64) * K + kk[None, :], mask=mv[:, None], other=0.0).to(tl.float32)
    b = tl.load(b_ptr + v, mask=mv, other=0.0)
    for n in tl.static_range(N):
        x = tl.load(x_ptr + n * K + kk).to(tl.float32)
        y = tl.sum(w * x[None, :], 1) + b + tl.load(base_ptr + n * V + v, mask=mv, other=0.0)
        if cap > 0:                                   # softcap: tanh(y / cap) * cap
            e = tl.exp(-2.0 * tl.abs(y / cap))
            th = (1.0 - e) / (1.0 + e)
            y = tl.where(y < 0, -th, th) * cap
        tl.store(out_ptr + n * V + v, y, mask=mv)


def fold_logits(x, w, b, base, cap: float = 0.0):
    """x [n, K], w [V, K], b [V] fp32, base [n, V] fp32 (contiguous) -> softcap(base + b + x @ w^T, cap) [n, V] fp32
    (cap 0: no softcap)."""
    n, K = x.shape
    V = w.shape[0]
    out = torch.empty(n, V, dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _fold_logits_kernel[(triton.cdiv(V, 64),)](x.contiguous(), w, b, base.contiguous(), out, V, float(cap or 0.0),
                                                   N=n, K=K, BV=64, num_warps=4)
    return out
