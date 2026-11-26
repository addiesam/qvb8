"""Layer kind: fused mHC routing weights (Triton) -- sigmoid read / write gates + the whole Sinkhorn of the stream-mix
matrix, ONE launch per call. Written for SenseNova-Loop (models/sensenova/adapter.py, MHCRouter).

Why: the reference computes, per router call, pre = sigmoid, post = 2 * sigmoid and Sinkhorn(mix) as 20 rounds of two
logsumexp on a 2 x 2 matrix -- 418 kernel launches per router + merge, 96 calls per decode step. Measured on the dev GPU
(decode shape [1, 8, 2, 4096] bf16): CPU enqueue 1.39 ms vs GPU busy 0.56 ms per call, ~133 ms of a 228 ms verify step.

Numerics: bit-identical to the reference ops, for any number of rows:
  * libdevice expf / logf and IEEE division (torch's CUDA maths), not Triton's fast exp / div;
  * torch.logsumexp's formula, log(sum(exp(x - max))) + max;
  * 2 streams only: every reduction is a sum of two terms, which has no order. torch reduces 4 terms in a
    shape-dependent order (interleaved along -1, sequential along -2), so more streams cannot be matched bit for bit.
The logits (RMS-norm + linear) and the small matmuls (branch = pre @ residual, the merge) stay in torch: cuBLAS rounds
those in a shape-dependent way, so a fused version flips greedy tokens.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

STREAMS = 2


@triton.jit
def _sigmoid(x):
    return libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x))         # torch's CUDA sigmoid: 1 / (1 + expf(-x))


@triton.jit
def _route_kernel(logits_ptr, pre_ptr, post_ptr, mix_ptr, S: tl.constexpr, ITERS: tl.constexpr):
    row = tl.program_id(0)
    lrow = logits_ptr + row * (2 * S + S * S)
    ss = tl.arange(0, S)
    tl.store(pre_ptr + row * S + ss, _sigmoid(tl.load(lrow + ss)))
    tl.store(post_ptr + row * S + ss, 2.0 * _sigmoid(tl.load(lrow + S + ss)))
    lg = tl.load(lrow + 2 * S + ss[:, None] * S + ss[None, :])
    for _ in tl.static_range(ITERS):
        mx = tl.max(lg, axis=1)
        lg = lg - (libdevice.log(tl.sum(libdevice.exp(lg - mx[:, None]), axis=1)) + mx)[:, None]
        mx = tl.max(lg, axis=0)
        lg = lg - (libdevice.log(tl.sum(libdevice.exp(lg - mx[None, :]), axis=0)) + mx)[None, :]
    tl.store(mix_ptr + row * S * S + ss[:, None] * S + ss[None, :], libdevice.exp(lg))


def mhc_route(logits: torch.Tensor, iterations: int):
    """logits [..., 2S + S*S] fp32 (S = 2) -> pre [..., S], post [..., S], mix [..., S, S], all fp32."""
    *lead, K = logits.shape
    S = STREAMS
    lg = logits.reshape(-1, K).contiguous()
    T = lg.shape[0]
    pre = torch.empty(T, S, device=lg.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    mix = torch.empty(T, S, S, device=lg.device, dtype=torch.float32)
    if T:
        with torch.cuda.device(lg.device):                                   # Triton launches on the current device
            _route_kernel[(T,)](lg, pre, post, mix, S=S, ITERS=int(iterations), num_warps=1)
    return pre.view(*lead, S), post.view(*lead, S), mix.view(*lead, S, S)
