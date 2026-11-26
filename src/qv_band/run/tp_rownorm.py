"""Row-deterministic RMSNorms for Qwen3.5 / Qwen3.8 (Triton): one program per row, one fixed reduction order, so a
row's result is bit-identical whether it is normalised alone (decode, T=1) or in a block (DFlash verify, T=8).
torch's reduction picks its strategy from the tensor shape: on the 27B, layer 1's input norm differed in the last
bits between a 1-row and an 8-row call, which moved a near-tie and broke "speculative output == plain output".

Same math as transformers' Qwen3_5RMSNorm ((x * rsqrt(mean(x^2) + eps)) * (1 + w), fp32, cast back) and
Qwen3_5RMSNormGated (normalise in fp32, cast, * w in the input dtype, * silu(gate) in fp32, cast back).
`install(model)` swaps the forward of every such module of the model.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_kernel(X, W, G, Y, N, eps, stride, PLUS_ONE: tl.constexpr, GATED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    msk = offs < N
    x = tl.load(X + row * stride + offs, mask=msk, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    xn = x * (1.0 / tl.sqrt(var + eps))
    w = tl.load(W + offs, mask=msk, other=0.0).to(tl.float32)
    if GATED:
        y = xn.to(tl.float16).to(tl.float32)
        y = (w * y).to(tl.float16).to(tl.float32)
        g = tl.load(G + row * stride + offs, mask=msk, other=0.0).to(tl.float32)
        y = y * (g / (1.0 + tl.exp(-g)))
    elif PLUS_ONE:
        y = xn * (1.0 + w)
    else:
        y = xn * w
    tl.store(Y + row * stride + offs, y.to(tl.float16), mask=msk)


def rms_rows(x, w, eps, plus_one=True, gate=None):
    shape = x.shape
    N = shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    y = torch.empty_like(x2)
    g2 = gate.reshape(-1, N).contiguous() if gate is not None else x2
    _rms_kernel[(x2.shape[0],)](x2, w, g2, y, N, eps, N, PLUS_ONE=plus_one, GATED=gate is not None,
                                BLOCK=triton.next_power_of_2(N), num_warps=4 if N <= 1024 else 8)
    return y.reshape(shape)


def install(model) -> int:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    norm, gated = (M.Qwen3_5RMSNorm,), (M.Qwen3_5RMSNormGated,)
    try:                                         # the MoE family (Qwen3.6-35B-A3B) has its own copies of the classes
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as MM
        norm, gated = norm + (MM.Qwen3_5MoeRMSNorm,), gated + (MM.Qwen3_5MoeRMSNormGated,)
    except Exception:                            # noqa: BLE001
        pass
    n = 0
    for m in model.modules():
        if isinstance(m, norm):
            m.forward = (lambda mm: lambda x: rms_rows(x, mm.weight, mm.eps, True))(m)
            n += 1
        elif isinstance(m, gated):
            m.forward = (lambda mm: lambda x, gate: rms_rows(x, mm.weight, mm.variance_epsilon, False,
                                                             gate.to(x.dtype)))(m)
            n += 1
    return n
