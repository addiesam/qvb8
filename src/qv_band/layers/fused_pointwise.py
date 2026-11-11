"""Layer kind: fused element-wise Triton kernels (clamped-SiLU MLP activation, RMSNorm, RoPE, sigmoid gate).
Written for Ripple prefill; the adapter names the module classes to bind (install_fused).

Profile of one 4096-token prefill chunk on the dev GPU (13.2 s GPU): GEMMs 6.9 s, weight decode 1.0 s, and ~2.5 s in
the two ops below, each written in the shipped modeling file as 4-6 separate full-tensor passes:

  RicursiveMLP     gate.clamp(max=t), up.clamp(-t, t), silu(gate), * up       -> 1.25 s on (4096, 40960) tensors
  RicursiveRMSNorm .float(), pow(2), mean, * rsqrt, .to(bf16), weight *        -> ~1.3 s (4 per layer + q/k norms)

Each becomes ONE Triton kernel: read inputs once, write the result once.

Numerics follow the reference op by op, in fp32 with the same rounding points:
  * MLP: silu(min(g,t)) is rounded to the input dtype before the multiply, as torch does (silu returns bf16),
    and the product is rounded once -- bit-identical to the reference.
  * RMSNorm: the row sum of squares is reduced in a different order than torch's mean kernel, so the
    normalised value can differ in the last bit (|rel| ~1e-7 before the bf16 rounding). Not bit-identical.

Installed by a model adapter (today: models/ripple/adapter.py).
QV_RIPPLE_FUSED=0 turns it off.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _mlp_act_kernel(g_ptr, u_ptr, o_ptr, n, t, HAS_T: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0)
    if HAS_T:
        g = tl.minimum(g, t)
        u = tl.minimum(tl.maximum(u.to(tl.float32), -t), t).to(u.dtype)    # clamp is exact in bf16
    s = tl.math.div_rn(g, 1.0 + tl.extra.cuda.libdevice.exp(-g))           # torch silu: x / (1 + exp(-x))
    s = s.to(u.dtype).to(tl.float32)                                       # silu() returns the input dtype
    tl.store(o_ptr + offs, (s * u.to(tl.float32)).to(u.dtype), mask=m)


def mlp_act(gate: torch.Tensor, up: torch.Tensor, t: float | None) -> torch.Tensor:
    g, u = gate.contiguous(), up.contiguous()
    out = torch.empty_like(u)
    n = u.numel()
    if n == 0:                                   # a zero-size grid is a launch error
        return out
    BLOCK = 2048
    _mlp_act_kernel[(triton.cdiv(n, BLOCK),)](g, u, out, n, float(t or 0.0), HAS_T=bool(t), BLOCK=BLOCK,
                                               num_warps=8)
    return out


@triton.jit
def _rms_kernel(x_ptr, w_ptr, o_ptr, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=m, other=0.0)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / D
    y = (xf * tl.math.rsqrt(var + eps)).to(x.dtype)                        # .to(input_dtype)
    w = tl.load(w_ptr + cols, mask=m, other=0.0)
    # weight * y in torch's promoted dtype (bf16 weight: bf16 product; fp32 weight: fp32)
    tl.store(o_ptr + row * D + cols, w.to(tl.float32) * y.to(tl.float32), mask=m)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    D = x.shape[-1]
    xc = x.contiguous()
    out = torch.empty(x.shape, dtype=torch.promote_types(weight.dtype, x.dtype), device=x.device)
    rows = xc.numel() // D
    if rows == 0:
        return out
    BLOCK = triton.next_power_of_2(D)
    _rms_kernel[(rows,)](xc, weight, out, D, float(eps), BLOCK=BLOCK, num_warps=8 if D >= 2048 else 4)
    return out


def install_fused(model, mlp_class: str, norm_class: str) -> int:
    """Swap the forward of every module whose class is named `mlp_class` / `norm_class`. Returns how many modules."""
    if os.environ.get("QV_RIPPLE_FUSED", "1") == "0" or not torch.cuda.is_available():
        return 0
    n = 0
    for mod in model.modules():
        name = type(mod).__name__
        if name == mlp_class:
            def fwd(x, _m=mod):
                return _m.down_proj(mlp_act(_m.gate_proj(x), _m.up_proj(x), _m.clamp_threshold))
            mod.forward = fwd
            n += 1
        elif name == norm_class:
            def fwd(x, _m=mod):
                if not x.is_cuda:
                    return type(_m).forward(_m, x)
                return rms_norm(x, _m.weight, _m.eps)
            mod.forward = fwd
            n += 1
    return n


# -- attention-side element-wise ops (used by layers/sliding_attention.py) -------------------------------------
@triton.jit
def _rope_kernel(x_ptr, cos_ptr, sin_ptr, o_ptr, H, D: tl.constexpr):
    # x, o: (B*T*H, D) contiguous rows; cos/sin: (B*T, D). OLMo-core RoPE in fp32:
    # x*cos + rotate_half(x)*sin, rotate_half(x) = cat(-x[D/2:], x[:D/2]).
    row = tl.program_id(0).to(tl.int64)
    bt = row // H
    half: tl.constexpr = D // 2
    c = tl.arange(0, half)
    x1 = tl.load(x_ptr + row * D + c).to(tl.float32)
    x2 = tl.load(x_ptr + row * D + half + c).to(tl.float32)
    c1 = tl.load(cos_ptr + bt * D + c).to(tl.float32)
    c2 = tl.load(cos_ptr + bt * D + half + c).to(tl.float32)
    s1 = tl.load(sin_ptr + bt * D + c).to(tl.float32)
    s2 = tl.load(sin_ptr + bt * D + half + c).to(tl.float32)
    o1 = x1 * c1 + (-x2) * s1
    o2 = x2 * c2 + x1 * s2
    tl.store(o_ptr + row * D + c, o1.to(o_ptr.dtype.element_ty))
    tl.store(o_ptr + row * D + half + c, o2.to(o_ptr.dtype.element_ty))


def rope_bthd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, T, H, D) contiguous; cos/sin: (B, T, D). Returns RoPE(x) in x's dtype, same layout."""
    B, T, H, D = x.shape
    assert D % 2 == 0 and (D // 2) & (D // 2 - 1) == 0, f"rope_bthd: head_dim {D} must be 2 x a power of two"
    xc = x.contiguous()
    out = torch.empty_like(xc)
    if xc.numel() == 0:
        return out
    # cos/sin come from position_ids, whose batch can be 1 for a B-row input: the kernel indexes them by
    # (b, t), so give them the input's batch explicitly (an unexpanded batch-1 table was an out-of-bounds read).
    if cos.shape[0] != B:
        cos, sin = cos.expand(B, -1, -1), sin.expand(B, -1, -1)
    assert cos.shape == (B, T, D), f"rope_bthd: cos {tuple(cos.shape)} for input {tuple(x.shape)}"
    _rope_kernel[(B * T * H,)](xc, cos.contiguous(), sin.contiguous(), out, H, D=D, num_warps=1)
    return out


@triton.jit
def _gate_kernel(o_ptr, g_ptr, y_ptr, n, BLOCK: tl.constexpr):
    # out * sigmoid(gate.float()).to(out.dtype): sigmoid in fp32, rounded to out's dtype, product rounded once.
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    o = tl.load(o_ptr + offs, mask=m, other=0.0)
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.math.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-g)).to(o.dtype).to(tl.float32)
    tl.store(y_ptr + offs, (o.to(tl.float32) * s).to(o.dtype), mask=m)


def gate_sigmoid(out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    o, g = out.contiguous(), gate.contiguous()
    y = torch.empty_like(o)
    n = o.numel()
    if n == 0:
        return y
    BLOCK = 2048
    _gate_kernel[(triton.cdiv(n, BLOCK),)](o, g, y, n, BLOCK=BLOCK, num_warps=8)
    return y


def enabled() -> bool:
    return os.environ.get("QV_RIPPLE_FUSED", "1") != "0" and torch.cuda.is_available()
