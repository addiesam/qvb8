"""Single-pass Triton versions of the qvb8 KV codec's bit pack / unpack / dequantise.

The torch versions in codec.py build several int64 temporaries per call (arange, shifts,
masks, index_add, ...). A prefill chunk packs and later unpacks the K and V of every
full-attention layer, which cost ~5% of the chunk. Same layout (LSB-first, bit-contiguous,
row-major), byte-for-byte identical results:

  pack(ids, bits)                       -> uint8
  unpack(packed, bits, n)               -> int32 ids
  dequant(packed, bits, n, min_s, tau_s, hd, dtype)
                                        -> min_s[c] + id * tau_s[c] (fp32 mul, then add,
                                           each rounded separately -- exactly the expression
                                           the encoder certified -- then cast to dtype)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _pack_k(IDS, OUT, n_ids, n_bytes, BITS: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)          # output byte
    m = b < n_bytes
    lo = b.to(tl.int64) * 8                                      # first bit of this byte
    first = lo // BITS                                           # the id that covers it
    acc = tl.zeros([BLOCK], tl.int32)
    for k in tl.static_range(K):
        i = first + k
        pos = i * BITS                                           # bit position of id i
        ok = m & (i < n_ids) & (pos < lo + 8)
        v = tl.load(IDS + i, mask=ok, other=0).to(tl.int32)
        rel = (pos - lo).to(tl.int32)                            # < 0: the id began in an earlier byte
        up = tl.where(rel >= 0, rel, 0)
        dn = tl.where(rel < 0, -rel, 0)
        acc |= tl.where(ok, ((v << up) >> dn) & 255, 0)
    tl.store(OUT + b, acc.to(tl.uint8), mask=m)


@triton.jit
def _load_ids(P, e, m, n_bytes, BITS: tl.constexpr):
    bp = e.to(tl.int64) * BITS
    b0 = bp >> 3
    sh = (bp & 7).to(tl.int32)
    x = tl.load(P + b0, mask=m & (b0 < n_bytes), other=0).to(tl.int32)
    if BITS + 7 > 8:
        x |= tl.load(P + b0 + 1, mask=m & (b0 + 1 < n_bytes), other=0).to(tl.int32) << 8
    if BITS + 7 > 16:
        x |= tl.load(P + b0 + 2, mask=m & (b0 + 2 < n_bytes), other=0).to(tl.int32) << 16
    return (x >> sh) & ((1 << BITS) - 1)


@triton.jit
def _unpack_k(P, OUT, n, n_bytes, BITS: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = e < n
    tl.store(OUT + e, _load_ids(P, e, m, n_bytes, BITS), mask=m)


@triton.jit
def _dequant_k(P, MIN, TAU, OUT, n, n_bytes, hd, BITS: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = e < n
    v = _load_ids(P, e, m, n_bytes, BITS)
    c = e % hd
    mn = tl.load(MIN + c, mask=m, other=0.0)
    tu = tl.load(TAU + c, mask=m, other=0.0)
    y = mn + v.to(tl.float32) * tu               # launched with fp fusion off: mul and add rounded apart
    tl.store(OUT + e, y.to(OUT.dtype.element_ty), mask=m)


def pack(ids: torch.Tensor, bits: int) -> torch.Tensor:
    ids = ids.reshape(-1).contiguous()
    n = ids.numel()
    n_bytes = (n * bits + 7) // 8
    out = torch.empty(n_bytes, device=ids.device, dtype=torch.uint8)
    _pack_k[(triton.cdiv(n_bytes, _BLOCK),)](ids, out, n, n_bytes, BITS=bits,
                                              K=(8 + bits - 1) // bits + 1, BLOCK=_BLOCK, num_warps=4)
    return out


def unpack(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    packed = packed.contiguous()
    out = torch.empty(n, device=packed.device, dtype=torch.int32)
    _unpack_k[(triton.cdiv(n, _BLOCK),)](packed, out, n, packed.numel(), BITS=bits, BLOCK=_BLOCK, num_warps=4)
    return out


def dequant(packed: torch.Tensor, bits: int, n: int, min_s: torch.Tensor, tau_s: torch.Tensor,
            hd: int, dtype: torch.dtype) -> torch.Tensor:
    packed = packed.contiguous()
    out = torch.empty(n, device=packed.device, dtype=dtype)
    _dequant_k[(triton.cdiv(n, _BLOCK),)](packed, min_s.contiguous(), tau_s.contiguous(), out, n,
                                           packed.numel(), hd, BITS=bits, BLOCK=_BLOCK,
                                           num_warps=4, enable_fp_fusion=False)
    return out
