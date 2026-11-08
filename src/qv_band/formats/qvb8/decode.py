"""QVB8 decode -- inference-side read path for the per-row virtual-band codec.

Deploy-only half of the QVB8 format: parses a .qvb8 blob and reconstructs
dense values from it. Calibration and packing (deciding min_s/tau_s/band ids
and writing a blob) live in qv_band.formats.qvb8.pack, which is NOT a
dependency of this module or of anything that imports it -- a deployed
server never needs to import calibration code, only this file.

Format constants, header layouts and the low-level bit-packing helpers are
defined here (not duplicated) because the calibration-side packer must
write bytes this reader can parse; qv_band.formats.qvb8.pack imports
them FROM here, not the other way around.

Each ROW is banded independently against a VIRTUAL codebook

    cb[i] = min_s + i * tau_s      with  min_s = row_min + tau
                                         tau_s = 2 * tau

so the codebook is never stored or searched -- entry i is one
multiply-add away from the index. Encode and decode are:

    id    = floor((w - row_min) / tau_s)
    value = min_s + id * tau_s

Certified: |value - w| <= tau for every non-escaped element, by
construction (id's bucket is tau_s wide and min_s sits at its centre). See
qv_band.formats.qvb8.pack's module docstring for the full certificate
scope and provenance -- this file only consumes the format it produces.

Storage layout (little-endian throughout):

    [ 64 B header ]
    [ Q * 4 B     ] row_min_s      fp32, the row's codebook origin
    [ Q * 4 B     ] row_tau_s      fp32, the row's bucket step
    [ Q * 1 B     ] row_nbits      uint8, bits per index in this row
    [ Q * 8 B     ] row_bit_off    uint64, this row's first index bit
    [ ceil(total_bits/8) B ] ids   bit-packed, row-major, LSB-first

Rows are bit-packed back to back with NO padding between them, so a
row's ids start at an arbitrary bit position -- row_bit_off records it.
Within a row every index is exactly row_nbits wide, so a reader strides
uniformly once it has the row's base, which is what keeps GEMV loads
coalesced.
"""
from __future__ import annotations

import struct

import numpy as np

from ...common.errors import Qvb8Error  # noqa: F401

MAGIC = 0x38425651        # 'QVB8'
VERSION = 11
HEADER_BYTES = 64
# v10: Q, P, n_rows_meta, reserved... ; tau as fp64 so the certified bound's
# constant survives a round trip exactly; then the escape encoding: mode
# (0 = every escape verbatim fp32, a v9 blob's zero padding reads as this;
#  1 = escapes split into an fp32 pool and an FP8 pool) and n_fp8, the size
# of the FP8 pool.
#
# v11: n_fp8 widened from a single byte (max 255) to a qword. A tensor
# whose FP8-pool escape count exceeds 255 -- routine at realistic tau on
# anything past a few hundred rows -- made struct.pack raise "'B' format
# requires 0 <= number <= 255" and the tensor was REFUSED outright; v10
# blobs already on disk only ever had n_fp8 <= 255 (anything larger
# failed to write in the first place), so their trailing byte plus the
# now-unused padding reads correctly as the low byte of a v11 qword --
# no re-pack needed, unpack_tensor branches on `version` regardless.
_HEADER_FMT = "<IIIIdQIQ20x"
_HEADER_FMT_V10 = "<IIIIdQIB27x"
# v9's header, kept so old packs still read: same first 33 bytes, the rest
# of the 64 was -- and is -- zero padding.
_HEADER_FMT_V9 = "<IIIIdQ32x"

# An index must fit the reader's accumulator...
MAX_NBITS = 32
# ...but fp32 runs out of precision long before that. The decode forms the
# product id*tau_s, whose fp32 ULP grows with the product: at nbits=14 one
# ULP is already 0.004*tau, at 16 it is 0.015*tau, and at 18 it is
# 0.061*tau. Past that, storage rounding alone breaches the certified
# bound no matter how the arithmetic is arranged -- measured: a row needing
# 17 bits reconstructs at 1.0033*tau, one needing 18 at 1.0223*tau.
# A row wider than this cannot be represented at this tau, so the codec
# refuses instead of returning a value it cannot certify.
MAX_CERTIFIED_NBITS = 14

# Mantissa bits of the dtype the kernel reconstructs into. Sets the peel
# threshold: ULP(v) <= 2^-FP16_MANTISSA_BITS * |v|, so ULP(v) <= tau holds
# exactly while |v| <= tau * 2^FP16_MANTISSA_BITS.
FP16_MANTISSA_BITS = 10


# Elements per bit-packing / bit-decoding block. The vectorised packer and
# reader each hold a block's worth of bit positions and one payload-sized
# byte array, so this caps a worker's transient the way _CHUNK_ELEMS caps
# calibration's -- it does not change a byte of the format (rows are packed
# back to back, block boundaries and all).
_PACK_BLOCK_ELEMS = 1 << 20

_E4M3_POS_CODES = 127     # codes 0..0x7E finite positive; 0x7F is NaN


def _e4m3_lut() -> np.ndarray:
    """256 float32 values, one per E4M3FN bit pattern."""
    global _E4M3_LUT
    if _E4M3_LUT is None:
        lut = np.empty(256, np.float32)
        for code in range(256):
            s = -1.0 if code & 0x80 else 1.0
            e = (code >> 3) & 0xF
            m = code & 0x7
            if e == 0:
                v = s * m * 2.0 ** -9          # subnormal: 2^-6 * m/8
            elif e == 15 and m == 7:
                v = np.nan                     # float8_e4m3fn's only NaN
            else:
                v = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
            lut[code] = np.float32(v)
        _E4M3_LUT = lut
    return _E4M3_LUT


_E4M3_LUT: np.ndarray | None = None


def _e4m3_decode(codes: np.ndarray, s8: np.ndarray) -> np.ndarray:
    """Stored value of an FP8 escape: the code's grid value times the row's
    FP8 scale, narrowed to fp16 -- the exact number the deployed decode
    reads, and the exact number the pack-time check certified. A product
    past fp16's range overflows to inf; the pack-time isfinite check is
    what routes such a value to the fp32 pool instead."""
    with np.errstate(over="ignore"):
        prod = _e4m3_lut()[codes] * s8
    return prod.astype(np.float16).astype(np.float32)


def _nbits_for(n_buckets: np.ndarray) -> np.ndarray:
    """Bits needed to address n_buckets distinct indices, at least 1."""
    nb = np.ceil(np.log2(np.maximum(n_buckets, 1))).astype(np.int64)
    return np.maximum(nb, 1)


def _combine_escapes(e32_idx: np.ndarray, e32_vals: np.ndarray,
                     e8_idx: np.ndarray, e8_codes: np.ndarray,
                     s8: np.ndarray, P: int):
    """One escape list from the fp32 and FP8 pools, ascending by flat index.

    The result is byte-identical to a v9 calibration's: escaped values that
    took the FP8 path appear as their decoded grid value, so every consumer
    (certified_error, the inference loaders) needs no new code path. The
    decoder reproduces exactly the value the pack-time check certified:
    lut[code] * s8[row] in fp32, narrowed to fp16.
    """
    if e8_idx.size == 0:
        return e32_idx, e32_vals
    e8_vals = _e4m3_decode(e8_codes, s8[e8_idx // P])
    esc_idx = np.concatenate([e32_idx, e8_idx])
    esc_vals = np.concatenate([e32_vals, e8_vals])
    order = np.argsort(esc_idx, kind="stable")
    return esc_idx[order], esc_vals[order]


# Dekker split factor for fp32: 2^ceil(24/2) + 1.
_DEKKER_SPLIT = np.float32(4097.0)


def fma32(a: np.ndarray, b, c: np.ndarray) -> np.ndarray:
    """Correctly-rounded a*b + c in fp32, matching the CUDA kernel's fmaf.

    A plain `c + a*b` rounds twice, once on the product and once on the
    sum, and diverges from the fused result wherever c and a*b nearly
    cancel. TwoProduct (Dekker split) and TwoSum recover both rounding
    errors exactly in fp32; adding them back reproduces the single-rounded
    value, so calibration certifies the number inference computes.

    numpy only: this runs in pack_model_qvb8's forked workers, where
    importing torch deadlocks the pool.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)   # scalar or per-row, broadcast
    prod = (a * b).astype(np.float32)
    ca = (_DEKKER_SPLIT * a).astype(np.float32)
    ah = (ca - (ca - a)).astype(np.float32)
    al = (a - ah).astype(np.float32)
    cb = (_DEKKER_SPLIT * b).astype(np.float32)
    bh = (cb - (cb - b)).astype(np.float32)
    bl = (b - bh).astype(np.float32)
    perr = (((ah * bh - prod) + ah * bl) + al * bh + al * bl).astype(np.float32)
    ssum = (c + prod).astype(np.float32)
    bv = (ssum - c).astype(np.float32)
    serr = ((c - (ssum - bv)) + (prod - bv)).astype(np.float32)
    return (ssum + (serr + perr)).astype(np.float32)


def reconstruct(min_s: np.ndarray, tau_s: np.float32, ids: np.ndarray,
                esc_idx: np.ndarray | None = None,
                esc_vals: np.ndarray | None = None) -> np.ndarray:
    """value = min_s + id * tau_s, via fma32.

    Bit-exact with the CUDA kernel's fmaf, so the error this certifies is
    the error inference produces.
    """
    ts = (np.broadcast_to(np.asarray(tau_s, np.float32).reshape(-1, 1),
                          ids.shape) if np.ndim(tau_s) else tau_s)
    out = fma32(ids.astype(np.float32), ts,
                np.ascontiguousarray(
                    np.broadcast_to(min_s[:, None], ids.shape),
                    dtype=np.float32))
    if esc_idx is not None and esc_idx.size:
        out.reshape(-1)[esc_idx] = esc_vals
    return out


def _row_bit_offsets(nbits: np.ndarray, P: int) -> np.ndarray:
    """Absolute bit offset of each row's first index, rows back to back."""
    row_bits = nbits.astype(np.int64) * P
    off = np.zeros(len(nbits), np.int64)
    if len(nbits) > 1:
        np.cumsum(row_bits[:-1], out=off[1:])
    return off


def _element_positions(nbits: np.ndarray, P: int, base_off: int = 0):
    """Bit position of every (row, col) element's FIRST index bit.

    Returns (pos, nb_e): pos[row*P + col] = base_off + row_off[row] +
    col * nbits[row], and nb_e is each element's row width. Those two are
    what let the per-bit loops index positions instead of walking rows in
    Python: an index's bit b sits at pos + b, and only elements whose row
    is wider than b may write or read there.
    """
    n = len(nbits)
    nb = nbits.astype(np.int64)
    erow = np.repeat(np.arange(n, dtype=np.int64), P)
    ecol = np.tile(np.arange(P, dtype=np.int64), n)
    pos = base_off + _row_bit_offsets(nb, P)[erow] + ecol * nb[erow]
    return pos, nb[erow]


def _pack_bits_block(ids: np.ndarray, nbits: np.ndarray, P: int):
    """Bit-pack ONE block of rows at their own widths, aligned at bit 0.

    Returns (bytes, total_bits). Callers OR the bytes into the payload at
    the block's own bit offset.
    """
    n = ids.shape[0]
    NB = int(nbits.max()) if n else 0
    if n == 0 or NB == 0:
        return np.zeros(0, np.uint8), 0
    pos, nb_e = _element_positions(nbits, P)
    total_bits = int(pos[-1] + nb_e[-1])
    # +NB: the last elements' writes of a bit past their own width land up
    # to NB-1 bits past the payload's last real bit. Those writes are zero
    # (masked out below, so the OR is a no-op) but the position must exist.
    bits = np.zeros(total_bits + NB + 8, np.uint8)
    flat_ids = ids.reshape(-1)
    span = np.arange(NB, dtype=np.uint64)
    for b in range(NB):
        bit = ((flat_ids >> span[b]) & np.uint64(1)).astype(np.uint8)
        bits[pos + b] |= bit & (nb_e > b).astype(np.uint8)
    return np.packbits(bits[:total_bits], bitorder="little"), total_bits


def _or_bits(dst: np.ndarray, bit_off: int, src: np.ndarray) -> None:
    """OR `src`'s bits into `dst` starting at absolute bit position bit_off.

    `bit_off` need not be byte-aligned, which is what keeps rows bit-packed
    back to back with no padding between them: each source byte's low part
    lands in place and its high part in the next byte.
    """
    n = src.size
    if n == 0:
        return
    byte = bit_off >> 3
    sh = bit_off & 7
    if sh == 0:
        dst[byte:byte + n] |= src
        return
    lo = ((src.astype(np.uint16) << np.uint16(sh)) & 0xFF).astype(np.uint8)
    hi = src >> np.uint8(8 - sh)
    dst[byte:byte + n] |= lo
    dst[byte + 1:byte + 1 + n] |= hi


def pack_bits_rows(ids: np.ndarray, nbits: np.ndarray,
                   rows_per_block: int | None = None):
    """Bit-pack each row's ids at its OWN width, rows back to back.

    Returns (buf uint8, bit_off uint64 (Q,)). Row r occupies bits
    [bit_off[r], bit_off[r] + nbits[r]*P), LSB-first within each index,
    with no padding between rows -- a row's width is uniform, so a reader
    only needs the row's base to stride through it.

    Packed row-BLOCK-wise: the per-bit loop runs over a block's elements in
    numpy rather than over a row's Python-level bits (the old form paid
    ~6us of interpreter per (row, bit) pair -- ~3.8M iterations for a
    248320x5120 embedding, more than the arithmetic it wrapped).
    """
    Q, P = ids.shape
    bit_off = _row_bit_offsets(nbits, P)
    total_bits = int((nbits.astype(np.int64) * P).sum())
    # + 8 trailing bytes: a reader lifting a field near the end reads a
    # whole word past it (see the CUDA kernel), and padding is cheaper
    # than branching on every element.
    buf = np.zeros((total_bits + 7) // 8 + 8, dtype=np.uint8)
    if Q == 0:
        return buf, bit_off.astype(np.uint64)
    rpb = rows_per_block or max(1, _PACK_BLOCK_ELEMS // max(1, P))
    for r0 in range(0, Q, rpb):
        r1 = min(r0 + rpb, Q)
        blk, _ = _pack_bits_block(ids[r0:r1], nbits[r0:r1], P)
        _or_bits(buf, int(bit_off[r0]), blk)
    return buf, bit_off.astype(np.uint64)


def _unpack_bits_block(buf: np.ndarray, nbits: np.ndarray, P: int,
                       base_bit: int = 0) -> np.ndarray:
    """Decode one block of rows whose fields start at `base_bit` in `buf`.

    A block is contiguous from `base_bit`, so its bits come out of `buf` in
    ONE np.unpackbits and are then gathered per bit. The old form walked
    rows in Python -- 2.7M numpy calls on (5120,) arrays for a 248320-row
    embedding; this is at most 14 calls on the whole block.
    """
    n = len(nbits)
    if n == 0:
        return np.zeros((0, P), np.int64)
    NB = int(nbits.max())
    pos, nb_e = _element_positions(nbits, P, base_bit)
    last = int(pos[-1] + nb_e[-1])                  # one past the last bit
    byte0 = base_bit >> 3
    nbytes = ((last + NB + 7) >> 3) - byte0
    bits = np.unpackbits(np.ascontiguousarray(buf[byte0:byte0 + nbytes]),
                         bitorder="little")
    rel = pos - (byte0 << 3)
    acc = np.zeros(n * P, np.uint16)
    for b in range(NB):
        acc |= bits[rel + b].astype(np.uint16) << np.uint16(b)
    # Bits past a row's own width were gathered out of the next element's
    # field; masking them off once is cheaper than masking every gather.
    acc &= (np.uint16(1) << nb_e.astype(np.uint16)) - np.uint16(1)
    return acc.astype(np.int64).reshape(n, P)


def unpack_bits_rows(buf: np.ndarray, nbits: np.ndarray,
                     bit_off: np.ndarray, Q: int, P: int,
                     r0: int = 0) -> np.ndarray:
    """Inverse of pack_bits_rows.

    Decodes len(nbits) rows starting at absolute row r0; bit_off values
    stay ABSOLUTE and so does the indexing into buf, so a caller can
    stream a blob chunk by chunk without re-slicing the payload.
    """
    nbits = np.asarray(nbits)
    n = len(nbits)
    out = np.zeros((n, P), np.int64)
    if n == 0:
        return out
    rpb = max(1, _PACK_BLOCK_ELEMS // max(1, P))
    for i0 in range(0, n, rpb):
        i1 = min(i0 + rpb, n)
        out[i0:i1] = _unpack_bits_block(buf, nbits[i0:i1], P,
                                        int(bit_off[r0 + i0]))
    return out


def read_header(blob: bytes):
    """(Q, P, tau, n_escapes) from the 64-byte header alone.

    Callers that only need the shape must NOT go through unpack_tensor:
    that copies every side array and the whole bit payload, which for a
    248320x5120 lm_head is ~1 GiB of pointless work.
    """
    magic, Q, P, version, tau, n_esc = struct.unpack(
        _HEADER_FMT_V9, blob[:HEADER_BYTES])
    if magic != MAGIC:
        raise Qvb8Error(f"not a QVB8 blob (magic {magic:#x})")
    if version not in (9, VERSION):
        raise Qvb8Error(f"QVB8 version {version}, expected {VERSION} (or 9)")
    return Q, P, tau, n_esc


def unpack_tensor(blob: bytes):
    """Inverse of pack_tensor. Returns (min_s, tau_s, nbits, bit_off, buf,
    Q, P, tau, esc_idx, esc_vals).

    v10 blobs hold escapes in two pools -- fp32 verbatim and FP8 codes with
    per-row scales -- and this merges them back into ONE (esc_idx, esc_vals)
    in the v9 shape, ascending by flat index. That is what keeps every
    downstream consumer (certified_error, the CUDA/CPU loaders, MTP) on the
    old code path: the FP8 decode happens exactly once, here, reproducing
    lut[code] * s8[row] narrowed to fp16 -- the same expression the packer
    certified against.
    """
    # version is the 4th field (offset 12) in both the v10 and v11 header
    # layouts -- they share the same prefix -- so it can be read before
    # picking which struct format the rest of the header needs.
    magic, _, _, version = struct.unpack_from("<IIII", blob, 0)
    if magic != MAGIC:
        raise Qvb8Error(f"not a QVB8 blob (magic {magic:#x})")
    if version not in (9, 10, VERSION):
        raise Qvb8Error(f"QVB8 version {version}, expected {VERSION} (or 9, 10)")
    header_fmt = _HEADER_FMT_V10 if version == 10 else _HEADER_FMT
    magic, Q, P, version, tau, n_esc, esc_mode, n_fp8 = struct.unpack(
        header_fmt, blob[:HEADER_BYTES])
    cur = HEADER_BYTES
    min_s = np.frombuffer(blob, "<f4", count=Q, offset=cur).copy(); cur += 4 * Q
    tau_s = np.frombuffer(blob, "<f4", count=Q, offset=cur).copy(); cur += 4 * Q
    nbits = np.frombuffer(blob, np.uint8, count=Q, offset=cur).astype(np.int64); cur += Q
    bit_off = np.frombuffer(blob, "<u8", count=Q, offset=cur).astype(np.int64); cur += 8 * Q
    if version == 9 or esc_mode == 0:
        if version == 9:
            n_fp8 = 0
        esc_idx = np.frombuffer(blob, "<i8", count=n_esc, offset=cur).copy(); cur += 8 * n_esc
        esc_vals = np.frombuffer(blob, "<f4", count=n_esc, offset=cur).copy(); cur += 4 * n_esc
    else:
        if n_fp8 > n_esc:
            raise Qvb8Error(f"header says {n_fp8} FP8 escapes of {n_esc} total")
        n32 = n_esc - n_fp8
        e32_idx = np.frombuffer(blob, "<i8", count=n32, offset=cur).copy(); cur += 8 * n32
        e32_vals = np.frombuffer(blob, "<f4", count=n32, offset=cur).copy(); cur += 4 * n32
        e8_idx = np.frombuffer(blob, "<i8", count=n_fp8, offset=cur).copy(); cur += 8 * n_fp8
        e8_codes = np.frombuffer(blob, np.uint8, count=n_fp8, offset=cur).copy(); cur += n_fp8
        s8 = np.frombuffer(blob, "<f4", count=Q, offset=cur).copy(); cur += 4 * Q
        esc_idx, esc_vals = _combine_escapes(e32_idx, e32_vals,
                                             e8_idx, e8_codes, s8, P)
    buf = np.frombuffer(blob, np.uint8, offset=cur).copy()
    return (min_s, tau_s, nbits, bit_off, buf, Q, P, tau,
            esc_idx, esc_vals)


def decode_tensor(blob: bytes) -> np.ndarray:
    """Full dense reconstruction from a QVB8 blob."""
    (min_s, tau_s, nbits, bit_off, buf, Q, P, _,
     esc_idx, esc_vals) = unpack_tensor(blob)
    ids = unpack_bits_rows(buf, nbits, bit_off, Q, P)
    return reconstruct(min_s, tau_s, ids, esc_idx, esc_vals)
