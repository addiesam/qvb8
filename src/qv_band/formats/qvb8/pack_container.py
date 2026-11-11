"""pack_container: QVB8 pack reader (release build -- the pack writer / calibration is not included)."""
from __future__ import annotations
import struct
from typing import Literal
import numpy as np
from .pack import Qvb8Error, unpack_bits_rows, _combine_escapes
SourceKind = Literal['bf16', 'fp8_block', 'q4_block', 'fp32']
_MAGIC = b'QVBD'
_VERSION = 1
_CALC_FORMULA = 0
_CALC_TABLE = 1
_SOURCE_CODES = {'bf16': 0, 'fp8_block': 1, 'q4_block': 2, 'fp32': 3}
_SOURCE_NAMES = {v: k for k, v in _SOURCE_CODES.items()}
_ESC32_WIDTH = {'bf16': 2, 'fp8_block': 2, 'q4_block': 2, 'fp32': 4}
_SOURCE_IS_FP32_IDENTITY = {'bf16': True, 'fp8_block': True, 'q4_block': True, 'fp32': True}
_RESTORE_FP16 = 0
_RESTORE_FP16_BF16 = 3
_RESTORE_NAMES = {_RESTORE_FP16: 'fp16', _RESTORE_FP16_BF16: 'fp16+bf16'}
_ROW_BLOCK_ELEMS = 4 << 20
_HEADER_FMT = '<4sBIIffBBB8x'
_HEADER_BYTES = struct.calcsize(_HEADER_FMT)
assert _HEADER_BYTES == 32, _HEADER_BYTES

def _decode_esc32(raw: np.ndarray, source: str) -> np.ndarray:
    """Inverse of _encode_esc32: on-disk esc32 payload -> fp32 values."""
    w = _ESC32_WIDTH[source]
    if w == 4:
        return raw.view(np.float32)
    if w == 2:
        return (raw.astype(np.uint32) << 16).view(np.float32)
    raise Qvb8Error(f'no esc32 width for source {source!r}')

def _unpack_header(blob: bytes) -> tuple:
    magic, ver, Q, P, tau, max_err, source_code, calc_method, restore_code = struct.unpack(_HEADER_FMT, blob[:_HEADER_BYTES])
    if magic != _MAGIC:
        raise Qvb8Error(f'not a qvband blob (magic {magic!r})')
    if ver != _VERSION:
        raise Qvb8Error(f'qvband version {ver}, reader supports {_VERSION}')
    if restore_code not in _RESTORE_NAMES:
        raise Qvb8Error(f'pack has restore code {restore_code}: fp8/nvfp4 restore targets were removed (fp16, and fp16+bf16, are the restore precisions); re-pack this tensor')
    return (int(Q), int(P), float(tau), float(max_err), _SOURCE_NAMES[source_code], int(calc_method), _RESTORE_NAMES[restore_code])

def calc_method_of(blob: bytes) -> int:
    """Peek a blob's calc_method without a full unpack -- lets a reader
    check for a legacy table pack without parsing twice."""
    return _unpack_header(blob)[5]

def unpack_formula(blob: bytes, *, skip_bit_decode: bool=False, fp16_list=None, where: str='', _compute_fp16: bool=False) -> dict:
    """Inverse of pack_formula. Returns a dict with Q, P, tau, max_err,
    source, min_s, tau_s, codes (Q,P) int64, esc_idx, esc_vals -- the
    latter two already MERGE the esc32 and esc8 pools back into one
    (flat index, exact fp32 value) pair per escape, ascending by flat
    index, matching pack_qvb8.py's unpack_tensor convention so a caller
    never has to know two pools exist on disk. Also returns nbits,
    bit_off, and the raw bit-packed `buf` -- a caller with a GPU decode
    kernel (see module_qvband.py's _dense_weight_formula) can use those
    directly instead of paying this function's own CPU unpack_bits_rows
    call, which measured as the dominant cost of loading a qvband pack
    (~0.6s of ~0.8s total on a 10240x5120 layer) and multiplies across
    a whole model's tensor count at load time.

    `skip_bit_decode=True` skips the CPU unpack_bits_rows call and
    returns `codes=None` -- use only when the caller is about to decode
    on GPU from the raw fields instead.
    """
    Q, P, tau, max_err, source, calc_method, restore = _unpack_header(blob)
    if calc_method != _CALC_FORMULA:
        raise Qvb8Error(f'unpack_formula got calc_method={calc_method}, expected {_CALC_FORMULA}')
    cur = _HEADER_BYTES
    min_s = np.frombuffer(blob, '<f4', count=Q, offset=cur).copy()
    cur += 4 * Q
    tau_s = np.frombuffer(blob, '<f4', count=Q, offset=cur).copy()
    cur += 4 * Q
    nbits = np.frombuffer(blob, np.uint8, count=Q, offset=cur).copy()
    cur += Q
    bit_off = _bit_offsets_from_nbits(nbits, P)
    payload_bytes = _bits_payload_bytes(nbits, P)
    buf = np.frombuffer(blob, np.uint8, count=payload_bytes + 8, offset=cur).copy()
    codes = None if skip_bit_decode else unpack_bits_rows(buf, nbits, bit_off, Q, P)
    cur += payload_bytes + 8
    n_esc_total, n8 = struct.unpack_from('<QQ', blob, cur)
    cur += 16
    n32 = int(n_esc_total) - int(n8)
    esc32_idx = np.frombuffer(blob, '<i8', count=n32, offset=cur).copy()
    cur += 8 * n32
    w32 = _ESC32_WIDTH[source]
    esc32_raw = np.frombuffer(blob, '<u4' if w32 == 4 else '<u2', count=n32, offset=cur).copy()
    cur += w32 * n32
    esc32_vals = _decode_esc32(esc32_raw, source)
    esc8_idx = np.frombuffer(blob, '<i8', count=n8, offset=cur).copy()
    cur += 8 * n8
    esc8_codes = np.frombuffer(blob, np.uint8, count=n8, offset=cur).copy()
    cur += n8
    s8 = np.frombuffer(blob, '<f4', count=Q, offset=cur).copy()
    cur += 4 * Q
    esc_idx, esc_vals = _combine_escapes(esc32_idx, esc32_vals, esc8_idx, esc8_codes, s8, P)
    if restore == 'fp16+bf16' and _compute_fp16:
        from qv_band.formats.qvb8.escape_split import split_esc32
        need = split_esc32(esc32_idx, esc32_vals, buf=buf, nbits=nbits, bit_off=bit_off, min_s=min_s, tau_s=tau_s, P=P, tau=tau)
        esc_idx_fp16, esc_vals_fp16 = _combine_escapes(esc32_idx[need], esc32_vals[need], esc8_idx, esc8_codes, s8, P)
    elif restore == 'fp16+bf16':
        if fp16_list is None:
            esc_idx_fp16 = esc_vals_fp16 = None
        else:
            from qv_band.formats.qvb8.escape_split import select_fp16
            keep = select_fp16(esc_idx, fp16_list, where)
            esc_idx_fp16, esc_vals_fp16 = (esc_idx[keep], esc_vals[keep])
    else:
        esc_idx_fp16, esc_vals_fp16 = (esc_idx, esc_vals)
    return dict(Q=Q, P=P, tau=tau, max_err=max_err, source=source, restore=restore, min_s=min_s, tau_s=tau_s, codes=codes, esc_idx=esc_idx, esc_vals=esc_vals, esc_idx_fp16=esc_idx_fp16, esc_vals_fp16=esc_vals_fp16, nbits=nbits, bit_off=bit_off, buf=buf)

def compute_fp16_table(blob: bytes):
    """(restore, full escape idx, fp16 escape idx) of a QVBD blob, by the tau check (escape_split.split_esc32).
    Used ONLY to write escape_tables.json (end of calibration, or the escape_split CLI) -- never at load."""
    d = unpack_formula(blob, skip_bit_decode=True, _compute_fp16=True)
    return (d['restore'], np.asarray(d['esc_idx'], np.int64), np.asarray(d['esc_idx_fp16'], np.int64))

def _bit_offsets_from_nbits(nbits: np.ndarray, P: int) -> np.ndarray:
    """pack_bits_rows lays rows back to back with no padding between
    them; each row's bit length is nbits[row]*P, so its absolute bit
    offset is the exclusive prefix sum of that."""
    row_bits = nbits.astype(np.int64) * P
    off = np.zeros(len(nbits), np.int64)
    np.cumsum(row_bits[:-1], out=off[1:])
    return off

def _bits_payload_bytes(nbits: np.ndarray, P: int) -> int:
    total_bits = int(nbits.astype(np.int64).sum()) * P
    return (total_bits + 7) // 8
