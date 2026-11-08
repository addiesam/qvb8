"""The .qvb8r file: one tensor per file.

Layout: b'QVBR' | u32 version | u32 header_len | header JSON | zero pad to 64 | arrays (64-aligned) | u32 crc32
The header holds Q, P, T, K, N, PB, the certification record copied from the QVB8 pack, and every array as
name -> [offset, dtype, shape]; the crc32 covers every byte before it.

Arrays: perm i32[Q] (sorted row -> original row), freq u16[K,N], tiles i32[nt,3] (class, first sorted row, rows),
tile_off u32[nt+1] (in words), states u64[nt,32], words u32[], drow i16[Q] (id = symbol + drow), min_s f32[Q],
tau_s f32[Q], esc_idx i64[E] (flat q*P+c), esc_val f32[E] (exact value), esc_delta f32[E] (exact - fp32 band value).
Version 2 = centred rows + 64-bit states / 32-bit words (codec.py). Version 1 files (16-bit words, no drow) are
refused: re-run the transcode from the QVB8 pack.
Slot tables, the block schedule and the read-ahead pad are derived at load (runtime_tables)."""
from __future__ import annotations

import json
import zlib

import numpy as np

from .codec import PB, slot_tables

MAGIC, VERSION = b'QVBR', 2
PAD_WORDS = 2048      # zero words after the stream: the kernels read up to RING + RING / 2 (384) words ahead


class QVB8RFormatError(ValueError):
    pass


def write(path: str, enc: dict, Q: int, P: int, min_s, tau_s, esc_idx, esc_val, esc_delta, cert: dict) -> int:
    arrs = dict(perm=enc['perm'], freq=enc['freq'], tiles=enc['tiles'], tile_off=enc['tile_off'], states=enc['states'],
                words=enc['words'], drow=enc['drow'], min_s=np.asarray(min_s, np.float32), tau_s=np.asarray(tau_s, np.float32),
                esc_idx=np.asarray(esc_idx, np.int64), esc_val=np.asarray(esc_val, np.float32),
                esc_delta=np.asarray(esc_delta, np.float32))
    hdr = dict(Q=int(Q), P=int(P), T=int(enc['T']), K=int(enc['K']), N=int(enc['N']), PB=PB, cert=cert, arrays={})
    while True:                                            # offsets depend on the header length: until it is stable
        hj = json.dumps(hdr).encode(); pos = (12 + len(hj) + 63) // 64 * 64
        for k, a in arrs.items():
            hdr['arrays'][k] = [pos, a.dtype.str, list(a.shape)]; pos = (pos + a.nbytes + 63) // 64 * 64
        if json.dumps(hdr).encode() == hj:
            break
    buf = bytearray(MAGIC + np.uint32(VERSION).tobytes() + np.uint32(len(hj)).tobytes() + hj)
    for k, a in arrs.items():
        off = hdr['arrays'][k][0]; assert off >= len(buf); buf += b'\0' * (off - len(buf)); buf += a.tobytes()
    buf += b'\0' * ((len(buf) + 63) // 64 * 64 - len(buf))
    buf += np.uint32(zlib.crc32(buf)).tobytes()
    with open(path, 'wb') as f:
        f.write(buf)
    return len(buf)


def read(path: str | None = None, blob: bytes | None = None) -> dict:
    """Header fields + arrays (numpy views into the file bytes). Raises QVB8RFormatError on a bad file."""
    raw = blob if blob is not None else open(path, 'rb').read()
    where = path or '<blob>'
    if raw[:4] != MAGIC:
        raise QVB8RFormatError(f'{where}: not a .qvb8r file')
    ver = int(np.frombuffer(raw, np.uint32, 1, 4)[0])
    if ver != VERSION:
        raise QVB8RFormatError(f'{where}: .qvb8r version {ver}, this reader knows {VERSION}'
                               + (' (re-transcode the pack: python -m qv_band.formats.qvb8r.pack)' if ver < VERSION else ''))
    if zlib.crc32(memoryview(raw)[:-4]) != int(np.frombuffer(raw, np.uint32, 1, len(raw) - 4)[0]):
        raise QVB8RFormatError(f'{where}: crc32 mismatch (truncated or corrupt file)')
    hl = int(np.frombuffer(raw, np.uint32, 1, 8)[0]); hdr = json.loads(bytes(raw[12:12 + hl]))
    if hdr.get('PB') != PB:
        raise QVB8RFormatError(f'{where}: PB {hdr.get("PB")} != {PB}')
    out = dict(hdr)
    for k, (off, dt, shape) in hdr['arrays'].items():
        n = int(np.prod(shape)) if shape else 1
        out[k] = np.frombuffer(raw, np.dtype(dt), n, off).reshape(shape)
    return out


def blocks_for(tiles: np.ndarray, tiles_per_block: int) -> np.ndarray:
    """Block schedule: runs of <= tiles_per_block consecutive tiles of one class -> (class, first tile, ntiles)."""
    b = []; i = 0; n = len(tiles)
    while i < n:
        j = i
        while j < n and j - i < tiles_per_block and tiles[j][0] == tiles[i][0]:
            j += 1
        b.append((tiles[i][0], i, j - i)); i = j
    return np.array(b, np.int32).reshape(-1, 3)


def runtime_tables(d: dict):
    """(slot tables, tiles i64, padded words, tile offsets i64) -- what a reader builds once at load."""
    tab = slot_tables(d['freq'])
    words = np.concatenate([d['words'], np.zeros(PAD_WORDS, np.uint32)])
    return tab, d['tiles'].astype(np.int64), words, d['tile_off'].astype(np.int64)
