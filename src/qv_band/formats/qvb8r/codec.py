"""codec: QVB8 pack reader (release build -- the pack writer / calibration is not included)."""
from __future__ import annotations
import numpy as np
PB = 12
MQ = 1 << PB
MAX_ID = 2048
MAX_FREQ = 1024
CENTRE = 256

def quant(c: np.ndarray) -> np.ndarray:
    """Counts -> frequencies summing to MQ; every present symbol gets >= 1 and none more than MAX_FREQ (the slot
    entry's limit). A class too peaked for that (centred near-constant rows, e.g. unused embedding rows, all land on
    CENTRE) gets the excess spread over the other symbols -- and their neighbours when fewer than MQ / MAX_FREQ are
    present -- at a few bits' cost for that class instead of the whole tensor staying QVB8."""
    c = np.asarray(c, np.float64)
    present = c > 0
    if len(c) * MAX_FREQ < MQ:
        raise ValueError(f'quant: {len(c)} symbols cannot hold {MQ} slots at <= {MAX_FREQ} each')
    while present.sum() * MAX_FREQ < MQ:
        grown = present.copy()
        grown[1:] |= present[:-1]
        grown[:-1] |= present[1:]
        present = grown
    k = int(present.sum())
    budget = MQ - k
    room = np.where(present, MAX_FREQ - 1, 0).astype(np.float64)
    raw = np.zeros_like(c)
    free = present & (c > 0)
    while True:
        tot = c[free].sum()
        left = budget - raw[~free & present].sum()
        r = np.where(free, c / tot * left if tot > 0 else 0, raw)
        over = free & (r > room)
        if not over.any():
            raw = r
            break
        raw[over] = room[over]
        free &= ~over
        if not free.any():
            spare = budget - raw.sum()
            z = present & (c == 0) & (raw < room)
            raw[z] += spare / z.sum()
            break
    f = np.where(present, np.floor(raw).astype(np.int64) + 1, 0)
    D = MQ - int(f.sum())
    frac = np.where(present & (f < MAX_FREQ), raw - np.floor(raw), -1)
    f[np.argsort(-frac)[:D]] += 1
    assert f.sum() == MQ and f.max() <= MAX_FREQ and (f[c > 0] >= 1).all()
    return f

def _classes(sd, K):
    return np.searchsorted(np.quantile(sd, np.linspace(0, 1, K + 1)[1:-1]), sd)

def pick_K(H: np.ndarray, sd: np.ndarray) -> int:
    """Number of row classes minimising coded bits + table bits. H: (R, N) per-row symbol counts, sd: row std."""
    R = len(sd)
    best = None
    for K in (1, 2, 4, 8, 16, 32, 64):
        if K > 1 and R < 64 * K:
            break
        cls = _classes(sd, K)
        bits = 0.0
        for k in range(K):
            c = H[cls == k].sum(0)
            if c.sum() == 0:
                continue
            f = quant(c)
            bits += -(c[c > 0] * np.log2(f[c > 0] / MQ)).sum()
        cost = bits + K * MQ * 32
        if best is None or cost < best[0]:
            best = (cost, K)
    return best[1]

def slot_tables(F: np.ndarray) -> np.ndarray:
    """(K, N) frequencies -> (K, 4096) packed slot entries."""
    F = F.astype(np.int64)
    K, N = F.shape
    C = np.concatenate([np.zeros((K, 1), np.int64), np.cumsum(F, 1)[:, :-1]], 1)
    tab = np.zeros((K, MQ), np.uint32)
    for k in range(K):
        s = np.repeat(np.arange(N), F[k])
        d = np.arange(MQ) - C[k][s]
        tab[k] = s | d << 11 | F[k][s] - 1 << 22
    return tab

def _layout(H, sd, K, T):
    """Classes, row order, quantised tables and tiles; None when the slot entry cannot hold the tensor."""
    R, N = H.shape
    cls = _classes(sd, K)
    perm = np.argsort(cls, kind='stable')
    cls_s = cls[perm]
    F = np.zeros((K, N), np.int64)
    for k in range(K):
        F[k] = quant(H[cls == k].sum(0) if (cls == k).any() else np.ones(N, np.int64))
    if F.max() > MAX_FREQ or N > MAX_ID:
        return None
    C = np.concatenate([np.zeros((K, 1), np.int64), np.cumsum(F, 1)[:, :-1]], 1)
    tiles = []
    starts = np.searchsorted(cls_s, np.arange(K))
    ends = np.searchsorted(cls_s, np.arange(K), 'right')
    for k in range(K):
        for r0 in range(starts[k], ends[k], T):
            tiles.append((k, r0, min(T, ends[k] - r0)))
    return (perm, F, C, np.array(tiles, np.int64))

def centre_rows(ids: np.ndarray) -> np.ndarray:
    """drow (Q,) int64: row mean on CENTRE, clipped so every symbol id - drow of the row is >= 0 and, when the row
    span allows, < MAX_ID."""
    lo, hi = (ids.min(1).astype(np.int64), ids.max(1).astype(np.int64))
    return np.clip(np.rint(ids.mean(1, dtype=np.float64)).astype(np.int64) - CENTRE, hi - (MAX_ID - 1), lo)
ROWS_PER_TILE_DIV = 4096

def rows_per_tile(Q: int) -> int:
    """Rows a warp decodes in sequence (one rANS tile). Each tile stores 32 start states (2048 bit): 2048 / (T * P)
    bit per id. Fewer tiles = fewer warps to keep the GPU busy; >= 4096 tiles is ~1.5 waves on a V100 (~2700
    resident warps) and ~2.4 on the dev GPU. Measured on the dev GPU (T 1 -> 2..4 on 10k..17k-row tensors): M=1 / M=8 within 3%,
    the M=9..16 WMMA path 2..14% slower, ids unchanged. The one place this is decided."""
    return max(1, min(8, Q // ROWS_PER_TILE_DIV))

def encode_tensor(ids: np.ndarray, T: int | None=None):
    """(Q, P) int ids -> (dict of arrays, None), or (None, reason) when the tensor must stay QVB8.
    T: rows per tile (default rows_per_tile(Q)); the reader takes it from the file, so any value decodes."""
    import torch
    from .module import _load_ext
    Q, P = ids.shape
    if P % 128:
        return (None, 'P % 128')
    drow = centre_rows(ids)
    if (ids.max(1) - drow >= MAX_ID).any():
        return (None, f'row span >= {MAX_ID}')
    sym = np.ascontiguousarray((ids - drow[:, None]).astype(np.int16))
    N = int(sym.max()) + 1
    H = _load_ext().row_hist(torch.from_numpy(sym), N).numpy().astype(np.int64)
    a = np.arange(N, dtype=np.float64)
    m1 = (H * a).sum(1) / P
    sd = np.sqrt(np.maximum((H * a * a).sum(1) / P - m1 * m1, 0))
    T = rows_per_tile(Q) if T is None else int(T)
    K = pick_K(H, sd)
    lay = _layout(H, sd, K, T)
    if lay is None:
        return (None, 'slot table limits')
    perm, F, C, tiles = lay
    words, off, states = _load_ext().encode(torch.from_numpy(np.ascontiguousarray(sym[perm])), torch.from_numpy(tiles), torch.from_numpy(F), torch.from_numpy(C))
    off = off.numpy()
    if off[-1] >= 2 ** 32:
        return (None, 'word stream > 2^32')
    return (dict(T=T, K=K, N=N, perm=perm.astype(np.int32), freq=F.astype(np.uint16), tiles=tiles.astype(np.int32), tile_off=off.astype(np.uint32), states=states.numpy().view(np.uint64), words=words.numpy().view(np.uint32), drow=drow.astype(np.int16)), None)
