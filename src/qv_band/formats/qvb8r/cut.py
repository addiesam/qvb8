"""Cut a .qvb8r table IN PLACE: any set of output rows x one input-column range, no re-encoding.

The rANS stream is read up to the cut: the decoder's state and word position at a cut point are a valid start for
the rest of the stream (the walk is deterministic). So a share is
  * for each run of the share's rows that sits consecutively inside one tile: a new tile whose start states are the
    decoder state at (first row, column c0) and whose words are the original words up to (last row, column c1);
  * with a column cut every row is its own tile (between two rows the stream carries the other cards' columns);
  * the same frequency tables, classes, drow, min_s, tau_s and escapes (exact values and deltas unchanged: the ids
    and the band are identical), re-indexed.
The ids a share decodes are bit-identical to the same rows / columns of the whole table. Cost: one CPU walk over
the touched part of the stream (no symbol written) and 32 start states per new tile.

Limits: c0 a multiple of 64 (a symbol pair: lane j holds columns 32 s + j, renormalised once per pair) and the share
width a multiple of 128 (every kernel's P). Anything else returns (None, reason) and the caller falls back.
Used by every Tensor Parallel loader (LLM run/tp_qwen35.py, DiT run/tp_dit.py, ...) through
QVBandLinearV8R.from_packfile(path, rows=..., cols=...)."""
from __future__ import annotations

import numpy as np
import torch

PAIR = 64          # columns per symbol pair


def _ranges(rows, Q):
    if rows is None:
        return np.arange(Q, dtype=np.int64)
    if isinstance(rows, tuple) and len(rows) == 2 and np.isscalar(rows[0]):
        rows = [rows]
    return np.concatenate([np.arange(int(a), int(b), dtype=np.int64) for a, b in rows])


def can_cut(P: int, cols) -> str | None:
    """None when (c0, c1) is a valid in-place column cut of width-P rows, else the reason."""
    if cols is None:
        return None
    c0, c1 = int(cols[0]), int(cols[1])
    if not (0 <= c0 < c1 <= P):
        return f"columns {c0}..{c1} outside 0..{P}"
    if c0 % PAIR:
        return f"column start {c0} not a multiple of {PAIR}"
    if (c1 - c0) % 128:
        return f"share width {c1 - c0} not a multiple of 128"
    return None


def cut_dict(d: dict, rows=None, cols=None):
    """d: a .qvb8r file dict (container.read). rows: list of (r0, r1) output-row ranges (share row order = this order)
    or None; cols: (c0, c1) or None. -> (share dict, keep mask over d['esc_idx']) or (None, reason)."""
    from .module import _load_ext
    Q, P, T = int(d["Q"]), int(d["P"]), int(d["T"])
    why = can_cut(P, cols)
    if why:
        return None, why
    c0, c1 = (0, P) if cols is None else (int(cols[0]), int(cols[1]))
    full_cols = (c0, c1) == (0, P)
    sel = _ranges(rows, Q)
    Qs, Ps = len(sel), c1 - c0
    if Qs == 0:
        return None, "no rows"
    perm = d["perm"].astype(np.int64)
    tiles = d["tiles"].astype(np.int64)
    off = d["tile_off"].astype(np.int64)
    states = np.ascontiguousarray(d["states"]).view(np.uint64).reshape(-1, 32)
    words = np.ascontiguousarray(d["words"]).view(np.uint32)
    inv = np.empty(Q, np.int64); inv[perm] = np.arange(Q)
    row_tile = np.repeat(np.arange(len(tiles)), tiles[:, 2])            # sorted row -> tile
    S2 = P // PAIR

    sp = inv[sel]                                     # each share row's position in the file's sorted order
    order = np.argsort(sp, kind="stable")             # share sorted row -> share row (keeps the class order)
    sps = sp[order]
    t_of = row_tile[sps]; k_of = sps - tiles[t_of, 1]

    # segments: (tile, first k, rows) -- runs of consecutive rows of one tile (<= T), every row alone on a column cut
    m = len(sps)
    brk = np.ones(m, bool)
    if full_cols and m > 1:
        brk[1:] = (t_of[1:] != t_of[:-1]) | (k_of[1:] != k_of[:-1] + 1)
        run0 = np.maximum.accumulate(np.where(brk, np.arange(m), 0))
        brk |= (np.arange(m) - run0) % T == 0
    starts = np.flatnonzero(brk)
    segs = np.stack([t_of[starts], k_of[starts], np.diff(np.append(starts, m))], 1).astype(np.int64)
    st_tile, st_pair = segs[:, 0], segs[:, 1] * S2 + c0 // PAIR
    en_pair = (segs[:, 1] + segs[:, 2] - 1) * S2 + c1 // PAIR
    whole_end = en_pair == tiles[st_tile, 2] * S2

    # walk only to the points the file does not already hold (a tile's start, a tile's end)
    si = np.flatnonzero(st_pair != 0); ei_ = np.flatnonzero(~whole_end)
    seg_of = np.concatenate([si, ei_]); which_of = np.concatenate([np.zeros(len(si), np.int8), np.ones(len(ei_), np.int8)])
    stop_t = st_tile[seg_of]; stop_p = np.concatenate([st_pair[si], en_pair[ei_]])
    new_states = states[st_tile].copy()
    w0, w1 = off[st_tile].copy(), off[st_tile + 1].copy()
    if len(stop_t):
        o = np.lexsort((stop_p, stop_t))
        tab = _tab(d)
        cs, cp = _load_ext().cut_points(torch.from_numpy(words.view(np.int32)), torch.from_numpy(off),
                                        torch.from_numpy(states.view(np.int64).reshape(-1)), torch.from_numpy(tiles),
                                        torch.from_numpy(tab.view(np.int32)), torch.from_numpy(stop_t[o]),
                                        torch.from_numpy(stop_p[o]))
        cs = cs.numpy().view(np.uint64); cp = cp.numpy()
        seg, wh = seg_of[o], which_of[o]
        a, b = wh == 0, wh == 1
        new_states[seg[a]] = cs[a]; w0[seg[a]] = cp[a]; w1[seg[b]] = cp[b]

    n = w1 - w0
    new_off = np.concatenate([[0], np.cumsum(n)])
    if new_off[-1] >= 2 ** 32:
        return None, "word stream > 2^32"
    # every segment's words, in order: copied run by run (adjacent segments merged into one slice) -- no index array,
    # which would cost 8 bytes per 4-byte word in host RAM on the loader threads
    new_words = np.empty(int(new_off[-1]), np.uint32)
    if len(n):
        brk = np.flatnonzero(w0[1:] != w1[:-1]) + 1                # where the next segment does not continue the last
        starts = np.concatenate([[0], brk]); ends = np.concatenate([brk, [len(n)]])
        for a, b in zip(starts.tolist(), ends.tolist()):
            new_words[new_off[a]:new_off[b]] = words[w0[a]:w1[b - 1]]
    first = np.concatenate([[0], np.cumsum(segs[:, 2])[:-1]])
    new_tiles = np.stack([tiles[st_tile, 0], first, segs[:, 2]], 1)

    tau = np.asarray(d["tau_s"], np.float32).reshape(-1)
    ei = np.asarray(d["esc_idx"], np.int64)
    er, ec = ei // P, ei % P
    pos = np.full(Q, -1, np.int64); pos[sel] = np.arange(Qs)
    keep = (pos[er] >= 0) & (ec >= c0) & (ec < c1)
    out = {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
    out.update(Q=Qs, P=Ps, T=int(segs[:, 2].max()),
               perm=order.astype(np.int32), freq=d["freq"], tiles=new_tiles.astype(np.int32),
               tile_off=new_off.astype(np.uint32), states=new_states, words=new_words,
               drow=np.asarray(d["drow"])[sel], min_s=np.asarray(d["min_s"], np.float32)[sel],
               tau_s=tau[sel] if tau.size == Q else np.full(Qs, tau[0], np.float32),
               esc_idx=pos[er[keep]] * Ps + (ec[keep] - c0),
               esc_val=np.asarray(d["esc_val"], np.float32)[keep],
               esc_delta=np.asarray(d["esc_delta"], np.float32)[keep])
    return out, keep


def _tab(d):
    from .codec import slot_tables
    return slot_tables(d["freq"])
