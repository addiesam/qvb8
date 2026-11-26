"""Cut a QVB8R table for Tensor Parallel and KEEP IT QVB8R (rANS size on the card).

A module loaded from a file is cut IN PLACE from the stream (cut.py: decoder states at the cut, no re-encoding).
The re-encode below is only the fallback (no file, or a cut cut.py refuses).

The rANS stream cannot be cut in place (tiles hold rows permuted by statistics class). So each card, at load:
  1. decodes the table on the GPU chunk by chunk and recovers the band ids exactly (min_s + id * tau_s, fma);
  2. keeps only its share: output rows (column-parallel) or input columns (row-parallel);
  3. re-encodes that share with the production rANS encoder (codec.encode_tensor: its own classes, tiles,
     frequency tables) -- lossless, the same ids;
  4. builds a QVBandLinearV8R from it in memory (nothing written), with the share's min_s / tau_s and the
     escapes that fall inside it (esc_delta = exact - fp32 band value, as the container defines it).
A share the encoder refuses (e.g. a column cut not a multiple of 128) falls back to a QVB8 share.
"""
from __future__ import annotations

import os

import numpy as np
import torch

# V100, 27B MLP shares: T=1 -> M=8 0.332 ms vs 0.364 (T=2, the pack default), +5 % bytes
SHARE_ROWS_PER_TILE = int(os.environ.get("QV_QVB8R_SHARE_T", "1"))


@torch.no_grad()
def _share_ids(lin, rows=None, cols=None):
    """ids (np.int32) of the share, plus the share's original row indices. rows: list of (r0, r1); cols: (c0, c1)."""
    from qv_band.formats.qvb8r.module import _load_ext
    ext = _load_ext()
    Q, P = lin.out_features, lin.in_features
    dev = lin.min_s.device
    if rows is not None:
        sel = torch.cat([torch.arange(a, b, device=dev) for a, b in rows])
    else:
        sel = torch.arange(Q, device=dev)
    c0, c1 = cols if cols is not None else (0, P)
    where = torch.full((Q,), -1, dtype=torch.int64, device=dev)            # original row -> share row
    where[sel] = torch.arange(sel.numel(), device=dev)
    out = np.empty((sel.numel(), c1 - c0), np.int32)
    mn_all = lin.min_s.float()
    ts_all = lin.tau_s.float() if lin.tau_s.numel() == Q else lin.tau_s.float().expand(Q)
    step = max(1, (32 << 20) // (8 * P))
    for c in lin.chunks:
        b0, nb, base, nrows, o0, o1 = c[:6]
        W = ext.decode(*lin._core(lin.blocks8), P, 2, b0, nb, base, nrows)    # fp32, sorted rows, no escapes
        for a in range(0, nrows, step):
            Ws = W[a:a + step]
            orow = lin.chunk_orows[o0 + a:o0 + a + Ws.shape[0]]
            dst = where[orow]
            keep = dst >= 0
            if not bool(keep.any()):
                continue
            Ws, orow, dst = Ws[keep], orow[keep], dst[keep]
            mn, ts = mn_all[orow].unsqueeze(1), ts_all[orow].unsqueeze(1)
            ids = torch.round((Ws - mn) / ts)
            if (torch.addcmul(mn, ids, ts) - Ws).abs().max().item() > 0:
                raise ValueError("qvb8r ids not recovered exactly")
            out[dst.cpu().numpy()] = ids[:, c0:c1].to(torch.int32).cpu().numpy()
        del W
    return out, sel


def _row_bias(lin, rows, device):
    """The share's bias: the bias of its output rows (any cut keeps every row's bias whole only when rows are cut;
    a column cut's bias belongs after the all-reduce, so the caller adds it there and the share gets none)."""
    if lin.bias is None:
        return None
    b = lin.bias.detach()
    if rows is not None:
        b = torch.cat([b[a:c] for a, c in rows])
    return b.to("cpu" if device == "ref" else device)


def slice_qvb8r(lin, rows=None, cols=None, device="cuda"):
    """QVBandLinearV8R share (rows=[(r0, r1), ...] output rows, or cols=(c0, c1) input columns), cut in place from its
    file (cut.py) or else re-encoded. Row cuts carry their bias rows; column cuts carry no bias (the caller's)."""
    from qv_band.formats.qvb8r.codec import encode_tensor
    from qv_band.formats.qvb8r.module import QVBandLinearV8R
    Q, P = lin.out_features, lin.in_features
    if rows is None and cols is None:
        return lin.to(device), None
    path = getattr(lin, "path", None)
    if path and os.path.exists(path):
        # in place from the file's stream (cut.py): no decode to ids, no re-encode -- every model family
        from qv_band.formats.qvb8r.cut import can_cut
        why = can_cut(P, cols)
        if why is None:
            sh = QVBandLinearV8R.from_packfile(path, device=device, rows=None if rows is None else list(rows), cols=cols,
                                               bias=_row_bias(lin, rows, device) if cols is None else None)
            sh.bf16_gemm = lin.bf16_gemm
            return sh, None
    if getattr(lin, "on_disk", False):                       # a header-only module: the fallback needs the table
        lin = QVBandLinearV8R.from_packfile(lin.path, device=device)
    ids, sel = _share_ids(lin, rows, cols)
    c0, c1 = cols if cols is not None else (0, P)
    Qs, Ps = ids.shape
    sel_np = sel.cpu().numpy()
    min_s = lin.min_s.float().cpu().numpy()[sel_np]
    ts_full = lin.tau_s.float().cpu().numpy()
    tau_s = ts_full[sel_np] if ts_full.size == Q else np.full(Qs, ts_full.reshape(-1)[0], np.float32)
    # escapes inside the share (the fp16 table: what the fp16 paths use)
    ef = lin.esc_flat.cpu().numpy().astype(np.int64)
    ev = lin.esc_val.float().cpu().numpy()
    er, ec = ef // P, ef % P
    pos = np.full(Q, -1, np.int64)
    pos[sel_np] = np.arange(Qs)
    m = (pos[er] >= 0) & (ec >= c0) & (ec < c1)
    nr, nc = pos[er[m]], ec[m] - c0
    band = (np.float32(1) * min_s[nr] + ids[nr, nc].astype(np.float32) * tau_s[nr]).astype(np.float32)
    band = torch.addcmul(torch.from_numpy(min_s[nr]), torch.from_numpy(ids[nr, nc].astype(np.float32)),
                         torch.from_numpy(tau_s[nr])).numpy()                  # the decoder's fma
    esc_idx, esc_val = nr * Ps + nc, ev[m]
    o = np.argsort(esc_idx, kind="stable")
    esc_idx, esc_val, band = esc_idx[o], esc_val[o], band[o]
    enc, why = encode_tensor(ids, T=SHARE_ROWS_PER_TILE)   # 1 row per tile: more warps decode in parallel
    if enc is None:
        return None, why
    d = dict(enc, Q=Qs, P=Ps, min_s=min_s.astype(np.float32), tau_s=tau_s.astype(np.float32),
             esc_idx=esc_idx, esc_val=esc_val.astype(np.float32), esc_delta=(esc_val - band).astype(np.float32),
             cert={})
    return QVBandLinearV8R(d, device=device, bias=_row_bias(lin, rows, device) if cols is None else None), None
