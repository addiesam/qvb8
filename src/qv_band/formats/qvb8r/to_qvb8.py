"""QVB8R table -> QVB8 table, LOSSLESS, at load time: so a QVB8R pack can be cut for Tensor Parallel.

QVB8R stores the same band ids as QVB8, rANS-coded in row TILES with permuted rows (rows sorted by statistics
class): a head's rows are spread over many tiles, so a share cannot be cut out of the coded stream. The ids
themselves are unchanged by the coding: the GPU decoder returns min_s + id * tau_s exactly (fp32), so
    id = round((value - min_s) / tau_s)
recovers them bit for bit (checked: re-decoding gives the same values), they are bit-packed at their row
width as a QVB8 table, and the existing byte-copy cuts (formats/qvb8/tp.py) apply. The cards then hold
their shares in QVB8 (13-22 % larger than QVB8R for those tables). A pack-time PM-aligned tiling (each head
range / column block coded as its own tile set) would keep the QVB8R size; see the design doc.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def qvb8r_to_qvb8(lin, check=True):
    """QVBandLinearV8R (on a GPU) -> QVBandLinearV8 on the CPU with the same values and escape tables.
    Streams the table in the module's own decode chunks (~CHUNK_BYTES each): pass 1 finds each row's width,
    pass 2 packs each chunk's rows straight into the host buffer -- peak GPU memory is one chunk, not the table
    (a 248k x 5120 output head would need 5 GB as one fp32 decode)."""
    from qv_band.formats.qvb8.module import QVBandLinearV8
    from qv_band.formats.qvb8r.module import _load_ext
    ext = _load_ext()
    Q, P = lin.out_features, lin.in_features
    dev = lin.min_s.device
    mn_all, ts_all = lin.min_s.float(), lin.tau_s.float()
    if ts_all.numel() != Q:
        ts_all = ts_all.expand(Q)

    def pieces():
        """(original rows, ids int64) in slices of ~32 MB of ids, chunk by chunk (bounded GPU memory)."""
        for c in lin.chunks:
            b0, nb, base, rows, o0, o1 = c[:6]
            W = ext.decode(*lin._core(lin.blocks8), P, 2, b0, nb, base, rows)      # fp32, sorted rows, no escapes
            step = max(1, (32 << 20) // (8 * P))
            for a in range(0, rows, step):
                Ws = W[a:a + step]
                orow = lin.chunk_orows[o0 + a:o0 + a + Ws.shape[0]]
                mn, ts = mn_all[orow].unsqueeze(1), ts_all[orow].unsqueeze(1)
                ids = torch.round((Ws - mn) / ts)
                if check:
                    err = (torch.addcmul(mn, ids, ts) - Ws).abs().max().item()   # the kernel decodes with an fma
                    if err > 0:
                        raise ValueError(f"qvb8r ids not recovered exactly (max diff {err:g})")
                yield orow, ids.to(torch.int64)
            del W

    mx = torch.zeros(Q, dtype=torch.int64, device=dev)
    for orow, ids in pieces():
        mx[orow] = ids.amax(1)
    nbits_t = torch.ceil(torch.log2(mx.clamp(min=1).double() + 1)).to(torch.int64).clamp(min=1)
    if P % 8:
        raise ValueError("qvb8r -> qvb8 needs P % 8 == 0")
    row_bytes = nbits_t * P // 8
    byte_off = torch.zeros(Q + 1, dtype=torch.int64, device=dev)
    byte_off[1:] = torch.cumsum(row_bytes, 0)
    buf = torch.zeros(int(byte_off[-1]) + 8, dtype=torch.uint8)                     # host
    for orow, ids in pieces():
        dst, by = _pack_rows_gpu(ids, nbits_t[orow], P, byte_off[orow])
        buf[dst.cpu()] = by.cpu()
    nbits = nbits_t.cpu().numpy().astype(np.uint8)
    return QVBandLinearV8(min_s=lin.min_s.float().cpu(), nbits=torch.from_numpy(nbits),
                          bit_off=(byte_off[:-1] * 8).cpu(), ids=buf,
                          Q=Q, P=P, tau_s=lin.tau_s.float().cpu(),
                          esc_idx=lin.esc_flat.cpu(), esc_vals=lin.esc_val.float().cpu(),
                          esc_idx_bf16=lin.esc_flat_bf16.cpu(), esc_vals_bf16=lin.esc_val_bf16.float().cpu())


def _pack_rows_gpu(ids, nbits, P, row_off):
    """QVB8 bit layout for a set of rows: ids [r, P] at their row widths nbits [r], row r starting at byte
    row_off[r] (rows are byte-aligned when P % 8 == 0). Returns (byte positions, bytes) to scatter into the
    table's buffer -- the same bytes as decode.pack_bits_rows."""
    weights = (1 << torch.arange(8, device=ids.device, dtype=torch.int64))
    dsts, bys = [], []
    for w in torch.unique(nbits).tolist():
        sel = torch.nonzero(nbits == w).flatten()
        sh = torch.arange(w, device=ids.device, dtype=torch.int64)
        rpc = max(1, (64 << 20) // (P * w * 8))                                  # ~64 MB of int64 temporaries
        for a in range(0, sel.numel(), rpc):
            rr = sel[a:a + rpc]
            bits = (ids[rr].unsqueeze(-1) >> sh) & 1                             # [r, P, w], LSB first
            by = (bits.reshape(rr.numel(), -1, 8) * weights).sum(-1).to(torch.uint8)
            dsts.append((row_off[rr].unsqueeze(1) + torch.arange(by.shape[1], device=ids.device)).flatten())
            bys.append(by.flatten())
    return torch.cat(dsts), torch.cat(bys)


def load_linear_any(path, device_for_decode):
    """A pack table as a CPU QVBandLinearV8, whichever of the two formats the file is."""
    from qv_band.formats.qvb8.module import QVBandLinearV8
    if path.endswith(".qvb8r"):
        from qv_band.formats.qvb8r.module import QVBandLinearV8R
        lin = QVBandLinearV8R.from_packfile(path, device=device_for_decode)
        out = qvb8r_to_qvb8(lin)
        del lin
        return out
    return QVBandLinearV8.from_packfile(path, device="cpu")
