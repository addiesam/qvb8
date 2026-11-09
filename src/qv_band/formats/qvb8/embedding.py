"""nn.Embedding backed by a QVB8 per-row virtual-band pack.

An embedding lookup is `W[token_id]`, so unlike the GEMV there is no
reduction: decoding row r is just

    value[c] = min_s[r] + id[r][c] * tau_s

for that one row. That makes this far simpler than the linear kernel --
and worth having rather than streaming the tensor dense, because
embed_tokens is 248320 x 5120: 0.97 GiB packed against 2.37 GiB in bf16.
Excluding it would hand back more than the whole codec saves.

"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class QVBandEmbeddingV8(nn.Module):
    """Embedding whose rows are decoded on demand from a QVB8 pack."""

    def _apply(self, fn, recurse=True):
        # model.to(torch.bfloat16) must not round min_s / tau_s to bf16 (see apply_keeping_fp32_metadata)
        from .module import apply_keeping_fp32_metadata
        return apply_keeping_fp32_metadata(self, fn, recurse)

    def __init__(self, *, min_s, nbits, bit_off, ids, Q, P, tau_s,
                 esc_idx, esc_vals, out_dtype=torch.bfloat16):
        super().__init__()
        self.num_embeddings = int(Q)
        self.embedding_dim = int(P)
        self.register_buffer("tau_s", tau_s, persistent=False)
        self._out_dtype = out_dtype
        self.register_buffer("min_s", min_s, persistent=False)
        self.register_buffer("nbits", nbits, persistent=False)
        self.register_buffer("bit_off", bit_off, persistent=False)
        self.register_buffer("ids", ids, persistent=False)
        # Escapes are held as a dense (position -> value) map only for the
        # rows that have any; embeddings on this model have none, and the
        # lookup stays a plain gather when the map is empty.
        self.register_buffer("esc_idx", esc_idx, persistent=False)
        self.register_buffer("esc_vals", esc_vals, persistent=False)

    @classmethod
    def from_packfile(cls, path: str, *, out_dtype=torch.bfloat16,
                      tau: float | None = None, device: str = "cuda",
                      blob: bytes | None = None):
        if blob is None:
            with open(path, "rb") as fh:
                blob = fh.read()
        if blob[:4] == b"QVBD":
            from qv_band.formats.qvb8.pack_container import calc_method_of, _CALC_TABLE
            if calc_method_of(blob) == _CALC_TABLE:
                raise NotImplementedError(
                    "qvb8 embedding pack is a table (calc_method=1) pack -- only "
                    "the formula has an embedding reader; stream this tensor "
                    "dense instead.")
            from qv_band.formats.qvb8.pack_container import unpack_formula
            from qv_band.formats.qvb8.escape_split import fp16_list_for
            d = unpack_formula(blob, skip_bit_decode=True, fp16_list=fp16_list_for(path), where=str(path))
            min_s, tau_s, nb, bit_off, buf = (
                d["min_s"], d["tau_s"], d["nbits"], d["bit_off"], d["buf"])
            Q, P = d["Q"], d["P"]
            # The output dtype is fixed here, so one table is kept: the bf16 table for bf16 output, else the fp16 one
            # (formats/qvb8/escape_split.py).
            # bf16 output: the bf16 table. fp16 output: the fp16 table, from escape_tables.json (none -> error).
            # fp32 output: the full list (exact values; no rounding of the weight).
            if out_dtype == torch.float16:
                if d["esc_idx_fp16"] is None:
                    from qv_band.formats.qvb8.escape_split import fp16_missing_error
                    raise fp16_missing_error(str(path))
                esc_idx, esc_vals = d["esc_idx_fp16"], d["esc_vals_fp16"]
            else:
                esc_idx, esc_vals = d["esc_idx"], d["esc_vals"]
        else:
            from qv_band.formats.qvb8.pack import unpack_tensor
            (min_s, tau_s, nb, bit_off, buf, Q, P, _tau,
             esc_idx, esc_vals) = unpack_tensor(blob)
        dev = torch.device(device)
        return cls(
            min_s=torch.from_numpy(min_s).to(dev),
            nbits=torch.from_numpy(nb.astype(np.int64)).to(dev),
            bit_off=torch.from_numpy(bit_off.astype(np.int64)).to(dev),
            ids=torch.from_numpy(buf).to(dev),
            Q=Q, P=P, tau_s=torch.from_numpy(tau_s).to(dev),
            esc_idx=torch.from_numpy(esc_idx.astype(np.int64)).to(dev),
            esc_vals=torch.from_numpy(esc_vals.astype(np.float32)).to(dev),
            out_dtype=out_dtype)

    def _decode_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Decode the given row indices to (n, P) fp32, on device.

        Rows may have DIFFERENT widths, so the bit positions are built per
        row from its own nbits and bit_off rather than from a global
        stride. A field spans at most 3 bytes, so 4 byte-gathers and one
        shift/mask lift it whatever its width -- the same trick the CUDA
        kernel uses, expressed in torch so this needs no second kernel.
        """
        dev = self.ids.device
        n = rows.numel()
        P = self.embedding_dim
        nb = self.nbits[rows].view(n, 1)                  # (n, 1)
        base = self.bit_off[rows].view(n, 1)              # (n, 1)
        cols = torch.arange(P, device=dev, dtype=torch.int64).view(1, P)
        bit = base + cols * nb                            # (n, P)
        byi = bit >> 3
        sh = bit & 7
        acc = self.ids[byi].to(torch.int64)
        for k in range(1, 4):
            acc |= self.ids[byi + k].to(torch.int64) << (8 * k)
        ids = (acc >> sh) & ((torch.ones_like(nb) << nb) - 1)
        # tau_s is per row, like min_s: index it by the SAME rows and
        # broadcast over columns. Treating it as a scalar here left the
        # whole (Q,) array in the expression and blew up on the first
        # embedding lookup.
        return (self.min_s[rows].view(n, 1)
                + ids.to(torch.float32) * self.tau_s[rows].view(n, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dev_in = input.device                 # table may live in host RAM
        flat = input.reshape(-1).to(self.ids.device)
        out = self._decode_rows(flat)
        if self.esc_idx.numel():
            # Escapes are flat (row*P + col) positions; only those falling
            # in the requested rows apply.
            P = self.embedding_dim
            esc_row = self.esc_idx // P
            esc_col = self.esc_idx - esc_row * P
            for i, r in enumerate(flat.tolist()):
                m = esc_row == r
                if bool(m.any()):
                    # Cast to the destination's dtype. `out` comes from
                    # _decode_rows, which is fp32 by construction, but
                    # esc_vals is a registered BUFFER and so a module-wide
                    # .to(torch.bfloat16) -- which imagegen does to the whole
                    # text encoder after the QV-band swap -- silently narrowed
                    # it too. index_put_ then refuses the mixed dtypes.
                    #
                    # This never showed up on the Z-Image tau=1e-3 pack
                    # because that embedding has NO escape values (esc_idx
                    # empty, so the loop is skipped); the tau=1e-4 pack has
                    # 19658 of them and failed on the first forward with
                    #   "Index put requires the source and destination dtypes
                    #    match, got Float ... and BFloat16 ..."
                    # So the trigger is the PACK, not the venv or the torch:
                    # any embedding with escapes hits this.
                    out[i, esc_col[m]] = self.esc_vals[m].to(out.dtype)
        return out.to(self._out_dtype).reshape(*input.shape, self.embedding_dim).to(dev_in)

    @property
    def weight(self):
        """Dense (Q, P) view, for callers that need the whole table."""
        rows = torch.arange(self.num_embeddings, device=self.ids.device)
        return self._decode_rows(rows).to(self._out_dtype)
