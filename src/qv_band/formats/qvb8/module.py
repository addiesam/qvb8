"""nn.Linear backed by a QVB8 per-row virtual-band pack.

    lin = QVBandLinearV8.from_packfile(path, device="cuda")
    y = lin(x)            # x: (..., P) -> (..., Q)

Its own loader and its own CUDA extension (backends/cuda_qvb8).
"""
from __future__ import annotations

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F




def _load_ext():
    """The QVB8 CUDA extension for the current GPU, built on first use (common/jit_ext.py).

    No setup step and no prebuilt .so: the build is keyed on the source, the GPU
    arch and this interpreter's torch, so a kernel change, another machine or
    another venv can never pick up a stale or ABI-mismatched binary (both used
    to happen: an unrebuilt .cu gave an all-black image, a .so from another venv
    failed with an undefined c10_cuda_check_implementation symbol). A missing
    toolkit raises here rather than silently falling back to a dense
    reconstruction, whose memory cost is the opposite of the point.
    """
    import os
    from qv_band.common.jit_ext import load_cuda_ext
    here = os.path.join(os.path.dirname(__file__), "backends", "cuda_qvb8")
    # per current card (jit_ext caches per device): one global copy would pin the first card's arch
    return load_cuda_ext("qvb8_cuda", [os.path.join(here, "qvb8_cuda.cu")],
                         cflags=["-O3"], cuda_cflags=["-O3", "--use_fast_math"])



FP32_METADATA = ("min_s", "tau_s", "esc_vals", "esc_vals_bf16")


def apply_keeping_fp32_metadata(module: nn.Module, fn, recurse=True):
    """nn.Module._apply with the fp32 decode metadata of `module` protected: taken out while `fn` runs (dtype casts, device moves),
    put back as the SAME values on whatever device the packed ids ended up on."""
    keep = {n: module._buffers[n] for n in FP32_METADATA if module._buffers.get(n) is not None}
    for n in keep:
        module._buffers[n] = None
    try:
        out = nn.Module._apply(module, fn, recurse)
    finally:
        ref = module._buffers.get("ids")
        dev = ref.device if ref is not None else (next(iter(keep.values())).device if keep else None)
        for n, b in keep.items():
            module._buffers[n] = b.to(dev)
    return out


class QVBandLinearV8(nn.Module):
    # Opt-in, per module: a bf16 caller with M > 8 decodes the band straight
    # to bf16 and runs ONE bf16 GEMM (bias fused in the cuBLASLt epilogue,
    # bf16 out). Set by loaders of models whose native precision is bf16
    # (diffusion transformers). Off by default so every other caller (LLM
    # prefill, Sonic) keeps the certified fp16 path unchanged.
    bf16_gemm = False
    # True only when the pack certifies |orig - bf16(band)| <= tau (header
    # restore code 3, `--restore bf16`). The bf16 GEMM runs ONLY then: on an
    # fp16-only pack it would put a few hundred weights at up to ~1.04 tau,
    # outside the certificate (tech_doc/restore_precision_policy.md, rule 1).
    bf16_certified = False
    # True for an fp16+bf16 pack loaded without escape_tables.json: an fp16 forward raises (no fp16 table)
    fp16_table_missing = False
    """y = x @ W^T (+ bias), W resident in VRAM ONLY as the bit-packed
    virtual-band index (+ min_s/tau_s/escapes) -- never a decoded weight.
    The fp32 value min_s + idx*tau_s is narrowed to fp16 (the one restore
    precision this format defines) and multiplied in the same kernel on every
    forward call, so nothing dense is cached in between. The certified bound is
    |fp16(min_s + idx*tau_s) - w| <= tau for every element, escapes included.

    Two escape tables (formats/qvb8/escape_split.py):
      esc_idx / esc_vals            the fp16 table -- every fp16 / fp32 path (GEMV, fp16 decode, experts)
      esc_idx_bf16 / esc_vals_bf16  the bf16 table -- only the paths that round the weight to bf16
                                    (mma with bf16 activations, decode_bf16). The same tensors when the pack
                                    has no bf16-only escapes.
    """

    def __init__(self, *, min_s, nbits, bit_off, ids, Q, P, tau_s,
                 esc_idx, esc_vals, bias=None, esc_idx_bf16=None, esc_vals_bf16=None):
        super().__init__()
        self.in_features = int(P)
        self.out_features = int(Q)
        # Per-row bucket step: each row's is sized against its own peak
        # magnitude so one outlier row cannot shrink every other row's
        # bucket. A buffer, not a scalar, and it follows .to(device).
        self.register_buffer("tau_s", tau_s, persistent=False)
        self.register_buffer("min_s", min_s, persistent=False)
        self.register_buffer("nbits", nbits, persistent=False)
        self.register_buffer("bit_off", bit_off, persistent=False)
        self.register_buffer("ids", ids, persistent=False)
        # escapes sorted by flat position (q*P + c): the correction kernels add each output's escapes in this order,
        # one thread per output, so the result is bit-reproducible (qvb8_fix_escapes / qvb8_mma_fix_escapes)
        def _sorted(ei, ev):
            if ei.numel() > 1 and not bool((ei[1:] >= ei[:-1]).all()):
                o = torch.argsort(ei, stable=True); ei, ev = ei[o].contiguous(), ev[o].contiguous()
            return ei, ev
        esc_idx, esc_vals = _sorted(esc_idx, esc_vals)
        if esc_idx_bf16 is None:
            esc_idx_bf16, esc_vals_bf16 = esc_idx, esc_vals
        else:
            esc_idx_bf16, esc_vals_bf16 = _sorted(esc_idx_bf16, esc_vals_bf16)
        self.register_buffer("esc_idx", esc_idx, persistent=False)
        self.register_buffer("esc_vals", esc_vals, persistent=False)
        self.register_buffer("esc_idx_bf16", esc_idx_bf16, persistent=False)
        self.register_buffer("esc_vals_bf16", esc_vals_bf16, persistent=False)
        self.bias = None if bias is None else nn.Parameter(bias,
                                                           requires_grad=False)
    def escapes(self, dtype=torch.float16):
        """(esc_idx, esc_vals) for a decode that rounds the weight to `dtype`: the bf16 table for bf16, else fp16."""
        if dtype == torch.bfloat16:
            return self.esc_idx_bf16, self.esc_vals_bf16
        if dtype == torch.float16 and self.fp16_table_missing:
            from qv_band.formats.qvb8.escape_split import fp16_missing_error
            raise fp16_missing_error(getattr(self, "_where", "QVBandLinearV8"))
        return self.esc_idx, self.esc_vals

    def row_subset(self, rows: torch.Tensor) -> "QVBandLinearV8":
        """A linear holding only output rows `rows` (1-D LongTensor on this
        module's device), sharing this module's packed `ids` buffer.

        Rows of a QVB8 pack are independent -- each carries its own min_s,
        tau_s, bit width and bit offset -- so a subset needs only those four
        per-row vectors sliced, and ZERO extra weight memory. Used to run a
        cheap draft head over a restricted vocabulary (the full model still
        verifies every draft, so this changes acceptance, never output).
        Escapes are remapped to the new row numbering.
        """
        rows = rows.to(self.min_s.device).long()
        P = self.in_features
        remap = torch.full((self.out_features,), -1, dtype=torch.long,
                           device=rows.device)
        remap[rows] = torch.arange(rows.numel(), device=rows.device)

        def sub(esc_idx, esc_vals):
            if not esc_idx.numel():
                return esc_idx, esc_vals
            q = esc_idx // P
            keep = remap[q] >= 0
            return remap[q[keep]] * P + esc_idx[keep] % P, esc_vals[keep]
        esc_idx, esc_vals = sub(self.esc_idx, self.esc_vals)
        same = self.esc_idx_bf16 is self.esc_idx
        ei_b, ev_b = (None, None) if same else sub(self.esc_idx_bf16, self.esc_vals_bf16)
        m = type(self)(
            min_s=self.min_s[rows], nbits=self.nbits[rows],
            bit_off=self.bit_off[rows], ids=self.ids,
            Q=int(rows.numel()), P=P, tau_s=self.tau_s[rows],
            esc_idx=esc_idx, esc_vals=esc_vals, bias=None,
            esc_idx_bf16=ei_b, esc_vals_bf16=ev_b)
        m.bf16_certified = self.bf16_certified
        m.fp16_table_missing = self.fp16_table_missing
        return m

    @classmethod
    def from_packfile(cls, path: str, *, bias=None, tau: float | None = None,
                      device: str = "cuda", blob: bytes | None = None):
        """`tau` is accepted for call-site compatibility (the loaders pass the
        manifest's) and ignored: every pack carries its own certified tau."""
        if blob is None:
            with open(path, "rb") as fh:
                blob = fh.read()
        dev = torch.device(device)
        b = None
        esc_bf16 = (None, None)
        fp16_missing = False
        if bias is not None:
            b = (bias.detach() if isinstance(bias, torch.Tensor)
                 else torch.tensor(bias)).to(dev)
        if blob[:4] != b"QVBD":
            from qv_band.formats.qvb8.pack import unpack_tensor
            (min_s, tau_s, nb, bit_off, buf, Q, P, _tau,
             esc_idx, esc_vals) = unpack_tensor(blob)
        else:
            # The pack container (formats/qvb8/pack_container.py). Only the
            # formula calc_method (min_s + idx*tau_s) is read; a table pack is a
            # pre-unification file and must be re-packed.
            from qv_band.formats.qvb8.pack_container import (
                calc_method_of, unpack_formula, _CALC_TABLE)
            if calc_method_of(blob) == _CALC_TABLE:
                raise NotImplementedError(
                    "qvb8 table (calc_method=1) packs are not read by "
                    "QVBandLinearV8; this is a pre-unification file, re-pack it.")
            from qv_band.formats.qvb8.escape_split import fp16_list_for
            d = unpack_formula(blob, skip_bit_decode=True, fp16_list=fp16_list_for(path), where=str(path))
            min_s, tau_s, nb, bit_off, buf = (
                d["min_s"], d["tau_s"], d["nbits"], d["bit_off"], d["buf"])
            Q, P = d["Q"], d["P"]
            if d["esc_idx_fp16"] is None:
                # fp16+bf16 pack without escape_tables.json: one table (the full list, exact values); an fp16 forward
                # refuses (fp16_table_missing), a bf16 run is unchanged
                esc_idx, esc_vals = d["esc_idx"], d["esc_vals"]
                fp16_missing = True
            else:
                esc_idx, esc_vals = d["esc_idx_fp16"], d["esc_vals_fp16"]
                if d["esc_idx_fp16"] is not d["esc_idx"]:
                    esc_bf16 = (torch.from_numpy(d["esc_idx"].astype(np.int64)).to(dev),
                                torch.from_numpy(d["esc_vals"].astype(np.float32)).to(dev))
        mod = cls(
            min_s=torch.from_numpy(min_s.astype(np.float32)).to(dev),
            nbits=torch.from_numpy(nb.astype(np.uint8)).to(dev),
            bit_off=torch.from_numpy(bit_off.astype(np.int64)).to(dev),
            ids=torch.from_numpy(buf).to(dev),
            Q=Q, P=P, tau_s=torch.from_numpy(tau_s.astype(np.float32)).to(dev),
            esc_idx=torch.from_numpy(esc_idx.astype(np.int64)).to(dev),
            esc_vals=torch.from_numpy(esc_vals.astype(np.float32)).to(dev),
            bias=b, esc_idx_bf16=esc_bf16[0], esc_vals_bf16=esc_bf16[1])
        mod.bf16_certified = blob[:4] == b"QVBD" and d.get("restore") == "fp16+bf16"
        mod.fp16_table_missing = fp16_missing
        mod._where = str(path)
        return mod

    def _apply(self, fn, recurse=True):
        # Module.to(dtype) converts every floating buffer. The decode metadata (min_s / tau_s, and the exact escape values) is
        # fp32 and must stay bit-exact: the kernels read it as float, and casting it to bf16 -- even if it is cast back to fp32
        # afterwards -- leaves each row's min / step rounded to 8 mantissa bits, i.e. weights off by up to ~tau (a diffusion
        # pipeline casting its text encoder to bf16 did exactly that). Only the DEVICE follows.
        out = apply_keeping_fp32_metadata(self, fn, recurse)
        self.__dict__.pop("_mma_cache", None)       # perm/tile_nb follow the buffers: rebuild after a move
        self.__dict__.pop("_rb_cache", None)
        return out

    def _mma_order(self, ext):
        """(perm, tile_nb) for ext.mma_multi, or None when this module must keep the GEMV.

        The tensor-core kernel multiplies the restored weight rounded once: fp16(min_s + id*tau_s)
        for an fp16 pack (the certified fp16 restore the M>8 decode path uses, where the GEMV keeps
        the fp32 value), bf16(min_s + id*tau_s) for a bf16-certified pack (what decode_bf16 makes),
        and only for what the kernel can read exactly: P a multiple of 128, aligned ids. Escapes (weights stored exactly
        because they left their row's band) are corrected after the main pass. Any row width works: `perm` sorts the rows by
        width so a 16-row tile mostly shares one width in 4..8 bits (compile-time shifts,
        the fast path); `tile_nb` is that width, 0 for a mixed tile, and any tile outside
        4..8 bits (the few outlier rows of a real pack) takes the any-width path. Computed
        once, then cached."""
        cached = self.__dict__.get("_mma_cache", False)
        if cached is not False:
            return cached
        if torch.cuda.is_current_stream_capturing():
            return None                            # the width check below reads the device
        order = None
        if (hasattr(ext, "mma_multi")
                and self.ids.is_cuda
                and torch.cuda.get_device_capability(self.ids.device)[0] >= 8   # m16n8k16 is sm_80+
                and (not self.bf16_certified or "bf16" in (ext.mma_multi.__doc__ or ""))   # a library built before bf16 support
                and self.in_features % 128 == 0
                and self.ids.is_cuda and self.ids.data_ptr() % 4 == 0):
            nb = self.nbits.to(torch.int16)
            if not (1 <= int(nb.min()) and int(nb.max()) <= 14):   # 14 = the widest the format certifies
                self._mma_cache = None
                return None
            Q = nb.numel()
            perm = torch.argsort(nb, stable=True).to(torch.int32)
            srt = nb[perm.long()]
            n_tiles = (Q + 15) // 16
            srt = torch.cat([srt, torch.full((n_tiles * 16 - Q,), -1, dtype=torch.int16,
                                             device=srt.device)]).view(n_tiles, 16)
            uniform = srt.min(1).values == srt.max(1).values
            tile_nb = torch.where(uniform, srt[:, 0], torch.zeros_like(srt[:, 0])).to(torch.int32)
            order = (perm.contiguous(), tile_nb.contiguous())
            # Per-weight launch metadata, computed here once (host syncs; never inside a graph capture):
            #   esc_starts -- start of each escaped row's run in the sorted esc_idx (+ final count), for the
            #                 warp-per-row escape fix;
            #   general    -- 1 if any tile is outside the lean kernel's 4..12-bit widths (mixed, 1-3, 13-14 bits).
            # Held by the module, so they live and die with the weight.
            # one escape-row index per table: the fp16 kernel mode corrects with the fp16 table, the bf16 mode with the bf16 one
            general = int(bool(((tile_nb < 4) | (tile_nb > 12)).any()))
            self._mma_extra = {False: (self._esc_row_starts(self.esc_idx), general),
                               True: (self._esc_row_starts(self.esc_idx_bf16), general)}
        self._mma_cache = order
        return order

    def _esc_row_starts(self, esc_idx):
        """Start of each escaped row's run in the sorted esc_idx (+ final count), for the warp-per-row escape fix."""
        n_esc = esc_idx.numel()
        if not n_esc:
            return torch.empty(0, dtype=torch.int64, device=self.ids.device)
        rows = torch.div(esc_idx, self.in_features, rounding_mode="floor")
        brk = torch.ones(n_esc, dtype=torch.bool, device=rows.device)
        brk[1:] = rows[1:] != rows[:-1]
        return torch.cat([brk.nonzero().view(-1),
                          torch.full((1,), n_esc, dtype=torch.int64, device=rows.device)]).contiguous()

    def _rb_order(self, ext):
        """(perm, tile_nb) for ext.gemv_rb on GPUs without the mma path (sm_70),
        or None. Rows sorted by bit width into 16-row tiles, as _mma_order;
        cached. Opt-in (QVB8_RB=1): 1.05-1.6x at kernel level for M=2..8, no
        end-to-end gain measured on the pipelined 2x V100 server."""
        cached = self.__dict__.get("_rb_cache", False)
        if cached is not False:
            return cached
        if torch.cuda.is_current_stream_capturing():
            return None
        order = None
        if (hasattr(ext, "gemv_rb") and self.ids.is_cuda
                and os.environ.get("QVB8_RB", "0") == "1"
                and torch.cuda.get_device_capability(self.ids.device)[0] < 8
                and self.in_features % 8 == 0 and self.ids.data_ptr() % 4 == 0):
            nb = self.nbits.to(torch.int16)
            Q = nb.numel()
            perm = torch.argsort(nb, stable=True).to(torch.int32)
            srt = nb[perm.long()]
            n_tiles = (Q + 15) // 16
            srt = torch.cat([srt, torch.full((n_tiles * 16 - Q,), -1, dtype=torch.int16,
                                             device=srt.device)]).view(n_tiles, 16)
            uniform = srt.min(1).values == srt.max(1).values
            tile_nb = torch.where(uniform, srt[:, 0], torch.zeros_like(srt[:, 0])).to(torch.int32)
            order = (perm.contiguous(), tile_nb.contiguous())
        self._rb_cache = order
        return order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ext = _load_ext()
        orig = x.shape
        x2d = x.reshape(-1, orig[-1])
        if x2d.device != self.min_s.device:
            x2d = x2d.to(self.min_s.device)
        # Honour autocast like nn.Linear does: an fp32 input inside
        # torch.autocast(dtype=bf16/fp16) is computed and returned in that
        # dtype. Wan2.2 keeps its residual stream fp32 and relies on autocast
        # for every linear; without this the module took the fp32 generic path
        # (fp16 cast, fp32 result, separate bias pass) and handed fp32 on to
        # the rest of the block -- 40% of an S2V DiT forward was those copies.
        out_dtype = x.dtype
        if x2d.dtype == torch.float32 and torch.is_autocast_enabled(x2d.device.type):
            ad = torch.get_autocast_dtype(x2d.device.type)
            if ad in (torch.bfloat16, torch.float16):
                x2d = x2d.to(ad)
                out_dtype = ad
        M = x2d.shape[0]
        if self.fp16_table_missing and x2d.dtype == torch.float16:
            from qv_band.formats.qvb8.escape_split import fp16_missing_error
            raise fp16_missing_error(getattr(self, "_where", "QVBandLinearV8"))
        # qvb8_forward's M>4 path calls qvb8_decode, whose CUDA host
        # wrapper reads float32 metadata/activations. A bf16 caller (Z-Image
        # text encoder, or any mixed-precision pipeline) otherwise reaches
        # data_ptr<float>() with a BFloat16 tensor and raises
        # "expected scalar type Float but found BFloat16".
        # Do NOT force X to fp32 here.
        #
        # Every M branch inside qvb8_forward already accepts any dtype: the
        # M>8 path calls saturating_half(X) itself, and qvb8_gemv_multi's
        # launcher does the same before reaching the kernel. Nothing in the
        # dispatch reads X as fp32 -- only the METADATA (ids/min_s/tau_s/
        # bit_off) is data_ptr<float>(), and that is a separate argument.
        #
        # This used to be x2d.float().contiguous(), unconditionally, and for a
        # caller running in fp16 (Sonic's UNet does) that was a full-size
        # round trip on the activation: fp16 -> fp32 here, then fp32 -> fp16
        # again inside saturating_half. Measured on the largest Sonic linear
        # (Q=1280, P=23040, M=2048), the cast alone cost 1.67 ms, and the
        # whole module forward took 5.05 ms -- against 2.77 ms for a plain
        # F.linear on the same operands with no codec at all. The decode
        # kernel was contributing nothing; the cast was the entire overhead,
        # and it made QVB8 slower than the dense fp16 GEMM it replaces.
        #
        # bf16 is handed straight to the extension, whose saturating_half reads
        # it directly (cast_sat_half_bf16): no full-size bf16 -> fp32 copy, and
        # the result is bit-identical (bf16 -> fp32 is exact).
        xk = x2d.contiguous()
        # 8..16 rows: a speculative verify (7 drafts + 1 per request, so two requests make 16).
        # The tensor-core kernel decodes each weight once for all of them and returns fp32 like
        # the GEMV; below 8 rows the GEMV is faster, above 16 the decode-once GEMM is.
        # A bf16-certified pack is certified in bf16, so it takes the bf16 kernel with bf16 activations;
        # with other activation dtypes it keeps the old route.
        # v1: up to 48 rows (a batched verify: up to 3 requests x 16 tokens) -- the kernel decodes each weight
        # once for all token groups; above 16 rows the alternative decodes the whole matrix to bf16 (~3x the traffic).
        # Kernel mode: bf16 (weights rounded to bf16) for a bf16-certified pack with bf16 activations, with the bf16
        # escape table; otherwise fp16 mode with the fp16 table -- also for a bf16-certified pack run in fp16
        # (--dtype fp16), whose fp16 certificate holds on its own.
        # An fp16+bf16 pack takes the mma only with fp16 or bf16 activations (as before: the fp16 mode saturates X to
        # fp16, so an fp32 stream keeps the GEMV / decode paths); an fp16 pack keeps its old route for any dtype.
        bf16_mode = bool(self.bf16_certified) and x2d.dtype == torch.bfloat16
        mma_ok = (not self.bf16_certified) or x2d.dtype in (torch.float16, torch.bfloat16)
        order = self._mma_order(ext) if 8 <= M <= _MMA_MAX_M and mma_ok else None
        if order is not None:
            starts, general = self._mma_extra[bf16_mode]
            ei, ev = (self.esc_idx_bf16, self.esc_vals_bf16) if bf16_mode else (self.esc_idx, self.esc_vals)
            y = ext.mma_multi(xk, self.ids, self.min_s, self.nbits, self.bit_off,
                              self.in_features, self.tau_s, ei, ev, *order,
                              bf16_mode, starts, general)
            y = y.reshape(*orig[:-1], self.out_features)
            if self.bias is not None:
                y = y + self.bias.to(y.dtype)
            return y.to(out_dtype)
        rb = self._rb_order(ext) if order is None and M <= 16 else None
        if rb is not None and 2 <= M <= 8:
            # sm_70: register-blocked GEMV (several same-width rows per warp);
            # one row stays on v3 (measured faster there)
            y = ext.gemv_rb(xk, self.ids, self.min_s, self.nbits, self.bit_off,
                            self.in_features, self.tau_s, self.esc_idx, self.esc_vals, *rb)
            y = y.reshape(*orig[:-1], self.out_features)
            if self.bias is not None:
                y = y + self.bias.to(y.dtype)
            return y.to(out_dtype)
        # The split GEMV (pre-Ampere, 9..16 rows) reads X through saturating_half; a bf16 caller that the
        # bf16-GEMM branch below serves exactly must not be diverted into it (values past 65504 would clip).
        bf16_exact = (x2d.dtype == torch.bfloat16 and M > 8 and self.bf16_gemm and self.bf16_certified)
        if order is None and not bf16_exact and _gemv_split(self.ids, M):
            y = forward_gemv(ext, xk, self.ids, self.min_s, self.nbits,
                             self.bit_off, self.in_features, self.tau_s,
                             self.esc_idx, self.esc_vals, rb=rb)
            y = y.reshape(*orig[:-1], self.out_features)
            if self.bias is not None:
                y = y + self.bias.to(y.dtype)
            return y.to(out_dtype)
        # An fp16 caller (Sonic's UNet) with M > 8 decodes the certified
        # fp16 weight and runs F.linear: cuBLASLt, fp16 result, and the
        # bias added INSIDE the GEMM epilogue (one rounding, like a dense
        # fp16 nn.Linear). The generic path below stores an fp32 result
        # (protection for bf16/fp32 callers against fp16's 65504 ceiling,
        # which a dense fp16 model has anyway), then adds the bias and
        # casts back in separate full-size passes -- ~18 bytes of traffic
        # per output element, ~40% of a Sonic UNet step when profiled.
        # M <= 8 is unchanged (GEMV kernels).
        if bf16_exact:
            # The model's own precision end to end. Measured on Qwen-Image-
            # Edit (768x448, 20 steps) the default path below spent ~22% of
            # GPU time AROUND its GEMM: a bf16 -> fp16 cast of X, an fp32
            # result, a separate fp32 bias pass, and an fp32 -> bf16 cast.
            # Here the fp32 band value fma(id, tau_s, min_s) is rounded once
            # to bf16 inside the decode kernel; only that reconstruction is
            # fp32. Precision: |orig - bf16(band)| is <= tau for a pack
            # certified with bf16 restore, and within ~1.04 tau at tau=1e-3
            # for an fp16-certified pack (357 of 1.5e9 weights over tau).
            W = ext.decode_bf16(self.ids, self.min_s, self.nbits,
                                self.bit_off, self.in_features, self.tau_s,
                                self.esc_idx_bf16, self.esc_vals_bf16)
            b = None if self.bias is None else self.bias.to(torch.bfloat16)
            y = F.linear(xk, W, b)
            return y.reshape(*orig[:-1], self.out_features)
        if x2d.dtype == torch.float16 and M > 8:
            W = ext.decode(self.ids, self.min_s, self.nbits, self.bit_off,
                           self.in_features, self.tau_s, self.esc_idx,
                           self.esc_vals)
            b16 = None if self.bias is None else self.bias.to(torch.float16)
            y = F.linear(xk, W, b16)
            return y.reshape(*orig[:-1], self.out_features)
        else:
            y = ext.forward(xk, self.ids, self.min_s,
                            self.nbits, self.bit_off, self.in_features,
                            self.tau_s, self.esc_idx, self.esc_vals)
        y = y.reshape(*orig[:-1], self.out_features)
        if self.bias is not None:
            b = self.bias.to(y.dtype)
            # y is a fresh kernel output, so an fp16 result takes the bias in
            # place (no extra full-size allocation and pass).
            y = y.add_(b) if y.dtype == torch.float16 else y + b
        return y.to(out_dtype)


# Largest row count sent to ext.mma_multi (16 = the old limit). Clamped to the kernel's own limit (48 = 6 token
# groups of 8): a larger value would make the launcher's TORCH_CHECK fail at run time.
_MMA_MAX_M = max(8, min(48, int(__import__("os").environ.get("QV_MMA_MAX_M", "48"))))


def _gemv_split(ids: torch.Tensor, M: int) -> bool:
    """True when an M-row call should run as <=8-row GEMV passes.

    Pre-Ampere (V100) has no mma kernel, so ext.forward's 9..16-row route --
    the speculative-verify shape -- decodes the WHOLE matrix every decode step:
    a transient the size of the dense weight (1.3 GB for a 27B lm_head shard)
    and slower than reading the packed ids twice."""
    if not (8 < M <= 16 and ids.is_cuda):
        return False
    pre = _PRE_AMPERE.get(ids.device.index)
    if pre is None:             # cached: this runs on every verify-shaped call
        pre = _PRE_AMPERE[ids.device.index] = torch.cuda.get_device_capability(ids.device)[0] < 8
    return pre


_PRE_AMPERE: dict = {}


def forward_gemv(ext, x2d, ids, min_s, nbits, bit_off, P, tau_s, esc_idx,
                 esc_vals, rb=None) -> torch.Tensor:
    """ext.forward, split into <=8-row GEMV passes where _gemv_split says so.
    `rb`: (perm, tile_nb) from QVBandLinearV8._rb_order -> the register-blocked
    sm_70 kernel for each <=8-row pass."""
    def one(x):
        if rb is not None and 2 <= x.shape[0] <= 8:
            return ext.gemv_rb(x, ids, min_s, nbits, bit_off, P, tau_s, esc_idx, esc_vals, *rb)
        return ext.forward(x, ids, min_s, nbits, bit_off, P, tau_s, esc_idx, esc_vals)
    if not _gemv_split(ids, x2d.shape[0]):
        return one(x2d)
    return torch.cat([one(x2d[i:i + 8].contiguous()) for i in range(0, x2d.shape[0], 8)])


def enable_bf16_gemm(model: nn.Module) -> tuple[int, int]:
    """Put every bf16-certified QVBandLinearV8 in `model` on the bf16 GEMM.

    The one switch every loader calls (diffusion transformers, embedding,
    reranker, LLMs). Packs are calibrated for fp16 AND bf16 first; the dtype
    that runs depends on the GPU (tech_doc/restore_precision_policy.md). Only
    modules whose pack carries the bf16 certificate (header restore code 3)
    are switched; an fp16-only pack stays on its certified fp16 path. Returns
    (switched, packed_linears) so the caller can log a pack that is fp16-only.
    """
    from qv_band.formats.qvb8r.module import QVBandLinearV8R        # qvb8r: the same ids, rANS-coded; same bf16 policy
    n = tot = 0
    for m in model.modules():
        if isinstance(m, (QVBandLinearV8, QVBandLinearV8R)):
            tot += 1
            if m.bf16_certified:
                m.bf16_gemm = True
                n += 1
    return n, tot


def footprint_report(module) -> dict:
    """What a packed module actually costs, and whether it is worth it.

    A QVB8 pack is meant to be SMALLER resident memory than the dense weights
    it replaces. It can fail at that, silently, and the failure is not obvious
    from the load log: the pipeline loads, the image comes out, and the only
    symptom is a peak far above what "compressed" implies.

    The reason is the escape set. Values outside the per-row virtual band are
    stored verbatim and listed by position, and each one costs an int64 index
    plus an fp32 value -- 12 bytes, against the 2 bytes of the plain fp16
    weight it is standing in for. So a band tight enough that most values
    escape is not a compressed model at all; it is a dense model in a format
    three times more expensive than dense.

    Measured on Z-Image-Turbo (6.15B params), identical code, one flag apart:

        tau=1e-3   ids  8.02 GB   escapes  0.03 GB (     5M)   1.31 B/param
        tau=1e-4   ids  8.46 GB   escapes 36.82 GB (6137M)   7.36 B/param

    6.137 of 6.15 billion weights escaped at tau=1e-4: peak VRAM 52.5 GB,
    against 13.1 GB for tau=1e-3. Same pictures.

    So this reports the ratio and lets the CALLER decide -- a 2.0 bytes/param
    yardstick is the dense-fp16 line, below which the pack is doing its job.
    Measuring is cheap (a walk over buffers that are already resident) and it
    is the only way to notice before allocating the memory rather than after.
    """
    import torch
    from qv_band.formats.qvb8.module import QVBandLinearV8 as _Lin

    ids = escapes = meta = 0
    n_esc = n_params = n_lin = 0
    for _, m in module.named_modules():
        if not isinstance(m, _Lin):
            continue
        n_lin += 1
        n_params += m.out_features * m.in_features
        ids += m.ids.numel() * m.ids.element_size()
        tabs = [m.esc_idx, m.esc_vals]
        if m.esc_idx_bf16.data_ptr() != m.esc_idx.data_ptr():      # a separate bf16 table costs its own memory
            tabs += [m.esc_idx_bf16, m.esc_vals_bf16]
        for t in tabs:
            if torch.is_tensor(t):
                n_esc += t.numel()
                escapes += t.numel() * t.element_size()
        for t in (m.min_s, m.tau_s, m.nbits, m.bit_off):
            if torch.is_tensor(t):
                meta += t.numel() * t.element_size()
    total = ids + escapes + meta
    return {
        'linears': n_lin,
        'params': n_params,
        'escape_count': n_esc,
        'ids_bytes': ids,
        'escape_bytes': escapes,
        'meta_bytes': meta,
        'total_bytes': total,
        'bytes_per_param': (total / n_params) if n_params else 0.0,
        'escape_ratio': (n_esc / n_params) if n_params else 0.0,
    }
