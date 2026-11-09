"""qvb8r readers: QVBandLinearV8R (GPU, decodes the rANS-coded ids on the fly) and QVBandEmbeddingV8R (CPU rows).

QVBandLinearV8R.forward picks its kernel by token count M through one table (_PATHS), the way QVB8's module does:
  M = 1      gemv      fp32 band value
  M = 2..8   multi     fp32 band value (bf16 X read exactly)
  M = 9..16  tc        fp16 band value, WMMA (as QVB8's mma path), escape correction against fp16(band)
  M > 16     chunked   decode ~CHUNK_BYTES of rows at a time to the activation dtype and F.linear each chunk, so a
                       prefill never holds the whole dense weight (lm_head alone would be 2.5 GB)
Escapes (ids the band could not certify) are stored with their exact value and delta = exact - band; on the M <= 16
paths one kernel adds x[m, c] * delta to y[m, q]."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import container as QF



def _load_ext():
    """The qvb8r extension for the current GPU (kernels + the CPU embedding reader), built on first use and cached
    per source / GPU arch / torch (common/jit_ext.py) -- no setup step."""
    from qv_band.common.jit_ext import load_cuda_ext
    here = os.path.join(os.path.dirname(__file__), "backends", "cuda_qvb8r")
    # per current card (jit_ext caches per device); no --use_fast_math: fmaf(id, tau, min) and its fp16 rounding
    # must match the certification
    return load_cuda_ext("qvb8r_cuda", [os.path.join(here, "qvb8r_cuda.cu"), os.path.join(here, "qvb8r_cpu.cpp")],
                         cflags=["-O3", "-fopenmp"], cuda_cflags=["-O3"], ldflags=["-fopenmp"])


WPB, TC_WARPS = 8, 16         # tiles per block of the gemv/multi/decode and the tc kernels (qvb8r_cuda.cu)


def _t(a, device):
    # copy: the arrays are read-only views into the file bytes, which are freed after load
    return torch.from_numpy(np.array(a, copy=True, order="C")).to(device)


_FP16_HEADROOM = 16.0     # fallback below: scale X down by this (a power of two: exact) so an fp16 product cannot overflow


def _mm_f32(xh: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """(M, P) fp16 @ (n, P)^T fp16 -> fp32 (fp32 accumulate AND fp32 result). F.linear would round the result to fp16,
    where a value past 65504 becomes inf. torch >= 2.8 returns fp32 straight from the GEMM; older torch gets the same
    sum through an exact power-of-two down-scale of X (the result is then rounded to fp16 at 1/16 of the size)."""
    try:
        return torch.mm(xh, W.t(), out_dtype=torch.float32)
    except TypeError:
        return F.linear(xh / _FP16_HEADROOM, W).float() * _FP16_HEADROOM


class QVBandLinearV8R(nn.Module):
    # Rows of dense weight decoded per matmul in the M > 16 path. A whole tensor in ONE decode + ONE F.linear gives the
    # same fp16 results as QVB8's path (cuBLAS picks its kernel, and so its accumulation order, per matmul SHAPE: 32 MB
    # row chunks made the results differ by an ulp here and there although every weight is bit-identical) and fewer,
    # larger GEMMs (MiniMax-H3 denoise 101 s -> 81 s). Every layer tensor of the 27B (<= 178 MB) and of MiniMax-H3
    # (<= 308 MB) is one chunk; lm_head (2.5 GB) still is not.
    CHUNK_BYTES = 512 << 20
    # Opt-in per module (enable_bf16_gemm, set by loaders of bf16-native models), as QVB8's: a bf16 activation with M > 8
    # decodes the band to bf16 and runs bf16 GEMMs -- the model's own precision, with bf16's range. Only for a pack whose
    # certificate covers bf16 (restore "fp16+bf16"): the fp16 path below would clamp a bf16 stream at 65504, and a bf16
    # network (MiniMax-H3 DiT: massive activations) turns NaN there.
    bf16_gemm = False

    def __init__(self, d: dict, bias=None, device="cuda", path: str | None = None):
        super().__init__()
        self.path = path
        tab, tiles, words, off = QF.runtime_tables(d)
        Q, P = int(d["Q"]), int(d["P"])
        self.out_features, self.in_features, self.T = Q, P, int(d["T"])
        dev = device
        # device="disk": a skeleton whose buffers are None -- the weights stay in the file at `path` and a disk-mode
        # BlockStreamer rebuilds them (rebuild_buffers) for each visit to the card. Nothing lands in host RAM.
        self.on_disk = dev == "disk"
        # Every tensor is a (non-persistent) buffer, as in QVBandLinearV8: module.to(), and loaders that place
        # buffers per card (pipeline placement), move them like any other module's.
        if self.on_disk:
            dev = "cpu"                      # the bias (small) is created on the host; the streamer places it
            buf = lambda name, a: self.register_buffer(name, None, persistent=False)
        else:
            buf = lambda name, a: self.register_buffer(name, _t(a, dev), persistent=False)
        buf("words", words.view(np.int32)); buf("off", off)
        buf("states", d["states"].view(np.int64)); buf("tiles", tiles)
        buf("drow", d["drow"].astype(np.int32))
        buf("tab", tab.view(np.int32)); buf("perm", d["perm"])
        blocks8 = QF.blocks_for(tiles, WPB)
        buf("blocks8", blocks8); buf("blocks16", QF.blocks_for(tiles, TC_WARPS))
        buf("min_s", d["min_s"]); buf("tau_s", d["tau_s"])
        self.cert = d.get("cert", {})
        self.bf16_certified = self.cert.get("restore") == "fp16+bf16"
        # Two escape tables (formats/qvb8/escape_split.py): the file holds the bf16 table (the full list); the fp16 table
        # comes ONLY from escape_tables.json (validated, nothing computed). fp16 paths (gemv, multi, tc, _chunked, dense
        # fp16) use the fp16 table; _chunked_bf16 and dense(bf16) the bf16 one. No file on an fp16+bf16 pack: one table
        # (the full list, exact values) and an fp16 forward refuses.
        ei_b = d["esc_idx"].astype(np.int64)
        ev_b = np.asarray(d["esc_val"], np.float32)
        need = np.ones(len(ei_b), bool)
        self.fp16_table_missing = False
        self._where = str(path)
        if self.bf16_certified:
            from qv_band.formats.qvb8.escape_split import fp16_list_for, select_fp16
            lst = fp16_list_for(path)
            if lst is None:
                self.fp16_table_missing = True
            else:
                need = select_fp16(ei_b, lst, str(path))
        d = dict(d, esc_idx=ei_b[need], esc_val=ev_b[need], esc_delta=np.asarray(d["esc_delta"], np.float32)[need])
        ei = d["esc_idx"].astype(np.int64)
        # the kernel correction (fix_escapes) groups the escapes by output q (CSR, sorted by q then c): one thread per
        # (m, q) sums its group in this fixed order -- bit-reproducible, no atomics
        o = np.argsort(ei, kind="stable"); es = ei[o]
        uq, cnt = np.unique(es // P, return_counts=True)
        buf("esc_uq", uq.astype(np.int64)); buf("esc_ptr", np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64))
        buf("esc_c", es % P)
        buf("esc_delta", d["esc_delta"][o])
        # the tc path multiplies fp16(band): its correction is exact - fp16(band); band = exact - delta (fp32)
        ev = d["esc_val"][o].astype(np.float32)
        band = ev - d["esc_delta"][o].astype(np.float32)
        buf("esc_delta16", ev - band.astype(np.float16).astype(np.float32))
        buf("esc_flat", ei); buf("esc_val", d["esc_val"])
        buf("esc_flat_bf16", ei_b); buf("esc_val_bf16", ev_b)
        self.bias = None if bias is None else nn.Parameter(bias.detach().to(dev), requires_grad=False)
        # M > 16: chunks of whole blocks (~CHUNK_BYTES of decoded weight each) and the escapes inside each.
        # Per chunk: (first block, blocks, first sorted row, rows, [o0, o1) into chunk_orows, [e0, e1) into chunk_e*)
        inv = np.empty(Q, np.int64); inv[d["perm"]] = np.arange(Q)
        rows_per_block = [int(tiles[t0:t0 + n, 2].sum()) for _, t0, n in blocks8]
        first_row = [int(tiles[t0, 1]) for _, t0, n in blocks8]
        budget = max(1, self.CHUNK_BYTES // (2 * P)); self.chunks = []; b = 0
        orows, spans = [], []; o0 = 0
        while b < len(blocks8):
            e, rows = b, 0
            while e < len(blocks8) and (e == b or rows + rows_per_block[e] <= budget):
                rows += rows_per_block[e]; e += 1
            base = first_row[b]
            orows.append(d["perm"][base:base + rows].astype(np.int64))
            spans.append((b, e - b, base, rows, o0, o0 + rows))
            o0 += rows; b = e

        def chunk_escapes(e_idx, e_val):
            """Each chunk's escapes (flat in the chunk's sorted rows) and its [e0, e1) range."""
            sp = inv[e_idx // P] if len(e_idx) else np.zeros(0, np.int64)
            flat, vals, rng, e0 = [], [], [], 0
            for (_, _, base, rows, _, _) in spans:
                sel = np.nonzero((sp >= base) & (sp < base + rows))[0]
                flat.append((sp[sel] - base) * P + e_idx[sel] % P); vals.append(e_val[sel])
                rng.append((e0, e0 + len(sel))); e0 += len(sel)
            return (np.concatenate(flat).astype(np.int64) if flat else np.zeros(0, np.int64),
                    np.concatenate(vals).astype(np.float32) if vals else np.zeros(0, np.float32), rng)
        f16, v16, r16 = chunk_escapes(ei, d["esc_val"])
        fb, vb, rb = chunk_escapes(ei_b, ev_b)
        self.chunks = [sp + r for sp, r in zip(spans, r16)]             # fp16 table ranges (_chunked)
        self.chunks_bf16 = [sp + r for sp, r in zip(spans, rb)]         # bf16 table ranges (_chunked_bf16)
        buf("chunk_orows", np.concatenate(orows))
        buf("chunk_eflat", f16); buf("chunk_evals", v16)
        buf("chunk_eflat_bf16", fb); buf("chunk_evals_bf16", vb)

    def rebuild_buffers(self) -> dict:
        """Disk mode: read this module's file again and return its buffers as host (pageable) tensors -- the same bytes a
        resident load builds. The file's pages are dropped from the page cache afterwards."""
        import os
        tmp = type(self)(QF.read(self.path), device="cpu", path=self.path)
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return dict(tmp._buffers)

    @classmethod
    def from_packfile(cls, path: str, *, bias=None, tau: float | None = None, device: str = "cuda",
                      blob: bytes | None = None):
        """`tau` is accepted for call-site compatibility and ignored: each file carries its own certified tau."""
        return cls(QF.read(path, blob), bias=bias, device=device, path=path)

    # decode metadata: the kernels read these as fp32; a dtype cast (module.to(torch.bfloat16), pipe.to(...)) must never narrow them
    _FP32_BUFFERS = ("min_s", "tau_s", "esc_val", "esc_delta", "esc_delta16", "chunk_evals",
                     "esc_val_bf16", "chunk_evals_bf16")

    def _apply(self, fn, recurse=True):
        """module.to(...) / .half() / .bfloat16(): everything follows (a bias changes dtype), the fp32 metadata only changes DEVICE.
        (They are taken out of the module while fn runs: casting them to bf16 and back would keep the dtype but lose the bits.)"""
        keep = {n: self._buffers[n] for n in self._FP32_BUFFERS if self._buffers.get(n) is not None}
        for n in keep:
            self._buffers[n] = None
        try:
            out = super()._apply(fn, recurse)
        finally:
            ref = self._buffers.get("words")
            dev = ref.device if ref is not None else next(iter(keep.values())).device if keep else None
            for n, b in keep.items():
                self._buffers[n] = b.to(dev)
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    # -- kernel paths ------------------------------------------------------------------------------------------
    def _core(self, blocks):
        return (self.words, self.off, self.states, self.tiles, blocks, self.tab, self.perm, self.min_s, self.tau_s,
                self.drow)

    def _escapes(self, xf, y, delta):
        y = y.contiguous(); _load_ext().fix_escapes(xf, self.esc_uq, self.esc_ptr, self.esc_c, delta, y); return y

    def _gemv(self, x2):
        xf = x2.float().contiguous()
        y = _load_ext().gemv(xf.view(-1), *self._core(self.blocks8))
        return self._escapes(xf, y.view(1, -1), self.esc_delta)

    def _multi(self, x2):
        # The kernel reads X transposed (P, MB): one vector load gives a lane all MB rows of its column. Reading (M, P)
        # in place (no copy) measured 2.6x slower on a V100 (MB scalar loads per symbol): keep the copy.
        xf = x2.float().contiguous()
        xk = x2.contiguous() if x2.dtype == torch.bfloat16 else xf
        return self._escapes(xf, _load_ext().multi(xk, *self._core(self.blocks8)), self.esc_delta)

    def _tc(self, x2):
        xf = x2.float().contiguous()
        xh = xf.clamp(-65504, 65504).half().contiguous()                     # saturating fp16, as QVB8's mma path
        y = _load_ext().tc(xh, *self._core(self.blocks16), self.T)
        return self._escapes(xf, y, self.esc_delta16)

    def _chunked_bf16(self, x2):
        """bf16 GEMM, the bias fused into it (one rounding, as QVB8's cuBLASLt epilogue): the result already holds the bias."""
        y = torch.empty(x2.shape[0], self.out_features, device=x2.device, dtype=torch.bfloat16)
        bias = None if self.bias is None else self.bias.to(torch.bfloat16)
        for b0, nb, base, rows, o0, o1, e0, e1 in self.chunks_bf16:
            W = _load_ext().decode(*self._core(self.blocks8), self.in_features, 1, b0, nb, base, rows)
            if e1 > e0:
                W.view(-1)[self.chunk_eflat_bf16[e0:e1]] = self.chunk_evals_bf16[e0:e1].to(torch.bfloat16)
            y[:, self.chunk_orows[o0:o1]] = F.linear(x2, W, None if bias is None else bias[self.chunk_orows[o0:o1]])
        return y

    def _chunked(self, x2):
        # fp16 weight x saturating fp16 X, fp32 accumulate: QVB8's M > 8 precision (a bf16 weight would carry 3
        # fewer mantissa bits)
        # cast then clamp == clamp in fp32 then cast (|x| > 65504 becomes 65504 or inf, both clamp to 65504),
        # without a full fp32 copy of X (34 MB for a 512-token MLP down_proj input)
        xh = x2.half()
        xh = xh.clamp(-65504, 65504) if xh is x2 else xh.clamp_(-65504, 65504)   # fp16 in: .half() is x2 itself, never clamp the caller's tensor in place
        y = torch.empty(x2.shape[0], self.out_features, device=x2.device, dtype=x2.dtype)
        # fp32 or bf16 activation: the product is accumulated AND returned in fp32, then rounded ONCE to the activation dtype
        # (QVB8's path; an fp16-output GEMM would round twice, and a value past 65504 becomes inf: an fp32 STREAM -- a network
        # trained in bf16 on a GPU without bf16, e.g. MiniMax-H3 on a V100 -- would turn NaN). An fp16 activation: F.linear.
        wide = x2.dtype in (torch.float32, torch.bfloat16)
        if len(self.chunks) == 1 and x2.dtype != torch.bfloat16:
            # ONE chunk (every layer up to CHUNK_BYTES of decoded weight): put the decoded WEIGHT rows back in output order
            # (a contiguous row copy, Q x P fp16) and let the GEMM write y directly. The old y[:, orows] = ... was a strided
            # column scatter of the whole M x Q result -- Z-Image 1 MP on a V100: 325 ms of index_put per step. Same values:
            # each output element is the same row . column product.
            b0, nb, base, rows, o0, o1, e0, e1 = self.chunks[0]
            W = _load_ext().decode(*self._core(self.blocks8), self.in_features, 0, b0, nb, base, rows)
            if e1 > e0:
                W.view(-1)[self.chunk_eflat[e0:e1]] = self.chunk_evals[e0:e1].half()
            Wo = torch.empty_like(W).index_copy_(0, self.chunk_orows[o0:o1], W)
            return _mm_f32(xh, Wo).to(x2.dtype) if wide else F.linear(xh, Wo)
        for b0, nb, base, rows, o0, o1, e0, e1 in self.chunks:
            W = _load_ext().decode(*self._core(self.blocks8), self.in_features, 0, b0, nb, base, rows)
            if e1 > e0:
                W.view(-1)[self.chunk_eflat[e0:e1]] = self.chunk_evals[e0:e1].half()
            y[:, self.chunk_orows[o0:o1]] = _mm_f32(xh, W).to(x2.dtype) if wide else F.linear(xh, W)
        return y

    _PATHS = ("_gemv",) * 2 + ("_multi",) * 7 + ("_tc",) * 8 + ("_chunked",)   # index = min(M, 17)

    def forward(self, x):
        shp = x.shape; x2 = x.reshape(-1, shp[-1])
        if x2.device != self.words.device:
            x2 = x2.to(self.words.device)
        # honour autocast like nn.Linear (and QVBandLinearV8): fp32 in under autocast -> computed/returned in its dtype
        out_dtype = x.dtype
        if x2.dtype == torch.float32 and torch.is_autocast_enabled(x2.device.type):
            ad = torch.get_autocast_dtype(x2.device.type)
            if ad in (torch.bfloat16, torch.float16):
                x2, out_dtype = x2.to(ad), ad
        if x2.shape[0] == 0:
            return x2.new_empty(*shp[:-1], self.out_features, dtype=out_dtype)
        if self.fp16_table_missing and x2.dtype == torch.float16:
            from qv_band.formats.qvb8.escape_split import fp16_missing_error
            raise fp16_missing_error(self._where)
        path = self._PATHS[min(x2.shape[0], 17)]
        if self.bf16_gemm and self.bf16_certified and x2.dtype == torch.bfloat16 and x2.shape[0] > 8:
            path = "_chunked_bf16"
        y = getattr(self, path)(x2)
        if self.bias is not None and path != "_chunked_bf16":
            y = y + self.bias.to(y.dtype)
        return y.to(out_dtype).reshape(*shp[:-1], self.out_features)

    def dense(self, dtype=torch.bfloat16):
        """The exact weight (Q, P) in `dtype` (escapes applied) -- for audits / re-packing, never on the hot path."""
        code = {torch.float16: 0, torch.bfloat16: 1}.get(dtype, 2)
        W = _load_ext().decode(*self._core(self.blocks8), self.in_features, code, 0, self.blocks8.shape[0], -1, 0)
        if dtype == torch.float16 and self.fp16_table_missing:
            from qv_band.formats.qvb8.escape_split import fp16_missing_error
            raise fp16_missing_error(self._where)
        ef, ev = (self.esc_flat_bf16, self.esc_val_bf16) if dtype == torch.bfloat16 else (self.esc_flat, self.esc_val)
        if ef.numel():
            W.view(-1)[ef] = ev.to(W.dtype)
        return W.to(dtype)


class QVBandEmbeddingV8R(nn.Module):
    """Rows decoded on the CPU (C++/OpenMP); only the looked-up rows reach `device`."""

    def __init__(self, d: dict, out_dtype=torch.bfloat16, device="cuda", path: str | None = None):
        super().__init__()
        tab, tiles, words, off = QF.runtime_tables(d)
        Q, P = int(d["Q"]), int(d["P"])
        self.num_embeddings, self.embedding_dim, self._out_dtype, self.device = Q, P, out_dtype, device
        inv = np.empty(Q, np.int32); inv[d["perm"]] = np.arange(Q, dtype=np.int32)
        row_tile = np.empty(Q, np.int32)
        for t, (_, r0, nr) in enumerate(tiles):
            row_tile[r0:r0 + nr] = t
        ei = d["esc_idx"].astype(np.int64)
        ev = np.asarray(d["esc_val"], np.float32)
        cert = d.get("cert", {})
        # output dtype fixed here: the bf16 table for bf16 output, else the fp16 table (formats/qvb8/escape_split.py)
        if out_dtype == torch.float16 and cert.get("restore") == "fp16+bf16":       # bf16 / fp32 output: the full list
            from qv_band.formats.qvb8.escape_split import fp16_list_for, select_fp16, fp16_missing_error
            lst = fp16_list_for(path)
            if lst is None:
                raise fp16_missing_error(str(path))
            need = select_fp16(ei, lst, str(path))
            ei, ev = ei[need], ev[need]
        er = ei // P; o = np.argsort(er, kind="stable")
        ptr = np.zeros(Q + 1, np.int64); np.add.at(ptr, er + 1, 1); ptr = np.cumsum(ptr)
        C = lambda a: torch.from_numpy(np.array(a, copy=True, order="C"))
        self._c = (C(words.view(np.int32)), C(off), C(d["states"].view(np.int64)), C(tiles), C(tab.view(np.int32)),
                   C(inv), C(row_tile), C(d["min_s"]), C(d["tau_s"]), C(d["drow"].astype(np.int32)), C(ptr),
                   C((ei % P)[o].astype(np.int32)),
                   C(ev[o]))

    @classmethod
    def from_packfile(cls, path: str, *, out_dtype=torch.bfloat16, tau: float | None = None, device: str = "cuda",
                      blob: bytes | None = None):
        return cls(QF.read(path, blob), out_dtype=out_dtype, device=device, path=path)

    def extra_repr(self):
        return f"{self.num_embeddings}, {self.embedding_dim}, rows decoded on CPU"

    @property
    def weight(self):
        """Dense (Q, P) table on `device`, for callers that need all of it (as QVBandEmbeddingV8.weight).
        Decoded in row chunks so the CPU never holds the whole table in fp32."""
        out = torch.empty(self.num_embeddings, self.embedding_dim, dtype=self._out_dtype, device=self.device)
        step = max(1, (64 << 20) // (4 * self.embedding_dim))
        for r0 in range(0, self.num_embeddings, step):
            rows = torch.arange(r0, min(r0 + step, self.num_embeddings), dtype=torch.int64)
            out[r0:r0 + len(rows)] = _load_ext().embed_rows(rows, *self._c, self.embedding_dim).to(self._out_dtype)
        return out

    def forward(self, ids):
        flat = ids.reshape(-1).to("cpu", torch.int64)
        y = _load_ext().embed_rows(flat, *self._c, self.embedding_dim)
        return y.to(self._out_dtype).to(ids.device if ids.device.type != "cpu" else self.device) \
                .reshape(*ids.shape, self.embedding_dim)
