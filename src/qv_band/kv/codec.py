"""qvb8 KV-cache codec (production): per-channel affine, certified.

Per-channel affine KV codec: fixed bits per element, one (min_s, tau_s) per
channel (head_dim column) of a (seq, head_dim) segment.

    id    = round((x - min_c) / step_c),   step_c = (max_c - min_c) / (2**B - 1)
    recon = min_c + id * step_c            (== min_s[c] + id * tau_s[c])

WHY. The fixed-tau codec (kv_band.py) spends bits on the SPAN of the whole
tensor, and a K/V tensor's span is set by a few outlier channels that are large
at every token. Taking min/max per CHANNEL removes that between-channel range, so
the same bit width resolves each channel at its own scale. Measured on the 27B
(16k ctx, attention-output relative error vs fp16; scratchpad kv_eq9.py):

    bits   per-token absmax (int8-style)   per-channel min/max
      4            0.0999                       0.0165
      5            0.0461                       0.0082
      6            0.0224                       0.0040
      8            0.0055                       0.0010

and the production fixed-tau codec was 0.0965 at ~5.2 bits (tau 0.25) and 0.205
at ~4.3 bits (tau 0.5). min/max cover every value, so there is NO escape channel.

Same decode form as the weight format (`min_s + id * tau_s`, one FMA); the
certified per-element error bound is tau_s[c] / 2 (+ fp32 rounding).

Interface matches what BandedKVLayer needs from a banded tensor (see BandedKV).

This REPLACED the old fixed-tau codec (the old kv_band.py, archived under
backup/kv_channel_20261002/removed/): one min per segment/head, bits set by the whole
span, an escape channel for out-of-range values. Defaults: tau 0.1, 6-bit floor.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch

_STEP_FLOOR = 1e-12

# Certified per-element error (--kv-tau-threshold overrides it per cache). Chosen
# 2026-10-02: ppl within 0.002 of dense KV on the 27B and 0.003 on the 35B.
KV_DEFAULT_TAU = 0.1
# Bit-width FLOOR; heads rise above it when the certificate needs more bits.
KV_BITS_FLOOR = int(os.environ.get("QV_BAND_KV_BITS", "6"))


def _pack_dev(ids: torch.Tensor, bits: int) -> torch.Tensor:
    """LSB-first bit-contiguous pack of flat int ids (< 2**bits) -> uint8 (same
    layout as kv_band._pack_bits). Fields never overlap, so index_add == OR."""
    if ids.is_cuda and 1 <= bits <= 16:
        from . import codec_kernels
        return codec_kernels.pack(ids, bits)
    n = ids.numel()
    pos = torch.arange(n, device=ids.device, dtype=torch.int64) * bits
    byi, sh = pos >> 3, pos & 7
    v = ids.to(torch.int64) << sh
    nbytes = (n * bits + 7) // 8
    out = torch.zeros(nbytes + 4, device=ids.device, dtype=torch.int64)
    for j in range((bits + 7) // 8 + 1):
        out.index_add_(0, byi + j, (v >> (8 * j)) & 255)
    return out[:nbytes].to(torch.uint8)


def _unpack_dev(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    if packed.is_cuda and 1 <= bits <= 16:
        from . import codec_kernels
        return codec_kernels.unpack(packed, bits, n).to(torch.int64)
    pad = torch.cat([packed, packed.new_zeros(4)])
    pos = torch.arange(n, device=packed.device, dtype=torch.int64) * bits
    byi, sh = pos >> 3, pos & 7
    acc = pad[byi].to(torch.int64)
    for j in range(1, (bits + 7) // 8 + 1):
        acc |= pad[byi + j].to(torch.int64) << (8 * j)
    return (acc >> sh) & ((1 << bits) - 1)


class _Dir:
    """Shim so code that reads `bk.direction.<x>` (unband_kv_4d, metrics) works."""
    def __init__(self, owner):
        self._o = owner
    shape = property(lambda s: s._o.orig_shape)
    outlier_count = property(lambda s: 0)
    outlier_ratio = property(lambda s: 0.0)
    n_bits = property(lambda s: s._o.bits)

    def decode_all_torch(self, device, dtype=torch.float32):
        return self._o.decode_all_torch(device, dtype)


@dataclass(eq=False)
class BandedKV:
    packed: torch.Tensor            # uint8, bits*seq*head_dim bits, row-major
    min_s: torch.Tensor             # (head_dim,) fp32
    tau_s: torch.Tensor             # (head_dim,) fp32 step per channel
    bits: int
    orig_shape: tuple               # (seq, head_dim)
    orig_dtype: torch.dtype
    tau: float                      # nominal request (unused by the codec; API parity)
    direction: _Dir = field(init=False)

    def __post_init__(self):
        self.direction = _Dir(self)

    def nbytes(self) -> int:
        return int(self.packed.numel() + 4 * (self.min_s.numel() + self.tau_s.numel()))

    outlier_count = property(lambda s: 0)
    outlier_ratio = property(lambda s: 0.0)

    @property
    def max_error(self) -> float:
        return float(self.tau_s.max()) * 0.5

    def decode_all_torch(self, device, dtype=torch.float32) -> torch.Tensor:
        seq, hd = self.orig_shape
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:          # "cuda" = the current card (pipeline: the layer's)
            dev = torch.device("cuda", torch.cuda.current_device())
        # a segment in host RAM (--kv-host) is read through a temporary device copy: it STAYS on the host
        packed, min_s, tau_s = self.packed, self.min_s, self.tau_s
        if packed.device != dev:
            packed = packed.to(dev, non_blocking=packed.is_pinned())
            min_s, tau_s = min_s.to(dev), tau_s.to(dev)
        if dev.type == "cuda" and 1 <= self.bits <= 16:
            from . import codec_kernels
            return codec_kernels.dequant(packed, self.bits, seq * hd, min_s, tau_s,
                                         hd, dtype).reshape(seq, hd)
        ids = _unpack_dev(packed, self.bits, seq * hd).reshape(seq, hd)
        return (min_s + ids.to(torch.float32) * tau_s).to(dtype)

    def truncate(self, n_rows: int) -> "BandedKV":
        """First `n_rows` tokens as a new segment, with NO re-quantisation.

        ids are stored row-major and bit-contiguous, so the first n_rows*head_dim
        ids are a byte prefix of `packed`. min_s/tau_s are shared, so every kept
        element decodes to exactly the value it did before and the certificate
        (|err| <= tau against the ORIGINAL values) still holds. Re-banding the
        decoded rows instead would quantise a quantised value: error up to ~2*tau.
        """
        seq, hd = self.orig_shape
        if not 0 < n_rows <= seq:
            raise ValueError(f"truncate({n_rows}) outside 1..{seq}")
        if n_rows == seq:
            return self
        nbytes = (n_rows * hd * self.bits + 7) // 8
        return BandedKV(
            packed=self.packed[:nbytes].clone(), min_s=self.min_s, tau_s=self.tau_s,
            bits=self.bits, orig_shape=(n_rows, hd), orig_dtype=self.orig_dtype,
            tau=self.tau)

    def take_rows(self, rows: torch.Tensor) -> "BandedKV":
        """Keep only the given token rows (ascending indices), WITHOUT re-quantising:
        the ids are unpacked, row-selected and repacked at the same bit width with
        the same per-channel min_s/tau_s, so every kept element decodes to exactly
        the value it did before and the certificate against the originals holds."""
        seq, hd = self.orig_shape
        rows = rows.to(torch.int64)
        if rows.numel() == 0 or int(rows.min()) < 0 or int(rows.max()) >= seq:
            raise ValueError(f"take_rows indices must lie in 0..{seq - 1} and be non-empty")
        if rows.numel() == seq and bool((rows == torch.arange(seq, device=rows.device)).all()):
            return self
        dev = self.packed.device
        ids = _unpack_dev(self.packed, self.bits, seq * hd).reshape(seq, hd)
        kept = ids[rows.to(dev)]
        return BandedKV(
            packed=_pack_dev(kept.reshape(-1), self.bits), min_s=self.min_s,
            tau_s=self.tau_s, bits=self.bits, orig_shape=(int(rows.numel()), hd),
            orig_dtype=self.orig_dtype, tau=self.tau)

    def decode_all(self) -> torch.Tensor:
        return self.decode_all_torch("cpu", torch.float32)

    def decode_row(self, row_idx: int) -> torch.Tensor:
        return self.decode_all()[row_idx]

    def fallback_tensor(self):
        return self.decode_all().numpy()


MAX_BITS = 16


class KVCertificateError(ValueError):
    """tau is below what this codec + storage dtype can certify for a tensor."""


def band_kv_heads(t: torch.Tensor, orig_dtype: torch.dtype, bits: int,
                       tau: float = 0.0, certify: bool = True) -> list:
    """(heads, seq, head_dim) tensor -> one BandedKV per head. Works on
    whatever device `t` is on; no host copy of the tensor.

    CERTIFIED (certify=True and tau > 0): every element satisfies
    |orig_dtype(recon) - x| <= tau, the same contract as the fixed-tau codec, but
    VERIFIED on the real decode instead of derived from a margin: the encoder
    evaluates the exact fp32 expression the decoder uses (min_s + id*tau_s), casts
    it to orig_dtype, and measures the worst error. `bits` is the FLOOR: each head
    is raised (up to 16) until that measured error is <= tau. Because x is itself
    an orig_dtype value, a fine enough grid reproduces it exactly, so huge
    magnitudes (where one bf16 ulp > 2*tau) simply reach error 0 at more bits
    rather than being refused. Per-channel ranges never exceed the tensor span, so
    this needs no more bits than the fixed-tau codec. No escapes.
    Raises KVCertificateError if even 16 bits cannot meet tau.
    certify=False (or tau<=0) = fixed `bits`, bound is only step/2 per channel.
    """
    h, s, d = t.shape
    x = t.detach().to(torch.float32)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn = x.amin(1, keepdim=True)                       # (h,1,d)
    mx = x.amax(1, keepdim=True)
    rng = (mx - mn)
    do_cert = certify and tau > 0
    if do_cert:
        worst = rng.amax((1, 2)).cpu().tolist()        # per head, one sync
        # starting guess: smallest B whose step/2 <= tau; verified below
        need = [max(bits, math.ceil(math.log2(w / (2.0 * tau) + 1.0))) if w > 0 else bits
                for w in worst]
    else:
        need = [bits] * h
    # Verify ALL heads at once (one GPU->CPU sync per round, not per head per
    # bit): per-head bit widths live in a (h,1,1) tensor; heads that fail their
    # measured error get +1 bit and the round repeats. Usually 1-2 rounds.
    bvec = [min(max(v, bits), MAX_BITS) for v in need]
    while True:
        from qv_band.common.xfer import h2d_small
        Lt = h2d_small(torch.tensor([float((1 << b) - 1) for b in bvec]), x.device).view(h, 1, 1)
        step = (rng / Lt).clamp_min(_STEP_FLOOR)
        ids = torch.round((x - mn) / step).clamp_(min=0)
        ids = torch.minimum(ids, Lt)
        if not do_cert:
            break
        recon = (mn + ids * step).to(orig_dtype).to(torch.float32)   # decoder's math
        errs = (recon - x).abs().amax((1, 2)).cpu().tolist()          # one sync per round
        bad = [i for i in range(h) if errs[i] > tau]
        if not bad:
            break
        if any(bvec[i] >= MAX_BITS for i in bad):
            i = next(i for i in bad if bvec[i] >= MAX_BITS)
            raise KVCertificateError(
                f"head {i}: {MAX_BITS}-bit per-channel error {errs[i]:.4g} > tau={tau:g}")
        for i in bad:
            bvec[i] += 1
    out = []
    for i in range(h):
        out.append(BandedKV(
            packed=_pack_dev(ids[i].to(torch.int64).reshape(-1), bvec[i]),
            min_s=mn[i, 0].contiguous(), tau_s=step[i, 0].contiguous(), bits=bvec[i],
            orig_shape=(s, d), orig_dtype=orig_dtype, tau=tau))
    return out


def unband_kv_4d(banded: list, batch: int, heads: int,
                 device: torch.device | str = "cpu") -> torch.Tensor:
    """Rebuild the (batch, heads, seq, head_dim) tensor from one BandedKV per head."""
    if not banded:
        return torch.empty(batch, heads, 0, 0, device=device)
    seq, dim = banded[0].orig_shape
    dev = torch.device(device)
    dt = banded[0].orig_dtype
    mats = [bk.decode_all_torch(dev, dt) for bk in banded]       # fp32 -> dt happens once, in the decode
    return torch.stack(mats, dim=0).reshape(batch, heads, seq, dim)
