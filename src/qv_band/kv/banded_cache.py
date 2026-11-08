"""Resident v7-KV cache: K/V held in BandedKV form, never dense.

This is the piece that turns kv_band.py's compression from a MEASUREMENT
(compress_cache_inplace, which bands the prefill population, reports a
ratio, then immediately writes a dense reconstruction back -- "not yet
realised as VRAM", as chat.py prints) into an actual resident memory
saving: the compressed BandedKV objects ARE the stored state between
calls, and a dense tensor exists only transiently inside update() for
attention to consume.

Structure mirrors qv_band.kv.int8_kv_cache exactly, and for the same
reasons -- subclass the official DynamicLayer, swap it in only at
full_attention positions, leave linear-attention layers on their stock
class (they hold a fixed-size recurrent conv/SSM state, not a growing
per-token K/V population, so there is nothing to compress there). Read
that module's docstring for why transformers' own QuantizedCache cannot
be used on hybrid Qwen3.5 models.

WHY A RESIDUAL BUFFER (same shape as the int8 cache's)
------------------------------------------------------
band_kv_heads() still costs a min/max scan plus an encode/verify pass
per (layer, head) tensor. Doing that on every single
appended token would dominate decode. New tokens therefore land in a small
dense residual buffer and are banded in bulk once it exceeds residual_length,
exactly as the int8 layer flushes its own residual.

WHAT THIS DOES NOT DO
---------------------
Attention still receives a DENSE tensor. That is not a shortcut: it is
what the stock attention kernels require, and the int8 cache does the
same (it dequantizes its whole int8 store on every update() too). The
saving is in what stays resident BETWEEN calls, not in what attention
reads during one.

Measured on the 4B pack (real model K/V, RSS over 400 copies, seq=115,
4 heads, head_dim=256): BandedKV resident cost is 34.6% of the dense
fp16 equivalent, against ~50.8% for the int8 layer -- but note the
reconstruction cost is higher here (a closed-form per-element formula,
kv_band.py's FixedTauDirection, vs. int8's elementwise multiply), which
is the trade this class exists to expose.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import torch

from transformers.cache_utils import (
    Cache, DynamicLayer, LinearAttentionCacheLayerMixin,
)

# get_layer_types_and_kwargs / DYNAMIC_LAYER_TYPE_MAPPING were removed from
# this transformers version's cache_utils (the dispatch they did is now
# inlined into DynamicCache.__init__ instead) -- see
# qv_band.kv.int8_kv_cache's identical fallback (same upstream removal
# hits both modules) for the full rationale; kept in sync with that one.
try:
    from transformers.cache_utils import (
        get_layer_types_and_kwargs,
        DYNAMIC_LAYER_TYPE_MAPPING,
    )
except ImportError:
    from transformers.cache_utils import (
        DynamicSlidingWindowLayer, LinearAttentionLayer, LinearAttentionAndFullAttentionLayer,
    )

    DYNAMIC_LAYER_TYPE_MAPPING = {
        "full_attention": DynamicLayer,
        "sliding_attention": DynamicSlidingWindowLayer,
        "chunked_attention": DynamicSlidingWindowLayer,
        "mamba": LinearAttentionLayer,
        "conv": LinearAttentionLayer,
        "linear_attention": LinearAttentionLayer,
        "moe": LinearAttentionLayer,
        "hybrid": LinearAttentionAndFullAttentionLayer,
    }

    def get_layer_types_and_kwargs(decoder_config):
        """Reproduces DynamicCache.__init__'s layer_types derivation for
        this transformers version. BandedKVCache below only ever
        instantiates full_attention (BandedKVLayer, taking its own kwargs,
        not this dict) or the mapping's class with no args, so sliding/
        chunked attention -- which needs sliding_window=... on construction
        -- is unsupported here, same as int8_kv_cache.py's fallback."""
        sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
            decoder_config, "attention_chunk_size", None)
        layer_types = getattr(decoder_config, "layer_types", None)
        if layer_types is None:
            layer_types = ["sliding_attention" if sliding_window is not None
                          else "full_attention"] * decoder_config.num_hidden_layers
        if hasattr(decoder_config, "num_kv_shared_layers"):
            layer_types = layer_types[: -decoder_config.num_kv_shared_layers]
        if "sliding_attention" in layer_types or "chunked_attention" in layer_types:
            raise NotImplementedError(
                "BandedKVCache's layer dispatch does not support sliding/"
                "chunked attention layer types in this transformers version "
                "-- only linear_attention/full_attention hybrids (e.g. "
                "Qwen3.5/3.6) are supported")
        return layer_types, {}

from qv_band.kv.codec import (
    unband_kv_4d, KV_DEFAULT_TAU, KV_BITS_FLOOR, band_kv_heads,
)

__all__ = ["BandedKVLayer", "BandedKVCache", "NemotronHBandedCache"]


# Byte budget for ONE layer's dense-reconstruction cache (see
# BandedKVLayer._unband_cached).
#
# This used to be a count of ENTRIES (4096) with no byte limit at all,
# which defeated the whole point of the banded store: each entry is a
# full dense (1, heads, seg_len, head_dim) reconstruction of a segment,
# so a long context kept its ENTIRE history in dense form on the GPU
# right next to the compressed copy -- and per slot, so two concurrent
# requests held two full dense histories. Measured on a real server:
# two Cline requests pushed VRAM to ~80GB where one used <40GB, with
# the banded store itself reporting near-zero; the dense cache was
# effectively the whole KV cache.
#
# A byte budget makes the cache what it was meant to be: a bounded
# accelerator for recently-touched segments, not a second copy of
# everything. Segments evicted from it are simply re-decoded on demand.
#
# Re-decoding used to be ruinously expensive -- it ran in numpy on the
# CPU and then copied the expansion across the bus, measured at 67% of
# total wall time. FixedTauDirection.decode_all_torch now decodes ON the
# GPU straight from the packed band ids, which removes that cost.
#
# DEFAULT: generous, NOT off. A 0.8B/5k-token benchmark showed only a
# ~1.4x penalty for running with no cache, and this defaulted to 0 on
# that basis. That was the wrong basis and it caused a real production
# regression: on the 9B at 24k context, decode collapsed from ~5 tok/s
# to ~0.5 tok/s. The reason is that the cost is PER DECODE STEP and
# scales with context -- a decode step produces ONE token but rebuilds
# the whole history for it, whereas prefill amortises the same work
# over a 512-token chunk (which is why prefill only fell 290->170 while
# decode fell 10x).
#
# So the cache is not a luxury at long context: it is what keeps decode
# O(new tokens) instead of O(context) per step.
#
# 512MB/layer measured on the 9B at 23801 tokens: prefill 349 tok/s,
# against 199 tok/s with the cache off. 4096MB measured 346 tok/s --
# i.e. indistinguishable, so the larger budget reserves VRAM for no
# gain and 512 is the right default. Set QV_BAND_UNBAND_CACHE_MB lower
# to trade speed back for VRAM, or 0 to disable it entirely.
_UNBAND_CACHE_BYTES = int(os.environ.get(
    "QV_BAND_UNBAND_CACHE_MB", "512")) * 2 ** 20


# Outlier ratio above which a segment is ALSO kept dense instead of relying
# on its escape side channel alone. A segment with a very high outlier count
# spends more bytes on (index, value) escape pairs than the dense form costs,
# so past this point the escape channel is a compression REGRESSION.
# 0.05 = 5% of elements escaped.
#
# KEEPING THE DENSE COPY IS OFF BY DEFAULT, and that is deliberate. The
# budget below is PER LAYER, not per cache: a 48-layer model at the old 128MB
# default could retain 6GB of UNCOMPRESSED K/V on top of the banded store --
# which is exactly the failure this whole cache exists to prevent (see
# _UNBAND_CACHE_BYTES' comment for the same mistake already made once, and
# the measured 80GB two-request VRAM blowout it caused). It OOM'd a MoE model
# outright when this was introduced.
#
# The escape side channel is already lossless: an out-of-range value is
# stored verbatim, so a dense copy buys SPEED on high-outlier segments, never
# correctness. Opt in only when profiling shows decode is decode-bound:
#   QV_BAND_DENSE_FALLBACK_MB=32   (per layer; 32 x n_layers is your cost)
# and keep the total comfortably under the VRAM headroom the model leaves.
_OUTLIER_FALLBACK_RATIO = float(os.environ.get(
    "QV_BAND_OUTLIER_FALLBACK_RATIO", "0.05"))
_DENSE_FALLBACK_BYTES = int(os.environ.get(
    "QV_BAND_DENSE_FALLBACK_MB", "0")) * 2 ** 20

# Both caches above are PER-LAYER budgets, and a per-layer budget is a
# multiplication waiting to happen: N layers each allowed to retain B bytes
# can retain N*B, which no single layer's accounting ever sees. These two
# process-wide caps bound the multiplication. They are deliberately low
# enough to be safe for a model whose weights already fill the card, and
# they bound a *cache*, so under pressure the right outcome is a cache miss
# (a re-decode), never an OOM.
_UNBAND_CACHE_CAP_BYTES = int(os.environ.get(
    "QV_BAND_GLOBAL_UNBAND_CACHE_MB", "1024")) * 2 ** 20
_DENSE_FALLBACK_CAP_BYTES = int(os.environ.get(
    "QV_BAND_GLOBAL_DENSE_FALLBACK_MB", "512")) * 2 ** 20
# Joined K/V history cache: a decode accelerator, never a second KV store.
_JOINED_CACHE_BYTES = int(os.environ.get(
    "QV_BAND_JOINED_CACHE_MB", "128")) * 2 ** 20
_JOINED_CACHE_CAP_BYTES = int(os.environ.get(
    "QV_BAND_GLOBAL_JOINED_CACHE_MB", "512")) * 2 ** 20

# Append-in-place dense K/V views for the ragged attention path (see
# BandedKVLayer.ragged_kv_views). Unlike the joined cache this is the PRIMARY
# decode-time view, so it is sized to hold real context: 512 MB/layer is
# ~128k tokens of (4 kv heads x 256 dim x fp16) K+V. Past either cap a slot
# falls back to the (slower, always-correct) rebuild-per-step path.
# Tried on the dev GPU (v1, 2026-11-06): 1 GB/layer + 10 GB global so Ripple's
# 16 full-attention layers keep the view at 75k tokens. A/B on a 74,764-token
# prompt, same acceptance: decode 4.5 / 5.7 tok/s with the big caps vs
# 4.9 / 6.3 with these defaults -- no gain, so the defaults stay. Override with
# QV_BAND_DENSE_VIEW_MB / QV_BAND_GLOBAL_DENSE_VIEW_MB. A failed allocation
# falls back to the rebuild path (see ragged_kv_views) instead of raising.
_SM_VIEW_BYTES = int(os.environ.get(
    "QV_BAND_DENSE_VIEW_MB", "512")) * 2 ** 20
_SM_VIEW_CAP_BYTES = int(os.environ.get(
    "QV_BAND_GLOBAL_DENSE_VIEW_MB", "4096")) * 2 ** 20

# Set by fused_attn.install(): a layer may then skip the dense reconstruction.
FUSED_ENABLED = False
_FUSED_MIN_BANDED = 1024     # below this the dense path is as cheap
_FUSED_MAX_Q_CACHE: dict = {}


def _fused_max_q(device) -> int | None:
    """Longest query the fused attention takes on this GPU (None = any length). Before sm_80 the dense path
    runs emulated bf16 and the WMMA kernel (kv/cuda_attn) tiles any query, so prefill chunks go fused too;
    from sm_80 on, prefill keeps the dense / flash path and only verify-sized queries (<= 16) go fused."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _FUSED_MAX_Q_CACHE:
        _FUSED_MAX_Q_CACHE[idx] = None if torch.cuda.get_device_capability(idx)[0] < 8 else 16
    return _FUSED_MAX_Q_CACHE[idx]

_GLOBAL_UNBAND_BYTES = 0
_GLOBAL_DENSE_FALLBACK_BYTES = 0
_GLOBAL_JOINED_BYTES = 0
_GLOBAL_SM_BYTES = 0


def _unband_global_add(n: int) -> None:
    global _GLOBAL_UNBAND_BYTES
    _GLOBAL_UNBAND_BYTES += n


def _unband_global_subtract(n: int) -> None:
    global _GLOBAL_UNBAND_BYTES
    # Clamp at 0: a double-drop (forget path after an eviction already
    # decremented) must not leave the global counter negative, which would
    # silently disable the very cap protecting against the OOM.
    _GLOBAL_UNBAND_BYTES = max(0, _GLOBAL_UNBAND_BYTES - n)


def _dense_fallback_global_add(n: int) -> None:
    global _GLOBAL_DENSE_FALLBACK_BYTES
    _GLOBAL_DENSE_FALLBACK_BYTES += n


def _dense_fallback_global_subtract(n: int) -> None:
    global _GLOBAL_DENSE_FALLBACK_BYTES
    _GLOBAL_DENSE_FALLBACK_BYTES = max(0, _GLOBAL_DENSE_FALLBACK_BYTES - n)


def _joined_global_add(n: int) -> None:
    global _GLOBAL_JOINED_BYTES
    _GLOBAL_JOINED_BYTES += n


def _joined_global_subtract(n: int) -> None:
    global _GLOBAL_JOINED_BYTES
    _GLOBAL_JOINED_BYTES = max(0, _GLOBAL_JOINED_BYTES - n)


class BandedKVLayer(DynamicLayer):
    """A DynamicLayer whose flushed history is stored as v7-KV BandedKV
    objects (one per (batch, head) slice) instead of dense tensors.

    `residual_length` keeps the most recent tokens dense. Those are the
    tokens most likely to be rolled back by speculative decoding, and
    re-banding on every append would cost a full calibration per token
    (see the module docstring); they are banded in bulk once the buffer
    exceeds the threshold.

    PER-SLOT BOOKKEEPING (continuous batching). BOTH the banded store
    (`_banded_k`/`_banded_v`/`_seg_len`/`_banded_len`) AND the dense
    residual (`_resid_k`/`_resid_v`) are indexed by batch position:
    `self._banded_k[bi]` is batch element `bi`'s own list of segments,
    `self._resid_k[bi]` its own (1, heads, seq_i, head_dim) residual
    tensor. Each slot flushes, crops, and grows its residual entirely
    independently of every other slot -- there is no shared tensor
    forcing lockstep any more (an earlier version kept the residual as
    ONE (batch, heads, seq, head_dim) tensor, which forced every slot to
    hold the exact same residual length; that constraint is gone).

    update() still returns ONE dense (batch, heads, seq_max, head_dim)
    tensor per call, since that is what the attention kernel needs --
    slots shorter than the batch's longest get LEFT-padded to seq_max
    (see _pad_and_stack/get_padding_mask) rather than being forced to
    literally share a tensor. add_slot()/drop_slot() admit/remove one
    slot between update() calls; `crop()` still takes an optional `slot`
    to roll back ONE batch element without touching the others.
    """

    def __init__(self, tau: float = KV_DEFAULT_TAU, residual_length: int = 128,
                 window: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.residual_length = residual_length
        # Sliding-window attention layer (e.g. Ripple's local layers): the FULL history is still stored (banded),
        # so crop / prefix reuse / per-slot scheduling work exactly as for a full-attention layer, but attention
        # only ever reads the last `window` keys -- update() and the varlen path return just that tail
        # (window_kv_slot), never the whole history. None = full attention (every Qwen layer): unchanged.
        self.window = window
        # Banded store, PER BATCH POSITION: self._banded_k[bi] is a LIST OF
        # SEGMENTS in token order for batch element bi. Each segment is
        # itself a list over that element's `heads` of BandedKV -- one
        # flush's worth, with its own codebook/scale basis. Segments are
        # never merged or re-banded (see _flush_residual). All lists are
        # kept the same length as the batch (self._heads is set once
        # lazy_initialization knows it); a batch element with nothing
        # banded yet just has an empty segment list.
        self._banded_k: list[list] = []
        self._banded_v: list[list] = []
        self._seg_len: list[list] = []      # tokens in each segment, per slot
        self._banded_len: list = []         # banded token count, per slot
        self._heads: int | None = None      # heads per batch element (constant)
        # PER-SLOT dense residual (Phase 2 -- continuous batching): each
        # slot keeps its OWN (1, heads, seq_i, head_dim) tensor instead of
        # one shared (batch, heads, seq, head_dim) tensor. This is what
        # actually lets slots diverge -- a shared tensor forces every slot
        # to hold the SAME residual length (its seq axis has one size),
        # which is what made "every slot must advance in lockstep" a real
        # constraint, not just an unenforced convention. self._dtype/
        # self._device are cached from the first append (needed to build
        # an empty placeholder for a slot with nothing appended yet).
        self._resid_k: list = []
        self._resid_v: list = []
        self._dtype: torch.dtype | None = None
        self._device = None
        # Lightweight, cumulative observability for profiling; no CUDA sync in
        # the request path.
        self.metrics = {
            "flushes": 0, "flush_tokens": 0, "outliers": 0,
            "fallback_segments": 0, "fallback_unretained": 0,
            "forget_tokens": 0, "forget_segments": 0,
        }
        # id(segment) -> (segment, dense tensor) for segments whose outlier
        # ratio exceeded _OUTLIER_FALLBACK_RATIO. Bounded by bytes, LRU.
        self._dense_fallback: "OrderedDict[int, tuple]" = OrderedDict()
        self._dense_fallback_bytes = 0

    # -- storage helpers ---------------------------------------------------
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        # Deliberately NOT calling super().lazy_initialization(): the base
        # DynamicLayer sets self.keys/self.values to key_states/value_states
        # directly, which this class no longer uses as its residual storage
        # (see _resid_k/_resid_v). Set the flag it manages ourselves instead.
        self.is_initialized = True
        batch = key_states.shape[0]
        self._banded_k = [[] for _ in range(batch)]
        self._banded_v = [[] for _ in range(batch)]
        self._seg_len = [[] for _ in range(batch)]
        self._banded_len = [0] * batch
        self._heads = None
        self._dtype = key_states.dtype
        self._device = key_states.device
        self._resid_k = [self._empty_resid(key_states) for _ in range(batch)]
        self._resid_v = [self._empty_resid(value_states) for _ in range(batch)]
        self._pad_lens: list | None = None
        # nogather's per-segment dense-reconstruction cache -- see
        # _unband_cached's docstring.
        self._unband_cache: OrderedDict[int, tuple] = OrderedDict()
        self._unband_bytes = 0
        # Per-slot cache of the CONCATENATED banded history, bounded by
        # bytes both per layer and process-wide. Eviction is a cache miss,
        # never a correctness event.
        self._joined_cache: "OrderedDict[int, tuple]" = OrderedDict()
        self._joined_bytes = 0

    def _empty_resid(self, like: torch.Tensor) -> torch.Tensor:
        """A (1, heads, 0, head_dim) placeholder for one slot's residual,
        same heads/head_dim/dtype/device as `like` (a real (batch, heads,
        seq, head_dim) tensor from update())."""
        return like.new_zeros((1, like.shape[1], 0, like.shape[3]))

    def _check_batch_size(self, batch: int) -> None:
        """Phase 2: batch size may now GROW (a new slot admitted) or SHRINK
        (a finished slot dropped) between update() calls -- add_slot/
        drop_slot below are the supported way to do that explicitly; this
        check only guards against update() being handed a batch size that
        neither matches the current slot count NOR came through one of
        those methods, which would otherwise silently misalign every
        per-slot list against the wrong batch position."""
        if batch != len(self._resid_k):
            raise RuntimeError(
                f"BandedKVLayer.update(): batch size is {batch}, but this "
                f"cache holds {len(self._resid_k)} slots -- use add_slot()/"
                "drop_slot() to change slot count between update() calls, "
                "not a different incoming batch size.")

    def add_slot(self, key_like: torch.Tensor, value_like: torch.Tensor) -> int:
        """Admit a new, empty slot (a new request joining the batch).
        Returns its slot index. `key_like`/`value_like`: any (1, heads, 0,
        head_dim)-compatible tensor on this cache's dtype/device, used only
        to read heads/head_dim -- e.g. the new request's own first
        key_states before its first update() call. Must be called BEFORE
        the update() whose incoming batch includes this new slot, since
        update() itself refuses a batch-size change (see _check_batch_size)."""
        if not self.is_initialized:
            self.lazy_initialization(key_like, value_like)
            return 0
        bi = len(self._resid_k)
        self._drop_joined_cache()
        self._banded_k.append([])
        self._banded_v.append([])
        self._seg_len.append([])
        self._banded_len.append(0)
        self._resid_k.append(self._empty_resid(key_like))
        self._resid_v.append(self._empty_resid(value_like))
        return bi

    def export_slot(self, bi: int) -> dict:
        """One slot's cached history, cheap enough to keep around at 100k+ tokens.
        Banded segments are immutable (a flush appends, a crop replaces), so they
        are SHARED by reference; only the short dense residual is copied."""
        return {"banded_k": list(self._banded_k[bi]), "banded_v": list(self._banded_v[bi]),
                "seg_len": list(self._seg_len[bi]), "banded_len": self._banded_len[bi],
                "heads": self._heads, "resid_k": self._resid_k[bi].clone(), "resid_v": self._resid_v[bi].clone()}

    def import_slot(self, bi: int, state: dict) -> None:
        """Put a history from export_slot() into the (empty) slot `bi`."""
        if self._banded_len[bi] or self._resid_k[bi].shape[-2]:
            raise RuntimeError(f"import_slot: slot {bi} is not empty")
        self._banded_k[bi] = list(state["banded_k"])
        self._banded_v[bi] = list(state["banded_v"])
        self._seg_len[bi] = list(state["seg_len"])
        self._banded_len[bi] = state["banded_len"]
        if self._heads is None:                 # set by the first flush; a fresh layer has none yet
            self._heads = state["heads"]
        self._resid_k[bi] = state["resid_k"].clone()
        self._resid_v[bi] = state["resid_v"].clone()
        self._drop_joined_cache()

    def drop_slot(self, bi: int) -> None:
        """Remove a finished/cancelled slot. Every slot AFTER `bi` shifts
        down one index -- callers (the scheduler) must remap their own
        slot->request tracking accordingly, the same contract array
        deletion always has."""
        for lst in (self._banded_k, self._banded_v, self._seg_len,
                    self._banded_len, self._resid_k, self._resid_v):
            del lst[bi]
        # _joined_cache is keyed by slot index, and every slot after bi
        # just moved -- a stale entry would hand one slot another's
        # history. Drop it wholesale; it refills on the next forward.
        self._drop_joined_cache()

    def _band_tensor(self, t: torch.Tensor, bi: int) -> list:
        """Band a (1, heads, seq, head_dim) single-slot tensor via the
        fixed-tau codec: index = floor((value-min)/(2*tau)), no codebook
        search, no frequency table to go stale -- every flush calibrates
        independently at near-zero cost (see kv_band.py's
        FixedTauDirection docstring).
        """
        b, h, s, d = t.shape
        assert b == 1, f"_band_tensor takes one slot at a time, got batch={b}"
        # Band from fp32 (exact for bf16/fp16 inputs); the codec certifies the
        # error after casting back to orig_dtype. _decode_banded casts to the
        # model dtype on the way out, so nothing downstream sees fp32.
        src = t.float() if t.dtype in (torch.float16, torch.bfloat16) else t
        orig_dtype = t.dtype

        # Per-channel certified codec (kv_codec.py): runs where the data is
        # (GPU or CPU), no escapes, every element within self.tau.
        return band_kv_heads(src[0], orig_dtype, KV_BITS_FLOOR, self.tau)

    def _flush_residual(self, bi: int) -> None:
        """Band slot `bi`'s dense residual and append ONE new segment to
        that slot's own segment list. Slots flush INDEPENDENTLY now (Phase
        2) -- one slot crossing residual_length does not touch any other
        slot's residual or segment list, which is exactly what per-slot
        residual storage (_resid_k/_resid_v) exists to allow.
        """
        r = self._resid_k[bi]
        if r is None or r.numel() == 0:
            return
        self.metrics["flushes"] += 1
        self.metrics["flush_tokens"] += int(r.shape[-2])
        _, h, s, d = r.shape
        self._heads = h
        # APPEND a new banded SEGMENT; never re-band what is already
        # banded. An earlier (pre-batching) version decoded the whole
        # existing store, concatenated the residual densely, and re-banded
        # the lot on every flush -- that made residual_length meaningless
        # (every flush re-derived one codebook over the entire history),
        # cost O(total_seq) per flush instead of O(residual_length), and
        # threw away already-paid calibration work each time.
        for slot_store, resid in ((self._banded_k, self._resid_k),
                                  (self._banded_v, self._resid_v)):
            seg = self._band_tensor(resid[bi], bi)
            slot_store[bi].append(seg)
            outliers = sum(bk.outlier_count for bk in seg)
            self.metrics["outliers"] += outliers
            # The dense decode is the expensive part: it materialises a
            # full history-sized tensor. Do NOT run it unless the copy can
            # actually be RETAINED -- doing it and then discarding the result
            # (because dense fallback is off, or the segment exceeds budget)
            # burns the work AND allocates exactly the transient peak that
            # OOMs a tight-VRAM model, for no benefit whatsoever.
            if (outliers > _OUTLIER_FALLBACK_RATIO * s * self._heads
                    and self._dense_fallback_enabled()
                    and self._device is not None and self._dtype is not None):
                self.metrics["fallback_segments"] += 1
                dense = unband_kv_4d(seg, 1, h, device=self._device)
                self._register_dense_fallback(seg, dense.to(self._dtype))
                del dense
        self._seg_len[bi].append(s)
        self._banded_len[bi] += s
        self._resid_k[bi] = self._resid_k[bi][..., :0, :]
        self._resid_v[bi] = self._resid_v[bi][..., :0, :]

    @staticmethod
    def _dense_fallback_enabled() -> bool:
        """True only if a dense copy could actually be retained.

        Both the per-layer and the cross-layer budget can be 0, and a segment
        larger than the budget is still rejected inside
        _register_dense_fallback -- so the cheap check lives here, BEFORE the
        decode, and the authoritative one stays there.
        """
        return (_DENSE_FALLBACK_BYTES > 0
                and _DENSE_FALLBACK_CAP_BYTES > 0)

    def _register_dense_fallback(self, seg: list, tensor: torch.Tensor) -> None:
        """Remember a DENSE copy of a high-outlier segment, byte-bounded.

        A dense segment is decoded exactly once here; every later read is a
        plain tensor reuse. Bounded like _unband_cache, and invalidated
        wholesale wherever the segment lists are (reset/drop_slot/crop/
        forget) -- dense copies are what must never survive a conversation.
        """
        key = id(seg)
        hit = self._dense_fallback.get(key)
        if hit is not None and hit[0] is seg:
            self._dense_fallback.move_to_end(key)
            return
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > min(_DENSE_FALLBACK_BYTES, _DENSE_FALLBACK_CAP_BYTES):
            # Cannot be retained; segment still decodes correctly, just
            # without the fast path. Checked against BOTH caps: a segment
            # that clears the per-layer cap on its own would, repeated
            # across every layer, exceed the cross-layer cap N times over
            # while every individual layer's accounting still looked fine.
            self.metrics["fallback_unretained"] += 1
            return
        while (self._dense_fallback
               and (self._dense_fallback_bytes + nbytes > _DENSE_FALLBACK_BYTES
                    or _GLOBAL_DENSE_FALLBACK_BYTES + nbytes > _DENSE_FALLBACK_CAP_BYTES)):
            _k, (_old_seg, old_t) = self._dense_fallback.popitem(last=False)
            freed = old_t.numel() * old_t.element_size()
            self._dense_fallback_bytes -= freed
            _dense_fallback_global_subtract(freed)
        self._dense_fallback[key] = (seg, tensor)
        self._dense_fallback_bytes += nbytes
        _dense_fallback_global_add(nbytes)

    def _unband_cached(self, seg: list, dtype: torch.dtype, device) -> torch.Tensor:
        """unband_kv_4d(seg, ...), cached by id(seg) + (device, dtype).

        Without this cache, _decode_banded re-decodes AND re-uploads EVERY
        accumulated segment from scratch on EVERY forward call. `seg` (one
        flush's list of per-head BandedKV) is immutable once flushed and
        keeps a stable id() across later flushes (append-only; a segment
        is never rebuilt in place, see _flush_residual's docstring), so
        caching on it here is safe. Bounded + FIFO-evicted: no GC hook
        fires exactly when a segment dies (crop() dropping it, or a new
        conversation replacing the whole store).
        """
        key = id(seg)
        # A registered DENSE fallback short-circuits the whole decode -- this
        # is the read-side payoff of the outlier-ratio policy.
        dense = self._dense_fallback.get(key)
        if dense is not None and dense[0] is seg and dense[1].device.type == device.type \
                and dense[1].dtype == dtype:
            self._dense_fallback.move_to_end(key)
            return dense[1]
        hit = self._unband_cache.get(key)
        if hit is not None and hit[0] is seg and hit[1] == device and hit[2] == dtype:
            # Touch: move to the end so eviction below is LRU, not FIFO
            # -- a long context re-reads its oldest segments every
            # forward, and FIFO would evict exactly the ones about to be
            # needed again.
            self._unband_cache.move_to_end(key)
            return hit[3]
        t = unband_kv_4d(seg, 1, self._heads, device=device).to(dtype)
        nbytes = t.numel() * t.element_size()
        # Evict least-recently-used entries until this one fits. Bounded
        # by BYTES, not entry count -- see _UNBAND_CACHE_BYTES for why
        # that distinction was a real, measured memory bug.
        while (self._unband_cache
               and (self._unband_bytes + nbytes > _UNBAND_CACHE_BYTES
                    or _GLOBAL_UNBAND_BYTES + nbytes > _UNBAND_CACHE_CAP_BYTES)):
            _k, old = self._unband_cache.popitem(last=False)
            freed = old[3].numel() * old[3].element_size()
            self._unband_bytes -= freed
            _unband_global_subtract(freed)
        if nbytes > min(_UNBAND_CACHE_BYTES, _UNBAND_CACHE_CAP_BYTES):
            # Single segment bigger than the whole budget: hand it back
            # without caching rather than blowing the budget. Both caps,
            # for the same cross-layer reason as _register_dense_fallback.
            return t
        self._unband_cache[key] = (seg, device, dtype, t)
        self._unband_bytes += nbytes
        _unband_global_add(nbytes)
        return t

    def _decode_banded(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode every slot's banded segments and stack back into a real
        (batch, heads, seq, head_dim) tensor PER SLOT (each slot's own
        banded length, no padding here -- see update() for where slots get
        padded to a common batch width). Returns (list of per-slot K
        tensors, list of per-slot V tensors), one entry per batch position,
        possibly different seq lengths.
        """
        dev = self._device if self._device is not None else "cpu"
        b = len(self._banded_k)
        per_slot_k, per_slot_v = [], []
        for bi in range(b):
            segs_k, segs_v = self._banded_k[bi], self._banded_v[bi]
            if not segs_k:
                hd = self._resid_k[bi].shape[-1] if self._resid_k[bi] is not None else 0
                h = self._heads or (self._resid_k[bi].shape[1] if self._resid_k[bi] is not None else 1)
                per_slot_k.append(torch.zeros(1, h, 0, hd, dtype=dtype, device=dev))
                per_slot_v.append(torch.zeros(1, h, 0, hd, dtype=dtype, device=dev))
                continue
            ks = [self._unband_cached(seg, dtype, dev) for seg in segs_k]
            vs = [self._unband_cached(seg, dtype, dev) for seg in segs_v]
            per_slot_k.append(torch.cat(ks, dim=-2) if len(ks) > 1 else ks[0])
            per_slot_v.append(torch.cat(vs, dim=-2) if len(vs) > 1 else vs[0])
        return per_slot_k, per_slot_v

    # -- official API ------------------------------------------------------
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reads self._step_real_lens (continuous-batching callers only,
        see ContinuousBatchEngine.step) instead of taking a real_lens
        KWARG -- the stock model forward (Qwen3_5TextModel.forward ->
        past_key_values.update(key_states, value_states, self.layer_idx))
        calls update() itself with no way for an outside caller to inject
        an extra per-call argument through that path, so the only place
        left to hand this cache extra per-step information is an
        attribute set on the cache object BEFORE calling model(...) --
        ContinuousBatchEngine.step() does exactly that
        (self.past._step_real_lens = [...]) right before its forward, and
        clears it right after. None (the default outside that one caller)
        preserves the exact old behaviour: every column of
        key_states[bi] is real.

        real_lens[bi]'s columns are taken from the END of
        key_states[bi]/value_states[bi] (the LAST real_lens[bi] columns),
        matching this whole file's LEFT-padding convention -- real
        content right-aligned, pad columns at the start (see
        _build_left_pad_inputs, the only input builder a continuous-
        batching caller uses; there is no right-padded caller any more).

        WHAT THIS SOLVES: a caller sharing ONE forward across slots of
        genuinely different new-token counts must pad the shorter rows to
        build a single batch tensor at all. update() has no way to tell
        pad columns from real content on its own (every column is just a
        normal K/V tensor value) -- appending them to a slot's residual
        was a real, confirmed bug: crop() removes committed tokens from
        the END, but BandedKVLayer._pad_and_stack (used to build the
        DENSE tensor attention actually reads) is hardcoded to LEFT-pad,
        which is independent of whichever side a caller's own input_ids
        padding used. Whichever side update()'s INPUT pad columns land
        on, cropping them back out after committing them can strip
        genuine content instead of the pad (confirmed by direct
        reproduction: a decoding request degenerated into a repeated
        token a few steps after a differently-shaped row joined the same
        batched forward). Dropping the pad columns HERE, before they are
        ever appended, root-fixes the problem instead of trying to undo
        it afterward.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        segs = getattr(self, "_step_segments", None)
        if segs is not None:
            # PACKED step (ContinuousBatchEngine, pure-attention models): one row holding every slot's
            # tokens back to back, no padding; segment j = (slot, offset, length) of that row.
            self._pad_lens = None
            for bi, off, n in segs:
                k_bi = key_states[:, :, off:off + n]
                v_bi = value_states[:, :, off:off + n]
                is_prefill = (self._banded_len[bi] == 0 and self._resid_k[bi].shape[-2] == 0 and n > 1)
                self._resid_k[bi] = torch.cat([self._resid_k[bi], k_bi], dim=-2)
                self._resid_v[bi] = torch.cat([self._resid_v[bi], v_bi], dim=-2)
                if self._resid_k[bi].shape[-2] > self.residual_length or is_prefill:
                    self._flush_residual(bi)
            return key_states, value_states
        self._check_batch_size(key_states.shape[0])
        b = key_states.shape[0]
        # Reset every call, BEFORE any padding decision below -- a stale
        # _pad_lens from a PREVIOUS update() (where slots had diverged)
        # would otherwise leak forward via get_padding_mask()'s getattr
        # fallback on a LATER call that needs no padding at all (slots
        # re-synced, or batch=1), making a caller apply a mask for
        # padding that is not actually in this call's returned tensor.
        self._pad_lens = None
        real_lens = getattr(self, "_step_real_lens", None)

        for bi in range(b):
            k_bi = key_states[bi:bi + 1]
            v_bi = value_states[bi:bi + 1]
            if real_lens is not None:
                rl = real_lens[bi]
                if rl < k_bi.shape[-2]:
                    k_bi = k_bi[..., -rl:, :] if rl > 0 else k_bi[..., :0, :]
                    v_bi = v_bi[..., -rl:, :] if rl > 0 else v_bi[..., :0, :]
            # TRUE prefill only, PER SLOT: the very first forward into an
            # empty slot. NOT simply "seq > 1" -- speculative decoding's
            # verify step also passes multiple tokens ([last_token] +
            # draft), and treating that as prefill banded the DRAFT --
            # unverified tokens committed to the banded store on every
            # step, forcing a decode-slice-reband in crop() on every
            # rejection, and making residual_length meaningless again
            # because every step flushed. Drafts must stay in the dense
            # residual until they are accepted.
            resid_len = self._resid_k[bi].shape[-2]
            is_prefill = (self._banded_len[bi] == 0 and resid_len == 0
                         and k_bi.shape[-2] > 1)
            self._resid_k[bi] = torch.cat(
                [self._resid_k[bi], k_bi], dim=-2)
            self._resid_v[bi] = torch.cat(
                [self._resid_v[bi], v_bi], dim=-2)
            # Flush when THIS SLOT's residual exceeds its budget, OR on
            # that slot's true prefill -- independently of every other
            # slot (Phase 2). Without the prefill clause a prompt shorter
            # than residual_length never compresses at all -- measured: a
            # 115-token prompt at residual_length=128 left banded_bytes=0
            # and the whole cache dense.
            if (self._resid_k[bi].shape[-2] > self.residual_length
                    or is_prefill):
                self._flush_residual(bi)

        # Ragged/varlen callers (ContinuousBatchEngine) never read this
        # method's dense return value -- they call
        # reconstruct_per_slot() themselves, consume it inside the
        # attention kernel, and drop it. Building a dense reconstruction
        # here anyway, and (worse) keeping a reference to it on the
        # layer, kept a FULL DENSE COPY OF EVERY SLOT'S ENTIRE HISTORY
        # resident on all six banded layers between calls.
        #
        # Measured on a real 23801-token prompt: banded store 0.080GiB,
        # the retained dense copy 0.272GiB -- 3.4x the compressed store
        # it exists to compress, doubling again with a second concurrent
        # request. That is the opposite of what a certified-bounded
        # compressed cache is for, and it is where this server's VRAM
        # was actually going.
        if getattr(self, "_step_real_lens", None) is not None:
            return key_states, value_states

        if self.window is not None:
            # Sliding layer: attention needs only the last window-1 keys before each new token, so return that
            # tail (plus the new tokens) instead of the whole history. The fused kernel has no window -> not used.
            q_new = key_states.shape[-2]
            win_k, win_v = [], []
            for bi in range(b):
                k_i, v_i = self.window_kv_slot(bi, self.window - 1 + q_new, key_states.dtype, key_states.device)
                win_k.append(k_i)
                win_v.append(v_i)
            if len({t.shape[-2] for t in win_k}) == 1:
                return torch.cat(win_k, dim=0), torch.cat(win_v, dim=0)
            return self._pad_and_stack(win_k, win_v)

        # Fused decode attention (fused_attn.py): the attention wrapper reads the
        # banded ids + dense residual itself, so no O(context) tensor is built
        # here. Single slot, a verify-width query, and a banded history worth it.
        _mq = _fused_max_q(key_states.device) if FUSED_ENABLED and key_states.is_cuda else 0
        if (FUSED_ENABLED and b == 1 and 1 <= key_states.shape[-2] and (_mq is None or key_states.shape[-2] <= _mq)
                and self._banded_len[0] >= _FUSED_MIN_BANDED):
            from . import fused_attn
            fk = getattr(self, "_fused", None)
            if fk is None:
                fk = self._fused = fused_attn.FusedKV()
            if fk.sync(self) is not None:
                fused_attn.set_pending(self)
                return key_states, value_states

        from . import fused_attn as _fa
        if getattr(_fa._TLS, "require", False):
            # pipe_prefill.py builds no attention mask: only the fused path is valid there
            raise RuntimeError("pipelined prefill reached the dense attention path")
        per_slot_k, per_slot_v = self.reconstruct_per_slot(key_states.dtype)
        lens = [t.shape[-2] for t in per_slot_k]
        if len(set(lens)) == 1:
            return torch.cat(per_slot_k, dim=0), torch.cat(per_slot_v, dim=0)
        return self._pad_and_stack(per_slot_k, per_slot_v)

    def reconstruct_per_slot(self, dtype: torch.dtype) -> tuple[list, list]:
        """Each slot's full (banded ++ residual) history as an UNPADDED
        (1, heads, seq_i, head_dim) tensor -- the shared computation
        both update()'s dense (padded) return value and the ragged/
        varlen path build from. Factored out of update() so the varlen
        path never needs a padded intermediate at all.
        """
        if not any(self._banded_k):
            return list(self._resid_k), list(self._resid_v)
        bk_list, bv_list = self._decode_banded_joined(dtype)
        full_k = [torch.cat([bk_list[bi], self._resid_k[bi]], dim=-2)
                 if self._resid_k[bi].shape[-2] else bk_list[bi]
                 for bi in range(len(bk_list))]
        full_v = [torch.cat([bv_list[bi], self._resid_v[bi]], dim=-2)
                 if self._resid_v[bi].shape[-2] else bv_list[bi]
                 for bi in range(len(bv_list))]
        return full_k, full_v

    def window_kv_slot(self, bi: int, need: int, dtype: torch.dtype, dev) -> tuple[torch.Tensor, torch.Tensor]:
        """The LAST `need` tokens of slot bi's history (banded ++ residual) as (1, heads, n, head_dim), n <= need.
        Decodes only the trailing segments that overlap that range (through the per-segment unband cache), so a
        sliding-window layer costs O(window) per step, not O(context)."""
        rk, rv = self._resid_k[bi], self._resid_v[bi]
        r = rk.shape[-2]
        if r >= need or not self._banded_k[bi]:
            return rk[..., r - min(r, need):, :], rv[..., r - min(r, need):, :]
        ks, vs, have = [rk], [rv], r
        for seg_k, seg_v, n_seg in zip(reversed(self._banded_k[bi]), reversed(self._banded_v[bi]),
                                       reversed(self._seg_len[bi])):
            ks.insert(0, self._unband_cached(seg_k, dtype, dev))
            vs.insert(0, self._unband_cached(seg_v, dtype, dev))
            have += n_seg
            if have >= need:
                break
        k = torch.cat([t.to(dtype) for t in ks], dim=-2)
        v = torch.cat([t.to(dtype) for t in vs], dim=-2)
        n = min(need, k.shape[-2])
        return k[..., k.shape[-2] - n:, :], v[..., v.shape[-2] - n:, :]

    def _decode_seg_dense(self, seg: list, dtype: torch.dtype, dev) -> torch.Tensor:
        """One segment decoded to (1, heads, seq, head_dim), uncached (the
        caller keeps the result in its own buffer). A registered dense
        fallback copy is reused when there is one."""
        dense = self._dense_fallback.get(id(seg))
        if dense is not None and dense[0] is seg and dense[1].dtype == dtype:
            return dense[1]
        return unband_kv_4d(seg, 1, self._heads, device=dev).to(dtype)

    def ragged_kv_views(self, dtype: torch.dtype, rows) -> dict | None:
        """Each listed slot's full (banded ++ residual) history as a pair of
        contiguous SEQ-MAJOR views (1, seq_i, heads, head_dim) into a
        per-slot buffer that persists across steps -- or None when a buffer
        would exceed the byte budget, in which case the caller uses
        reconstruct_per_slot().

        WHY: the ragged attention path used to rebuild every slot's history
        per layer per step -- torch.cat(banded, residual), then transpose to
        the (seq, heads, dim) layout the kernel wants, then cat across slots
        -- three O(context) copies to produce one token. Here the banded part
        is decoded into the buffer ONCE, when its segment is appended, and a
        decode step only rewrites the short dense residual behind it
        (O(residual), <= residual_length + draft tokens). Layout is
        seq-major so a prefix slice is contiguous and goes to the kernel with
        no further copy.

        Validity is keyed on segment IDENTITY (the segment objects
        themselves are held, so a recycled id() can never alias): the
        longest common prefix of the buffer's segments and the slot's
        current segments stays valid, everything after is re-decoded. An
        append (flush) extends the prefix; a crop that drops trailing
        segments only shrinks it; a crop that re-bands the straddling
        segment replaces just that one.
        """
        global _GLOBAL_SM_BYTES
        views = getattr(self, "_sm_views", None)
        if views is None:
            views = self._sm_views = {}
            self._sm_bytes = 0
        dev = self._device if self._device is not None else "cpu"
        out = {}
        for bi in rows:
            rk, rv = self._resid_k[bi], self._resid_v[bi]
            r = rk.shape[-2]
            n_b = self._banded_len[bi]
            need = n_b + r
            h, d = rk.shape[1], rk.shape[3]
            segs_k, segs_v = self._banded_k[bi], self._banded_v[bi]
            e = views.get(bi)
            if e is not None and (e["dtype"] != dtype or e["h"] != h or e["d"] != d
                                  or e["dev"] != dev):
                _GLOBAL_SM_BYTES = max(0, _GLOBAL_SM_BYTES - e["bytes"])
                self._sm_bytes -= e["bytes"]
                e = views.pop(bi)
                e = None
            if e is None:
                e = {"dtype": dtype, "h": h, "d": d, "dev": dev, "cap": 0,
                     "bytes": 0, "k": None, "v": None, "segk": [], "segv": [],
                     "valid": 0}
            # Longest prefix of segments still identical to the buffer's.
            m = 0
            lim = min(len(e["segk"]), len(segs_k))
            while (m < lim and e["segk"][m] is segs_k[m]
                   and e["segv"][m] is segs_v[m]):
                m += 1
            valid = sum(self._seg_len[bi][:m])
            if need > e["cap"]:
                per_tok = 2 * h * d * torch.empty((), dtype=dtype).element_size()
                # Growth headroom must never push a slot that FITS over the
                # budget: cap it at what the budgets allow, so the view covers
                # exactly the advertised tokens (budget // per_tok), not
                # budget / 1.25 -- past that the slot fell off the fast path.
                room = min(_SM_VIEW_BYTES,
                           _SM_VIEW_CAP_BYTES - _GLOBAL_SM_BYTES + e["bytes"],
                           _SM_VIEW_BYTES - self._sm_bytes + e["bytes"])
                # Spare room for the next tokens: a fixed ~2k, not a quarter of the context. 25% of an
                # 80k-token slot was ~1.1 GB of unused dense K/V; a growth copy every ~2k tokens costs
                # milliseconds.
                new_cap = min(need + 2048, room // per_tok)
                nbytes = new_cap * per_tok
                if new_cap < need:
                    # Over budget: release this slot's buffer and let the
                    # caller rebuild the old way.
                    if bi in views:
                        _GLOBAL_SM_BYTES = max(0, _GLOBAL_SM_BYTES - e["bytes"])
                        self._sm_bytes -= e["bytes"]
                        del views[bi]
                    return None
                try:
                    nk = torch.empty(1, new_cap, h, d, dtype=dtype, device=dev)
                    nv = torch.empty(1, new_cap, h, d, dtype=dtype, device=dev)
                except torch.OutOfMemoryError:
                    # No room for a bigger view: drop this slot's view and use
                    # the rebuild path (always correct), as over budget.
                    nk = nv = None
                    if bi in views:
                        _GLOBAL_SM_BYTES = max(0, _GLOBAL_SM_BYTES - e["bytes"])
                        self._sm_bytes -= e["bytes"]
                        del views[bi]
                    return None
                if e["k"] is not None and valid:
                    nk[:, :valid].copy_(e["k"][:, :valid])
                    nv[:, :valid].copy_(e["v"][:, :valid])
                _GLOBAL_SM_BYTES = max(0, _GLOBAL_SM_BYTES - e["bytes"]) + nbytes
                self._sm_bytes += nbytes - e["bytes"]
                e.update(k=nk, v=nv, cap=new_cap, bytes=nbytes)
            pos = valid
            for si in range(m, len(segs_k)):
                kt = self._decode_seg_dense(segs_k[si], dtype, dev)
                vt = self._decode_seg_dense(segs_v[si], dtype, dev)
                sl = kt.shape[-2]
                e["k"][0, pos:pos + sl].copy_(kt[0].transpose(0, 1))
                e["v"][0, pos:pos + sl].copy_(vt[0].transpose(0, 1))
                pos += sl
            e["segk"], e["segv"] = list(segs_k), list(segs_v)
            e["valid"] = pos
            if r:
                e["k"][0, pos:pos + r].copy_(rk[0].transpose(0, 1))
                e["v"][0, pos:pos + r].copy_(rv[0].transpose(0, 1))
            views[bi] = e
            out[bi] = (e["k"][:, :need], e["v"][:, :need])
        return out

    def _decode_banded_joined(self, dtype: torch.dtype) -> tuple[list, list]:
        """_decode_banded(), but the per-slot concatenation of segments
        is cached and reused across calls.

        WHY: the banded portion of a slot's history only changes when a
        flush APPENDS a segment or a crop DROPS one -- never on an
        ordinary decode step, where only the dense residual grows. Yet
        reconstruct_per_slot ran a full torch.cat over every segment on
        every forward, allocating a history-sized tensor per layer per
        step. At 24k context that dominates a decode step, which
        produces a single token: the work is O(context) where it should
        be O(new tokens). Keying the cache on the segment LIST IDENTITY
        (its length plus the id() of each segment) makes an append or a
        crop miss naturally, while a plain decode step hits.
        """
        cache = getattr(self, "_joined_cache", None)
        if not isinstance(cache, OrderedDict):
            cache = self._joined_cache = OrderedDict()
        dev = self._device if self._device is not None else "cpu"
        out_k, out_v = [], []
        for bi in range(len(self._banded_k)):
            segs_k = self._banded_k[bi]
            key = (len(segs_k), tuple(id(s) for s in segs_k), str(dev), dtype)
            hit = cache.get(bi)
            if hit is not None and hit[0] == key:
                cache.move_to_end(bi)
                out_k.append(hit[1])
                out_v.append(hit[2])
                continue
            if hit is not None:
                old = cache.pop(bi)
                freed = old[1].numel() * old[1].element_size()
                freed += old[2].numel() * old[2].element_size()
                self._joined_bytes -= freed
                _joined_global_subtract(freed)
            k1, v1 = self._decode_banded_slot(bi, dtype, dev)
            nbytes = (k1.numel() * k1.element_size()
                      + v1.numel() * v1.element_size())
            if not hasattr(self, "_joined_bytes"):
                self._joined_bytes = 0
            while cache and (self._joined_bytes + nbytes > _JOINED_CACHE_BYTES
                             or _GLOBAL_JOINED_BYTES + nbytes > _JOINED_CACHE_CAP_BYTES):
                _old_bi, old = cache.popitem(last=False)
                freed = old[1].numel() * old[1].element_size()
                freed += old[2].numel() * old[2].element_size()
                self._joined_bytes -= freed
                _joined_global_subtract(freed)
            if nbytes <= min(_JOINED_CACHE_BYTES, _JOINED_CACHE_CAP_BYTES):
                cache[bi] = (key, k1, v1)
                self._joined_bytes += nbytes
                _joined_global_add(nbytes)
            out_k.append(k1)
            out_v.append(v1)
        return out_k, out_v

    def _decode_banded_slot(self, bi: int, dtype: torch.dtype, dev):
        """One slot's banded segments decoded and concatenated."""
        segs_k, segs_v = self._banded_k[bi], self._banded_v[bi]
        if not segs_k:
            hd = self._resid_k[bi].shape[-1] if self._resid_k[bi] is not None else 0
            h = self._heads or (self._resid_k[bi].shape[1]
                                if self._resid_k[bi] is not None else 1)
            z = torch.zeros(1, h, 0, hd, dtype=dtype, device=dev)
            return z, z.clone()
        ks = [self._unband_cached(seg, dtype, dev) for seg in segs_k]
        vs = [self._unband_cached(seg, dtype, dev) for seg in segs_v]
        return (torch.cat(ks, dim=-2) if len(ks) > 1 else ks[0],
                torch.cat(vs, dim=-2) if len(vs) > 1 else vs[0])
    def _pad_and_stack(self, per_slot_k: list, per_slot_v: list
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """LEFT-pad every slot's (1, heads, seq_i, head_dim) tensor up to
        the batch's max seq_i, then stack into one (batch, heads,
        seq_max, head_dim) tensor. Left-padding (not right) matches the
        standard batched-decode convention: real tokens stay right-
        aligned, so `cache_position`/rotary position ids computed from
        the END of the sequence are correct for every slot without a
        per-slot offset, and the causal mask this cache's OWN padding
        produces (see get_padding_mask) simply masks out the pad
        columns on the left, same shape as right-padding's mask would
        mask out the right -- just conventionally the side transformers'
        own generate() already expects for a left-padded batch.

        Records self._pad_lens (this call's per-slot pad width) so
        get_padding_mask() can build the matching key-side mask; a caller
        building attention_mask itself needs both pieces (the padded
        tensor's shape, and which columns in it are real vs pad).
        """
        seq_max = max(t.shape[-2] for t in per_slot_k)
        self._pad_lens = [seq_max - t.shape[-2] for t in per_slot_k]
        padded_k, padded_v = [], []
        for k, v, pad in zip(per_slot_k, per_slot_v, self._pad_lens):
            if pad:
                zk = k.new_zeros((1, k.shape[1], pad, k.shape[3]))
                zv = v.new_zeros((1, v.shape[1], pad, v.shape[3]))
                k = torch.cat([zk, k], dim=-2)
                v = torch.cat([zv, v], dim=-2)
            padded_k.append(k)
            padded_v.append(v)
        return torch.cat(padded_k, dim=0), torch.cat(padded_v, dim=0)

    def get_padding_mask(self) -> torch.Tensor | None:
        """(batch, seq_max) bool tensor, True where a position is REAL
        (not left-padding), matching the shape update()'s last return
        padded to. None if the last update() call needed no padding (the
        common case: every slot at equal progress, or batch=1) -- a
        caller should treat None as "no masking needed beyond the causal
        mask" rather than "call this again", since _pad_lens is only set
        on a call that actually padded.
        """
        pad_lens = self._pad_lens
        if pad_lens is None or not any(pad_lens):
            return None
        # seq_max derived from get_seq_length() per slot (banded+residual)
        # plus its pad width.
        lens = [self._banded_len[bi] + self._resid_k[bi].shape[-2]
               for bi in range(len(self._resid_k))]
        seq_max = max(l + p for l, p in zip(lens, pad_lens))
        mask = torch.zeros(len(pad_lens), seq_max, dtype=torch.bool,
                           device=self._device)
        for bi, (length, pad) in enumerate(zip(lens, pad_lens)):
            mask[bi, pad:pad + length] = True
        return mask

    def get_seq_length(self, slot: int | None = None) -> int:
        """Cache API is one scalar per layer conventionally, but Phase 2
        slots can genuinely diverge -- `slot` (default: max across all
        slots, matching the padded tensor width update() just returned)
        lets a caller ask about ONE slot specifically. Passing no slot
        keeps giving a sane answer for any single-slot (batch=1) caller,
        unchanged from before."""
        if not self._resid_k:
            return 0
        if slot is not None:
            return self._banded_len[slot] + self._resid_k[slot].shape[-2]
        return max(self._banded_len[bi] + self._resid_k[bi].shape[-2]
                  for bi in range(len(self._resid_k)))

    def crop(self, max_length: int, slot: int | None = None) -> None:
        """Roll the cache back, honouring HF's overloaded sign convention:
        negative means "drop this many", non-negative means "truncate to
        this absolute length". Speculative decoding relies on the negative
        form (see spec_decode.py's rollback path).

        `slot`: which batch element to crop. None (default) crops EVERY
        slot, each to its OWN max_length derived from ITS OWN current
        length (not a shared one) -- Phase 2's per-slot residual
        (_resid_k/_resid_v) means there is no shared tensor to keep in
        sync any more, so a "crop everyone the same relative amount"
        request naturally applies independently per slot without the
        divergence hazard the old shared-residual version had to guard
        against. An explicit `slot` index crops only that one.
        """
        if slot is not None:
            self._crop_slot(slot, max_length)
            return
        for bi in range(len(self._banded_len)):
            self._crop_slot(bi, max_length)

    def _crop_slot(self, bi: int, max_length: int) -> None:
        banded_len = self._banded_len[bi] if bi < len(self._banded_len) else 0
        residual = self._resid_k[bi].shape[-2] if self._resid_k[bi] is not None else 0
        seq_len = banded_len + residual
        if max_length <= 0:
            max_length = seq_len - abs(max_length)
        if seq_len <= max_length:
            return
        max_length = max(max_length, 0)

        if max_length >= banded_len:
            # Cut lands inside this slot's OWN dense residual; banded
            # store is intact. Each slot's residual is its own tensor now
            # (Phase 2), so truncating it never touches any other slot.
            keep = max_length - banded_len
            if keep < residual:
                self._resid_k[bi] = self._resid_k[bi][..., :keep, :]
                self._resid_v[bi] = self._resid_v[bi][..., :keep, :]
            return

        # Cut lands inside the banded store, so this slot's ENTIRE
        # residual is past max_length and must go too.
        self._resid_k[bi] = self._resid_k[bi][..., :0, :]
        self._resid_v[bi] = self._resid_v[bi][..., :0, :]

        # Whole segments before the cut survive untouched (each is
        # self-contained); the segment the cut falls inside is truncated by
        # slicing its packed ids (BandedKV.truncate: rows are a byte prefix and
        # the per-channel min_s/tau_s are kept, so nothing is re-quantised and
        # the certified error bound is not loosened); every later segment is
        # dropped.
        if max_length == 0:
            self._banded_k[bi], self._banded_v[bi], self._seg_len[bi] = [], [], []
            self._banded_len[bi] = 0
            self._drop_dense_fallbacks()
            return
        kept_k, kept_v, kept_len, acc = [], [], [], 0
        seg_list_k, seg_list_v, seg_lens = self._banded_k[bi], self._banded_v[bi], self._seg_len[bi]
        for si, seg_len in enumerate(seg_lens):
            if acc + seg_len <= max_length:
                kept_k.append(seg_list_k[si])
                kept_v.append(seg_list_v[si])
                kept_len.append(seg_len)
                acc += seg_len
                continue
            # this segment straddles the cut
            need = max_length - acc
            if need > 0:
                kept_k.append([bk.truncate(need) for bk in seg_list_k[si]])
                kept_v.append([bk.truncate(need) for bk in seg_list_v[si]])
                kept_len.append(need)
                acc += need
            break
        self._banded_k[bi], self._banded_v[bi], self._seg_len[bi] = kept_k, kept_v, kept_len
        self._banded_len[bi] = acc
        self._drop_dense_fallbacks()

    def plan_forget(self, token_indices=None, segment_indices=None,
                    slot: int | None = None) -> dict:
        """What a forget WOULD touch -- no mutation, no rebuild, no decode.

        Selective forget is the one cache operation expensive enough to be
        worth previewing: forget_tokens() rewrites every segment a target
        falls in (ids repacked, no decode/re-quantise) to drop a handful of positions. Callers doing privacy
        deletion need to audit scope BEFORE paying that, and a dry run must
        be safe to call on a live cache mid-generation.
        """
        n_slots = len(self._banded_k)
        slots = list(range(n_slots)) if slot is None else [slot]
        for bi in slots:
            if bi < 0 or bi >= n_slots:
                raise IndexError(f"slot {bi} out of range")
        out = {"slots": slots, "token_indices": [], "segment_indices": [],
               "would_remove_tokens": 0, "would_remove_segments": 0,
               "affected": []}
        if token_indices:
            out["token_indices"] = sorted({int(i) for i in token_indices if int(i) >= 0})
        if segment_indices:
            out["segment_indices"] = sorted({int(i) for i in segment_indices if int(i) >= 0})
        for bi in slots:
            segs = self._seg_len[bi] if bi < len(self._seg_len) else []
            total = self._banded_len[bi] if bi < len(self._banded_len) else 0
            # forget_tokens() masks the FULL reconstructed history, which is
            # banded ++ residual. Counting only the banded part would
            # under-report every position still living in the residual.
            resid = 0
            if self._resid_k and bi < len(self._resid_k) and self._resid_k[bi] is not None:
                resid = int(self._resid_k[bi].shape[-2])
            full = total + resid
            ntok = sum(1 for i in out["token_indices"] if i < full)
            nseg = sum(1 for i in out["segment_indices"] if i < len(segs))
            if ntok or nseg:
                out["affected"].append({"slot": bi, "tokens": ntok,
                                        "segments": nseg, "total_tokens": full})
            out["would_remove_tokens"] += ntok
            out["would_remove_segments"] += nseg
        return out

    def forget_tokens(self, token_indices, slot: int | None = None,
                      audit: list | None = None) -> int:
        """Remove selected absolute token positions.

        Segment by segment, without decoding the history: a segment with none of
        the targets is kept as is, one with all of them is dropped, and a partly
        hit one is rebuilt by BandedKV.take_rows (ids row-selected and repacked
        with the same per-channel min_s/tau_s -- no re-quantisation, so the
        certified bound is not loosened). Targets in the dense residual are masked
        out of it directly.
        """
        targets = {int(i) for i in token_indices if int(i) >= 0}
        if not targets:
            return 0
        slots = range(len(self._banded_k)) if slot is None else (slot,)
        removed = 0
        if audit is not None:
            audit.append(self.plan_forget(token_indices=targets, slot=slot))
        for bi in slots:
            if bi < 0 or bi >= len(self._banded_k):
                raise IndexError(f"slot {bi} out of range")
            banded_len = self._banded_len[bi]
            resid = self._resid_k[bi]
            n_resid = int(resid.shape[-2]) if resid is not None else 0
            valid = sorted(i for i in targets if i < banded_len + n_resid)
            n = len(valid)
            if not n:
                continue
            new_k, new_v, new_len, acc = [], [], [], 0
            for si, seg_len in enumerate(self._seg_len[bi]):
                hit = {t - acc for t in valid if acc <= t < acc + seg_len}
                if not hit:
                    new_k.append(self._banded_k[bi][si]); new_v.append(self._banded_v[bi][si])
                    new_len.append(seg_len)
                elif len(hit) < seg_len:
                    dev = self._banded_k[bi][si][0].packed.device
                    rows = torch.tensor([r for r in range(seg_len) if r not in hit],
                                        dtype=torch.int64, device=dev)
                    new_k.append([bk.take_rows(rows) for bk in self._banded_k[bi][si]])
                    new_v.append([bk.take_rows(rows) for bk in self._banded_v[bi][si]])
                    new_len.append(int(rows.numel()))
                acc += seg_len
            self._banded_k[bi], self._banded_v[bi], self._seg_len[bi] = new_k, new_v, new_len
            self._banded_len[bi] = sum(new_len)
            if n_resid:
                drop = [t - banded_len for t in valid if t >= banded_len]
                if drop:
                    mask = torch.ones(n_resid, dtype=torch.bool, device=resid.device)
                    mask[drop] = False
                    self._resid_k[bi] = self._resid_k[bi][..., mask, :]
                    self._resid_v[bi] = self._resid_v[bi][..., mask, :]
            self._drop_joined_cache()
            self._drop_unband_cache()
            self._drop_sm_views()
            self._drop_dense_fallbacks()
            self.metrics["forget_tokens"] += n
            removed += n
        return removed

    def forget_segments(self, segment_indices, slot: int | None = None,
                        audit: list | None = None) -> int:
        """Drop complete compressed segments without decoding them."""
        wanted = {int(i) for i in segment_indices if int(i) >= 0}
        if not wanted:
            return 0
        slots = range(len(self._banded_k)) if slot is None else (slot,)
        removed = 0
        if audit is not None:
            audit.append(self.plan_forget(segment_indices=wanted, slot=slot))
        for bi in slots:
            if bi < 0 or bi >= len(self._banded_k):
                raise IndexError(f"slot {bi} out of range")
            valid = set(range(len(self._seg_len[bi])))
            drop = wanted & valid
            keep = sorted(valid - drop)
            self._banded_k[bi] = [self._banded_k[bi][i] for i in keep]
            self._banded_v[bi] = [self._banded_v[bi][i] for i in keep]
            self._seg_len[bi] = [self._seg_len[bi][i] for i in keep]
            self._banded_len[bi] = sum(self._seg_len[bi])
            self._drop_joined_cache()
            self._drop_unband_cache()
            self._drop_dense_fallbacks()
            self.metrics["forget_segments"] += len(drop)
            removed += len(drop)
        return removed

    def metrics_report(self) -> dict:
        out = dict(self.metrics)
        out["segments"] = sum(len(x) for x in self._banded_k)
        out["banded_tokens"] = sum(self._banded_len)
        out["outlier_count"] = sum(
            bk.direction.outlier_count
            for slots in (self._banded_k, self._banded_v) for slot in slots
            for seg in slot for bk in seg
        )
        out["fallback_required"] = sum(
            1 for slots in (self._banded_k, self._banded_v) for slot in slots
            for seg in slot for bk in seg if bk.direction.outlier_ratio > 0.05
        )
        return out

    def _drop_joined_cache(self) -> None:
        """Release joined dense-history entries and their byte accounting."""
        for entry in getattr(self, "_joined_cache", {}).values():
            if entry and len(entry) == 3:
                k, v = entry[1], entry[2]
                _joined_global_subtract(k.numel() * k.element_size()
                                        + v.numel() * v.element_size())
        self._joined_cache = OrderedDict()
        self._joined_bytes = 0
        self._drop_sm_views()

    def _drop_sm_views(self) -> None:
        """Release the append-in-place dense views and their accounting."""
        global _GLOBAL_SM_BYTES
        for e in getattr(self, "_sm_views", {}).values():
            _GLOBAL_SM_BYTES = max(0, _GLOBAL_SM_BYTES - e["bytes"])
        self._sm_views = {}
        self._sm_bytes = 0

    def _drop_unband_cache(self) -> None:
        for _k, (_seg, _dev, _dt, t) in self._unband_cache.items():
            _unband_global_subtract(t.numel() * t.element_size())
        self._unband_cache = OrderedDict()
        self._unband_bytes = 0

    def _drop_dense_fallbacks(self) -> None:
        """Release every dense fallback copy. Called from EVERY path that
        drops or rebuilds segments -- a retained dense copy is a full
        uncompressed KV history, so leaking one is a memory bug, and
        returning a stale one is a correctness bug."""
        for _k, (_seg, t) in self._dense_fallback.items():
            _dense_fallback_global_subtract(t.numel() * t.element_size())
        self._dense_fallback = OrderedDict()
        self._dense_fallback_bytes = 0

    def reset(self) -> None:
        # Deliberately NOT calling super().reset(): DynamicLayer.reset()
        # touches self.keys/self.values, which this class no longer uses.
        batch = len(self._banded_k) if self._banded_k else 0
        self._banded_k = [[] for _ in range(batch)]
        self._banded_v = [[] for _ in range(batch)]
        self._seg_len = [[] for _ in range(batch)]
        self._banded_len = [0] * batch
        if self._dtype is not None:
            self._resid_k = [self._empty_resid_like(bi) for bi in range(batch)]
            self._resid_v = [self._empty_resid_like(bi) for bi in range(batch)]
        self._pad_lens = None
        # MUST clear this, not just self._banded_k/_v: _unband_cache holds
        # this layer's PREVIOUS conversation's segment objects, keyed by
        # id(seg). CPython can and does reuse a freed list's memory
        # address for a NEW segment built after this reset -- leaving a
        # stale entry would risk _unband_cached returning the OLD
        # conversation's (freed, stale) reconstructed tensor for what
        # looks like the same id(), a real cross-conversation data leak.
        self._fused = None            # uint8 id buffers of the fused decode path
        self._drop_joined_cache()
        self._drop_unband_cache()
        self._drop_dense_fallbacks()

    def _empty_resid_like(self, bi: int) -> torch.Tensor:
        """Rebuild slot bi's empty residual placeholder using this
        layer's cached heads/head_dim/dtype/device -- used by reset(),
        which must not assume any live (non-empty) tensor is still
        around to copy shape metadata from."""
        old = self._resid_k[bi] if bi < len(self._resid_k) else None
        heads = old.shape[1] if old is not None else (self._heads or 1)
        head_dim = old.shape[3] if old is not None else 1
        return torch.zeros(1, heads, 0, head_dim, dtype=self._dtype, device=self._device)

    # -- measurement -------------------------------------------------------
    def memory_bytes(self) -> dict:
        """Resident bytes: banded payload + dense residual.

        The dense baseline uses the cache tensors' OWN element_size(), not a
        hardcoded 2. This model runs fp32 K/V (measured: element_size()==4),
        so a hardcoded fp16 baseline reported ratios 2x too high -- it made
        an all-dense cache look like "2.000 of fp16" and a real 23%-of-fp32
        result read as "50%", which is what made this codec look no better
        than the int8 layer when it was in fact better.
        """
        banded = 0
        for slot in self._banded_k:
            for seg in slot:
                banded += sum(bk.nbytes() for bk in seg)
        for slot in self._banded_v:
            for seg in slot:
                banded += sum(bk.nbytes() for bk in seg)
        residual = 0
        elems = 0
        esz = self._resid_k[0].element_size() if self._resid_k else 4
        for rk, rv in zip(self._resid_k, self._resid_v):
            residual += (rk.numel() + rv.numel()) * esz
            elems += rk.numel() + rv.numel()
        total_banded = sum(self._banded_len) if self._banded_len else 0
        if self._heads is not None and total_banded:
            h = self._heads
            first_seg = next(slot[0] for slot in self._banded_k if slot)
            d = first_seg[0].orig_shape[1]
            elems += 2 * h * total_banded * d
        return {
            "dense_fallback_bytes": self._dense_fallback_bytes,
            "banded_bytes": banded,
            "residual_bytes": residual,
            "total_bytes": banded + residual,
            "dense_bytes": elems * esz,
            "elem_size": esz,
            "elems": elems,
        }


class _StatelessLayer:
    """Placeholder cache entry for a block type that holds no state
    (Nemotron-H's moe/mlp blocks). It exists only so Cache.layers has
    exactly one entry per model layer; every stateful code path skips it
    via its own isinstance checks (BandedKVLayer /
    LinearAttentionCacheLayerMixin)."""

    supports_early_init = False
    is_compileable = True
    is_sliding = False
    is_croppable = False

    def crop(self, tokens_to_remove: int) -> None:
        """No state to crop (BandedKVCache.crop's slot=None branch calls
        this on every non-BandedKVLayer layer)."""


class NemotronHMambaLayer(LinearAttentionCacheLayerMixin):
    """Per-layer Mamba-2 state (conv window + SSM) for one Nemotron-H
    mamba block.

    The model never calls update_conv_state/update_recurrent_state on
    this layer: run/nemotron_h._mamba2_forward reads and writes the
    tensors behind NemotronHBandedCache's conv_states/ssm_states views
    (in-place .copy_), and those views read this layer's
    conv_states[0]/recurrent_states[0] on every access. That is what lets
    the engine's generic linear-attention helpers
    (_grow_linear_attention_batch / _drop_linear_attention_batch /
    _snapshot_recurrent_state / _restore_recurrent_state_slot) manage the
    batch axis exactly as they do for Qwen3.5's GatedDeltaNet layers:
    they edit THIS dict, and the model sees the edit through the view.

    The state is allocated lazily -- the model's forward never calls
    lazy_initialization, so the owning cache's _qv_ensure_mamba_batch
    creates the tensors at the first forward's batch size (zero state ==
    start of sequence, the model's own convention).
    """

    def __init__(self, conv_dim: int, kernel_size: int, ssm_shape: tuple,
                 ssm_dtype: torch.dtype, number_of_states: int = 1, **kwargs):
        super().__init__(number_of_states=number_of_states)
        self._qv_conv_dim = conv_dim
        self._qv_kernel_size = kernel_size
        self._qv_ssm_shape = ssm_shape
        self._qv_ssm_dtype = ssm_dtype

    def lazy_initialization(self, conv_states: torch.Tensor | None = None,
                            recurrent_states: torch.Tensor | None = None,
                            state_idx: int = 0) -> None:
        """Satisfies the mixin's ABC. The model does not call this (see
        class docstring); it is provided so a caller can seed the state
        explicitly, from the given tensors' shapes."""
        if conv_states is not None:
            self.conv_states[state_idx] = torch.zeros_like(conv_states)
            self.is_conv_states_initialized[state_idx] = True
            self.conv_kernel_size[state_idx] = conv_states.shape[-1]
            if self.device is None:
                self.dtype, self.device = conv_states.dtype, conv_states.device
        if recurrent_states is not None:
            self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
            self.is_recurrent_states_initialized[state_idx] = True
            if self.device is None:
                self.dtype, self.device = recurrent_states.dtype, recurrent_states.device

    def update_conv_state(self, conv_states: torch.Tensor, state_idx: int = 0,
                          **kwargs) -> torch.Tensor:
        """No-op passthrough: the model writes the conv window in place
        through the cache-level view (see class docstring), never through
        this method."""
        return conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor,
                               state_idx: int = 0, **kwargs) -> torch.Tensor:
        """No-op passthrough (see update_conv_state)."""
        return recurrent_states

    def crop(self, tokens_to_remove: int) -> None:
        """No-op: a fixed-size recurrent state has no per-token rollback
        (the engine rolls it back with _snapshot_recurrent_state /
        _restore_recurrent_state_slot instead -- the same contract
        Qwen3.5's LinearAttentionLayer has)."""

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """No-op (beam search is not a supported path for this cache)."""


def _qv_dtype(v) -> torch.dtype:
    """A config dtype field as a torch dtype ("float32"/"bfloat16"/...)."""
    if isinstance(v, torch.dtype):
        return v
    return getattr(torch, str(v), torch.float32)


class BandedKVCache(Cache):
    """`DynamicCache` drop-in storing full-attention K/V as v7-KV bands.

    Layer dispatch mirrors DynamicCache.__init__ exactly, except that
    `full_attention` positions get a BandedKVLayer. Every other layer type
    (notably `linear_attention`) keeps its official class, so hybrid models
    work and the recurrent state is never touched.
    """

    def __init__(
        self,
        config,
        tau: float = KV_DEFAULT_TAU,
        residual_length: int = 128,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        decoder_config = config.get_text_config(decoder=True)
        layer_types, layer_kwargs = get_layer_types_and_kwargs(decoder_config)

        layers = []
        sw = getattr(decoder_config, "sliding_window", None)
        if getattr(decoder_config, "use_sliding_window", None) is False:
            sw = None                     # HF semantics: the window is switched off for this config
        for layer_type in layer_types:
            if layer_type == "full_attention":
                layers.append(BandedKVLayer(tau=tau,
                                            residual_length=residual_length,
                                            **layer_kwargs))
            elif sw and (layer_type in ("sliding_attention", "chunked_attention")
                         or getattr(DYNAMIC_LAYER_TYPE_MAPPING.get(layer_type), "is_sliding", False)):
                # Sliding-window layer: banded like a full-attention layer (so per-slot scheduling, crop and prefix
                # reuse all work), attention reads only the window (BandedKVLayer.window). Qwen has none.
                layers.append(BandedKVLayer(tau=tau, residual_length=residual_length, window=int(sw)))
            else:
                layers.append(DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs))

        super().__init__(
            layers=layers,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )
        self.tau = tau
        self.residual_length = residual_length

    def metrics_report(self) -> dict:
        reports = [layer.metrics_report() for layer in self.layers
                   if isinstance(layer, BandedKVLayer) and layer.is_initialized]
        keys = ("flushes", "flush_tokens", "outliers", "fallback_segments",
                "fallback_required", "forget_tokens", "forget_segments",
                "segments", "banded_tokens")
        return {k: sum(int(r.get(k, 0)) for r in reports) for k in keys}

    def memory_report(self) -> dict:
        tot = {"banded_bytes": 0, "residual_bytes": 0, "total_bytes": 0,
               "dense_bytes": 0, "elems": 0, "layers": 0}
        for layer in self.layers:
            if isinstance(layer, BandedKVLayer) and layer.is_initialized:
                m = layer.memory_bytes()
                for k in ("banded_bytes", "residual_bytes", "total_bytes",
                          "dense_bytes", "elems"):
                    tot[k] += m[k]
                tot["layers"] += 1
        if tot["elems"]:
            tot["bytes_per_elem"] = tot["total_bytes"] / tot["elems"]
        if tot["dense_bytes"]:
            # ratio against the cache's REAL dense cost, not an assumed fp16
            tot["ratio_vs_dense"] = tot["total_bytes"] / tot["dense_bytes"]
        return tot

    def add_slot(self, key_like: torch.Tensor, value_like: torch.Tensor) -> int:
        """Admit a new slot on every BandedKVLayer in this cache (each
        full_attention layer keeps its own per-slot state, so all of them
        must be told about a new slot together, or they desync on the
        very next update()).

        LINEAR-ATTENTION LAYERS ARE NOT HANDLED HERE. Qwen3.5's hybrid
        layers (LinearAttentionLayer / the recurrent conv+SSM state) have
        no per-slot API of their own in this transformers version -- they
        hold ONE fixed-size recurrent state per batch position, tied to
        the whole Cache's batch axis, and admitting a slot into them
        requires either a real upstream API this version does not expose,
        or growing their state tensors directly (not attempted here). A
        caller of THIS method on a hybrid model must also handle those
        layers' own slot admission, or the batch axis across layer types
        will disagree. Safe to call as-is only when every layer in this
        cache is a BandedKVLayer (no linear-attention layers present).
        """
        idx = None
        for layer in self.layers:
            if isinstance(layer, BandedKVLayer):
                idx = layer.add_slot(key_like, value_like)
        if idx is None:
            raise RuntimeError(
                "BandedKVCache.add_slot(): no BandedKVLayer found (empty "
                "or non-hybrid cache) -- nothing to add a slot to.")
        return idx

    def drop_slot(self, bi: int) -> None:
        """Remove slot `bi` from every BandedKVLayer. Same linear-
        attention caveat as add_slot() -- see its docstring."""
        for layer in self.layers:
            if isinstance(layer, BandedKVLayer):
                layer.drop_slot(bi)

    def plan_forget(self, token_indices=None, segment_indices=None,
                    slot: int | None = None) -> dict:
        """Cache-wide dry run: union of every banded layer's plan."""
        plans = [l.plan_forget(token_indices, segment_indices, slot)
                 for l in self.layers if isinstance(l, BandedKVLayer)]
        if not plans:
            return {"slots": [], "token_indices": [], "segment_indices": [],
                    "would_remove_tokens": 0, "would_remove_segments": 0,
                    "affected": []}
        return {
            "slots": plans[0]["slots"],
            "token_indices": plans[0]["token_indices"],
            "segment_indices": plans[0]["segment_indices"],
            "would_remove_tokens": sum(p["would_remove_tokens"] for p in plans),
            "would_remove_segments": sum(p["would_remove_segments"] for p in plans),
            "affected": [a for p in plans for a in p["affected"]],
        }

    def forget_tokens(self, token_indices, slot: int | None = None,
                      audit: list | None = None) -> int:
        return sum(layer.forget_tokens(token_indices, slot=slot, audit=audit)
                   for layer in self.layers if isinstance(layer, BandedKVLayer))

    def forget_segments(self, segment_indices, slot: int | None = None,
                        audit: list | None = None) -> int:
        return sum(layer.forget_segments(segment_indices, slot=slot, audit=audit)
                   for layer in self.layers if isinstance(layer, BandedKVLayer))

    def crop(self, max_length: int, slot: int | None = None) -> None:
        """Override the base Cache.crop(max_length), which calls every
        layer's .crop(max_length) with NO slot argument -- exactly the
        old whole-batch-uniform crop behaviour, still correct when
        slot=None. Passing slot=bi propagates it to every BandedKVLayer's
        own per-slot crop (see that class's crop() docstring); every
        other (non-BandedKVLayer) layer type is SKIPPED in this mode --
        only the banded full_attention layers have per-slot state to
        target, and a linear-attention layer's recurrent state has no
        per-slot crop of its own (see spec_decode.py's _snapshot_
        recurrent_state/_restore_recurrent_state_slot for how THAT gets
        rolled back per-slot instead, which every per-slot caller in this
        codebase already pairs with a slot-crop call).

        REAL BUG FIXED HERE: this used to call `layer.crop(max_length)`
        (no slot) on every non-BandedKVLayer regardless of whether `slot`
        was given -- on Qwen3.5's hybrid stack those are
        LinearAttentionLayer instances, which do NOT support crop() at
        all without `activate_past_recording()` armed first (stock
        transformers raises `RuntimeError: crop was called, but the
        current layer does not track past states` -- confirmed by direct
        reproduction, the very first per-slot crop call after this
        engine went into real use). The per-slot callers in spec_decode.py
        (ContinuousBatchEngine.step, speculative_generate_batched) both
        already call _restore_recurrent_state_slot separately for exactly
        this purpose, so calling layer.crop() on these layers in slot
        mode was never correct to begin with -- it either crashed (this
        model family) or silently no-opped on a family where crop
        happened to be armed, masking that the recurrent state was never
        actually rolled back by it. slot=None (the original whole-batch
        crop a non-continuous-batching caller uses) is unaffected -- that
        path still calls every layer's plain crop(max_length), unchanged.
        """
        for layer in self.layers:
            if isinstance(layer, BandedKVLayer):
                layer.crop(max_length, slot=slot)
            elif slot is None:
                layer.crop(max_length)


class NemotronHBandedCache(BandedKVCache):
    """BandedKVCache for a Nemotron-H hybrid stack (Mamba-2 / attention /
    LatentMoE, one block type per layer, config.layers_block_type).

    BandedKVCache itself derives its layer list from
    get_layer_types_and_kwargs, which cannot see layers_block_type (it is
    not a transformers layer_types field) and would build one
    BandedKVLayer per layer -- the mamba blocks would then find no
    conv_states/ssm_states on the cache and crash on the first prefill
    (AttributeError: 'BandedKVCache' object has no attribute
    'conv_states'). This subclass builds the real layout instead:
      - "attention" -> BandedKVLayer (banded KV, unchanged)
      - "mamba"     -> NemotronHMambaLayer (per-slot conv + SSM state)
      - "moe"/"mlp" -> _StatelessLayer (no state)
    and exposes the model-level surface the shipped modeling code reads:
    conv_states / ssm_states dicts keyed by layer index,
    has_previous_state, trace. Everything else (add_slot/drop_slot/crop/
    metrics) is inherited from BandedKVCache and already skips the
    non-BandedKVLayer entries.
    """

    def __init__(self, config, tau: float = KV_DEFAULT_TAU,
                 residual_length: int = 128, offloading: bool = False,
                 offload_only_non_sliding: bool = False):
        decoder_config = config.get_text_config(decoder=True)
        block_types = [str(t) for t in getattr(decoder_config, "layers_block_type", [])]
        if not block_types:
            raise ValueError(
                "NemotronHBandedCache requires a config with "
                "layers_block_type (the Nemotron-H puzzle config)")
        # Mamba state geometry, uniform across mamba blocks -- the same
        # derivation as the model's own NemotronHMamba2Mixer.__init__ /
        # NemotronHHybridDynamicCache.__init__:
        #   conv_states[i] = [batch, inter + 2*groups*ssm, conv_kernel]
        #   ssm_states[i]  = [batch, mamba_heads, mamba_head_dim, ssm]
        mamba_heads = int(getattr(decoder_config, "mamba_num_heads", 0))
        mamba_head_dim = int(getattr(decoder_config, "mamba_head_dim", 0))
        ssm_state_size = int(getattr(decoder_config, "ssm_state_size", 0))
        n_groups = int(getattr(decoder_config, "n_groups", 1))
        conv_kernel = int(getattr(decoder_config, "conv_kernel", 4))
        if not (mamba_heads and mamba_head_dim and ssm_state_size):
            raise ValueError(
                "NemotronHBandedCache: the config is missing the mamba "
                "state dimensions (mamba_num_heads/mamba_head_dim/"
                "ssm_state_size)")
        inter = mamba_heads * mamba_head_dim
        conv_dim = inter + 2 * n_groups * ssm_state_size
        ssm_dtype = _qv_dtype(getattr(decoder_config, "mamba_ssm_cache_dtype", "float32"))

        layers = []
        for bt in block_types:
            if bt == "attention":
                layers.append(BandedKVLayer(tau=tau, residual_length=residual_length))
            elif bt == "mamba":
                layers.append(NemotronHMambaLayer(
                    conv_dim, conv_kernel,
                    (mamba_heads, mamba_head_dim, ssm_state_size), ssm_dtype))
            else:                                   # moe / mlp: no state
                layers.append(_StatelessLayer())
        # BandedKVCache.__init__ would rebuild the layer list from
        # get_layer_types_and_kwargs; go straight to Cache with ours.
        Cache.__init__(self, layers=layers, offloading=offloading,
                       offload_only_non_sliding=offload_only_non_sliding)
        self.tau = tau
        self.residual_length = residual_length
        # Model-level surface (see class docstring):
        self.has_previous_state = False
        self.trace = None

    @property
    def conv_states(self) -> dict:
        """layer index -> that mamba block's conv window, read through the
        layer's own dict so the engine's batch-axis edits (grow/drop) are
        visible here without any bookkeeping on this side. The model
        writes in place (.copy_) through this view, which is why a
        read-only view is sufficient."""
        return {i: lyr.conv_states[0] for i, lyr in enumerate(self.layers)
                if isinstance(lyr, NemotronHMambaLayer)}

    @property
    def ssm_states(self) -> dict:
        """layer index -> that mamba block's SSM state (see conv_states)."""
        return {i: lyr.recurrent_states[0] for i, lyr in enumerate(self.layers)
                if isinstance(lyr, NemotronHMambaLayer)}

    def _qv_ensure_mamba_batch(self, b: int, dtype, device) -> None:
        """Allocate (or re-allocate) every mamba block's conv/SSM state at
        batch `b`, zeroed. Called from the patched mamba forward when the
        state is absent (the first forward) or its batch axis no longer
        matches the incoming batch (a new engine after the last slot has
        left, or a direct caller that changed the batch size without the
        engine's grow/drop). Zero state == start of sequence, so
        re-allocation is a clean reset in every case it is reached."""
        for lyr in self.layers:
            if isinstance(lyr, NemotronHMambaLayer):
                cs, rs = lyr.conv_states[0], lyr.recurrent_states[0]
                if cs is None or rs is None or cs.shape[0] != b or rs.shape[0] != b:
                    break
        else:
            return
        for lyr in self.layers:
            if not isinstance(lyr, NemotronHMambaLayer):
                continue
            lyr.conv_states[0] = torch.zeros(b, lyr._qv_conv_dim, lyr._qv_kernel_size,
                                             dtype=dtype, device=device)
            lyr.recurrent_states[0] = torch.zeros(b, *lyr._qv_ssm_shape,
                                                  dtype=lyr._qv_ssm_dtype, device=device)
            lyr.is_conv_states_initialized[0] = True
            lyr.is_recurrent_states_initialized[0] = True
            lyr.conv_kernel_size[0] = lyr._qv_kernel_size
            lyr.has_previous_state[0] = True
            if lyr.device is None:
                lyr.device, lyr.dtype = device, dtype
