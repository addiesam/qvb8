"""Cache kind: sliding-window KV layer with a rollback margin (bf16 KV path). Used by: Ripple.

The cache object itself is the engine's (DynamicCache for bf16 KV, BandedKVCache for qvb8 KV, exactly as for
Qwen). This module contributes:
  * MarginSlidingLayer -- the cache layer for its 72 sliding-window layers. HF's DynamicSlidingWindowLayer keeps
    exactly sliding_window-1 keys, so a speculative verify of k tokens followed by a rejection cannot be undone
    (its crop() refuses once the window is full). This layer keeps MARGIN extra keys, so dropping up to MARGIN
    rejected tokens is exact. Registered with HF as layer type "margin_sliding" (cache_utils auto-registers
    subclasses that declare _layer_type).
  * rollback(cache, n_drop) -- exact O(1) rollback of the last n_drop tokens on every layer; the engine calls it
    through model._qv_rollback (spec_decode's hook for models without recurrent state).
The 8 global layers stay "full_attention": DynamicLayer (bf16) or BandedKVLayer (qvb8, certified codec).
"""
from __future__ import annotations

import torch
from transformers.cache_utils import DynamicSlidingWindowLayer

LAYER_TYPE = "margin_sliding"


class MarginSlidingLayer(DynamicSlidingWindowLayer):
    _layer_type = LAYER_TYPE
    MARGIN = 64                    # >= the largest verify block (16) with room for chunked rollbacks
    default_window = 1024          # set from the model config by the adapter

    def __init__(self, sliding_window: int | None = None, **kwargs):
        super().__init__(sliding_window=sliding_window or self.default_window)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]
        full_k = torch.cat([self.keys, key_states], dim=-2)
        full_v = torch.cat([self.values, value_states], dim=-2)
        keep = self.sliding_window - 1 + self.MARGIN
        self.keys = full_k[:, :, -keep:, :]
        self.values = full_v[:, :, -keep:, :]
        # Attention applies the window itself (flash-attn window_size, bottom-right causal), so returning a few
        # keys beyond the window is harmless.
        return full_k, full_v

    def can_drop(self, n: int) -> bool:
        if n <= 0:
            return True
        held = self.keys.shape[-2] if self.is_initialized else 0
        need_after = min(self.cumulative_length - n, self.sliding_window - 1)
        return n <= held and held - n >= need_after

    def drop(self, n: int) -> None:
        if n <= 0:
            return
        self.keys = self.keys[:, :, :-n, :]
        self.values = self.values[:, :, :-n, :]
        self.cumulative_length -= n

    # Prompt-anchor hook (spec_decode._save_anchor / _reuse_from_anchor): an agent client (Cline) resends
    # prompt + re-rendered reply + new input; exact-prefix reuse misses, so the engine rolls the cache back to the
    # previous prompt's end. This layer cannot crop that far (it holds window+margin keys), so it snapshots its
    # window at the anchor and restores it. ~2.2 MB per layer at W=1024 (64 layers ~ 0.28 GB), kept on the device.
    def anchor_snapshot(self):
        if not self.is_initialized:
            return None
        return (self.keys.clone(), self.values.clone(), self.cumulative_length)

    def anchor_restore(self, snap) -> None:
        if snap is None:
            self.keys = self.keys[:, :, :0] if self.is_initialized else self.keys
            self.values = self.values[:, :, :0] if self.is_initialized else self.values
            self.cumulative_length = 0
            return
        k, v, n = snap
        self.keys, self.values, self.cumulative_length = k.clone(), v.clone(), n

    def crop(self, max_length: int) -> None:
        """HF convention: negative = drop that many; non-negative = truncate to that absolute length."""
        n = -max_length if max_length < 0 else self.cumulative_length - max_length
        if not self.can_drop(n):
            raise RuntimeError(f"MarginSlidingLayer: cannot drop {n} tokens (margin {self.MARGIN})")
        self.drop(n)


def rollback(cache, n_drop: int) -> bool:
    """Drop the last n_drop tokens from every layer, exactly. False (cache untouched) if any layer cannot."""
    if n_drop <= 0:
        return True
    layers = getattr(cache, "layers", None)
    if not layers:
        return False
    for L in layers:
        if isinstance(L, MarginSlidingLayer) and not L.can_drop(n_drop):
            return False
    for L in layers:
        if isinstance(L, MarginSlidingLayer):
            L.drop(n_drop)
        else:                       # DynamicLayer / BandedKVLayer: both support HF's negative crop
            L.crop(-n_drop)
    return True
