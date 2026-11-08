"""Install Ripple's runtime pieces on a loaded model (called once, right after loading).

  1. attention  -> engine attention registry (flash-attn / qvb8 fused), see attention.py
  2. cache      -> sliding layers become "ripple_sliding" (RippleSlidingLayer, exact rollback), global layers stay
                   "full_attention" (DynamicLayer for bf16 KV, BandedKVLayer for qvb8 KV) -- see cache.py
  3. rollback   -> model._qv_rollback for the engine's speculative loop (no full-prefix rebuild on rejection)
"""
from __future__ import annotations

import functools

from qv_band.models.ripple.attention import install_attention
from qv_band.models.ripple.cache import LAYER_TYPE, RippleSlidingLayer, rollback


def install(model) -> None:
    if getattr(model, "_qv_ripple_installed", False):
        return
    cfg = model.config
    try:
        import flash_attn  # noqa: F401
    except ImportError as e:
        raise RuntimeError("Ripple runtime needs flash-attn (bottom-right causal + sliding window)") from e
    orig_types = list(cfg.layer_types)
    window = int(cfg.sliding_window)
    n = install_attention(model, orig_types, window)
    # The skeleton was built with sdpa because the shipped class refuses flash-attn at init; the attention is
    # ours now, so switch the registry entry it uses.
    cfg._attn_implementation = "flash_attention_2"
    RippleSlidingLayer.default_window = window
    cfg.layer_types = [LAYER_TYPE if t == "sliding_attention" else t for t in orig_types]
    cfg._qv_orig_layer_types = orig_types
    model._qv_rollback = functools.partial(rollback)
    from qv_band.models.ripple.chat_format import RippleChatFormat
    model._qv_chat_format = RippleChatFormat()     # reasoning only when output opens <think>; Hermes JSON tool calls
    from qv_band.models.ripple.fused import install_fused
    n_fused = install_fused(model)                 # Triton MLP activation + RMSNorm (QV_RIPPLE_FUSED=0: off)
    model._qv_ripple_installed = True
    n_slide = sum(t == "sliding_attention" for t in orig_types)
    print(f"ripple runtime: {n} attention layers -> flash-attn registry ({n_slide} sliding W={window}, "
          f"{n - n_slide} global NoPE); KV: qvb8 -> all layers BandedKVLayer (sliding ones window={window}), "
          f"bf16 -> RippleSlidingLayer (margin {RippleSlidingLayer.MARGIN}); exact rollback hook; "
          f"{n_fused} MLP/RMSNorm modules fused",
          flush=True)
