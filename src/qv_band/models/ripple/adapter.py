"""Ripple-65B adapter: which framework mechanisms the model uses, and its defaults.

  attention   layers/sliding_attention (64 sliding-window RoPE layers, 16 global NoPE layers)
  cache       qvb8 KV: every layer is a BandedKVLayer (sliding ones read only their window);
              bf16 KV: kv/sliding_cache.MarginSlidingLayer on the sliding layers; exact rollback hook
  pointwise   layers/fused_pointwise on RicursiveMLP / RicursiveRMSNorm
  drafter     DFlash2 with NoPE layers (decode/dflash_nope_draft, chosen from the drafter's own config)
  chat        models/ripple/chat_format.py
"""
from __future__ import annotations

import os

from qv_band.engine.adapter import ModelAdapter, ServeDefaults, register

DRAFT_DIR = "/mnt/backup/models/ricursiveAI/Ripple-65B-Preview-DFlash2"
DRAFT_PACK = "/mnt/backup/qvb8/Ripple-65B-Preview-DFlash2_qvb8_5e-3"


class RippleAdapter(ModelAdapter):
    name = "ripple"
    skeleton_attn_impl = "sdpa"        # the shipped RicursiveForCausalLM refuses flash-attn at construction

    def matches(self, config) -> bool:
        return getattr(config, "model_type", None) == "ricursive"

    def serve_defaults(self) -> ServeDefaults:
        return ServeDefaults(kv_cache="qvb8", prefill_chunk=4096, drafter="dflash", draft_k=15, attn_impl="sdpa")

    def install(self, model, *, model_dir, config, tok) -> None:
        import functools
        from qv_band.kv.sliding_cache import LAYER_TYPE, MarginSlidingLayer, rollback
        from qv_band.layers.fused_pointwise import install_fused
        from qv_band.layers.sliding_attention import install_attention
        from qv_band.models.ripple.chat_format import RippleChatFormat
        cfg = model.config
        try:
            import flash_attn  # noqa: F401
        except ImportError as e:
            raise RuntimeError("Ripple needs flash-attn (bottom-right causal + sliding window)") from e
        orig_types = list(cfg.layer_types)
        window = int(cfg.sliding_window)
        n = install_attention(model, orig_types, window)
        # The skeleton was built with sdpa (skeleton_attn_impl); the attention forward is the framework's now, so
        # switch the registry entry it uses.
        cfg._attn_implementation = "flash_attention_2"
        MarginSlidingLayer.default_window = window
        cfg.layer_types = [LAYER_TYPE if t == "sliding_attention" else t for t in orig_types]
        cfg._qv_orig_layer_types = orig_types
        model._qv_rollback = functools.partial(rollback)
        model._qv_chat_format = RippleChatFormat()     # reasoning only when output opens <think>; JSON tool calls
        n_fused = install_fused(model, "RicursiveMLP", "RicursiveRMSNorm")     # QV_RIPPLE_FUSED=0: off
        n_slide = sum(t == "sliding_attention" for t in orig_types)
        print(f"ripple runtime: {n} attention layers -> flash-attn registry ({n_slide} sliding W={window}, "
              f"{n - n_slide} global NoPE); KV: qvb8 -> all layers BandedKVLayer (sliding ones window={window}), "
              f"bf16 -> MarginSlidingLayer (margin {MarginSlidingLayer.MARGIN}); exact rollback hook; "
              f"{n_fused} MLP/RMSNorm modules fused", flush=True)

    def default_draft_dir(self, pack_dir):
        return os.environ.get("RIPPLE_DRAFT", DRAFT_DIR)

    def default_draft_pack(self, pack_dir):
        p = os.environ.get("RIPPLE_DRAFT_PACK", DRAFT_PACK)
        return p if os.path.isdir(p) else None


ADAPTER = register(RippleAdapter())
