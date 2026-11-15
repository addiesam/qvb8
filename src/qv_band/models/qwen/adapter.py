"""Qwen adapter (Qwen3.5 / 3.6 / 3.8, dense and MoE). Every mechanism it uses is a framework kind: full attention,
Gated-DeltaNet recurrent state, MoE experts, qvb8 KV, DFlash drafter. Nothing to install."""
from __future__ import annotations

from qv_band.engine.adapter import ModelAdapter, register


class QwenAdapter(ModelAdapter):
    name = "qwen"

    def matches(self, config) -> bool:
        mt = getattr(config, "model_type", "") or ""
        return mt.startswith("qwen")

    def default_draft_dir(self, pack_dir):
        from qv_band.decode.dflash_draft import find_dflash_draft
        return find_dflash_draft(pack_dir)


ADAPTER = register(QwenAdapter())
