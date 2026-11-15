"""Nemotron-H Omni adapter (Nemotron-3.5-Bolt): which framework mechanisms the model uses.

  stack     layers/mamba2_hybrid (Mamba-2 / attention / latent-MoE skeleton from the checkpoint's own modeling files,
            sparse MoE, CUDA-graph fast paths)
  cache     kv/banded_cache.HybridBandedCache (chosen from the config's layers_block_type)
  loop      decode/hybrid_generate
  drafter   LiLiCorr (chosen from the drafter's own config)
  vision    C-RADIO tower (layers/cradio_vision, ai_engine/generation/vision_cradio)

NOT RUN since it was written (2026-11-10): no Nemotron pack is on the the dev GPU machine. It carries, unchanged, what
run_inference.py did inline before.
"""
from __future__ import annotations

import json
import os

from qv_band.engine.adapter import ModelAdapter, register


class NemotronAdapter(ModelAdapter):
    name = "nemotron"
    dense_fp32_modules = ("NemotronHTopkRouter",)      # sigmoid router, run in fp32 by the model's own forward
    vision_adapter = "ai_engine.generation.vision_cradio:CRadioVisionAdapter"

    def matches(self, config) -> bool:
        llm = getattr(config, "llm_config", None)
        return llm is not None and getattr(llm, "model_type", "") == "nemotron_h_puzzle"

    def text_config(self, config):
        return config.llm_config

    def build_skeleton(self, model_dir, config, attn_impl):
        from qv_band.layers.mamba2_hybrid import build_text_skeleton
        return build_text_skeleton(model_dir, config, attn_impl)

    def install(self, model, *, model_dir, config, tok) -> None:
        from qv_band.layers.mamba2_hybrid import install_fast_paths
        model._qv_eos_ids = self._eos_ids(model_dir, config, tok)
        print(f">>> nemotron fast paths: {install_fast_paths(model)}", flush=True)

    def generate_fn(self):
        from qv_band.decode.hybrid_generate import hybrid_generate
        return hybrid_generate

    @staticmethod
    def _eos_ids(model_dir, config, tok) -> set:
        """Every id that ends a reply: tokenizer, wrapper and text config, generation_config.json."""
        ids = set()

        def add(v):
            for x in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(x, int):
                    ids.add(x)
        add(tok.eos_token_id)
        add(getattr(config, "eos_token_id", None))
        add(getattr(config.llm_config, "eos_token_id", None))
        try:
            add(json.load(open(os.path.join(model_dir, "generation_config.json"))).get("eos_token_id"))
        except OSError:
            pass
        return ids


ADAPTER = register(NemotronAdapter())
