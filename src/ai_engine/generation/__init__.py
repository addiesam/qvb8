"""The generation core: turn a packed model into a stream of tokens.

    gen_core.py   DEFAULTS, GenParams, load_model, build_drafter,
                  encode_prompt, run_generate
    loader.py     load_qvb8_llm: the one correct order for the three steps

This is the layer every text interface shares. chat.py, llm_server.py and
each pipeline differ only in how a request arrives and how its output is
delivered; everything between is here, so it cannot drift between them. It
drifted once -- a max_new default bound to ctx_size caused a 70 GB VRAM
blowup in one interface while the KV-cache flags existed in only one of the
two.

WHY IT IS HERE AND NOT IN qv_band
---------------------------------
qv_band defines the compression: the codecs, the pack readers, the weight
formats, the drafters. This package defines none of that -- it assembles
those parts into something servable, and adds the decisions that belong to
an application rather than to a format (which drafter, how much shadow
budget, what the memory ceiling does).

Keeping it here is also what keeps the dependency graph acyclic. qv_band
never imports this module -- its own calibration/eval/inference do not use
it -- so when this lived inside qv_band it forced qv_band to depend on
ai_engine for nothing, while ai_engine's servers imported straight back.
"""
from ai_engine.generation.gen_core import (          # noqa: F401
    DEFAULTS, DRAFTER_CHOICES, KV_CACHE_CHOICES, KV_CACHE_PRESETS,
    NO_ABORT, DrafterBuildError, GenParams, ModelBundle,
    add_generation_args, apply_kv_shadow, build_drafter, encode_prompt,
    kv_shadow_covers_tokens, parse_kv_shadow, load_model,
    params_from_args, resolve_device, run_generate,
)
from ai_engine.generation.loader import (            # noqa: F401
    LoadedLLM, load_qvb8_llm,
)

__all__ = [
    # gen_core
    "DEFAULTS", "DRAFTER_CHOICES", "KV_CACHE_CHOICES", "KV_CACHE_PRESETS",
    "NO_ABORT", "DrafterBuildError", "GenParams",
    "ModelBundle", "add_generation_args", "apply_kv_shadow", "build_drafter",
    "encode_prompt", "kv_shadow_covers_tokens", "parse_kv_shadow",
    "load_model", "params_from_args", "resolve_device", "run_generate",
    # loader
    "LoadedLLM", "load_qvb8_llm",
]
