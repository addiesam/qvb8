"""The one way to load a QV-band packed LLM.

Four call sites were each re-assembling the same three steps:

    apply_kv_shadow(model_dir, params)   # must happen BEFORE load_model
    bundle = load_model(model_dir, params)
    drafter = build_drafter(params.drafter, bundle, ...)

llm_server.py and chat.py got all three. script_to_video's llm_loader, video_gen's
llm_adapter and agent_tools/loop.py skipped apply_kv_shadow entirely --
so the same model loaded differently depending on who asked, which is exactly
the class of bug this module removes.

WHY apply_kv_shadow IS NOT OPTIONAL
-----------------------------------
Importing qv_band freezes the shadow budgets. gen_core's own comment:

    # Before load_model(): importing qv_band freezes the shadow budgets, so
    # --kv-shadow has to be resolved and exported first.

Call it too late and the budget is computed against the wrong shape, or not
exported at all, and the KV cache sizes itself wrong. A loader that lets you
skip it is a loader that will eventually be called without it.

Every decision -- shadow, device, memory ceiling, model load, drafter kind and context -- is made here, identically,
for everyone. A drafter that cannot be built stops the load; nothing is replaced by n-gram.

This module used to be engine.loader, then ai_engine.generation.loader, and is
now here beside the gen_core it sequences. What it sequences -- reading a
QV-band pack folder -- is defined by qv_band, but the ORDER is an
application concern, and the shadow-before-import constraint above only
makes sense next to the code that does the importing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ai_engine.generation.gen_core import (
    ModelBundle, apply_kv_shadow, build_drafter, load_model,
)
from ai_engine.runtime.memlimit import rss_gb


@dataclass
class LoadedLLM:
    """A loaded QVB8 model plus the drafter it should generate with."""

    bundle: ModelBundle
    drafter: object | None
    shadow: dict
    params: object

    @property
    def model(self):
        return self.bundle.model

    @property
    def tok(self):
        return self.bundle.tok

    @property
    def device(self) -> str:
        """'cuda:0' / 'xpu:0' / 'cpu' -- the plain name, no rank."""
        if not self.bundle.use_gpu:
            return "cpu"
        return (self.bundle.gpu_dev.split(":", 1)[0]
                if ":" in self.bundle.gpu_dev else "cuda")

    def describe(self) -> str:
        """One block of banner text: device, KV mode, load time, RSS."""
        lines = [f"device : {self.device}",
                 f"kv     : {self.params.describe_kv()}",
                 f"ready  : {self.bundle.load_seconds:.1f}s, "
                 f"RSS {rss_gb():.1f} GB"]
        return "\n".join(lines)


def load_qvb8_llm(model_dir: str, params, *, verbose: bool = True,
                  require_model: bool = True) -> LoadedLLM:
    """Load a QVB8 packed LLM the one correct way.

    model_dir must already exist unless require_model=False (the pipeline
    loaders pass their configured path and want the clear error; agent/loop.py
    resolves a path from an env var and would rather get None back).

    A drafter that cannot be built stops the load with DrafterBuildError.
    """
    if require_model and not os.path.isdir(model_dir):
        raise FileNotFoundError(f"LLM model folder not found: {model_dir}")

    # MUST precede load_model: importing qv_band freezes the shadow budgets.
    shadow = apply_kv_shadow(model_dir, params)
    bundle = load_model(model_dir, params, verbose=verbose)
    drafter = (build_drafter(params.drafter, bundle, max_ctx=params.draft_ctx,
                             params=params)
               if params.draft_k > 0 else None)
    return LoadedLLM(bundle=bundle, drafter=drafter, shadow=shadow, params=params)


__all__ = ["LoadedLLM", "load_qvb8_llm"]
