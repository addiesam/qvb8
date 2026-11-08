"""Tensor-key <-> module-path resolution shared by every run_inference*
variant (CUDA, CPU, tensor-parallel). Pure name-mangling, no device/backend
logic, so one copy serves all three loaders.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch.nn as nn


def tensor_key_to_module(key: str) -> str:
    """'model.language_model.layers.0.mlp.gate_proj.weight' -> 'model.layers.0.mlp.gate_proj'.

    The text-only causal-LM (from text_config) strips the multimodal
    'language_model.' segment but keeps the leading 'model.' prefix.

    A routed-expert synthetic key ("<base>#expert<e>") has its base tensor
    NAME translated the same way (there is no ".weight" suffix on a bare
    nn.Parameter like experts.gate_up_proj, so that strip is a no-op for
    it) -- the caller is responsible for stripping "#expert<e>" first via
    _parse_expert_key if it wants the base-tensor's module path; this
    function alone does not recognise the synthetic suffix.
    """
    if key.endswith(".weight"):
        key = key[: -len(".weight")]
    for pfx, repl in (("language_model.backbone.", "model."),    # Nemotron-H Omni: backbone == the text model
                      ("model.language_model.", "model."), ("language_model.", "")):
        if key.startswith(pfx):
            return repl + key[len(pfx):]
    return key


def _resolve(model: nn.Module, dotted: str):
    obj = model
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def make_resolve_key(meta_keys: set) -> Callable[[str], Optional[str]]:
    """A checkpoint-key -> meta-parameter-name resolver closed over one
    model's meta_keys set.

    Three things a raw checkpoint key may need before it matches a name in
    meta_keys, tried in order:
      1. itself, unchanged
      2. with '.language_model.' collapsed to '.' (the multimodal-wrapper
         segment text-only skeletons never have)
      3. with a leading 'model.' ADDED (some checkpoints' tensor keys omit
         the leading 'model.' the skeleton's own parameter/buffer names
         use -- confirmed on real checkpoints: Qwen3-Embedding-0.6B/4B ship
         keys like 'embed_tokens.weight' with no prefix, while
         Qwen3-Reranker-* -- same architecture, same skeleton -- keeps
         'model.' prefixed keys. Without this, EVERY tensor in such a
         checkpoint fails to resolve here, silently loading zero dense
         weights (the dense-stream pass has no "tensor not found" error,
         only a per-tensor skip) -- found via a real Qwen3-Embedding-4B
         dense load that completed with "loaded 0 dense tensors" and
         crashed deep inside attention on an untouched meta tensor instead
         of failing loudly up front.)
    """
    def _resolve_key(ckpt_key: str) -> Optional[str]:
        if ckpt_key in meta_keys:
            return ckpt_key
        stripped = ckpt_key.replace(".language_model.", ".")
        if stripped in meta_keys:
            return stripped
        if ckpt_key.startswith("language_model."):
            nemo = ckpt_key.replace("language_model.backbone.", "model.", 1) \
                if ckpt_key.startswith("language_model.backbone.") else ckpt_key[len("language_model."):]
            if nemo in meta_keys:
                return nemo
        prefixed = "model." + ckpt_key
        return prefixed if prefixed in meta_keys else None

    return _resolve_key
