"""ModelAdapter protocol + registry: the single seam between the engine (mechanism) and a model (policy).

Invariants the engine relies on (see tech_doc/engine_design.md):
  I1 certificate   |w - restore(w_hat)| <= tau on the executed path; restore dtype is part of the pack type
  I2 losslessness  speculative output == plain greedy, except at near-ties within bf16 kernel noise
  I3 cache         after commit(n) the cache equals a sequential feed of the committed prefix
  I4 untrusted     correctness never depends on the drafter (it only costs speed)
  I5 progress      every verify step commits >= 1 token

Discovery: in-tree adapters register on import of qv_band.models.<name>.adapter; out-of-tree packages
declare an entry point in group "qv_band.models" pointing at a ModelAdapter instance or class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# -- runtime protocols ---------------------------------------------------------------------------------
@runtime_checkable
class Target(Protocol):
    """The verified model. Owns its cache; `commit` must satisfy I3."""

    def prefill(self, ids) -> tuple[Any, Any]: ...            # -> (logits, taps)
    def verify(self, tokens) -> tuple[Any, Any]: ...          # one forward over k+1 tokens
    def commit(self, n: int) -> None: ...                     # keep n of the verified tokens


@runtime_checkable
class Drafter(Protocol):
    """Untrusted proposer (I4): any output, however wrong, must leave the result unchanged."""

    def observe_features(self, first_pos: int, feats) -> None: ...
    def propose(self, last_token: int, k: int, temperature: float = 0.0,
                top_k: int = 0, top_p: float = 1.0): ...
    def feedback(self, n_accepted: int, n_proposed: int) -> None: ...


# -- adapter -------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ServeDefaults:
    """Launch defaults a model wants; every one is overridable by an explicit flag."""
    kv_cache: str = "qvb8"
    prefill_chunk: int = 8192
    drafter: str = "dflash"
    draft_k: int = 8
    attn_impl: str | None = None          # None = engine default (flash-attn if available)
    extra: dict = field(default_factory=dict)


class ModelAdapter:
    """Facts about one model family: which framework mechanisms it uses and its defaults. No kernels, no decode
    loop, no cache code -- those live in the framework (layers/, kv/, decode/) and are named by kind.

    The loader, the decode entry points and the server call only these members; they never test for a model.
    Subclass, set `name`, override what differs, and `register()` an instance.
    """

    name: str = ""
    skeleton_attn_impl: str | None = None     # attention impl the shipped class must be BUILT with (None = engine's)
    dense_fp32_modules: tuple = ()            # class names whose packed weight is decoded once to a dense parameter
    vision_adapter: str | None = None         # "module:Class" of the image-input adapter (None = the generic one)

    def matches(self, config) -> bool:
        raise NotImplementedError

    def serve_defaults(self) -> ServeDefaults:
        return ServeDefaults()

    def text_config(self, config):
        """The decoder's config inside the checkpoint's top-level config."""
        return getattr(config, "text_config", config)

    def build_skeleton(self, model_dir: str, config, attn_impl: str):
        """(model, text_config) built under init_empty_weights, or None = AutoModelForCausalLM.from_config."""
        return None

    def install(self, model, *, model_dir: str, config, tok) -> None:
        """Called once after the weights are in place: choose the layer / cache kinds for this model."""

    def generate_fn(self):
        """The decode loop for one request, or None = spec_decode.speculative_generate."""
        return None

    def build_drafter(self, model, draft_dir: str, *, device: str = "cuda", pack_dir: str | None = None):
        from qv_band.decode.dflash_draft import build_dflash_drafter   # picks the drafter kind from its config
        return build_dflash_drafter(model, draft_dir, device=device, pack_dir=pack_dir)

    def default_draft_dir(self, pack_dir: str) -> str | None:
        return None

    def default_draft_pack(self, pack_dir: str) -> str | None:
        return None


# -- registry ------------------------------------------------------------------------------------------
_REGISTRY: list[ModelAdapter] = []
_ENTRY_POINTS_LOADED = False
_IN_TREE = ("ripple",)              # imported lazily so the engine never imports models at load time


def register(adapter: ModelAdapter) -> ModelAdapter:
    if any(a.name == adapter.name for a in _REGISTRY):
        raise ValueError(f"adapter {adapter.name!r} already registered")
    _REGISTRY.append(adapter)
    return adapter


def registered() -> list[str]:
    _discover()
    return [a.name for a in _REGISTRY]


def _discover() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    import importlib
    for name in _IN_TREE:
        try:
            importlib.import_module(f"qv_band.models.{name}.adapter")
        except ModuleNotFoundError:        # a release without that model part
            pass
    from importlib.metadata import entry_points
    for ep in entry_points(group="qv_band.models"):
        obj = ep.load()
        a = obj() if isinstance(obj, type) else obj
        if all(r.name != a.name for r in _REGISTRY):
            register(a)


def find_adapter(config) -> ModelAdapter | None:
    """The first registered adapter whose matches(config) is true; None = generic engine path.
    `config` is the checkpoint's top-level config (the loader) -- adapters match on that."""
    _discover()
    hits = [a for a in _REGISTRY if a.matches(config)]
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous adapters for this config: {[a.name for a in hits]}")
    return hits[0] if hits else None
