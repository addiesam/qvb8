"""Serve the LLM (python -m ai_engine.cli.serve PACK --host ... --port ...; several cards: add --pipeline cuda:0,cuda:1).
"""
from __future__ import annotations

from ai_engine.cli import kv_offload, pipeline_flags
kv_offload.apply_from_argv()
pipeline_flags.apply_from_argv()      # --kv-offload ...: must precede the first qv_band.kv import


def tolerate_autotune_launch_oom() -> None:
    """Let Triton's autotuner skip a config whose LAUNCH runs out of memory.

    Triton already skips configs that fail with OutOfResources, but a config
    whose spilled registers need more local memory than the card has left
    fails at launch with a plain RuntimeError ("Triton Error [CUDA]: out of
    memory") -- and that took the whole request down. On a 16 GB V100 with a
    27B loaded this is exactly what fla's fp16 Gated-DeltaNet kernels hit on
    their first autotune. A launch OOM is not sticky, so the config is simply
    scored as unusable, like OutOfResources.
    """
    try:
        from triton.runtime.autotuner import Autotuner
    except Exception:
        return
    if getattr(Autotuner._bench, "_qv_oom_tolerant", False):
        return
    orig = Autotuner._bench

    def _bench(self, *args, config, **meta):
        try:
            return orig(self, *args, config=config, **meta)
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise
            return [float("inf"), float("inf"), float("inf")]

    _bench._qv_oom_tolerant = True
    Autotuner._bench = _bench


def main() -> None:
    tolerate_autotune_launch_oom()
    from ai_engine.servers.llm_server import main as server_main
    server_main()


if __name__ == "__main__":
    main()