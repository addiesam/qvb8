"""Serve the LLM with the CUDA kernels this GPU can actually EXECUTE.

Why this exists: a compiled CUDA extension imports fine on any box and only
fails when it is CALLED. `causal_conv1d`'s wheel here was built for sm_120
(Blackwell); on a V100 (sm_70, Volta) `import causal_conv1d` succeeds and
`causal_conv1d_fn(...)` then raises

    AcceleratorError: CUDA error: no kernel image is available
    for execution on the device

transformers' Qwen3.5/3.8 Gated-DeltaNet layers bind their fast-path kernels
at CONSTRUCTION time based only on "is the package importable" -- there is no
per-call capability check -- so an unpatched load binds a kernel that cannot
run and dies on the first forward, minutes after a successful-looking load.
qv_band's CPU loader already does this exact patch (run_inference_cpu.py:101);
the GPU path did not, which is the gap closed here.

flash-attn needs the opposite treatment and is handled by an env var, not a
patch: it is only reachable through `_attn_implementation`, so QV_ATTN_IMPL=sdpa
(the torch-native path, valid on every capability) is enough.

Run exactly like the server it wraps:

    torchrun --nproc_per_node=2 -m ai_engine.cli.serve PACK --host ... --port ...
"""
from __future__ import annotations

import sys


def _can_execute_causal_conv1d() -> bool:
    """True iff causal_conv1d's kernel can actually launch on this GPU."""
    try:
        import torch
        from causal_conv1d import causal_conv1d_fn
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(1, 8, 16, device="cuda", dtype=torch.bfloat16)
        w = torch.zeros(8, 3, device="cuda", dtype=torch.bfloat16)
        causal_conv1d_fn(x, w)
        torch.cuda.synchronize()
        return True
    except Exception as e:                    # no sm_XX kernel for this card
        print(f"[serve] causal_conv1d cannot run here: {type(e).__name__}: "
              f"{str(e)[:120]}", file=sys.stderr, flush=True)
        return False


def disable_unusable_kernels(verbose: bool = True) -> None:
    """Null out fast-path kernels that import but cannot execute on this GPU.

    Only touches what is actually broken: `fla` is Triton (JIT-compiled per
    device) and runs fine on sm_70, so it is deliberately left alone.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q35
    except Exception:
        return                                # not this architecture; nothing to do

    if _can_execute_causal_conv1d():
        return

    # Only the causal-conv pair. `fla`'s scan is Triton (JIT-compiled per
    # device) and verified running on sm_70, and torch_chunk_gated_delta_rule
    # is the real attribute name here -- chunk_gated_delta_rule /
    # fused_recurrent_gated_delta_rule do not exist on this module, so
    # nulling the names the CPU loader lists would silently do nothing here.
    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        fn = getattr(q35, name, None)
        if fn is None and not hasattr(q35, name):
            continue
        # Newer transformers call these unconditionally (the decorated
        # function carries its pure-torch body in __wrapped__); older ones
        # test them for None. Use the torch body when there is one, else None.
        setattr(q35, name, getattr(fn, "__wrapped__", None))
    if verbose:
        print("[serve] disabled causal_conv1d kernels -> pure-torch "
              "Gated-DeltaNet conv (this GPU has no matching cubin); "
              "fla's Triton scan stays enabled", flush=True)


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
    # Under torchrun every rank must touch only its own card. The probe below
    # runs on device="cuda", which is cuda:0 until set_device: without this,
    # rank 1 leaves a ~364 MiB CUDA context on card 0 -- memory a 16 GB V100
    # does not have to spare.
    import os
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    disable_unusable_kernels()
    tolerate_autotune_launch_oom()
    from ai_engine.servers.llm_server import main as server_main
    server_main()


if __name__ == "__main__":
    main()