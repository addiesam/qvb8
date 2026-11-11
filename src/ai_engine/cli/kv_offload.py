"""KV-cache host-RAM tiering as server flags, with the V100 launchers' meaning (scripts/v100/Qwen3.8-27B_2e-3.sh --window /
--cold-block). The engine reads QV_KV_* once, when qv_band.kv.fused_attn is imported, so the flags must reach the
environment BEFORE that import: call apply_from_argv() first thing in an entry point.

    --kv-window N|auto   N >= 1 = number of NEWEST TOKENS per layer kept in VRAM (not GB); older KV (its packed copy and
                         its id window) lives in pinned host RAM              -> QV_KV_HOST=1, QV_KV_GPU_WINDOW=N
                         N = 1 is the most host RAM the engine allows (the GPU still holds one cold block + the current
                         prefill chunk). There is no 0: the engine's own 0 means unlimited.
                         auto = all in VRAM while there is room; a layer whose KV buffer must grow while free VRAM would
                         drop below --kv-gpu-reserve-mb freezes its window there, and only its oldest blocks go to host
                         RAM from then on (the V100 launcher's mode)         -> QV_KV_HOST=1, QV_KV_GPU_WINDOW=auto
                         Not given: no offload -- the whole KV cache stays in VRAM and grows until the memory ceiling /
                         OOM (QV_KV_* of the environment apply unchanged).
    --kv-gpu-reserve-mb M  auto only: VRAM kept free, MB (default 2048)       -> QV_KV_GPU_RESERVE_MB
    --kv-host            packed KV that attention no longer reads goes to pinned host RAM: a sliding-window layer's
                         segments older than window + 1024 tokens, a full-attention slot's older than 1024 tokens once
                         it attends fused (its uint8 id buffer). Decode speed unchanged; a new turn's prefill reads the
                         old history back once. --kv-window implies it.   -> QV_KV_HOST=1
                         (On a the dev GPU CPU and GPU share one memory: nothing is saved there.)
    --kv-cold-block T    host-RAM block size, tokens (default 4096)           -> QV_KV_COLD_BLOCK

Older KV is streamed back over PCIe on every attention call and attended exactly (merged by log-sum-exp), so the output
equals the all-VRAM path; only time is added. The QV_KV_* variables are how the flags reach the engine (it reads them
once, at import): set the flags, not the variables.
"""
import argparse
import os
import sys


def add_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--kv-window", default=None, metavar="N|auto",
                    help="newest N tokens per layer kept in VRAM (N >= 1; 1 = most in host RAM), older KV in host RAM; "
                         "auto = spill only when free VRAM would drop below --kv-gpu-reserve-mb. "
                         "Not given: no offload, all KV in VRAM. See ai_engine/cli/kv_offload.py")
    ap.add_argument("--kv-gpu-reserve-mb", type=int, default=None, metavar="M",
                    help="--kv-window auto: VRAM kept free, MB (default 2048)")
    ap.add_argument("--kv-host", action="store_true", default=None,
                    help="packed KV that attention no longer reads -> pinned host RAM (see ai_engine/cli/kv_offload.py)")
    ap.add_argument("--kv-cold-block", type=int, default=None, metavar="T",
                    help="--kv-window: host-RAM KV block, tokens (default 4096)")


def apply_from_argv(argv=None) -> None:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_flags(ap)
    a, _ = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    if a.kv_window is not None:
        w = a.kv_window.strip().lower()
        if not (w == "auto" or (w.isdigit() and int(w) >= 1)):
            raise SystemExit(f"--kv-window {a.kv_window!r}: expected auto or a token count >= 1 (1 = most in host RAM; "
                             f"to keep all KV in VRAM leave the flag out)")
        os.environ["QV_KV_HOST"], os.environ["QV_KV_GPU_WINDOW"] = "1", w
    if a.kv_host:
        os.environ["QV_KV_HOST"] = "1"
    if a.kv_gpu_reserve_mb is not None:
        os.environ["QV_KV_GPU_RESERVE_MB"] = str(a.kv_gpu_reserve_mb)
    if a.kv_cold_block is not None:
        os.environ["QV_KV_COLD_BLOCK"] = str(a.kv_cold_block)
