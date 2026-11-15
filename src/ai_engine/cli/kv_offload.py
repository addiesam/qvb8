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
    --prefix-store gpu|host  where the scheduler keeps saved prompt prefixes (its linear-attention state and dense KV
                         residual, ~150 MB each for a 27B; one per --prefill-chunk of a long prompt, up to 12). gpu (default):
                         restore is instant. host: pinned host RAM, copied back on a restore -- a full 16 GB card has no
                         room for them (27B on one V100: out of memory on the second chunk of any prompt over ~4k tokens)
                                                                           -> QV_PREFIX_STORE
                         shared: one pool in /dev/shm (RAM, never disk) for every server on the machine with the same
                         pack and settings; a prefix is kept once and any of them restores it as views of the pool
                         (no copy). Size: --prefix-shared-gb (12) -> QV_PREFIX_SHARED_GB; name in /dev/shm:
                         --prefix-shared-name (qv_prefix_<pack folder>)            -> QV_PREFIX_SHARED_NAME
    --prefix-checkpoint-every N  mid-prompt checkpoints of a long prefill on a fixed grid of positions N, 2N, ...
                         (a multiple of --prefill-chunk; default: the smallest multiple of it >= 8192); a restore
                         re-prefills at most N tokens                      -> QV_PREFIX_CKPT_EVERY
    --prefix-state exact|qvb8  what a checkpoint keeps of the linear-attention state (~150 MB fp32 per checkpoint for a
                         27B). exact (default): an fp32 copy. qvb8: the KV cache's codec at the MODEL's tau (read from the
                         pack manifest), a head that cannot meet it in 16 bits kept exact; every prefill rounds the state
                         the same way at each grid position, so a restore equals an uncached run. ~0.24x the size; the
                         2026-11-12 PoC at tau 2e-3: perplexity change < 0.001, top-1 agreement 99.3-99.75 %
                                                                           -> QV_PREFIX_STATE

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
    ap.add_argument("--prefix-checkpoint-every", type=int, default=None, metavar="N",
                    help="mid-prompt checkpoints on a grid of N tokens (a multiple of --prefill-chunk; default >= 8192), exact states")
    ap.add_argument("--prefix-state", choices=("exact", "qvb8"), default=None,
                    help="checkpointed linear-attention state: exact fp32 (default) or qvb8 at the model's tau "
                         "(see ai_engine/cli/kv_offload.py)")
    ap.add_argument("--prefix-store", choices=("gpu", "host", "shared"), default=None,
                    help="saved prompt prefixes of the scheduler: gpu (default), pinned host RAM, or one host-RAM pool "
                         "shared by every server on the machine (see ai_engine/cli/kv_offload.py)")
    ap.add_argument("--prefix-shared-gb", type=float, default=None,
                    help="--prefix-store shared: the pool's size in GiB (default 12)")
    ap.add_argument("--kv-remote", default=None, metavar="URL",
                    help="a KV host (ai_engine.servers.kv_host: the V100 or the DRAM PC) for the OLDEST KV blocks of long "
                         "requests: it keeps them and computes their attention (with --kv-window: the blocks beyond it)")
    ap.add_argument("--prefix-backup", default=None, metavar="URL",
                    help="--prefix-store shared: a prefix store holding copies (the DRAM PC); prefixes it holds are evicted "
                         "here first, so a small pool loses nothing")
    ap.add_argument("--prefix-shared-name", default=None,
                    help="--prefix-store shared: the pool's name in /dev/shm (default qv_prefix_<pack folder>); the "
                         "launcher creates <name>.arena so its RAM is not charged to one server")


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
    if a.prefix_store is not None:
        os.environ["QV_PREFIX_STORE"] = a.prefix_store
    if a.kv_remote is not None:
        os.environ["QV_KV_REMOTE"] = a.kv_remote
    if a.prefix_backup is not None:
        os.environ["QV_PREFIX_BACKUP"] = a.prefix_backup
    if a.prefix_shared_name is not None:
        os.environ["QV_PREFIX_SHARED_NAME"] = a.prefix_shared_name
    if a.prefix_shared_gb is not None:
        os.environ["QV_PREFIX_SHARED_GB"] = str(a.prefix_shared_gb)
    if a.prefix_state is not None:
        os.environ["QV_PREFIX_STATE"] = a.prefix_state
    if a.prefix_checkpoint_every is not None:
        if a.prefix_checkpoint_every < 1:
            raise SystemExit("--prefix-checkpoint-every: a token count >= 1")
        os.environ["QV_PREFIX_CKPT_EVERY"] = str(a.prefix_checkpoint_every)

