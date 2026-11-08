"""Single process-wide memory ceiling, shared by llm_server.py and chat.py.

One number, --max-memory-gb, ABSOLUTE (31 means 31 * 1e9 bytes, never a
fraction of anything). The guard it maps to depends on the hardware:

  * CPU only: caps host RAM via RLIMIT_AS. There is no VRAM to speak of.
  * Discrete GPU / unified memory: caps CUDA memory with TWO guards
    together, not one -- see below for why a single guard was not enough
    (measured: a real request OOM'd with "48G ceiling" printed and no
    effect at all).

WHY TWO GUARDS, NOT JUST THE POLL. The original design used only
vram_abort_check(): the generation loop polls it once per DECODE STEP, so
it can only intervene BETWEEN steps. A prefill forward -- the whole prompt
in one shot, which is exactly what a large context (a long Cline
conversation, say) sends -- is a SINGLE step. If that one forward's own
activations plus the KV cache it allocates exceed the ceiling, the process
OOMs inside that one CUDA call, before the poll ever runs again. The
--max-memory-gb print was truthful about what it does; what it does turned
out not to cover this case.

So there are now two guards, doing two different jobs:

  1. torch.cuda.set_per_process_memory_fraction -- a REAL allocator-level
     limit, checked by CUDA itself on every allocation, including inside a
     single forward. This is what actually stops a prefill-sized spike.
     It caps RESERVED bytes (live + cached-free + fragmentation), not live
     bytes, and takes a fraction -- so it trips at a device-dependent
     number somewhat BELOW the nominal --max-memory-gb (measured: a 31 GB
     request tripped around 25-29 GB live). That undershoot is the
     deliberate trade: an early, hard stop beats a late, absent one.
  2. vram_abort_check(), still polled once per decode step -- gives a
     CLEAN stop (return what was generated so far) for the case it CAN
     catch: a long decode loop whose KV cache grows past the ceiling one
     token at a time. When guard 1 fires first (a single-step spike),
     generation raises OutOfMemoryError instead of returning cleanly --
     there is nothing to return, no tokens were produced by that step.

Caveats:

  * Guard 1 undershoots the nominal ceiling by an amount that varies with
    the device's total memory; treat --max-memory-gb as "the process will
    not exceed roughly this," not an exact byte count. Guard 2 is exact
    but only fires between steps.
  * RLIMIT_AS is NOT used on the GPU path: it caps virtual address space,
    and the CUDA driver reserves tens of GB it never faults in (measured:
    ~9 GB live OOM'd against a 44 GB RLIMIT_AS). CPU-only runs still use
    RLIMIT_AS since there is no CUDA allocator to guard instead.
  * Bounds THIS process only; a second heavy job on the box still competes
    for the same pool.
"""
from __future__ import annotations

import os


def _total_ram_gb() -> float | None:
    """Total physical RAM in GB, or None if it cannot be determined."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def rss_gb() -> float:
    """Resident set size of THIS process in GB, or NaN if unavailable.

    The host-total counterpart is _total_ram_gb; this is the per-process
    figure both applications print in their ready banner ("ready in 4.2s,
    RSS 12.3 GB"). NaN rather than 0.0 so a caller formatting it prints
    "nan" and the gap is visible instead of looking like a real measurement.

    psutil is an optional extra (see the memlimit extra), so its absence is
    expected rather than exceptional.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return float("nan")


def has_unified_memory() -> bool:
    """True on Grace-class boxes (GH200, ...) where CUDA device memory
    and host RAM are the same physical pool. Detected via
    `nvidia-smi --query-gpu=memory.total`: a discrete card reports a real
    number there, a unified part reports "[N/A]" because there is no
    separate VRAM total to report. Returns False (i.e. "assume discrete")
    if nvidia-smi is missing or the query fails, which is the safe default
    for a normal GPU box.
    """
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and "N/A" in out.stdout
    except Exception:
        return False


def _apply_host_limit(max_gb: float, *, verbose: bool) -> bool:
    try:
        import resource
    except ImportError:                      # non-POSIX
        if verbose:
            print(f"[mem] --max-memory-gb {max_gb:g} ignored on host RAM: "
                  f"RLIMIT_AS is not available on this platform", flush=True)
        return False

    nbytes = int(max_gb * 1e9)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)

    # Never raise the hard limit (unprivileged processes cannot anyway) and
    # never silently exceed an existing one set by whoever launched us.
    if hard != resource.RLIM_INFINITY and nbytes > hard:
        if verbose:
            print(f"[mem] --max-memory-gb {max_gb:g} exceeds the inherited "
                  f"hard limit {hard/1e9:.1f} GB; clamping to it", flush=True)
        nbytes = hard

    try:
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, hard))
    except (ValueError, OSError) as e:
        if verbose:
            print(f"[mem] could not set host memory limit to {max_gb:g} "
                  f"GB: {e}", flush=True)
        return False

    if verbose:
        total = _total_ram_gb()
        of_total = f" ({nbytes/1e9/total:.0%} of {total:.0f} GB)" if total else ""
        print(f"[mem] host RAM ceiling {nbytes/1e9:.1f} GB{of_total} "
              f"-- allocations past this raise MemoryError", flush=True)
        if total and nbytes / 1e9 > total:
            print("[mem] NOTE: ceiling exceeds physical RAM, so it will "
                  "not bind before the machine does", flush=True)
    return True


# Set by _apply_vram_limit so a generation loop's per-step abort_check
# (see vram_abort_check) can enforce the SAME absolute ceiling by polling
# torch.cuda.memory_allocated(). None means "no VRAM ceiling in force".
VRAM_CEILING_BYTES: int | None = None


def _apply_vram_limit(max_gb: float, *, verbose: bool) -> bool:
    """Register an ABSOLUTE VRAM ceiling: --max-memory-gb 31 means "this
    process must not hold much more than 31 GB of CUDA memory."

    Two guards, both set here -- see this module's docstring for the full
    story of why one alone was not enough (it printed a ceiling and OOM'd
    anyway on a large-prompt request):

      1. torch.cuda.set_per_process_memory_fraction(max_gb / total_device_
         memory) -- the CUDA allocator enforces this itself, including
         inside a single forward pass (a big prefill), which is exactly
         the case per-step polling structurally cannot catch. Caps
         RESERVED bytes, so the real trip point runs somewhat below
         max_gb; that undershoot is accepted in exchange for it actually
         firing before the process OOMs the box.
      2. VRAM_CEILING_BYTES, polled by vram_abort_check() once per decode
         step -- gives a clean stop (return what was generated) for a long
         decode loop that grows past the ceiling gradually, which guard 1
         would otherwise turn into a hard OutOfMemoryError with nothing
         returned.
    """
    global VRAM_CEILING_BYTES
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        want_bytes = int(max_gb * 1e9)
        VRAM_CEILING_BYTES = want_bytes

        total_bytes = torch.cuda.get_device_properties(0).total_memory
        # Guard 1 sits ABOVE the poll ceiling on purpose. Setting the
        # allocator fraction to exactly want_bytes made the two guards
        # coincide: the allocator raised OutOfMemoryError AT the ceiling, so
        # memory_allocated() never reached it and guard 2 (the poll that
        # returns what was generated so far) could never fire -- the clean
        # stop this function documents was unreachable, and a long decode
        # over the ceiling died with a hard OOM instead.
        #
        # The headroom is what a single decode step can plausibly need past
        # the point where the poll would have stopped us. If even that is
        # exhausted, guard 1 still holds: the process gets a hard OOM rather
        # than taking the whole box down with it.
        headroom = max(2 * 1024 ** 3, want_bytes // 8)   # >= 2 GiB, >= 12.5%
        cap_bytes = want_bytes + headroom
        fraction = min(cap_bytes / total_bytes, 1.0)
        fraction_applied = fraction < 1.0
        if fraction_applied:
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)

        if verbose:
            if fraction_applied:
                print(f"[mem] VRAM ceiling {max_gb:g} GB: per-step poll at "
                      f"{want_bytes/1e9:.1f} GB (clean stop) + allocator "
                      f"hard cap {cap_bytes/1e9:.1f} GB "
                      f"({fraction:.1%} of {total_bytes/1e9:.0f} GB, catches a "
                      f"large prefill)", flush=True)
            else:
                print(f"[mem] --max-memory-gb {max_gb:g} GB >= this device's "
                      f"{total_bytes/1e9:.0f} GB total; no allocator fraction "
                      f"set (nothing to cap against), per-step poll only",
                      flush=True)
        return True
    except Exception as e:
        # Reset the ceiling this function itself may have already set
        # (VRAM_CEILING_BYTES = want_bytes, above) before failing partway
        # through -- an exception here means the allocator-level cap was
        # never actually installed, but leaving the global set would make
        # vram_abort_check() believe a ceiling is in force anyway, polling
        # and aborting a generation that has no real hard cap backing it
        # (worse than not calling this function at all: silent, spurious
        # aborts instead of the documented "no guard, no crash" contract).
        VRAM_CEILING_BYTES = None
        if verbose:
            print(f"[mem] could not set VRAM limit to {max_gb:g} GB: {e}",
                  flush=True)
        return False


def vram_abort_check(plan=None) -> bool:
    """Zero-arg callable for speculative_generate(abort_check=...): returns
    True once live accelerator memory has reached the --max-memory-gb ceiling,
    so the generation loop stops at its next step boundary and returns what it
    has produced so far. Returns False (never aborts) when no ceiling is in
    force or there is no accelerator. See _apply_vram_limit.

    Polls CUDA *or* XPU. This used to check torch.cuda only, so on Intel
    hardware the ceiling was registered and announced but never enforced --
    the run OOMed instead of stopping cleanly. Pass a resolved DevicePlan to
    skip re-detection per step; with None it auto-detects, which costs one
    cuda.is_available() per call and is fine at step cadence.
    """
    if VRAM_CEILING_BYTES is None:
        return False
    from ai_engine.runtime.device import resolve_device, vram_used_bytes
    plan = plan or resolve_device("auto")
    return vram_used_bytes(plan) >= VRAM_CEILING_BYTES


def apply_unified_memory_limit(max_gb: float | None, device: str = "auto",
                               *, verbose: bool = True) -> bool:
    """Apply --max-memory-gb as whichever guard(s) match this process's
    actual hardware. `device` is the resolved value of --device ("auto"
    resolves to CUDA if available, else CPU, exactly like the callers'
    own device selection).

    `max_gb` of None or <= 0 means "no limit" and is honoured as such -- an
    explicit opt-out, not an error. Returns True if at least one guard is
    now in force.
    """
    if not max_gb or max_gb <= 0:
        if verbose:
            print("[mem] no process memory limit (unbounded)", flush=True)
        return False

    try:
        import torch
        if device == "auto":
            use_cuda = torch.cuda.is_available()
        else:
            # "cuda" or a concrete "cuda:<int>" both mean the CUDA path.
            use_cuda = device == "cuda" or device.startswith("cuda:")
    except ImportError:
        use_cuda = False

    if not use_cuda:
        return _apply_host_limit(max_gb, verbose=verbose)

    # Both discrete GPU and unified memory: the ceiling is enforced by
    # polling live CUDA memory once per decode step (see _apply_vram_limit
    # and vram_abort_check). RLIMIT_AS is deliberately not added on either
    # -- on unified memory it trips far below max_gb of real use because of
    # the CUDA driver's un-faulted address reservations; on a discrete GPU
    # it does not see device memory at all.
    return _apply_vram_limit(max_gb, verbose=verbose)


def add_memory_limit_arg(ap, default: float | None = None) -> None:
    """Register --max-memory-gb on an ArgumentParser, worded identically in
    every entry point that offers it. Call apply_unified_memory_limit()
    with the parsed value AFTER --device is resolved -- which guard this
    number becomes depends on whether the run is CPU, discrete GPU, or
    unified memory."""
    ap.add_argument("--max-memory-gb", type=float, default=default,
                    help="hard ceiling on process memory in GB "
                         f"(default {'unlimited' if not default else format(default, 'g')}). "
                         "CPU run: bounds host RAM. Discrete-GPU run: "
                         "bounds VRAM. Unified-memory run (GH200, "
                         "...): bounds both, since they are the same "
                         "physical pool there. 0 disables it.")
