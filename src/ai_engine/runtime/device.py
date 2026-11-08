"""Device resolution and the memory ceiling, for every modality.

This is the one place that answers "which device do we run on, and what is
our memory budget?". It used to live inside gen_core.py, which meant only the
LLM path honoured it -- image_gen, sonic_qvb8, lipsync and qwen_local each
rolled their own, and each got it wrong in a different way:

    image_gen     device was an opaque string, compared as `device == "cuda"`,
                  so "cuda:1" reported zero peak and "auto" was unhandled.
    sonic_qvb8    `0 if device == 'cpu' else (split(':')[1] if ':' in ...)`
                  turns "auto" and "xpu" into device_id 0 -- it silently picks
                  the wrong GPU instead of asking.
    lipsync       the same hand-rolled split, same failure.
    qwen_local    hardcoded "cuda:0" and refused anything else outright.

And the LLM path itself had a hole: memlimit.vram_abort_check polled
torch.cuda only, so on XPU the ceiling was registered but never enforced.

Accepted device values, everywhere, in every modality:

    "auto"      CUDA if present, else XPU, else CPU
    "cuda"      CUDA if present, else falls back to the host guard
    "cuda:N"    pin to that exact index (single-process multi-GPU)
    "xpu"       Intel XPU if present
    "cpu"       never touch an accelerator

The tensor-parallel remap also lives here: under torchrun a rank handed the
bare "cuda" actually runs on cuda:{LOCAL_RANK}, resolved inside
qv_band.run.run_inference.load_qv_model and never handed back. Recomputing the
same remap is what stops a rank sending its tensors to the wrong card --
observed as MTP hanging forever behind an unresolved implicit sync.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")

DEVICE_CHOICES = ("auto", "all", "cuda", "xpu", "cpu")


def valid_device(device: str) -> bool:
    """True for a bare DEVICE_CHOICES name or a concrete 'cuda:<int>'."""
    if device in DEVICE_CHOICES:
        return True
    return device.startswith("cuda:") and device[5:].isdigit()


@dataclass(frozen=True)
class DevicePlan:
    """A resolved device. Construct through resolve_device(), not directly."""

    gpu_dev: str | None      # "cuda:0" / "cuda" / "xpu", or None for CPU
    use_gpu: bool
    use_cuda: bool
    use_xpu: bool
    requested: str           # what the caller asked for, for error messages
    all_gpus: bool = False   # "all": let HF shard across every visible GPU
    gpu_count: int = 0       # visible CUDA devices

    @property
    def name(self) -> str:
        """Short human name: 'cuda:1', 'xpu', or 'cpu'."""
        return self.gpu_dev or "cpu"

    @property
    def index(self) -> int | None:
        """The ordinal for APIs that want an int (Sonic(device_id=...),
        torch.cuda.mem_get_info(...)). None for XPU and CPU."""
        if self.use_cuda and self.gpu_dev and ":" in self.gpu_dev:
            return int(self.gpu_dev.split(":", 1)[1])
        return 0 if self.use_cuda else None

    @property
    def hf_device_map(self) -> str:
        """The string to hand straight to from_pretrained(device_map=...).

        "auto" when more than one CUDA device is visible so HF shards across
        them, "cuda:0" for a single card, "cpu"/"xpu" otherwise.
        """
        if not self.use_gpu:
            return "cpu"
        if self.use_xpu:
            return "xpu"
        return "auto" if self.gpu_count > 1 else "cuda:0"

    def torch_device(self) -> Any:
        """A torch.device, for .to() / .Generator(device=...)."""
        import torch
        return torch.device(self.name)

    def accelerator(self):
        """The torch module for this device (torch.cuda / torch.xpu), or None."""
        if self.use_cuda:
            import torch
            return torch.cuda
        if self.use_xpu:
            import torch
            return torch.xpu
        return None

    def __str__(self) -> str:
        return self.name


def resolve_device(device: str = "auto", *, verbose: bool = False,
                   strict: bool = False) -> DevicePlan:
    """Resolve a device value to a DevicePlan.

    strict=False (default): an unavailable accelerator falls back to the host,
    which is what the LLM path has always done.

    strict=True: a request for a GPU that is not present RAISES, so a machine
    that must never fall back to a slow CPU run fails immediately instead of
    quietly taking hours. This is what the TTS/ASR/aligner settings use, and
    what script_to_video's own resolve_device() has always done -- the two
    policies are now one function with a flag instead of two functions.

    An unrecognised *string* always raises: that is a typo, not a hardware fact.
    """
    if not valid_device(device):
        raise ValueError(
            f"device must be one of {DEVICE_CHOICES} or 'cuda:<int>', "
            f"got {device!r}")

    import torch
    have_cuda = torch.cuda.is_available()
    have_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    gpu_count = torch.cuda.device_count() if have_cuda else 0

    def _fail(msg: str):
        raise SystemExit(
            f"device {device!r} {msg}. Use 'auto' to allow the CPU fallback, "
            f"or 'cpu' to force it.")

    if device == "cpu":
        gpu_dev = None
    elif device == "auto":
        gpu_dev = ("cuda" if have_cuda else "xpu" if have_xpu else None)
    elif device in ("cuda", "all"):
        if strict and not have_cuda:
            _fail("was requested but no CUDA device is available")
        gpu_dev = "cuda" if have_cuda else None
    elif device == "xpu":
        if strict and not have_xpu:
            _fail("was requested but no XPU device is available")
        gpu_dev = "xpu" if have_xpu else None
    else:  # cuda:<int>
        idx = int(device[5:])
        if not have_cuda:
            # An index into hardware that is not there is not a pin: treat it
            # like bare "cuda" (host fallback, or a hard failure when strict)
            # instead of reporting use_cuda=True on a CPU-only machine.
            if strict:
                _fail("was requested but no CUDA device is available")
            gpu_dev = None
        elif idx >= gpu_count:
            _fail(f"is out of range; only {gpu_count} CUDA device(s) are "
                  f"visible (valid: cuda:0..cuda:{gpu_count - 1}). "
                  f"Check CUDA_VISIBLE_DEVICES.")
        else:
            gpu_dev = device

    # TP remap: a rank given the generic "cuda" runs on cuda:{LOCAL_RANK}.
    # A caller that already pinned "cuda:<int>" is left alone.
    if gpu_dev == "cuda" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        gpu_dev = f"cuda:{os.environ.get('LOCAL_RANK', '0')}"

    use_gpu = gpu_dev is not None
    use_cuda = use_gpu and gpu_dev.startswith("cuda")
    use_xpu = use_gpu and gpu_dev.startswith("xpu")
    plan = DevicePlan(gpu_dev=gpu_dev, use_gpu=use_gpu, use_cuda=use_cuda,
                      use_xpu=use_xpu, requested=device,
                      all_gpus=(device == "all"), gpu_count=gpu_count)

    if verbose and device != "auto" and plan.name != device:
        print(f"[device] asked for {device!r}, using {plan.name!r}",
              flush=True)
    return plan


def prepare_inference(pack_dir: str | None = None, *, device: str = "auto",
                      max_memory_gb: float | None = None,
                      cpu_threads: int | None = 1, verbose: bool = True,
                      plan: DevicePlan | None = None) -> DevicePlan:
    """Resolve the device and put the memory ceiling in force.

    Every modality's load path starts here, so --max-memory-gb means the
    same thing whether the model is a 9B LLM, a diffusion transformer, a
    lip-sync UNet or a TTS decoder.

    pack_dir, when given, is checked for a manifest.json so a typo fails as
    "not a pack folder" instead of somewhere inside the loader.

    cpu_threads defaults to 1: qv_band's own kernel pool already saturates
    every core, and torch's default intra-op pool contends with it for the
    small slice of ops still on torch (measured 1.62 -> 1.70 tok/s forcing
    torch down to 1). Pass None to leave torch's threading alone.
    """
    from ai_engine.runtime.memlimit import apply_unified_memory_limit

    if pack_dir is not None:
        if not os.path.isdir(pack_dir):
            raise FileNotFoundError(f"pack folder does not exist: {pack_dir}")
        if not os.path.isfile(os.path.join(pack_dir, "manifest.json")):
            raise FileNotFoundError(
                f"no manifest.json in {pack_dir} (not a QV-band pack folder)")

    plan = plan or resolve_device(device, verbose=verbose)

    if not plan.use_gpu and cpu_threads:
        import torch
        torch.set_num_threads(cpu_threads)

    # Before the model loads, so the ceiling covers the weights too.
    apply_unified_memory_limit(max_memory_gb, plan.name, verbose=verbose)
    return plan


def load_timed(fn: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
    """Call fn, return (result, elapsed_seconds). Every loader reports a
    load time, and they were each open-coding their own time.time() pair."""
    import time
    t0 = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - t0


def accelerator_stats(plan: DevicePlan) -> dict:
    """{'device', 'allocated_gb', 'peak_gb'} for CUDA or XPU; zeros on CPU.

    Replaces the `if device == "cuda"` string comparisons that made the image
    path report zero peak whenever the device was "cuda:1" or "auto".
    """
    acc = plan.accelerator()
    if acc is None:
        return {"device": "cpu", "allocated_gb": 0.0, "peak_gb": 0.0}
    try:
        return {"device": plan.name,
                "allocated_gb": acc.memory_allocated() / 1e9,
                "peak_gb": acc.max_memory_allocated() / 1e9}
    except Exception:
        return {"device": plan.name, "allocated_gb": 0.0, "peak_gb": 0.0}


def vram_used_bytes(plan: DevicePlan) -> int:
    """Live accelerator memory, or 0 on CPU. Used by the abort check so the
    ceiling is enforced on XPU too -- the previous CUDA-only poll left the
    ceiling registered but inert on Intel hardware."""
    acc = plan.accelerator()
    if acc is None:
        return 0
    try:
        return int(acc.memory_allocated())
    except Exception:
        return 0


__all__ = ["DEVICE_CHOICES", "DevicePlan", "resolve_device", "valid_device",
           "prepare_inference", "load_timed", "accelerator_stats",
           "vram_used_bytes"]
