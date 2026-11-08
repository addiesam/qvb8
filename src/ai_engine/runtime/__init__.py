"""Device resolution and process resource limits -- modality-agnostic.

This is the layer every backend and every app sits on: which device do we
run on, and what is our memory budget? It used to be a sibling `engine`
package, and it moved here because it is the true bottom of the ladder --
nothing in it knows what a weight format is, or that a server exists.

    device.py      resolve_device / DevicePlan / prepare_inference
    memlimit.py    the --max-memory-gb ceiling, polled per decode step
    sampling.py    Qwen think/no-think presets
    seed.py        set_seed -- reproducibility for anything sampled

image, lip-sync, TTS, ASR and LLM all resolve a device through
device.resolve_device, so one device string means one thing everywhere.
The same goes for seeding: a TTS wav and an LLM completion are both
sampled generations, so both seed through runtime.seed.set_seed rather
than each re-implementing which RNGs to touch.
"""
from ai_engine.runtime.seed import set_seed                    # noqa: F401
from ai_engine.runtime.memlimit import (          # noqa: F401
    add_memory_limit_arg, apply_unified_memory_limit, has_unified_memory,
    rss_gb, vram_abort_check,
)
from ai_engine.runtime.sampling import (          # noqa: F401
    CHAT_SKIP_SPECIAL_TOKENS, GREEDY_PRESET, NO_THINK_PRESET,
    PIPELINE_SKIP_SPECIAL_TOKENS, THINK_PRESET,
    resolve_sampling, sampling_preset,
)
from ai_engine.runtime.device import (           # noqa: F401
    DEVICE_CHOICES, DevicePlan, accelerator_stats, load_timed,
    prepare_inference, resolve_device, valid_device, vram_used_bytes,
)

__all__ = [
    # memlimit
    "add_memory_limit_arg", "apply_unified_memory_limit", "has_unified_memory",
    "rss_gb", "vram_abort_check",
    # sampling
    "CHAT_SKIP_SPECIAL_TOKENS", "GREEDY_PRESET", "NO_THINK_PRESET",
    "PIPELINE_SKIP_SPECIAL_TOKENS", "THINK_PRESET", "resolve_sampling",
    "sampling_preset",
    # device
    "DEVICE_CHOICES", "DevicePlan", "accelerator_stats", "load_timed",
    "prepare_inference", "resolve_device", "valid_device", "vram_used_bytes",
    # seed
    "set_seed",
]
