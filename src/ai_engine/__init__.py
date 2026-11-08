"""ai_engine: run and serve models, whatever format the weights are in.

This package collects everything between "a folder full of weights" and "an
HTTP response" into one place, so that adding a new weight format or a new
kind of server does not mean scattering changes across a dozen sibling
projects.

    runtime/     device resolution, memory ceiling, sampling presets.
                 The bottom of the ladder -- knows nothing about weight
                 formats or servers.
    web/         the hand-rolled BaseHTTPRequestHandler and the MCP stdio
                 loop every server in this package shares.
    generation/  the text-generation core: GenParams, model load, drafter
                 selection, the single speculative_generate() call.
    backends/    what a weight format has to be able to do. One file per
                 format, registered in backends/__init__.py.
    servers/     long-lived: a resident model, request state, a queue.
    workers/     short-lived: load once, run a batch of jobs, exit. Driven
                 as a subprocess by whichever pipeline needs them.
    pipelines/   one-shot CLIs that build a model, do a thing, and exit.
    cli/         interactive or one-shot text interfaces.

WHY SERVERS AND WORKERS ARE SEPARATE FOLDERS
--------------------------------------------
They have genuinely different lifecycles and confusing them is expensive.
A server holds a model resident for the process lifetime and arbitrates
concurrent requests; a worker loads a multi-billion-parameter model once,
renders a batch of jobs and exits -- reload-per-job is the difference
between a run that takes minutes and one that takes hours, which is exactly
why workers/sonic_qvb8.py and workers/lipsync.py exist as subprocess entry
points at all. Nothing in servers/ should be read as a batch script, and
nothing in workers/ should be given a socket.

THE DEPENDENCY LADDER
----------------------
    runtime/, web/          format- and modality-agnostic plumbing
        ^                        ^
        |                        |
    generation/  --------->  qv_band (codecs, pack readers, drafters)
        ^
        |
    backends/, servers/, workers/, pipelines/, cli/

qv_band is a pure library and imports nothing from this package. It used to
be otherwise: the generation core lived in qv_band.engine while needing
ai_engine's device resolution, so the two projects depended on each other.
Nothing inside qv_band ever imported that subpackage, which is what showed
it did not belong there.
"""
from ai_engine.runtime import (          # noqa: F401
    DEVICE_CHOICES, DevicePlan, accelerator_stats, add_memory_limit_arg,
    apply_unified_memory_limit, has_unified_memory, load_timed,
    prepare_inference, resolve_device, resolve_sampling, rss_gb,
    sampling_preset, valid_device, vram_abort_check, vram_used_bytes,
)
from ai_engine.web import QVHTTPHandler    # noqa: F401

__all__ = [
    "DEVICE_CHOICES", "DevicePlan", "QVHTTPHandler", "accelerator_stats",
    "add_memory_limit_arg", "apply_unified_memory_limit", "has_unified_memory",
    "load_timed", "prepare_inference", "resolve_device", "resolve_sampling",
    "rss_gb", "sampling_preset", "valid_device", "vram_abort_check",
    "vram_used_bytes",
]