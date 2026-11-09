"""Pack-format registry.

A pack directory's manifest names its format; everything that has to build
modules from it looks the classes up HERE rather than branching on the
name. Adding a format means adding one entry, not hunting for every
`if fmt == ...` in the loader, the MTP head and whatever comes next.

    spec = get_format(manifest["format"])
    lin  = spec.linear.from_packfile(path, device=dev)
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class FormatSpec:
    """How to read one pack format.

    name      manifest["format"] value, and the pack files' extension
    linear    nn.Linear replacement class
    embedding nn.Embedding replacement class
    experts   MoE expert-stack class, or None if the format has no reader.
              None means "stream that stack dense": a caller must NOT fall
              back to another format's class, which would decode the bytes
              with the wrong codec and return silent garbage.
    """
    name: str
    linear: type
    embedding: type
    experts: type | None
    sharded_linear: type | None = None   # tensor-parallel reader
    cpu_linear: type | None = None       # CPU-only reader
    # CPU embedding reader, when the format has one whose constructor
    # differs from `embedding` (the earlier format's takes mmap_file/cache_rows, not
    # device). None means `embedding` already works on CPU (QVB8's does:
    # from_packfile(..., device="cpu")).
    cpu_embedding: type | None = None
    # Callable(linear_instance) -> dense (Q, P) tensor. Each format plugs
    # its own decoder in here, so nothing downstream tests format names.
    decoder: "object | None" = None
    # Callable(model) run once by the loader after every module is swapped
    # (e.g. qvb8r fuses same-input linears into one launch). None: nothing.
    finalize: "object | None" = None

    def filename(self, tensor_name: str) -> str:
        """Pack filename for a tensor, in this format's extension."""
        import re
        return re.sub(r"[^A-Za-z0-9]", "_", tensor_name) + "." + self.name


_REGISTRY: dict[str, FormatSpec] = {}


def _build_registry() -> None:
    if _REGISTRY:
        return
    def _opt(mod: str, cls: str):
        """Import a backend class if its module is available."""
        try:
            return getattr(__import__(mod, fromlist=[cls]), cls)
        except Exception:
            return None

    from qv_band.formats.qvb8.module import QVBandLinearV8
    from qv_band.formats.qvb8.embedding import QVBandEmbeddingV8
    def _decode_v8(lin):
        from qv_band.formats.qvb8.module import _load_ext
        return _load_ext().decode(lin.ids, lin.min_s, lin.nbits, lin.bit_off,
                                  lin.in_features, lin.tau_s,
                                  lin.esc_idx, lin.esc_vals)

    from qv_band.formats.qvb8.backends_experts import (QVBandExpertsV8,
                                                 QVBandShardedLinearV8,
                                                 QVBandLinearCPUV8)
    _REGISTRY["qvb8"] = FormatSpec("qvb8", QVBandLinearV8,
                                   QVBandEmbeddingV8, QVBandExpertsV8,
                                   sharded_linear=QVBandShardedLinearV8,
                                   cpu_linear=QVBandLinearCPUV8,
                                   decoder=_decode_v8)
    # fp8n/fp8q (native hardware FP8, FP8Linear/FP8Experts) moved to the
    # separate qv_band_fp8 repo -- this repo stays pure bf16 (QVB8).

    # "qvband" is QVB8's on-disk container name (pack/pack_qvband.py).
    # QVB8 is the format version; fp16 (and by default bf16 too) is its restore precision
    # within it, tagged in the pack's own header and dispatched
    # internally by QVBandLinearV8/QVBandEmbeddingV8's from_packfile.
    # Same classes as "qvb8", own `name` since FormatSpec.filename
    # derives the on-disk extension from it and packs are written as
    # "<tensor>.qvband".
    _REGISTRY["qvband"] = dataclasses.replace(_REGISTRY["qvb8"], name="qvband")

    # qvb8r: the same band ids, rANS-coded (formats/qvb8r). A qvb8r pack
    # dir keeps a few tensors as .qvband (too small / outside the codec's
    # limits); the loaders read each tensor by its own extension, so those
    # resolve to the "qvband" entry above. No MoE expert reader: expert
    # stacks of a qvb8r pack stream dense (see FormatSpec.experts).
    from qv_band.formats.qvb8r.module import QVBandLinearV8R, QVBandEmbeddingV8R
    from qv_band.formats.qvb8r.fuse import fuse_same_input
    _REGISTRY["qvb8r"] = FormatSpec("qvb8r", QVBandLinearV8R,
                                    QVBandEmbeddingV8R, None,
                                    # fp32 band value (escapes exact); decode_pack rounds ONCE to the dtype asked for
                                    decoder=lambda lin: lin.dense(__import__("torch").float32),
                                    finalize=fuse_same_input)


def get_format(name: str | None) -> FormatSpec:
    _build_registry()
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValueError(
            f"pack format {name!r} is not supported -- known formats: "
            f"{', '.join(sorted(_REGISTRY))}")
    return spec


def require(spec: FormatSpec, backend: str):
    """The class `spec` provides for `backend`, or a clear error.

    Lets a path that only some formats support (tensor-parallel, CPU,
    MoE experts) state its requirement once instead of testing format
    names.
    """
    cls = getattr(spec, backend, None)
    if cls is None:
        raise NotImplementedError(
            f"pack format {spec.name!r} has no {backend} reader -- this "
            f"path needs one, and reading the packs with another format's "
            f"class would return silent garbage.")
    return cls


def decode_pack(path: str, spec: FormatSpec | None = None,
                device: str = "cuda", dtype=None):
    """Dense (Q, P) weights from a pack file, whatever wrote it (bf16 unless `dtype`).

    The ONE place that turns a pack back into raw weights, for the callers
    that genuinely need them -- re-packing a model into another codec, a
    numeric audit, a conv/bare tensor that must stay dense. Inference never comes
    here: a linear runs its packed kernel, which is the whole point of the format.

    Kept out of the module classes deliberately. Both codecs needed this
    and each grew its own dense() method, which put a packing-time concern
    on the inference hot-path class and duplicated the same idea twice.

    `spec` defaults to the format the file extension names. A QVB8 pack on
    `device="cpu"` is decoded by the CPU reader (numpy): the CUDA decode kernel
    must never be handed host tensors (it faults with an illegal memory access).
    """
    import torch
    dtype = torch.bfloat16 if dtype is None else dtype
    if spec is None:
        spec = get_format(path.rsplit(".", 1)[-1])
    if str(device).startswith("cpu") and spec.name in ("qvb8", "qvband"):
        return spec.cpu_linear.from_packfile(path)._weights().to(dtype)
    decode = require(spec, "decoder")
    # A format without a CPU reader (qvb8r: rANS decoded by CUDA kernels) is decoded on the GPU and handed back on the host.
    run_on = "cuda" if str(device).startswith("cpu") and torch.cuda.is_available() else device
    lin = spec.linear.from_packfile(path, device=run_on)
    try:
        W = decode(lin).to(dtype)
        return W.cpu() if run_on != device else W
    finally:
        del lin


def known_formats() -> tuple[str, ...]:
    _build_registry()
    return tuple(sorted(_REGISTRY))
