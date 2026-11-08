"""qvb8r: QVB8 band ids losslessly rANS-coded, decoded on the fly by the GPU kernels (13-22% smaller than QVB8).
codec.py (encoder), container.py (.qvb8r file), pack.py (QVB8 pack -> qvb8r pack), module.py (readers),
fuse.py (same-input fusion), backends/cuda_qvb8r (kernels + CPU embedding reader)."""
from .module import QVBandLinearV8R, QVBandEmbeddingV8R

__all__ = ["QVBandLinearV8R", "QVBandEmbeddingV8R"]
