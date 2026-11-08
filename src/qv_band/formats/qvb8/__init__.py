"""QVB8: per-row virtual-band packs. Calibration/pack writer (pack*.py, source_detect.py) and
readers (module.py, embedding.py, backends_experts.py, ...) plus CUDA kernels (backends/cuda_qvb8)."""
from .module import QVBandLinearV8
__all__ = ["QVBandLinearV8"]
