"""Reproducibility: seed every RNG the sampler can draw from.

Moved here from script_to_video/generate_segments.py. Seeding is not a
property of any one pipeline -- it is a property of running a model, the
same way device resolution is -- so a chat reply and a video line that both
need a repeatable wav ask the same resolver for the same thing.

WHY IT TOUCHES SO MANY GENERATORS. Any of them left unseeded makes the run
non-reproducible, and which one actually gets sampled varies by model and
device: torch's global RNG for the sampler, plus the CUDA generators on an
accelerator, plus numpy for anything that draws outside torch. Seeding only
one of them was the bug this replaced.

DETERMINISM IS OPTIONAL AND DEFAULTS OFF FOR SPEED. cudnn.deterministic
forces algorithm selection and costs real throughput; a run that wants the
same bytes back on a second run asks for it explicitly.
"""
from __future__ import annotations

import random

__all__ = ['set_seed']


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG a sampled generation can draw from.

    deterministic=True additionally pins cuDNN's algorithm choice, which is
    what makes two runs byte-identical on the same hardware. Off by default
    because it costs throughput, and most callers only need the sampling
    stream pinned, not bitwise-identical kernels.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False