"""Move a tensor between GPUs WITHOUT a peer-to-peer copy.

A plain `t.to("cuda:1")` from cuda:0 makes PyTorch enable peer access and copy
GPU to GPU directly. On a host whose IOMMU (Intel VT-d / AMD-Vi) translates
device DMA and has no passthrough (`iommu=pt`), that path is broken for PCIe
GPUs: the copy silently delivers zeros and the card then MMU-faults (seen on
2x V100 SXM2-on-PCIe: Xid 62 microcontroller halt, then Xid 31, driver hung
until reboot). Pipeline placement only moves small tensors between cards --
the hidden state at a stage boundary, rotary cos/sin, a few weights at load --
so staging them through host memory costs microseconds and never touches the
peer path. NCCL (tensor parallel) detects this itself and is unaffected.
"""
from __future__ import annotations

import torch


def to_device(t: torch.Tensor, device) -> torch.Tensor:
    """`t` on `device`; a GPU-to-different-GPU move goes through host memory."""
    dev = torch.device(device)
    if t.device == dev:
        return t
    if t.is_cuda and dev.type == "cuda":
        return t.to("cpu").to(dev)
    return t.to(dev)


class _PinRing:
    """Reusable pinned staging slots for small host->device copies.

    torch's .pin_memory() on a fresh small tensor calls cudaHostAlloc whenever
    no cached block of that size exists, and cudaHostAlloc stalls the cards.
    A pageable copy instead waits for the target stream to drain. Measured:
    both kept a V100 pipeline stage idle ~40% of a prefill. A slot is reused
    only after its previous copy finished (event), so data is never clobbered.
    """

    def __init__(self, slots: int = 256, nbytes: int = 1 << 17):
        self.buf = [None] * slots
        self.ev = [None] * slots
        self.nbytes = nbytes
        self.i = 0
        import threading
        self.lock = threading.Lock()

    def h2d(self, t: torch.Tensor, device) -> torch.Tensor:
        n = t.numel() * t.element_size()
        if n > self.nbytes:
            return t.pin_memory().to(device, non_blocking=True)
        with self.lock:
            s = self.i
            self.i = (self.i + 1) % len(self.buf)
            if self.ev[s] is not None:
                self.ev[s].synchronize()
            if self.buf[s] is None:
                self.buf[s] = torch.empty(self.nbytes, dtype=torch.uint8, pin_memory=True)
            src = self.buf[s][:n].view(t.dtype).view(t.shape)
            src.copy_(t)
            dev = torch.device(device)
            out = src.to(dev, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(dev))
            self.ev[s] = ev
        return out


_RING = None


def h2d_small(t: torch.Tensor, device) -> torch.Tensor:
    """A small CPU tensor on `device` without a stream sync or cudaHostAlloc."""
    global _RING
    if torch.device(device).type != "cuda":
        return t.to(device)
    if _RING is None:
        _RING = _PinRing()
    return _RING.h2d(t.contiguous(), device)
