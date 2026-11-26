"""Weight streaming: the packed weights of a stack of blocks live in pinned host RAM and visit the GPU one block at a time.

A model whose packed weights do not fit the card (MiniMax-H3 transformer 20.5 GB, Qwen-Image-Edit transformer ~19 GB, on a 16 GB
V100) keeps only a window of `ahead + 1` blocks in VRAM: the block that is running and the next `ahead` ones, copied over a
second stream while the current one computes. For a video / image DiT every weight is used for thousands of tokens per pass, so
the copy (one block ~0.4 GB over PCIe) is hidden behind the compute (~1 s per block at 720p). The result is bit-identical to the
resident run: the same bytes, the same kernels.

Not for a model whose compute per weight is small (an LLM decoding 1-8 tokens per pass is bound by exactly this copy).

    streamer = BlockStreamer(model.transformer_blocks, "cuda:0", ahead=2, cyclic=True)   # blocks hold CPU buffers on entry

Only the buffers of QVBand* modules are streamed; every other parameter / buffer of a block is moved to the card once and stays.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

_PROFILE = os.environ.get("QV_STREAM_PROFILE") == "1"
_CACHES = ("_mma_cache", "_rb_cache")        # lazily built per-device tensors of QVBandLinearV8: must not outlive the block's visit


RESIDENT_FRACTION = 0.7  # a component whose packed weights are more than this share of the card's VRAM is streamed from host RAM
HOST_HEADROOM_GB = 6.0   # host RAM left for the OS, python, the page cache


def pack_bytes(cdir: str) -> int:
    """Packed size of a component directory (its manifest)."""
    import json
    import os
    return sum(v["bytes"] for v in json.load(open(os.path.join(cdir, "manifest.json")))["tensors"].values())


def vram_budget_bytes() -> int | None:
    """The VRAM this process may use per card (QV_VRAM_BUDGET_GB in GiB, as cards are sold, set by a server's --vram-budget-gb), or None = the whole card."""
    import os
    v = os.environ.get("QV_VRAM_BUDGET_GB", "").strip()
    return int(float(v) * 2**30) if v else None


def card_bytes(dev) -> int:
    """Size of the card as this process sees it: the whole card, or the budget when one is set (the smaller)."""
    total = torch.cuda.get_device_properties(dev).total_memory
    b = vram_budget_bytes()
    return total if b is None else min(total, b)


def free_bytes(dev) -> int:
    """Free VRAM as this process sees it: the card's free memory, capped by budget - what this process already reserved."""
    free = torch.cuda.mem_get_info(dev)[0]
    b = vram_budget_bytes()
    return free if b is None else max(0, min(free, b - torch.cuda.memory_reserved(dev)))


def apply_vram_budget() -> None:
    """Make the budget real on every visible card (an allocation past it raises OOM, as on a smaller card). Call once at start."""
    b = vram_budget_bytes()
    if b is None or not torch.cuda.is_available():
        return
    # A budget makes every cached byte count: the default allocator hoards dead
    # segments (measured 0.9-1.8 GiB reserved-but-unallocated at OOM time on an
    # 8 GiB budget -- the VAE decode's 600 MiB allocation died on that waste).
    # Expandable segments grow/shrink in 2 MiB units, so reserved ~= allocated.
    # Must be set before the first CUDA allocation, which is why this runs here
    # and not in the pipelines.
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, b / total), i)


def needs_streaming(cdir: str, dev) -> bool:
    """True when this component's packed weights would take more than RESIDENT_FRACTION of the card (activations need the rest)."""
    return pack_bytes(cdir) > RESIDENT_FRACTION * card_bytes(dev)


def host_can_keep(*sizes: int) -> bool:
    """Can the host RAM hold all of `sizes` (bytes) at once, with HOST_HEADROOM_GB spare? (/proc/meminfo MemTotal)"""
    with open("/proc/meminfo") as f:
        total = int(next(l for l in f if l.startswith("MemTotal")).split()[1]) * 1024
    return sum(sizes) + HOST_HEADROOM_GB * 1e9 <= total


def mark_placed(model: nn.Module) -> None:
    """A streamed model is placed already: a later model.to(device) / pipe.to(device) must not pull its host weights onto the card."""
    model.to = lambda *a, **k: model
    model.cuda = lambda *a, **k: model


def is_streamed_pack_file(pack_file: str, prefix: str) -> bool:
    import os
    return os.path.basename(pack_file).startswith(prefix)


def _is_packed(mod: nn.Module) -> bool:
    return type(mod).__name__.startswith("QVBand")



_PIN_FAIL_SAID = False


def pin_exact(t: "torch.Tensor") -> "torch.Tensor":
    """A pinned host copy of `t` that costs exactly its bytes. torch's pin_memory() goes through the caching host allocator:
    every size is rounded up to a power of two (a 0.6 GiB block takes 1 GiB) and freed blocks are kept by the allocator for
    good (measured on the V100: 3.0 GiB of blocks -> 5.0 GiB RSS, still 5.0 after del). Here: ordinary host memory, page-
    locked with cudaHostRegister, unlocked when the returned tensor goes (a view that outlives it is then just pageable)."""
    import weakref
    if t.is_pinned() or not torch.cuda.is_available() or t.numel() == 0:
        return t
    base = torch.empty(t.numel() * t.element_size(), dtype=torch.uint8)
    out = base.view(t.dtype).view(t.shape)
    out.copy_(t.detach().to("cpu"))
    cr = torch.cuda.cudart()
    rc = int(cr.cudaHostRegister(base.data_ptr(), base.numel(), 0))
    if rc != 0:
        cr.cudaGetLastError()                     # clear it: else the NEXT torch CUDA call reports this error
        global _PIN_FAIL_SAID
        if not _PIN_FAIL_SAID:
            _PIN_FAIL_SAID = True
            print(f"[block_stream] cudaHostRegister failed (error {rc}, {base.numel()} bytes): kept pageable", flush=True)
        return out
    weakref.finalize(out, _unregister, base)      # `base` stays alive until it is unlocked
    return out


def _unregister(base):
    torch.cuda.cudart().cudaHostUnregister(base.data_ptr())

class BlockStreamer:
    def __init__(self, blocks, device, *, ahead: int = 2, cyclic: bool = False):
        self.blocks = list(blocks)
        self.dev = torch.device(device)
        self.ahead, self.cyclic = int(ahead), bool(cyclic)
        self.copy = torch.cuda.Stream(self.dev)
        self.host: list[list] = []                # per block: (module, buffer name, pinned host tensor)
        self.gpu: dict[int, list] = {}            # resident blocks: (module, name, device tensor)
        self.ready: dict[int, torch.cuda.Event] = {}
        self.host_bytes = 0
        self.disk: list[list] = []                # per block: packed modules built on_disk (weights re-read from their file)
        self.disk_bytes = 0
        self.pending: dict = {}                   # block -> Future of a disk rebuild (runs on a worker thread)
        self._pool = None
        self._hooks = []
        # disk blocks, two stages (SSD -> pinned host -> card), as the layer-offload pipelines do (PIPO, FlexGen): READERS
        # threads read + rebuild up to READ_AHEAD blocks ahead into pinned host RAM, while the card window stays `ahead`
        # blocks (no extra VRAM). Defaults 1 / 0 = the one-stage path below (rebuild straight onto the card).
        self.readers = int(os.environ.get("QV_STREAM_READERS", "1"))
        self.read_ahead = int(os.environ.get("QV_STREAM_READ_AHEAD", "0"))
        self.staged: dict = {}                    # block -> Future of its pinned host entries (read + rebuilt)
        self._readers = None
        for i, blk in enumerate(self.blocks):
            entries, on_disk = [], []
            for mod in blk.modules():
                if _is_packed(mod) and getattr(mod, "on_disk", False):
                    for name, p in list(mod._parameters.items()):          # the bias: small, resident
                        if p is not None:
                            mod._parameters[name] = nn.Parameter(p.data.to(self.dev), requires_grad=False)
                    on_disk.append(mod)
                elif _is_packed(mod):
                    for name, p in list(mod._parameters.items()):          # a packed linear's bias: small, resident
                        if p is not None:
                            mod._parameters[name] = nn.Parameter(p.data.to(self.dev), requires_grad=False)
                    for name, buf in list(mod._buffers.items()):
                        if buf is None:
                            continue
                        h = pin_exact(buf.detach().to("cpu").contiguous())   # (a dram:shared copy: pinned already)
                        mod._buffers[name] = h
                        entries.append((mod, name, h)); self.host_bytes += h.numel() * h.element_size()
                else:
                    for name, p in list(mod._parameters.items()):
                        if p is not None:
                            mod._parameters[name] = nn.Parameter(p.data.to(self.dev), requires_grad=False)
                    for name, buf in list(mod._buffers.items()):
                        if buf is not None:
                            mod._buffers[name] = buf.to(self.dev)
            self.host.append(entries); self.disk.append(on_disk)
            self._hooks.append(blk.register_forward_pre_hook(self._pre(i)))
            self._hooks.append(blk.register_forward_hook(self._post(i)))

    # ---- copy side
    def _nxt(self, i: int, k: int):
        j = i + k
        if j < len(self.blocks):
            return j
        return j % len(self.blocks) if self.cyclic else None

    def _queue(self, i):
        if i is None or i in self.gpu or i in self.pending:
            return
        if self.disk[i]:
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="disk_stream")
            if self.read_ahead > 0:
                self._stage(i)
                self.pending[i] = self._pool.submit(self._from_staged, i)
            else:
                self.pending[i] = self._pool.submit(self._from_disk, i)
            return
        with torch.cuda.stream(self.copy):
            entries = [(m, n, h.to(self.dev, non_blocking=True)) for m, n, h in self.host[i]]
            ev = torch.cuda.Event(); ev.record(self.copy)
        self.gpu[i], self.ready[i] = entries, ev

    def _stage(self, i):
        """Start reading block i into pinned host RAM (READERS threads), if it streams from disk and is not on its way."""
        if i is None or not self.disk[i] or i in self.staged or i in self.gpu or i in self.pending:
            return
        if self._readers is None:
            from concurrent.futures import ThreadPoolExecutor
            self._readers = ThreadPoolExecutor(max_workers=max(1, self.readers), thread_name_prefix="disk_read")
        self.staged[i] = self._readers.submit(self._read_host, i)

    def _read_host(self, i):
        """Reader thread: block i's packed buffers rebuilt from their files, pinned (the copy to the card is then a plain
        DMA). Returns the entries; the pinned copies go (unlocked, back to the OS) once they are on the card."""
        out = []
        for mod, bufs in self._rebuild_all(i):
            for n, b in bufs.items():
                if b is not None:
                    out.append((mod, n, pin_exact(b)))
        return out

    def _rebuild_all(self, i):
        """Block i's on-disk modules (one weight each) rebuilt from their files at once, one weight per thread (the common
        loader's worker count: QV_LOAD_WORKERS, capped by host RAM), in module order."""
        from qv_band.run.weight_load import workers
        mods = self.disk[i]
        n = min(len(mods), workers())
        if n <= 1:
            return [(m, m.rebuild_buffers()) for m in mods]
        if getattr(self, "_mod_pool", None) is None:
            from concurrent.futures import ThreadPoolExecutor
            self._mod_pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="disk_mod")
        dev = self.dev
        idx = None if dev.type != "cuda" else (dev.index if dev.index is not None else torch.cuda.current_device())

        def one(m):
            if idx is not None:
                torch.cuda.set_device(idx)
            return m, m.rebuild_buffers()
        return list(self._mod_pool.map(one, mods))

    def _from_staged(self, i):
        """Copy worker: wait for block i's pinned host copy, put it on the card on the copy stream."""
        host = self.staged.pop(i).result()
        with torch.cuda.device(self.dev), torch.cuda.stream(self.copy):
            entries = [(m, n, h.to(self.dev, non_blocking=True)) for m, n, h in self.host[i]]
            entries += [(m, n, h.to(self.dev, non_blocking=True)) for m, n, h in host]
            ev = torch.cuda.Event(); ev.record(self.copy)
            ev.synchronize()                     # the pinned sources go back to the allocator once this returns
        return entries, ev

    def _from_disk(self, i):
        """Worker thread: rebuild block i's packed buffers from their files and copy them to the card on the copy stream.
        Pageable host tensors: each copy is staged by the driver and the host copy is freed as soon as it returns, so
        host RAM holds about one module's buffers at a time (plus any pinned entries of the block)."""
        with torch.cuda.device(self.dev), torch.cuda.stream(self.copy):
            entries = [(m, n, h.to(self.dev, non_blocking=True)) for m, n, h in self.host[i]]
            for mod, bufs in self._rebuild_all(i):
                for n, b in bufs.items():
                    if b is not None:
                        entries.append((mod, n, b.to(self.dev, non_blocking=True)))
            ev = torch.cuda.Event(); ev.record(self.copy)
            ev.synchronize()                     # the pageable sources may be freed once this returns
        return entries, ev

    def _collect(self, i):
        f = self.pending.pop(i, None)
        if f is not None:
            self.gpu[i], self.ready[i] = f.result()

    # ---- compute side
    def _pre(self, i):
        def hook(mod, args):
            prof = _PROFILE and self._prof_begin(i)
            self._queue(i)
            self._collect(i)
            cur = torch.cuda.current_stream(self.dev)
            if prof:
                self.ready[i].synchronize(); self._prof_mark("wait")
            cur.wait_event(self.ready[i])
            for m, n, g in self.gpu[i]:
                g.record_stream(cur)                  # allocated on the copy stream, used (and freed) on this one
                m._buffers[n] = g
            for k in range(1, self.ahead + 1):
                self._queue(self._nxt(i, k))
            for k in range(self.ahead + 1, self.ahead + self.read_ahead + 1):
                self._stage(self._nxt(i, k))           # deeper: only into pinned host RAM
        return hook

    def _post(self, i):
        def hook(mod, args, out):
            if _PROFILE and getattr(self, "_pt", None) is not None:
                torch.cuda.synchronize(self.dev); self._prof_mark("compute")
                if i == len(self.blocks) - 1:
                    self._prof_end()
            for m, n, h in self.host[i]:
                m._buffers[n] = h                    # the block goes back to the host copy; the device tensors are released
                for c in _CACHES:
                    m.__dict__.pop(c, None)
            for m in self.disk[i]:
                for n in list(m._buffers):
                    m._buffers[n] = None             # back to the file
                for c in _CACHES:
                    m.__dict__.pop(c, None)
            self.gpu.pop(i, None); self.ready.pop(i, None)
            if self.cyclic and i == len(self.blocks) - 1:
                for k in range(0, self.ahead):         # the next pass starts at block 0: have it on its way
                    self._queue(k % len(self.blocks))
                for k in range(self.ahead, self.ahead + self.read_ahead):
                    self._stage(k % len(self.blocks))
        return hook

    # ---- QV_STREAM_PROFILE=1: per streamed block, the time the compute stream waited for the block's copy vs its compute
    # (both synchronized, so the run is slower while profiling); one line per pass over the streamed blocks
    def _prof_begin(self, i):
        import time
        torch.cuda.synchronize(self.dev)
        if i == 0 or not hasattr(self, "_pacc"):
            self._pacc = {"wait": 0.0, "compute": 0.0, "n": 0}
        self._pt = time.perf_counter(); self._pacc["n"] += 1
        return True

    def _prof_mark(self, what):
        import time
        t = time.perf_counter(); self._pacc[what] += t - self._pt; self._pt = t

    def _prof_end(self):
        a = self._pacc
        print(f"[block_stream] pass over {a['n']} streamed blocks on {self.dev}: waited for copies {a['wait']:.2f} s, "
              f"computed {a['compute']:.2f} s ({self.host_bytes / 1e9:.1f} GB pinned, {len(self.disk)} blocks)", flush=True)
        self._pt = None; self._pacc = {"wait": 0.0, "compute": 0.0, "n": 0}

    def resident_blocks(self) -> int:
        return len(self.gpu)

    def release(self):
        """Forget the pinned host copies (the caller is about to drop the model): hooks off, nothing referenced from here."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._drain()
        self.host, self.host_bytes, self.disk = [], 0, []
        self.gpu.clear(); self.ready.clear()

    def evict(self):
        """Drop the copies on the card (a cyclic streamer prefetches the next pass's first blocks when a pass ends: after the
        LAST pass of a request they wait there for nothing, through the VAE decode). Hooks and host copies stay: the next
        pass copies the blocks again."""
        self._drain()
        self.gpu.clear(); self.ready.clear()

    def _drain(self):
        for f in self.pending.values():
            try:
                f.result()
            except Exception:
                pass
        self.pending.clear()
        for f in self.staged.values():
            try:
                f.result()
            except Exception:
                pass
        self.staged.clear()
        if self._readers is not None:
            self._readers.shutdown(wait=True); self._readers = None
        if self._pool is not None:
            self._pool.shutdown(wait=True); self._pool = None
        if getattr(self, "_mod_pool", None) is not None:
            self._mod_pool.shutdown(wait=True); self._mod_pool = None

    def close(self):
        self._drain()
        for h in self._hooks:
            h.remove()
        self._hooks = []
        for i, entries in enumerate(self.host):
            for m, n, h in entries:
                m._buffers[n] = h
        self.gpu.clear(); self.ready.clear()
