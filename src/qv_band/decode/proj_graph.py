"""CUDA-graph replay of a layer's input projections as ONE group at decode / verify.

WHY: the V100 host (Xeon E5-2673 v4) is too slow per thread for the ~4,900 kernel launches of a 27B verify step; the
GPU idles while Python issues them (moe_graph.install_mlp_graphs fixed the MLPs). Each Gated-DeltaNet layer then calls
four projections on the SAME input -- in_proj_qkv, in_proj_z, in_proj_b, in_proj_a -- each a Python module call plus
its kernels. Here the first of them replays one graph computing all four; the other three return the stored results
when called with that same input tensor (identity check), so the stock forward is untouched.

The first output is returned as a clone (the conv cache may keep a slice of it past this step); the rest are the
graph's static outputs, consumed within the same layer call. Rows 2..MAX_ROWS only (verify / decode); anything else
(prefill chunks, a different input) runs eagerly. QV_PROJ_GRAPH=0 turns it off.
"""
from __future__ import annotations

import os

import torch

MAX_ROWS = int(os.environ.get("QV_PROJ_GRAPH_MAX_ROWS", "16"))

GROUPS = (("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),)   # Qwen3_5GatedDeltaNet


class _Lead(torch.nn.Module):
    """Replaces the group's first projection: runs (or replays) all of them."""

    def __init__(self, mods, pool):
        super().__init__()
        self.mods = torch.nn.ModuleList(mods)
        self._pool = pool
        self._g = {}
        self._seen = set()
        self.last_in = None          # the input the stash belongs to (identity)
        self.stash = None

    def _eager(self, x):
        outs = [m(x) for m in self.mods]
        self.last_in, self.stash = x, outs
        return outs[0]

    def forward(self, x):
        rows = x.numel() // x.shape[-1]
        if not x.is_cuda or torch.cuda.is_current_stream_capturing() or not (2 <= rows <= MAX_ROWS):
            self.last_in = None
            return self.mods[0](x)
        key = (tuple(x.shape), x.dtype, x.device)
        ent = self._g.get(key)
        if ent is None:
            if key not in self._seen or key in self._g:      # first sighting, or capture failed before
                self._seen.add(key)
                return self._eager(x)
            ent = self._capture(x, key)
            if ent is None:
                return self._eager(x)
        g, sin, souts = ent
        sin.copy_(x)
        g.replay()
        self.last_in, self.stash = x, souts
        return souts[0].clone()

    def _capture(self, x, key):
        sin = x.detach().clone()
        s = torch.cuda.Stream(device=x.device)
        s.wait_stream(torch.cuda.current_stream(x.device))
        try:
            with torch.cuda.stream(s):
                [m(sin) for m in self.mods]                  # warm on the capture stream
            torch.cuda.current_stream(x.device).wait_stream(s)
            g = torch.cuda.CUDAGraph()
            from qv_band.formats.qvb8.moe_graph import pool_for
            pool = pool_for(x.device) if self._pool == "per-device" else self._pool
            with torch.cuda.device(x.device), torch.cuda.graph(g, pool=pool, stream=s, capture_error_mode="thread_local"):   # see moe_graph
                souts = [m(sin) for m in self.mods]
        except Exception as e:                               # not capturable: stay eager for this shape
            print(f"[proj-graph] capture failed for {key[0]} ({type(e).__name__}: {e}); eager", flush=True)
            torch.cuda.synchronize(x.device)
            self._g[key] = None
            return None
        self._g[key] = (g, sin, souts)
        return self._g[key]


class _Follow(torch.nn.Module):
    """Replaces a later projection of the group: the lead's stored result for the same input, else its own call."""

    def __init__(self, lead: _Lead, idx: int, mod):
        super().__init__()
        self.__dict__["_lead"] = lead                        # not a submodule: no double registration
        self.idx = idx
        self.inner = mod

    def forward(self, x):
        lead = self._lead
        if lead.last_in is x and lead.stash is not None:
            return lead.stash[self.idx]
        return self.inner(x)


def install(model) -> int:
    """Group the input projections of every module that has all of a GROUPS entry. Returns how many groups."""
    if os.environ.get("QV_PROJ_GRAPH", "1") == "0" or not torch.cuda.is_available():
        return 0
    pool = "per-device"       # one pool per card (moe_graph.pool_for): layers may be split over two GPUs
    n = 0
    for mod in list(model.modules()):
        for names in GROUPS:
            if not all(isinstance(getattr(mod, nm, None), torch.nn.Module) for nm in names):
                continue
            if isinstance(getattr(mod, names[0]), _Lead):
                continue
            mods = [getattr(mod, nm) for nm in names]
            lead = _Lead(mods, pool)
            setattr(mod, names[0], lead)
            for i, nm in enumerate(names[1:], 1):
                setattr(mod, nm, _Follow(lead, i, mods[i]))
            n += 1
    return n
