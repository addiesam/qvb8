"""Same-input fusion: sibling qvb8r linears that read the same activation (q/k/v, gate/up, qkv/z) run as ONE launch.

Their rANS data is concatenated into one virtual tensor (row / class / word offsets shifted), so every kernel path
runs unchanged on it; the output is split back per member. Measured on the 27B: 0.75-0.96x the separate launches at
M <= 16, bit-identical there (each row's sum is computed the same way).

The loader calls fuse_same_input(model) once after swapping (the format's `finalize` hook in the registry)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import container as QF
from .module import QVBandLinearV8R

# (parent attribute, sibling names) whose members are fed the same tensor by the model's forward
SAME_INPUT_GROUPS = (
    ("self_attn", ("q_proj", "k_proj", "v_proj")),
    ("linear_attn", ("in_proj_qkv", "in_proj_z")),
    ("mlp", ("gate_proj", "up_proj")),
)


def concat(ds: list[dict]) -> tuple[dict, list[int]]:
    """Concatenate qvb8r tensors with the same P into one (rows of ds[0] first). Returns (dict, row splits)."""
    P = int(ds[0]["P"])
    if any(int(d["P"]) != P for d in ds):
        raise ValueError("qvb8r concat: members must share P")
    N = max(int(d["N"]) for d in ds)
    out = dict(P=P, N=N, T=max(int(d["T"]) for d in ds), Q=sum(int(d["Q"]) for d in ds),
               K=sum(int(d["K"]) for d in ds))
    perm, freq, tiles, toff, states, words, drow, mn, ts, ei, ev, ed = ([] for _ in range(12))
    rq = rk = rw = 0
    for d in ds:
        perm.append(d["perm"].astype(np.int64) + rq)
        f = np.zeros((int(d["K"]), N), np.uint16); f[:, :d["freq"].shape[1]] = d["freq"]; freq.append(f)
        t = d["tiles"].astype(np.int64).copy(); t[:, 0] += rk; t[:, 1] += rq; tiles.append(t)
        toff.append(d["tile_off"][:-1].astype(np.int64) + rw); states.append(d["states"]); words.append(d["words"])
        drow.append(d["drow"])
        mn.append(d["min_s"]); ts.append(d["tau_s"])
        ei.append(d["esc_idx"].astype(np.int64) + rq * P); ev.append(d["esc_val"]); ed.append(d["esc_delta"])
        rq += int(d["Q"]); rk += int(d["K"]); rw += len(d["words"])
    out.update(perm=np.concatenate(perm).astype(np.int32), freq=np.concatenate(freq),
               tiles=np.concatenate(tiles).astype(np.int32),
               tile_off=np.concatenate(toff + [np.array([rw])]).astype(np.int64), states=np.concatenate(states),
               words=np.concatenate(words), drow=np.concatenate(drow), min_s=np.concatenate(mn), tau_s=np.concatenate(ts),
               esc_idx=np.concatenate(ei), esc_val=np.concatenate(ev), esc_delta=np.concatenate(ed))
    return out, np.cumsum([int(d["Q"]) for d in ds])[:-1].tolist()


class FusedGroup(nn.Module):
    """One launch for linears that read the same input: the first member called computes every member's output."""

    def __init__(self, ds: list[dict], device):
        super().__init__()
        d, self.splits = concat(ds)
        self.lin = QVBandLinearV8R(d, device=device)
        self._x = None; self._ver = None; self._outs = None; self._left = set()

    def get(self, x, i):
        # Reuse only for the SAME tensor object (held, so its storage cannot be recycled under us) and only until
        # every member has taken its part once; anything else recomputes. An address/version key can match a new
        # tensor allocated where the old one was and hand out a stale result.
        # _version also catches the same tensor modified in place between two members' calls.
        # (Python-side reuse: not valid under CUDA-graph replay -- no qvb8r path captures graphs.)
        if self._x is not x or x._version != self._ver or i not in self._left:
            self._outs = torch.tensor_split(self.lin(x), self.splits, -1)
            self._x = x; self._ver = x._version; self._left = set(range(len(self._outs)))
        self._left.discard(i); out = self._outs[i]
        if not self._left:
            self._x = self._outs = None
        return out


class FusedMember(nn.Module):
    def __init__(self, group: FusedGroup, i: int, in_features: int, out_features: int, bias):
        super().__init__()
        # the group is shared by the members: keep it out of the module tree so it is not registered N times
        object.__setattr__(self, "group", group)
        self.i, self.in_features, self.out_features = i, in_features, out_features
        self.bias = bias

    def forward(self, x):
        y = self.group.get(x, self.i)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, fused member {self.i}"


def fuse_same_input(model: nn.Module) -> int:
    """Replace every same-input group of qvb8r linears by members of one fused launch. Returns groups fused."""
    n = 0
    # Collect the owners first and hold NO module references across groups: a list of every module (or a loop
    # variable) would keep each replaced linear alive until the end, so the separate AND the fused copies of the
    # whole model would coexist (measured 13.4 -> 22 GB at load on the 27B: an OOM on a 16 GB card).
    owners = [(owner, names) for parent in model.modules() for sub, names in SAME_INPUT_GROUPS
              if (owner := getattr(parent, sub, None)) is not None
              and all(isinstance(getattr(owner, a, None), QVBandLinearV8R) and getattr(owner, a).path for a in names)]
    for owner, names in owners:
        first = getattr(owner, names[0])
        dev = first.words.device
        ds = [QF.read(getattr(owner, a).path) for a in names]          # host bytes
        meta = [(getattr(owner, a).in_features, getattr(owner, a).out_features, getattr(owner, a).bias) for a in names]
        del first
        for a in names:                                    # drop the separate modules BEFORE the fused one exists
            setattr(owner, a, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        g = FusedGroup(ds, dev)
        del ds
        owner._qvb8r_fused = g                             # registered once, under the shared owner
        for i, (a, (pin, pout, bias)) in enumerate(zip(names, meta)):
            setattr(owner, a, FusedMember(g, i, pin, pout, bias))
        n += 1
    if n and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n
