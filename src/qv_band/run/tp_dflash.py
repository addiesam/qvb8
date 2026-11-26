"""Split a DFlash drafter over the TP cards by the same ratio as the target.

Each draft layer: q/k/v by KV head (its query heads follow), gate/up by 128-wide blocks of rows; o_proj and
down_proj by the matching input columns with one all-reduce each (2 per layer, 10 for the 5-layer 27B drafter).
Replicated: fc + hidden_norm (they make the context feature every layer reads), norms, the dynamic causal convs
(they act on the full hidden vector), the candidate selector. The context K/V buffers keep only this card's KV
heads. The hidden vector after each all-reduce is the same on every card, so every card drafts the same block.

Packed linears must be QVB8 (QVBandLinearV8: cut from the packed bits); dense tensors are sliced directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qv_band.formats.qvb8.module import QVBandLinearV8
from qv_band.formats.qvb8.tp import (ColumnParallelLinear, RowParallelLinear, TPGroup, split_weighted)


class _DenseRowParallel(nn.Module):
    """A dense weight's input columns; the partial products are summed over the cards."""

    def __init__(self, w: torch.Tensor, tp: TPGroup):
        super().__init__()
        self.w, self.tp = w, tp

    def forward(self, x):
        return self.tp.all_reduce(F.linear(x, self.w))


def _col(w, rows, tp):
    """Output rows (a list of (a, b) ranges) of a linear: packed -> its own module, dense -> a sliced tensor."""
    if isinstance(w, QVBandLinearV8):
        return ColumnParallelLinear.from_linear(w, tp, rows=rows, device=w.ids.device)
    if isinstance(w, torch.Tensor):
        return torch.cat([w[a:b] for a, b in rows], 0).contiguous()
    if type(w).__name__ == "QVBandLinearV8R":       # the common loader: cut in place from the rANS stream
        from qv_band.run.weight_load import load_weight
        dev = w.min_s.device
        return ColumnParallelLinear(load_weight(w.path, rows=rows, device=str(dev)), tp, w.out_features)
    raise TypeError(f"cannot split a {type(w).__name__} drafter linear (needs QVB8, QVB8R or dense)")


def _row(w, cols, tp):
    c0, c1 = cols
    if isinstance(w, QVBandLinearV8):
        return RowParallelLinear.from_linear(w, tp, cols=(c0, c1), device=w.ids.device)
    if isinstance(w, torch.Tensor):
        return _DenseRowParallel(w[:, c0:c1].contiguous(), tp)
    if type(w).__name__ == "QVBandLinearV8R":
        from qv_band.run.weight_load import load_weight
        dev = w.min_s.device
        bias = None if w.bias is None else w.bias.detach().to(dev)
        return RowParallelLinear(load_weight(w.path, cols=(c0, c1), device=str(dev)), tp, w.in_features, bias)
    raise TypeError(f"cannot split a {type(w).__name__} drafter linear (needs QVB8, QVB8R or dense)")


def split_drafter(d, tp: TPGroup, weights=None, intermediate_size: int | None = None):
    """Cut drafter `d` (DFlashDrafter) in place to this card's share; returns d."""
    if tp.world == 1:
        return d
    w = tuple(weights) if weights else (1,) * tp.world
    hd, nh, nkv = d.hd, d.n_heads, d.n_kv
    g = nh // nkv
    k0, k1 = split_weighted(nkv, w, tp.rank)
    q0, q1 = k0 * g, k1 * g
    inter = intermediate_size or _out_features(d.layers[0]["mlp.gate_proj"])
    m0, m1 = split_weighted(inter, w, tp.rank, unit=128)
    for L in d.layers:
        L["self_attn.q_proj"] = _col(L["self_attn.q_proj"], [(q0 * hd, q1 * hd)], tp)
        L["self_attn.k_proj"] = _col(L["self_attn.k_proj"], [(k0 * hd, k1 * hd)], tp)
        L["self_attn.v_proj"] = _col(L["self_attn.v_proj"], [(k0 * hd, k1 * hd)], tp)
        L["self_attn.o_proj"] = _row(L["self_attn.o_proj"], (q0 * hd, q1 * hd), tp)
        L["mlp.gate_proj"] = _col(L["mlp.gate_proj"], [(m0, m1)], tp)
        L["mlp.up_proj"] = _col(L["mlp.up_proj"], [(m0, m1)], tp)
        L["mlp.down_proj"] = _row(L["mlp.down_proj"], (m0, m1), tp)
    d.n_heads, d.n_kv = q1 - q0, k1 - k0
    d._reset_state()                                   # context K/V buffers with this card's KV heads only
    torch.cuda.empty_cache()
    return d


def _out_features(w):
    return w.out_features if hasattr(w, "out_features") else w.shape[0]
