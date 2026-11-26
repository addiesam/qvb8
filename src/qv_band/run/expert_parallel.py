"""EXPERT PARALLEL for every mixture-of-experts model: ONE implementation of the split, the routing mask and the combine.

Card r owns the WHOLE experts [e0, e1) of every MoE layer (nothing is cut inside an expert; ranges may be uneven by
weights for unequal cards). Per MoE layer every card:
  1. routes ALL tokens (router replicated, tiny): the same top-k on every card;
  2. keeps the pairs whose expert it owns (`localize`: foreign pairs get weight 0), runs ITS experts on them;
  3. adds its share of any shared / dense branch (a column / row split of that branch);
  4. ONE reduce (all-reduce) over the cards: the sum is routed experts + shared branch.
A model plugs in only what differs: its router, its local experts (qvb8 per-expert tables, a qvbq stack, host experts ...),
its shared branch and the reduce. Used by Qwen3.5/3.6 MoE (run/tp_moe.py, tp_graph) and Inkling (models/inkling).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qv_band.formats.qvb8.tp import split_weighted


def expert_range(n_experts: int, rank: int, world: int, weights=None) -> tuple[int, int]:
    """[e0, e1) of rank's experts: equal shares, or in proportion to `weights` (one per card, at least one expert each)."""
    return split_weighted(n_experts, tuple(weights) if weights else (1,) * world, rank)


def expert_ranges(n_experts: int, world: int, weights=None) -> list[tuple[int, int]]:
    return [expert_range(n_experts, r, world, weights) for r in range(world)]


def localize(top_idx: torch.Tensor, top_w: torch.Tensor, e0: int, n_local: int):
    """Routed (idx, weight) [T, k] -> (local idx, weight) with the weight of every pair this card does NOT own set to 0
    (and its index to 0, so a gather stays in range)."""
    loc = top_idx - e0
    own = (loc >= 0) & (loc < n_local)
    return torch.where(own, loc, torch.zeros_like(loc)), torch.where(own, top_w, torch.zeros_like(top_w))


def route_softmax_topk(logits: torch.Tensor, top_k: int):
    """Qwen3.5/3.6 MoE routing: softmax over experts, top-k, weights renormalised to sum 1."""
    probs = F.softmax(logits, dtype=torch.float, dim=-1)
    top, idx = torch.topk(probs, top_k, dim=-1)
    return (top / top.sum(dim=-1, keepdim=True)).to(logits.dtype), idx


class ExpertParallelMoE(nn.Module):
    """The generic EP block. router(x) -> logits; route(logits, k) -> (weights, idx); experts(x, loc_idx, weights) -> [T, H]
    (this card's experts only); shared(x) -> [T, H] partial or None; reduce(y) -> y summed over the cards."""

    def __init__(self, router, experts, e0: int, n_local: int, top_k: int, hidden: int, reduce, *,
                 route=route_softmax_topk, shared=None):
        super().__init__()
        self.router, self.experts, self.shared = router, experts, shared
        self.e0, self.n_local, self.top_k, self.hidden = e0, n_local, top_k, hidden
        self.reduce, self.route = reduce, route

    def local_experts(self, x, loc, w, w_all):
        """Hook: the card's routed experts (w = the owned pairs' weights; w_all kept for subclasses with exact side paths)."""
        return self.experts(x, loc, w) if self.n_local > 0 else torch.zeros_like(x)

    def forward(self, h):
        shp = h.shape
        x = h.reshape(-1, self.hidden)
        top, idx = self.route(self.router(x), self.top_k)
        loc, w = localize(idx, top, self.e0, self.n_local)
        y = self.local_experts(x, loc, w, w)
        if self.shared is not None:
            y = y + self.shared(x)
        return self.reduce(y).reshape(shp)
