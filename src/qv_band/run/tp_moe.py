"""TP for mixture-of-experts blocks (Qwen3.5/3.6 MoE, e.g. Qwen3.6-35B-A3B): EXPERT PARALLEL, one expert = one PM.

A QVB8 MoE pack stores every expert as its own tables (gate_up_proj#expertN, down_proj#expertN), so an expert is a
natural partial matrix: card r holds the whole experts of its range [e0, e1) (uneven ranges by --tp-weights),
nothing is cut inside an expert.

Per block, every card:
  1. routes all tokens (router replicated: 256 x hidden, tiny) -- same top-k on every card;
  2. runs ITS experts for the tokens routed to them (other experts' weights are 0 for it);
  3. runs its share of the shared expert (columns of gate/up, rows of down: as a dense MLP);
  4. adds sigmoid(shared_gate(x)) * shared_partial -- the gate is a per-token scalar, so it distributes over the
     partial sums -- and ALL-REDUCES ONCE: sum over cards = routed experts + gated shared expert.

One all-reduce per MoE block, the same count as a dense MLP.
"""
from __future__ import annotations

import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from qv_band.formats.qvb8.module import QVBandLinearV8
from qv_band.formats.qvb8r.to_qvb8 import load_linear_any
from qv_band.formats.qvb8.tp import ColumnParallelLinear, RowParallelLinear, split_weighted
from qv_band.run.expert_parallel import expert_range as _ep_range, localize, route_softmax_topk


class TPMoeBlock(nn.Module):
    def __init__(self, router, experts, e0, n_local, top_k, shared_gate_up, shared_down, shared_gate, tp,
                 hidden):
        super().__init__()
        self.router, self.experts = router, experts
        self.e0, self.n_local, self.top_k = e0, n_local, top_k
        self.sg, self.su = shared_gate_up                       # column shares of shared gate_proj, up_proj
        self.sd = shared_down                                    # row share (local part; no reduce here)
        self.shared_gate = shared_gate
        self.tp, self.hidden = tp, hidden
        # experts with escape entries (a handful per model) cannot go into the fused stack: they stay in it with
        # their escapes stripped and routing weight 0, and run exactly here (dense, graph-safe, no host sync)
        self.exact = nn.ModuleDict()
        if experts is not None:
            for i in range(n_local):
                gu, dn = experts.gate_up_experts[i], experts.down_experts[i]
                if gu.esc_idx.numel() or dn.esc_idx.numel():
                    self.exact[str(i)] = nn.ModuleList([gu, dn])
                    experts.gate_up_experts[i], experts.down_experts[i] = _no_esc(gu), _no_esc(dn)
        self.exact_ids = [int(k) for k in self.exact]

    def forward(self, h):
        shp = h.shape
        x = h.reshape(-1, self.hidden)
        top, idx = route_softmax_topk(self.router(x), self.top_k)       # run/expert_parallel.py: the common EP routing
        loc, w = localize(idx, top, self.e0, self.n_local)
        w_f = w
        if self.exact_ids:
            w_f = w.clone()
            for i in self.exact_ids:
                w_f = torch.where(loc == i, torch.zeros_like(w_f), w_f)
        y = self.experts(x, loc, w_f) if self.n_local > 0 else torch.zeros_like(x)
        for i in self.exact_ids:
            gu, dn = self.exact[str(i)]
            wi = (w * (loc == i).to(w.dtype)).sum(-1, keepdim=True)
            g, u = gu(x).chunk(2, dim=-1)
            y = y + (dn(F.silu(g) * u) * wi).to(y.dtype)
        if self.sd is not None:
            sh = self.sd(F.silu(self.sg(x)) * self.su(x))
            y = y + torch.sigmoid(self.shared_gate(x)) * sh
        return self.tp.all_reduce(y).reshape(shp)


def _no_esc(m):
    """A view of a QVB8 linear without its escape entries (shares the packed ids)."""
    import copy
    c = copy.copy(m)
    c._buffers = dict(m._buffers)
    c._parameters, c._modules = dict(m._parameters), dict(m._modules)
    c.esc_idx = m.esc_idx[:0].clone()
    c.esc_vals = m.esc_vals[:0].clone()
    return c


def _shard_path(path, rank):
    """An expert file of this card: from $QV_EP_SHARD_DIR/rank<R>/ when the pack was pre-split (scripts/tools/split_pack_ep.py),
    else from the pack folder."""
    d = os.environ.get("QV_EP_SHARD_DIR")
    if d:
        p = os.path.join(d, f"rank{rank}", os.path.basename(path))
        if os.path.exists(p):
            return p
    return path


def expert_range(num_experts, tp, weights=None):
    return _ep_range(num_experts, tp.rank, tp.world, weights)


def build_moe_blocks(model, pack, man, cfg, tp, device, weights=None, prefix="model.language_model.",
                     verbose=False):
    """Replace every layer's HF MoE block (on meta) by a TPMoeBlock with this card's experts and shared share."""
    from qv_band.formats.qvb8.backends_experts import QVBandExpertsV8
    E, H = cfg.num_experts, cfg.hidden_size
    e0, e1 = expert_range(E, tp, weights)
    w = tuple(weights) if weights else (1,) * tp.world
    s0, s1 = split_weighted(cfg.shared_expert_intermediate_size, w, tp.rank, unit=128)
    per_layer = {}
    pat = re.compile(re.escape(prefix) + r"layers\.(\d+)\.mlp\.(.+)")
    for key, ent in man.items():
        m = pat.match(key)
        if m:
            per_layer.setdefault(int(m.group(1)), {})[m.group(2)] = os.path.join(pack, ent["pack"])
    for li, files in sorted(per_layer.items()):
        def lin(name):
            return load_linear_any(files[name], device)
        router = lin("gate.weight").to(device)
        shared_gate = lin("shared_expert_gate.weight").to(device)
        sg = ColumnParallelLinear.from_linear(lin("shared_expert.gate_proj.weight"), tp, rows=[(s0, s1)],
                                              device=device)
        su = ColumnParallelLinear.from_linear(lin("shared_expert.up_proj.weight"), tp, rows=[(s0, s1)],
                                              device=device)
        sd = RowParallelLinear.from_linear(lin("shared_expert.down_proj.weight"), tp, cols=(s0, s1),
                                           device=device).local
        gu = {e - e0: _shard_path(files[f"experts.gate_up_proj#expert{e:03d}"], tp.rank) for e in range(e0, e1)}
        dn = {e - e0: _shard_path(files[f"experts.down_proj#expert{e:03d}"], tp.rank) for e in range(e0, e1)}
        experts = QVBandExpertsV8.from_packfiles(gu, dn, num_experts=e1 - e0, device=device) if e1 > e0 else None
        blk = TPMoeBlock(router, experts, e0, e1 - e0, cfg.num_experts_per_tok, (sg, su), sd, shared_gate, tp, H)
        model.model.layers[li].mlp = blk
        if verbose and li % 10 == 0:
            print(f"[tp{tp.rank}] MoE layer {li}: experts {e0}..{e1 - 1}, shared cols {s0}..{s1}, "
                  f"VRAM {torch.cuda.memory_allocated(device) / 1e9:.2f} GB", flush=True)
    return e0, e1
