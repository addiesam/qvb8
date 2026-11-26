"""Tensor parallel (TP) for diffusion transformers (DiT) -- several cards of ONE machine, inside ONE process.

Why in one process: an image / video server is one process (text encoder, scheduler, VAE, job queue). Only the
transformer is worth splitting, so only the transformer is: every card gets its own copy of it in which the big
linears hold that card's share (Megatron layout) and everything small (norms, modulation, embedders) is whole.

    attention   to_q / to_k / to_v (+ add_q/k/v_proj)  column split by heads     -> attn.heads = heads / cards
                to_out.0 (+ to_add_out)                  row split, then all-reduce
    feed-forward  w1, w3  |  net.0.proj (GELU, GEGLU)   column split
                  w2      |  net.2                       row split, then all-reduce

A forward call runs the whole transformer forward on every card at once (one Python thread per card, the same
inputs); the row-split linears meet in an all-reduce (a thread barrier and a card-to-card copy). Every card sums the
partial results in the same order, so all cards hold bit-identical activations; card 0's output is returned. The
attention code, RoPE, masks, fused kernels: unchanged -- they see fewer heads, that is all. Norms over ALL heads
(Wan's qk-norm "across heads") are replaced by a split version that all-reduces the sum of squares.

The result is the same computation up to floating-point summation order (each output-projection sum is split into
per-card partial sums); it is not bit-identical to the one-card forward.

    from qv_band.run.tp_dit import enable_tp
    enable_tp(pipe.transformer, ["cuda:0", "cuda:1"])        # QVB8 / QVB8R / dense nn.Linear
"""
from __future__ import annotations

import copy
import os
import threading
import time
import types

import torch
from torch import nn

from qv_band.formats.qvb8.tp import ColumnParallelLinear, RowParallelLinear, split_even, split_weighted

_PEER = {}


def peer_ok(a, b) -> bool:
    """Card-to-card copies a -> b give the right bytes (they do not on every board: some PCIe switches corrupt peer
    copies). Checked once per pair; QV_TP_DIT_VIA_HOST=1 / 0 forces the answer."""
    a, b = torch.device(a), torch.device(b)
    if a == b or a.type != "cuda" or b.type != "cuda":
        return True
    force = os.environ.get("QV_TP_DIT_VIA_HOST", "auto")
    if force in ("0", "1"):
        return force == "0"
    k = (a.index, b.index)
    if k not in _PEER:
        x = torch.randn(1 << 22, device=a)
        ok = torch.equal(x.cpu(), x.to(b).cpu())
        x2 = torch.randn(1 << 22, device=b)
        ok = ok and torch.equal(x2.cpu(), x2.to(a).cpu())
        _PEER[k] = _PEER[(b.index, a.index)] = ok
        if not ok:
            print(f"[tp-dit] card-to-card copies {a} <-> {b} are corrupt on this board: copies go through host memory",
                  flush=True)
    return _PEER[k]


def xcopy(t: torch.Tensor, dev) -> torch.Tensor:
    """t on card `dev`, correct on every board (through host memory where card-to-card copies are corrupt)."""
    dev = torch.device("cpu" if dev == "ref" else dev)        # "ref": a share still in its file keeps its bias on the host
    if t.device == dev:
        return t
    if t.device.type == "cuda" and dev.type == "cuda" and not peer_ok(t.device, dev):
        return t.cpu().to(dev)
    return t.to(dev)


COL_ATTN = ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")
ROW_ATTN = (("to_out", 0), ("to_add_out", None))


class Hub:
    """The cards of one transformer and their all-reduce (threads; one thread per card)."""

    def __init__(self, devices):
        self.devs = [torch.device(d) for d in devices]
        self.world = len(self.devs)
        self.bar = threading.Barrier(self.world)
        self.slot = [None] * self.world
        self.ev = [None] * self.world
        self.via_host = any(not peer_ok(a, b) for a in self.devs for b in self.devs)
        self.pinned = [{} for _ in self.devs]             # per card: pinned host staging buffers by (shape, dtype)
        self.n_reduce = 0
        self.prof = os.environ.get("QV_TP_PROF") == "1"      # QV_TP_PROF=1: seconds in all-reduce (total / waiting at the barrier) per card
        self.t_all = [0.0] * self.world
        self.t_wait = [0.0] * self.world

    def _stage(self, r, y):
        k = (tuple(y.shape), y.dtype)
        b = self.pinned[r].get(k)
        if b is None:
            b = self.pinned[r][k] = torch.empty(y.shape, dtype=y.dtype, pin_memory=True)
        return b

    def all_reduce(self, y: torch.Tensor, r: int) -> torch.Tensor:
        dev = y.device
        st = torch.cuda.current_stream(dev)
        if self.prof:
            t0 = time.perf_counter()
            st.synchronize()                              # the card's own work up to here is NOT the reduce: separate it
            t1 = time.perf_counter()
            self.t_comp = getattr(self, "t_comp", [0.0] * self.world)
            self.t_comp[r] += t1 - t0
        if self.via_host:                                 # my partial -> pinned host memory, the others' from there
            hb = self._stage(r, y)
            hb.copy_(y, non_blocking=True)
            st.synchronize()
            self.slot[r] = hb
        else:
            self.slot[r] = y
            e = torch.cuda.Event()
            e.record(st)
            self.ev[r] = e
        if self.prof:
            tw = time.perf_counter()
        self.bar.wait()
        if self.prof:
            self.t_wait[r] += time.perf_counter() - tw
        acc = None
        for j in range(self.world):                       # the same order on every card: identical sums
            t = y if j == r else self.slot[j]
            if j != r:
                if self.via_host:
                    t = t.to(dev, non_blocking=True)
                else:
                    self.ev[j].synchronize()
                    t = t.to(dev) if t.device != dev else t
            acc = t.float() if acc is None else acc + t.float()
        st.synchronize()                                  # the others' tensors are read: they may go now
        self.bar.wait()
        if r == 0:
            self.n_reduce += 1
        if self.prof:
            self.t_all[r] += time.perf_counter() - t1
        return acc.to(y.dtype)


def _hub_gather(self, y, r):
    """Concatenate every card's y along the last dim, in card order (exact: no arithmetic)."""
    dev = y.device
    st = torch.cuda.current_stream(dev)
    if self.via_host:
        hb = self._stage(r, y)
        hb.copy_(y, non_blocking=True)
        st.synchronize()
        self.slot[r] = hb
    else:
        self.slot[r] = y
        e = torch.cuda.Event()
        e.record(st)
        self.ev[r] = e
    self.bar.wait()
    parts = []
    for j in range(self.world):
        t = y if j == r else self.slot[j]
        if j != r:
            if self.via_host:
                t = t.to(dev, non_blocking=True)
            else:
                self.ev[j].synchronize()
                t = t.to(dev) if t.device != dev else t
        parts.append(t)
    out = torch.cat(parts, -1)
    st.synchronize()
    self.bar.wait()
    return out


Hub.all_gather = _hub_gather


class ThreadTP:
    """The TPGroup interface (rank, world, all_reduce) for one card of a Hub."""

    def __init__(self, hub: Hub, rank: int, weights=None):
        self.hub, self.rank, self.world, self.group = hub, rank, hub.world, None
        self.weights = weights                         # share of every split per card (None: even)

    def split(self, n, unit=1):
        if self.weights is None:
            return split_even(n, self.world, self.rank, unit)
        return split_weighted(n, self.weights, self.rank, unit)

    graphable = False

    def all_reduce(self, y):
        return self.hub.all_reduce(y, self.rank)

    def all_gather_last(self, y, full=None):
        return self.hub.all_gather(y, self.rank)


class DenseCol(nn.Module):
    """A dense nn.Linear's output rows (ranges) on one card."""

    def __init__(self, lin: nn.Linear, ranges, device):
        super().__init__()
        idx = torch.cat([torch.arange(a, b) for a, b in ranges])
        self.weight = nn.Parameter(xcopy(lin.weight.detach()[idx.to(lin.weight.device)], device), requires_grad=False)
        self.bias = None if lin.bias is None else nn.Parameter(
            xcopy(lin.bias.detach()[idx.to(lin.bias.device)], device), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, int(idx.numel())

    def forward(self, x):
        return nn.functional.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class DenseRow(nn.Module):
    """A dense nn.Linear's input columns [c0, c1) on one card; all-reduce, then the bias."""

    def __init__(self, lin: nn.Linear, c0, c1, tp, device):
        super().__init__()
        self.tp = tp
        self.weight = nn.Parameter(xcopy(lin.weight.detach()[:, c0:c1].contiguous(), device), requires_grad=False)
        self.bias = None if lin.bias is None else nn.Parameter(xcopy(lin.bias.detach(), device), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        y = self.tp.all_reduce(nn.functional.linear(x, self.weight.to(x.dtype)))
        return y if self.bias is None else y + self.bias.to(y.dtype)


class SplitRMSNorm(nn.Module):
    """RMSNorm over a dimension split across the cards (qk-norm across all heads): the sum of squares is all-reduced."""

    def __init__(self, norm, c0, c1, full, tp, device):
        super().__init__()
        self.tp, self.full = tp, full
        self.eps = getattr(norm, "eps", None) or getattr(norm, "variance_epsilon", 1e-6)
        w = getattr(norm, "weight", None)
        self.weight = None if w is None else nn.Parameter(xcopy(w.detach()[c0:c1].contiguous(), device), requires_grad=False)

    def forward(self, x):
        xf = x.float()
        ss = self.tp.all_reduce(xf.pow(2).sum(-1, keepdim=True))
        y = xf * torch.rsqrt(ss / self.full + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.to(x.dtype)


def _is_qv(lin) -> bool:
    return hasattr(lin, "ids") and hasattr(lin, "nbits")


def _is_qvr(lin) -> bool:
    return type(lin).__name__ == "QVBandLinearV8R"


def _on_card(lin, device):
    """A QVB8R linear held in host RAM: a temporary copy on `device` (its slicer decodes on a card)."""
    if getattr(lin, "path", None) or lin.min_s.device.type == "cuda":    # a file: cut.py reads the share from it
        return lin
    return copy.deepcopy(lin).to(device)


def _col(lin, ranges, tp, device, gather=False):
    if _is_qvr(lin):
        from qv_band.formats.qvb8r.tp_slice import slice_qvb8r
        local, why = slice_qvb8r(_on_card(lin, device), rows=list(ranges), device=device)
        if local is None:
            raise RuntimeError(f"tp_dit: cannot split a QVB8R linear by rows: {why}")
        return ColumnParallelLinear(local, tp, lin.out_features, gather=gather)
    if _is_qv(lin):
        from qv_band.formats.qvb8.tp import slice_rows
        return ColumnParallelLinear(slice_rows(lin, ranges, device), tp, lin.out_features, gather=gather)
    if isinstance(lin, nn.Linear):
        d = DenseCol(lin, ranges, device)
        return ColumnParallelLinear(d, tp, lin.out_features, gather=True) if gather else d
    raise TypeError(f"tp_dit: cannot split {type(lin).__name__}")


def _row(lin, c0, c1, tp, device):
    if _is_qvr(lin):
        from qv_band.formats.qvb8r.tp_slice import slice_qvb8r
        local, why = slice_qvb8r(_on_card(lin, device), cols=(c0, c1), device=device)
        if local is None:
            raise RuntimeError(f"tp_dit: cannot split a QVB8R linear by columns: {why}")
        bias = None if lin.bias is None else xcopy(lin.bias.detach(), device)
        local.bias = None
        return RowParallelLinear(local, tp, lin.in_features, bias)
    if _is_qv(lin):
        from qv_band.formats.qvb8.tp import slice_cols
        bias = None if lin.bias is None else xcopy(lin.bias.detach(), device)
        if (c0, c1) == (0, lin.in_features):
            raise ValueError("tp_dit: a row split needs two or more cards")
        return RowParallelLinear(slice_cols(lin, c0, c1, device), tp, lin.in_features, bias)
    if isinstance(lin, nn.Linear):
        return DenseRow(lin, c0, c1, tp, device)
    raise TypeError(f"tp_dit: cannot split {type(lin).__name__}")


def _plan(tr, world):
    """[(parent module, attribute or index, kind, args)] for every linear to split, and the attention modules."""
    todo, attns = [], []
    for name, m in tr.named_modules():
        heads = getattr(m, "heads", None)
        if isinstance(heads, int) and hasattr(m, "to_q") and hasattr(m, "to_out"):
            hd = m.to_q.out_features // heads
            attns.append((name, m, heads, hd))
            for a in COL_ATTN:
                lin = getattr(m, a, None)
                if lin is not None:
                    if lin.out_features % hd:
                        raise ValueError(f"tp_dit: {name}.{a}: {lin.out_features} rows are not whole heads of {hd}")
                    todo.append((m, a, "col_heads", hd))
            for a, i in ROW_ATTN:
                lin = getattr(m, a, None)
                if lin is not None and i is not None:
                    lin = lin[i]
                if lin is not None:
                    todo.append((m, (a, i), "row_heads", hd))
            for a in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
                nm = getattr(m, a, None)
                w = getattr(nm, "weight", None)
                if w is not None and w.numel() == m.to_q.out_features and w.numel() != hd:
                    todo.append((m, a, "norm_across", hd))
            continue
        cls = type(m).__name__
        if cls == "FeedForward" and hasattr(m, "w1") and hasattr(m, "w2"):          # SwiGLU (Z-Image)
            todo += [(m, "w1", "col", None), (m, "w3", "col", None), (m, "w2", "row", None)]
        elif cls == "FeedForward" and hasattr(m, "net"):                            # diffusers FeedForward
            act = m.net[0]
            if type(act).__name__ in ("GEGLU", "SwiGLU"):
                todo.append((act, "proj", "col_geglu", None))
            elif hasattr(act, "proj"):
                todo.append((act, "proj", "col", None))
            else:
                continue
            last = [i for i, x in enumerate(m.net) if hasattr(x, "in_features")][-1]
            todo.append((m.net, last, "row", None))
    # every other BIG linear (Qwen-Image's per-block modulation: 3072 -> 18432): its output rows split, then gathered
    # (exact -- no sum is split), so its weights are not held whole on every card
    big = int(os.environ.get("QV_TP_DIT_GATHER_MIN", str(16 << 20)))
    planned = {id(_get(p, k)) for p, k, _, _ in todo}
    names = dict(tr.named_modules())
    for name, m in names.items():
        if id(m) in planned or "." not in name:
            continue
        if not (_is_qv(m) or _is_qvr(m) or type(m) is nn.Linear):
            continue
        if m.in_features * m.out_features < big or m.out_features % (128 * world):
            continue
        pname, _, leaf = name.rpartition(".")
        parent = names[pname]
        key = int(leaf) if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else leaf
        todo.append((parent, key, "col_gather", None))
    return todo, attns


def _slice_plain(lin, device, rows=None, cols=None):
    """A plain share of a (small) linear -- LoRA A / B: QVB8, QVB8R or dense; no collective attached."""
    if _is_qvr(lin):
        from qv_band.formats.qvb8r.tp_slice import slice_qvb8r
        out, why = slice_qvb8r(_on_card(lin, device), rows=rows, cols=cols, device=device)
        if out is None:
            raise RuntimeError(f"tp_dit: LoRA share: {why}")
        return out
    if _is_qv(lin):
        from qv_band.formats.qvb8.tp import slice_cols, slice_rows
        return slice_rows(lin, rows, device) if rows is not None else slice_cols(lin, cols[0], cols[1], device)
    w = getattr(lin, "weight", None)
    if w is None:
        raise TypeError(f"tp_dit: cannot split LoRA module {type(lin).__name__}")
    if rows is not None:
        idx = torch.cat([torch.arange(a, b) for a, b in rows]).to(w.device)
        w = w.detach()[idx]
    else:
        w = w.detach()[:, cols[0]:cols[1]].contiguous()
    m = copy.copy(lin)
    m._parameters = dict(lin._parameters)
    m._parameters["weight"] = nn.Parameter(xcopy(w, device), requires_grad=False)
    if "bias" in m._parameters and m._parameters["bias"] is not None:
        m._parameters["bias"] = None
    m._buffers, m._modules = dict(lin._buffers), dict(lin._modules)
    for k in ("in_features", "out_features"):
        if hasattr(m, k):
            setattr(m, k, w.shape[1] if k == "in_features" else w.shape[0])
    return m


def _carry_lora(src, share, kind, ranges_or_cols, device):
    """LoRA adapters (lora_addon: src.lora_adapters + a forward hook) onto the share's local linear. Column share: B's
    rows; row share: A's columns (each card's delta is then a partial sum, completed by the all-reduce)."""
    ads = getattr(src, "lora_adapters", None)
    if not ads:
        return
    local = getattr(share, "local", share)
    new = nn.ModuleDict()
    for name, ad in ads.items():
        a = nn.Module()
        a.scale = ad.scale
        if kind == "col":
            a.A = replicate(ad.A, device)
            a.B = _slice_plain(ad.B, device, rows=ranges_or_cols)
        else:
            a.A = _slice_plain(ad.A, device, cols=ranges_or_cols)
            a.B = replicate(ad.B, device)
        new[name] = a
    local.lora_adapters = new
    for h in src._forward_hooks.values():
        local.register_forward_hook(h)


def _get(parent, key):
    if isinstance(key, tuple):
        a, i = key
        x = getattr(parent, a)
        return x if i is None else x[i]
    if isinstance(key, int):
        return parent[key]
    return getattr(parent, key)


def _set(parent, key, v):
    if isinstance(key, tuple):
        a, i = key
        if i is None:
            setattr(parent, a, v)
        else:
            getattr(parent, a)[i] = v
    elif isinstance(key, int):
        parent[key] = v
    else:
        setattr(parent, key, v)


def _shard(lin_or_norm, kind, hd, tp, device, unit=128):
    sh, how = _shard0(lin_or_norm, kind, hd, tp, device, unit)
    if how is not None:                      # LoRA add-ons are small: a share kept off the card holds them in host RAM
        _carry_lora(lin_or_norm, sh, *how, "cpu" if device == "ref" else device)
    return sh


def _shard0(lin_or_norm, kind, hd, tp, device, unit=128):
    """-> (share, (\"col\", row ranges) | (\"row\", (c0, c1)) | None)"""
    r, w = tp.rank, tp.world
    if kind == "col_gather":
        rg = [tp.split(lin_or_norm.out_features, unit)]
        return _col(lin_or_norm, rg, tp, device, gather=True), ("col", rg)
    if kind in ("col_heads", "col", "col_geglu"):
        if kind == "col_geglu":                                 # [value | gate] halves: the same share of each
            h = lin_or_norm.out_features // 2
            a, b = tp.split(h, unit)
            rg = [(a, b), (h + a, h + b)]
        else:
            rg = [tp.split(lin_or_norm.out_features, hd if kind == "col_heads" else unit)]
        return _col(lin_or_norm, rg, tp, device), ("col", rg)
    if kind in ("row_heads", "row"):
        c0, c1 = tp.split(lin_or_norm.in_features, hd if kind == "row_heads" else unit)
        return _row(lin_or_norm, c0, c1, tp, device), ("row", (c0, c1))
    if kind == "norm_across":
        full = lin_or_norm.weight.numel()
        c0, c1 = tp.split(full, hd)
        return SplitRMSNorm(lin_or_norm, c0, c1, full, tp, device), None
    raise ValueError(kind)


def _copy_flags(src, dst):
    """Per-linear switches set after loading (bf16 GEMM, ...) carried over to a share of that linear."""
    inner = getattr(dst, "local", dst)
    for k in ("bf16_gemm",):
        if hasattr(src, k):
            setattr(inner, k, getattr(src, k))


def _move_loose(rep, dev):
    """Tensors a module keeps as plain attributes (caches, RoPE tables; also inside helper objects) -- .to() does
    not move those."""
    seen = set()

    def mv(obj, depth=0):
        if depth > 2 or id(obj) in seen:
            return
        seen.add(id(obj))
        d = getattr(obj, "__dict__", None)
        if d is None:
            return
        for k, v in list(d.items()):
            if k in ("_parameters", "_buffers", "_modules", "tp", "hub"):
                continue
            if torch.is_tensor(v) and v.device != torch.device(dev) and v.numel() > 1:
                d[k] = xcopy(v, dev)
            elif isinstance(v, (list, tuple)) and v and all(torch.is_tensor(t) for t in v):
                d[k] = type(v)(xcopy(t, dev) for t in v)
            elif isinstance(v, dict) and v and all(torch.is_tensor(t) for t in v.values()):
                d[k] = {kk: xcopy(t, dev) for kk, t in v.items()}
            elif not isinstance(v, (nn.Module, ThreadTP, Hub, type, types.FunctionType, types.MethodType,
                                    types.BuiltinFunctionType)) and hasattr(v, "__dict__"):
                mv(v, depth + 1)

    for m in rep.modules():
        mv(m)


def _tree_to(x, dev):
    if torch.is_tensor(x):
        return xcopy(x, dev)
    if isinstance(x, (list, tuple)):
        t = [_tree_to(v, dev) for v in x]
        return type(x)(t) if not hasattr(x, "_fields") else type(x)(*t)
    if isinstance(x, dict):
        return {k: _tree_to(v, dev) for k, v in x.items()}
    return x


def replicate(tr, device, memo=None):
    """A copy of module `tr` on `device`, built without a second copy on tr's card: every parameter / buffer is moved
    first (deepcopy then reuses the moved tensors); `memo` may already map modules to ready-made replacements."""
    memo = {} if memo is None else memo
    skip = set()
    for m in memo.values():
        if isinstance(m, nn.Module):
            skip.update(id(t) for t in m.parameters())
            skip.update(id(t) for t in m.buffers())
    covered = set()                 # every module INSIDE a ready-made replacement (its card-0 share's children too)
    for m in tr.modules():
        if id(m) in memo:
            covered.update(id(c) for c in m.modules())
    for m in tr.modules():
        if id(m) in covered:
            continue
        for k, p in m._parameters.items():
            if p is not None and id(p) not in memo:
                memo[id(p)] = nn.Parameter(xcopy(p.detach(), device), requires_grad=p.requires_grad)
        for k, b in m._buffers.items():
            if b is not None and id(b) not in memo:
                memo[id(b)] = xcopy(b, device)
    saved = []
    if "forward" in tr.__dict__:    # the top module's instance forward (TP / counters / CFG wrappers): never copied
        saved.append((tr, "forward", tr.__dict__.pop("forward")))
    for m in tr.modules():          # instance-level wrappers and the parallel bookkeeping are not copied
        f = m.__dict__.get("forward")
        if isinstance(f, (TPDiT, CFGParallel)) or getattr(f, "__self__", None).__class__.__name__ in (
                "CFGBatch", "PairBatch", "S2VPairBatch", "CFGParallel"):
            saved.append((m, "forward", m.__dict__.pop("forward")))
        for k in ("_tp_dit", "_cfg_par_rep", "_cfg_par_dev", "_cfg_batch"):
            if k in m.__dict__:
                saved.append((m, k, m.__dict__.pop(k)))
    try:
        rep = copy.deepcopy(tr, memo)
    finally:
        for m, k, v in saved:
            m.__dict__[k] = v
    _rebind_closures(rep, memo)
    _move_loose(rep, device)
    return rep


def _copy_cell(c, memo):
    """A closure cell's value for the copy: modules / tensors map to the copies; what cannot be copied (locks, counters,
    threads -- bookkeeping shared by every card's forward) is shared as it is."""
    try:
        v = c.cell_contents
    except ValueError:                       # an empty cell
        return None
    try:
        return copy.deepcopy(v, memo)
    except (TypeError, copy.Error):
        return v


def _rebind_closures(rep, memo):
    """Functions stored on a module (patched forwards: `emb.forward = forward` closing over the ORIGINAL module's
    bound method) are not copied by deepcopy; rebuild them with their closure cells pointing at the copies."""
    import types
    for m in rep.modules():
        for k, f in list(vars(m).items()):
            if isinstance(f, types.FunctionType) and f.__closure__:
                cells = tuple(types.CellType(_copy_cell(c, memo)) for c in f.__closure__)
                g = types.FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, cells)
                g.__kwdefaults__ = f.__kwdefaults__
                setattr(m, k, g)


class CFGParallel:
    """Classifier-free guidance on two cards (data parallel inside one job): a whole copy of the transformer on a
    second card runs the conditional forward while card 0 runs the unconditional one -- one hand-over per step
    (the cond result), instead of TP's two all-reduces per block. The result is the same as running the two
    forwards one after the other on card 0 (the same kernels on the same kind of card).

    The pipeline calls transformer(cond) and then transformer(uncond), and reads the cond result only after the
    uncond call (the same pattern cfg_batch relies on): the cond call starts the copy's forward in a thread and returns
    a placeholder, the uncond call runs on card 0, then the cond result is copied into the placeholder. Installed per
    request by cfg_batch_scope (only while CFG is really on)."""

    def __init__(self, rep, device, orig):
        self.rep, self.dev, self.orig = rep, torch.device(device), orig
        self.rep_fwd = rep.forward
        self.pending = None
        self.batched = 0

    @staticmethod
    def _placeholder(args, kw):
        x = kw.get("hidden_states")
        if x is None and args:
            x = args[0]
        if isinstance(x, list) and len(x) == 1 and torch.is_tensor(x[0]):
            return [torch.empty_like(x[0])], "list"
        if torch.is_tensor(x) and not kw.get("return_dict", True):
            return (torch.empty_like(x),), "tuple"
        return None, None

    def _start(self, args, kw):
        box = {}
        grad, inf = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
        src = torch.cuda.current_stream()
        src.synchronize()                                      # the inputs are complete before the copy reads them

        def run():
            try:
                torch.cuda.set_device(self.dev)
                with torch.inference_mode(inf), torch.set_grad_enabled(grad):
                    a, k = _tree_to(args, self.dev), _tree_to(kw, self.dev)
                    box["out"] = self.rep_fwd(*a, **k)
                    torch.cuda.current_stream(self.dev).synchronize()
            except BaseException as e:                         # noqa: BLE001
                box["err"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t, box

    def _finish(self):
        if self.pending is None:
            return
        (t, box), ph, kind = self.pending
        self.pending = None
        t.join()
        if "err" in box:
            raise box["err"]
        out = box["out"]
        src = out[0] if kind == "list" else out[0]
        ph[0].copy_(xcopy(src, ph[0].device))

    def forward(self, *args, **kw):
        ph, kind = self._placeholder(args, kw)
        if ph is None:
            self._finish()
            return self.orig(*args, **kw)
        if self.pending is None:                               # cond call: start it on the other card
            self.pending = (self._start(args, kw), ph, kind)
            return ph
        try:                                                   # uncond call: here, at the same time
            out = self.orig(*args, **kw)
        finally:
            self._finish()
        self.batched += 1
        return out

    __call__ = forward


def enable_cfg_parallel(tr, device, verbose=True):
    """CFG parallel for transformer `tr`: a whole copy on `device`, made at its first use (after the loader's last patch
    of `tr`); cfg_batch_scope uses it while CFG is on."""
    tr._cfg_par_dev = (str(device), verbose)
    return tr


def cfg_parallel_copy(tr):
    """(copy, device) for CFG parallel, or None when tr has none. The copy is made once, at the first CFG request."""
    rep = tr.__dict__.get("_cfg_par_rep")
    if rep is not None:
        return rep
    d = tr.__dict__.get("_cfg_par_dev")
    if d is None:
        return None
    device, verbose = d
    r = replicate(tr, device)
    torch.cuda.empty_cache()
    tr._cfg_par_rep = (r, device)
    if verbose:
        n = sum(t.numel() * t.element_size() for t in list(r.parameters()) + list(r.buffers()))
        print(f"[cfg-parallel] {type(tr).__name__}: a copy on {device} ({n / 1e9:.2f} GB)", flush=True)
    return tr._cfg_par_rep


def _bytes(mods):
    """Bytes the modules take on a card (a table still in its file, "disk" mode: its file size)."""
    seen, n = set(), 0
    for top in mods:
        for m in top.modules():
            if getattr(m, "on_disk", False) and getattr(m, "path", None) and id(m) not in seen:
                seen.add(id(m)); n += os.path.getsize(m.path)
            for t in list(m._parameters.values()) + list(m._buffers.values()):
                if t is not None and id(t) not in seen:
                    seen.add(id(t)); n += t.numel() * t.element_size()
    return n


def _load_disk_left(tr, device):
    """Tables loaded header-only ("disk") that were not split: load them whole on `device` (same flags, same bias)."""
    from qv_band.formats.qvb8r.module import QVBandLinearV8R
    for name, m in list(tr.named_modules()):
        for k, c in list(m._modules.items()):
            if isinstance(c, QVBandLinearV8R) and c.on_disk:
                full = QVBandLinearV8R.from_packfile(c.path, device=str(device),
                                                     bias=None if c.bias is None else c.bias.detach())
                _copy_flags(c, full)
                m._modules[k] = full


def _card_weights(tr, devices, todo=()):
    """Share of the split weights per card. QV_TP_DIT_WEIGHTS='1,2'; default: even when the cards have about the same free
    memory, else in proportion to what each card has free less an activation reserve (QV_TP_DIT_RESERVE_GB, default 3):
    card 0 usually also holds the text encoder and the VAE. Cards >= 1 also get their own copy of every part that is not
    split (norms, embeddings, small linears) at the first call: that is taken off their free memory first."""
    w = os.environ.get("QV_TP_DIT_WEIGHTS")
    if w:
        return [float(x) for x in w.split(",")]
    res = float(os.environ.get("QV_TP_DIT_RESERVE_GB", "3")) * 1e9
    split = _bytes(_get(p, k) for p, k, _, _ in todo)
    rest = max(0, _bytes([tr]) - split)                       # what every other card copies whole
    free = []
    for i, d in enumerate(devices):
        d = torch.device(d)
        if d.type != "cuda":
            return None
        free.append(max(1e9, torch.cuda.mem_get_info(d)[0] - res - (rest if i else 0)))
    if os.environ.get("QV_TP_DIT_DEBUG") == "1":
        print(f"[tp-dit] split weights {split / 1e9:.1f} GB, unsplit copy per extra card {rest / 1e9:.1f} GB", flush=True)
    if len(set(str(d) for d in devices)) < len(devices) or max(free) < 1.15 * min(free):
        return None
    return free


def _fwd_for(orig, tr, rep):
    """The forward TP saved for card 0 (tr's own, before any wrapper went on top), rebound to card r's copy."""
    if getattr(orig, "__self__", None) is tr:
        return orig.__func__.__get__(rep)
    return rep.forward


def _set_heads(attn, heads, hd):
    """This card's head count, and every width derived from it that attention code reads (diffusers Qwen-Image:
    head_dim = attn.inner_dim // attn.heads)."""
    attn.heads = heads
    for k in ("inner_dim", "inner_kv_dim"):
        if isinstance(getattr(attn, k, None), int):
            setattr(attn, k, heads * hd)


class TPDiT:
    """Installed as transformer.forward: runs every card's copy at once, returns card 0's output."""

    def __init__(self, tr, devices, verbose=True):
        world = len(devices)
        self.hub = Hub(devices)
        todo, attns = _plan(tr, world)
        self.weights = _card_weights(tr, devices, todo)
        self.tps = [ThreadTP(self.hub, r, self.weights) for r in range(world)]
        # every card's share of every split module, built straight on its card; card 0's replace the originals in tr
        # (the full weights go). The other cards' copies of the transformer are made at the first call (_build): by
        # then the loader has applied its last patches to tr (fused kernels, GEMM switches), and the copies take them.
        self.memos = [None] + [{} for _ in range(1, world)]
        dbg = os.environ.get("QV_TP_DIT_DEBUG") == "1"
        if dbg:
            print(f"[tp-dit] weights {self.weights}; free " +
                  ", ".join(f"{d}: {torch.cuda.mem_get_info(torch.device(d))[0] / 1e9:.1f} GB" for d in devices), flush=True)
        from qv_band.run.weight_load import load_parallel as run_parallel, warm, workers
        warm(devices)

        def shards(item):                                       # every card's share of one module (QVB8R: cut in place)
            parent, key, kind, hd = item
            m = _get(parent, key)
            out = []
            for r in range(world):
                sh = _shard(m, kind, hd, self.tps[r], devices[r])
                _copy_flags(m, sh)
                out.append(sh)
            return out
        t0 = time.time()
        built = run_parallel(shards, todo)                      # QV_LOAD_WORKERS threads, results in order
        for (parent, key, kind, hd), shs in zip(todo, built):
            for r in range(1, world):
                self.memos[r][id(shs[0])] = shs[r]
            _set(parent, key, shs[0])
        del built
        if dbg:
            print(f"[tp-dit] shares built in {time.time() - t0:.0f}s on {workers()} threads: " + ", ".join(
                f"{d} {torch.cuda.memory_allocated(torch.device(d)) / 1e9:.1f} GB" for d in devices), flush=True)
        _load_disk_left(tr, devices[0])
        self.heads = {name: [self.tps[r].split(heads * hd, hd) for r in range(world)] for name, m, heads, hd in attns}
        self.heads = {name: [(b - a) // hd for a, b in v] for (name, v), (_, _, _, hd) in zip(self.heads.items(), attns)}
        self.hds = {name: hd for name, m, heads, hd in attns}
        for name, m, heads, hd in attns:
            _set_heads(m, self.heads[name][0], hd)
        self.tr, self.reps, self.fwds = tr, None, None
        d0 = torch.device(devices[0])
        if any(t.device != d0 for t in list(tr.parameters()) + list(tr.buffers())):
            for m in tr.modules():                              # what is still in host RAM / elsewhere -> card 0
                for k, p in list(m._parameters.items()):
                    if p is not None and p.device != d0:
                        m._parameters[k] = nn.Parameter(xcopy(p.detach(), d0), requires_grad=p.requires_grad)
                for k, b in list(m._buffers.items()):
                    if b is not None and b.device != d0:
                        m._buffers[k] = xcopy(b, d0)
            _move_loose(tr, d0)
        self.devs = self.hub.devs
        self.n_split = len(todo)
        torch.cuda.empty_cache()
        if verbose:
            h0 = next(iter(self.heads.values()), [])
            print(f"[tp-dit] {type(tr).__name__}: {len(todo)} modules split over {devices} "
                  f"({len(attns)} attentions, heads per card {h0}), all-reduce "
                  f"{'through host memory' if self.hub.via_host else 'card to card'}", flush=True)

    def _build(self):
        if os.environ.get("QV_TP_DIT_DEBUG") == "1":
            print("[tp-dit] at first forward: " + ", ".join(
                f"{d} alloc {torch.cuda.memory_allocated(d) / 1e9:.2f} GB, free {torch.cuda.mem_get_info(d)[0] / 1e9:.2f} GB"
                for d in self.devs), flush=True)
        reps = [self.tr]
        for r in range(1, len(self.devs)):
            if os.environ.get("QV_TP_DIT_DEBUG") == "1":
                inshare = {id(x) for sh in self.memos[r] for x in ()}
                split_ids = set()
                for m in self.tr.modules():
                    if id(m) in self.memos[r]:
                        split_ids |= {id(x) for x in list(m.parameters()) + list(m.buffers())}
                rest, seen = {}, set()
                for n, t in list(self.tr.named_parameters()) + list(self.tr.named_buffers()):
                    if t is None or id(t) in split_ids or id(t) in seen:
                        continue
                    seen.add(id(t))
                    key = ".".join(x for x in n.split(".") if not x.isdigit())
                    rest[key] = rest.get(key, 0) + t.numel() * t.element_size()
                top = sorted(rest.items(), key=lambda kv: -kv[1])[:10]
                print(f"[tp-dit] unsplit to copy: {sum(rest.values()) / 1e9:.2f} GB: "
                      f"{[(k, round(v / 1e6)) for k, v in top]}", flush=True)
                hits = []
                for mn, mm in self.tr.named_modules():
                    for k, v in mm.__dict__.items():
                        if k in ("_parameters", "_buffers", "_modules") or k.startswith("__"):
                            continue
                        vals = v if isinstance(v, (list, tuple)) else list(v.values()) if isinstance(v, dict) else [v]
                        for x in vals:
                            if isinstance(x, nn.Module) and id(x) not in self.memos[r]:
                                hits.append((_bytes([x]), f"{mn}.{k}:{type(x).__name__}"))
                            elif isinstance(x, torch.Tensor) and x.numel() > 1:
                                hits.append((x.numel() * x.element_size(), f"{mn}.{k}:tensor"))
                hits.sort(reverse=True)
                print(f"[tp-dit] attrs outside the module tree: {sum(b for b, _ in hits) / 1e9:.2f} GB: "
                      f"{[(n, round(b / 1e6)) for b, n in hits[:8]]}", flush=True)
            rep = replicate(self.tr, self.devs[r], self.memos[r])
            for name, hs in self.heads.items():
                _set_heads(rep.get_submodule(name), hs[r], self.hds[name])
            reps.append(rep)
            self.memos[r] = None
        torch.cuda.empty_cache()
        self.reps = reps
        self.fwds = [self.orig0] + [_fwd_for(self.orig0, self.tr, rep) for rep in reps[1:]]

    def __call__(self, *args, **kw):
        if self.reps is None:
            self._build()
        world = len(self.reps)
        outs, errs = [None] * world, [None] * world
        grad, inf = torch.is_grad_enabled(), torch.is_inference_mode_enabled()

        def run(r):
            try:
                torch.cuda.set_device(self.devs[r])
                with torch.inference_mode(inf), torch.set_grad_enabled(grad):
                    a, k = (args, kw) if r == 0 else (_tree_to(args, self.devs[r]), _tree_to(kw, self.devs[r]))
                    outs[r] = self.fwds[r](*a, **k)
                    torch.cuda.current_stream(self.devs[r]).synchronize()
            except BaseException as e:                                  # noqa: BLE001
                errs[r] = e
                self.hub.bar.abort()

        ths = [threading.Thread(target=run, args=(r,), daemon=True) for r in range(1, world)]
        for t in ths:
            t.start()
        run(0)
        for t in ths:
            t.join()
        if any(e is not None for e in errs):
            self.hub.bar.reset()
            e = next(e for e in errs if e is not None and not isinstance(e, threading.BrokenBarrierError)) \
                if any(e is not None and not isinstance(e, threading.BrokenBarrierError) for e in errs) else \
                next(e for e in errs if e is not None)
            raise e
        return outs[0]


def enable_tp(tr, devices, verbose=True) -> TPDiT:
    """Split `tr` (a diffusers transformer, already on devices[0]) over `devices`. Idempotent."""
    if getattr(tr, "_tp_dit", None) is not None:
        return tr._tp_dit
    if len(devices) < 2:
        raise ValueError("tp_dit: need two or more cards")
    t = TPDiT(tr, list(devices), verbose)
    t.orig0 = tr.forward                       # card 0 runs tr's own forward (with any wrapper installed before TP)
    # callers that read inspect.signature(transformer.forward) to choose their kwargs (MiniMax-H3's denoiser) see the
    # real forward's parameters, not a bare (*args, **kwargs)
    import inspect
    try:
        t.__signature__ = inspect.signature(t.orig0)
    except (TypeError, ValueError):
        pass
    t.__wrapped__ = t.orig0
    tr._tp_dit = t
    tr.forward = t
    return t
