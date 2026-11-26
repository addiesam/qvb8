"""Tensor-parallel QVB8 linears (Megatron / SGLang style).

One model, N GPUs, every GPU working on the SAME token at the same time. A block's
first projections are split by OUTPUT rows (column-parallel: each GPU owns its heads
or its part of the MLP, no communication); its last projection is split by INPUT
columns (row-parallel: each GPU makes a partial sum, then ONE all-reduce). Two
collectives per transformer layer instead of one gather after every linear (the
design removed on issued ~500 collectives per decode step).

Both splits are exact for QVB8 because the certificate is per weight:

  * a row of a pack carries its own min_s, tau_s, bit width and bit offset, so a set
    of rows is a slice of the per-row vectors plus their bits;
  * a column range of a row is a fixed-width bit range of that row (its ids all have
    the row's width), so it is re-packed once at load. The values are unchanged:
    every restored weight is still within tau of the original.

The local pieces are ordinary QVBandLinearV8 modules, so every kernel path (GEMV,
register-blocked sm_70, tensor-core mma, CUDA graphs) runs unchanged on each GPU.

QVB8R (rANS-coded) packs are not split here: a column range of an entropy-coded row
is not addressable without decoding it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from qv_band.formats.qvb8.module import QVBandLinearV8


# ----------------------------------------------------------------------------- split bounds

def split_even(n: int, world: int, rank: int, unit: int = 1) -> tuple[int, int]:
    """[start, end) of rank's share of n items, in whole units (a head, a 128-wide
    tile). n must divide into `world` equal shares of whole units: a ragged split
    would give the ranks different shapes, which the collectives do not allow."""
    if n % unit:
        raise ValueError(f"{n} is not a whole number of units of {unit}")
    units = n // unit
    if units % world:
        raise ValueError(f"{units} units of {unit} do not split evenly over {world} GPUs")
    per = units // world * unit
    return rank * per, (rank + 1) * per


def split_ceil(n: int, world: int, rank: int) -> tuple[int, int]:
    """[start, end) for a GATHERED output (lm_head vocab): ceil(n / world) per rank, the last rank takes the rest.
    Any world size works (248320 over 3 cards); all_gather_last(y, full=n) pads to the common width and trims."""
    per = -(-n // world)
    return min(n, rank * per), min(n, (rank + 1) * per)


def split_weighted(n: int, weights, rank: int, unit: int = 1) -> tuple[int, int]:
    """[start, end) of rank's share of n items, in whole units, in proportion to `weights`
    (one per rank, e.g. (3, 1): a big card takes 3/4). Row/column-parallel layers may be
    uneven (each rank's own shapes; the all-reduce output is full size on every rank).
    Every rank gets at least one unit."""
    if n % unit:
        raise ValueError(f"{n} is not a whole number of units of {unit}")
    units, w = n // unit, [float(x) for x in weights]
    if len(w) < 1 or min(w) <= 0 or units < len(w):
        raise ValueError(f"cannot split {units} units of {unit} by weights {weights}")
    tot, cum, bounds = sum(w), 0.0, [0]
    for i, x in enumerate(w[:-1]):
        cum += x
        b = int(round(units * cum / tot))
        b = max(b, bounds[-1] + 1)                      # at least one unit for this rank
        b = min(b, units - (len(w) - 1 - i))            # and leave one for each later rank
        bounds.append(b)
    bounds.append(units)
    return bounds[rank] * unit, bounds[rank + 1] * unit


# ----------------------------------------------------------------------------- slicing a pack

def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _fields(lin: QVBandLinearV8) -> dict:
    """The pack fields of a loaded linear, on the CPU, as numpy."""
    same = lin.esc_idx_bf16 is lin.esc_idx
    return dict(
        Q=lin.out_features, P=lin.in_features,
        min_s=_np(lin.min_s), tau_s=_np(lin.tau_s), nbits=_np(lin.nbits).astype(np.uint8),
        bit_off=_np(lin.bit_off).astype(np.int64), buf=_np(lin.ids),
        esc=(_np(lin.esc_idx), _np(lin.esc_vals)),
        esc_bf16=None if same else (_np(lin.esc_idx_bf16), _np(lin.esc_vals_bf16)),
        bias=None if lin.bias is None else lin.bias.detach().cpu(),
    )


def _build(f: dict, *, Q: int, P: int, min_s, tau_s, nbits, bit_off, buf, esc, esc_bf16,
           bias, src: QVBandLinearV8, device) -> QVBandLinearV8:
    dev = torch.device(device)
    m = QVBandLinearV8(
        min_s=torch.from_numpy(np.ascontiguousarray(min_s, np.float32)).to(dev),
        nbits=torch.from_numpy(np.ascontiguousarray(nbits, np.uint8)).to(dev),
        bit_off=torch.from_numpy(np.ascontiguousarray(bit_off, np.int64)).to(dev),
        ids=torch.from_numpy(np.ascontiguousarray(buf, np.uint8)).to(dev),
        Q=Q, P=P,
        tau_s=torch.from_numpy(np.ascontiguousarray(tau_s, np.float32)).to(dev),
        esc_idx=torch.from_numpy(np.ascontiguousarray(esc[0], np.int64)).to(dev),
        esc_vals=torch.from_numpy(np.ascontiguousarray(esc[1], np.float32)).to(dev),
        bias=None if bias is None else bias.to(dev),
        esc_idx_bf16=None if esc_bf16 is None else torch.from_numpy(np.ascontiguousarray(esc_bf16[0], np.int64)).to(dev),
        esc_vals_bf16=None if esc_bf16 is None else torch.from_numpy(np.ascontiguousarray(esc_bf16[1], np.float32)).to(dev),
    )
    m.bf16_certified = src.bf16_certified
    m.fp16_table_missing = src.fp16_table_missing
    m._where = getattr(src, "_where", "QVBandLinearV8")
    return m


def _row_ranges(rows) -> list[tuple[int, int]]:
    """rows as a list of [a, b) ranges: an int pair, a list of pairs, or an index array."""
    if isinstance(rows, tuple) and len(rows) == 2 and all(isinstance(v, (int, np.integer)) for v in rows):
        return [(int(rows[0]), int(rows[1]))]
    if isinstance(rows, list) and rows and isinstance(rows[0], tuple):
        return [(int(a), int(b)) for a, b in rows]
    idx = np.asarray(rows, np.int64)
    if idx.size == 0:
        return []
    brk = np.nonzero(np.diff(idx) != 1)[0]
    starts = np.r_[idx[0], idx[brk + 1]]
    ends = np.r_[idx[brk] + 1, idx[-1] + 1]
    return list(zip(starts.tolist(), ends.tolist()))


def _esc_rows(esc, ranges, P):
    """Escapes of the selected rows, renumbered to the new row order."""
    idx, vals = esc
    out_i, out_v, base = [], [], 0
    for a, b in ranges:
        keep = (idx >= a * P) & (idx < b * P)
        out_i.append(idx[keep] - a * P + base * P)
        out_v.append(vals[keep])
        base += b - a
    if not out_i:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(out_i), np.concatenate(out_v)


def _copy_spans(buf: np.ndarray, start_bits: np.ndarray, n_bits: np.ndarray):
    """Copy one bit span per row as whole bytes, back to back (no decoding). The new bit
    offset keeps each span's position inside its first byte: the kernels read ids from
    any bit offset, so nothing is shifted. Returns (new buffer, new bit offsets)."""
    start_bits = start_bits.astype(np.int64)
    lo = start_bits >> 3
    hi = (start_bits + n_bits.astype(np.int64) + 7) >> 3
    lens = hi - lo
    base = np.zeros(lens.size, np.int64)
    np.cumsum(lens[:-1], out=base[1:])
    total = int(lens.sum())
    src = np.repeat(lo - base, lens) + np.arange(total, dtype=np.int64)
    out = np.zeros(total + 8, np.uint8)                  # +8: kernels may read a word past the end
    out[:total] = buf[src]
    return out, (base << 3) + (start_bits & 7)


def slice_rows(lin: QVBandLinearV8, rows, device=None) -> QVBandLinearV8:
    """A linear holding only output rows `rows` (a range (a, b), a list of ranges, or an
    index array), with ONLY those rows' bits: the memory really is split.

    Each kept row's bits are copied as whole bytes; its bit offset keeps the sub-byte phase.
    Several ranges (a fused projection's interleaved share, e.g. q|k|v) work the same way."""
    f = _fields(lin)
    P = f["P"]
    ranges = _row_ranges(rows)
    sel = np.concatenate([np.arange(a, b) for a, b in ranges]) if ranges else np.zeros(0, np.int64)
    nb = f["nbits"][sel]
    buf, bit_off = _copy_spans(f["buf"], f["bit_off"][sel], nb.astype(np.int64) * P)
    bias = None if f["bias"] is None else f["bias"][torch.from_numpy(sel)]
    return _build(f, Q=int(sel.size), P=P, min_s=f["min_s"][sel], tau_s=f["tau_s"][sel], nbits=nb,
                  bit_off=bit_off, buf=buf, esc=_esc_rows(f["esc"], ranges, P),
                  esc_bf16=None if f["esc_bf16"] is None else _esc_rows(f["esc_bf16"], ranges, P),
                  bias=bias, src=lin, device=device or lin.ids.device)


def _esc_cols(esc, c0, c1, P):
    idx, vals = esc
    r, c = idx // P, idx % P
    keep = (c >= c0) & (c < c1)
    return r[keep] * (c1 - c0) + (c[keep] - c0), vals[keep]


def slice_cols(lin: QVBandLinearV8, c0: int, c1: int, device=None) -> QVBandLinearV8:
    """A linear holding input columns [c0, c1) of every row (no bias: the caller adds it
    once, after the all-reduce). Each row's ids are fixed-width, so the columns are one
    bit range per row, copied as whole bytes (no decoding, no re-packing)."""
    f = _fields(lin)
    Q, P = f["Q"], f["P"]
    if not (0 <= c0 < c1 <= P):
        raise ValueError(f"column slice [{c0}, {c1}) out of range for P={P}")
    Pl = c1 - c0
    nb = f["nbits"].astype(np.int64)
    buf, bit_off = _copy_spans(f["buf"], f["bit_off"] + c0 * nb, Pl * nb)
    return _build(f, Q=Q, P=Pl, min_s=f["min_s"], tau_s=f["tau_s"], nbits=f["nbits"],
                  bit_off=bit_off, buf=buf, esc=_esc_cols(f["esc"], c0, c1, P),
                  esc_bf16=None if f["esc_bf16"] is None else _esc_cols(f["esc_bf16"], c0, c1, P),
                  bias=None, src=lin, device=device or lin.ids.device)


# ----------------------------------------------------------------------------- parallel modules

class TPGroup:
    """Who this GPU is in a tensor-parallel group. world == 1 means no communication,
    so the same model code runs on one GPU and on N."""

    def __init__(self, rank: int = 0, world: int = 1, group=None):
        self.rank, self.world, self.group = int(rank), int(world), group
        # cross-machine mode (init_distributed with xnode="gloo"): NCCL inside each machine, gloo between machines
        self.xnode = False
        self.node_group = self.leader_group = None
        self.node_world, self.node_leader, self.is_leader, self.n_nodes = 1, 0, True, 1

    @property
    def graphable(self) -> bool:
        """CUDA graphs can hold every collective (one machine, or NCCL across machines). A gloo hop between
        machines goes through host memory and cannot be captured: the decoders then run eagerly."""
        return not self.xnode

    @classmethod
    def from_dist(cls, group=None):
        import torch.distributed as dist
        if not dist.is_initialized():
            return cls()
        tp = cls(dist.get_rank(group), dist.get_world_size(group), group)
        x = _XNODE.get("tp")
        if x is not None and group is None:
            tp.__dict__.update({k: v for k, v in x.__dict__.items() if k not in ("rank", "world", "group")})
        return tp

    def all_reduce(self, y: torch.Tensor) -> torch.Tensor:
        if self.world > 1:
            import torch.distributed as dist
            if not self.xnode:
                dist.all_reduce(y, group=self.group)
                return y
            # hierarchical: sum inside the machine (NCCL), machine leaders sum over the network (gloo, host
            # memory, fp32), each leader hands the total back to its machine (NCCL). Every rank gets the same bits.
            if self.node_world > 1:
                dist.all_reduce(y, group=self.node_group)
            if self.n_nodes > 1 and self.is_leader:
                h = y.detach().float().cpu()
                dist.all_reduce(h, group=self.leader_group)
                y.copy_(h)
            if self.node_world > 1:
                dist.broadcast(y, self.node_leader, group=self.node_group)
        return y

    def all_gather_last(self, y: torch.Tensor, full: int | None = None) -> torch.Tensor:
        """Concatenate the ranks' pieces along the last dimension. full: total width when the pieces follow
        split_ceil (the last may be narrower): every piece is padded to ceil(full / world), the result trimmed."""
        if self.world == 1:
            return y
        if full is not None and full % self.world:
            per = -(-full // self.world)
            if y.shape[-1] < per:
                y = torch.nn.functional.pad(y, (0, per - y.shape[-1]))
            return self.all_gather_last(y)[..., :full]
        import torch.distributed as dist
        lead = y.shape[:-1]
        y2 = y.reshape(-1, y.shape[-1]).contiguous()
        if self.xnode:                                  # through host memory over the gloo world group
            out = torch.empty(self.world * y2.shape[0], y2.shape[1], dtype=y.dtype)   # gloo: a flat (world*r, c) output
            dist.all_gather_into_tensor(out, y2.cpu(), group=self.group)
            out = out.view(self.world, *y2.shape).to(y.device)
        else:
            out = torch.empty(self.world, *y2.shape, dtype=y.dtype, device=y.device)
            dist.all_gather_into_tensor(out, y2, group=self.group)
        return out.permute(1, 0, 2).reshape(*lead, self.world * y.shape[-1])


_XNODE = {}


def init_distributed(device, timeout=None, xnode=None):
    """Start the process group of a TP service on one or several machines; returns its TPGroup.

    xnode (or env QV_TP_XNODE) -- how machines talk to each other:
      "nccl" (default)  one NCCL world. Fastest when every machine has the same NCCL build and a fast fabric
                        (InfiniBand / RoCE / >= 10 GbE); CUDA graphs keep working.
      "gloo"            NCCL inside each machine, gloo (TCP, host memory) between machines. Works across GPU
                        generations, CUDA / NCCL versions and any network (Wi-Fi included); decoding then runs
                        without CUDA graphs. GLOO_SOCKET_IFNAME picks each machine's interface.
    Launch on every machine: torchrun --nnodes N --node-rank i --nproc-per-node <its cards> --master-addr <rank 0 host> ..."""
    import os
    import socket
    import torch.distributed as dist
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return TPGroup()
    xnode = (xnode or os.environ.get("QV_TP_XNODE", "nccl")).lower()
    kw = {} if timeout is None else {"timeout": timeout}
    if xnode != "gloo":
        dist.init_process_group("nccl", device_id=torch.device(device), **kw)
        return TPGroup.from_dist()
    dist.init_process_group("gloo", **kw)
    rank, world = dist.get_rank(), dist.get_world_size()
    hosts = [None] * world
    dist.all_gather_object(hosts, (os.environ.get("GROUP_RANK") or socket.gethostname(), rank))
    nodes = {}
    for h, r in hosts:
        nodes.setdefault(h, []).append(r)
    order = sorted(nodes.values(), key=lambda rs: rs[0])
    tp = TPGroup(rank, world, None)
    tp.xnode, tp.n_nodes = True, len(order)
    for rs in order:                                    # every rank takes part in creating every group
        g = dist.new_group(rs, backend="nccl", **kw) if len(rs) > 1 else None
        if rank in rs:
            tp.node_group, tp.node_world, tp.node_leader, tp.is_leader = g, len(rs), rs[0], rank == rs[0]
    leaders = [rs[0] for rs in order]
    tp.leader_group = dist.new_group(leaders, backend="gloo", **kw) if len(leaders) > 1 else None
    _XNODE["tp"] = tp
    if rank == 0:
        print(f"[tp] cross-machine TP: {world} ranks on {len(order)} machines {order}: NCCL inside a machine, "
              f"gloo between machines, no CUDA graphs", flush=True)
    return TPGroup.from_dist()


class ColumnParallelLinear(nn.Module):
    """This GPU's output rows of a linear. Without gather the output stays split (the next
    layer is head- or MLP-local); with gather it is concatenated over the GPUs (lm_head)."""

    def __init__(self, local: QVBandLinearV8, tp: TPGroup, full_out: int, gather: bool = False):
        super().__init__()
        self.local, self.tp, self.gather = local, tp, gather
        self.in_features = local.in_features
        self.out_features = full_out if gather else local.out_features
        self.full_out_features = full_out

    @classmethod
    def from_linear(cls, lin: QVBandLinearV8, tp: TPGroup, rows=None, unit: int = 1,
                    gather: bool = False, device=None):
        if rows is None:
            rows = split_even(lin.out_features, tp.world, tp.rank, unit)
        if _row_ranges(rows) == [(0, lin.out_features)]:       # whole table: nothing to cut
            return cls(lin.to(device) if device else lin, tp, lin.out_features, gather)
        return cls(slice_rows(lin, rows, device), tp, lin.out_features, gather)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.local(x)
        return self.tp.all_gather_last(y, self.out_features) if self.gather else y


class RowParallelLinear(nn.Module):
    """This GPU's input columns of a linear: a partial sum, completed by one all-reduce.
    The bias is added once, after the reduce, on every GPU (so every GPU ends with the
    same, full output)."""

    def __init__(self, local: QVBandLinearV8, tp: TPGroup, full_in: int, bias=None):
        super().__init__()
        self.local, self.tp = local, tp
        self.in_features, self.local_in_features = full_in, local.in_features
        self.out_features = local.out_features
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    @classmethod
    def from_linear(cls, lin: QVBandLinearV8, tp: TPGroup, cols=None, unit: int = 1, device=None):
        c0, c1 = cols if cols is not None else split_even(lin.in_features, tp.world, tp.rank, unit)
        dev = device or lin.ids.device
        bias = None if lin.bias is None else lin.bias.detach().to(dev)
        if (c0, c1) == (0, lin.in_features):                   # whole table: nothing to cut
            local = lin.to(dev)
            local.bias = None
            return cls(local, tp, lin.in_features, bias)
        return cls(slice_cols(lin, c0, c1, dev), tp, lin.in_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tp.all_reduce(self.local(x))
        return y if self.bias is None else y + self.bias.to(y.dtype)
