"""QVB8 sharded / CPU / expert-stack backends.

The tensor-parallel, CPU and MoE-expert readers the format registry
(common/formats.py) offers for QVB8: same constructor keywords and
`from_packfile` signature as QVBandLinearV8, so nothing downstream has to know
which reader it holds.

QVB8 makes all three simple for one reason: a row is self-contained. Its
origin (min_s), its width (nbits) and its bit offset are per row, so a row
slice is just an array slice -- no codebook to re-base, no anchors to
recompute.
"""
from __future__ import annotations


import numpy as np
import numpy as np
import torch
import torch.nn as nn

from qv_band.formats.qvb8.module import QVBandLinearV8, _load_ext, forward_gemv


def shard_bounds(Q: int, world: int, rank: int) -> tuple[int, int, int]:
    """Even row split with a fixed per-rank stride (ceil) so every rank's output
    has the same width for all_gather; the last rank is zero-padded.
    Returns (start, end, stride), where end - start may be < stride on the tail."""
    stride = -(-Q // world)  # ceil
    start = min(rank * stride, Q)
    end = min(start + stride, Q)
    return start, end, stride


def _unpack_blob(blob: bytes):
    """(min_s, tau_s, nbits, bit_off, buf, Q, P, tau, esc_idx, esc_vals) of a pack blob,
    whichever container it is in: the current one (magic QVBD, pack_container.py) or
    the legacy standalone QVB8 blob."""
    if blob[:4] == b"QVBD":
        from qv_band.formats.qvb8.pack_container import unpack_formula
        d = unpack_formula(blob, skip_bit_decode=True)
        # The sharded GEMV decodes in fp16/fp32. Blob only (no path, so no escape_tables.json): the pack's full list,
        # exact values -- correct; an fp16 run of an fp16+bf16 pack is refused up front (require_fp16_tables).
        return (d["min_s"], d["tau_s"], d["nbits"], d["bit_off"], d["buf"],
                d["Q"], d["P"], d["tau"], d["esc_idx"], d["esc_vals"])
    from qv_band.formats.qvb8.pack import unpack_tensor
    return unpack_tensor(blob)


def _blob_shape(blob: bytes) -> tuple[int, int]:
    """(Q, P) from the header alone, for either container (see _unpack_blob)."""
    if blob[:4] == b"QVBD":
        from qv_band.formats.qvb8.pack_container import _unpack_header
        return _unpack_header(blob)[:2]
    from qv_band.formats.qvb8.pack import read_header
    return read_header(blob)[:2]


def _slice_rows(blob: bytes, start: int, end: int):
    """The QVB8 fields for rows [start, end), as a self-contained slice.

    The bit payload is REPACKED to hold only these rows. Handing over the
    whole buffer and keeping the original offsets would also decode
    correctly -- rows are independent -- but then every rank of a
    tensor-parallel run holds the entire payload, which is exactly what
    sharding exists to avoid (measured: a 0.92 GiB lm_head would cost
    6.4 GiB across 8 ranks instead of 0.92 GiB).

    Row widths vary, so the slice is rebuilt bit by bit rather than cut at
    a byte boundary: row r's field starts at an arbitrary bit, and only
    row 0 of the slice is guaranteed to land anywhere convenient.
    """
    import numpy as np
    (min_s, tau_s, nbits, bit_off, buf, Q, P, tau,
     esc_idx, esc_vals) = _unpack_blob(blob)
    if not (0 <= start < end <= Q):
        raise ValueError(f"row slice [{start}, {end}) out of range for Q={Q}")

    sub_nbits = nbits[start:end]
    # Rows are bit-contiguous (pack_bits_rows), so the slice is one byte
    # range with the offsets rebased by a whole number of bytes: the
    # sub-byte phase of each row is preserved and the reader strides by bit
    # offset anyway. Decoding to ids and repacking gave the same bytes at
    # ~100x the CPU cost (it dominated tensor-parallel load time).
    bit_off64 = bit_off[start:end].astype(np.int64)
    byte_lo = int(bit_off64[0]) >> 3
    bit_hi = int(bit_off64[-1]) + int(sub_nbits[-1]) * P
    byte_hi = (bit_hi + 7) >> 3
    sub_buf = np.zeros(byte_hi - byte_lo + 8, dtype=np.uint8)  # 8B read pad
    sub_buf[:byte_hi - byte_lo] = buf[byte_lo:byte_hi]
    sub_off = (bit_off64 - (byte_lo << 3)).astype(np.uint64)

    keep = (esc_idx >= start * P) & (esc_idx < end * P)
    return {
        "min_s": min_s[start:end],
        "nbits": sub_nbits,
        "bit_off": sub_off,
        "ids": sub_buf,
        "Q": end - start,
        "P": P,
        "tau_s": tau_s[start:end],
        # Escape positions are flat (row*P + col), so they rebase by the
        # same row shift.
        "esc_idx": esc_idx[keep] - start * P,
        "esc_vals": esc_vals[keep],
    }


class QVBandShardedLinearV8(nn.Module):
    """Output-row-sharded QVB8 linear.

    Each rank holds rows [start, end) and computes that slice of Y;
    forward() all-gathers the ranks' columns into the full output, so the
    rest of the model sees an ordinary replicated activation. Same
    contract as a row-parallel linear: each rank holds rows [start, end).

    A row slice is exact here by construction: every row is banded against
    its OWN min and width, so rows never share state that would have to be
    re-derived for a subset.
    """

    def __init__(self, packfile: str, *, rank: int, world: int,
                 device: str = "cuda", bias=None):
        super().__init__()
        with open(packfile, "rb") as fh:
            blob = fh.read()
        Q = _blob_shape(blob)[0]
        start, end, stride = shard_bounds(Q, world, rank)
        self.rank, self.world, self.stride = rank, world, stride
        self.shard_start, self.shard_end = start, end
        self._local_rows = end - start

        f = _slice_rows(blob, start, end)
        dev = torch.device(device)
        self.in_features = int(f["P"])
        self.out_features = int(Q)          # the FULL output width
        self.local_out_features = int(f["Q"])
        self.register_buffer("tau_s",
                             torch.from_numpy(f["tau_s"]).to(dev),
                             persistent=False)
        self.register_buffer("min_s", torch.from_numpy(f["min_s"]).to(dev),
                             persistent=False)
        self.register_buffer("nbits",
                             torch.from_numpy(f["nbits"].astype(np.uint8)).to(dev),
                             persistent=False)
        self.register_buffer("bit_off",
                             torch.from_numpy(f["bit_off"].astype(np.int64)).to(dev),
                             persistent=False)
        self.register_buffer("ids", torch.from_numpy(f["ids"]).to(dev),
                             persistent=False)
        self.register_buffer("esc_idx",
                             torch.from_numpy(f["esc_idx"].astype(np.int64)).to(dev),
                             persistent=False)
        self.register_buffer("esc_vals",
                             torch.from_numpy(f["esc_vals"].astype(np.float32)).to(dev),
                             persistent=False)
        self.bias = None if bias is None else nn.Parameter(bias.to(dev),
                                                           requires_grad=False)

    @classmethod
    def from_packfile(cls, packfile: str, *, rank: int, world: int,
                      bias=None, device: str = "cuda"):
        # Keyword order is rank/world first so a
        # caller can swap the class without touching the call site.
        return cls(packfile, rank=rank, world=world, device=device, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        ext = _load_ext()
        orig = x.shape
        x2d = x.reshape(-1, orig[-1])
        if x2d.device != self.min_s.device:
            x2d = x2d.to(self.min_s.device)
        M = x2d.shape[0]

        if self._local_rows > 0:
            local = forward_gemv(ext, x2d.contiguous(), self.ids, self.min_s,
                                 self.nbits, self.bit_off, self.in_features,
                                 self.tau_s, self.esc_idx, self.esc_vals
                                 ).to(x.dtype)
        else:
            # A rank past the end of the tensor owns no rows at all, which
            # happens whenever world does not divide Q.
            local = torch.zeros(M, 0, dtype=x.dtype, device=x2d.device)

        # Zero-pad to the common stride so every rank contributes an equal-size
        # chunk; the padding is trimmed after the gather. shard_bounds gives the
        # LAST rank fewer rows whenever world does not divide Q (e.g. Q=10240
        # over 3 ranks -> 3414/3414/3412), and the collective requires
        # identical shapes, so this is not optional.
        if local.shape[1] < self.stride:
            pad = torch.zeros(M, self.stride - local.shape[1],
                              dtype=local.dtype, device=local.device)
            local = torch.cat([local, pad], dim=1)

        if self.world > 1:
            # This rank only holds rows [start, end), so the remaining
            # output columns live on other ranks and MUST be gathered.
            # Returning `local` without a process group would silently hand
            # back a truncated activation, so fail loudly instead.
            if not dist.is_initialized():
                raise RuntimeError(
                    f"{type(self).__name__} was built with world={self.world} "
                    f"but torch.distributed is not initialised -- this rank "
                    f"holds only rows [{self.shard_start}, {self.shard_end}) "
                    f"of {self.out_features} and cannot complete the output.")
            # ONE collective into one buffer. The list form of all_gather
            # allocates `world` tensors, copies each way, and needs a cat
            # afterwards: per linear that is a dozen extra launches, and a
            # decode step has ~500 linears on a CPU-bound path.
            gathered = torch.empty(self.world, M, self.stride,
                                   dtype=local.dtype, device=local.device)
            dist.all_gather_into_tensor(gathered, local.contiguous())
            y = gathered.permute(1, 0, 2).reshape(
                M, self.world * self.stride)[:, :self.out_features]
        else:
            y = local[:, :self.out_features]

        y = y.reshape(*orig[:-1], self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(x.dtype)


class _FusedMember(nn.Module):
    """Stands in for one projection of a QVBandFusedShardedGroup."""

    def __init__(self, group, idx: int, in_features: int, out_features: int):
        super().__init__()
        object.__setattr__(self, "_group", group)     # not a registered child
        self._idx = idx
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._group.member_forward(self._idx, x)


class QVBandFusedShardedGroup:
    """Several QVBandShardedLinearV8 that read the SAME input, run as one.

    A decode step is bound by launch count and by the collective after every
    sharded linear (~500 per token on a 27B), not by the weight kernel. The
    projections of one block that share an input (qkv/z/b/a, q/k/v, gate/up)
    are therefore concatenated row-wise into one pack: ONE kernel call and
    ONE all_gather give every member's output. Rows are independent in QVB8,
    so the result is bit-identical to running the members separately.

    The first member called computes everything; the others take their slice
    from the cache, which is released once every member has been served.
    Keyed on the input tensor's identity+version, and holding a reference to
    it, so a different tensor can never be mistaken for the cached one.
    """

    def __init__(self, members: list):
        m0 = members[0]
        self.world, self.rank = m0.world, m0.rank
        self.in_features = m0.in_features
        self.strides = [m.stride for m in members]
        self.outs = [m.out_features for m in members]
        self.locals = [m.local_out_features for m in members]
        dev = m0.min_s.device
        P = self.in_features
        ids, bit_off, esc_idx, esc_vals = [], [], [], []
        byte_base = row_base = 0
        for m in members:
            ids.append(m.ids)
            bit_off.append(m.bit_off + 8 * byte_base)
            esc_idx.append(m.esc_idx + row_base * P)
            esc_vals.append(m.esc_vals)
            byte_base += m.ids.numel()
            row_base += m.local_out_features
        self.ids = torch.cat(ids)
        self.bit_off = torch.cat(bit_off)
        self.esc_idx = torch.cat(esc_idx)
        self.esc_vals = torch.cat(esc_vals)
        self.min_s = torch.cat([m.min_s for m in members])
        self.tau_s = torch.cat([m.tau_s for m in members])
        self.nbits = torch.cat([m.nbits for m in members])
        self.device = dev
        self._n = len(members)
        self._key = None
        self._x = None
        self._out = None
        self._left = 0

    def _compute(self, x: torch.Tensor):
        import torch.distributed as dist
        ext = _load_ext()
        x2d = x.reshape(-1, x.shape[-1])
        M = x2d.shape[0]
        local = forward_gemv(ext, x2d.contiguous(), self.ids, self.min_s, self.nbits,
                             self.bit_off, self.in_features, self.tau_s,
                             self.esc_idx, self.esc_vals).to(x.dtype)
        # local is (M, sum(local rows)); every member divides evenly by world
        # (checked when the group is built), so local rows == stride.
        gathered = torch.empty(self.world, M, local.shape[1],
                               dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(gathered, local.contiguous())
        outs, off = [], 0
        for stride, out_f in zip(self.strides, self.outs):
            part = gathered[:, :, off:off + stride]
            outs.append(part.permute(1, 0, 2).reshape(M, self.world * stride)[:, :out_f])
            off += stride
        return outs

    def member_forward(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        key = (x.data_ptr(), x._version, tuple(x.shape), x.dtype)
        if self._key != key or self._left <= 0:
            self._out = self._compute(x)
            self._key, self._x, self._left = key, x, self._n
        y = self._out[idx]
        self._left -= 1
        if self._left <= 0:
            self._key = self._x = self._out = None
        return y.reshape(*x.shape[:-1], self.outs[idx])


# Projections of one block that consume the same hidden state.
_FUSE_GROUPS = (("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
                ("q_proj", "k_proj", "v_proj"),
                ("gate_proj", "up_proj"))


def fuse_sharded_siblings(model) -> int:
    """Replace each group of sibling QVBandShardedLinearV8 by fused members.
    Returns the number of groups fused. Skips any group whose shards are not
    all even across the ranks or that carries a bias."""
    n = 0
    # Hold only the PARENT modules: a list of every module would keep the old
    # per-projection weights alive while their fused copies are built.
    parents = [mod for mod in model.modules()
               if any(hasattr(mod, nms[0]) for nms in _FUSE_GROUPS)]
    for module in parents:
        for names in _FUSE_GROUPS:
            subs = [getattr(module, nm, None) for nm in names]
            if not all(isinstance(m, QVBandShardedLinearV8) for m in subs):
                continue
            w = subs[0].world
            if (w < 2 or any(m.bias is not None for m in subs)
                    or any(m.out_features % w for m in subs)
                    or len({m.in_features for m in subs}) != 1):
                continue
            group = QVBandFusedShardedGroup(subs)
            for i, (nm, m) in enumerate(zip(names, subs)):
                setattr(module, nm, _FusedMember(group, i, m.in_features,
                                                 m.out_features))
            del subs, m
            n += 1
    return n


class QVBandLinearCPUV8(nn.Module):
    """CPU QVB8 linear.

    Decodes with numpy through the packer's own reference reconstruction,
    so the CPU path is the codec's definition rather than a second
    implementation that could drift from it. Same public shape as
    QVBandLinearCPU.
    """

    def __init__(self, blob: bytes, bias=None):
        super().__init__()
        Q, P = _blob_shape(blob)
        self.in_features, self.out_features = int(P), int(Q)
        self._blob = blob
        self._W = None
        self.bias = None if bias is None else nn.Parameter(bias,
                                                           requires_grad=False)

    @classmethod
    def from_packfile(cls, path: str, *, bias=None,
                      mmap_file: bool = True) -> "QVBandLinearCPUV8":
        # `mmap_file` mirrors QVBandLinearCPU.from_packfile's signature so
        # the two are interchangeable. QVB8 reads the blob whole (its rows
        # are bit-packed back to back, so there is no per-row range to map
        # lazily), and the flag is accepted and ignored rather than making
        # the caller know which format it holds.
        del mmap_file
        with open(path, "rb") as fh:
            return cls(fh.read(), bias=bias)

    def _weights(self) -> torch.Tensor:
        """Dense weights, decoded once.

        A CPU run has no kernel to fuse decode into the matmul, so the
        alternative is decoding per call -- which for any real model is
        far more expensive than the memory this holds.
        """
        if self._W is None:
            if self._blob[:4] == b"QVBD":
                from qv_band.formats.qvb8.pack_container import unpack_formula
                from qv_band.formats.qvb8.decode import reconstruct
                d = unpack_formula(self._blob)
                # fp16 is the restore precision (what the CUDA decode stores
                # and the certificate covers), so narrow the fp32 formula
                # value, escapes included, exactly like the GPU path does.
                W = reconstruct(d["min_s"], d["tau_s"], d["codes"],
                                d["esc_idx"], d["esc_vals"]
                                ).astype(np.float16).astype(np.float32)
            else:
                from qv_band.formats.qvb8.pack import decode_tensor
                W = decode_tensor(self._blob)
            self._W = torch.from_numpy(W)
            # Drop the packed bytes: the dense weights supersede them, and
            # holding both is the compressed size on top of the dense one
            # for the rest of the process's life.
            self._blob = None
        return self._W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._weights()
        y = torch.nn.functional.linear(x.to(W.dtype), W)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(x.dtype)


class QVBandExpertsV8(nn.Module):
    """MoE expert stack backed by QVB8 packs.

    Holds one QVBandLinearV8 per expert per projection and runs HF's own
    routing maths over them (the fused decode path is build_stack below).
    """

    def __init__(self, gate_up_experts, down_experts, *, num_experts: int,
                 act_fn=None):
        super().__init__()
        # Validate the expert lists up front. Without it a short
        # list is only caught when routing happens to select a missing
        # expert, as a bare "IndexError: index 2 is out of range" from the
        # ModuleList -- no mention of experts, the stack, or how many were
        # expected. from_packfiles already checks its packfile maps, but
        # this constructor is public and is what a caller building experts
        # by hand reaches.
        gate_up_experts = list(gate_up_experts)
        down_experts = list(down_experts)
        if len(gate_up_experts) != num_experts:
            raise ValueError(f"expected {num_experts} gate_up_proj experts, "
                             f"got {len(gate_up_experts)}")
        if len(down_experts) != num_experts:
            raise ValueError(f"expected {num_experts} down_proj experts, "
                             f"got {len(down_experts)}")
        self.num_experts = int(num_experts)
        self.gate_up_experts = nn.ModuleList(gate_up_experts)
        self.down_experts = nn.ModuleList(down_experts)
        self.act_fn = act_fn if act_fn is not None else nn.SiLU()
        # Stacked layout for the fused decode kernel (see build_stack): None =
        # not built yet, False = cannot be used for this stack.
        self._stk = None

    @classmethod
    def from_packfiles(cls, gate_up_packfiles: dict[int, str],
                       down_packfiles: dict[int, str], *,
                       num_experts: int, act_fn=None,
                       device: str = "cuda",
                       tau: float | None = None) -> "QVBandExpertsV8":
        """`tau` is accepted for call-site compatibility and ignored (every
        pack carries its own certified tau)."""
        missing_gu = [e for e in range(num_experts) if e not in gate_up_packfiles]
        missing_dn = [e for e in range(num_experts) if e not in down_packfiles]
        if missing_gu or missing_dn:
            raise ValueError(
                f"QVBandExpertsV8 requires every expert packed -- missing "
                f"gate_up: {missing_gu[:10]}, down: {missing_dn[:10]}")
        gu = [QVBandLinearV8.from_packfile(gate_up_packfiles[e], device=device)
              for e in range(num_experts)]
        dn = [QVBandLinearV8.from_packfile(down_packfiles[e], device=device)
              for e in range(num_experts)]
        return cls(gu, dn, num_experts=num_experts, act_fn=act_fn)

    # -- fused decode path ------------------------------------------------
    def build_stack(self) -> bool:
        """Concatenate every expert's packed ids into ONE buffer per
        projection, with per-row metadata laid out (expert, row), so a single
        grouped kernel can serve any set of selected experts (see
        qvb8_moe_experts_forward in qvb8_cuda.cu). Each expert module keeps
        working: its `ids` becomes a VIEW of the stacked buffer and the original
        allocation is freed, so the peak extra memory is one expert.

        Refused (returns False, loop path stays in use) when any expert carries
        an escape entry, differing shapes, a non-SiLU
        activation, or the device is not CUDA. Cached."""
        if self._stk is not None:
            return self._stk is not False
        self._stk = False
        if "silu" not in type(self.act_fn).__name__.lower():
            return False
        gu, dn = list(self.gate_up_experts), list(self.down_experts)
        for mods in (gu, dn):
            if not mods or not mods[0].ids.is_cuda:
                return False
            if any(m.esc_idx.numel() for m in mods):
                return False
            if any(m.out_features != mods[0].out_features
                   or m.in_features != mods[0].in_features for m in mods):
                return False
            if mods[0].in_features % 8:
                return False
        if gu[0].out_features != 2 * dn[0].in_features \
                or gu[0].in_features != dn[0].out_features:
            return False

        def stack(mods):
            dev = mods[0].ids.device
            offs, total = [], 0
            for m in mods:
                offs.append(total)
                total += (m.ids.numel() + 15) // 16 * 16
            ids_all = torch.zeros(total + 64, dtype=torch.uint8, device=dev)
            for off, m in zip(offs, mods):
                n = m.ids.numel()
                ids_all[off:off + n].copy_(m.ids)
                m.ids = ids_all[off:off + n]            # view; frees the original
            return dict(
                ids=ids_all,
                min_s=torch.stack([m.min_s for m in mods]).contiguous(),
                tau_s=torch.stack([m.tau_s for m in mods]).contiguous(),
                nbits=torch.stack([m.nbits for m in mods]).contiguous(),
                bit_off=torch.stack([m.bit_off + o * 8 for m, o in zip(mods, offs)]).contiguous(),
                Q=int(mods[0].out_features), P=int(mods[0].in_features))
        self._stk = (stack(gu), stack(dn))
        return True

    # Above this many tokens the per-expert GEMM path (decode + cuBLAS) wins.
    FUSED_MAX_TOKENS = 8

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        n_tok = top_k_index.shape[0]
        if (n_tok <= self.FUSED_MAX_TOKENS and hidden_states.is_cuda
                and self.build_stack()):
            g, d = self._stk
            return _load_ext().moe_experts(
                hidden_states.contiguous(), top_k_index.contiguous(),
                top_k_weights.float(),
                g["ids"], g["min_s"], g["tau_s"], g["nbits"], g["bit_off"], g["Q"], g["P"],
                d["ids"], d["min_s"], d["tau_s"], d["nbits"], d["bit_off"], d["Q"], d["P"])
        return self._forward_loop(hidden_states, top_k_index, top_k_weights)

    def _forward_loop(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                      top_k_weights: torch.Tensor) -> torch.Tensor:
        """Run routed experts with one sorted routing plan.

        The old path built an ``E x top_k x tokens`` one-hot tensor, reduced it
        for every expert, and then ran ``where`` for every hit.  That is a
        sizeable temporary allocation for wide MoE stacks.  Sorting the
        flattened expert ids once replaces those repeated reductions with
        one stable partition; each expert still receives its own independent
        output and the same top-k weighted index_add.
        """
        final = torch.zeros_like(hidden_states)
        n_tokens, top_k = top_k_index.shape
        flat_expert = top_k_index.reshape(-1)
        flat_weight = top_k_weights.reshape(-1)
        token_ids = torch.arange(n_tokens, device=hidden_states.device,
                                 dtype=torch.long).repeat_interleave(top_k)
        order = torch.argsort(flat_expert, stable=True)
        sorted_expert = flat_expert.index_select(0, order)
        counts = torch.bincount(sorted_expert, minlength=self.num_experts)
        # ONE GPU->CPU sync for the whole routing plan, not one per expert.
        # `int(counts[e])` inside the loop below was a separate device sync
        # per expert: a 128-expert, 40-layer MoE decode step paid ~5120
        # synchronisations for ONE token, and every one of them drains the
        # pipeline and blocks the next kernel launch. That is the difference
        # between the MoE running at full speed and at roughly half. One
        # `.tolist()` here moves the same counts to the host once and the
        # loop below then reads plain Python ints.
        counts_host = counts.tolist()
        start = 0
        for e in range(self.num_experts):
            count = counts_host[e]
            if count == 0:
                continue
            end = start + count
            sel = order[start:end]
            tok = token_ids.index_select(0, sel)
            cur = hidden_states.index_select(0, tok)
            gate, up = self.gate_up_experts[e](cur).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = self.down_experts[e](h)
            h = h * flat_weight.index_select(0, sel)[:, None]
            final.index_add_(0, tok, h.to(final.dtype))
            start = end
        return final
