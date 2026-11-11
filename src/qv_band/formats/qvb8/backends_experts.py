"""QVB8 CPU / expert-stack backends.

The CPU and MoE-expert readers the format registry (common/formats.py) offers for QVB8: same constructor keywords
and `from_packfile` signature as QVBandLinearV8, so nothing downstream has to know which reader it holds.
"""
from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn

from qv_band.formats.qvb8.module import QVBandLinearV8, _load_ext


def _blob_shape(blob: bytes) -> tuple[int, int]:
    """(Q, P) from the header alone, for either container (QVBD or the legacy standalone QVB8 blob)."""
    if blob[:4] == b"QVBD":
        from qv_band.formats.qvb8.pack_container import _unpack_header
        return _unpack_header(blob)[:2]
    from qv_band.formats.qvb8.pack import read_header
    return read_header(blob)[:2]


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

    # Up to this many tokens the fused kernel (attention-free, one launch per projection) runs; above it the
    # per-expert GEMM loop. The loop costs about the same from 16 to 64 tokens (one host sync + up to 2 x experts
    # GEMMs per layer), the fused kernel grows with the tokens. Measured on the dev GPU (35B-A3B, DFlash K=8, ms per step,
    # scheduler, 2026-11-10): 2 requests (16 tok) 261 loop / 128 fused; 3 (24 tok) 288 / 163; 6 (48 tok) 421 / 313;
    # 8 (64 tok) 447 / 362; one request 85 either way. So several users share a step only when the limit covers
    # their tokens. sm_70 (V100) keeps 8: not measured there.
    FUSED_MAX_TOKENS = 8
    FUSED_MAX_TOKENS_SM80 = 64
    _limit_by_dev: dict = {}

    @classmethod
    def fused_max_tokens(cls, device) -> int:
        lim = cls._limit_by_dev.get(device)
        if lim is None:
            sm80 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
            lim = cls._limit_by_dev[device] = cls.FUSED_MAX_TOKENS_SM80 if sm80 else cls.FUSED_MAX_TOKENS
        return lim

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        n_tok = top_k_index.shape[0]
        if (hidden_states.is_cuda and n_tok <= self.fused_max_tokens(hidden_states.device)
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
