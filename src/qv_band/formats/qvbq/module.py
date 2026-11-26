"""QVBQ: low-precision weight codes kept as-is, decoded inside the matmul.

Pack file `<tensor>.qvbq` (safetensors) holds
    q  uint8 codes  [E, N, K/2] (E2M1) or [E, N, K] (E4M3/E5M2)
    s  scale        [E, SN, SK]  uint8 (e4m3 / E8M0) or fp32
    f  optional fp32 per-row factor [E, N]
and metadata {code, scale, GN, GK, N, K, E}. A 2-D linear is E=1. The bytes are
the checkpoint's own (pack_qvbq.py copies them bit-exact), so the certificate is
|restore(source) - restore(pack)| = 0.

Expert stacks keep gate/up as ONE [gate; up] row block (pack_qvbq de-interleaves
row-interleaved checkpoints when the scale is row-wise, a bit-exact permutation).
"""
from __future__ import annotations

import inspect

import os
import torch
import torch.nn as nn
from safetensors import safe_open

from .kernel import (CODE_E2M1, CODE_E4M3, CODE_E5M2, SCALE_E4M3, SCALE_E8M0,
                     SCALE_F32, moe_align, qvbq_matmul)

CODES = {"e2m1": CODE_E2M1, "e4m3": CODE_E4M3, "e5m2": CODE_E5M2}
SCALES = {"e4m3": SCALE_E4M3, "e8m0": SCALE_E8M0, "f32": SCALE_F32}
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


class QVBQWeight:
    """Codes + scales of one [E, N, K] weight, on one device."""

    def __init__(self, q, s, f, *, code: str, scale: str, GN: int, GK: int, N: int, K: int):
        self.q, self.s, self.f = q, s, f
        self.code_name, self.scale_name = code, scale
        self.code, self.scale_kind = CODES[code], SCALES[scale]
        self.GN, self.GK, self.N, self.K = int(GN), int(GK), int(N), int(K)
        self.E = q.shape[0]
        self.SN, self.SK = s.shape[1], s.shape[2]
        if code == "e2m1" and K % 2:
            raise ValueError("E2M1 needs an even K")

    @classmethod
    def from_file(cls, path: str, device="cuda", experts=None) -> "QVBQWeight":
        """experts=(e0, e1): only that range of the stack is read (expert parallel: each card its own experts)."""
        if experts is None:
            with safe_open(path, "pt", device=str(device)) as f:
                meta = f.metadata()
                q, s = f.get_tensor("q"), f.get_tensor("s")
                fac = f.get_tensor("f") if "f" in f.keys() else None
        else:
            e0, e1 = experts
            with safe_open(path, "pt", device="cpu") as f:
                meta = f.metadata()
                q, s = f.get_slice("q")[e0:e1], f.get_slice("s")[e0:e1]
                fac = f.get_slice("f")[e0:e1] if "f" in f.keys() else None
            if str(device) != "cpu":
                q, s = q.to(device), s.to(device)
                fac = None if fac is None else fac.to(device)
        return cls(q.view(torch.uint8), s if meta["scale"] == "f32" else s.view(torch.uint8),
                   fac, code=meta["code"], scale=meta["scale"], GN=int(meta["GN"]),
                   GK=int(meta["GK"]), N=int(meta["N"]), K=int(meta["K"]))

    def per_expert_nbytes(self) -> int:
        return self.nbytes() // self.E

    def empty_like(self, E: int, device) -> "QVBQWeight":
        """A stack of E expert slots with this weight's format (a GPU expert cache)."""
        z = lambda t: None if t is None else torch.empty((E,) + tuple(t.shape[1:]), dtype=t.dtype, device=device)
        return QVBQWeight(z(self.q), z(self.s), z(self.f), code=self.code_name, scale=self.scale_name,
                          GN=self.GN, GK=self.GK, N=self.N, K=self.K)

    def copy_expert_(self, slot: int, src: "QVBQWeight", e: int) -> None:
        self.q[slot].copy_(src.q[e], non_blocking=True)
        self.s[slot].copy_(src.s[e], non_blocking=True)
        if self.f is not None:
            self.f[slot].copy_(src.f[e], non_blocking=True)

    def pin_(self) -> "QVBQWeight":
        """Host copy page-locked at its exact size (swap/block_stream.pin_exact): the GPU copies straight from it."""
        from qv_band.swap.block_stream import pin_exact
        self.q, self.s = pin_exact(self.q), pin_exact(self.s)
        self.f = None if self.f is None else pin_exact(self.f)
        return self

    def nbytes(self) -> int:
        n = self.q.numel() + self.s.numel() * self.s.element_size()
        return n + (0 if self.f is None else self.f.numel() * 4)

    @torch.no_grad()
    def dense(self, idx=None, dtype=torch.float32) -> torch.Tensor:
        """Reference decode [n, N, K] (audits / tests only; inference uses the kernel)."""
        idx = torch.arange(self.E, device=self.q.device) if idx is None else idx
        q = self.q.index_select(0, idx)
        if self.code == CODE_E2M1:
            lut = torch.tensor(_E2M1, device=q.device)
            v = torch.stack((lut[(q & 15).long()], lut[(q >> 4).long()]), -1).reshape(len(idx), self.N, self.K)
        else:
            v = q.view(torch.float8_e4m3fn if self.code == CODE_E4M3 else torch.float8_e5m2).float()
        s = self.s.index_select(0, idx)
        s = (s.view(torch.float8_e4m3fn).float() if self.scale_kind == SCALE_E4M3 else
             torch.exp2(s.float() - 127) if self.scale_kind == SCALE_E8M0 else s.float())
        s = s.repeat_interleave(self.GN, 1)[:, :self.N].repeat_interleave(self.GK, 2)[:, :, :self.K]
        v = v * s
        if self.f is not None:
            v = v * self.f.index_select(0, idx)[:, :, None]
        return v.to(dtype)


class QVBQLinear(nn.Module):
    """nn.Linear replacement; the E=1 case of the QVBQ kernel."""

    def __init__(self, w: QVBQWeight, bias=None):
        super().__init__()
        assert w.E == 1
        self.w = w
        self.in_features, self.out_features = w.K, w.N
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.weight = torch.empty(0, dtype=torch.bfloat16, device=w.q.device)   # dtype probe only

    @classmethod
    def from_packfile(cls, path: str, bias=None, device="cuda", **_):
        return cls(QVBQWeight.from_file(path, device), bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).to(torch.float16 if x.dtype == torch.float16 else torch.bfloat16).contiguous()
        M = x2.shape[0]
        ids = torch.zeros(M, 1, dtype=torch.long, device=x.device)
        si, be = moe_align(ids, 1, 16)
        y = qvbq_matmul(x2, self.w, si, be, M, 1, out_dtype=x2.dtype)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*shp[:-1], self.out_features).to(x.dtype)


def _glu(act_fn):
    """act(gate, up): two-argument GLU modules (e.g. softcapped SwiGLU) are called
    as-is, one-argument activations as act(gate) * up (the HF MoE convention)."""
    if act_fn is None:
        act_fn = nn.SiLU()
    fwd = act_fn.forward if isinstance(act_fn, nn.Module) else act_fn
    try:
        n = len([p for p in inspect.signature(fwd).parameters.values()
                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "self"])
    except (TypeError, ValueError):
        n = 1
    return act_fn if n >= 2 else (lambda g, u: act_fn(g) * u)


class QVBQExperts(nn.Module):
    """MoE expert stack; forward(hidden, top_k_index, top_k_weights) like QVBandExpertsV8."""

    def __init__(self, gate_up: QVBQWeight, down: QVBQWeight, *, num_experts: int, act_fn=None,
                 e0: int = 0, dtype=torch.bfloat16):
        super().__init__()
        if gate_up.E != down.E or not (0 <= e0 and e0 + gate_up.E <= num_experts):
            raise ValueError(f"experts [{e0}, {e0 + gate_up.E}) of {num_experts}: got {gate_up.E}/{down.E}")
        self.num_experts = num_experts                   # of the whole model (router ids)
        self.e0, self.E = e0, gate_up.E                  # this module's experts: [e0, e0 + E) (expert parallel)
        self.gu, self.dn = gate_up, down
        self.act_fn = act_fn
        self._glu = _glu(act_fn)
        self.dtype = dtype                               # matmul input dtype: bf16, or fp16 on sm_70

    @staticmethod
    def stack_paths(gate_up_packfiles: dict, down_packfiles: dict, num_experts: int):
        """Every expert of a stack names the same .qvbq stack file -> (gate_up path, down path)."""
        def one(d, what):
            paths = {d[e] for e in range(num_experts) if e in d}
            if len(d) != num_experts or len(paths) != 1:
                raise ValueError(f"QVBQExperts: {what} must be one stack file for all "
                                 f"{num_experts} experts, got {len(paths)} files / {len(d)} experts")
            return paths.pop()
        return one(gate_up_packfiles, "gate_up"), one(down_packfiles, "down")

    @classmethod
    def from_packfiles(cls, gate_up_packfiles: dict, down_packfiles: dict, *, num_experts: int,
                       act_fn=None, device="cuda", tau=None, experts=None, dtype=torch.bfloat16):
        """experts=(e0, e1): this card holds only those experts (expert parallel; the caller sums the cards)."""
        gp, dp = cls.stack_paths(gate_up_packfiles, down_packfiles, num_experts)
        e0 = 0 if experts is None else experts[0]
        return cls(QVBQWeight.from_file(gp, device, experts), QVBQWeight.from_file(dp, device, experts),
                   num_experts=num_experts, act_fn=act_fn, e0=e0, dtype=dtype)

    def _out_dtype(self):
        return torch.float32 if self.dtype == torch.float16 else self.dtype   # fp16: matmul outputs kept fp32

    def _pairs(self, xs, e, w, gu, dn, E):
        """Rows xs [n, H] (one per token-expert pair), local expert e [n], route weight w [n] -> [n, H] fp32."""
        n = xs.shape[0]
        si, be = moe_align(e.view(-1, 1), E, 16)
        h = qvbq_matmul(xs, gu, si, be, n, 1, out_dtype=self._out_dtype())
        g, u = h.chunk(2, dim=-1)
        a = self._glu(g, u).to(self.dtype).contiguous()
        return qvbq_matmul(a, dn, si, be, n, 1, weights=w.float().contiguous(), x_is_pairs=True,
                           out_dtype=torch.float32)

    def _forward_small(self, x, top_k_index, top_k_weights):
        """sm < 80, few pairs (decode): the expert of each pair straight into kernel_v100 (no moe_align: no sort, no bincount)."""
        from .kernel_v100 import small_matmul
        T, H = x.shape
        k = top_k_index.shape[1]
        e = top_k_index.reshape(-1).to(torch.int32) - self.e0
        pe = torch.where((e >= 0) & (e < self.E), e, torch.full_like(e, -1)).contiguous()
        h = small_matmul(x, self.gu, pe, T * k, k, out_dtype=torch.float32)
        g, u = h.chunk(2, dim=-1)
        a = self._glu(g, u).to(self.dtype).contiguous()
        o = small_matmul(a, self.dn, pe, T * k, k, weights=top_k_weights.reshape(-1).float().contiguous(),
                         x_is_pairs=True, out_dtype=torch.float32)
        return o.view(T, k, H).sum(1)

    def _forward_dense(self, x, top_k_index, top_k_weights):
        """sm < 80, many pairs (prefill): per touched expert decode to dense fp16 (kernel_v100.dequant_expert), then the
        tensor-core mm for that expert's rows. Sums in fp32."""
        from .kernel_v100 import dequant_expert
        T, H = x.shape
        k = top_k_index.shape[1]
        e = top_k_index.reshape(-1) - self.e0
        ok = (e >= 0) & (e < self.E)
        flat = torch.where(ok, e, torch.full_like(e, self.E))
        order = torch.argsort(flat, stable=True)
        cnt = torch.bincount(flat, minlength=self.E + 1)[:self.E].tolist()
        wts = top_k_weights.reshape(-1).float()
        out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
        bg = torch.empty(self.gu.N, self.gu.K, dtype=x.dtype, device=x.device)
        bd = torch.empty(self.dn.N, self.dn.K, dtype=x.dtype, device=x.device)
        start = 0
        for ex, c in enumerate(cnt):
            if c == 0:
                continue
            rows = order[start:start + c]
            start += c
            tok = rows // k
            dequant_expert(self.gu, ex, bg)
            h = x[tok] @ bg.t()
            g, u = h.float().chunk(2, dim=-1)
            a = self._glu(g, u).to(x.dtype)
            dequant_expert(self.dn, ex, bd)
            o = (a @ bd.t()).float() * wts[rows][:, None]
            out.index_add_(0, tok, o)
        return out

    def forward(self, hidden_states, top_k_index, top_k_weights):
        T, H = hidden_states.shape
        k = top_k_index.shape[1]
        x = hidden_states.to(self.dtype).contiguous()
        if x.is_cuda and T * k <= int(os.environ.get("QV_QVBQ_SMALL_PAIRS", "64")):
            return self._forward_small(x, top_k_index, top_k_weights).to(hidden_states.dtype)
        if x.is_cuda and torch.cuda.get_device_capability(x.device) < (8, 0):
            return self._forward_dense(x, top_k_index, top_k_weights).to(hidden_states.dtype)
        if self.e0 == 0 and self.E == self.num_experts:  # every expert here: all pairs in one pass
            E = self.num_experts
            ids = top_k_index.clamp(min=0, max=E - 1)
            wts = torch.where((top_k_index >= 0) & (top_k_index < E), top_k_weights,
                              torch.zeros_like(top_k_weights))
            si, be = moe_align(ids, E, 16)
            h = qvbq_matmul(x, self.gu, si, be, T * k, k, out_dtype=self._out_dtype())
            g, u = h.chunk(2, dim=-1)
            a = self._glu(g, u).to(self.dtype).contiguous()
            o = qvbq_matmul(a, self.dn, si, be, T * k, k, weights=wts.reshape(-1).float().contiguous(),
                            x_is_pairs=True, out_dtype=self._out_dtype())
            return o.view(T, k, H).float().sum(1).to(hidden_states.dtype)
        # a range of the experts (expert parallel): only the pairs routed here; the others' share is 0 on this card
        e = top_k_index.reshape(-1) - self.e0
        sel = ((e >= 0) & (e < self.E)).nonzero().squeeze(1)
        out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
        if sel.numel():
            t = sel // k
            out.index_add_(0, t, self._pairs(x[t], e[sel], top_k_weights.reshape(-1)[sel], self.gu, self.dn, self.E))
        return out.to(hidden_states.dtype)


class QVBQHostExperts(QVBQExperts):
    """The expert stack in page-locked HOST RAM, the experts in use cached on the card (a card too small for them all:
    one 16 GB V100). Per call: the experts the router chose are brought into the cache (least recently used out);
    a call needing more experts than the cache holds (a long prefill chunk) runs them in groups. The matmul is the same
    kernel over the cache's slots, so the results are the resident stack's."""

    def __init__(self, gate_up: QVBQWeight, down: QVBQWeight, *, num_experts: int, slots: int, device,
                 act_fn=None, e0: int = 0, dtype=torch.bfloat16):
        super().__init__(gate_up.pin_(), down.pin_(), num_experts=num_experts, act_fn=act_fn, e0=e0, dtype=dtype)
        from collections import OrderedDict
        self.slots = max(1, min(int(slots), self.E))
        self.dev = torch.device(device)
        self.cgu, self.cdn = gate_up.empty_like(self.slots, self.dev), down.empty_like(self.slots, self.dev)
        self.slot_of: "OrderedDict[int, int]" = OrderedDict()            # local expert -> slot, LRU order
        self.free = list(range(self.slots))
        self.remap = torch.full((self.E,), -1, dtype=torch.long, device=self.dev)
        self.loads = 0

    @classmethod
    def from_packfiles(cls, gate_up_packfiles: dict, down_packfiles: dict, *, num_experts: int, slots: int,
                       act_fn=None, device="cuda", experts=None, dtype=torch.bfloat16, **_):
        gp, dp = cls.stack_paths(gate_up_packfiles, down_packfiles, num_experts)
        e0 = 0 if experts is None else experts[0]
        return cls(QVBQWeight.from_file(gp, "cpu", experts or (0, num_experts)),
                   QVBQWeight.from_file(dp, "cpu", experts or (0, num_experts)),
                   num_experts=num_experts, slots=slots, device=device, act_fn=act_fn, e0=e0, dtype=dtype)

    def cache_nbytes(self) -> int:
        return self.cgu.nbytes() + self.cdn.nbytes()

    def _bring(self, group) -> None:
        keep = set(group)
        for e in group:
            if e in self.slot_of:
                self.slot_of.move_to_end(e)
                continue
            if self.free:
                sl = self.free.pop()
            else:
                old = next(x for x in self.slot_of if x not in keep)        # least recently used, not needed now
                sl = self.slot_of.pop(old)
                self.remap[old] = -1
            self.cgu.copy_expert_(sl, self.gu, e)
            self.cdn.copy_expert_(sl, self.dn, e)
            self.slot_of[e] = sl
            self.remap[e] = sl
            self.loads += 1

    def forward(self, hidden_states, top_k_index, top_k_weights):
        T, H = hidden_states.shape
        k = top_k_index.shape[1]
        x = hidden_states.to(self.dtype).contiguous()
        e = top_k_index.reshape(-1) - self.e0
        w = top_k_weights.reshape(-1)
        ok = (e >= 0) & (e < self.E)
        need = torch.unique(e[ok]).tolist()
        out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
        cached = [x_ for x_ in need if x_ in self.slot_of]
        order = cached + [x_ for x_ in need if x_ not in self.slot_of]  # cached ones first: fewer copies
        for g0 in range(0, len(order), self.slots):
            group = order[g0:g0 + self.slots]
            self._bring(group)
            gt = torch.tensor(group, dtype=torch.long, device=x.device)
            sel = (ok & torch.isin(e, gt)).nonzero().squeeze(1)
            t = sel // k
            out.index_add_(0, t, self._pairs(x[t], self.remap[e[sel]], w[sel], self.cgu, self.cdn, self.slots))
        return out.to(hidden_states.dtype)


def decode_linear(lin) -> torch.Tensor:
    """FormatSpec.decoder: dense (N, K) fp32 of an QVBQLinear."""
    return lin.w.dense()[0]


class QVBQEmbedding(nn.Module):
    """nn.Embedding replacement: decodes only the looked-up rows (E=1, row-wise scale)."""

    def __init__(self, w: QVBQWeight, out_dtype=torch.bfloat16):
        super().__init__()
        assert w.E == 1
        self.w, self.out_dtype = w, out_dtype
        self.num_embeddings, self.embedding_dim = w.N, w.K

    @classmethod
    def from_packfile(cls, path: str, out_dtype=torch.bfloat16, device="cuda", **_):
        return cls(QVBQWeight.from_file(path, device), out_dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        w = self.w
        r = ids.reshape(-1).long()
        q = w.q[0].index_select(0, r)
        if w.code == CODE_E2M1:
            lut = torch.tensor(_E2M1, device=q.device)
            v = torch.stack((lut[(q & 15).long()], lut[(q >> 4).long()]), -1).reshape(len(r), w.K)
        else:
            v = q.view(torch.float8_e4m3fn if w.code == CODE_E4M3 else torch.float8_e5m2).float()
        s = w.s[0].index_select(0, r // w.GN)
        s = (s.view(torch.float8_e4m3fn).float() if w.scale_kind == SCALE_E4M3 else
             torch.exp2(s.float() - 127) if w.scale_kind == SCALE_E8M0 else s.float())
        v = v * s.repeat_interleave(w.GK, 1)[:, :w.K]
        if w.f is not None:
            v = v * w.f[0].index_select(0, r)[:, None]
        return v.to(self.out_dtype).reshape(*ids.shape, w.K)
