"""Fused pointwise kernels for the Qwen3.5 / 3.6 / 3.8 decoder: RMSNorm, gated
RMSNorm and SwiGLU, one kernel launch each instead of 8-10 eager ops.

Why: profiled on the 27B, a 4096-token prefill chunk spends about a third of its
time in eager elementwise ops (fp32 upcasts, pow/mean/mul/type_as, silu*mul,
casts) that are memory-bound and launch-bound. These kernels do the same
arithmetic in the same order and the same intermediate roundings as the eager
code (including the bf16 roundings in the gated norm and SwiGLU), reading and
writing each tensor once.

install() replaces `forward` on the transformers classes only after checking that
the class's source (as an AST, so comments do not matter) is exactly the one this
file reproduces; anything else keeps running eagerly.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import torch
import triton
import triton.language as tl

_MAX_N = 16384            # widest row a single program normalises (fp32 in registers)


@triton.jit
def _rmsnorm_k(X, W, Y, n_rows, N, eps, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_N)
    m = (rows[:, None] < n_rows) & (cols[None, :] < N)
    off = rows[:, None] * N + cols[None, :]
    x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / N + eps)
    w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
    y = x * inv[:, None] * (1.0 + w)[None, :]            # (x * rsqrt) * (1 + w), as eager
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=m)


@triton.jit
def _rmsnorm_gated_k(X, Z, W, Y, n_rows, N, eps, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
                     ROUND_B: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_N)
    m = (rows[:, None] < n_rows) & (cols[None, :] < N)
    off = rows[:, None] * N + cols[None, :]
    x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / N + eps)
    a = (x * inv[:, None]).to(Y.dtype.element_ty).to(tl.float32)     # hidden.to(input_dtype)
    w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
    b = w[None, :] * a
    if ROUND_B:                                                      # weight in the input dtype: bf16 * bf16 -> bf16
        b = b.to(Y.dtype.element_ty).to(tl.float32)
    z = tl.load(Z + off, mask=m, other=0.0).to(tl.float32)
    y = b * (z / (1.0 + tl.exp(-z)))
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=m)


@triton.jit
def _swiglu_k(G, U, Y, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    g = tl.load(G + i, mask=m, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(Y.dtype.element_ty).to(tl.float32)   # silu() returns the input dtype
    u = tl.load(U + i, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + i, (s * u).to(Y.dtype.element_ty), mask=m)


_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _on_gpu(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in _DTYPES and not x.requires_grad and x.numel() > 0


def _usable(x: torch.Tensor) -> bool:
    """For the row-wise norms, which hold a whole row in registers."""
    return _on_gpu(x) and x.shape[-1] <= _MAX_N


def _tile(N: int) -> tuple[int, int, int]:
    bn = triton.next_power_of_2(N)
    br = max(1, 8192 // bn)
    return bn, br, (4 if bn * br <= 8192 else 8)


def rmsnorm_zero_centered(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    N = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    y = torch.empty_like(x2)
    bn, br, nw = _tile(N)
    with torch.cuda.device(x2.device):   # triton launches on the CURRENT card: a layer on another card must set it
        _rmsnorm_k[(triton.cdiv(x2.shape[0], br),)](x2, weight, y, x2.shape[0], N, eps,
                                                    BLOCK_N=bn, BLOCK_R=br, num_warps=nw)
    return y.reshape(x.shape)


def rmsnorm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    N = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    z2 = gate.reshape(-1, N).contiguous()
    y = torch.empty_like(x2)
    bn, br, nw = _tile(N)
    with torch.cuda.device(x2.device):
        _rmsnorm_gated_k[(triton.cdiv(x2.shape[0], br),)](
            x2, z2, weight, y, x2.shape[0], N, eps, BLOCK_N=bn, BLOCK_R=br,
            ROUND_B=(weight.dtype == x.dtype), num_warps=nw)
    return y.reshape(x.shape)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    g, u = gate.contiguous(), up.contiguous()
    y = torch.empty_like(g)
    n = g.numel()
    with torch.cuda.device(g.device):
        _swiglu_k[(triton.cdiv(n, 4096),)](g, u, y, n, BLOCK=4096, num_warps=8)
    return y


# ---------------------------------------------------------------------------
# Patching the transformers classes
# ---------------------------------------------------------------------------
_REF_RMSNORM_FORWARD = """
def forward(self, x):
    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)
"""
_REF_RMSNORM_NORM = """
def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
"""
_REF_GATED_FORWARD = """
def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    hidden_states = self.weight * hidden_states.to(input_dtype)
    hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))
    return hidden_states.to(input_dtype)
"""
_REF_MLP_FORWARD = """
def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
"""


def _ast_of(fn_or_src) -> str:
    src = fn_or_src if isinstance(fn_or_src, str) else inspect.getsource(fn_or_src)
    return ast.dump(ast.parse(textwrap.dedent(src)))


def _is_ref(cls, name: str, ref: str) -> bool:
    fn = cls.__dict__.get(name)
    return fn is not None and _ast_of(fn) == _ast_of(ref)


_PATCHED: set = set()


def _patch_rmsnorm(cls) -> bool:
    if not (_is_ref(cls, "forward", _REF_RMSNORM_FORWARD) and _is_ref(cls, "_norm", _REF_RMSNORM_NORM)):
        return False
    orig = cls.forward

    def forward(self, x):
        if not _usable(x) or self.weight.dtype not in _DTYPES:
            return orig(self, x)
        return rmsnorm_zero_centered(x, self.weight, self.eps)
    cls.forward = forward
    return True


def _patch_gated(cls) -> bool:
    if not _is_ref(cls, "forward", _REF_GATED_FORWARD):
        return False
    orig = cls.forward

    def forward(self, hidden_states, gate):
        if (not _usable(hidden_states) or gate.shape != hidden_states.shape
                or gate.dtype != hidden_states.dtype or self.activation != "silu"
                or self.weight.dtype not in (hidden_states.dtype, torch.float32)):
            return orig(self, hidden_states, gate)
        return rmsnorm_gated(hidden_states, gate, self.weight, self.variance_epsilon)
    cls.forward = forward
    return True


def _patch_mlp(cls) -> bool:
    if not _is_ref(cls, "forward", _REF_MLP_FORWARD):
        return False
    orig = cls.forward

    def forward(self, x):
        if getattr(self.config, "hidden_act", None) != "silu" or not _on_gpu(x):
            return orig(self, x)
        gate, up = self.gate_proj(x), self.up_proj(x)
        # elementwise: no limit on the intermediate width (17408 on the 27B)
        if gate.dtype != up.dtype or gate.shape != up.shape or not _on_gpu(gate):
            return self.down_proj(self.act_fn(gate) * up)
        return self.down_proj(swiglu(gate, up))
    cls.forward = forward
    return True


def install() -> int:
    """Patch every matching Qwen3.5-family class once. Returns how many were
    replaced (0 when transformers' code is not the version this reproduces)."""
    import importlib
    n = 0
    for mod in ("qwen3_5", "qwen3_5_moe"):
        try:
            m = importlib.import_module(f"transformers.models.{mod}.modeling_{mod}")
        except Exception:
            continue
        for name, cls in vars(m).items():
            if not inspect.isclass(cls) or cls.__module__ != m.__name__ or cls in _PATCHED:
                continue
            if name.endswith("VisionMLP"):
                continue
            ok = False
            if name.endswith("RMSNormGated"):
                ok = _patch_gated(cls)
            elif name.endswith("RMSNorm"):
                ok = _patch_rmsnorm(cls)
            elif name.endswith("MLP"):
                ok = _patch_mlp(cls)
            if ok:
                _PATCHED.add(cls)
                n += 1
    return n
