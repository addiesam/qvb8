"""QVB8 multi-row GEMM on V100 (sm_70) fp16 tensor cores, for 2..16 rows (DFlash verify, small batches).

`install()` routes QVBandLinearV8.forward for QVB8_WMMA_MIN_M (5) <= M <= 16 on cards older than sm_80 to qvb8_wmma_sm70.cu;
M = 1 and every other case keep the existing kernels. The weight is the certified fp16 restore
fp16(min_s + id * tau_s), the one the M>8 decode path and the sm_80 mma path multiply. On by default for every QVB8
user (LLM PP / TP, DiT, text encoders): measured on the V100, a DFlash verify (M=8) of a 17408x5120 MLP linear
0.536 ms -> 0.311 ms, attention 0.242 -> 0.155 ms; M=1 keeps the GEMV (faster there). QVB8_WMMA=0 turns it off.
"""
from __future__ import annotations

import os

import torch

from qv_band.formats.qvb8.module import QVBandLinearV8

_HERE = os.path.join(os.path.dirname(__file__), "backends", "cuda_qvb8")
_ORIG = QVBandLinearV8.forward
_SMS = {}
FACTOR = os.environ.get("QVB8_WMMA_FACTOR", "1") == "1"   # y = mu*sum(x) + tau*sum(id*x): exact integer weights
MIN_M = int(os.environ.get("QVB8_WMMA_MIN_M", "5"))   # V100, 27B half tables: GEMV wins at M <= 4, this kernel from 5
MAX_M = 128


def ext(device=None):
    from qv_band.common.jit_ext import load_cuda_ext
    return load_cuda_ext("qvb8_wmma_sm70", [os.path.join(_HERE, "qvb8_wmma_sm70.cu")], device=device)


def _split(dev, Q, P):
    sms = _SMS.get(dev)
    if sms is None:
        sms = _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    nblk = -(-Q // 64)
    # ~16 blocks per SM (split-K): measured on the V100, a 27B layer's 6 linears at M=8: 4/SM 1.53 ms, 16/SM 1.43 ms
    return max(1, min(P // 128, (16 * sms) // max(1, nblk)))


def wmma_forward(self, x):
    shape = x.shape
    x2d = x.reshape(-1, shape[-1])
    M = x2d.shape[0]
    dev = x2d.device
    if (not (MIN_M <= M <= MAX_M) or not x2d.is_cuda or self.fp16_table_missing
            or torch.cuda.get_device_capability(dev)[0] >= 8 or self.ids.data_ptr() % 4
            or (x2d.dtype == torch.bfloat16 and self.bf16_certified)):     # bf16 end to end: its own exact path
        return _ORIG(self, x)
    if M > 16:
        # a batched verify (several users x a draft block): passes of <= 16 rows. The old route decoded the whole
        # matrix to fp16 first -- a 1.2 GB temporary for a TP lm_head share, and about the same time
        # (measured: M=32 0.62 ms, 2 x M=16 0.64 ms on a 17408x5120 MLP linear)
        n = -(-M // 16)
        step = -(-M // n)
        return torch.cat([wmma_forward(self, x2d[i:i + step]) for i in range(0, M, step)]) \
            .reshape(*shape[:-1], self.out_features)
    P, Q = self.in_features, self.out_features
    xh = x2d if x2d.dtype == torch.float16 else x2d.float().clamp(-65504.0, 65504.0).to(torch.float16)
    y = ext(dev).qvb8_wmma(xh, self.ids, self.min_s, self.nbits, self.bit_off, self.tau_s,
                           self.esc_idx, self.esc_vals, P, _split(dev.index, Q, P), FACTOR)
    if self.bias is not None:
        y = y + self.bias.float()
    return y.to(x.dtype).reshape(*shape[:-1], Q)


def install():
    QVBandLinearV8.forward = wmma_forward


def uninstall():
    QVBandLinearV8.forward = _ORIG


if os.environ.get("QVB8_WMMA", "1") == "1":
    install()
