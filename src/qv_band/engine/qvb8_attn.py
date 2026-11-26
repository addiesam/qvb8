"""Attention over a qvb8 KV cache (engine/static_decode.StaticQKV) for ANY model: closed 8-bit blocks + dense tail.

    o, lse = qvb8_attention(q, cache, vis, scale)

q [S, Hq, T, D] (GQA: query head h reads kv head h // (Hq / Hkv)), vis [S, T] long = how many TAIL keys query t sees
(causal inside the tail; every closed block is older than every query, so fully visible). Returns o [S, Hq, T, D] fp32
(normalised) and lse [S, Hq, T] fp32, so a caller can merge more keys exactly (cold KV in host RAM, a prompt chunk).

The closed blocks go through one kernel split over the key axis, reading lc (keys held as ids) from the DEVICE, so the
call is CUDA-graph safe and costs what the CONTEXT costs, not the buffer:
  sm >= 80   the Triton kernel below (grid S*Hkv x P parts x row tiles)
  sm < 80    kv/cuda_attn's WMMA kernel (fwd_static: ids splits + the tail + the merge in two launches), since Triton has no tensor-core dot on Volta (measured on V100:
             the Triton kernel spilled 1-3 KB per thread -- 0.6-1.1 GB of driver local memory -- and ran ~1 ms per layer)
Each key tile lies inside one qvb8 block: k = min + id * step of that block, decoded on the fly.
The tail (<= block + extra keys) is a plain fp16 matmul. Parts and tail are merged by log-sum-exp.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _blk_attn(q_ptr, kid_ptr, vid_ptr, kmn_ptr, kst_ptr, vmn_ptr, vst_ptr, lc_ptr, o_ptr, l_ptr,
              R, scale, sqs, sqh, sqr, sis, sih, sij, sms, smh, smb, sos, soh, sop, sor, sls, slh, slp,
              HKV: tl.constexpr, RP: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, CH: tl.constexpr,
              QBLK: tl.constexpr):
    pid = tl.program_id(0)
    part = tl.program_id(1)
    rb = tl.program_id(2)                                                  # a tile of RP query rows
    s = pid // HKV
    h = pid % HKV
    Lc = tl.load(lc_ptr + s)
    lo = part * CH
    hi = tl.minimum(lo + CH, Lc)
    rr = rb * RP + tl.arange(0, RP)
    rd = tl.arange(0, D)
    rmask = rr < R
    q = tl.load(q_ptr + s * sqs + h * sqh + rr[:, None] * sqr + rd[None, :], mask=rmask[:, None], other=0.0)
    m_i = tl.full((RP,), float("-inf"), tl.float32)
    l_i = tl.zeros((RP,), tl.float32)
    acc = tl.zeros((RP, D), tl.float32)
    for j0 in range(lo, hi, BN):
        rj = j0 + tl.arange(0, BN)
        jm = rj < hi
        mo = s * sms + h * smh + (j0 // QBLK) * smb + rd                  # BN divides QBLK: one block per tile
        ids = tl.load(kid_ptr + s * sis + h * sih + rj[:, None] * sij + rd[None, :], mask=jm[:, None], other=0)
        k = (tl.load(kmn_ptr + mo)[None, :] + ids.to(tl.float32) * tl.load(kst_ptr + mo)[None, :]).to(tl.float16)
        sc = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        sc = tl.where(jm[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        ids = tl.load(vid_ptr + s * sis + h * sih + rj[:, None] * sij + rd[None, :], mask=jm[:, None], other=0)
        v = (tl.load(vmn_ptr + mo)[None, :] + ids.to(tl.float32) * tl.load(vst_ptr + mo)[None, :]).to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v).to(tl.float32)
        m_i = m_new
    ok = l_i > 0
    out = tl.where(ok[:, None], acc / tl.where(ok, l_i, 1.0)[:, None], 0.0)
    lse = tl.where(ok, m_i + tl.log(tl.where(ok, l_i, 1.0)), float("-inf"))
    tl.store(o_ptr + s * sos + h * soh + part * sop + rr[:, None] * sor + rd[None, :], out, mask=rmask[:, None])
    tl.store(l_ptr + s * sls + h * slh + part * slp + rr, lse, mask=rmask)


def _parts(cap: int, block: int) -> tuple[int, int]:
    """(P, CH): the key axis in P parts of CH keys (CH a multiple of the qvb8 block), at most 64 parts."""
    nb = cap // block
    per = max(1, -(-nb // 64))
    CH = per * block
    return -(-cap // CH), CH


def _closed_triton(q, cache, scale, T, G):
    """-> o [S, Hkv, P, R, D] (normalised), lse [S, Hkv, P, R]; rows r = t * G + g."""
    S, Hq, _, D = q.shape
    Hkv = Hq // G
    R = G * T
    qg = q.view(S, Hkv, G, T, D).transpose(2, 3).reshape(S, Hkv, R, D).to(torch.float16).contiguous()
    RP = min(max(16, triton.next_power_of_2(R)), 32)
    NR = -(-R // RP)
    P, CH = _parts(cache.cap, cache.block)
    o = torch.empty(S, Hkv, P, NR * RP, D, dtype=torch.float32, device=q.device)
    l = torch.empty(S, Hkv, P, NR * RP, dtype=torch.float32, device=q.device)
    BN = 32 if cache.block % 32 == 0 else cache.block
    km = cache.kmin                                                          # [S, nb, Hkv, D]
    _blk_attn[(S * Hkv, P, NR)](qg, cache.kid, cache.vid, cache.kmin, cache.kstep, cache.vmin, cache.vstep,
                                cache.lc.contiguous(), o, l, R, float(scale), qg.stride(0), qg.stride(1), qg.stride(2),
                                cache.kid.stride(0), cache.kid.stride(1), cache.kid.stride(2),
                                km.stride(0), km.stride(2), km.stride(1),
                                o.stride(0), o.stride(1), o.stride(2), o.stride(3), l.stride(0), l.stride(1), l.stride(2),
                                HKV=Hkv, RP=RP, D=D, BN=BN, CH=CH, QBLK=cache.block, num_warps=4, num_stages=1)
    return o[:, :, :, :R], l[:, :, :, :R]


def _wmma(ext, q, cache, vis, scale, T, G):
    """Volta: kv/cuda_attn fwd_static -- ids splits + the tail split + the merge, two launches, final o / lse."""
    S, Hq, _, D = q.shape
    Hkv = Hq // G
    R = G * T
    plan = getattr(cache, "_wmma_plan", None)
    if plan is None:                                                         # splits of the whole capacity (fixed)
        n_tiles = cache.cap // 32
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        want = max(1, min(n_tiles, 2 * sms // max(1, S * Hkv)))
        tps = -(-n_tiles // want)
        plan = cache._wmma_plan = (tps, -(-n_tiles // tps))
    tps, NSP = plan
    o = torch.empty(S, NSP + 1, Hkv, R, D, dtype=torch.float32, device=q.device)
    m = torch.empty(S, NSP + 1, Hkv, R, dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    out = torch.empty(S, Hq, T, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(S, Hq, T, dtype=torch.float32, device=q.device)
    ext.fwd_static(q, cache.kid, cache.vid, cache.kmin, cache.kstep, cache.vmin, cache.vstep, cache.kt, cache.vt,
                   cache.lc.contiguous(), vis.contiguous(), o, m, l, out, lse, tps, cache.block, float(scale))
    return out, lse


def _ext(device):
    if torch.cuda.get_device_capability(device)[0] >= 8:
        return None
    from qv_band.kv.fused_attn import _cuda_ext
    return _cuda_ext(device)


def qvb8_attention(q, cache, vis, scale: float):
    S, Hq, T, D = q.shape
    Hkv = cache.kid.shape[1]
    G = Hq // Hkv
    R = G * T
    ext = _ext(q.device)
    if ext is not None and D in (128, 256) and cache.block % 32 == 0 and cache.kt.dtype == torch.float16:
        q = q.to(torch.float16)
        if q.stride(3) != 1:
            q = q.contiguous()
        return _wmma(ext, q, cache, vis.long(), scale, T, G)
    o, l = _closed_triton(q.contiguous(), cache, scale, T, G)
    P = o.shape[2]
    # the dense tail: keys j < vis[s, t] of query t (rows r = t * G + g)
    qg = q.reshape(S, Hkv, G, T, D).transpose(2, 3).reshape(S, Hkv, R, D).to(cache.kt.dtype)   # q may be strided
    kt, vt = cache.kt, cache.vt                                             # [S, Hkv, Bt, D]
    st = torch.matmul(qg, kt.transpose(-1, -2)).float() * scale            # [S, Hkv, R, Bt]
    Bt = kt.shape[2]
    see = torch.arange(Bt, device=q.device).view(1, 1, Bt) < vis.view(S, T, 1).expand(S, T, G).reshape(S, R, 1)
    st = st.masked_fill(~see.unsqueeze(1), float("-inf"))
    mt = st.amax(-1, keepdim=True).clamp(min=-1e30)
    pt = torch.exp(st - mt)
    lt = pt.sum(-1, keepdim=True)
    ot = torch.matmul(pt.to(vt.dtype), vt).float() / lt.clamp(min=1e-30)  # [S, Hkv, R, D]
    lset = torch.where(lt[..., 0] > 0, mt[..., 0] + torch.log(lt[..., 0].clamp(min=1e-30)),
                       torch.full_like(mt[..., 0], float("-inf")))
    # merge the parts and the tail by log-sum-exp
    L = torch.cat([l, lset.unsqueeze(2)], 2)                               # [S, Hkv, P + 1, R]
    M = L.amax(2, keepdim=True).clamp(min=-1e30)
    w = torch.exp(L - M)
    den = w.sum(2)                                                          # [S, Hkv, R]
    out = ((w[:, :, :P, :, None] * o).sum(2) + w[:, :, P, :, None] * ot) / den.clamp(min=1e-30)[..., None]
    lse = M[:, :, 0] + torch.log(den.clamp(min=1e-30))
    out = out.view(S, Hkv, T, G, D).transpose(2, 3).reshape(S, Hq, T, D)
    lse = lse.view(S, Hkv, T, G).transpose(2, 3).reshape(S, Hq, T)
    return out, lse
