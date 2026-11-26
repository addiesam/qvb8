"""Inkling attention as one flash kernel: O(L) memory instead of HF eager's O(L^2).

HF eager builds the relative-position bias as a full [B, H, Lq, Lk] tensor (gather of
rel_logits over the distance matrix), adds it to a full score matrix, concatenates the
sink column and softmaxes -- ~4 * H * Lq * Lk * 4 bytes (22 GB per tensor at 15k tokens).
Here the bias stays compact, rel[B, H, Lq, E] (E = rel_extent), and is looked up inside
the kernel by distance:

    d(i, j) = d0 + i - j               d0 = q_offset - kv_offset (keys are contiguous positions)
    visible: d >= 0, and d < W on sliding layers
    s(i, j) = scale * q_i . k_j + (rel[i, d] if 0 <= d < E else 0)
    p = softmax over {s(i, j) : visible j} U {sink_i}  (sink has no value)
    out_i = sum_j p(i, j) v_j

Online softmax starts from m = sink, l = 1, so the sink only enters the denominator.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qv_load(id_ptr, mn_ptr, st_ptr, t_ptr, b, hk, rj, rd, jmask, Lc, sib, sih, sij, smb, smh, smk,
             stb, sth, stj, QBLK: tl.constexpr):
    """Keys / values [BN, D] for positions rj: from the 8-bit ids below Lc (min + id * step of their block), else from
    the dense tail at rj - Lc (engine/static_decode.StaticQKV layout)."""
    inq = (rj < Lc) & jmask
    ind = (rj >= Lc) & jmask
    ids = tl.load(id_ptr + b * sib + hk * sih + rj[:, None] * sij + rd[None, :], mask=inq[:, None], other=0)
    blk = rj // QBLK
    mn = tl.load(mn_ptr + b * smb + hk * smh + blk[:, None] * smk + rd[None, :], mask=inq[:, None], other=0.0)
    sp = tl.load(st_ptr + b * smb + hk * smh + blk[:, None] * smk + rd[None, :], mask=inq[:, None], other=0.0)
    tj = tl.where(ind, rj - Lc, 0)
    tv = tl.load(t_ptr + b * stb + hk * sth + tj[:, None] * stj + rd[None, :], mask=ind[:, None], other=0.0)
    return tl.where(inq[:, None], mn + ids.to(tl.float32) * sp, tv.to(tl.float32))


@triton.jit
def _attn_fwd(q_ptr, k_ptr, v_ptr, rel_ptr, sink_ptr, o_ptr,
              Lq, Lk, E, d0, W, scale, lk_ptr, d0_ptr, slot_ptr,
              kid_ptr, vid_ptr, kmin_ptr, kstep_ptr, vmin_ptr, vstep_ptr, lc_ptr, sib, sih, sij, smb, smh, smk,
              sqb, sqh, sqi, skb, skh, skj, svb, svh, svj,
              srb, srh, sri, ssb, ssh, sob, soi, soh,
              H: tl.constexpr, GROUPS: tl.constexpr, D: tl.constexpr,
              SLIDING: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DYN: tl.constexpr,
              QKV: tl.constexpr, QBLK: tl.constexpr, SLOT: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    if DYN:                                                # per-row key count / offset from device tensors (CUDA graphs)
        Lk = tl.load(lk_ptr + b)
        d0 = tl.load(d0_ptr + b)
    kb = b                                                 # the K / V batch row
    if SLOT:                                               # ragged rows: row b reads slot slot[b]'s keys
        kb = tl.load(slot_ptr + b)
    Lc = 0
    if QKV:                                                # qvb8 keys: [0, Lc) as 8-bit ids (min + id * step), then the tail
        Lc = tl.load(lc_ptr + kb)
    hk = h // GROUPS
    ri = pid_m * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    imask = ri < Lq
    q = tl.load(q_ptr + b * sqb + h * sqh + ri[:, None] * sqi + rd[None, :], mask=imask[:, None], other=0.0)
    sink = tl.load(sink_ptr + b * ssb + h * ssh + ri, mask=imask, other=0.0).to(tl.float32)
    m_i = tl.where(imask, sink, 0.0)
    l_i = tl.where(imask, 1.0, 1.0)
    acc = tl.zeros((BM, D), dtype=tl.float32)
    i_lo = pid_m * BM
    i_hi = tl.minimum(i_lo + BM, Lq) - 1
    j_hi = tl.minimum(Lk, d0 + i_hi + 1)                  # causal: j <= d0 + i
    j_lo = 0
    if SLIDING:
        j_lo = tl.maximum(0, d0 + i_lo - W + 1)            # window: j > d0 + i - W
    if not (SLIDING and DYN):                              # static window buffers: tiles from the window start, as when
        j_lo = (j_lo // BN) * BN                           # the window sat at index 0 (same tiles -> same sums)
    for j0 in range(j_lo, j_hi, BN):
        rj = j0 + tl.arange(0, BN)
        jmask = rj < Lk
        if QKV:
            k = _qv_load(kid_ptr, kmin_ptr, kstep_ptr, k_ptr, kb, hk, rj, rd, jmask, Lc, sib, sih, sij, smb, smh, smk,
                         skb, skh, skj, QBLK).to(q.dtype)
        else:
            k = tl.load(k_ptr + kb * skb + hk * skh + rj[:, None] * skj + rd[None, :], mask=jmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        d = d0 + ri[:, None] - rj[None, :]
        vis = (d >= 0) & jmask[None, :] & imask[:, None]
        if SLIDING:
            vis = vis & (d < W)
        inb = vis & (d < E)
        bias = tl.load(rel_ptr + b * srb + h * srh + ri[:, None] * sri + d, mask=inb, other=0.0).to(tl.float32)
        s = tl.where(vis, s + bias, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if QKV:
            v = _qv_load(vid_ptr, vmin_ptr, vstep_ptr, v_ptr, kb, hk, rj, rd, jmask, Lc, sib, sih, sij, smb, smh, smk,
                         svb, svh, svj, QBLK).to(q.dtype)
        else:
            v = tl.load(v_ptr + kb * svb + hk * svh + rj[:, None] * svj + rd[None, :], mask=jmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(o_ptr + b * sob + ri[:, None] * soi + h * soh + rd[None, :], out.to(o_ptr.dtype.element_ty),
             mask=imask[:, None])


def inkling_attention(q, k, v, rel, sinks, d0, scale: float, window: int | None, lk=None, qkv=None, slot=None):
    """q [B,H,Lq,D], k/v [B,Hk,Lk,D], rel [B,H,Lq,E], sinks [B,H,Lq] -> out [B,Lq,H,D].
    d0 an int (keys = contiguous positions from q_offset - d0), or with lk: d0 [B] and lk [B] int32 device tensors =
    per-row offset and number of valid keys (static decode buffers: nothing the host reads, CUDA-graph safe; a sliding
    layer's window is then the last `window` of the lk keys).
    qkv: an engine/static_decode.StaticQKV -- qvb8 keys / values read in the kernel (k / v are then its dense tails).
    slot [B] (with lk): ragged rows -- q row b attends slot slot[b]'s keys (a speculative verify: one row per token)."""
    B, H, Lq, D = q.shape
    Hk, Lk = k.shape[1], k.shape[2]
    if slot is not None:
        slot = slot.contiguous()
    E = rel.shape[-1]
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    rel = rel.contiguous()
    sinks = sinks.contiguous()
    o = torch.empty(B, Lq, H, D, dtype=q.dtype, device=q.device)
    BM = 64 if Lq > 16 else 16
    BN = 32 if qkv is not None else 64           # qvb8 keys: ids + min + step + tail tiles per step (shared memory)
    grid = (B * H, triton.cdiv(Lq, BM))
    dyn = lk is not None
    if qkv is not None:
        Q = qkv
        qargs = (Q.kid, Q.vid, Q.kmin, Q.kstep, Q.vmin, Q.vstep, Q.lc.contiguous(), Q.kid.stride(0), Q.kid.stride(1),
                 Q.kid.stride(2), Q.kmin.stride(0), Q.kmin.stride(2), Q.kmin.stride(1))   # (slot, head, block)
        Lk = Q.cap + Q.block
    else:
        qargs = (o, o, o, o, o, o, o, 0, 0, 0, 0, 0, 0)
    _attn_fwd[grid](q, k, v, rel, sinks, o, Lq, Lk, E, 0 if dyn else int(d0), int(window or 0), float(scale),
                    lk if dyn else o, d0 if dyn else o, slot if slot is not None else o, *qargs,
                    q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2), rel.stride(0), rel.stride(1), rel.stride(2),
                    sinks.stride(0), sinks.stride(1), o.stride(0), o.stride(1), o.stride(2),
                    H=H, GROUPS=H // Hk, D=D, SLIDING=window is not None, BM=BM, BN=BN, DYN=dyn,
                    QKV=qkv is not None, QBLK=qkv.block if qkv is not None else 1, SLOT=slot is not None,
                    num_warps=4, num_stages=2 if torch.cuda.get_device_capability(q.device)[0] >= 8 and qkv is None else 1)  # sm_70: 96 KB smem
    return o
