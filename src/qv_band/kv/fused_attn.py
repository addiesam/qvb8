"""Fused qvb8-KV decode attention: reads the packed ids directly, no dense copy.

The codec is per-channel affine, x[t, c] = min[c] + id[t, c] * step[c] with one
(min, step) pair per (segment, kv head). Attention never needs x itself:

    q . k_t      = sum_c (q_c * step_c) * id[t, c]  +  q . min
    sum_t p_t v_t = step * (p @ id_v)  +  min * sum_t p_t

so each tile is one integer->half convert and a dot on the raw ids; the dequant
FMA is folded into the (tiny) query/accumulator side. Decode is memory-bound,
so reading 1 byte/element instead of a 2-byte bf16 shadow halves the traffic
AND the shadow buffer (2 bytes/elem) is not needed at all.

Scope: any query length -- decode, verify and chunked prefill (queries are
tiled in row blocks, keys in BLOCK_N tiles: flash-attention style, so no
O(context) or O(q x context) tensor is ever built). Heads that needed > 8 bits are not
representable here (layout_from_segments returns None -> caller keeps the dense
path). The dense residual tail is attended separately and merged by LSE.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

BLOCK_N = 32


@triton.jit
def _fwd_split(Q, KI, VI, KC, VC, KMIN, KSTEP, VMIN, VSTEP,
               T_START, T_LEN, T_SEG,
               O, M_OUT, L_OUT,
               q_pos0, n_tiles, tiles_per_split, scale, n_rows,
               stride_ki_n, stride_ki_h, stride_qr, stride_qh,
               G: tl.constexpr, BLOCK_M: tl.constexpr,
               D: tl.constexpr, BLOCK_N: tl.constexpr, NHKV: tl.constexpr):
    # grid (kv head, split, query-row block). Query rows of one kv head are
    # r = i * G + g (token i, group member g); a block holds BLOCK_M of them,
    # so any query length works -- decode, verify AND prefill chunks. The
    # per-segment affine is applied to q on the fly (no per-segment q copy).
    h = tl.program_id(0)
    sp = tl.program_id(1)
    rb = tl.program_id(2)
    rows = rb * BLOCK_M + tl.arange(0, BLOCK_M)
    qi = rows // G
    qg = rows % G
    row_ok = rows < n_rows
    d = tl.arange(0, D)
    qpos = q_pos0 + qi
    qmax = q_pos0 + (tl.minimum(rb * BLOCK_M + BLOCK_M, n_rows) - 1) // G

    q = tl.load(Q + qi[:, None] * stride_qr + (h * G + qg)[:, None] * stride_qh + d[None, :],
                mask=row_ok[:, None], other=0.0)                      # model dtype

    qf = q.to(tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    prev = -1
    vmin = tl.zeros([D], tl.float32)
    vstep = tl.zeros([D], tl.float32)
    vc = 0.0
    qs_hi = tl.zeros([BLOCK_M, D], tl.float16)
    qs_lo = tl.zeros([BLOCK_M, D], tl.float16)
    qb = tl.zeros([BLOCK_M], tl.float32)

    t0 = sp * tiles_per_split
    t1 = tl.minimum(t0 + tiles_per_split, n_tiles)
    for t in range(t0, t1):
        start = tl.load(T_START + t)
        if start <= qmax:                                   # causal: tile visible at all
            nlen = tl.load(T_LEN + t)
            seg = tl.load(T_SEG + t)
            if seg != prev:                              # new segment: rescale q once
                pidx = (seg * NHKV + h) * D + d
                kstep = tl.load(KSTEP + pidx)
                kmid = tl.load(KMIN + pidx)
                kc = tl.load(KC + seg * NHKV + h)
                vmin = tl.load(VMIN + pidx)
                vstep = tl.load(VSTEP + pidx)
                vc = tl.load(VC + seg * NHKV + h)
                # K = kmid + kstep * (id - kc): q.k = (q*kstep).id + q.kmid - kc*sum(q*kstep)
                qs32 = qf * kstep[None, :]
                qs_hi = qs32.to(tl.float16)
                qs_lo = (qs32 - qs_hi.to(tl.float32)).to(tl.float16)   # hi + lo ~ fp32
                qb = tl.sum(qf * kmid[None, :], axis=1) - kc * tl.sum(qs32, axis=1)
                prev = seg

            n = tl.arange(0, BLOCK_N)
            n_ok = n < nlen
            kp = KI + (start + n)[:, None] * stride_ki_n + h * stride_ki_h + d[None, :]
            k = tl.load(kp, mask=n_ok[:, None], other=0).to(tl.float16)   # raw ids
            kt = tl.trans(k)
            s = tl.dot(qs_hi, kt, out_dtype=tl.float32) + tl.dot(qs_lo, kt, out_dtype=tl.float32)
            s = (s + qb[:, None]) * scale
            kpos = start + n
            ok = n_ok[None, :] & (kpos[None, :] <= qpos[:, None]) & row_ok[:, None]
            s = tl.where(ok, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)

            vp = VI + (start + n)[:, None] * stride_ki_n + h * stride_ki_h + d[None, :]
            v = tl.load(vp, mask=n_ok[:, None], other=0).to(tl.float16)
            # V = vmid + vstep * (id - vc): centre on the accumulator side. Both terms use
            # the SAME fp16-rounded p, so its rounding cancels instead of being amplified.
            p16 = p.to(tl.float16)
            pv = tl.dot(p16, v, out_dtype=tl.float32)
            ps = tl.sum(p16.to(tl.float32), axis=1)
            acc = acc * alpha[:, None] + (pv - vc * ps[:, None]) * vstep[None, :] + ps[:, None] * vmin[None, :]
            m_i = m_new

    base = (sp * NHKV + h) * n_rows
    o_ptr = O + (base + rows)[:, None] * D + d[None, :]
    tl.store(o_ptr, acc, mask=row_ok[:, None])
    tl.store(M_OUT + base + rows, m_i, mask=row_ok)
    tl.store(L_OUT + base + rows, l_i, mask=row_ok)


class QVLayout:
    """uint8 ids for one slot/layer + per-segment affine params + tile table.

    Built once when segments are appended; a decode step only needs
    `attend`. `ki`/`vi`: (cap, Hkv, D) uint8; params: (nseg, Hkv, D) fp32.
    """

    def __init__(self, ki, vi, kmin, kstep, vmin, vstep, seg_len, device, block_n=BLOCK_N,
                 kc=None, vc=None):
        self.block_n = block_n
        n_seg = len(seg_len)
        # kmin / vmin hold the range CENTRE when kc / vc are given (see _centre)
        self.kc = kc if kc is not None else torch.zeros(n_seg, kmin.shape[1], dtype=torch.float32, device=device)
        self.vc = vc if vc is not None else torch.zeros(n_seg, kmin.shape[1], dtype=torch.float32, device=device)
        self.ki, self.vi = ki, vi
        self.kmin, self.kstep, self.vmin, self.vstep = kmin, kstep, vmin, vstep
        self.n = int(sum(seg_len))
        starts, lens, segs, pos = [], [], [], 0
        for si, sl in enumerate(seg_len):
            for a in range(0, sl, block_n):
                starts.append(pos + a)
                lens.append(min(block_n, sl - a))
                segs.append(si)
            pos += sl
        self.t_start = torch.tensor(starts, dtype=torch.int32, device=device)
        self.t_len = torch.tensor(lens, dtype=torch.int32, device=device)
        self.t_seg = torch.tensor(segs, dtype=torch.int32, device=device)
        self.n_tiles = len(starts)


def layout_from_segments(segs_k, segs_v, seg_len, device):
    """segs_*: per segment, a list (one BandedKV per kv head). None if any head
    uses more than 8 bits (ids would not fit a byte)."""
    from .codec import _unpack_dev
    nseg, h = len(segs_k), len(segs_k[0])
    n = sum(seg_len)
    d = segs_k[0][0].orig_shape[1]
    ki = torch.empty(n, h, d, dtype=torch.uint8, device=device)
    vi = torch.empty_like(ki)
    P = [torch.empty(nseg, h, d, dtype=torch.float32, device=device) for _ in range(4)]
    KC = torch.zeros(nseg, h, dtype=torch.float32, device=device)
    VC = torch.zeros_like(KC)
    pos = 0
    for si in range(nseg):
        sl = seg_len[si]
        for hi in range(h):
            for store, ids_t, mn, st in ((segs_k, ki, P[0], P[1]), (segs_v, vi, P[2], P[3])):
                bk = store[si][hi]
                if bk.bits > 8:
                    return None
                ids = _unpack_dev(bk.packed.to(device), bk.bits, sl * d).reshape(sl, d)
                ids_t[pos:pos + sl, hi] = ids.to(torch.uint8)
                c = ((1 << bk.bits) - 1) / 2.0
                mn[si, hi] = (bk.min_s + bk.tau_s * c).to(device)   # range centre
                st[si, hi] = bk.tau_s.to(device)
                (KC if store is segs_k else VC)[si, hi] = c
        pos += sl
    return QVLayout(ki, vi, P[0], P[1], P[2], P[3], list(seg_len), device, kc=KC, vc=VC)


_EXT = {}
import os as _os
_KV_HOST = _os.environ.get("QV_KV_HOST", "0") == "1"


def _cuda_ext(device):
    """The WMMA kernel (cuda_attn/qvattn_cuda.cu) for GPUs before sm_80, where
    Triton's tl.dot has no tensor-core path and runs ~50x slower. None when the
    extension cannot be built here or QV_ATTN_CUDA=0."""
    import os
    idx = torch.device(device).index
    if idx is None:                                      # "cuda" = the current card, not card 0
        idx = torch.cuda.current_device()
    if idx not in _EXT:
        ext = None
        want = os.environ.get("QV_ATTN_CUDA", "auto")
        if want != "0" and (want == "1" or torch.cuda.get_device_capability(idx)[0] < 8):
            from qv_band.common.jit_ext import load_cuda_ext
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_attn")
            try:                                   # built on first use for this GPU (common/jit_ext.py)
                ext = load_cuda_ext("qvattn_cuda", [os.path.join(d, "qvattn_cuda.cu")], cflags=["-O3"],
                                    cuda_cflags=["-O3", "--use_fast_math"], device=idx)
            except Exception as e:                 # no usable toolkit: the Triton path still works, only slower
                print(f"[fused_attn] qvattn_cuda unavailable ({e}); using Triton", flush=True)
        _EXT[idx] = ext
    return _EXT[idx]


def attend(q, lay: QVLayout, q_pos0: int, scale: float, n_splits: int | None = None,
           num_warps: int = 4, num_stages: int = 1, block_m: int | None = None):
    """q: (MQ, Hq, D) bf16/fp16 queries at absolute positions q_pos0..q_pos0+MQ-1.
    Returns (o (MQ, Hq, D) fp32, lse (MQ, Hq) fp32) over the banded keys only
    (key position <= query position); rows with no visible key get lse = -inf.
    Memory is O(MQ * Hq * D * n_splits): nothing grows with the context."""
    mq, hq, d = q.shape
    hkv = lay.ki.shape[1]
    g = hq // hkv
    rows = mq * g
    if block_m is None:
        block_m = 16 if rows <= 16 else 32
    n_rb = triton.cdiv(rows, block_m)
    if n_splits is None:
        # enough programs to fill the GPU; long query blocks need fewer splits
        want = 4 * torch.cuda.get_device_properties(q.device).multi_processor_count
        n_splits = max(1, min(lay.n_tiles, want // (hkv * n_rb)))
    tps = math.ceil(lay.n_tiles / n_splits)
    n_splits = math.ceil(lay.n_tiles / tps)
    q = q.contiguous()
    kscale = scale
    # The WMMA kernel serves head_dim 128 / 256 with 32-key tiles; anything else stays on the Triton kernel.
    ext = _cuda_ext(q.device) if d in (128, 256) and lay.block_n == 32 else None
    if ext is not None:
        block_m = 64                                     # fixed by the CUDA kernel
        n_rb = triton.cdiv(rows, block_m)
        want = 2 * torch.cuda.get_device_properties(q.device).multi_processor_count
        n_splits = max(1, min(lay.n_tiles, want // (hkv * n_rb)))
        tps = math.ceil(lay.n_tiles / n_splits)
        n_splits = math.ceil(lay.n_tiles / tps)
        # fp16 queries (no bf16 on sm_70): fold the softmax scale in first (exact maths, |q| shrinks by sqrt(d))
        # and saturate, so an outlier channel cannot become inf -> a NaN row.
        q = (q.float() * scale).clamp_(-65504.0, 65504.0).to(torch.float16)
        kscale = 1.0
    o = torch.empty(n_splits, hkv, rows, d, dtype=torch.float32, device=q.device)
    m = torch.empty(n_splits, hkv, rows, dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    if ext is not None:
        ext.fwd(q, lay.ki, lay.vi, lay.kmin, lay.kstep, lay.vmin, lay.vstep,
                lay.kc.contiguous(), lay.vc.contiguous(), lay.t_start, lay.t_len, lay.t_seg,
                o, m, l, q_pos0, tps, float(kscale), g)
    else:
        _fwd_split[(hkv, n_splits, n_rb)](
            q, lay.ki, lay.vi, lay.kc, lay.vc, lay.kmin, lay.kstep, lay.vmin, lay.vstep,
            lay.t_start, lay.t_len, lay.t_seg, o, m, l,
            q_pos0, lay.n_tiles, tps, scale, rows,
            lay.ki.stride(0), lay.ki.stride(1), q.stride(0), q.stride(1),
            G=g, BLOCK_M=block_m, D=d, BLOCK_N=lay.block_n, NHKV=hkv,
            num_warps=num_warps, num_stages=num_stages)
    if n_splits == 1:
        lsum, mx_s = l[0], m[0]
        out = o[0] / lsum.clamp_min(1e-30)[..., None]
        mx_s = torch.where(torch.isinf(mx_s), torch.zeros_like(mx_s), mx_s)
    else:                                                # combine splits
        mx = m.max(dim=0).values
        mx_s = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
        w = torch.exp(m - mx_s)                          # (S, H, R)
        lsum = (l * w).sum(0)
        out = (o * w[..., None]).sum(0) / lsum.clamp_min(1e-30)[..., None]
    lse = torch.where(lsum > 0, mx_s + torch.log(lsum.clamp_min(1e-30)), torch.full_like(lsum, float("-inf")))
    out = out.reshape(hkv, mq, g, d).permute(1, 0, 2, 3).reshape(mq, hq, d)
    lse = lse.reshape(hkv, mq, g).permute(1, 0, 2).reshape(mq, hq)
    return out, lse


# ---------------------------------------------------------------------------
# Integration with BandedKVLayer / transformers attention.
#
#   BandedKVLayer.update()  -> FusedKV.sync(layer): keep a uint8 id buffer that
#                              follows the layer's banded segments (only NEW or
#                              replaced segments are unpacked, as in
#                              ragged_kv_views), and register the layer as
#                              "pending" for the attention call that follows.
#   attention wrapper       -> banded part with the kernel, dense residual with
#                              a few torch ops, merged by log-sum-exp.
# No O(context) tensor is built per step.
# ---------------------------------------------------------------------------
import threading

_TLS = threading.local()


# Option B tiering: only the newest QV_KV_GPU_WINDOW tokens of each layer's ids
# stay in VRAM; older ones move to pinned host RAM in QV_KV_COLD_BLOCK-token
# blocks and are streamed back (double-buffered on a copy stream) for every
# attention call. Each part is attended exactly and merged by log-sum-exp, so
# the result equals the all-GPU path; only PCIe time is added. 0 = all on GPU.
# "auto": all on GPU while the card has room; when a layer's id buffer must grow and the card would drop below
# QV_KV_GPU_RESERVE_MB free, that layer's window is frozen at its current size and from then on only its oldest
# blocks move to host RAM (offload only past the VRAM limit).
_WIN_ENV = (_os.environ.get("QV_KV_GPU_WINDOW", "0") or "0").strip().lower()
_AUTO_WIN = _WIN_ENV == "auto"
_GPU_WIN = 0 if _AUTO_WIN else int(_WIN_ENV)
_RESERVE = int(_os.environ.get("QV_KV_GPU_RESERVE_MB", "2048")) << 20


def _vram_room(dev, nbytes: int) -> bool:
    """True if allocating nbytes more on dev still leaves _RESERVE free (counting the allocator's cached blocks)."""
    free, _ = torch.cuda.mem_get_info(dev)
    free += torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
    return free - nbytes >= _RESERVE
_COLD_BLOCK = int(_os.environ.get("QV_KV_COLD_BLOCK", "4096") or 4096)
_HOST_LAG = 1024       # keep the newest packed segments on the GPU (rollback reads them)
_HOST_BATCH = 4096     # move packed copies to host this many tokens at a time


def _tile_table(seg_len, a, b, dev, block_n=BLOCK_N):
    """Tiles covering absolute positions [a, b), t_start relative to a; a tile
    never straddles a segment (it carries one segment's affine params)."""
    starts, lens, segs, pos = [], [], [], 0
    for si, sl in enumerate(seg_len):
        s0, s1 = max(pos, a), min(pos + sl, b)
        for x in range(s0, s1, block_n):
            starts.append(x - a)
            lens.append(min(block_n, s1 - x))
            segs.append(si)
        pos += sl
        if pos >= b:
            break
    # one pinned host->device copy, no stream sync (a pageable copy waits for
    # the card's whole queue and stalls the CPU that should be queueing ahead)
    from qv_band.common.xfer import h2d_small
    allv = h2d_small(torch.tensor(starts + lens + segs, dtype=torch.int32), dev)
    n = len(starts)
    return allv[:n], allv[n:2 * n], allv[2 * n:], n


def _mk_lay(ki, vi, P, C, ns, tt):
    lay = QVLayout.__new__(QVLayout)
    lay.block_n = BLOCK_N
    lay.ki, lay.vi = ki, vi
    lay.kmin, lay.kstep, lay.vmin, lay.vstep = (x[:ns] for x in P)
    lay.kc, lay.vc = C[0][:ns], C[1][:ns]
    lay.t_start, lay.t_len, lay.t_seg, lay.n_tiles = tt
    return lay


_STAGE = {}       # (device, dtype) -> (copy stream, 2 k slots, 2 v slots)


def _staging(dev, dt, h, d):
    key = (str(dev), dt, h, d)            # the buffers' shape is (COLD_BLOCK, h, d): part of the key
    if key not in _STAGE:
        mk = lambda: [torch.empty(_COLD_BLOCK, h, d, dtype=dt, device=dev) for _ in range(2)]
        _STAGE[key] = (torch.cuda.Stream(device=dev), mk(), mk())
    return _STAGE[key]


class FusedKV:
    """Per-layer (slot 0) id view of the banded segments: GPU window + host tier."""

    def __init__(self):
        self.segk, self.segv = [], []
        self.ki = self.vi = None      # GPU window: absolute positions [base, valid)
        self.cap = 0
        self.base = 0
        self.valid = 0
        self.cold = []                # (a, b, host k, host v, tile table) blocks below base
        self.P = None                 # kmid, kstep, vmid, vstep : (cap_seg, H, D)
        self.C = None                 # kc, vc : (cap_seg, H) id centres
        self.cap_seg = 0
        self.lay = None
        self.bad = False
        self.win = _GPU_WIN           # GPU window, tokens (0 = unlimited); "auto" sets it when VRAM runs short

    def _grow(self, need_tok, need_seg, h, d, dev, seg_len=0):
        """need_tok: window length (from base) that must fit. seg_len: the segment being appended (a prefill chunk): the
        id buffer needs GPU window + cold block + that many spare rows before eviction can make room for it."""
        if self.ki is None or need_tok > self.cap:
            cap = max(need_tok + need_tok // 4, need_tok + 4096)
            if self.win:
                cap = max(need_tok, min(cap, self.win + _COLD_BLOCK + seg_len))
            dt = self.ki.dtype if self.ki is not None else torch.uint8
            ki = torch.empty(cap, h, d, dtype=dt, device=dev)
            vi = torch.empty_like(ki)
            n = self.valid - self.base
            if self.ki is not None and n:
                ki[:n] = self.ki[:n]
                vi[:n] = self.vi[:n]
            self.ki, self.vi, self.cap = ki, vi, cap
        if self.P is None or need_seg > self.cap_seg:
            cs = max(need_seg + need_seg // 2, need_seg + 64)
            P = [torch.empty(cs, h, d, dtype=torch.float32, device=dev) for _ in range(4)]
            C = [torch.zeros(cs, h, dtype=torch.float32, device=dev) for _ in range(2)]
            if self.P is not None:
                for a, b in zip(P + C, self.P + self.C):
                    a[:self.cap_seg] = b
            self.P, self.C, self.cap_seg = P, C, cs

    _host_moves = ()
    _host_ev = None

    def _finish_host_moves(self):
        """Swap in the pinned host copies issued by the previous sync (long
        finished by now: waiting on its event costs nothing)."""
        if self._host_moves:
            if self._host_ev is not None:
                self._host_ev.synchronize()
            for bk, pin in self._host_moves:
                bk.packed = pin
        self._host_moves = []
        self._host_ev = None

    _host_upto = 0                  # segments [0, _host_upto) already queued for host

    def _queue_host_moves(self, sk, sv, lens, dev):
        """Packed copies -> host RAM for segments older than _HOST_LAG tokens,
        batched: one pinned buffer and async copies per batch (attention reads
        only the id buffer; the packed copy serves rollback / fallback / prefix
        save). Swapped in on the next sync, when the copies are long done."""
        if self._host_upto > len(sk):
            self._host_upto = 0
        end = sum(lens) - _HOST_LAG
        pos, j = sum(lens[:self._host_upto]), self._host_upto
        while j < len(sk) and pos + lens[j] <= end:
            pos += lens[j]
            j += 1
        if j - self._host_upto < 1 or pos - sum(lens[:self._host_upto]) < _HOST_BATCH:
            return
        items = [bk for si in range(self._host_upto, j) for st in (sk, sv) for bk in st[si]
                 if bk.packed.is_cuda]
        self._host_upto = j
        if not items:
            return
        tot = sum(bk.packed.numel() * bk.packed.element_size() for bk in items)
        pin = torch.empty(tot, dtype=torch.uint8, pin_memory=True)
        off = 0
        for bk in items:
            nb = bk.packed.numel() * bk.packed.element_size()
            dst = pin[off:off + nb].view(bk.packed.dtype).view(bk.packed.shape)
            dst.copy_(bk.packed, non_blocking=True)
            self._host_moves.append((bk, dst))
            off += nb
        self._host_ev = torch.cuda.Event()
        self._host_ev.record(torch.cuda.current_stream(dev))

    def _widen(self):
        """Switch the id buffers to 2 bytes (int16 holding uint16 bit patterns)."""
        self.ki = self.ki.to(torch.int16).bitwise_and_(0xFF)
        self.vi = self.vi.to(torch.int16).bitwise_and_(0xFF)
        self.cold = [(a, b, hk.to(torch.int16).pin_memory(), hv.to(torch.int16).pin_memory(), tt)
                     for a, b, hk, hv, tt in self.cold]
        print(f"[fused_attn] a head needs > 8 bits: this layer's ids widened to 16 bit", flush=True)

    def _evict(self, lens, dev):
        """Move the oldest window blocks to pinned host RAM."""
        while self.win and self.valid - self.base > self.win + _COLD_BLOCK:
            B = _COLD_BLOCK
            hk = torch.empty((B,) + tuple(self.ki.shape[1:]), dtype=self.ki.dtype, pin_memory=True)
            hv = torch.empty_like(hk, pin_memory=True)
            hk.copy_(self.ki[:B])
            hv.copy_(self.vi[:B])
            a = self.base
            self.cold.append((a, a + B, hk, hv, _tile_table(lens, a, a + B, dev)))
            n = self.valid - a - B
            self.ki[:n] = self.ki[B:B + n].clone()
            self.vi[:n] = self.vi[B:B + n].clone()
            self.base = a + B

    def sync(self, layer):
        """The QVLayout for slot 0's current banded segments (GPU window part), or None."""
        if self.bad:
            return None
        from .codec import _unpack_dev
        sk, sv, lens = layer._banded_k[0], layer._banded_v[0], layer._seg_len[0]
        m, lim = 0, min(len(self.segk), len(sk))
        while m < lim and self.segk[m] is sk[m] and self.segv[m] is sv[m]:
            m += 1
        if self.lay is not None and m == len(sk) == len(self.segk):
            return self.lay
        dev = layer._device if layer._device is not None else self.ki.device
        h, d = len(sk[0]), sk[0][0].orig_shape[1]
        pos = sum(lens[:m])
        self._host_upto = min(self._host_upto, m)    # rolled-back segments re-queue
        if pos < self.base:                 # history rewritten inside the host tier
            self.cold, self.base, m, pos = [], 0, 0, 0
        self.valid = pos
        self._grow(0, len(sk), h, d, dev)
        self._finish_host_moves()
        cvals = []                          # (si, k centres, v centres): one copy at the end
        for si in range(m, len(sk)):
            sl = lens[si]
            need = pos + sl - self.base
            if (_AUTO_WIN and not self.win and self.ki is not None and need > self.cap
                    and self.cap > sl + 2 * _COLD_BLOCK):
                grow_to = max(need + need // 4, need + 4096)
                if not _vram_room(dev, 2 * grow_to * h * d * self.ki.element_size()):
                    # VRAM limit reached: keep this layer's buffer as it is and spill its oldest blocks from now on
                    self.win = self.cap - sl - _COLD_BLOCK
                    self._evict(lens, dev)
                    print(f"[fused_attn] VRAM limit at {pos} tokens: this layer now keeps the newest {self.win} on "
                          f"{dev}, older KV in host RAM", flush=True)
                    need = pos + sl - self.base
            self._grow(need, len(sk), h, d, dev, sl)
            cvals.append((si, [0.0] * h, [0.0] * h))
            w = pos - self.base
            for hi in range(h):
                for store, is_k, mn, st in ((sk, True, self.P[0], self.P[1]),
                                            (sv, False, self.P[2], self.P[3])):
                    bk = store[si][hi]
                    if bk.bits > 8 and self.ki.dtype == torch.uint8:
                        if _cuda_ext(dev) is None:      # the Triton kernel reads bytes only
                            self.bad = True
                            return None
                        self._widen()                   # 9..16-bit head: 2-byte ids from now on
                    ids = _unpack_dev(bk.packed.to(dev), bk.bits, sl * d).reshape(sl, d)
                    # int16 holds the uint16 bit pattern (the kernel reads it unsigned)
                    (self.ki if is_k else self.vi)[w:w + sl, hi] = ids.to(torch.int32).to(self.ki.dtype)
                    c = ((1 << bk.bits) - 1) / 2.0
                    mn[si, hi] = (bk.min_s + bk.tau_s * c).to(dev)    # range centre
                    st[si, hi] = bk.tau_s.to(dev)
                    cvals[-1][1 if is_k else 2][hi] = c
            pos += sl
            self.valid = pos
            self._evict(lens, dev)
        if cvals:
            from qv_band.common.xfer import h2d_small
            rows = h2d_small(torch.tensor([cv[1] + cv[2] for cv in cvals], dtype=torch.float32), dev)
            idx = h2d_small(torch.tensor([cv[0] for cv in cvals], dtype=torch.long), dev)
            self.C[0][idx] = rows[:, :h]
            self.C[1][idx] = rows[:, h:]
        if _KV_HOST:
            self._queue_host_moves(sk, sv, lens, dev)
        self.segk, self.segv = list(sk), list(sv)
        ns = len(sk)
        self.lay = _mk_lay(self.ki, self.vi, self.P, self.C, ns,
                           _tile_table(lens, self.base, self.valid, dev))
        return self.lay

    def attend_cold(self, q, q_pos0, scale):
        """(o, lse) over the host tier, streamed block by block; None if empty."""
        if not self.cold:
            return None
        dev, dt = q.device, self.ki.dtype
        h, d = self.ki.shape[1], self.ki.shape[2]
        cs, sk, sv = _staging(dev, dt, h, d)
        main = torch.cuda.current_stream(dev)
        ns = self.lay.kmin.shape[0]
        ready = [torch.cuda.Event(), torch.cuda.Event()]
        done = [None, None]

        def issue(j):
            s = j % 2
            if done[s] is not None:
                cs.wait_event(done[s])
            with torch.cuda.stream(cs):
                sk[s].copy_(self.cold[j][2], non_blocking=True)
                sv[s].copy_(self.cold[j][3], non_blocking=True)
                ready[s].record(cs)

        cs.wait_stream(main)                  # staging free of the previous call
        issue(0)
        if len(self.cold) > 1:
            issue(1)
        o = lse = None
        for j, (a, b, _, _, tt) in enumerate(self.cold):
            s = j % 2
            main.wait_event(ready[s])
            o2, l2 = attend(q, _mk_lay(sk[s], sv[s], self.P, self.C, ns, tt), q_pos0 - a, scale)
            o, lse = (o2, l2) if o is None else _merge_lse(o, lse, o2, l2)
            done[s] = torch.cuda.Event()
            done[s].record(main)
            if j + 2 < len(self.cold):
                issue(j + 2)
        return o, lse


def _merge_lse(o1, lse1, o2, lse2):
    m = torch.maximum(lse1, lse2)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w1, w2 = torch.exp(lse1 - m), torch.exp(lse2 - m)
    den = w1 + w2
    o = (o1 * w1[..., None] + o2 * w2[..., None]) / den.clamp_min(1e-30)[..., None]
    lse = torch.where(den > 0, m + torch.log(den.clamp_min(1e-30)), torch.full_like(den, float("-inf")))
    return o, lse


def _merge(o1, lse1, o2, lse2):
    m = torch.maximum(lse1, lse2)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w1, w2 = torch.exp(lse1 - m), torch.exp(lse2 - m)
    den = (w1 + w2).clamp_min(1e-30)
    return (o1 * w1[..., None] + o2 * w2[..., None]) / den[..., None]


def _resid_part(layer, q, q_pos0, scale, banded_len):
    """Attention of q (MQ,Hq,D) over the dense residual tail -> (o, lse) fp32."""
    rk, rv = layer._resid_k[0][0], layer._resid_v[0][0]      # (H, r, D)
    mq, hq, d = q.shape
    h, r = rk.shape[0], rk.shape[1]
    g = hq // h
    qf = q.float().view(mq, h, g, d)
    s = torch.einsum("qhgd,hrd->qhgr", qf, rk.float()) * scale
    kpos = banded_len + torch.arange(r, device=q.device)
    qpos = q_pos0 + torch.arange(mq, device=q.device)
    s = s.masked_fill((kpos[None, :] > qpos[:, None])[:, None, None, :], float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)[..., None])
    o = torch.einsum("qhgr,hrd->qhgd", p, rv.float())
    return o.reshape(mq, hq, d), lse.reshape(mq, hq)


def fused_layer_attention(layer, query, scale):
    """query (1, Hq, q, D) -> (1, q, Hq, D). Banded history via the kernel, the
    dense residual tail via torch, merged exactly (log-sum-exp)."""
    fk = layer._fused
    lay = fk.lay
    q = query[0].transpose(0, 1).contiguous()                 # (MQ, Hq, D)
    mq = q.shape[0]
    banded_len = layer._banded_len[0]
    r = layer._resid_k[0].shape[-2]
    q_pos0 = banded_len + r - mq
    o1 = lse1 = None
    if lay.n_tiles:
        o1, lse1 = attend(q, lay, q_pos0 - fk.base, scale)
    cold = fk.attend_cold(q, q_pos0, scale)
    if cold is not None:
        o1, lse1 = cold if o1 is None else _merge_lse(o1, lse1, *cold)
    if r:
        o2, lse2 = _resid_part(layer, q, q_pos0, scale, banded_len)
        o1 = _merge(o1, lse1, o2, lse2)
    return o1.to(query.dtype)[None]


def pending():
    """The layer registered by the update() that just ran, consumed once."""
    layer = getattr(_TLS, "layer", None)
    _TLS.layer = None
    return layer


def set_pending(layer):
    _TLS.layer = layer


_INSTALLED = {}


def install(model) -> bool:
    """Wrap the model's attention function so a layer that update() registered
    as fused is attended by fused_layer_attention. Only for the attention
    implementation the model really uses; anything else -> not installed."""
    from transformers.modeling_utils import AttentionInterface
    from qv_band.kv import banded_cache
    cfg = getattr(model, "config", None)
    name = getattr(cfg, "_attn_implementation", None) or getattr(
        getattr(cfg, "text_config", None), "_attn_implementation", None)
    base = AttentionInterface._global_mapping.get(name)
    if name not in ("flash_attention_2", "sdpa") or base is None:
        return False
    if name not in _INSTALLED:
        def wrapper(module, query, key, value, attention_mask, *a, _orig=base, **kw):
            layer = pending()
            if layer is None:
                return _orig(module, query, key, value, attention_mask, *a, **kw)
            scale = kw.get("scaling")
            if scale is None:
                scale = query.shape[-1] ** -0.5
            return fused_layer_attention(layer, query, scale), None
        AttentionInterface.register(name, wrapper)
        _INSTALLED[name] = base
    banded_cache.FUSED_ENABLED = True
    return True
