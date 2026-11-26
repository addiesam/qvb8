"""Static decode for any BatchTarget: one-token steps of every live request as ONE CUDA graph replay.

A model running its own forward (engine/adapter.BatchTarget) keeps per-request state in Python objects whose tensors grow
(torch.cat): fine for a prompt, but a decode step is then thousands of tiny kernel launches and the GPU waits on Python. Here the state of
the decoding requests lives in FIXED buffers, one row per slot, updated IN PLACE, positions in device tensors -- so the
step is capturable and replays as one graph.

The engine (target_engine.TargetBatchEngine) runs prompts through the target's usual forward, then moves the request into a
slot (static_load) and decodes it here. A target opts in by implementing:

    static_init(slots, max_len)   allocate its buffers (use the helpers below) and these four tensors:
                                  st_tok [S] long (input token per slot), st_active [S] bool, st_pos [S] long,
                                  st_logits [S, vocab] fp32 (output)
    static_load(slot, state)      copy a request's state (after its prompt) into slot `slot`; set st_pos[slot]
    static_step()                 one token for EVERY slot (inactive slots change nothing): read st_tok / st_active,
                                  write st_logits, advance st_pos[active] -- graph-safe: no host reads, fixed shapes
    static_capacity               max_len: a request whose context would pass it stays on the eager path

Generic state kinds (the model picks per layer):
    StaticKV(slots, heads, cap, dim, window=False)   keys / values, oldest first, count n[S]: a full layer appends;
                                                     window=True keeps the newest `cap` (a cap + WINDOW_SPARE buffer: append,
                                                     move the newest cap - 1 to the front once it fills)
    conv_push(state, x, active)                      a causal conv history [S, K-1, C]: drop the oldest, append x
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

WINDOW_SPARE = 128              # spare rows of a window buffer: one move of cap - 1 rows per ~WINDOW_SPARE tokens


@triton.jit
def _window_move(k_ptr, v_ptr, n_ptr, act_ptr, H, BUF, KEEP, T, D: tl.constexpr, BR: tl.constexpr):
    """Slot s active with no room for T more keys (n + T > BUF): rows [n - KEEP, n) -> [0, KEEP), of every head.
    Ascending chunks, the source n - KEEP > BR rows ahead of the destination (WINDOW_SPARE >= BR + T) and a barrier per
    chunk (the warps of a program are not in step), so the in-place move never reads a row already written."""
    pid = tl.program_id(0)
    s = pid // H
    n = tl.load(n_ptr + s)
    if (tl.load(act_ptr + s) != 0) & (n + T > BUF):
        base = pid.to(tl.int64) * BUF * D
        rd = tl.arange(0, D)
        for r0 in range(0, KEEP, BR):
            r = r0 + tl.arange(0, BR)
            m = (r < KEEP)[:, None]
            src = base + (r + n - KEEP)[:, None] * D + rd[None, :]
            dst = base + r[:, None] * D + rd[None, :]
            kk = tl.load(k_ptr + src, mask=m)
            vv = tl.load(v_ptr + src, mask=m)
            tl.debug_barrier()                            # every warp has read this chunk before any writes it
            tl.store(k_ptr + dst, kk, mask=m)
            tl.store(v_ptr + dst, vv, mask=m)


class StaticKV:
    """K and V of one cache layer for S slots: [S, H, buf, D], valid keys [0, n) oldest first. buf = cap, or with
    window=True cap + WINDOW_SPARE: the newest cap keys are [n - cap, n) (the attention masks by the window)."""

    def __init__(self, slots: int, heads: int, cap: int, dim: int, dtype, device, window: bool = False):
        buf = int(cap) + (WINDOW_SPARE if window else 0)
        self.k = torch.zeros(slots, heads, buf, dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.n = torch.zeros(slots, dtype=torch.long, device=device)
        self.cap, self.window = int(cap), bool(window)

    def load(self, slot: int, k, v) -> None:
        """k / v [1 or None, H, L, D] (oldest first): the newest `cap` of them go to slot `slot`."""
        k, v = (k[0] if k.dim() == 4 else k), (v[0] if v.dim() == 4 else v)
        L = min(k.shape[1], self.cap)
        self.k[slot, :, :L].copy_(k[:, k.shape[1] - L:])
        self.v[slot, :, :L].copy_(v[:, v.shape[1] - L:])
        self.n[slot] = L

    def clear(self, slot: int) -> None:
        self.n[slot] = 0

    def push(self, k_new, v_new, active) -> None:
        """Append one key / value per slot (k_new [S, H, 1, D]) where active; a full window buffer first moves its
        newest cap - 1 keys to the front."""
        S, H, buf, D = self.k.shape
        if self.window:
            at = self._make_room(1, active)
        else:
            at = self.n.clamp(max=buf - 1)
        idx = at.view(S, 1, 1, 1).expand(S, H, 1, D)
        a = active.view(S, 1, 1, 1)
        self.k.scatter_(2, idx, torch.where(a, k_new.to(self.k.dtype), self.k.gather(2, idx)))
        self.v.scatter_(2, idx, torch.where(a, v_new.to(self.v.dtype), self.v.gather(2, idx)))
        self.n.copy_(torch.where(active, (at + 1).clamp(max=buf), self.n))

    def _make_room(self, T, active):
        """Window: where an active slot has no room for T more keys, its newest cap - 1 move to the front (n follows)."""
        S, H, buf, D = self.k.shape
        with torch.cuda.device(self.k.device):
            _window_move[(S * H,)](self.k, self.v, self.n, active, H, buf, self.cap - 1, T, D=D, BR=64, num_warps=4)
        return torch.where(active & (self.n + T > buf), torch.full_like(self.n, self.cap - 1), self.n)

    def write_rows(self, k, v, slot, j, row_active, active, T) -> None:
        """Ragged rows (a speculative verify): key / value row r (k [R, H, D]) at n[slot[r]] + j[r] where row_active,
        NOT counted (commit() keeps some). A window first makes room for T keys (its n may drop to cap - 1)."""
        buf = self.k.shape[2]
        if self.window:
            self.n.copy_(self._make_room(T, active))
        p = (self.n[slot] + j).clamp(max=buf - 1)
        a = row_active.view(-1, 1, 1)
        self.k[slot, :, p] = torch.where(a, k.to(self.k.dtype), self.k[slot, :, p])
        self.v[slot, :, p] = torch.where(a, v.to(self.v.dtype), self.v[slot, :, p])

    def commit(self, keep) -> None:
        """keep [S]: of the keys write() put after n, the first keep[s] count."""
        self.n.copy_((self.n + keep).clamp(max=self.k.shape[2]))

    def nbytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()

    @staticmethod
    def bytes_for(slots, heads, cap, dim, dtype, window=False) -> int:
        """What StaticKV(slots, heads, cap, dim, dtype, window=window) allocates."""
        buf = int(cap) + (WINDOW_SPARE if window else 0)
        return 2 * slots * heads * buf * dim * torch.empty(0, dtype=dtype).element_size() + 8 * slots


def qvb8_encode(x):
    """x [..., B, D] -> (ids uint8 [..., B, D], min [..., D], step [..., D]) fp32: per channel over the B tokens,
    x = min + id * step, |error| <= step / 2 (8-bit ids: never coarser than the qvb8 KV codec's own bit choice)."""
    xf = x.float()
    lo, hi = xf.amin(-2), xf.amax(-2)
    step = (hi - lo) / 255.0
    ids = torch.where(step.unsqueeze(-2) > 0, torch.round((xf - lo.unsqueeze(-2)) / step.clamp(min=1e-30).unsqueeze(-2)),
                      torch.zeros_like(xf)).clamp(0, 255).to(torch.uint8)
    return ids, lo, step


@triton.jit
def _seal_kernel(t_ptr, ids_ptr, mn_ptr, st_ptr, seal_ptr, bi_ptr, H, BT, CAP, NB, D, B: tl.constexpr, DB: tl.constexpr):
    """Slot s sealing: tail rows [0, B) of head h, channels [d0, d0+DB) -> 8-bit ids at block bi[s] + per-channel min /
    step (qvb8_encode: x = min + id * step, step = (max - min) / 255); then tail rows [B, BT) move down by B."""
    s, h = tl.program_id(0), tl.program_id(1)
    if tl.load(seal_ptr + s) != 0:
        d = tl.program_id(2) * DB + tl.arange(0, DB)
        r = tl.arange(0, B)
        tb = t_ptr + ((s * H + h) * BT).to(tl.int64) * D
        x = tl.load(tb + r[:, None] * D + d[None, :]).to(tl.float32)
        lo = tl.min(x, 0)
        step = (tl.max(x, 0) - lo) / 255.0
        q = libdevice.rint(libdevice.div_rn(x - lo[None, :], tl.maximum(step, 1e-30)[None, :]))
        q = tl.where(step[None, :] > 0, tl.minimum(tl.maximum(q, 0.0), 255.0), 0.0)
        bi = tl.load(bi_ptr + s)
        ib = ids_ptr + ((s * H + h) * CAP).to(tl.int64) * D
        tl.store(ib + (bi * B + r)[:, None] * D + d[None, :], q.to(tl.uint8))
        pb = ((s * NB + bi) * H + h) * D + d
        tl.store(mn_ptr + pb, lo)
        tl.store(st_ptr + pb, step)
        for i in range(0, BT - B):                       # the rows past the block (a speculative step's extra) move down
            tl.store(tb + i * D + d, tl.load(tb + (B + i) * D + d))


class StaticQKV:
    """qvb8 K / V of one GROWING cache layer for S slots, CUDA-graph safe -- THE KV store of every static decoder
    (Inkling static slots, Qwen TP single / batch / speculative decoders).
    Closed blocks of `block` tokens: 8-bit ids [S, H, cap, D] + per (block, channel) min / step [S, cap/block, H, D]
    (the layout of kv/cuda_attn's WMMA kernel: one block's params of all heads together);
    the open block: dense tail [S, H, block + extra, D] (extra = room for a speculative step's tokens before commit).
    A tail of >= block tokens is sealed (its first block encoded) inside the same graph (fixed shapes, chosen per slot
    by torch.where): push() for one token per step, write() + set_len() for T tokens of which only some are kept.
    Keys j < lc = nc*block are read from the ids (min + id * step), the rest from the tail; engine/qvb8_attn reads this
    layout directly (no dense copy)."""

    def __init__(self, slots: int, heads: int, cap: int, dim: int, dtype, device, block: int = 256, extra: int = 0):
        cap = (int(cap) + block - 1) // block * block
        nb = cap // block
        self.block, self.cap, self.extra = int(block), cap, int(extra)
        self.kid = torch.zeros(slots, heads, cap, dim, dtype=torch.uint8, device=device)
        self.vid = torch.zeros_like(self.kid)
        self.kmin = torch.zeros(slots, nb, heads, dim, dtype=torch.float32, device=device)
        self.kstep, self.vmin, self.vstep = (torch.zeros_like(self.kmin) for _ in range(3))
        self.kt = torch.zeros(slots, heads, block + self.extra, dim, dtype=dtype, device=device)
        self.vt = torch.zeros_like(self.kt)
        self.nc = torch.zeros(slots, dtype=torch.long, device=device)      # closed blocks
        self.nt = torch.zeros(slots, dtype=torch.long, device=device)      # tokens in the tail
        self.window = False

    @property
    def n(self):
        return self.nc * self.block + self.nt

    @property
    def lc(self):
        return self.nc * self.block                                         # keys held as ids

    def clear(self, slot: int) -> None:
        self.nc[slot] = 0
        self.nt[slot] = 0

    def reset(self) -> None:
        self.nc.zero_()
        self.nt.zero_()

    # -- eager (prompt) side ---------------------------------------------------------------------------------------
    def _put(self, slot: int, k, v, b0: int) -> None:
        """k / v [H, L, D] = the keys from block b0 on: whole blocks encoded, the rest into the tail."""
        B, L = self.block, k.shape[1]
        nc = L // B
        if (b0 + nc) * B > self.cap or L - nc * B > self.kt.shape[2]:
            raise ValueError(f"{b0 * B + L} tokens do not fit {self.cap}")
        if nc:
            for x, ids, mn, st in ((k, self.kid, self.kmin, self.kstep), (v, self.vid, self.vmin, self.vstep)):
                i, lo, sp = qvb8_encode(x[:, :nc * B].reshape(x.shape[0], nc, B, x.shape[-1]))
                ids[slot, :, b0 * B:(b0 + nc) * B].copy_(i.reshape(x.shape[0], nc * B, -1))
                mn[slot, b0:b0 + nc].copy_(lo.transpose(0, 1))
                st[slot, b0:b0 + nc].copy_(sp.transpose(0, 1))
        r = L - nc * B
        if r:
            self.kt[slot, :, :r].copy_(k[:, nc * B:])
            self.vt[slot, :, :r].copy_(v[:, nc * B:])
        self.nc[slot], self.nt[slot] = b0 + nc, r

    def load(self, slot: int, k, v, keep: int = 0) -> None:
        """Slot `slot` holds k / v ([1 or None, H, L, D], oldest first). Closed blocks wholly below `keep` (tokens
        already in this slot, unchanged) stay as they are: no second rounding of a kept prefix."""
        k, v = (k[0] if k.dim() == 4 else k), (v[0] if v.dim() == 4 else v)
        b0 = min(int(keep) // self.block, int(self.nc[slot]))
        self._put(slot, k[:, b0 * self.block:], v[:, b0 * self.block:], b0)

    def append(self, slot: int, k, v) -> None:
        """Eager: k / v [1 or None, H, T, D] after the slot's keys (any T; full blocks sealed)."""
        k, v = (k[0] if k.dim() == 4 else k), (v[0] if v.dim() == 4 else v)
        nt = int(self.nt[slot])
        self._put(slot, torch.cat([self.kt[slot, :, :nt], k.to(self.kt.dtype)], 1),
                  torch.cat([self.vt[slot, :, :nt], v.to(self.vt.dtype)], 1), int(self.nc[slot]))

    def truncate(self, slot: int, n: int) -> None:
        """Keep the first n keys: a closed block cut in two goes back to the tail (decoded)."""
        if n >= int(self.n[slot]):
            return
        B = self.block
        nb = n // B
        if nb < int(self.nc[slot]) and n % B:
            k, v = self.dense(slot)
            self.kt[slot, :, :n % B].copy_(k[:, nb * B:n])
            self.vt[slot, :, :n % B].copy_(v[:, nb * B:n])
        elif nb == int(self.nc[slot]):
            pass                                                            # the cut is inside the tail
        self.nc[slot], self.nt[slot] = nb, n % B

    def drop_front(self, nblk: int):
        """Every slot: the oldest nblk closed blocks out (returned dense [S, H, nblk*block, D]), the rest shifted down."""
        B, S = self.block, self.kid.shape[0]
        out = []
        for ids, mn, st, t in ((self.kid, self.kmin, self.kstep, self.kt), (self.vid, self.vmin, self.vstep, self.vt)):
            H, D = ids.shape[1], ids.shape[3]
            x = ids[:, :, :nblk * B].float().reshape(S, H, nblk, B, D) * st[:, :nblk].transpose(1, 2).unsqueeze(3) \
                + mn[:, :nblk].transpose(1, 2).unsqueeze(3)
            out.append(x.reshape(S, H, nblk * B, D).to(t.dtype))
            ids[:, :, :self.cap - nblk * B].copy_(ids[:, :, nblk * B:].clone())
            mn[:, :mn.shape[1] - nblk].copy_(mn[:, nblk:].clone())
            st[:, :st.shape[1] - nblk].copy_(st[:, nblk:].clone())
        self.nc.sub_(nblk).clamp_(min=0)
        return out

    # -- graph side ---------------------------------------------------------------------------------------------------
    def _seal(self, seal) -> None:
        """Where seal [S]: encode the tail's first block into block nc, shift the tail down by a block, nc += 1.
        One gated launch per K / V (a slot that does not seal costs a flag read)."""
        S, H, Bt, D = self.kt.shape
        B = self.block
        bi = self.nc.clamp(max=self.cap // B - 1)
        sl = seal.to(torch.int32)
        with torch.cuda.device(self.kt.device):
            for t, ids, mn, st in ((self.kt, self.kid, self.kmin, self.kstep), (self.vt, self.vid, self.vmin, self.vstep)):
                _seal_kernel[(S, H, D // 32)](t, ids, mn, st, sl, bi, H, Bt, self.cap, mn.shape[1], D,
                                              B=B, DB=32, num_warps=4)
        self.nc.copy_(torch.where(seal, self.nc + 1, self.nc))
        self.nt.copy_(torch.where(seal, self.nt - B, self.nt))

    def push(self, k_new, v_new, active) -> None:
        """One key / value per slot (k_new [S, H, 1, D]) where active."""
        S, H, Bt, D = self.kt.shape
        a = active.view(S, 1, 1, 1)
        idx = self.nt.clamp(max=Bt - 1).view(S, 1, 1, 1).expand(S, H, 1, D)
        self.kt.scatter_(2, idx, torch.where(a, k_new.to(self.kt.dtype), self.kt.gather(2, idx)))
        self.vt.scatter_(2, idx, torch.where(a, v_new.to(self.vt.dtype), self.vt.gather(2, idx)))
        self.nt.copy_(torch.where(active, self.nt + 1, self.nt))
        self._seal(active & (self.nt >= self.block) & (self.nc < self.cap // self.block))

    def write(self, k, v, n0) -> None:
        """k / v [S, H, T, D] at key positions n0[S] .. n0+T-1 (n0 >= lc; not counted until set_len)."""
        S, H, T, D = k.shape
        off = (n0 - self.lc).clamp(min=0)
        idx = (off.view(S, 1, 1, 1) + torch.arange(T, device=k.device).view(1, 1, T, 1)) \
            .clamp(max=self.kt.shape[2] - 1).expand(S, H, T, D)
        self.kt.scatter_(2, idx, k.to(self.kt.dtype))
        self.vt.scatter_(2, idx, v.to(self.vt.dtype))

    def set_len(self, n) -> None:
        """n [S] = keys kept per slot after a step (written by write()); seals at most one block (T <= extra + 1)."""
        self.nt.copy_((n - self.lc).clamp(min=0, max=self.kt.shape[2]))
        self._seal((self.nt >= self.block) & (self.nc < self.cap // self.block))

    def write_rows(self, k, v, slot, j) -> None:
        """Ragged rows (a speculative verify): key / value row r (k [R, H, D]) at key position n[slot[r]] + j[r]
        (into the tail; not counted until commit)."""
        off = (self.n[slot] - self.lc[slot] + j).clamp(max=self.kt.shape[2] - 1)
        self.kt[slot, :, off] = k.to(self.kt.dtype)
        self.vt[slot, :, off] = v.to(self.vt.dtype)

    def commit(self, keep) -> None:
        """keep [S]: of the keys write_rows() put after n, the first keep[s] count (seals a full block)."""
        self.set_len(self.n + keep)

    def attn(self, q, vis, scale):
        """q [S, Hq, T, D]; vis [S, T] tail keys visible per query -> o fp32 [S, Hq, T, D], lse [S, Hq, T]."""
        from qv_band.engine.qvb8_attn import qvb8_attention
        return qvb8_attention(q, self, vis, scale)

    def dense(self, slot: int):
        """[H, n, D] of one slot (prompt reuse, spills, tests): the ids decoded + the tail."""
        nc, nt, B = int(self.nc[slot]), int(self.nt[slot]), self.block
        out = []
        for ids, mn, st, t in ((self.kid, self.kmin, self.kstep, self.kt), (self.vid, self.vmin, self.vstep, self.vt)):
            H, D = ids.shape[1], ids.shape[3]                               # explicit: nc may be 0
            x = ids[slot, :, :nc * B].float().reshape(H, nc, B, D) * st[slot, :nc].transpose(0, 1).unsqueeze(2) \
                + mn[slot, :nc].transpose(0, 1).unsqueeze(2)
            out.append(torch.cat([x.reshape(H, nc * B, D).to(t.dtype), t[slot, :, :nt]], 1))
        return out

    def nbytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in (self.kid, self.vid, self.kmin, self.kstep, self.vmin,
                                                           self.vstep, self.kt, self.vt))

    @staticmethod
    def bytes_for(slots, heads, cap, dim, dtype, block: int = 256, extra: int = 0) -> int:
        """What StaticQKV(slots, heads, cap, dim, dtype, block=block, extra=extra) allocates."""
        cap = (int(cap) + block - 1) // block * block
        es = torch.empty(0, dtype=dtype).element_size()
        return slots * heads * dim * (2 * cap + 4 * 4 * (cap // block) + 2 * (block + extra) * es) + 16 * slots


@triton.jit
def _conv_commit_kernel(st_ptr, sc_ptr, keep_ptr, TS, C, K1: tl.constexpr, BC: tl.constexpr):
    s = tl.program_id(0)
    keep = tl.load(keep_ptr + s)
    if keep > 0:
        c = tl.program_id(1) * BC + tl.arange(0, BC)
        m = c < C
        for i in tl.static_range(K1):                    # row i <- row keep+i of [state (K1) | scratch]: ascending,
            jj = keep + i                                # the source always ahead (keep >= 1), one thread per column
            a = tl.load(st_ptr + (s * K1 + jj) * C + c, mask=m & (jj < K1), other=0.0)
            b = tl.load(sc_ptr + (s * TS + jj - K1) * C + c, mask=m & (jj >= K1), other=0.0)
            tl.store(st_ptr + (s * K1 + i) * C + c, tl.where(jj < K1, a, b), mask=m)


def conv_commit(state, scratch, keep) -> None:
    """A speculative step's conv history: state [S, K-1, C] fp32 <- the last K-1 of [state | the first keep[s] inputs
    in scratch [S, Ts, C]] (keep 0: unchanged). One launch."""
    S, K1, C = state.shape
    with torch.cuda.device(state.device):
        _conv_commit_kernel[(S, triton.cdiv(C, 512))](state, scratch, keep, scratch.shape[1], C, K1=K1, BC=512,
                                                      num_warps=4)


def conv_push(state, x, active) -> None:
    """state [S, K-1, C] (oldest first) <- drop the oldest row, append x [S, C], where active."""
    new = torch.cat([state[:, 1:], x.unsqueeze(1).to(state.dtype)], 1)
    state.copy_(torch.where(active.view(-1, 1, 1), new, state))


class StaticStepper:
    """Slots + the captured step of one target (one CUDA graph for every slot at once)."""

    def __init__(self, target, slots: int | None = None, max_len: int | None = None, shrink: bool = True,
                 reserve: int | None = None):
        self.t = target
        self.slots = int(slots or os.environ.get("QV_STATIC_SLOTS", "4"))
        self.max_len = int(max_len or os.environ.get("QV_STATIC_MAX_LEN", "32768"))
        # the slot buffers take what the card can spare AND leave QV_STATIC_RESERVE_GB free for prompts (eager prefill):
        # the context halves down to 1024 tokens, then the slots halve -- a full card (experts cached on it) gets fewer,
        # shorter slots. A target that says what its buffers take (static_bytes) is sized before anything is allocated.
        # reserve: bytes to leave free (measure_prefill_peak at engine start), at least QV_STATIC_RESERVE_GB
        reserve = max(float(os.environ.get("QV_STATIC_RESERVE_GB", "2")) * 2**30, float(reserve or 0))
        self.reserve = reserve
        dev = torch.cuda.current_device()
        sizes, sl_ = [], self.slots
        while sl_ >= 1:
            ml = self.max_len
            while True:
                sizes.append((sl_, ml))
                if not shrink or ml <= 1024:
                    break
                ml //= 2
            if not shrink:
                break
            sl_ //= 2
        torch.cuda.empty_cache()
        free_now = torch.cuda.mem_get_info(dev)[0]
        if hasattr(self.t, "static_bytes"):
            fit = [z for z in sizes if self.t.static_bytes(*z) + reserve <= free_now]
            if not fit:
                raise RuntimeError(f"static decode: no room for {sizes[-1][0]} slot(s) of {sizes[-1][1]} tokens "
                                   f"({free_now / 2**30:.1f} GiB free, {reserve / 2**30:.1f} GiB kept for prompts)")
            sizes = fit
        why = ""
        for i, (sl_, ml) in enumerate(sizes):
            if i:
                print(f"[static] {self.slots} slots x {self.max_len} tokens {why}: trying {sl_} x {ml}", flush=True)
            self.slots, self.max_len = sl_, ml
            try:
                self.t.static_init(self.slots, self.max_len)
                torch.cuda.empty_cache()
                if torch.cuda.mem_get_info(dev)[0] >= reserve:
                    break
                why = "would leave less than the reserve free"
            except torch.OutOfMemoryError:
                why = "out of memory"
            if hasattr(self.t, "static_free"):
                self.t.static_free()
            torch.cuda.empty_cache()
        else:
            raise RuntimeError(f"static decode: no room for {sizes[-1][0]} slot(s) of {sizes[-1][1]} tokens ({why})")
        self.free = list(range(self.slots))
        self.graph = None
        self.pool = torch.cuda.graph_pool_handle()        # the step / verify / commit graphs run one at a time
        from qv_band.engine.spec import SpecPlanner
        self.pgraph, self.vgraph, self.cg = {}, {}, {}      # speculative: proposal / verify / commit graphs
        self.planner = SpecPlanner()                       # their measured costs + the draft plan
        # a target whose step cannot be captured (e.g. experts fetched from host RAM on demand) says static_graphable =
        # False: its static step then runs eagerly (still the fixed slot buffers)
        self.use_graph = os.environ.get("QV_STATIC_GRAPH", "1") == "1" and bool(getattr(target, "static_graphable", True))
        self.dev = self.t.st_tok.device

    def warm(self) -> None:
        """Capture now every graph decoding will use (the step; a speculative target's proposal / verify / commit), so
        the memory they take is there before the engine checks what prompts have left."""
        if self.use_graph and self.graph is None:
            self._capture()
        t = self.t
        if hasattr(t, "st_vtok") and not self.planner.vcost:
            self._spec_measure(0)
            t.st_acc.zero_()
            self._run(self.cg, 0, t.static_commit)
        torch.cuda.synchronize(self.dev)

    def close(self) -> None:
        """Give everything back (graphs, slot buffers): the engine sizes again, smaller."""
        self.graph, self.pgraph, self.vgraph, self.cg = None, {}, {}, {}
        if hasattr(self.t, "static_free"):
            self.t.static_free()
        if getattr(self.t, "_qv_static_stepper", None) is self:
            self.t._qv_static_stepper = None
        torch.cuda.synchronize(self.dev)
        torch.cuda.empty_cache()

    def acquire(self):
        return self.free.pop(0) if self.free else None

    def release(self, slot: int) -> None:
        if slot is not None and slot not in self.free:
            self.free.append(slot)
            self.free.sort()

    def load(self, slot: int, state) -> None:
        self.t.static_load(slot, state)

    def _capture(self):
        t = self.t
        t.st_active.zero_()                                  # warm-up and capture change nothing
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(2):
                t.static_step()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s, pool=self.pool):
            t.static_step()
        torch.cuda.synchronize(self.dev)
        self.graph = g

    def _graph(self, fn):
        """fn() captured (warm-up twice with no slot active: nothing changes), or None where the target is not capturable."""
        t = self.t
        if not self.use_graph:
            return None
        t.st_active.zero_()
        if hasattr(t, "st_acc"):
            t.st_acc.zero_()
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s, pool=self.pool):
            fn()
        torch.cuda.synchronize(self.dev)
        return g

    def _run(self, graphs, key, fn, costs=None):
        """Replay graphs[key] (captured on first use), or fn() eagerly where the target is not capturable. costs: a
        new graph is also timed into costs[key] (only for steps that do not advance a state: proposal, verify)."""
        import time
        t = self.t
        if key not in graphs:
            act, acc = t.st_active.clone(), t.st_acc.clone()
            graphs[key] = self._graph(fn)
            t.st_active.copy_(act)                           # (the capture's warm-up ran with no slot active, keep 0)
            t.st_acc.copy_(acc)
            if costs is not None:
                run = graphs[key].replay if graphs[key] is not None else fn
                run()
                torch.cuda.synchronize(self.dev)
                t0 = time.perf_counter()
                run()
                torch.cuda.synchronize(self.dev)
                self.planner.observe(costs, key, time.perf_counter() - t0, exact=True)
        (graphs[key].replay if graphs[key] is not None else fn)()

    # speculative step (engine/spec.py protocol), in phases: plan (the drafting card only) -> verify + commit (every card
    # of a Tensor Parallel group runs these two and _spec_measure in the same order: they hold the all-reduces;
    # engine/proc_tp._ProcStepper.spec sends them)
    @torch.no_grad()
    def spec(self, rows: dict) -> dict:
        """{slot: (last token, (temperature, top_k, top_p))} -> {slot: tokens emitted (the accepted drafts + the target's
        next)}: greedy, or speculative sampling per slot. One proposal graph per number of slots (a step with a sampled
        slot proposes eagerly: its drafts are drawn); the verify runs RAGGED rows (each slot's last token + its drafts,
        back to back: no row for an idle slot), one graph per row count; one commit graph."""
        if not self.planner.vcost:
            self._spec_measure(min(rows))
        order, drafts, dists = self._spec_plan(rows)
        R, inputs = self._spec_rows(order, rows, drafts)
        self._spec_load(inputs)
        self._spec_verify(R)
        out, keep = self._spec_accept(order, drafts, inputs, R, rows, dists)
        self._spec_commit(keep)
        return out

    @property
    def spec_margin(self) -> int:
        """Positions a speculative request needs past its last token (the verify writes them before the commit)."""
        t = self.t
        return t.st_vtok.shape[0] // self.slots + 1 if hasattr(t, "st_vtok") else 0

    @torch.no_grad()
    def _spec_measure(self, sl):
        """This card's costs (once): the proposal of one slot (a drafting card), a verify of 1..TV rows of slot sl
        (neither advances a state), each graph captured and timed."""
        t, P = self.t, self.planner
        Rm = t.st_vtok.shape[0]
        TV = Rm // self.slots
        t.st_active.zero_()
        t.st_active[sl] = True
        if hasattr(t, "static_propose"):
            t.st_pslot[0], t.st_panchor[0] = sl, 0
            self._run(self.pgraph, 1, lambda: t.static_propose(1), P.pcost)
        t.st_vtok.zero_()
        t.st_vslot.fill_(sl)
        t.st_vstart.zero_()
        for R in range(1, TV + 1):
            t.st_vj.copy_(torch.tensor(list(range(R)) + [TV] * (Rm - R)))
            t.st_vrow.copy_(torch.arange(Rm) < R)
            self._run(self.vgraph, R, lambda R=R: t.static_verify(R), P.vcost)
        print(f"[static] speculative step: proposal {P.pcost.get(1, 0.0) * 1e3:.1f} ms, verify " +
              ", ".join(f"{R} rows {P.vcost[R] * 1e3:.1f}" for R in sorted(P.vcost)) + " ms", flush=True)

    def _spec_plan(self, rows):
        """-> (slots in order, {slot: drafts to verify}, {slot: draft distributions or None}): the proposal (unless the
        planner skips it) + the plan."""
        t, P = self.t, self.planner
        order = sorted(rows)
        n = len(order)
        if not P.want_proposal():
            return order, {sl: [] for sl in order}, {sl: None for sl in order}
        t.st_pslot[:n].copy_(torch.tensor(order))
        t.st_panchor[:n].copy_(torch.tensor([rows[sl][0] for sl in order]))
        sps = [rows[sl][1] if rows[sl][1][0] > 0 else None for sl in order]
        if any(sps):
            t.static_propose(n, sps)                          # drawn drafts: eager
        else:
            self._run(self.pgraph, n, lambda: t.static_propose(n), P.pcost)
        dt, dp = t.st_draft[:n].cpu().tolist(), t.st_dprob[:n].cpu().tolist()
        k = P.plan(dp, t.static_max_draft)
        drafts = {sl: dt[i][:k[i]] for i, sl in enumerate(order)}
        dists = {sl: (list(t.st_ddist[i, :k[i]].unbind(0)) if sps[i] else None) for i, sl in enumerate(order)}
        return order, drafts, dists

    def _spec_rows(self, order, rows, drafts):
        """The verify's ragged rows as host lists (what a Tensor Parallel group sends its other cards)."""
        Rm = self.t.st_vtok.shape[0]
        TV = Rm // self.slots
        tok, slot, j, start = [], [], [], [0] * self.slots
        for sl in order:
            start[sl] = len(tok)
            seq = [rows[sl][0]] + drafts[sl]
            tok += seq
            slot += [sl] * len(seq)
            j += range(len(seq))
        R, pad = len(tok), Rm - len(tok)                      # rows past R: inactive, the padding row TV of a slot
        return R, (tok + [0] * pad, slot + [slot[0]] * pad, j + [TV] * pad, R, start, order)

    def _spec_load(self, inputs):
        t = self.t
        tok, slot, j, R, start, order = inputs
        Rm = t.st_vtok.shape[0]
        act = torch.zeros(self.slots, dtype=torch.bool)
        act[list(order)] = True
        t.st_active.copy_(act)
        t.st_vtok.copy_(torch.tensor(tok))
        t.st_vslot.copy_(torch.tensor(slot))
        t.st_vj.copy_(torch.tensor(j))
        t.st_vrow.copy_(torch.arange(Rm) < R)
        t.st_vstart.copy_(torch.tensor(start))

    def _spec_verify(self, R):
        t = self.t
        self._run(self.vgraph, R, lambda: t.static_verify(R), self.planner.vcost)

    def _spec_accept(self, order, drafts, inputs, R, rows, dists):
        from qv_band.engine.spec import verify_row
        start = inputs[4]
        lg = self.t.st_vlogits
        tgt = lg[:R].argmax(-1).cpu().tolist()
        out, keep = {}, [0] * self.slots
        for sl in order:
            d, r0, sp = drafts[sl], start[sl], rows[sl][1]
            a, nxt = verify_row(d, dists[sl], lg[r0:r0 + len(d) + 1], sp[0], sp[1], sp[2], True,
                                tgt[r0:r0 + len(d) + 1])
            out[sl], keep[sl] = d[:a] + [nxt], a + 1
        return out, keep

    def _spec_commit(self, keep):
        t = self.t
        t.st_acc.copy_(torch.tensor(keep))
        self._run(self.cg, 0, t.static_commit)

    @torch.no_grad()
    def step(self, toks: dict) -> dict:
        """{slot: input token} -> {slot: logits [vocab] fp32} (views into the target's output buffer)."""
        t = self.t
        act = torch.zeros(self.slots, dtype=torch.bool)
        tok = torch.zeros(self.slots, dtype=torch.long)
        for sl, x in toks.items():
            act[sl], tok[sl] = True, int(x)
        if self.use_graph and self.graph is None:
            try:
                self._capture()
            except Exception as e:                       # noqa: BLE001 -- run the same step eagerly instead
                print(f"[static] CUDA graph capture failed ({e!r}): eager static steps", flush=True)
                self.use_graph, self.graph = False, None
                torch.cuda.synchronize(self.dev)
        t.st_tok.copy_(tok, non_blocking=True)
        t.st_active.copy_(act, non_blocking=True)
        if self.graph is not None:
            self.graph.replay()
        else:
            t.static_step()
        return {sl: t.st_logits[sl] for sl in toks}


def free_bytes(target) -> int:
    """Free bytes on the target's tightest card (a Tensor Parallel target: target.free_bytes() over its cards)."""
    if hasattr(target, "free_bytes"):
        return int(target.free_bytes())
    dev = torch.device(getattr(getattr(target, "m", None), "device", "cuda"))
    torch.cuda.empty_cache()
    return int(torch.cuda.mem_get_info(dev)[0])


def size_stepper(target, prefill_tokens: int):
    """The engine's static stepper, sized so that a prompt still fits: one real prefill of `prefill_tokens` measured
    first (nothing else allocated: a measurement must not run out of memory -- on a Tensor Parallel group that stops
    every card), then slots sized and every decode graph captured (warm); while the tightest card has less than 1.25x
    that peak free, everything is given back and the slots shrink (context halves to 1024, then the slots halve).
    None: no room."""
    slots = int(os.environ.get("QV_STATIC_SLOTS", "4"))
    max_len = int(os.environ.get("QV_STATIC_MAX_LEN", "32768"))
    peak = measure_prefill_peak(target, prefill_tokens)
    while True:
        st = get_stepper(target, reserve=int(1.25 * peak), slots=slots, max_len=max_len)
        st.warm()
        free = free_bytes(target)
        if free >= 1.25 * peak:
            print(f"[static] {st.slots} slots x {st.max_len} tokens, graphs captured; a {prefill_tokens}-token prefill "
                  f"peaks at {peak / 2**30:.2f} GiB, {free / 2**30:.2f} GiB free", flush=True)
            return st
        slots, max_len = st.slots, st.max_len
        print(f"[static] {slots} slots x {max_len} tokens leave {free / 2**30:.2f} GiB free, a prefill needs "
              f"{1.25 * peak / 2**30:.2f}: smaller", flush=True)
        st.close()
        if max_len > 1024:
            max_len //= 2
        elif slots > 1:
            slots //= 2
        else:
            return None


def measure_prefill_peak(target, tokens: int) -> int:
    """Bytes one eager prefill of `tokens` tokens takes above what the cards hold now, on the card where it is most
    (a Tensor Parallel target: target.prefill_peak over its cards) -- the room the static slots must leave free for
    prompts. A scratch state, dropped after."""
    if hasattr(target, "prefill_peak"):
        return int(target.prefill_peak(tokens))
    dev = torch.device(getattr(getattr(target, "m", None), "device", "cuda"))
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    st = target.new_state()
    target.forward([(st, [0] * int(tokens))])
    torch.cuda.synchronize(dev)
    peak = torch.cuda.max_memory_allocated(dev) - base
    del st
    torch.cuda.empty_cache()
    return int(peak)


def get_stepper(target, reserve: int | None = None, slots: int | None = None, max_len: int | None = None):
    """The target's ONE stepper (created on first use; None when the target has no static decode or QV_STATIC_DECODE=0).
    Shared by the batch engine and the one-request paths (image chats): slots are taken from the same pool."""
    st = getattr(target, "_qv_static_stepper", None)
    if st is None and os.environ.get("QV_STATIC_DECODE", "1") != "0" and \
            (hasattr(target, "static_init") or hasattr(target, "static_stepper")):
        st = (target.static_stepper(reserve, slots, max_len) if hasattr(target, "static_stepper")
              else StaticStepper(target, slots, max_len, reserve=reserve))
        target._qv_static_stepper = st
    return st


class SlotDecode:
    """One request's decode after its prompt: through a static slot when one is free and the context fits, else the
    target's eager forward. dec(token) -> logits [vocab] fp32; dec.spec(token) -> tokens (a greedy request of a target
    that drafts by itself, engine/spec.py: the accepted drafts + the target's next); close() frees the slot."""

    def __init__(self, target, state, context: int, speculative: bool = False):
        self.t, self.state, self.slot, self.st = target, state, None, None
        self.speculative = speculative and getattr(target, "spec_propose", None) is not None
        try:
            st = get_stepper(target)
        except Exception as e:                           # noqa: BLE001 -- eager keeps working
            print(f"[static] off: {e!r}", flush=True)
            st = None
        margin = st.spec_margin if st is not None and self.speculative else 0
        if st is not None and context + margin <= st.max_len and \
                (not self.speculative or getattr(target, "static_verify", None) is not None):
            slot = st.acquire()
            if slot is not None:
                st.load(slot, state)
                self.st, self.slot, self.state = st, slot, None
        if self.speculative and self.slot is None:
            from qv_band.engine.spec import SpecPlanner
            self.planner = getattr(target, "_qv_spec_planner", None) or SpecPlanner()
            target._qv_spec_planner = self.planner         # one eager cost table per target

    def __call__(self, tok):
        if self.slot is not None:
            return self.st.step({self.slot: tok})[self.slot]
        return self.t.forward([(self.state, [tok])])[0]

    def spec(self, tok, sp=(0.0, 0, 1.0)) -> list:
        """tok: the last token; sp: (temperature, top_k, top_p) -> tokens emitted (greedy or speculative sampling)."""
        if self.slot is not None:
            return self.st.spec({self.slot: (tok, sp)})[self.slot]
        from qv_band.engine.spec import eager_step
        return eager_step(self.t, self.planner, [(self.state, tok, sp)])[0]

    def close(self):
        if self.slot is not None:
            self.st.release(self.slot)
            self.slot = None
