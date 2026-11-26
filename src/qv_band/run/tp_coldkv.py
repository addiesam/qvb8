"""Cold KV for TP long context (framework level): positions older than the VRAM hot window, in pinned host RAM.

Storage: per attention layer, a list of SEGMENTS (<= `seg` positions each, identical boundaries in every layer).
A segment is either
  * packed  -- the production certified KV codec (qv_band.kv.codec: per-channel affine, bits raised per head until
               every element is within tau; default tau 0.1, 6-bit floor, as the DP engine): ~0.4x of fp16 over PCIe;
  * raw     -- fp16 (tau 0: bit-exact with the no-offload path).
Each card stores only its own KV heads (its TP share).

Decode streams the segments back every step (double-buffered on a copy stream), dequantises them on the GPU and
merges their attention exactly by log-sum-exp with the hot window.

Nothing is ever written to disk: the cold KV lives in host RAM only (disk is read-only, for weights).
"""
from __future__ import annotations

import os

import torch

COLD_TAU = float(os.environ.get("QV_TP_COLD_TAU", "0.1"))
COLD_BITS = int(os.environ.get("QV_TP_COLD_BITS", "6"))


def _pin(t):
    h = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    h.copy_(t)
    return h


class _Seg:
    __slots__ = ("n", "raw", "k", "v")

    def __init__(self, n, raw, k, v):
        self.n, self.raw, self.k, self.v = n, raw, k, v      # raw: k, v pinned [nkv, n, hd] fp16
        #                                                       packed: k, v = list per head of (pinned bytes, bits,
        #                                                       min_s gpu [hd], tau_s gpu [hd])

    def nbytes(self):
        if self.raw:
            return 2 * self.k.numel() * self.k.element_size()
        return sum(p.numel() for kv in (self.k, self.v) for p, _, _, _ in kv)


class ColdStore:
    def __init__(self, layers, device, seg=8192, tau=COLD_TAU, bits=COLD_BITS):
        self.layers, self.dev, self.seg, self.tau, self.bits = list(layers), device, seg, tau, bits
        self.segs = {li: [] for li in self.layers}
        self.stream = torch.cuda.Stream(device)
        self._stage = None

    # -------------------------------------------------------------- size
    def length(self, li=None):
        li = self.layers[0] if li is None else li
        return sum(s.n for s in self.segs[li])

    def nbytes(self):
        return sum(s.nbytes() for L in self.segs.values() for s in L)

    def clear(self):
        for li in self.layers:
            self.segs[li] = []

    # -------------------------------------------------------------- write
    def _encode(self, t):
        """t [nkv, n, hd] fp16 on GPU -> per head (pinned bytes, bits, min_s, tau_s)."""
        from qv_band.kv.codec import band_kv_heads
        out = []
        for b in band_kv_heads(t, t.dtype, self.bits, tau=self.tau, certify=True):
            out.append((_pin(b.packed), b.bits, b.min_s.to(self.dev).float().contiguous(),
                        b.tau_s.to(self.dev).float().contiguous()))
        return out

    def append(self, li, k, v):
        """Append positions (k, v: [nkv, n, hd] on GPU) at the end of layer li's cold part."""
        n = k.shape[1]
        L = self.segs[li]
        a = 0
        while a < n:
            room = self.seg - L[-1].n if (L and L[-1].n < self.seg and L[-1].raw) else 0
            if room > 0:                                  # top up the last raw segment (fp16 is cheap to extend)
                b = min(n, a + room)
                last = L[-1]
                L[-1] = _Seg(last.n + b - a, True, torch.cat([last.k, _pin(k[:, a:b])], 1),
                             torch.cat([last.v, _pin(v[:, a:b])], 1))
                L[-1].k, L[-1].v = _pin(L[-1].k), _pin(L[-1].v)
                a = b
                continue
            b = min(n, a + self.seg)
            if self.tau > 0:
                L.append(_Seg(b - a, False, self._encode(k[:, a:b].contiguous()),
                              self._encode(v[:, a:b].contiguous())))
            else:
                L.append(_Seg(b - a, True, _pin(k[:, a:b]), _pin(v[:, a:b])))
            a = b

    def truncate(self, p):
        """Keep positions [0, p)."""
        for li in self.layers:
            L, out, acc = self.segs[li], [], 0
            for s in L:
                if acc + s.n <= p:
                    out.append(s)
                    acc += s.n
                    continue
                keep = p - acc
                if keep > 0:                              # cut a segment: decode it and keep its head as raw fp16
                    k, v = self._to_gpu(s)
                    out.append(_Seg(keep, True, _pin(k[0, :, :keep]), _pin(v[0, :, :keep])))
                    acc += keep
                break
            self.segs[li] = out

    # -------------------------------------------------------------- read
    def _dequant(self, parts, n, hd, out):
        from qv_band.kv.codec_kernels import dequant
        for h, (pk, bits, mn, ts) in enumerate(parts):
            out[0, h, :n].copy_(dequant(pk, bits, n * hd, mn, ts, hd, torch.float16).view(n, hd))

    def _to_gpu(self, s):
        if s.raw:
            return s.k.to(self.dev)[None], s.v.to(self.dev)[None]
        hd = s.k[0][2].numel()
        k = torch.empty(1, len(s.k), s.n, hd, dtype=torch.float16, device=self.dev)
        v = torch.empty_like(k)
        self._dequant([(p.to(self.dev), b, m, t) for p, b, m, t in s.k], s.n, hd, k)
        self._dequant([(p.to(self.dev), b, m, t) for p, b, m, t in s.v], s.n, hd, v)
        return k, v

    def read_into(self, li, K, V, n):
        """Positions [0, n) of layer li into K, V [1, nkv, >= n, hd] (GPU)."""
        a = 0
        for s in self.segs[li]:
            if a >= n:
                break
            k, v = self._to_gpu(s)
            m = min(s.n, n - a)
            K[:, :, a:a + m].copy_(k[:, :, :m])
            V[:, :, a:a + m].copy_(v[:, :, :m])
            a += s.n

    def blocks(self, li):
        """Yield (k, v) [1, nkv, n, hd] fp16 GPU views of each segment, the next one's bytes in flight meanwhile."""
        L = self.segs[li]
        if not L:
            return
        cur, cs = torch.cuda.current_stream(self.dev), self.stream
        cs.wait_stream(cur)                      # staging buffers may still be read by the previous layer
        if self._stage is None:
            self._stage = [dict(), dict()]
        ev_copy, ev_use = [None, None], [None, None]
        staged = [None, None]

        def issue(j):
            s, b = L[j], j % 2
            st = self._stage[b]
            with torch.cuda.stream(cs):
                if ev_use[b] is not None:
                    cs.wait_event(ev_use[b])
                if s.raw:
                    shp = tuple(s.k.shape)
                    key = ("raw", shp)
                    if key not in st:
                        st.clear()
                        st[key] = (torch.empty(shp, dtype=torch.float16, device=self.dev),
                                   torch.empty(shp, dtype=torch.float16, device=self.dev))
                    gk, gv = st[key]
                    gk.copy_(s.k, non_blocking=True)
                    gv.copy_(s.v, non_blocking=True)
                    staged[b] = ("raw", gk, gv)
                else:
                    parts = []
                    for kv in (s.k, s.v):
                        hp = []
                        for h, (p, bits, mn, ts) in enumerate(kv):
                            key = ("p", len(parts), h)
                            buf = st.get(key)
                            if buf is None or buf.numel() < p.numel():
                                buf = st[key] = torch.empty(max(p.numel(), 1 << 20), dtype=torch.uint8,
                                                            device=self.dev)
                            buf[:p.numel()].copy_(p, non_blocking=True)
                            hp.append((buf[:p.numel()], bits, mn, ts))
                        parts.append(hp)
                    staged[b] = ("pack", parts)
                ev_copy[b] = torch.cuda.Event()
                ev_copy[b].record(cs)

        issue(0)
        hd = None
        for j, s in enumerate(L):
            if j + 1 < len(L):
                issue(j + 1)
            b = j % 2
            cur.wait_event(ev_copy[b])
            if staged[b][0] == "raw":
                k, v = staged[b][1][None], staged[b][2][None]
            else:
                parts = staged[b][1]
                hd = parts[0][0][2].numel()
                key = ("deq", s.n)
                dq = self._stage[b].get(key)
                if dq is None:
                    dq = self._stage[b][key] = (torch.empty(1, len(parts[0]), s.n, hd, dtype=torch.float16,
                                                            device=self.dev),
                                                torch.empty(1, len(parts[0]), s.n, hd, dtype=torch.float16,
                                                            device=self.dev))
                self._dequant(parts[0], s.n, hd, dq[0])
                self._dequant(parts[1], s.n, hd, dq[1])
                k, v = dq
            yield k, v
            ev_use[b] = torch.cuda.Event()
            ev_use[b].record(cur)
