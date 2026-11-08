"""Two-card overlapped chunked prefill for pipeline placement (QV_PIPELINE_SPLIT).

Sequential chunked prefill leaves one card idle at all times: card A runs
layers [0, split) for a chunk, then card B runs [split, L) while A waits. Here
each chunk runs in its own worker thread and passes through two ordered stages:

    stage A: embed + layers [0, split)    on card A
    stage B: layers [split, L)            on card B

A stage admits chunks strictly in order, so every layer still sees chunk c
before chunk c + 1 -- the KV / recurrent state evolves exactly as in the
sequential loop (same kernels, same order per layer) and the result is
identical. What changes is only that chunk c + 1 can be issued to card A while
card B still works on chunk c.

The two stages must never wait on each other's card, or the overlap is lost:
  * rotary cos/sin and positions for the whole run are computed once up
    front, before any chunk is in flight, and sliced per chunk; the
    embedding is looked up in host RAM (a cross-card copy through host
    memory mid-run would synchronise the source card's whole queue,
    including the other chunk's work);
  * the stage-boundary hidden state is copied on a side stream after an
    event, not on the default stream.

Requires the fused qvb8 attention path (it ignores the attention mask, so no
mask -- whose size depends on the cache length -- has to be built while the
other chunk is mid-flight). The caller runs the first chunk(s) sequentially
until the fused path is active and the last chunk sequentially for logits.
"""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from qv_band.common.xfer import to_device


import os as _os
_DBG = _os.environ.get("QV_PIPE_DEBUG") == "1"
_INFLIGHT = int(_os.environ.get("QV_PIPE_INFLIGHT", "2"))


class _Stage:
    """Admits chunk tickets strictly in order."""

    def __init__(self, first: int):
        self._cv = threading.Condition()
        self._next = first

    def enter(self, c: int) -> None:
        with self._cv:
            self._cv.wait_for(lambda: self._next == c)

    def leave(self, c: int) -> None:
        with self._cv:
            self._next = c + 1
            self._cv.notify_all()


class PipePrefill:
    """Per-model helper, built once (holds the per-card rotary copies)."""

    def __init__(self, model, split: int):
        lm = model.model
        self.lm = lm
        self.split = split
        self.layers = list(lm.layers[: lm.config.num_hidden_layers])
        self.types = lm.config.layer_types
        self.dev_a = next(self.layers[0].parameters()).device
        self.dev_b = next(self.layers[-1].parameters()).device
        if self.dev_a == self.dev_b:
            raise ValueError("pipeline prefill needs two cards")
        self.rot_b = lm.rotary_emb
        self.copy_stream = torch.cuda.Stream(device=self.dev_a)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipe-prefill")
        self._bufs = {}
        self._fail = False

    def _host_buf(self, h):
        """Pinned boundary buffers, reused: (chunk slot) -> one per in-flight
        chunk, sized for the largest chunk seen (no cudaHostAlloc per chunk)."""
        need = h.numel()
        key = threading.get_ident()
        b, ev = self._bufs.get(key, (None, None))
        if ev is not None:
            ev.synchronize()              # its last H2D (to card B) has finished
        if b is None or b.numel() < need or b.dtype != h.dtype:
            b = torch.empty(need, dtype=h.dtype, pin_memory=True)
        self._bufs[key] = (b, None)
        return b[:need].view(h.shape)

    def _tables(self, h_like, p_lo: int, p_hi: int):
        """Rotary cos/sin + text positions for [p_lo, p_hi), computed ONCE on
        the main card by the model's own rotary module (before any chunk is in
        flight) and copied to card A: per-chunk slices then match the
        sequential path exactly and no stage touches the other card."""
        pid = torch.arange(p_lo, p_hi, device=self.dev_b).view(1, 1, -1).expand(4, 1, -1)
        cos, sin = self.rot_b(h_like, pid[1:])
        tb = (cos, sin, pid[0])
        ta = tuple(to_device(t, self.dev_a) for t in tb)
        return {self.dev_a: ta, self.dev_b: tb}

    @staticmethod
    def _pos(tab, a: int, b: int):
        cos, sin, tp = tab
        return (cos[:, a:b], sin[:, a:b]), tp[:, a:b]

    def _run_layers(self, h, lo, hi, pe, tpid, past, capture, feats):
        for i in range(lo, hi):
            h = self.layers[i](h, position_embeddings=pe, attention_mask=None,
                               position_ids=tpid, past_key_values=past, use_cache=True)
            if i in capture:
                feats[i] = h
        return h

    def _chunk(self, c, ids, p0, past, capture, sa: _Stage, sb: _Stage, tabs, off):
        """One chunk through both stages. On ANY failure (here or in an earlier
        chunk) the chunk still passes through both stages in order without
        work, so later chunks never wait forever on a turn that is never
        given; the caller re-raises the first error."""
        st = {"a": False, "b": False}
        try:
            if self._fail:
                raise RuntimeError("pipelined prefill: an earlier chunk failed")
            return self._chunk_work(c, ids, p0, past, capture, sa, sb, tabs, off, st)
        except BaseException:
            self._fail = True
            raise
        finally:
            for k, stage in (("a", sa), ("b", sb)):
                if not st[k]:
                    stage.enter(c)
                    stage.leave(c)

    @torch.no_grad()      # grad mode is per thread: the workers do not inherit the caller's
    def _chunk_work(self, c, ids, p0, past, capture, sa, sb, tabs, off, st):
        from qv_band.kv import fused_attn
        fused_attn._TLS.require = True        # worker thread: fused attention only
        n = ids.shape[-1]
        feats: dict = {}
        # Embedding BEFORE taking the card-A turn: the host-RAM lookup decodes
        # rows on the CPU (~0.15 s per 1024 tokens) and would otherwise hold
        # card A idle at the start of every chunk; here it overlaps the
        # previous chunk's card-A work. The H2D copy writes a fresh tensor, so
        # its position in card A's queue does not matter.
        prev = torch.cuda.current_device()
        emb = self.lm.embed_tokens
        edev = next(iter(emb.buffers()), None)
        edev = edev.device if edev is not None else self.dev_a
        h = emb(ids.to(edev))
        torch.cuda.set_device(self.dev_a)
        if h.device.type == "cpu":            # pinned staging, reused per thread
            key = ("emb", threading.get_ident())
            b, ev = self._bufs.get(key, (None, None))
            if ev is not None:
                ev.synchronize()
            if b is None or b.numel() < h.numel() or b.dtype != h.dtype:
                b = torch.empty(h.numel(), dtype=h.dtype, pin_memory=True)
            stage = b[:h.numel()].view(h.shape)
            stage.copy_(h)
            h = stage.to(self.dev_a, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.dev_a))
            self._bufs[key] = (b, ev)
        else:
            h = to_device(h, self.dev_a)
        torch.cuda.set_device(prev)
        sa.enter(c)
        try:
            torch.cuda.set_device(self.dev_a)
            pe, tpid = self._pos(tabs[self.dev_a], off, off + n)
            h = self._run_layers(h, 0, self.split, pe, tpid, past, capture, feats)
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.dev_a))
            if _DBG:
                print(f"[pipe-dbg] chunk {c} stageA issued: A alloc "
                      f"{torch.cuda.memory_allocated(self.dev_a)/1e9:.2f} max "
                      f"{torch.cuda.max_memory_allocated(self.dev_a)/1e9:.2f} B alloc "
                      f"{torch.cuda.memory_allocated(self.dev_b)/1e9:.2f}", flush=True)
        finally:
            torch.cuda.set_device(prev)
            sa.leave(c)
            st["a"] = True
        # Boundary copy on a side stream: waits for THIS chunk's card-A work
        # only, not for the next chunk already queued behind it.
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(ev)
            host = self._host_buf(h)
            host.copy_(h, non_blocking=True)
            fa = {i: t.to("cpu", non_blocking=True) for i, t in feats.items()}
        self.copy_stream.synchronize()
        sb.enter(c)
        try:
            torch.cuda.set_device(self.dev_b)
            hb = host.to(self.dev_b, non_blocking=True)
            ev_h = torch.cuda.Event()
            ev_h.record(torch.cuda.current_stream(self.dev_b))
            self._bufs[threading.get_ident()] = (self._bufs[threading.get_ident()][0], ev_h)
            feats = {i: t.to(self.dev_b, non_blocking=True) for i, t in fa.items()}
            pe, tpid = self._pos(tabs[self.dev_b], off, off + n)
            self._run_layers(hb, self.split, len(self.layers), pe, tpid, past, capture, feats)
        finally:
            torch.cuda.set_device(prev)
            sb.leave(c)
            st["b"] = True
        if capture:
            return torch.cat([feats[i] for i in capture], dim=-1)[0]
        return None

    def run(self, ids, past, p0: int, bounds, capture_layers, on_done):
        """Prefill ids[:, b0:b1] for consecutive (b0, b1) in `bounds` (relative
        to absolute position p0). on_done(i, b0, b1, features) is called in
        chunk order from the calling thread; returning True aborts (chunks
        already issued still complete, so the cache stays consistent)."""
        capture = list(capture_layers or [])
        # Two stage threads + the caller share the GIL. After a stage blocks on
        # its card (codec certification reads a few bytes back), it must get the
        # GIL back fast or its card idles: the default 5 ms switch interval
        # measured ~37 ms bubbles per sync. Restored on exit.
        old_si = sys.getswitchinterval()
        sys.setswitchinterval(float(_os.environ.get("QV_PIPE_SWITCH", "0.0002")))
        try:
            return self._run(ids, past, p0, bounds, capture, on_done)
        finally:
            sys.setswitchinterval(old_si)

    def _run(self, ids, past, p0, bounds, capture, on_done):
        ids = ids.cpu()                       # once, before any chunk is in flight
        lo = bounds[0][0]
        dt = getattr(self.lm.embed_tokens, "_out_dtype", None) or self.lm.embed_tokens.weight.dtype
        h_like = torch.empty(1, 1, 1, dtype=dt, device=self.dev_b)      # rotary output dtype
        tabs = self._tables(h_like, p0 + lo, p0 + bounds[-1][1])
        sa, sb = _Stage(0), _Stage(0)
        self._fail = False
        futs = []
        nxt = 0
        stopped = False
        err = None
        for i in range(len(bounds)):
            while not stopped and nxt < len(bounds) and nxt <= i + _INFLIGHT - 1:   # chunks in flight
                b0, b1 = bounds[nxt]
                futs.append(self.pool.submit(self._chunk, nxt, ids[:, b0:b1], p0 + b0,
                                             past, capture, sa, sb, tabs, b0 - lo))
                nxt += 1
            if i >= len(futs):
                break
            try:
                f = futs[i].result()
            except BaseException as e:      # wait for the rest, then re-raise the first
                err = err or e
                stopped = True
                continue
            finally:
                # Drop the finished future: it holds the chunk's drafter features
                # (1 x chunk x 5*hidden, ~52 MB per 1024 tokens). Keeping every
                # future alive grew the main card ~50 KB per prompt token.
                futs[i] = None
            if err is None and on_done(i, bounds[i][0], bounds[i][1], f) and not stopped:
                stopped = True      # chunks already issued still finish and report
        if err is not None:
            raise err
        return nxt
