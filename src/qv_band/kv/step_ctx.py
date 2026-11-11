"""The continuous-batching engine's per-forward step context, kept per THREAD.

One forward of the engine carries a layout every layer needs: the packed segments (slot, offset, length) of a
packed step, each row's real length and which linear-attention rows start cold. It used to be written onto shared
objects (each cache layer, the cache, the linear-attention cache) for the duration of the forward. With several
prefill chunks in flight at once on different cards (multi-card overlapped prefill), each chunk's forward runs in
its own thread and needs its own layout: so it lives here, thread-local, set by the forward's caller.

    with step_ctx.active(segs=..., real_lens=..., cold_rows=...):
        model.model(...)
    step_ctx.current()        # inside the forward: the StepCtx of this thread, or None
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

_TLS = threading.local()


class StepCtx:
    __slots__ = ("segs", "real_lens", "cold_rows", "capture", "capture_dev", "capture_layers", "want_pre_norm",
                 "pre_norm")

    def __init__(self, segs=None, real_lens=None, cold_rows=None, capture=None, capture_dev=None,
                 capture_layers=None):
        self.segs = segs                    # packed step: [(slot, offset, length)]
        self.real_lens = real_lens          # padded ragged step: real tokens per row
        self.cold_rows = cold_rows          # per row: no linear-attention state yet
        self.capture = capture              # {layer index: hidden state} sink for drafter taps, or None
        self.capture_dev = capture_dev      # where captured hidden states go (the main card)
        self.capture_layers = frozenset(capture_layers or ())   # which tap layers write into `capture`
        self.want_pre_norm = False
        self.pre_norm = None                # the final norm's input, when asked for


def current() -> StepCtx | None:
    return getattr(_TLS, "ctx", None)


@contextmanager
def active(**kw):
    prev = getattr(_TLS, "ctx", None)
    ctx = StepCtx(**kw)
    _TLS.ctx = ctx
    try:
        yield ctx
    finally:
        _TLS.ctx = prev
