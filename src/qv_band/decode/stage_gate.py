"""Stage gates for overlapped prefill over N cards (pipeline placement, run/placement.py).

Each prefill round ("chunk") runs the model's own forward in its own thread, holding a ticket (its round number).
The pipeline hooks (run_inference._install_pipeline_hooks) call enter(stage, ticket) before the first layer of a card
and leave(stage, ticket) after its last layer. A card admits rounds strictly in ticket order, so every layer sees
round c before round c + 1 -- the KV evolves exactly as in a sequential loop and the result is identical -- while
round c + 1 can run on card i as soon as round c has moved on to card i + 1.

leave() first runs the callback registered for (stage, ticket), if any: a prefix checkpoint snapshots THAT card's
layers there, the only moment they hold exactly the state after round `ticket`.
A forward without a ticket (every other caller) is never gated.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

_TLS = threading.local()


def ticket():
    return getattr(_TLS, "ticket", None)


@contextmanager
def as_ticket(t: int):
    prev = getattr(_TLS, "ticket", None)
    _TLS.ticket = t
    try:
        yield
    finally:
        _TLS.ticket = prev


class StageGate:
    def __init__(self, n_stages: int, stage_layers: list[tuple[int, int]]):
        self.n = n_stages
        self.stage_layers = stage_layers           # [(first layer, end layer)] per card
        self._cv = threading.Condition()
        self._next = [0] * n_stages
        self._issued = 0
        self.on_leave: dict = {}                   # (stage, ticket) -> callable, run inside leave()

    def new_run(self, n_rounds: int) -> int:
        """Reserve tickets for n_rounds rounds; every card expects the first of them next. Returns the first."""
        with self._cv:
            t0 = self._issued
            self._issued += n_rounds
            self._next = [t0] * self.n
            self._cv.notify_all()
            return t0

    def enter(self, s: int, t: int) -> None:
        with self._cv:
            self._cv.wait_for(lambda: self._next[s] == t)

    def leave(self, s: int, t: int) -> None:
        cb = self.on_leave.pop((s, t), None)
        try:
            if cb is not None:
                cb()
        finally:
            with self._cv:
                self._next[s] = t + 1
                self._cv.notify_all()

    def release(self, t: int) -> None:
        """A round that failed (or is skipped after an earlier failure) still passes every card it has not left, so
        later rounds never wait for a turn that is never given."""
        for s in range(self.n):
            with self._cv:
                pending = self._next[s] <= t
            if pending:
                self.enter(s, t)
                self.on_leave.pop((s, t), None)
                with self._cv:
                    self._next[s] = t + 1
                    self._cv.notify_all()
