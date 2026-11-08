"""Background file-read prefetch for the pack-swap loop.

Profiling the real fp8 27B load (497 tensors) found disk I/O is 24% of
from_packfile's own time (24s of 100s), the rest GPU compute (decode +
certify kernels) that a background thread cannot speed up. Overlapping
that 24% with the GPU work already in flight for the PREVIOUS tensor --
by having a background thread read the NEXT tensor's bytes while the
main thread is still inside from_packfile for the current one -- is the
only part of load time this can realistically shrink, and it costs zero
extra VRAM: the prefetch only ever holds raw file bytes on the CPU side,
never touches the GPU, and the swap loop's own control flow (which
kernel/class handles which tensor) is completely unchanged.
"""
from __future__ import annotations

import queue
import threading


class PackfilePrefetcher:
    """Reads pack files ahead of a caller that consumes them in order.

    Usage: give it the exact list of paths the caller will request, in
    the order it will request them (a swap loop's `tensors.items()` is
    already in manifest order, i.e. known in full up front). A single
    background thread reads them in that order into an unbounded
    result dict; no depth limit is needed since each tensor's GPU
    compute already takes longer than the next tensor's file read (see
    module docstring), so the reader thread never gets meaningfully far
    ahead of the consumer in practice. `get(path)` blocks until that
    path's bytes are ready, so calling `get` for paths out of the
    prefetched order still works (just without the overlap benefit) --
    the prefetcher is a pure speed optimization, never a correctness
    dependency.
    """

    # Max bytes read ahead of the consumer. Unbounded read-ahead buffered most
    # of the pack in host RAM (22 GB x 2 tensor-parallel ranks -> kernel OOM
    # on a 31 GB host); a few hundred MB keeps the I/O overlap.
    MAX_AHEAD_BYTES = 512 * 1024 * 1024

    def __init__(self, paths: list[str]):
        self._paths = list(paths)
        self._ahead = 0  # bytes read but not yet handed to get()
        # A path may legitimately appear more than once (e.g. a tied
        # embedding whose pack is read a second time for lm_head), so
        # results are queues-per-path, not a plain dict -- popping a
        # single value per path would starve the second `get()` call
        # for a duplicate that was only ever read once.
        self._results: dict[str, list] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._queue: queue.Queue[str | None] = queue.Queue()
        for p in self._paths:
            self._queue.put(p)
        self._queue.put(None)  # sentinel: stop the worker
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            path = self._queue.get()
            if path is None:
                return
            with self._cond:
                # Wait for the consumer to catch up (an empty buffer always
                # admits one file, so a single huge tensor cannot deadlock).
                while self._ahead >= self.MAX_AHEAD_BYTES:
                    self._cond.wait()
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                # Let the caller's own open() raise the real error --
                # this thread's only job is to warm the OS page cache /
                # have bytes ready sooner, never to be the sole reader.
                data = None
            with self._cond:
                self._results.setdefault(path, []).append(data)
                self._ahead += len(data) if data else 0
                self._cond.notify_all()

    def get(self, path: str) -> bytes | None:
        """Block until `path`'s bytes are ready, then return them (or
        None if the background read failed, in which case the caller
        should fall back to its own open())."""
        with self._cond:
            while not self._results.get(path):
                self._cond.wait()
            data = self._results[path].pop(0)
            self._ahead -= len(data) if data else 0
            self._cond.notify_all()
            return data

    def close(self):
        # Drain any unread paths, but keep re-queuing a sentinel so the
        # worker's next queue.get() always finds something -- draining
        # could otherwise pop the ORIGINAL sentinel itself, leaving the
        # worker blocked forever on an empty queue with nothing left to
        # wake it.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put(None)
