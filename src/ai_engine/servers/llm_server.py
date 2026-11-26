"""Local HTTP API for a QV-band packed model (.qvband).

The model costs ~4 s to load and ~6 GB resident, so it is loaded once at
startup and shared by every request; a per-request load would dominate the
response time entirely.  Generation itself is serialised behind a lock,
because two concurrent forwards would each stream the full 6.7 GB of packed
weights and thrash a machine whose measured RAM bandwidth is ~7 GB/s -- they
would not run twice as fast, they would run more than twice as slow.

Built on http.server rather than FastAPI/Flask: neither is installed, and at
~0.6 tok/s the framework overhead is far below the noise floor.  Adding a
dependency here would buy nothing.

Endpoints
    GET  /health                  liveness plus what is loaded
    GET  /v1/models               OpenAI-shaped model list
    POST /v1/completions          {"prompt": ..., "max_tokens": ...}
    POST /v1/chat/completions     {"messages": [...], "max_tokens": ...}
    POST /generate                {"prompt"|"messages", ...}  native form

The two /v1 routes follow the OpenAI request/response shape so existing
clients work unchanged:

    curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
         -d '{"messages":[{"role":"user","content":"香港今日天氣點?"}],"max_tokens":60}'

Set "stream": true on any generate route for Server-Sent Events, which matters
at this speed -- a 200-token answer takes minutes, and you want the tokens as
they arrive rather than after.

  python -m ai_engine.servers.llm_server ./models/Qwen3.5-9B_qvb8_fp16_1e-3 --port 8080
  python -m ai_engine.servers.llm_server ./models/Qwen3.8-27B_qvb8_1e-3_fp16 --host 0.0.0.0

First argument is a self-contained pack folder (config, tokenizer and
weights all live there) -- just a model path, nothing else needed to pick
a model. Either pack format works: the folder's manifest names it and the
format registry supplies the readers, so nothing here branches on the
codec.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import re
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from ai_engine.web.base_handler import QVHTTPServer

from ai_engine.web.base_handler import QVHTTPHandler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ai_engine.runtime.sampling import resolve_sampling
from ai_engine.runtime.memlimit import rss_gb, vram_abort_check
from ai_engine.generation.gen_core import (
    DEFAULTS, GenParams, add_generation_args,
    params_from_args, build_drafter, run_generate)
from ai_engine.generation.loader import load_qvb8_llm
from ai_engine.servers.llm_server_stream import (StreamMixin, _SSEOut, _StreamState,  # noqa: F401
                                                 _ThinkSplitter)

MODEL_ID = "qv-band-local"

# All generation defaults (ctx size, output budget, KV mode/tau/entropy,
# drafter, sampling, memory ceiling) live in ai_engine.generation.gen_core.DEFAULTS
# -- the ONE definition, shared with chat.py.

# STATE["params"] is the base GenParams built at load() from the launch
# flags; each request clones it with per-call overrides (max_tokens,
# temperature, ...). STATE["bundle"] is gen_core's ModelBundle (model,
# tok, resolved device). The loose per-field keys below are kept
# only for /health's JSON and backward-compatible reads.
# STATE["prefix_cache"]: single-conversation KV-cache reuse (see
# qv_band.decode.spec_decode.PrefixCache) across consecutive requests -- an
# OpenAI-style client resends the FULL message history every call, so a
# request whose tokens start with exactly what the last one left the cache
# holding skips re-prefilling that shared prefix, the same win llama.cpp's
# server gets from its single-slot cache. Saved/read behind STATE["lock"]
# (generation is already fully serialised there), reset on /v1/... when a
# request's tokens diverge -- PrefixCache itself detects that and simply
# does not reuse, so there is nothing to clear explicitly; a stale entry is
# just replaced by the next call's returned cache.
STATE: dict = {"model": None, "tok": None, "lock": threading.Lock(),
               "pack": None, "served": 0, "tokens": 0,
               "params": None, "bundle": None, "prefix_cache": None,
               "ctx_size": DEFAULTS.CTX_SIZE, "kv_tau": DEFAULTS.KV_TAU,
               "req_seq": 0, "req_seq_lock": threading.Lock()}

# Per-request sequence number, purely for log correlation -- lets an
# operator watching the console tell requests apart (e.g. confirm two
# Cline connections really overlapped once the batched scheduler lands,
# vs the current STATE["lock"]-serialised path where request N+1's
# prefill only starts once request N's own generation finishes). NOT a
# request id in the OpenAI-API sense (no such field exists on chat
# completion responses) -- purely a server-side log tag, assigned once
# per do_POST call under its own tiny lock so two concurrently-arriving
# requests never race onto the same number.
def _next_req_id() -> int:
    with STATE["req_seq_lock"]:
        STATE["req_seq"] += 1
        return STATE["req_seq"]

# Largest non-streaming response served. A non-streaming request holds an
# idle socket for the whole generation, and most HTTP clients apply a read
# timeout well under that; past this size the client is told to stream
# instead. Streaming has no such limit -- it is bounded by
# --max-output-tokens like any other request.
NONSTREAM_MAX_TOKENS = DEFAULTS.MAX_OUTPUT_TOKENS

# Seconds of scheduler idleness before the CUDA allocator cache is returned
# to the driver (once per idle period). 30 min by default so a pause between
# chat turns never pays the re-allocation; QV_IDLE_TRIM_S=0 trims at once.
IDLE_TRIM_S = float(os.environ.get("QV_IDLE_TRIM_S", "1800"))

# --no-scheduler / QV_NO_SCHEDULER=1: one switch, read once at start-up, the same on every machine (the dev GPU, V100,
# sm_80+ cards). On (default): the continuous-batching scheduler serves every request. Off: one request at a time
# behind STATE["lock"]. load() builds the scheduler or not from this value; a request takes its path once, on
# arrival (STATE["scheduler"] set or None) -- no step or token loop tests it.
_SCHEDULER_OFF = os.environ.get("QV_NO_SCHEDULER", "").lower() in (
    "1", "true", "yes", "on")


def _set_scheduler(enabled: bool) -> None:
    """Record the --no-scheduler decision in the module global load() reads."""
    global _SCHEDULER_OFF
    _SCHEDULER_OFF = not enabled


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# QV_REST_TRIM=1 (default): after every request, give back what the card holds
# for nothing between requests -- the allocator's cached blocks AND the CUDA
# local-memory reservation. The second is the large one on Volta: Triton cannot
# use tensor cores there, so fla's Gated-DeltaNet kernels and the fused qvb8
# attention spill registers (chunk_fwd_kernel_o: 28 KB of stack per thread),
# and the driver keeps that spill space reserved for every resident thread
# after the kernel ends -- measured +1.4 GB per card from the first request on,
# +2.0 GB after a long prompt, none of it visible to torch. Resetting the stack
# limit releases it; the next request re-grows it in a few ms. That part is
# skipped while CUDA graphs are on (QVB8_MLP_GRAPH, see _rest_trim). Runs inside each
# request's lock, never from a background thread.
_REST_TRIM = os.environ.get("QV_REST_TRIM", "1") != "0"
_CUDART = None


def _rest_trim() -> None:
    global _CUDART
    import ctypes
    import torch
    # Pipeline placement spans every visible card in this one process; a
    # single-card server owns only its current one -- touching another card
    # would create a stray context there.
    if os.environ.get("QV_PIPELINE", "").strip():
        devices = sorted({torch.device(d).index for d in os.environ["QV_PIPELINE"].split(",") if d.strip()})
    else:
        devices = [torch.cuda.current_device()]
    for d in devices:
        torch.cuda.synchronize(d)
    torch.cuda.empty_cache()
    if os.environ.get("QVB8_MLP_GRAPH", "0") == "1":
        # NEVER with CUDA graphs: a captured graph keeps the local-memory
        # window the context had at capture, and shrinking it underneath
        # makes the next replay fault the GPU (Xid 13, SKEDCHECK05_LOCAL_
        # MEMORY_TOTAL_SIZE, both cards, seen on 2x V100). With graphs on,
        # only the allocator trim above runs.
        return
    try:
        if _CUDART is None:
            _CUDART = ctypes.CDLL("libcudart.so.12")
        # cudaLimitStackSize = 0. Setting it (even to its default) makes the
        # driver drop the grown local-memory reservation. Per device.
        cur = torch.cuda.current_device()
        for d in devices:
            torch.cuda.set_device(d)
            rc = _CUDART.cudaDeviceSetLimit(ctypes.c_int(0), ctypes.c_size_t(1024))
            if rc != 0 and not getattr(_rest_trim, "_warned", False):
                _rest_trim._warned = True
                print(f"[serve] rest-trim: cudaDeviceSetLimit returned {rc} on "
                      f"cuda:{d}; local memory stays reserved", flush=True)
        torch.cuda.set_device(cur)
    except OSError:
        pass                    # no libcudart to call: the allocator trim above still ran


def _maybe_empty_cache(bundle) -> None:
    """Trim the CUDA caching allocator ONLY when it is actually holding a
    lot of reserved-but-unused memory -- i.e. fragmentation really
    happened. torch.cuda.empty_cache() is a full device sync (~ms, and it
    forces the next allocations to go back to the driver), so calling it
    after every request serialised the server and was a large part of why
    llm_server.py ran slower than chat.py. SGLang keeps a persistent KV pool and
    never frees; this is the middle ground -- free only the slack.
    """
    if not bundle.use_gpu:
        return
    import torch
    dev_mod = torch.xpu if bundle.gpu_dev == "xpu" else torch.cuda
    if _REST_TRIM and dev_mod is torch.cuda:
        _rest_trim()
        return
    try:
        reserved = dev_mod.memory_reserved()
        allocated = dev_mod.memory_allocated()
    except Exception:
        dev_mod.empty_cache()
        return
    slack = reserved - allocated
    # > 2 GiB of cached free blocks, and they are most of what is reserved:
    # a genuinely fragmented pool worth returning. A normal steady-state
    # request leaves far less and is skipped.
    if slack > 2 * 2**30 and slack > reserved * 0.5:
        dev_mod.empty_cache()


# ---------------------------------------------------------------------------
# Continuous-batching scheduler
# ---------------------------------------------------------------------------
# WHY THIS EXISTS: llm_server.py's default path runs speculative_generate()
# behind ONE global STATE["lock"] -- every request is fully serialised, so
# two concurrent Cline connections queue rather than share a forward pass.
# qv_band.decode.spec_decode.ContinuousBatchEngine is the real fix: a
# persistent decode loop that admits a NEW request into the
# currently-running batch at the top of every step (not "collect a batch,
# run it to completion" -- see that class's own docstring for why the
# earlier round-based design here still reproduced serialised-looking
# behaviour for any two requests that were not almost exactly
# simultaneous). The three classes below are the piece that makes llm_server.py's
# HTTP layer actually USE it: a single background thread owns the engine
# and calls step() in a tight loop -- checking the request queue before
# EVERY step, not just before a round -- while each HTTP handler thread
# pushes its own request in and blocks reading ITS OWN output queue, so a
# slow client never blocks a fast one's tokens from arriving, and the
# scheduler thread is the only thing that ever touches the model/cache.
#
# WHY IT LIVES HERE AND NOT IN qv_band.engine
# -------------------------------------------
# It was engine.batch_scheduler, then briefly qv_band.engine.batch_scheduler.
# Both were wrong, for the same reason: this is not a reusable primitive,
# it is one server's request-routing policy. There is exactly one caller
# (load() below constructs it, generate()/stream() submit to it, main()
# reports on it) and no second consumer will ever appear -- chat.py has no
# HTTP layer to route, and the pipelines are single-shot scripts.
#
# Its only qv_band contact is one lazy import of ContinuousBatchEngine in
# _run() below, and llm_server.py ALREADY reaches into that same module
# directly for _IncStreamer and _REQ_ID. So the "keep qv_band details out
# of the app layer" argument that justified the move cuts the other way
# here: the app layer was never insulated from spec_decode, and splitting
# the scheduler out only moved the STATE["scheduler"] contract across a
# package boundary it has no reason to cross. Inlining it also means the
# wiring in load() and the consumption in generate() can be read in one
# file, which is where a reader actually looks.
#
# SCOPE (matches ContinuousBatchEngine's own documented gaps): only
# qvb8_resident (BandedKVCache) requests use this path. int8/bf16
# KV caches have NO per-slot state built for them -- load() must keep routing
# those through the OLD STATE["lock"] path rather than silently degrading
# them under this scheduler, which would either crash (batch-size
# mismatch) or hang.
#
# NOT YET SUPPORTED, same as ContinuousBatchEngine: prefix-cache reuse
# across requests (every request prefills cold), abort/disconnect
# detection mid-generation for one slot without stopping the loop (a
# disconnect currently just lets that slot run to max_new/EOS with nobody
# reading its out_q -- wasted compute, not a hang, since drop happens on
# completion regardless of whether a reader remains).


def _batch_paused():
    """Context for code that runs the model outside the scheduler (vision):
    pauses the batch step loop if there is one, else a no-op."""
    sch = STATE.get("scheduler")
    return sch.exclusive() if sch is not None else contextlib.nullcontext()


class Overloaded(Exception):
    """The server cannot take this request now (waiting queue full, or it waited longer than --queue-wait): HTTP 429 /
    503 with Retry-After. Never a fault of the request itself, so a router tries another replica."""

    def __init__(self, msg: str, code: int = 429, retry_after: int = 5):
        super().__init__(msg)
        self.code, self.retry_after = code, retry_after


@dataclass
class Limits:
    """Admission control of the batch scheduler. 0 = no limit.

    WHY. The scheduler used to admit every queued request into the live batch at once; under a burst the only brake was
    the VRAM ceiling check inside a step, which fails the WHOLE batch. Now a request enters the batch only when it fits:

      max_live         requests decoding / prefilling together (slots). One request's speed falls as slots are added
                       (each verify step covers every live slot), and a one-card V100 cannot prefill two long prompts.
      max_live_tokens  sum over the live requests of prompt tokens + max output tokens: what their KV cache can grow
                       to. This is the memory bound (KV bytes are proportional to tokens), stated in the unit a
                       request is known by BEFORE it runs.
      max_queue        requests waiting for admission; one more is refused at once with 429 + Retry-After (a client
                       or router can go elsewhere instead of holding a socket open on a server that is hours behind).
      queue_wait_s     a request that waited this long without being admitted gets 503 + Retry-After.

    Order is strict first-in first-out: the head of the queue is admitted when it fits, nothing overtakes it (a short
    request never starves a long one). An EMPTY batch always admits the head, whatever its size, so a request larger
    than max_live_tokens still runs -- alone."""
    max_live: int = 0
    max_live_tokens: int = 0
    max_queue: int = 0
    queue_wait_s: float = 0.0

    def any(self) -> bool:
        return bool(self.max_live or self.max_live_tokens or self.max_queue or self.queue_wait_s)


@dataclass
class _PendingRequest:
    """One HTTP request waiting for admission into the live batch.
    `out_q` is this request's own output channel -- the handler thread
    blocks reading it; the scheduler thread is the only writer."""
    ids: object                     # torch.Tensor (1, prompt_len)
    max_new: int
    temperature: float
    top_k: int
    top_p: float
    req_id: int
    out_q: "queue.Queue" = field(default_factory=queue.Queue)
    # Wall-clock time this request entered the live batch, set on
    # admission -- used only to report its own tok/s when it finishes.
    t_admitted: float = 0.0
    t_submitted: float = 0.0        # entered the waiting queue (Limits.queue_wait_s counts from here)
    cost: int = 0                   # prompt tokens + max_new: what this request can add to the live KV


class _StreamDone:
    """Sentinel pushed to a request's out_q when its generation is
    complete (whether by EOS, max_new, or the engine stopping)."""
    __slots__ = ("stats",)

    def __init__(self, stats):
        self.stats = stats


class BatchScheduler:
    """One instance per server process. start() launches the background
    step-loop thread; submit() is what the HTTP handlers below call."""

    dead: str | None = None      # the reason, once the step-loop thread has ended for good (see _die)
    limits = Limits()            # admission control (see Limits); an instance gets its own in __init__
    refused = expired = 0
    _excl_waiting = 0            # callers waiting for the model outside the batch (image requests)

    def __init__(self, model, tok, drafter_factory, k: int,
                 kv_tau: float, kv_residual: int,
                 idle_poll_s: float = 0.01,
                 prefill_chunk: int | None = None, limits: "Limits | None" = None):
        self.limits = limits or Limits()
        self.refused = self.expired = 0          # counters for /health
        self.model = model
        self.tok = tok
        self.drafter_factory = drafter_factory
        self.k = k
        self.kv_tau = kv_tau
        self.kv_residual = kv_residual
        # Forwarded to the engine so --prefill-chunk applies to the
        # continuous-batching path too; without this it silently ran at
        # spec_decode's module default no matter what was asked for.
        self.prefill_chunk = prefill_chunk
        # How long to block waiting for the FIRST request when the engine
        # has no live slots at all (nothing to step()) -- irrelevant once
        # any slot is live, since the loop then checks the queue with a
        # zero-wait poll between every single step() instead (see _run):
        # a request must never wait for an in-progress generation, only
        # for the current step's forward (tens of ms), which is what
        # actually delivers real continuous batching.
        self.idle_poll_s = idle_poll_s
        self._q: "queue.Queue[_PendingRequest]" = queue.Queue()
        # Keyed by req_id, NOT slot bi -- ContinuousBatchEngine.step()
        # shifts every later slot's bi down by one whenever an earlier
        # slot finishes and is dropped (drop_slot's own index-shift
        # contract), so a bi captured at admission time goes stale the
        # instant any OTHER request completes. req_id is stable for the
        # request's whole lifetime.
        self._pending: dict[int, _PendingRequest] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._engine = None
        # The scheduler thread holds _model_lock for every drain+step(), so a
        # caller that must run the model itself (the vision route) can take
        # it via exclusive() and be sure no batched forward is in flight.
        # _excl_waiting makes the loop stand aside first: a bare Lock is not
        # FIFO and a tight step loop would otherwise re-take it forever.
        self._model_lock = threading.Lock()
        self._excl_waiting = 0
        self._excl_guard = threading.Lock()
        # Requests whose client is gone or whose stop string was hit: the step loop drops their
        # slots between steps. Without this a disconnected request kept decoding until max_new
        # (minutes), taking a share of every step from the requests still being read.
        self._cancel_ids: set[int] = set()
        self._queued_ids: set[int] = set()        # submitted, not yet admitted (guarded by _cancel_lock)
        self._cancel_lock = threading.Lock()
        # Set (to the reason) when the step-loop thread ended for good: every waiting request has been failed, new
        # ones fail at once, and the state endpoint answers 503 so a router stops sending here.
        self.dead: str | None = None

    @contextlib.contextmanager
    def exclusive(self):
        """Run the model from another thread while the step loop is paused.

        step() flips the shared text config to the "qv_varlen" attention and
        points every attention module's _qv_active_past at the engine's cache
        for the duration of the call. A forward from any other thread in that
        window is routed into the batch's cache: it reads/writes the other
        requests' KV, and the batch's own tokens get corrupted. Live slots
        keep their state while paused and resume afterwards."""
        with self._excl_guard:
            self._excl_waiting += 1
        try:
            with self._model_lock:
                yield
        finally:
            with self._excl_guard:
                self._excl_waiting -= 1

    def cancel(self, req_id: int) -> None:
        """Ask the step loop to drop request `req_id` (still queued, or already live) at its
        next opportunity. Safe from any thread; a request that already finished is ignored."""
        with self._cancel_lock:
            self._cancel_ids.add(req_id)

    def _cancel_snapshot(self) -> set:
        with self._cancel_lock:
            return set(self._cancel_ids)

    def _apply_cancels(self) -> None:
        with self._cancel_lock:
            ids = set(self._cancel_ids)
        if not ids or self._engine is None:
            return
        done = set()
        for req_id in ids:
            for bi, slot in enumerate(self._engine.slots):
                if slot is not None and slot.req_id == req_id:
                    self._engine._drop(bi)
                    self._engine.finished.pop(req_id, None)
                    self._pending.pop(req_id, None)
                    print(f"[scheduler] req #{req_id} cancelled -- slot freed, batch now "
                          f"{[x.req_id for x in self._engine.slots if x is not None]}",
                          file=sys.__stdout__, flush=True)
                    done.add(req_id)
                    break
        with self._cancel_lock:
            # Matched ids are done. An id that is neither live nor still queued belongs to a request
            # that already finished (the client left just as it completed): nothing left to cancel.
            self._cancel_ids -= done | (ids - self._queued_ids)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="batch-scheduler",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the step-loop thread to exit, and unblock any caller
        whose request is still sitting in the submit queue (never admitted
        into a live batch, so it would otherwise never receive anything on
        its out_q -- the caller's `out_q.get()` would block forever, since
        nothing shuts down and drains this queue on its own once the loop
        thread has exited).

        A request already admitted into the live batch (in self._pending,
        not self._q) is NOT touched here: the step-loop thread checks
        self._stop itself and is responsible for finishing or aborting
        those in flight before it exits.
        """
        self._stop.set()
        while True:
            try:
                req = self._q.get_nowait()
            except queue.Empty:
                break
            with self._cancel_lock:
                self._queued_ids.discard(req.req_id)
            req.out_q.put(RuntimeError("scheduler stopped before this "
                                       "request was admitted"))

    def join(self, timeout: float | None = None) -> None:
        """Wait for the step-loop thread to actually exit.

        stop() only SETS the flag -- the thread may still be mid-step()
        (a forward pass takes tens of ms). The interpreter is about to
        finalize, so the caller needs to know the thread is really gone
        before stdout is torn down; otherwise the daemon thread can print()
        into a finalizing interpreter and abort the process with
        "_enter_buffered_busy: could not acquire lock".
        """
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout)

    def pool_state(self) -> dict:
        """Load and restorable prefixes, for the router (servers/router.py). Read from the HTTP thread while the
        engine runs: a best-effort snapshot (list copies), never a lock on the scheduler."""
        from ai_engine.servers.pool_keys import prefix_key
        with self._q.mutex:                      # a plain list(deque) can meet a put/get from another thread
            queued = list(self._q.queue)
        load = {"queued": len(queued), "queued_prompt_tokens": sum(int(r.ids.shape[-1]) for r in queued),
                "queued_decode_tokens": sum(int(r.max_new) for r in queued), "running": 0,
                "running_prefill_tokens": 0, "running_decode_tokens": 0}
        prefixes, held = [], []
        eng = self._engine
        if eng is not None:
            pending = dict(eng._pending_prompts)
            for sl in list(eng.slots):
                if sl is None or sl.done:
                    continue
                load["running"] += 1
                p = pending.get(sl.bi)
                load["running_prefill_tokens"] += int(p.shape[-1]) if p is not None else 0
                load["running_decode_tokens"] += max(0, int(sl.max_new) - len(sl.out_ids))
            sh = getattr(eng, "_shared", None)
            if sh is not None:                      # the machine's pool: every entry by its last whole block
                from ai_engine.servers.pool_xfer import keyed_entries
                ke = keyed_entries(sh)
                prefixes.extend(sorted({(L, h) for _, L, h, _ in ke}))
                if not os.environ.get("QV_PREFIX_LOCAL_ONLY"):       # a split model's entries are not mirrored
                    held.extend(sorted({(h, n) for _, _, h, n in ke}))
            else:
                cache = self.__dict__.setdefault("_pool_key_cache", {})
                live = set()
                for e in list(eng._prefix_store):
                    t = e["tokens"]
                    live.add(id(t))
                    if id(t) not in cache:
                        cache[id(t)] = (t, prefix_key(list(t)))  # the list is kept with its key: its id stays unique
                    k = cache[id(t)][1]
                    if k is not None:
                        prefixes.append(k)
                for i in [i for i in cache if i not in live]:
                    del cache[i]
        lim = self.limits
        load["live_tokens"] = self.live_tokens()
        # what a router needs to stop sending before it is refused: "accepting" False = the waiting queue is full
        return {"load": load, "prefixes": prefixes, "held": held,
                "limits": {"max_live": lim.max_live, "max_live_tokens": lim.max_live_tokens,
                           "max_queue": lim.max_queue, "queue_wait_s": lim.queue_wait_s},
                "accepting": not (lim.max_queue and load["queued"] >= lim.max_queue),
                "refused": self.refused, "expired": self.expired}

    def submit(self, ids, max_new: int, temperature: float, top_k: int,
               top_p: float, req_id: int) -> "queue.Queue":
        """Push a request; returns the Queue the caller must block-read
        from. Each item read is either an int (a real token id, emit it)
        or a _StreamDone (generation over -- .stats holds the final
        Stats). Admitted into the LIVE batch at the top of the scheduler
        thread's very next step() -- never behind a whole other
        generation finishing first."""
        req = _PendingRequest(ids=ids, max_new=max_new, temperature=temperature,
                              top_k=top_k, top_p=top_p, req_id=req_id)
        req.t_submitted = time.time()
        req.cost = int(ids.shape[-1]) + int(max_new)
        self.check_queue()                       # Overloaded when the waiting queue is full
        if self.dead is not None:                # nobody would ever read the queue
            req.out_q.put(RuntimeError(f"the server's scheduler stopped ({self.dead}): restart the server"))
            return req.out_q
        with self._cancel_lock:
            self._queued_ids.add(req_id)
        self._q.put(req)
        if self.dead is not None:                # it died between the check and the put: _die may have missed this one
            self._fail_queued(RuntimeError(f"the server's scheduler stopped ({self.dead}): restart the server"))
        return req.out_q

    def check_queue(self) -> None:
        """Raise Overloaded when one more request may not wait here (Limits.max_queue). Called by submit, and by the
        HTTP handler BEFORE it sends any header, so a refusal is a clean 429."""
        lim = self.limits
        # image requests wait for the model outside the batch queue (exclusive()): they count as waiting too
        waiting = self._q.qsize() + self._excl_waiting
        if lim.max_queue and waiting >= lim.max_queue:
            self.refused += 1
            raise Overloaded(f"{waiting} requests are already waiting on this server (--max-queue "
                             f"{lim.max_queue}): try again shortly", 429, self.retry_after())

    def retry_after(self) -> int:
        """Seconds a refused client should wait: the time the work in front of it takes at the speed of the last
        finished request, between 1 and 120."""
        tps = getattr(self, "_last_tps", 0.0) or 10.0
        with self._q.mutex:
            ahead = sum(r.max_new for r in self._q.queue)
        ahead += sum(max(0, r.max_new) for r in list(self._pending.values()))
        return int(min(120, max(1, ahead / tps / max(1, self.limits.max_live or len(self._pending) or 1))))

    def live_tokens(self) -> int:
        return sum(r.cost for r in list(self._pending.values()))

    def _fits(self, req: "_PendingRequest") -> bool:
        """May `req` enter the live batch now? (Limits.) An empty batch always admits."""
        lim, n = self.limits, len(self._pending)
        if n == 0:
            return True
        if lim.max_live and n >= lim.max_live:
            return False
        return not (lim.max_live_tokens and self.live_tokens() + req.cost > lim.max_live_tokens)

    def _fail_queued(self, err: Exception) -> None:
        while True:
            try:
                req = self._q.get_nowait()
            except queue.Empty:
                return
            with self._cancel_lock:
                self._queued_ids.discard(req.req_id)
            req.out_q.put(err)

    def _die(self, e: BaseException) -> None:
        """The step loop cannot go on (it raised outside a step, or the engine could not be rebuilt after a failed
        step). Before the thread just ended: every client kept waiting on its queue and the server looked
        healthy. Now every admitted and queued request gets the error, and the server reports itself unusable."""
        self.dead = f"{type(e).__name__}: {e}"
        print(f"[scheduler] FATAL: the step loop ended: {self.dead} -- every request fails until the server is "
              f"restarted", file=sys.__stdout__, flush=True)
        err = RuntimeError(f"the server's scheduler stopped ({self.dead}): restart the server")
        for req in list(self._pending.values()):
            req.out_q.put(err)
        self._pending.clear()
        self._fail_queued(err)

    def _drain_new_requests(self) -> None:
        """Admit every currently-queued request into the engine's live
        batch, right before the next step() -- this is what makes
        admission happen at STEP granularity instead of round
        granularity: a request that arrives while other slots are
        mid-generation is picked up here on the very next iteration of
        the loop below (worst case one in-flight step's latency, not one
        whole generation)."""
        while True:
            # Strict FIFO with admission control: look at the head; a cancelled or expired head leaves the queue, a
            # head that does not fit the live batch STAYS (and nothing behind it is admitted before it).
            with self._q.mutex:
                head = self._q.queue[0] if self._q.queue else None
            if head is None:
                return
            with self._cancel_lock:
                cancelled = head.req_id in self._cancel_ids
            waited = time.time() - head.t_submitted
            expired = bool(self.limits.queue_wait_s and waited > self.limits.queue_wait_s and self._pending)
            if not cancelled and not expired and not self._fits(head):
                if not getattr(head, "_held_logged", False):
                    head._held_logged = True
                    print(f"[scheduler] req #{head.req_id} waits for room: {len(self._pending)} live "
                          f"(max {self.limits.max_live or 'none'}), {self.live_tokens()} live tokens + {head.cost} "
                          f"(max {self.limits.max_live_tokens or 'none'}); {self._q.qsize()} waiting",
                          file=sys.__stdout__, flush=True)
                return
            try:
                req = self._q.get_nowait()       # this thread is the only consumer: it is the head just looked at
            except queue.Empty:
                return
            with self._cancel_lock:
                self._queued_ids.discard(req.req_id)
                if req.req_id in self._cancel_ids:
                    self._cancel_ids.discard(req.req_id)
                    print(f"[scheduler] req #{req.req_id} cancelled before admission",
                          file=sys.__stdout__, flush=True)
                    continue
            if expired:
                self.expired += 1
                print(f"[scheduler] req #{req.req_id} waited {waited:.0f}s for a slot (--queue-wait "
                      f"{self.limits.queue_wait_s:g}): refused", file=sys.__stdout__, flush=True)
                req.out_q.put(Overloaded(f"waited {waited:.0f}s for a free slot on this server: try again", 503,
                                         self.retry_after()))
                continue
            thread_name = threading.current_thread().name
            print(f"[scheduler] admitting req #{req.req_id} on thread="
                  f"{thread_name}, prompt {req.ids.shape[-1]} tok, "
                  f"live_slots_before={len(self._engine.slots) if self._engine else 0}",
                  file=sys.__stdout__, flush=True)

            def _make_on_token(rq):
                def _on_token(tok_id):
                    rq.out_q.put(tok_id)
                return _on_token

            try:
                self._engine.add_request(
                    req.ids, max_new=req.max_new, temperature=req.temperature,
                    top_k=req.top_k, top_p=req.top_p, req_id=req.req_id,
                    on_token=_make_on_token(req))
            except Exception as e:
                # Report to THIS request and keep the loop alive. Uncaught,
                # the exception left _run(), killed the scheduler thread,
                # and every client already blocked on out_q.get() waited
                # forever with the server looking idle. The failed request
                # never entered self._pending, so only its own queue needs
                # the exception; running requests keep streaming.
                print(f"[scheduler] admit failed for req #{req.req_id}: {e!r}",
                      file=sys.__stdout__, flush=True)
                traceback.print_exc(file=sys.__stdout__)
                req.out_q.put(e)
                continue
            req.t_admitted = time.time()
            self._pending[req.req_id] = req
            live = [s.req_id for s in self._engine.slots if s is not None]
            print(f"[scheduler] batch now {len(live)} live: {live} "
                  f"(waiting in queue: {self._q.qsize()})",
                  file=sys.__stdout__, flush=True)

    def _new_engine(self):
        """The model's batch engine: a BatchTarget (adapter with its own forward) runs on
        engine/target_engine.TargetBatchEngine, every HF-skeleton model on ContinuousBatchEngine."""
        if getattr(self.model, "_qv_batch_target", False):
            from qv_band.engine.target_engine import TargetBatchEngine
            return TargetBatchEngine(self.model, self.tok, prefill_chunk=self.prefill_chunk)
        from qv_band.decode.spec_decode import ContinuousBatchEngine
        return ContinuousBatchEngine(
            self.model, self.tok, drafter_factory=self.drafter_factory,
            k=self.k, kv_tau=self.kv_tau, int8_residual_length=self.kv_residual,
            prefill_chunk=self.prefill_chunk)

    def _run(self) -> None:
        try:
            self._run_loop()
        except BaseException as e:               # noqa: BLE001 - the thread is ending either way: say so
            traceback.print_exc(file=sys.__stdout__)
            self._die(e)

    def _run_loop(self) -> None:
        from qv_band.decode.spec_decode import ContinuousBatchEngine
        print("[scheduler] continuous-batching engine thread started",
              file=sys.__stdout__, flush=True)
        _enter_home(self.model)              # this thread's kernels launch on the model's card, not on cuda:0
        self._engine = self._new_engine()
        engine = self._engine
        engine.cancel_check = self._cancel_snapshot
        engine.over_ceiling = vram_abort_check
        last_active = time.monotonic()
        trimmed = False
        while not self._stop.is_set():
            if engine.has_work():
                last_active = time.monotonic()
            if not engine.has_work():
                # Nothing live: block on the queue instead of busy-polling
                # an idle engine. As soon as one request arrives it is
                # admitted and stepping starts immediately.
                # (wait WITHOUT taking the request out: taking it and putting it back moved it behind requests that
                # arrived meanwhile, which broke the first-in first-out order admission control relies on)
                with self._q.not_empty:
                    if not self._q.queue:
                        self._q.not_empty.wait(0.5)
                    first = bool(self._q.queue)
                if not first:
                    # Idle long enough: hand the allocator's cached-but-unused
                    # blocks back to the driver once (weights stay resident).
                    if (not trimmed and
                            time.monotonic() - last_active >= IDLE_TRIM_S):
                        trimmed = True
                        try:
                            import gc
                            import torch
                            gc.collect()
                            if torch.cuda.is_available():
                                before = torch.cuda.memory_reserved()
                                torch.cuda.empty_cache()
                                print(f"[scheduler] idle {IDLE_TRIM_S:.0f}s: "
                                      f"released "
                                      f"{(before - torch.cuda.memory_reserved()) / 2**30:.1f} GiB "
                                      f"of cached GPU memory",
                                      file=sys.__stdout__, flush=True)
                        except Exception as e:
                            print(f"[scheduler] idle trim failed: {e!r}",
                                  file=sys.__stdout__, flush=True)
                    continue
                trimmed = False              # _drain_new_requests below admits it

            while self._excl_waiting and not self._stop.is_set():
                time.sleep(0.002)
            step_err = None
            with self._model_lock:
                try:
                    self._apply_cancels()
                except Exception as e:
                    # Dropping a slot touches the shared cache; a failure may leave it inconsistent, so it
                    # takes the same path as a failed step (fail the batch, rebuild the engine) instead of
                    # killing this thread and leaving every client waiting.
                    step_err = e
                if step_err is None:
                    self._drain_new_requests()
                    if not engine.has_work():
                        continue
                    try:
                        engine.step()
                    except Exception as e:
                        step_err = e
            if step_err is not None:
                e = step_err
                print(f"[scheduler] step failed: {e!r}", file=sys.__stdout__, flush=True)
                # The traceback pins every frame of the failed forward, and with them its GPU tensors (~1.3 GB
                # measured on a one-card V100 after an OOM). Clients only need the message: drop the frames.
                import traceback as _tb
                _tb.clear_frames(e.__traceback__)
                try:
                    e = type(e)(*e.args)
                except Exception:
                    e = RuntimeError(repr(e))
                step_err = None
                # A step-level failure can't be attributed to one slot
                # cleanly (the batched forward covers all of them) --
                # fail every currently-admitted request the same way the
                # old round model did on a round failure, then reset the
                # engine so a future request gets a clean cache instead
                # of continuing to drive whatever state the exception
                # left behind.
                for req in self._pending.values():
                    req.out_q.put(e)
                self._pending.clear()
                # Release the failed engine's KV cache BEFORE building
                # the replacement. Dropping the reference alone leaves
                # the old cache resident until Python happens to collect
                # it, so the two would briefly coexist -- and the most
                # likely reason to be here at all is an OOM, which is
                # exactly when that must not happen.
                self._engine = None
                engine = None
                try:
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                self._engine = self._new_engine()
                engine = self._engine
                engine.cancel_check = self._cancel_snapshot
                engine.over_ceiling = vram_abort_check
                continue

            # A slot that finished this step is already dropped from
            # engine.slots inside step() itself -- detect completion here
            # by req_id (stable across bi shifts, see _pending's own
            # comment) no longer appearing among the engine's live slots.
            self._reap_finished()

    def _reap_finished(self) -> None:
        """Push _StreamDone (carrying the request's final Stats) for any
        request the engine just finished and dropped this step (see
        ContinuousBatchEngine._drop, which stashes (req_id, out_ids,
        stats) into engine.finished right before removing the slot)."""
        engine = self._engine
        for req_id, (_, _out_ids, stats) in list(engine.finished.items()):
            req = self._pending.pop(req_id, None)
            del engine.finished[req_id]
            if req is not None:
                req.out_q.put(_StreamDone(stats))
                dt = time.time() - getattr(req, "t_admitted", time.time())
                n = len(_out_ids)
                if n and dt > 0:
                    self._last_tps = n / dt      # for Retry-After estimates
                live = [s.req_id for s in engine.slots if s is not None]
                # Cheap here: a completion is rare compared to a step.
                kv = ""
                try:
                    m = engine.past.memory_report() if engine.past else {}
                    if m.get("total_bytes"):
                        kv = f", kv now {m['total_bytes'] / 2**30:.2f}GiB"
                except Exception:
                    pass
                print(f"[scheduler] req #{req_id} done: {n} tok in {dt:.1f}s "
                      f"({n / max(dt, 1e-9):.1f} tok/s) -- "
                      f"batch now {len(live)} live: {live}{kv}",
                      file=sys.__stdout__, flush=True)


def load(model_dir: str, think: bool, params: "GenParams"):
    """Load the model and stash a base GenParams + ModelBundle in STATE.

    The three steps -- apply_kv_shadow, load_model, build_drafter -- live in
    ai_engine.generation.loader so every caller does them identically; this
    function is now just: load it, keep the result, print the server's banner.

    A drafter that cannot be built stops the server with an error (no fallback).
    """
    llm = load_qvb8_llm(model_dir, params, verbose=True)
    bundle, drafter_obj, shadow = llm.bundle, llm.drafter, llm.shadow

    print(f"[serve] backend {bundle.gpu_dev or 'cpu'}   pack {bundle.pack}",
          flush=True)
    if getattr(bundle.model, "_qv_batch_target", False):
        print(f"[serve] KV cache: the model adapter's own ({getattr(bundle.model, '_qv_kv_desc', 'bf16')})", flush=True)
    else:
        print(f"[serve] KV cache: {params.describe_kv()}", flush=True)

    STATE.update(
        model=bundle.model, tok=bundle.tok, pack=bundle.pack, think=think,
        bundle=bundle, params=params,
        cuda=bundle.use_gpu, gpu_dev=bundle.gpu_dev,
        # The drafter actually built: "dflash" falls back to n-gram for a
        # model that has no DFlash draft model.
        drafter_name=(params.drafter if params.drafter not in ("dflash", "n-gram_with_dflash")
                      or hasattr(drafter_obj, "fork") or drafter_obj is None
                      else "n-gram"),
        drafter_obj=drafter_obj,
        # loose keys kept for /health JSON + backward-compatible reads
        ctx_size=params.ctx_size, max_memory_gb=params.max_memory_gb,
        kv_cache=params.kv_cache, kv_tau=params.effective_kv_tau,
        int8_kv=params.int8_kv, **shadow,
        temperature=params.temperature, top_k=params.top_k, top_p=params.top_p)

    # Continuous-batching scheduler (real multi-request parallel serving)
    # -- only wired up when this server's own config matches what
    # speculative_generate_batched actually supports: qvb8_resident (the
    # per-slot KV cache rewrite this backs onto). Every other config keeps
    # using the old STATE["lock"]-serialised path unchanged -- see
    # BatchScheduler's own docstring above for the full scope rationale.
    # A fresh drafter instance is needed PER SLOT (drafters
    # are stateful/per-sequence), so the scheduler gets a FACTORY
    # (build_drafter called again on demand), not the single shared
    # drafter_obj every other path reuses.
    STATE["scheduler"] = None
    # Pipeline placement over N cards (QV_PIPELINE, --pipeline) is served by the scheduler too: every per-slot cache
    # tensor lives on its layer's card (BandedKVLayer._adopt_device / add_slot) and drafter taps are gathered on the
    # main card (unified_scheduler.md section 4).
    _pipeline = bool(os.environ.get("QV_PIPELINE", "").strip())      # N-card placement: served by the scheduler too
    if getattr(bundle.model, "_qv_batch_target", False):
        # engine/target_engine runs the model whole on one card (its prefixes do use the machine pool):
        # N-card placement and remote stages are BandedKVCache features.
        for _env, _flag in (("QV_PIPELINE", "--pipeline"), ("QV_STAGES", "--pipeline HOST:PORT")):
            if os.environ.get(_env, "").strip():
                raise SystemExit(f"{_flag}: not supported for a model adapter target ({type(bundle.model).__name__})")
    if params.qvb8_resident and not _SCHEDULER_OFF:
        from qv_band.decode.spec_decode import ckpt_every_for
        try:
            ckpt_every_for(int(params.prefill_chunk))
        except ValueError as e:
            raise SystemExit(str(e))
    # --prefix-state qvb8: the checkpointed linear-attention state at the MODEL's tau, read here from the pack manifest
    # and handed to the engine (QV_PREFIX_STATE_TAU, internal). Heads that cannot meet it in 16 bits stay exact.
    if os.environ.get("QV_PREFIX_STATE", "exact") == "qvb8" and not params.qvb8_resident:
        print("[prefix] --prefix-state qvb8 has no effect here: the KV cache is not qvb8 (no prefix checkpoints of the "
              "linear-attention state) -- exact", flush=True)
        os.environ["QV_PREFIX_STATE"] = "exact"
    if os.environ.get("QV_PREFIX_STATE", "exact") == "qvb8":
        try:
            with open(os.path.join(str(bundle.pack), "manifest.json")) as _f:
                _ptau = json.load(_f).get("tau")
        except OSError as e:
            raise SystemExit(f"--prefix-state qvb8: cannot read the pack manifest for the model's tau ({e})")
        if not _ptau:
            raise SystemExit(f"--prefix-state qvb8: {bundle.pack}/manifest.json has no tau")
        os.environ["QV_PREFIX_STATE_TAU"] = str(float(_ptau))
    else:
        os.environ.pop("QV_PREFIX_STATE_TAU", None)
    if os.environ.get("QV_PREFIX_STORE") == "shared":
        # One pool per machine for every server with the SAME configuration (decode/shared_prefix.py): its name comes
        # from everything that decides a saved slot's bytes.
        if not (params.qvb8_resident and not _SCHEDULER_OFF):
            raise SystemExit("--prefix-store shared needs the scheduler (--kv-cache qvb8, no --no-scheduler)")
        # A split model (remote stages): this pool holds the layers that run HERE, each stage worker keeps its layers in
        # its own machine's pool (start it with --prefix-store shared too). The pool's fingerprint and the transfer
        # signature carry the split, so a whole-model server never meets these partial entries.
        # ... and the tensor-parallel processes (engine/proc_tp): an entry then holds one share per rank, never readable by a
        # server with another rank count
        _split = (os.environ.get("QV_STAGES", "").strip(), os.environ.get("QV_LAYERS_HERE", "").strip(),
                  *((f"tp{os.environ['WORLD_SIZE']}",) if int(os.environ.get("WORLD_SIZE", "1")) > 1 else ()))
        from qv_band.decode.shared_prefix import config_tag, pack_signature
        from qv_band.decode.spec_decode import ckpt_every_for
        _dm = str(getattr(params, "draft_model", "") or "")
        os.environ["QV_PREFIX_SHARED_TAG"] = config_tag(
            pack_signature(str(bundle.pack)), os.environ.get("QV_PREFIX_STATE_TAU"), params.effective_kv_tau, int(params.kv_residual),
            int(params.prefill_chunk), ckpt_every_for(int(params.prefill_chunk)), str(getattr(bundle.model, "dtype", "")),
            os.environ.get("QV_PIPELINE", ""), params.drafter, pack_signature(_dm) if _dm else "",
            int(params.draft_ctx), int(params.draft_k), _split)
        # Between MACHINES only what decides a saved prefix's bytes must match: not the prefill chunk (checkpoints sit
        # on the grid, the same positions for 512 and 12288) nor the card layout.
        _xparts = (pack_signature(str(bundle.pack)), os.environ.get("QV_PREFIX_STATE_TAU"), params.effective_kv_tau,
                   int(params.kv_residual), ckpt_every_for(int(params.prefill_chunk)),
                   params.drafter,                   # (model dtype: cast at restore, shared_prefix.get)
                   pack_signature(_dm) if _dm else "",      # (drafter context length: speed only)
                   *((_split,) if any(_split) else ()))
        os.environ["QV_PREFIX_XFER_SIG"] = config_tag(*_xparts)
        print(f"[prefix] transfer signature {os.environ['QV_PREFIX_XFER_SIG']} = {_xparts}", flush=True)
        if not os.environ.get("QV_POOL_TOKEN"):
            print("[prefix] WARNING: QV_POOL_TOKEN is not set: /v1/pool/export, /piece and /pull are open to the LAN "
                  "(saved prefixes hold prompt token ids). Set the same token on servers, router and prefix store.",
                  flush=True)
        os.environ.setdefault("QV_PREFIX_SHARED_NAME", "qv_prefix_" + os.path.basename(str(bundle.pack).rstrip("/"))
                              + ("_main" + _split[1].rstrip(":").replace(":", "-") if any(_split) else ""))
        if any(_split):
            os.environ["QV_PREFIX_LOCAL_ONLY"] = "1"       # partial entries: not exported, not mirrored
        from qv_band.decode.shared_prefix import store_from_env
        try:
            store_from_env()                   # opens (or creates) the pool now: a mismatch stops the server at start
        except (ValueError, OSError) as e:
            raise SystemExit(f"--prefix-store shared: {e}")
        if os.environ.get("QV_PREFIX_BACKUP"):
            from ai_engine.servers.pool_xfer import start_backup_watch
            start_backup_watch(os.environ["QV_PREFIX_BACKUP"])
    if os.environ.get("QV_STAGES", "").strip():
        # Layers on another machine (--pipeline HOST:PORT,local): the scheduler is the only path that serves them.
        if not (params.qvb8_resident and not _SCHEDULER_OFF):
            raise SystemExit("--pipeline with a remote stage needs the scheduler (--kv-cache qvb8, no --no-scheduler)")
        from qv_band.decode.remote_stage import attach_from_env
        attach_from_env(bundle.model)
    _target = bool(getattr(bundle.model, "_qv_batch_target", False))     # adapter with its own forward
    if (params.qvb8_resident or _target) and not _SCHEDULER_OFF:
        # A DFlash drafter carries gigabytes of weights: every slot gets a
        # fork() of the one already built (shared weights, own context)
        # instead of loading the draft model once per concurrent request.
        if params.draft_k > 0 and hasattr(drafter_obj, "fork"):
            drafter_factory = drafter_obj.fork
        else:
            drafter_factory = (
                (lambda: build_drafter(params.drafter, bundle,
                                       max_ctx=params.draft_ctx, params=params))
                if params.draft_k > 0 else None)
        scheduler = BatchScheduler(
            bundle.model, bundle.tok, drafter_factory, params.draft_k,
            params.effective_kv_tau, params.kv_residual,
            prefill_chunk=params.prefill_chunk, limits=STATE.get("limits"))
        lim = scheduler.limits
        if lim.any():
            print(f"[serve] admission: at most {lim.max_live or 'any number of'} live requests, "
                  f"{lim.max_live_tokens or 'any number of'} live tokens (prompt + output), "
                  f"{lim.max_queue or 'any number'} waiting, wait limit {lim.queue_wait_s or 'none'} s", flush=True)
        else:
            print("[serve] WARNING: no admission limits (--max-live / --max-live-tokens / --max-queue / --queue-wait): "
                  "every request is admitted at once and a burst is stopped only by the memory ceiling, which fails "
                  "the whole batch", flush=True)
        scheduler.start()
        STATE["scheduler"] = scheduler
        print("[serve] continuous-batching scheduler ACTIVE "
              f"({'model adapter target: engine/target_engine' if _target else 'qvb8_resident'}) -- concurrent "
              "requests share batched forward passes", flush=True)
    else:
        # Serialised path: one request at a time behind STATE["lock"], using
        # the stock attention implementation. Chosen by --no-scheduler, or
        # forced when the config does not support the batched engine
        # (non-resident KV cache).
        print("[serve] continuous-batching scheduler NOT active "
              f"(qvb8_resident={params.qvb8_resident}, "
              f"pipeline={_pipeline}, disabled={_SCHEDULER_OFF}) "
              "-- requests are serialised behind STATE[\"lock\"]",
              flush=True)

    # QV_EXPANDABLE_AFTER_LOAD=1 (with PYTORCH_CUDA_ALLOC_CONF=
    # expandable_segments:False at launch): load into plain segments, then
    # serve with expandable ones. Loading interleaves resident weights with
    # decode temporaries; in ONE expandable segment the freed temporaries
    # become sub-page holes that can never be unmapped (measured 1.39 GB per
    # card stuck on the 27B), while plain segments free them whole. Serving
    # keeps expandable segments: without them decode ran 11.6 vs 19.6 tok/s.
    if bundle.use_cuda and os.environ.get("QV_EXPANDABLE_AFTER_LOAD") == "1":
        import torch
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    # sm_70: install the WMMA Gated DeltaNet prefill kernels once at load, so the batching scheduler's forward
    # (which never goes through speculative_generate, where they used to be installed) does not fall back to fla's
    # Triton kernels (V100: slow, and they ran out of memory on a full card). Idempotent.
    if bundle.use_cuda:
        try:
            from qv_band.kv.gdn_volta import install as _install_gdn
            _install_gdn()
        except Exception as _e:                    # noqa: BLE001 - optional accelerator
            print(f"[gdn] not installed at load: {_e!r}", flush=True)

    # Start at rest: return what loading left cached (decoded temporaries,
    # the drafter build) -- the same trim every request ends with.
    if bundle.use_cuda and _REST_TRIM:
        _rest_trim()
        if os.environ.get("QV_MEM_DEBUG"):
            import torch
            ms = torch.cuda.memory_stats()
            print("[mem-debug] " + "  ".join(
                f"{k}={ms.get(k, 0) / 1e9:.2f}GB" for k in (
                    "allocated_bytes.all.current", "reserved_bytes.all.current",
                    "inactive_split_bytes.all.current",
                    "reserved_bytes.large_pool.current",
                    "reserved_bytes.small_pool.current")), flush=True)
            from collections import Counter
            free_by = Counter()
            for seg in torch.cuda.memory_snapshot():
                free_by[(seg.get("stream"), seg.get("segment_pool_id"), seg.get("is_expandable"))] += (
                    seg["total_size"] - seg["allocated_size"])
            print("[mem-debug] free bytes by (stream, pool, expandable): " + ", ".join(
                f"{k}={v / 1e9:.2f}GB" for k, v in free_by.most_common(6)), flush=True)
            if os.environ.get("QV_MEM_DEBUG") == "3":
                from qv_band.decode.spec_decode import _mem_groups
                print("[mem-debug] " + _mem_groups(), flush=True)

    print(f"[serve] ready in {bundle.load_seconds:.1f}s, "
          f"RSS {rss_gb():.1f} GB", flush=True)


def _denormalise_tool_calls(msgs: list[dict]) -> list[dict]:
    """Undo the OpenAI-wire-format string-encoding of tool_calls[].function.
    arguments before handing messages to the checkpoint's own chat template.

    parse_tool_calls() (the server's response side) deliberately encodes
    arguments as a JSON string, because that's the OpenAI wire format every
    client -- Cline included -- expects and will echo straight back on the
    next turn. But this checkpoint's chat template does `arguments|items`
    on it directly (line 120), which requires an actual mapping, not a
    string -- that mismatch is exactly what raised "Can only get item pairs
    from a mapping." here. Only tool_calls need touching; everything else
    passes through untouched.
    """
    out = []
    for m in msgs:
        calls = m.get("tool_calls")
        if not calls:
            out.append(m)
            continue
        m = dict(m)
        new_calls = []
        for c in calls:
            fn = c.get("function", c)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    pass
            new_calls.append({**c, "function": {**fn, "arguments": args}})
        m["tool_calls"] = new_calls
        out.append(m)
    return out


def build_prompt(body: dict) -> str:
    """Chat messages or a raw prompt, to the exact string the model sees.

    The checkpoint's own chat template has a native tool-calling contract
    (the "tools" list renders as a <tools> system block, and it expects
    <tool_call><function=NAME><parameter=X>val</parameter></function>
    </tool_call> back) -- passing "tools" straight through to
    apply_chat_template is all that's needed on the request side; the
    matching piece is parse_tool_calls() on the response side.
    """
    tok = STATE["tok"]
    msgs = body.get("messages")
    if msgs:
        msgs = _denormalise_tool_calls(msgs)
        fmt = _chat_format()
        if fmt is not None and hasattr(fmt, "prepare_messages"):
            msgs = fmt.prepare_messages(msgs)     # model-specific message shaping (Ripple: content-part lists -> text)
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            tools=body.get("tools") or None,
            enable_thinking=bool(body.get("think", STATE.get("think", True))))
    prompt = body.get("prompt", "")
    if body.get("chat"):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(body.get("think", STATE.get("think", True))))
    return prompt


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# apply_chat_template(enable_thinking=True) puts the OPENING <think> tag
# into the PROMPT itself (Qwen's own template convention -- the model is
# primed straight into reasoning mode, token one), so the model's own
# completion only ever contains the reasoning text followed by a bare
# closing </think>, never the opening tag. _THINK_RE above requires both
# tags in the same string and therefore never matches real generations --
# confirmed live: a response streamed to Cline as literal visible text
# "...我需要先詢問用戶想要什麼幫助。 </think>\n\n您好！..." -- the opening
# tag was already consumed by the prompt, invisible to this regex, so
# _strip_think() silently did nothing every single call it was ever
# reached from, not just this one time.
_THINK_CLOSE_RE = re.compile(r"^.*?</think>", re.DOTALL)








def _chat_format():
    """The loaded model's own output format (model._qv_chat_format, e.g. Ripple: reasoning only when the output
    opens <think>, Hermes JSON tool calls), or None -> this server's built-in (Qwen) handling."""
    return getattr(STATE.get("model"), "_qv_chat_format", None)


def _strip_think(text: str) -> str:
    """Drop the model's <think>...</think> reasoning block from a response.

    enable_thinking=True (the --think default) makes the checkpoint emit its
    planning inside <think> tags before the actual answer; callers of this
    API want the answer, not the 900+-token scratchpad that produced it.

    Handles both shapes: a self-contained <think>...</think> pair (the
    non-chat-template path, or a caller that fed the opening tag back in)
    and a bare leading ...</think> (the normal chat-template case, where
    the opening tag lives in the PROMPT, not the completion -- see
    _THINK_CLOSE_RE above). Tries the full-pair form first so a response
    that genuinely contains both tags does not get over-matched by the
    bare-closing form (which would also swallow real content preceding an
    unrelated, later </think>-shaped string that was never opened).

    A THIRD shape -- no </think> anywhere -- means generation was cut off
    (max_tokens/EOS) WHILE still inside the reasoning block: this model's
    <think> can run 1200-5700+ tokens uncapped with no awareness of any
    limit, so hitting a real request's max_output_tokens mid-thought is a
    routine occurrence, not a rare edge case. Returning the raw text
    unchanged in that case (what a plain regex substitution does, since
    there is nothing to match) hands the client pure internal scratchpad
    with no actual answer in it at all -- worse than the original bug,
    since it looks exactly like a real, complete response with nothing to
    signal that generation never got to answer the question. Detect this
    shape explicitly and say so instead of guessing at partial content.
    """
    fmt = _chat_format()
    if fmt is not None:
        return fmt.strip_reasoning(text)
    if _THINK_RE.search(text):
        return _THINK_RE.sub("", text).strip()
    if "</think>" in text:
        return _THINK_CLOSE_RE.sub("", text).strip()
    return ("[thinking was cut off before an answer was produced -- try "
            "again with a larger max_tokens, or --no-think]")


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Split the checkpoint's <tool_call> blocks out of generated text.

    Matches the template's own contract (see build_prompt): each block is
    <tool_call><function=NAME><parameter=K>v</parameter>...</function>
    </tool_call>. Returns (remaining_text_with_blocks_removed, tool_calls),
    tool_calls in OpenAI's shape (id/type/function.name/function.arguments
    as a JSON string) so existing OpenAI-compatible clients need no special
    casing. Values that parse as JSON are kept structured; anything else is
    passed through as a string arg, matching what the template itself does
    in reverse (tojson for mappings/sequences, str() otherwise).
    """
    fmt = _chat_format()
    if fmt is not None:
        return fmt.parse_tool_calls(text)
    calls: list[dict] = []

    def _consume(m: re.Match) -> str:
        fm = _FUNCTION_RE.search(m.group(1))
        if not fm:
            return ""
        name = fm.group(1).strip()
        args: dict = {}
        for pname, pval in _PARAMETER_RE.findall(fm.group(2)):
            pval = pval.strip()
            try:
                args[pname.strip()] = json.loads(pval)
            except (json.JSONDecodeError, ValueError):
                args[pname.strip()] = pval
        calls.append({
            "id": f"call_{len(calls)}_{int(time.time()*1000)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
        return ""

    remaining = _TOOL_CALL_RE.sub(_consume, text).strip()
    return remaining, calls


class _ClientGone(Exception):
    """The streaming client disconnected mid-generation.

    Not an error: a cancelled request is normal (Cline does it routinely).
    Raised from the stdout shim in _stream() to unwind speculative_generate,
    which otherwise runs to max_tokens for a client that will never read the
    result -- holding the global generation lock the whole time.
    """


def _stop_list(body: dict) -> list[str]:
    stop = body.get("stop")
    if not stop:
        return []
    return [stop] if isinstance(stop, str) else [s for s in stop if s]


def _truncate_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    """Cut text at the earliest stop string.  The model still decodes to
    max_tokens/EOS regardless -- speculative_generate has no interruption
    hook, and adding one risks the same class of correctness bug the MTP
    head had -- but the client only ever sees output truncated exactly
    where a real stop would have landed, which is what it actually cares
    about. Returns (text, stopped)."""
    cut = min((text.index(s) for s in stops if s in text), default=None)
    if cut is None:
        return text, False
    return text[:cut], True


def _request_params(max_tokens: int, draft: int,
                    temperature: float | None, top_k: int | None,
                    top_p: float | None) -> "GenParams":
    """The server's base GenParams (from launch flags, in STATE["params"])
    cloned with this request's overrides. The KV-cache config is fixed at
    load time and never varies per request; only the output budget,
    draft-k and the sampling knobs do. GenParams.__post_init__ re-clamps
    max_output_tokens to ctx_size."""
    base: GenParams = STATE["params"]
    import dataclasses as _dc
    return _dc.replace(
        base,
        max_output_tokens=(max_tokens if max_tokens
                           else base.max_output_tokens),
        draft_k=(draft if draft is not None else base.draft_k),
        temperature=(temperature if temperature is not None
                     else base.temperature),
        top_k=(top_k if top_k is not None else base.top_k),
        top_p=(top_p if top_p is not None else base.top_p),
    )


def _timings(st, n_out: int, dt: float, n_prompt: int) -> dict:
    """llama.cpp-style per-request timings. Prefill (prompt tokens through the model until the first token's logits)
    and decode are reported SEPARATELY: a long prompt with a normal decode rate otherwise looks 'slow overall'.
    Prefill fields are absent where the engine does not time it (the batching scheduler shares prefill steps);
    decode_tok_s = generated tokens / (request time - prefill time)."""
    t = {"prompt_n": int(n_prompt), "predicted_n": int(n_out)}
    tp = float(getattr(st, "t_prefill", 0) or 0)
    npf = int(getattr(st, "n_prefill", 0) or 0)
    if tp > 0 and npf > 0:
        t.update(prefill_n=npf, prefill_ms=round(tp * 1000, 1), prefill_tok_s=round(npf / tp, 1))
    td = dt - tp if tp > 0 else dt
    if n_out > 0 and td > 0:
        t.update(decode_ms=round(td * 1000, 1), decode_tok_s=round(n_out / td, 2))
    return t


def _timings_str(t: dict) -> str:
    parts = []
    if "prefill_tok_s" in t:
        parts.append(f"prefill {t['prefill_n']} tok in {t['prefill_ms'] / 1000:.2f}s ({t['prefill_tok_s']:.0f} tok/s)")
    if "decode_tok_s" in t:
        parts.append(f"decode {t['decode_tok_s']:.1f} tok/s")
    return "  |  ".join(parts)


def _model_family() -> str:
    """The source model's name from the pack manifest (Qwen3.8-27B for every pack of it), cached; the pack folder's
    name when the manifest has none."""
    fam = STATE.get("family")
    if fam is None:
        pack = str(STATE.get("pack") or "")
        try:
            with open(os.path.join(pack, "manifest.json")) as f:
                fam = os.path.basename(str(json.load(f).get("hf_model", "")).rstrip("/"))
        except (OSError, ValueError):
            fam = ""
        fam = STATE["family"] = fam or os.path.basename(pack.rstrip("/"))
    return fam


def _enter_home(model) -> None:
    """Make the model's home card (the pipeline's main card) the CURRENT device of the calling thread. Triton and the
    qvb8 extension launch on the current device, which is cuda:0 in every new thread (github issue #1)."""
    try:
        from qv_band.run.placement import home_device
        home = home_device(model)
        if home is not None and home.type == "cuda":
            import torch
            torch.cuda.set_device(home)
    except Exception:                        # noqa: BLE001 - CPU / XPU backends have no current CUDA device
        pass


def generate(text: str, max_tokens: int, draft: int, on_token=None,
             temperature: float | None = None, top_k: int | None = None,
             top_p: float | None = None, think: bool = True,
             req_id: int = 0, client_gone=None) -> dict:
    """One generation, serialised.  Returns text and timing.

    `req_id`: the server-side log tag for this request. The non-streaming
    route used to print NOTHING -- no "arrived", no "done", no timing --
    so a long non-streaming answer looked exactly like a hung server.
    """
    bundle = STATE["bundle"]
    tok = bundle.tok          # model is reached via run_generate(bundle, ...)
    _enter_home(bundle.model)
    ids = tok(text, return_tensors="pt").input_ids
    if bundle.use_gpu:
        ids = ids.to(bundle.gpu_dev)
    drafter = STATE["drafter_obj"] if draft > 0 else None

    # Drafting is not tied to greedy: speculative_generate does real
    # speculative sampling whenever temperature > 0. The one KV-mode ->
    # kwargs mapping is GenParams.to_speculative_kwargs(), inside
    # run_generate; nothing here re-derives it.
    rp = _request_params(max_tokens, draft, temperature, top_k, top_p)

    # Scheduler active: the scheduler thread is the ONLY thread that may run the model. A forward pass on this HTTP
    # thread would fire the scheduler's layer-capture hooks and see its global attention-implementation switch
    # (wrong-size feature tensors, "ragged_kv_views" on an empty slot). Submit like the streaming route does, collect.
    sch = STATE.get("scheduler")
    if sch is not None:
        print(f"[serve] {_ts()} req #{req_id} scheduler path", file=sys.__stdout__, flush=True)
        out_q = sch.submit(ids, rp.max_output_tokens, rp.temperature or 0.0, rp.top_k or 0, rp.top_p or 1.0, req_id)
        out, st, t0 = [], None, time.time()
        t_peek = time.monotonic()
        try:
            while True:
                # A non-streaming client that hung up: stop decoding for nobody (the streaming route always did). Checked
                # once a second by the clock, not only when the queue is idle: tokens arrive several times a second.
                if client_gone is not None and time.monotonic() - t_peek >= 1.0:
                    t_peek = time.monotonic()
                    if client_gone():
                        raise _ClientGone()
                try:
                    item = out_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                if isinstance(item, Exception):
                    raise item
                if hasattr(item, "stats"):          # _StreamDone
                    st = item.stats
                    break
                out.append(int(item))
        except BaseException:
            sch.cancel(req_id)
            raise
        dt = time.time() - t0
        STATE["served"] += 1
        STATE["tokens"] += len(out)
        _decoded = tok.decode(out, skip_special_tokens=True)
        line = (f"[serve] {_ts()} req #{req_id} done: {len(out)} tok in {dt:.1f}s "
                f"({len(out)/max(dt, 1e-9):.2f} tok/s) [scheduler]")
        if st is not None and st.proposed:
            line += (f"  |  drafted {st.proposed}, accepted {st.accepted} ({st.acceptance:.0%}), "
                     f"{st.steps} steps, draft {st.t_draft:.1f}s verify {st.t_verify:.1f}s")
        print(line, file=sys.__stdout__, flush=True)
        return {
            "text": _strip_think(_decoded) if think else _decoded,
            "prompt_tokens": int(ids.shape[1]),
            "completion_tokens": len(out),
            "seconds": round(dt, 2),
            "tokens_per_second": round(len(out) / dt, 3) if dt else None,
            "accepted": st.accepted if st else 0, "proposed": st.proposed if st else 0,
            "timings": _timings(st, len(out), dt, int(ids.shape[1])) if st else {},
        }

    with STATE["lock"]:
        t0 = time.time()
        prefix_cache_in = STATE["prefix_cache"]
        try:
            # Invalidate up front -- see _stream's identical guard for why:
            # an exception here means the cache object was mutated in
            # place past what the OLD handle's token_ids can account for,
            # so that handle must never be reused again. A clean return
            # overwrites this with the fresh handle on the next line.
            STATE["prefix_cache"] = None
            out, st, STATE["prefix_cache"] = run_generate(
                bundle, drafter, ids, rp, stream=False,
                prefix_cache=prefix_cache_in, return_cache=True)
        finally:
            # Trim the allocator only if it is genuinely fragmented -- an
            # unconditional empty_cache() per request serialised the server
            # (see _maybe_empty_cache).
            _maybe_empty_cache(bundle)
        dt = time.time() - t0

    STATE["served"] += 1
    STATE["tokens"] += len(out)
    _decoded = tok.decode(out, skip_special_tokens=True)
    line = (f"[serve] {_ts()} req #{req_id} done: {len(out)} tok in {dt:.1f}s "
            f"({len(out)/dt:.2f} tok/s)")
    tm = _timings(st, len(out), dt, int(ids.shape[1]))
    if _timings_str(tm):
        line += "  |  " + _timings_str(tm)
    if st.proposed:
        line += (f"  |  drafted {st.proposed}, accepted {st.accepted} "
                 f"({st.acceptance:.0%}), {st.steps} steps, draft {st.t_draft:.1f}s verify {st.t_verify:.1f}s")
    print(line, file=sys.__stdout__, flush=True)
    return {
        "text": _strip_think(_decoded) if think else _decoded,
        "prompt_tokens": int(ids.shape[1]),
        "completion_tokens": len(out),
        "seconds": round(dt, 2),
        "tokens_per_second": round(len(out) / dt, 3) if dt else None,
        "accepted": st.accepted, "proposed": st.proposed,
        "timings": tm,
    }



VISION_BODY_BYTES = 32 << 20     # image payload allowance per request with --vision on (base64 data URLs)

class Handler(StreamMixin, QVHTTPHandler):
    LOG_TAG = "serve"

    # ~6 bytes/token is a generous upper bound for UTF-8 text plus JSON
    # structure overhead, so this never rejects anything the --ctx-size
    # check in do_POST (which runs the real tokenizer and is the
    # authoritative limit, but only fires AFTER the full body is read into
    # memory and AFTER build_prompt()/tokenization has already paid the
    # cost of a huge string) would have accepted.
    # text: ~6 bytes per token of --ctx-size; with --vision on also room for images sent as base64 data URLs (a 1 MP
    # photo is ~1.8 MB of base64: the text-only limit, 983 KB at 163840 tokens, refused every real photo)
    MAX_BODY = staticmethod(lambda: STATE["ctx_size"] * 6 + (VISION_BODY_BYTES if STATE.get("vision", True) else 0))

    def _send_busy(self, e: "Overloaded") -> None:
        """429 / 503 with Retry-After: the request is fine, this server has no room for it now."""
        raw = json.dumps({"error": str(e), "retry_after": e.retry_after}).encode()
        self.send_response(e.code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Retry-After", str(e.retry_after))
        if self.ENABLE_CORS:
            self.send_header("Access-Control-Allow-Origin", self._cors_value())
            self.send_header("Vary", "Origin")
        self._end_close()
        self.end_headers()
        self.wfile.write(raw)

    def _peer_closed(self) -> bool:
        """True when the client has closed its side (a peek that reads end-of-stream); never blocks."""
        try:
            import select
            # select first: the connection has a timeout, and a bare recv on it WAITS for that timeout (then raises a
            # timeout, which is an OSError) -- an open, silent client must read as "still there", at once
            r, _, _ = select.select([self.connection], [], [], 0)
            return bool(r) and self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True

    def _body_too_large_message(self, n: int, max_body: int) -> str:
        return (f"request body is {n} bytes, over the {max_body}-byte limit "
                f"derived from --ctx-size ({STATE['ctx_size']} tokens)"
                + (f" + {VISION_BODY_BYTES >> 20} MB for images" if STATE.get("vision", True) else ""))

    # -- routes -----------------------------------------------------

    def do_GET(self):
        from ai_engine.servers import pool_xfer
        if pool_xfer.handle_get(self, self.path):        # /v1/pool/export, /v1/pool/piece (another machine pulls)
            return
        if self.path.rstrip("/") == "/v1/pool/state":
            sch = STATE.get("scheduler")
            if sch is not None and sch.dead is not None:
                return self._send(503, {"error": f"scheduler stopped ({sch.dead}): restart the server"})
            st = sch.pool_state() if sch is not None else {"load": None, "prefixes": []}
            p = STATE.get("params")
            st.update({"model": os.path.basename(str(STATE.get("pack") or "")), "pack": str(STATE.get("pack") or ""),
                       "ctx_size": STATE.get("ctx_size"), "vision": STATE.get("vision", True),
                       "kv_tau": getattr(p, "kv_tau", None), "dtype": getattr(p, "dtype", None),
                       "scheduler": sch is not None})
            st.update(pool_xfer.state_extra())           # which machine's pool, transfer signature
            st["family"] = _model_family()               # the source model's name: the router's group
            return self._send(200, st)
        if self.path.rstrip("/") in ("/health", ""):
            vram = {}
            if STATE.get("gpu_dev"):
                import torch
                _b = (torch.xpu if STATE.get("gpu_dev") == "xpu"
                      else torch.cuda)
                vram = {
                    "max_memory_gb": STATE.get("max_memory_gb"),
                    "vram_allocated_gb": round(_b.memory_allocated(0) / 1e9, 2),
                    "vram_reserved_gb": round(_b.memory_reserved(0) / 1e9, 2),
                    # peaks since the process started (GiB): what was actually allocated by tensors,
                    # and what the allocator reserved from the GPU (the difference is cache/fragmentation)
                    "vram_peak_allocated_gib": round(_b.max_memory_allocated(0) / 2**30, 2),
                    "vram_peak_reserved_gib": round(_b.max_memory_reserved(0) / 2**30, 2),
                }
            return self._send(200, {
                "status": "ok", "model": os.path.basename(STATE["pack"]),
                "pack": STATE["pack"],
                "device": STATE.get("gpu_dev") or "cpu",
                "drafter": STATE.get("drafter_name", "n-gram"),
                "vision": STATE.get("vision", True),
                "ctx_size": STATE["ctx_size"],
                "max_output_tokens": (STATE["params"].max_output_tokens
                                      if STATE.get("params") else None),
                "kv_cache": STATE.get("kv_cache"),
                "kv_tau": STATE.get("kv_tau"),
                "kv_shadow": STATE.get("kv_shadow_mode"),
                "kv_shadow_mb": STATE.get("kv_shadow_mb"),
                "kv_shadow_global_mb": STATE.get("kv_shadow_global_mb"),
                # Context the shadow budget actually holds cached. Compare
                # against ctx_size to see whether the decode accelerator
                # applies at the length you are running, or quietly does
                # not -- the distinction the old fixed default hid.
                "kv_shadow_covers_tokens": STATE.get("kv_shadow_covers_tokens"),
                **vram,
                "rss_gb": round(rss_gb(), 2),
                "requests_served": STATE["served"],
                "tokens_generated": STATE["tokens"],
                "busy": STATE["lock"].locked(),
                "scheduler_active": STATE.get("scheduler") is not None,
            })
        if self.path.rstrip("/") == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model",
                 "created": 0, "owned_by": "local"}]})
        self._send(404, {"error": "not found"})

    def _vision_chat(self, body, max_tokens, think, temperature, top_k, top_p,
                     req_id: int = 0):
        """Chat completion whose messages carry images (non-streaming; a
        "stream": true request is handled by _vision_chat_stream)."""
        from ai_engine.generation import vision as _vision
        bundle = STATE["bundle"]
        if body.get("stream"):
            return self._vision_chat_stream(body, max_tokens, think, temperature, top_k, top_p,
                                            req_id=req_id)
        # The image route used to print NOTHING on a successful request: no
        # "arrived", no "done", no token count, no timing. The text route
        # prints all four, so an operator watching the console could not tell
        # a working vision request from a hung one -- the server looks
        # completely idle while it is happily generating. Print the same
        # req #N lines the text path prints, so the two are comparable.
        print(f"[serve] {_ts()} req #{req_id} arrived "
              f"(thread={threading.current_thread().name}), vision",
              file=sys.__stdout__, flush=True)
        t0 = time.time()
        # Progress heartbeat. The non-streaming image path used to be SILENT
        # for its whole run: the client got no bytes and the console got no
        # lines, so a 2-minute generation looked exactly like a hung server.
        # on_token fires per sampled token; print a line every 64 so the
        # operator can see it advancing (and see the rate settle) instead of
        # staring at a silent terminal.
        prog = {"n": 0, "last": t0}

        def _on_token(_tid: int) -> None:
            prog["n"] += 1
            if prog["n"] % 64 == 0:
                now = time.time()
                print(f"[serve] {_ts()} req #{req_id} vision progress: "
                      f"{prog['n']} tok in {now - t0:.1f}s "
                      f"({prog['n']/(now-t0):.2f} tok/s)",
                      file=sys.__stdout__, flush=True)
                prog["last"] = now

        def _on_progress() -> None:
            # Called after every prefill chunk: the image tower is loaded and
            # the prompt is being prefilled -- the slowest part of a cold
            # request, and previously a total silence.
            print(f"[serve] {_ts()} req #{req_id} vision prefill: "
                  f"{time.time() - t0:.1f}s elapsed",
                  file=sys.__stdout__, flush=True)

        try:
            with STATE["lock"], _batch_paused():
                ad = _vision.adapter(bundle, STATE.get("drafter_obj"))
                r = ad.generate(_denormalise_tool_calls(body["messages"]),
                                tools=body.get("tools"), think=think,
                                max_new=max_tokens,
                                temperature=0.0 if temperature is None else float(temperature),
                                top_k=20 if top_k is None else int(top_k),
                                top_p=0.95 if top_p is None else float(top_p),
                                on_token=_on_token, on_progress=_on_progress,
                                prefill_chunk=getattr(STATE.get("params"), "prefill_chunk", None))
                _maybe_empty_cache(bundle)
        except ValueError as e:
            print(f"[serve] {_ts()} req #{req_id} vision failed: {e}",
                  file=sys.__stdout__, flush=True)
            return self._send(400, {"error": str(e)})
        except Exception as e:
            print(f"[serve] {_ts()} req #{req_id} vision failed: {type(e).__name__}: {e}",
                  file=sys.__stdout__, flush=True)
            traceback.print_exc()
            return self._send(500, {"error": f"vision: {e}"})
        STATE["served"] += 1
        STATE["tokens"] += r["completion_tokens"]
        dt = time.time() - t0
        n_out = r["completion_tokens"]
        print(f"[serve] {_ts()} req #{req_id} done: {n_out} tok in {dt:.1f}s "
              f"({n_out/dt:.2f} tok/s), {r['prompt_tokens']} prompt tok, "
              f"{r.get('images', 0)} image(s)"
              + (f"  |  vision encode {r['timings']['vision_ms'] / 1000:.2f}s  |  " + _timings_str(r["timings"])
                 if r.get("timings") else ""),
              file=sys.__stdout__, flush=True)
        text = _strip_think(r["text"]) if think else r["text"]
        text, hit_stop = _truncate_at_stop(text, _stop_list(body))
        content, tool_calls = parse_tool_calls(text)
        message = {"role": "assistant", "content": content or None}
        finish = "stop" if hit_stop or r["completion_tokens"] < max_tokens else "length"
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish = "tool_calls"
        created = int(time.time())
        return self._send(200, {
            "id": f"chatcmpl-{created}", "object": "chat.completion",
            "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": r["prompt_tokens"],
                      "completion_tokens": r["completion_tokens"],
                      "total_tokens": r["prompt_tokens"] + r["completion_tokens"]},
            "timings": r.get("timings")})


    def do_POST(self):
        route = self.path.rstrip("/")
        req_id = _next_req_id()
        try:
            body = self._body()
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(400, {"error": f"bad JSON: {e}"})

        from ai_engine.servers import pool_xfer
        if pool_xfer.handle_post(self, route, body):     # /v1/pool/pull, /v1/pool/release
            return
        if route == "/v1/pool/keys":
            # the router's prefix keys of a request, from this server's own prompt builder and tokenizer
            if STATE["model"] is None:
                return self._send(503, {"error": "model still loading"})
            try:
                from ai_engine.servers.pool_keys import block_keys
                ids = STATE["tok"](build_prompt(body), return_tensors="pt").input_ids[0].tolist()
            except Exception as e:                  # noqa: BLE001 - a bad body is the client's error
                return self._send(400, {"error": f"keys: {e}"})
            return self._send(200, {"n_tokens": len(ids), "keys": block_keys(ids)})
        if route not in ("/generate", "/v1/completions", "/v1/chat/completions"):
            return self._send(404, {"error": "not found"})

        if STATE["model"] is None:
            return self._send(503, {"error": "model still loading"})

        # int(...) here can raise on a malformed value (e.g. "max_tokens":
        # "abc") -- that used to be unguarded, and an uncaught exception in
        # do_POST is NOT caught by BaseHTTPRequestHandler (it only catches
        # TimeoutError), so the request thread died silently and the client
        # just hung waiting for a response that was never coming, instead
        # of getting a clean 400. Every request-parsing step that can raise
        # belongs inside the try block below.
        parsed = False                           # True once the request is understood: later errors are the server's
        try:
            # absent -> the server's base output budget (GenParams default),
            # NOT ctx_size (that binding once decoded 200k tokens on a
            # runaway generation). Resolve it to the concrete number now so
            # every downstream check (the non-stream guard, the
            # finish_reason comparison) sees the SAME value the generation
            # will actually use. _request_params re-clamps to ctx_size.
            _req_max = int(body.get("max_tokens") or body.get("max_new_tokens") or 0)
            max_tokens = min(
                _req_max or STATE["params"].max_output_tokens,
                STATE["ctx_size"])
            # Speculative draft tokens per step. Default = the server's
            # launch --draft (STATE["params"].draft_k), NOT a hardcoded
            # 8 -- an operator who launched with --draft 0 to disable drafting
            # must get that for requests that omit "draft". A request may
            # still override per-call; 0 disables (no rollback, the simple
            # path -- a rejected draft costs a cache rebuild since the
            # linear-attention layers have no partial rollback).
            draft = int(body.get("draft", STATE["params"].draft_k))
            # A request may override temperature/top_k/top_p directly, and/or
            # override "think" (which also shifts the server's --think/
            # --no-think preset -- see resolve_sampling). Left unset, these
            # fall back to the server's configured preset inside
            # generate()/_stream(); they now apply to drafted runs too, since
            # speculative_generate verifies with speculative sampling when
            # temperature > 0 instead of forcing greedy.
            # OpenAI-compatible clients (the vLLM / SGLang convention) switch thinking with chat_template_kwargs:
            # the same switch as "think" (servers/tp_server.py reads both the same way)
            _ctk = body.get("chat_template_kwargs")
            if "think" not in body and isinstance(_ctk, dict) and "enable_thinking" in _ctk:
                body["think"] = bool(_ctk["enable_thinking"])
            temperature = body.get("temperature")
            top_k = body.get("top_k")
            top_p = body.get("top_p")
            if "think" in body:
                temperature, top_k, top_p = resolve_sampling(
                    bool(body["think"]), False, temperature, top_k, top_p)
            if route == "/v1/chat/completions" and not body.get("messages"):
                # "not in body" alone let "messages": [] through -- build_prompt
                # treats an empty list as falsy and silently falls back to the
                # (also empty) "prompt" field, which then fails deep inside
                # tokenization with an opaque numpy error instead of a clear
                # 400 naming the actual problem.
                return self._send(400, {"error": "messages required"})

            # Opt-in capture (LLM_DUMP_DIR): each DISTINCT system prompt once (by hash) + one tiny params line per request
            # (temperature/top_p/max_tokens/...). Used to learn what a real client (Cline/OpenCode) sends; never breaks serving.
            _dd = os.environ.get("LLM_DUMP_DIR")
            if _dd:
                try:
                    import hashlib, time as _t
                    os.makedirs(_dd, exist_ok=True)
                    _ms = body.get("messages") or []
                    for _m in _ms:
                        if _m.get("role") == "system":
                            _txt = _m["content"] if isinstance(_m["content"], str) else json.dumps(_m["content"])
                            _fn = os.path.join(_dd, "system_" + hashlib.sha1(_txt.encode()).hexdigest()[:12] + ".txt")
                            if not os.path.exists(_fn):
                                open(_fn, "w").write(_txt)
                    import gzip                       # full body (messages + tools) so a bad output can be replayed exactly
                    os.makedirs(os.path.join(_dd, "bodies"), exist_ok=True)
                    _bid = f"{int(_t.time() * 1000)}_{os.getpid()}"
                    with gzip.open(os.path.join(_dd, "bodies", _bid + ".json.gz"), "wt") as _g:
                        json.dump(body, _g)
                    with open(os.path.join(_dd, "requests.jsonl"), "a") as _f:
                        _f.write(json.dumps({"ts": int(_t.time()), "body": _bid, "route": route, "n_messages": len(_ms),
                                             **{k: body.get(k) for k in ("temperature", "top_p", "top_k", "max_tokens", "think", "stream")},
                                             "has_tools": bool(body.get("tools"))}) + "\n")
                except Exception:
                    pass
            think_enabled = bool(body.get("think", STATE.get("think", True)))
            # image_url content -> the vision route (generation/vision.py);
            # text-only requests never take it.
            from ai_engine.generation import vision as _vision
            if route == "/v1/chat/completions" and _vision.has_images(body["messages"]) and not STATE.get("vision", True):
                return self._send(400, {"error": "image input is disabled on this server (started with --vision off; use --vision on, or LLM_VISION=on)"})
            if route == "/v1/chat/completions" and _vision.has_images(body["messages"]):
                if STATE.get("scheduler") is not None:
                    STATE["scheduler"].check_queue()     # image requests obey --max-queue as text requests do
                return self._vision_chat(body, max_tokens, think_enabled,
                                         temperature, top_k, top_p, req_id=req_id)
            text = build_prompt(body)
            if not text.strip():
                return self._send(400, {"error": "empty prompt"})
            # Cap prompt size before any forward pass runs, the same way
            # llama.cpp's --ctx-size does: prefill memory (attention/KV for
            # the 8 full-attention layers, plus the prefill activations)
            # scales with sequence length, and max_tokens above only bounds
            # OUTPUT length. Without this, a
            # single oversized request -- an agent tool dumping a whole
            # project tree into one prompt, say -- has no limit on how much
            # memory prefill alone can consume.
            n_prompt = STATE["tok"](text, return_tensors="pt").input_ids.shape[1]
            if n_prompt > STATE["ctx_size"]:
                return self._send(400, {"error": (
                    f"prompt has {n_prompt} tokens, exceeds the configured "
                    f"context size of {STATE['ctx_size']} (--ctx-size). "
                    f"Send a shorter prompt or restart with a larger "
                    f"--ctx-size.")})
            if n_prompt + max_tokens > STATE["ctx_size"]:
                max_tokens = STATE["ctx_size"] - n_prompt
                if max_tokens < 1:
                    return self._send(400, {"error": (
                        f"prompt has {n_prompt} tokens: no room for an answer in the context size of "
                        f"{STATE['ctx_size']} (--ctx-size)")})
            stream_opts = body.get("stream_options") or {}
            stops_in = _stop_list(body)
            parsed = True
            sch = STATE.get("scheduler")
            if sch is not None:
                sch.check_queue()            # full: 429 now, before a streamed answer has sent its 200
            if body.get("stream"):
                return self._stream(text, max_tokens, draft, route,
                                    has_tools=bool(body.get("tools")),
                                    think=think_enabled,
                                    stops=stops_in,
                                    temperature=temperature, top_k=top_k,
                                    top_p=top_p, req_id=req_id,
                                    include_usage=stream_opts.get("include_usage", True))
            # A non-streaming response holds a fully idle socket until the
            # whole generation finishes, and most HTTP clients (Cline
            # included) give up on a read timeout before that, showing a
            # spinner that never resolves rather than an error. Pinging the
            # client from a background thread was tried and is worse: that
            # thread and the handler both write to self.wfile with no
            # coordination, so the response can interleave and become
            # unparseable. Past NONSTREAM_MAX_TOKENS the request is refused
            # with a pointer to streaming, which works at any length.
            if max_tokens > NONSTREAM_MAX_TOKENS:
                return self._send(400, {"error": (
                    f"non-streaming requests over {NONSTREAM_MAX_TOKENS} "
                    f"tokens are rejected -- the socket would sit idle long "
                    f"enough to trip client read timeouts. Set "
                    f"\"stream\": true instead.")})
            # "arrived" for the non-streaming text route. The streaming route
            # printed this; this one did not, so a long non-streaming answer
            # produced no console output at all until it finished.
            print(f"[serve] {_ts()} req #{req_id} arrived "
                  f"(thread={threading.current_thread().name}), "
                  f"prompt {n_prompt} tok",
                  file=sys.__stdout__, flush=True)
            r = generate(text, max_tokens, draft, temperature=temperature,
                        top_k=top_k, top_p=top_p, think=think_enabled,
                        req_id=req_id, client_gone=self._peer_closed)
        except Overloaded as e:
            print(f"[serve] {_ts()} req #{req_id} refused ({e.code}): {e}", file=sys.__stdout__, flush=True)
            return self._send_busy(e)
        except _ClientGone:
            print(f"[serve] {_ts()} req #{req_id} client left before the answer: generation stopped",
                  file=sys.__stdout__, flush=True)
            self.close_connection = True
            return
        except Exception as e:
            if not parsed and isinstance(e, (ValueError, TypeError, KeyError, AttributeError)):
                # a field of the wrong type ("max_tokens": "abc", "stream_options": 3): the client's error, and a
                # router must not try the same request on every other replica as it does for a 5xx
                return self._send(400, {"error": f"bad request: {type(e).__name__}: {e}"})
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

        stops = stops_in
        r["text"], hit_stop = _truncate_at_stop(r["text"], stops)

        if route == "/generate":
            return self._send(200, r)

        created = int(time.time())
        model_id = MODEL_ID
        usage = {"prompt_tokens": r["prompt_tokens"],
                 "completion_tokens": r["completion_tokens"],
                 "total_tokens": r["prompt_tokens"] + r["completion_tokens"]}
        finish = "stop" if hit_stop else (
            "length" if r["completion_tokens"] >= max_tokens else "stop")
        if route == "/v1/completions":
            return self._send(200, {
                "id": f"cmpl-{created}", "object": "text_completion",
                "created": created, "model": model_id,
                "choices": [{"text": r["text"], "index": 0,
                             "finish_reason": finish}], "usage": usage, "timings": r.get("timings")})
        fmt = _chat_format()
        if fmt is not None and hasattr(fmt, "split") and fmt.split(r["text"])[2]:
            r["text"] = fmt.strip_reasoning(r["text"])
        content, tool_calls = parse_tool_calls(r["text"])
        message = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish = "tool_calls"
        return self._send(200, {
            "id": f"chatcmpl-{created}", "object": "chat.completion",
            "created": created, "model": model_id,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": message}], "usage": usage, "timings": r.get("timings")})



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="self-contained pack folder to serve")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 keeps it on this machine; 0.0.0.0 exposes "
                         "it to the LAN with NO authentication")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--think", dest="think", action="store_true", default=True,
                    help="let the model reason before answering (default). "
                         "Slow: measured 900+ planning tokens per call. A "
                         "request may override it per-call with a \"think\" "
                         "field in the JSON body.")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="disable reasoning: the template emits a closed "
                         "<think></think> so the model answers at token one.")
    # device / memory / ctx-size / KV mode+tau+entropy / drafter / draft-k
    # / draft-ctx -- the ONE definition, shared with chat.py, in gen_core.
    add_generation_args(ap)
    ap.add_argument("--temperature", type=float, default=None,
                    help="default per --think preset (Qwen's recommendation). "
                         "0 = greedy. A request overrides per-call with a "
                         "\"temperature\" field.")
    ap.add_argument("--top-k", type=int, default=None,
                    help="top-K logits kept (temperature > 0 only).")
    ap.add_argument("--top-p", type=float, default=None,
                    help="nucleus threshold (temperature > 0 only).")
    from ai_engine.cli.kv_offload import add_flags as _add_kv_offload_flags
    from ai_engine.cli.pipeline_flags import add_flags as _add_pipeline_flags
    _add_kv_offload_flags(ap)
    _add_pipeline_flags(ap)       # applied before the engine import by the entry point (cli/kv_offload.py)
    ap.add_argument("--greedy", action="store_true",
                    help="plain greedy/deterministic decoding (== "
                         "--temperature 0).")
    ap.add_argument("--vision", choices=("on", "off"), default=os.environ.get("LLM_VISION", "on").lower(),
                    help="image input (image_url chat content). on (default): the vision tower is built lazily on "
                         "the first image request, so a text-only workload pays no memory for it; off: such "
                         "requests are refused with 400 and nothing vision-related is ever loaded. (Env: LLM_VISION.)")
    ap.add_argument("--vision-tower", choices=("gpu", "host", "cpu"), default=None,
                    help="where the vision tower waits between image requests: gpu (default) or pinned host RAM, moved to "
                         "the card only while it encodes a picture (a full 16 GB card: 27B on one V100). (Env: LLM_VISION_TOWER.)")
    ap.add_argument("--no-scheduler", dest="scheduler", action="store_false",
                    default=True,
                    help="do not start the continuous-batching scheduler; "
                         "serve requests one at a time behind a lock (slower "
                         "with several users). Works on every machine; decided "
                         "once at start-up. (Env: QV_NO_SCHEDULER=1.)")
    ap.add_argument("--max-live", type=int, default=0, metavar="N",
                    help="admission: at most N requests in the live batch (0 = no limit); the others wait in order")
    ap.add_argument("--max-live-tokens", type=int, default=0, metavar="T",
                    help="admission: sum of prompt + max output tokens over the live requests (0 = no limit): the bound "
                         "on what their KV cache can grow to. A request larger than T still runs, alone")
    ap.add_argument("--max-queue", type=int, default=0, metavar="N",
                    help="admission: at most N requests waiting; one more gets 429 + Retry-After (0 = no limit)")
    ap.add_argument("--queue-wait", type=float, default=0.0, metavar="S",
                    help="admission: a request not admitted after S seconds gets 503 + Retry-After (0 = waits for ever)")
    args = ap.parse_args()
    STATE["limits"] = Limits(max(0, args.max_live), max(0, args.max_live_tokens), max(0, args.max_queue),
                             max(0.0, args.queue_wait))
    _set_scheduler(args.scheduler)

    args.temperature, args.top_k, args.top_p = resolve_sampling(
        args.think, args.greedy, args.temperature, args.top_k, args.top_p)

    # One GenParams from the launch flags -- llm_server.py holds it in
    # STATE["params"] and each request clones it (see _request_params).
    params = params_from_args(args)
    STATE["vision"] = (args.vision != "off")
    if args.vision_tower is not None:
        os.environ["LLM_VISION_TOWER"] = args.vision_tower
    # Bind the socket BEFORE loading the model: loading a 27B pack takes
    # minutes and ~24 GB of VRAM, and a port clash discovered afterwards
    # wastes all of that (the process then dies with "Address already in
    # use" only after the whole model is resident). Fail fast instead.
    if int(os.environ.get("RANK", "0")) > 0:
        # torchrun, a card other than rank 0 (engine/proc_tp): it never serves HTTP -- load() runs this card's share
        # and then rank 0's commands until it quits (the adapter's build_target does not return here)
        load(args.model, args.think, params)
        return
    try:
        srv = QVHTTPServer((args.host, args.port), Handler)
        from qv_band.common import tls as _tls
        _tls.wrap_server(srv)                 # TLS when the deployment sets QV_TLS_* (qv_band/common/tls.py)
    except OSError as exc:
        print(f"[serve] cannot bind {args.host}:{args.port}: {exc}",
              file=sys.stderr, flush=True)
        print("[serve] another server is probably already running on that "
              "port -- stop it, or pass -p <port>.", file=sys.stderr,
              flush=True)
        raise SystemExit(1) from exc
    load(args.model, args.think, params)

    print(f"[serve] http://{args.host}:{args.port}")
    print("  GET  /health")
    print("  POST /v1/chat/completions   (OpenAI-shaped)")
    print("  POST /v1/completions")
    print("  POST /generate")
    if STATE["scheduler"] is not None:
        print("\nContinuous-batching scheduler active: concurrent requests "
              "share batched forward passes (see [serve] req #N log lines "
              "to confirm overlap).\nCtrl-C to stop.", flush=True)
    else:
        print("\nGeneration is serialised: concurrent requests queue rather "
              "than compete for memory bandwidth.\nCtrl-C to stop.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopping")
        srv.shutdown()
    finally:
        # Shut the scheduler's step-loop thread down BEFORE the interpreter
        # starts tearing down: it is a daemon thread, and if it is still
        # alive at exit it may try to print() to a stdout that is already
        # finalizing -> "Fatal Python error: _enter_buffered_busy" and an
        # aborted core dump. stop() also drains queued-but-never-admitted
        # requests so no HTTP handler is left blocked on out_q.get().
        sched = STATE.get("scheduler")
        if sched is not None:
            try:
                sched.stop()
                sched.join(timeout=5.0)
            except Exception:
                traceback.print_exc()
        srv.server_close()


if __name__ == "__main__":
    main()
