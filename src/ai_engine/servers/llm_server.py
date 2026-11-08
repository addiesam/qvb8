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

  python -m ai_engine.servers.llm_server /mnt/backup/models/Qwen3.5-9B_qvb8_fp16_1e-3 --port 8080
  python -m ai_engine.servers.llm_server /mnt/data/models/Qwen3.8-27B_qvb8_tau1e-3 --host 0.0.0.0

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
from http.server import ThreadingHTTPServer

from ai_engine.web.base_handler import QVHTTPHandler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ai_engine.runtime.sampling import resolve_sampling
from ai_engine.runtime.memlimit import rss_gb, vram_abort_check
from ai_engine.generation.gen_core import (
    DEFAULTS, GenParams, NO_ABORT, add_generation_args,
    params_from_args, build_drafter, run_generate)
from ai_engine.generation.loader import load_qvb8_llm

MODEL_ID = "qv-band-local"

# All generation defaults (ctx size, output budget, KV mode/tau/entropy,
# drafter, sampling, memory ceiling) live in ai_engine.generation.gen_core.DEFAULTS
# -- the ONE definition, shared with chat.py.

# STATE["params"] is the base GenParams built at load() from the launch
# flags; each request clones it with per-call overrides (max_tokens,
# temperature, ...). STATE["bundle"] is gen_core's ModelBundle (model,
# tok, resolved device, TP remap). The loose per-field keys below are kept
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

# ---------------------------------------------------------------------------
# Tensor-parallel request dispatch (torchrun, WORLD_SIZE>1)
# ---------------------------------------------------------------------------
# speculative_generate() itself is TP-safe (see spec_decode.py: every rank
# broadcasts its drafted/accepted token from rank 0 so every rank's model
# input is byte-identical every step -- otherwise GEMM reduction-order float
# noise between the physical GPUs desyncs the ranks and hangs forever). But
# it still needs every rank to CALL it, with the identical arguments, at
# the same time. Only rank 0 can own the HTTP socket (every rank binding
# the same port is nonsensical), so the other ranks have no request of
# their own to call it with -- this dispatcher is what gives them one.
#
# Rank 0: every generate()/streaming call goes through _tp_dispatch first,
# which broadcasts the call's arguments (a small picklable dict) to every
# other rank before running the real work locally.
# Rank 1+: after the model loads, main() puts these ranks into
# _tp_follower_loop instead of starting an HTTP server -- they block on the
# same broadcast, receive rank 0's arguments when a request arrives, and
# run the identical generate() call (its own local devices, its own
# drafter instance, but the arguments -- and, thanks to speculative_generate's
# own internal sync, ultimately the token stream -- match rank 0's exactly).
#
# A clean shutdown is a broadcast too (kind="stop"), so ranks 1+ exit their
# loop instead of being torchrun SIGTERM'd -- which would leave rank 0's
# next request hanging on a broadcast nobody answers.
_TP_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
_TP_ACTIVE = _TP_WORLD > 1

# Largest non-streaming response served. A non-streaming request holds an
# idle socket for the whole generation, and most HTTP clients apply a read
# timeout well under that; past this size the client is told to stream
# instead. Streaming has no such limit -- it is bounded by
# --max-output-tokens like any other request.
NONSTREAM_MAX_TOKENS = DEFAULTS.MAX_OUTPUT_TOKENS
_TP_RANK = int(os.environ.get("RANK", "0"))
_TP_IS_LEADER = (not _TP_ACTIVE) or _TP_RANK == 0

# Seconds of scheduler idleness before the CUDA allocator cache is returned
# to the driver (once per idle period). 30 min by default so a pause between
# chat turns never pays the re-allocation; QV_IDLE_TRIM_S=0 trims at once.
IDLE_TRIM_S = float(os.environ.get("QV_IDLE_TRIM_S", "1800"))

# --no-scheduler / QV_NO_SCHEDULER=1: serve with the STATE["lock"]-serialised
# path instead of the continuous-batching scheduler. Useful for debugging or
# comparing against the batched path. (It used to be REQUIRED on a box without
# flash_attn; the batched path now falls back to per-sequence SDPA there, just
# more slowly -- see qv_band spec_decode._qv_varlen_attention_forward.) Set in
# main() from the flag; the env var alone also works for callers that build
# GenParams by hand.
_SCHEDULER_OFF = os.environ.get("QV_NO_SCHEDULER", "").lower() in (
    "1", "true", "yes", "on")


def _set_scheduler(enabled: bool) -> None:
    """Record the --no-scheduler decision in the module global load() reads."""
    global _SCHEDULER_OFF
    _SCHEDULER_OFF = not enabled


def _tp_dispatch(kind: str, payload: dict) -> None:
    """Rank 0 only: broadcast one request's dispatch payload to every other
    rank before running it locally. No-op when not running under TP.

    Call it ONLY while holding STATE["lock"]: a dispatch sent while another
    request is generating lands between that request's model collectives, and
    the follower reads a spec_decode broadcast as its next dispatch."""
    if not _TP_ACTIVE:
        return
    import torch.distributed as dist
    box = [{"kind": kind, "payload": payload}]
    dist.broadcast_object_list(box, src=0)


def _tp_follower_loop() -> None:
    """Non-leader ranks: block for rank 0's next dispatched request (or
    stop signal) and run it identically. Runs for the process lifetime in
    place of serving HTTP."""
    import torch.distributed as dist
    print(f"[serve] rank {_TP_RANK} following rank 0's requests "
          f"(no HTTP socket on this rank)", flush=True)
    while True:
        box = [None]
        dist.broadcast_object_list(box, src=0)
        msg = box[0]
        if msg is None or msg.get("kind") == "stop":
            print(f"[serve] rank {_TP_RANK} stopping", flush=True)
            return
        payload = msg.get("payload", {})
        try:
            if msg["kind"] == "generate":
                generate(**payload, _tp_dispatched=True)
            elif msg["kind"] == "stream":
                # The SSE path emits bytes to a real client socket rank 0
                # owns; this rank has no socket to write to and does not
                # need one -- it only needs to run speculative_generate in
                # lockstep. _stream_follower reproduces exactly the model
                # call _stream_leader_side makes (same drafter resolution,
                # same speculative_generate arguments), discarding output.
                _stream_follower(**payload)
        except Exception as e:
            # A follower's exception must never propagate out of this loop
            # (there is no HTTP response to send it to, and an uncaught
            # exception here would kill this rank while rank 0 carries on
            # -- the next request's broadcast would then hang forever
            # waiting for a rank that no longer exists). Log and keep
            # following; rank 0's own try/except around the same call
            # reports the real error to the client.
            print(f"[serve] rank {_TP_RANK} follower error (continuing): "
                  f"{e!r}", flush=True)


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
# request's lock on every rank (all ranks call this at the same point), never
# from a background thread: a follower rank idles inside an NCCL broadcast,
# and a device-synchronising call there would deadlock.
_REST_TRIM = os.environ.get("QV_REST_TRIM", "1") != "0"
_CUDART = None


def _rest_trim() -> None:
    global _CUDART
    import ctypes
    import torch
    # Pipeline placement spans every visible card in this one process; a
    # tensor-parallel rank (or a single-card server) owns only its current
    # one -- touching another card would create a stray context there.
    if int(os.environ.get("QV_PIPELINE_SPLIT", "0") or 0) > 0:
        devices = list(range(torch.cuda.device_count()))
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
# qvb8_resident (BandedKVCache) + non-TP requests use this path. int8/bf16
# KV caches and tensor-parallel (WORLD_SIZE>1) have NO per-slot state / no
# batched-forward broadcast built for them yet -- load() must keep routing
# those through the OLD STATE["lock"] path rather than silently degrading
# them under this scheduler, which would either crash (batch-size
# mismatch) or (TP) desync ranks and hang forever.
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


class _StreamDone:
    """Sentinel pushed to a request's out_q when its generation is
    complete (whether by EOS, max_new, or the engine stopping)."""
    __slots__ = ("stats",)

    def __init__(self, stats):
        self.stats = stats


class BatchScheduler:
    """One instance per server process. start() launches the background
    step-loop thread; submit() is what the HTTP handlers below call."""

    def __init__(self, model, tok, drafter_factory, k: int,
                 kv_tau: float, kv_residual: int,
                 idle_poll_s: float = 0.01,
                 prefill_chunk: int | None = None):
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
        with self._cancel_lock:
            self._queued_ids.add(req_id)
        self._q.put(req)
        return req.out_q

    def _drain_new_requests(self) -> None:
        """Admit every currently-queued request into the engine's live
        batch, right before the next step() -- this is what makes
        admission happen at STEP granularity instead of round
        granularity: a request that arrives while other slots are
        mid-generation is picked up here on the very next iteration of
        the loop below (worst case one in-flight step's latency, not one
        whole generation)."""
        while True:
            try:
                req = self._q.get_nowait()
            except queue.Empty:
                return
            with self._cancel_lock:
                self._queued_ids.discard(req.req_id)
                if req.req_id in self._cancel_ids:
                    self._cancel_ids.discard(req.req_id)
                    print(f"[scheduler] req #{req.req_id} cancelled before admission",
                          file=sys.__stdout__, flush=True)
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

    def _run(self) -> None:
        from qv_band.decode.spec_decode import ContinuousBatchEngine
        print("[scheduler] continuous-batching engine thread started",
              file=sys.__stdout__, flush=True)
        self._engine = ContinuousBatchEngine(
            self.model, self.tok, drafter_factory=self.drafter_factory,
            k=self.k, kv_tau=self.kv_tau, int8_residual_length=self.kv_residual,
            prefill_chunk=self.prefill_chunk)
        engine = self._engine
        engine.cancel_check = self._cancel_snapshot
        last_active = time.monotonic()
        trimmed = False
        while not self._stop.is_set():
            if engine.has_work():
                last_active = time.monotonic()
            if not engine.has_work():
                # Nothing live: block on the queue instead of busy-polling
                # an idle engine. As soon as one request arrives it is
                # admitted and stepping starts immediately.
                try:
                    first = self._q.get(timeout=0.5)
                except queue.Empty:
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
                trimmed = False
                self._q.put(first)   # let _drain_new_requests below consume it uniformly

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
                self._engine = ContinuousBatchEngine(
                    self.model, self.tok, drafter_factory=self.drafter_factory,
                    k=self.k, kv_tau=self.kv_tau,
                    int8_residual_length=self.kv_residual,
                    prefill_chunk=self.prefill_chunk)
                engine = self._engine
                engine.cancel_check = self._cancel_snapshot
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

    llm_server.py is the ONE caller that degrades a heavy drafter to n-gram rather
    than dying: a running server must survive a drafter it cannot build.
    """
    llm = load_qvb8_llm(model_dir, params, verbose=True,
                        fallback_to_ngram=True)
    bundle, drafter_obj, shadow = llm.bundle, llm.drafter, llm.shadow

    print(f"[serve] backend {bundle.gpu_dev or 'cpu'}   pack {bundle.pack}",
          flush=True)
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
    # per-slot KV cache rewrite this backs onto) and non-TP (no rank-
    # broadcast built for the batched path yet). Every other config keeps
    # using the old STATE["lock"]-serialised path unchanged -- see
    # BatchScheduler's own docstring above for the full scope rationale.
    # A fresh drafter instance is needed PER SLOT (drafters
    # are stateful/per-sequence), so the scheduler gets a FACTORY
    # (build_drafter called again on demand), not the single shared
    # drafter_obj every other path reuses.
    STATE["scheduler"] = None
    # Pipeline placement (layers split across cards, QV_PIPELINE_SPLIT) runs
    # only on the serialised path: the batched engine builds its per-slot
    # tensors on one device and concatenates across layers that now live on
    # two (RuntimeError "Expected all tensors to be on the same device").
    _pipeline = int(os.environ.get("QV_PIPELINE_SPLIT", "0") or 0) > 0
    if params.qvb8_resident and not _TP_ACTIVE and not _SCHEDULER_OFF and not _pipeline:
        # A DFlash drafter carries gigabytes of weights: every slot gets a
        # fork() of the one already built (shared weights, own context)
        # instead of loading the draft model once per concurrent request.
        if params.draft_k > 0 and hasattr(drafter_obj, "fork"):
            drafter_factory = drafter_obj.fork
        else:
            drafter_factory = (
                (lambda: build_drafter(params.drafter, bundle,
                                       max_ctx=params.draft_ctx,
                                       fallback_to_ngram=True, params=params))
                if params.draft_k > 0 else None)
        scheduler = BatchScheduler(
            bundle.model, bundle.tok, drafter_factory, params.draft_k,
            params.effective_kv_tau, params.kv_residual,
            prefill_chunk=params.prefill_chunk)
        scheduler.start()
        STATE["scheduler"] = scheduler
        print("[serve] continuous-batching scheduler ACTIVE "
              "(qvb8_resident, non-TP) -- concurrent requests share "
              "batched forward passes", flush=True)
    else:
        # Serialised path: one request at a time behind STATE["lock"], using
        # the stock attention implementation. Chosen by --no-scheduler, or
        # forced when the config does not support the batched engine
        # (non-resident KV cache, tensor parallel).
        print("[serve] continuous-batching scheduler NOT active "
              f"(qvb8_resident={params.qvb8_resident}, TP={_TP_ACTIVE}, "
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


class _ThinkSplitter:
    """Splits a token stream at the model's </think> into reasoning and answer
    text. The tag can arrive split across pieces ("</th" + "ink>"), so while in
    the reasoning part the last len(tag)-1 characters are held back until the
    next piece shows they are not the start of the tag."""
    CLOSE = "</think>"

    def __init__(self, in_reasoning: bool):
        self.reasoning = in_reasoning
        self._hold = ""

    def feed(self, piece: str) -> list[tuple[str, str]]:
        """-> [("reasoning_content" | "content", text), ...]"""
        self._hold += piece
        out: list[tuple[str, str]] = []
        while self.reasoning:
            if self.CLOSE in self._hold:
                head, _, self._hold = self._hold.partition(self.CLOSE)
                if head:
                    out.append(("reasoning_content", head))
                self.reasoning = False
                continue
            keep = len(self.CLOSE) - 1
            if len(self._hold) > keep:
                out.append(("reasoning_content", self._hold[:-keep]))
                self._hold = self._hold[-keep:]
            return out
        if self._hold:
            out.append(("content", self._hold))
            self._hold = ""
        return out

    def finish(self) -> list[tuple[str, str]]:
        """Whatever is still held when the stream ends (an unclosed think block
        stays reasoning)."""
        text, self._hold = self._hold, ""
        if not text:
            return []
        return [("reasoning_content" if self.reasoning else "content", text)]


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


def generate(text: str, max_tokens: int, draft: int, on_token=None,
             temperature: float | None = None, top_k: int | None = None,
             top_p: float | None = None, think: bool = True,
             _tp_dispatched: bool = False, req_id: int = 0) -> dict:
    """One generation, serialised.  Returns text and timing.

    `_tp_dispatched`: set by _tp_follower_loop when it calls this on behalf
    of a broadcast from rank 0 -- skips re-broadcasting (which would hang
    waiting for a rank that already IS the one running this call) and
    skips STATE["served"]/["tokens"] bookkeeping (meaningless off the
    leader, which is the only rank an HTTP client ever talks to).

    `req_id`: the server-side log tag for this request. The non-streaming
    route used to print NOTHING -- no "arrived", no "done", no timing --
    so a long non-streaming answer looked exactly like a hung server.
    """
    bundle = STATE["bundle"]
    tok = bundle.tok          # model is reached via run_generate(bundle, ...)
    ids = tok(text, return_tensors="pt").input_ids
    if bundle.use_gpu:
        ids = ids.to(bundle.gpu_dev)
    drafter = STATE["drafter_obj"] if draft > 0 else None

    # Drafting is not tied to greedy: speculative_generate does real
    # speculative sampling whenever temperature > 0. The one KV-mode ->
    # kwargs mapping is GenParams.to_speculative_kwargs(), inside
    # run_generate; nothing here re-derives it.
    rp = _request_params(max_tokens, draft, temperature, top_k, top_p)

    with STATE["lock"]:
        if _TP_ACTIVE and not _tp_dispatched:
            _tp_dispatch("generate", dict(
                text=text, max_tokens=max_tokens, draft=draft,
                temperature=temperature, top_k=top_k, top_p=top_p, think=think))
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


def _stream_follower(text: str, max_tokens: int, draft: int, has_tools: bool,
                     think: bool, stops: list[str] | None,
                     temperature: float | None, top_k: int | None,
                     top_p: float | None) -> None:
    """Non-leader-rank counterpart of Handler._stream: runs the identical
    speculative_generate call (same drafter resolution, same arguments,
    same abort-sync participation) with no HTTP socket to write to and no
    output worth keeping -- this rank's only job is to stay in lockstep
    with rank 0's real request. See _tp_follower_loop/_tp_dispatch's module
    docstring for why a follower rank needs this at all.

    `has_tools`/`think`/`stops` are accepted for signature parity with the
    leader's dispatch payload but unused here -- they only affect what
    Handler._stream buffers/truncates for the CLIENT'S benefit, not what
    speculative_generate computes, so this rank has no reason to replicate
    that bookkeeping."""
    bundle = STATE["bundle"]
    ids = bundle.tok(text, return_tensors="pt").input_ids
    if bundle.use_gpu:
        ids = ids.to(bundle.gpu_dev)
    drafter = STATE["drafter_obj"] if draft > 0 else None
    rp = _request_params(max_tokens, draft, temperature, top_k, top_p)

    with STATE["lock"]:
        try:
            # NO_ABORT: this rank has no client socket. Only rank 0's
            # abort_check is consulted; spec_decode broadcasts rank 0's
            # decision every step (tp_active branch), so this rank still
            # stops exactly when rank 0 (disconnect OR VRAM ceiling) does.
            run_generate(bundle, drafter, ids, rp, stream=False,
                         abort_check=NO_ABORT)
        finally:
            _maybe_empty_cache(bundle)


class Handler(QVHTTPHandler):
    LOG_TAG = "serve"

    # ~6 bytes/token is a generous upper bound for UTF-8 text plus JSON
    # structure overhead, so this never rejects anything the --ctx-size
    # check in do_POST (which runs the real tokenizer and is the
    # authoritative limit, but only fires AFTER the full body is read into
    # memory and AFTER build_prompt()/tokenization has already paid the
    # cost of a huge string) would have accepted.
    MAX_BODY = staticmethod(lambda: STATE["ctx_size"] * 6)

    def _body_too_large_message(self, n: int, max_body: int) -> str:
        return (f"request body is {n} bytes, over the {max_body}-byte limit "
                f"derived from --ctx-size ({STATE['ctx_size']} tokens)")

    # -- routes -----------------------------------------------------

    def do_GET(self):
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
        if _TP_ACTIVE:
            return self._send(400, {"error": "image input is not supported with tensor parallel"})
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

    def _vision_chat_stream(self, body, max_tokens, think, temperature, top_k, top_p,
                            req_id: int = 0):
        """SSE version of _vision_chat. The image route used to answer a streaming
        request with ONE JSON body after the whole generation (tower load + a
        thinking block of thousands of tokens), so a client saw no bytes for
        minutes and gave up or looked hung. Here headers and a keep-alive go out
        at once, tokens stream as they are sampled (reasoning_content before
        </think>, content after), and a client that goes away stops the run."""
        from ai_engine.generation import vision as _vision
        from qv_band.decode.spec_decode import _IncStreamer
        bundle = STATE["bundle"]
        has_tools = bool(body.get("tools"))
        stops = _stop_list(body)
        margin = max((len(x) for x in stops), default=1) - 1
        include_usage = body.get("stream_options", {}).get("include_usage", True)
        created = int(time.time())
        cid = f"chatcmpl-{created}"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", self._cors_value())
        self.send_header("Vary", "Origin")
        self._end_close()
        self.end_headers()

        def write(data: bytes) -> None:
            self.connection.settimeout(2.0)
            try:
                self.wfile.write(data)
                self.wfile.flush()
            finally:
                self.connection.settimeout(None)

        def sse(obj) -> None:
            write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")

        def gone() -> bool:
            try:
                import select
                r, _, _ = select.select([self.connection], [], [], 0)
                return bool(r) and self.connection.recv(
                    1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
            except (BlockingIOError, InterruptedError):
                return False
            except OSError:
                return True

        def delta(key: str, text: str) -> None:
            if text:
                sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": MODEL_ID, "choices": [{"index": 0, "delta": {key: text}}]})

        split = _ThinkSplitter(bool(think))
        st = {"sent": 0, "content": [], "n": 0}
        streamer = None
        t0 = time.time()
        # Same "arrived" line the text path prints -- without it a streaming
        # image request is invisible in the log until it finishes (or dies).
        print(f"[serve] {_ts()} req #{req_id} arrived "
              f"(thread={threading.current_thread().name}), vision stream",
              file=sys.__stdout__, flush=True)

        def route_pieces(pieces) -> None:
            for key, text in pieces:
                if key == "reasoning_content":
                    delta(key, text)
                    continue
                st["content"].append(text)
                if has_tools:                           # needs the whole text for parse_tool_calls
                    continue
                full = "".join(st["content"])
                hit, _ = _truncate_at_stop(full, stops)
                cut = min(max(len(full) - margin, st["sent"]), len(hit))
                delta("content", full[st["sent"]:cut])
                st["sent"] = cut

        try:
            write(b": ping\n\n")                      # bytes on the wire before the slow parts
            with STATE["lock"], _batch_paused():
                ad = _vision.adapter(bundle, STATE.get("drafter_obj"))            # first image request loads the tower
                write(b": ping\n\n")
                streamer = _IncStreamer(bundle.tok)

                def on_token(tid: int) -> None:
                    st["n"] += 1
                    route_pieces(split.feed(streamer.feed([tid])))

                r = ad.generate(
                    _denormalise_tool_calls(body["messages"]), tools=body.get("tools"),
                    think=think, max_new=max_tokens,
                    temperature=0.0 if temperature is None else float(temperature),
                    top_k=20 if top_k is None else int(top_k),
                    top_p=0.95 if top_p is None else float(top_p),
                    on_token=on_token, should_stop=gone,
                    on_progress=lambda: write(b": ping\n\n"),
                    prefill_chunk=getattr(STATE.get("params"), "prefill_chunk", None))
                _maybe_empty_cache(bundle)
            if r.get("stopped"):
                print(f"[serve] {_ts()} req #{req_id} vision stream: client gone after "
                      f"{time.time() - t0:.1f}s, {st['n']} tok",
                      file=sys.__stdout__, flush=True)
                return
            route_pieces(split.feed(streamer.finish()))  # a half-emitted codepoint held back
            route_pieces(split.finish())
            text, hit_stop = _truncate_at_stop("".join(st["content"]), stops)
            finish = "stop" if hit_stop or r["completion_tokens"] < max_tokens else "length"
            if has_tools:
                content, tool_calls = parse_tool_calls(text)
                last = {"tool_calls": tool_calls} if tool_calls else {"content": content or text or None}
                finish = "tool_calls" if tool_calls else finish
            else:
                delta("content", text[st["sent"]:])
                last = {}
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": MODEL_ID,
                     "choices": [{"index": 0, "delta": last, "finish_reason": finish}]}
            if include_usage:
                chunk["usage"] = {"prompt_tokens": r["prompt_tokens"],
                                  "completion_tokens": r["completion_tokens"],
                                  "total_tokens": r["prompt_tokens"] + r["completion_tokens"]}
                if r.get("timings"):
                    chunk["timings"] = r["timings"]
            sse(chunk)
            write(b"data: [DONE]\n\n")
            STATE["served"] += 1
            STATE["tokens"] += r["completion_tokens"]
            dt = time.time() - t0
            print(f"[serve] {_ts()} req #{req_id} done: {st['n']} tok in {dt:.1f}s "
                  f"({st['n']/dt:.2f} tok/s), {r['prompt_tokens']} prompt tok, "
                  f"{r.get('images', 0)} image(s), vision stream",
                  file=sys.__stdout__, flush=True)
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            print(f"[serve] {_ts()} req #{req_id} vision stream: connection lost "
                  f"after {time.time() - t0:.1f}s",
                  file=sys.__stdout__, flush=True)
        except Exception as e:
            print(f"[serve] {_ts()} req #{req_id} vision failed: {type(e).__name__}: {e}",
                  file=sys.__stdout__, flush=True)
            traceback.print_exc()
            try:
                sse({"error": {"message": f"vision: {e}", "type": type(e).__name__}})
                write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                pass

    def do_POST(self):
        route = self.path.rstrip("/")
        req_id = _next_req_id()
        try:
            body = self._body()
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(400, {"error": f"bad JSON: {e}"})

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

            think_enabled = bool(body.get("think", STATE.get("think", True)))
            # image_url content -> the vision route (generation/vision.py);
            # text-only requests never take it.
            from ai_engine.generation import vision as _vision
            if route == "/v1/chat/completions" and _vision.has_images(body["messages"]) and not STATE.get("vision", True):
                return self._send(400, {"error": "image input is disabled on this server (started with --vision off; use --vision on, or LLM_VISION=on)"})
            if route == "/v1/chat/completions" and _vision.has_images(body["messages"]):
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
            if body.get("stream"):
                return self._stream(text, max_tokens, draft, route,
                                    has_tools=bool(body.get("tools")),
                                    think=think_enabled,
                                    stops=_stop_list(body),
                                    temperature=temperature, top_k=top_k,
                                    top_p=top_p, req_id=req_id,
                                    include_usage=body.get(
                                        "stream_options", {}).get(
                                            "include_usage", True))
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
                        req_id=req_id)
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

        stops = _stop_list(body)
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

    def _stream(self, text: str, max_tokens: int, draft: int, route: str,
               has_tools: bool = False, think: bool = False,
               stops: list[str] | None = None,
               temperature: float | None = None, top_k: int | None = None,
               top_p: float | None = None, req_id: int = 0,
               include_usage: bool = False):
        """Server-sent events.  At ~0.6 tok/s a long answer is minutes away,
        so streaming is the difference between usable and not.

        When the request carried "tools", a <tool_call> block can appear
        anywhere in the generated text and must never reach the client as
        raw content (Cline etc. would render the literal XML). There is no
        cheap way to know a token is/isn't inside a future tool call as it
        streams, so with has_tools this buffers the whole generation instead
        of emitting live deltas, then runs parse_tool_calls() once at the
        end -- same parser the non-streaming path uses. Requests without
        tools are unaffected and keep the original token-by-token path.

        Thinking (enable_thinking=True) is streamed incrementally too, but
        on a SEPARATE wire field: everything up to (not including) the
        closing </think> tag is sent as delta.reasoning_content, matching
        the vLLM/SGLang/DeepSeek convention Cline and other OpenAI-
        compatible clients already parse into their own collapsible
        "thinking" UI -- see reasoning_content in streaming docs. This
        replaces the older behaviour of buffering the whole response until
        </think> closed and discarding everything before it: that made a
        long reasoning block (this checkpoint's <think> routinely runs
        1200-5700+ tokens) look like a multi-minute hang with zero bytes
        on the wire, even though decode was running at its normal rate the
        whole time -- the client had nothing to show, not because nothing
        was happening. Streaming reasoning_content fixes the PERCEIVED
        latency; it does not change decode throughput itself.

        has_tools keeps the old whole-response buffer: a <tool_call> block
        can appear anywhere in the generated text and must never reach the
        client as raw content, and unlike the </think> boundary there is
        no cheap way to know a token is/isn't inside a future tool call as
        it streams, so this still buffers to run parse_tool_calls() once
        at the end. Requests without tools use the incremental path below
        regardless of `think` -- thinking and non-thinking content share
        the same split-by-</think> logic, just with reasoning_content
        empty (immediate switch to content) when </think> never appears
        (e.g. --no-think, or a non-thinking checkpoint).

        "stop" strings can span more than one streamed piece, so a small
        trailing window (the longest stop string, minus one char) is always
        held back before emitting; once a stop is found in the held buffer,
        emission cuts off there and the rest of the generation (which still
        runs to completion internally -- see _truncate_at_stop) is dropped
        rather than sent.
        """
        # Only the POST-</think> content needs buffering for has_tools: a
        # <tool_call> block is emitted by the checkpoint's own chat template
        # convention only in the answer that follows </think>, never inside
        # the reasoning block itself (the template puts <think> first,
        # always closed before any tool-call or plain-answer text begins).
        # So reasoning can stream incrementally exactly like the no-tools
        # path -- only content AFTER </think> needs to wait for
        # parse_tool_calls() to run once at the end. `buffer_all` now means
        # "buffer post-</think> content", not "buffer everything"; a
        # request with think=False and has_tools=True still buffers from
        # the first byte, since there is no </think> to wait for.
        buffer_all = has_tools
        stops = stops or []
        margin = max((len(s) for s in stops), default=1) - 1

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", self._cors_value())
        self.send_header("Vary", "Origin")
        self._end_close()
        self.end_headers()

        bundle = STATE["bundle"]
        tok = bundle.tok          # model is reached via run_generate(bundle, ...)
        ids = tok(text, return_tensors="pt").input_ids
        if bundle.use_gpu:
            ids = ids.to(bundle.gpu_dev)
        created = int(time.time())
        model_id = MODEL_ID
        chat = route == "/v1/chat/completions"

        # Disconnect detection is a select()+recv() -- two syscalls. Doing
        # it per token (the abort_check runs once per decode step) is a
        # measurable serialising cost at this token rate, and a client that
        # left is fine to notice a fraction of a second later. Throttle to
        # a wall-clock interval; llama.cpp's server polls is_connection_
        # closed roughly the same way rather than every token.
        _PEER_POLL_S = 0.25
        _peer_last = [0.0, False]   # [last_check_time, last_result]

        # Bounds every socket write _pump() makes -- see its own comment
        # for why an unbounded write is a real risk (it runs on the thread
        # holding the global generation lock, for every emitted piece, not
        # just the heartbeat). A slow-but-alive client legitimately
        # reading at a trickle could in principle hit this and lose a
        # piece; that is the correct tradeoff since the alternative is an
        # unbounded stall of every other queued request on this server,
        # not just the slow one.
        _WRITE_TIMEOUT_S = 2.0

        def _peer_gone(force: bool = False) -> bool:
            now = time.monotonic()
            if not force and now - _peer_last[0] < _PEER_POLL_S:
                return _peer_last[1]
            _peer_last[0] = now
            try:
                import select
                sock = self.connection
                r, _, _ = select.select([sock], [], [], 0)
                gone = bool(r) and sock.recv(
                    1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
            except (BlockingIOError, InterruptedError):
                gone = False
            except OSError:
                gone = True
            _peer_last[1] = gone
            return gone

        # SSE chunks are coalesced and flushed on a wall-clock interval
        # rather than per token: a per-token write()+flush() syscall pair
        # serialises with the decode at this token rate, and the client
        # (Cline, a browser EventSource) renders on its own frame budget
        # anyway. SGLang and llama.cpp both batch their stream writes.
        _FLUSH_S = 0.05
        _wbuf: list[bytes] = []
        _wbuf_last = [time.monotonic()]

        def _sse(obj) -> None:
            _wbuf.append(b"data: " + json.dumps(obj, ensure_ascii=False).encode()
                         + b"\n\n")

        def _pump(force: bool = False) -> None:
            # self.wfile.write()/flush() are BLOCKING socket calls with no
            # timeout by default -- a slow or half-dead client (stopped
            # reading but never closed cleanly) can stall this write
            # indefinitely. Every call site runs on the thread holding
            # STATE["lock"] for the WHOLE generation (emit()'s per-token
            # writes during decode, not just _ping()'s 20s heartbeat), so
            # an unbounded stall here blocks every other queued request on
            # this server, not just the slow one. Bound with a send
            # timeout; on expiry the write raises socket.timeout, which
            # every caller already catches alongside BrokenPipeError/
            # ConnectionResetError (a stalled client is the same "nothing
            # left to deliver" outcome as a closed one from the caller's
            # side).
            if not _wbuf:
                return
            now = time.monotonic()
            if not force and now - _wbuf_last[0] < _FLUSH_S:
                return
            self.connection.settimeout(_WRITE_TIMEOUT_S)
            try:
                self.wfile.write(b"".join(_wbuf))
                self.wfile.flush()
            finally:
                self.connection.settimeout(None)
            _wbuf.clear()
            _wbuf_last[0] = now

        def _ping() -> None:
            # SSE keep-alive, passed to run_generate as on_heartbeat: fires
            # on a wall-clock timer (HEARTBEAT_INTERVAL_S, spec_decode.py)
            # during BOTH a long chunked prefill and a slow/stalled decode
            # stretch, not just prefill -- decode tokens are normally their
            # own heartbeat, but heavy draft rejection or GPU contention
            # can still leave a real silent gap. A `:`-prefixed line is a
            # COMMENT per the SSE spec, never a `data:` event, so every
            # conformant client (including Cline's) ignores its content
            # but still sees the byte traffic and resets its own body-read
            # timeout. Confirmed root cause of Cline's own "terminated:
            # BodyTimeoutError (UND_ERR_BODY_TIMEOUT)": a long silent
            # stretch producing zero streamed bytes is exactly what a
            # body-read timeout watches for -- Cline then cancels and
            # resubmits, landing a NEW request behind the old one in this
            # server's single-flight queue (see _peer_gone/the docstring
            # above) and compounding the delay each retry. force=True:
            # skip _pump's throttle, since the whole point is to land
            # bytes on the wire promptly, not batch them with real content
            # that has not started yet.
            # force=True's write is bounded the same as every other _pump()
            # call -- see that function's own comment. The heartbeat is
            # best-effort by design (a missed ping just means the client's
            # own body-read timeout fires sooner), so losing one to a
            # write timeout is always safe to swallow.
            _wbuf.append(b": ping\n\n")
            try:
                _pump(force=True)
            except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                pass  # let the next _peer_gone() poll notice and abort

        def emit(piece: str, reasoning: bool = False):
            if _peer_gone():
                raise _ClientGone()
            if chat:
                delta_key = "reasoning_content" if reasoning else "content"
                obj = {"id": f"chatcmpl-{created}",
                       "object": "chat.completion.chunk", "created": created,
                       "model": model_id,
                       "choices": [{"index": 0, "delta": {delta_key: piece}}]}
            else:
                # /v1/completions and /generate have no reasoning_content
                # slot in their wire shape -- reasoning text is dropped
                # there, same as this whole feature not existing pre-fix,
                # since only the chat-completions shape has a place to put
                # it and a client (Cline) actually parsing it.
                if reasoning:
                    return
                obj = {"text": piece}
            _sse(obj)
            _pump()

        buf: list[str] = []
        pending = [""]      # held-back trailing window, for stop matching
        stopped = [False]
        # `think`-only requests (has_tools handled separately, still fully
        # buffered): track whether </think> has closed yet, so held-back
        # pending text can be routed to reasoning_content before the close
        # and content after. _THINK_TAG is the tag itself, never emitted;
        # margin must also cover it so a tag split across two writes is
        # not missed the same way a stop string would not be.
        _THINK_TAG = "</think>"
        in_reasoning = [think]
        reasoning_margin = max(margin, len(_THINK_TAG) - 1) if think else margin

        # speculative_generate prints as it goes; capture that stream instead
        # of re-implementing the loop, by swapping stdout for a shim.
        # A model whose output OPENS its own <think> (Ripple) decides reasoning vs content from its first
        # characters instead of assuming the prompt opened the block (Qwen). None = Qwen behaviour, unchanged.
        _fmt = _chat_format()
        _gate = _fmt.new_stream_gate() if (_fmt is not None and think) else None

        class _Shim:
            def write(_s, piece):
                if not piece or stopped[0]:
                    return
                if _gate is not None:
                    piece, _dec = _gate.feed(piece)
                    if _dec is False:
                        in_reasoning[0] = False
                    if not piece:
                        return
                if buffer_all and not in_reasoning[0]:
                    # Past </think> (or think=False, so in_reasoning started
                    # False and there is no boundary to wait for) -- a
                    # <tool_call> can appear anywhere in this remaining
                    # text, so it must stay buffered for parse_tool_calls()
                    # at the end.
                    buf.append(piece)
                    return
                pending[0] += piece
                cut = min((pending[0].index(s) for s in stops if s in pending[0]),
                         default=None)
                if cut is not None:
                    stopped[0] = True
                    out = pending[0][:cut]
                    pending[0] = ""
                elif len(pending[0]) > reasoning_margin:
                    split = len(pending[0]) - reasoning_margin
                    # BUG FIX: a fixed-width margin (len(_THINK_TAG)-1
                    # trailing chars retained) is not enough on its own --
                    # it protects the SUFFIX of pending[0] from being
                    # released while it could still grow into the tag, but
                    # says nothing about the split point ITSELF landing
                    # inside a partial tag match. Concretely: pending[0] =
                    # "...this. </think>" splits at len-7, releasing
                    # "...this. <" (ending in the tag's OPENING '<') and
                    # retaining "/think>" (7 chars) -- neither the released
                    # prefix nor the retained suffix contains the full tag
                    # any more, so out.find(_THINK_TAG) below can never
                    # find it and in_reasoning[0] never flips. Confirmed:
                    # this silently misclassified an ENTIRE response as
                    # reasoning_content whenever the tag happened to land
                    # on this exact boundary -- a real, pre-existing bug,
                    # not new. Fix: while still in_reasoning, never let the
                    # released slice's own tail be a partial prefix of the
                    # tag -- pull the split point back to before that
                    # partial match starts, so the tag stays whole and
                    # detectable once enough of it has arrived.
                    if in_reasoning[0]:
                        window = pending[0][:split]
                        max_check = min(len(_THINK_TAG) - 1, split)
                        for i in range(max_check, 0, -1):
                            if window.endswith(_THINK_TAG[:i]):
                                split -= i
                                break
                    out = pending[0][:split]
                    pending[0] = pending[0][split:]
                else:
                    return
                if not out:
                    return
                # Split `out` at </think> if it closes inside this piece --
                # everything up to the tag is reasoning, the tag itself is
                # dropped (never shown to the client, same as before), the
                # remainder is real content. Only relevant while still
                # in_reasoning; once closed, every later write is plain
                # content and this whole branch is skipped for speed.
                pieces: list[tuple[str, bool]] = []  # (text, is_reasoning)
                if in_reasoning[0]:
                    tag_at = out.find(_THINK_TAG)
                    if tag_at >= 0:
                        if tag_at:
                            pieces.append((out[:tag_at], True))
                        rest = out[tag_at + len(_THINK_TAG):]
                        in_reasoning[0] = False
                        # pending[0] can ALREADY hold post-tag content at
                        # this exact moment -- the margin window that just
                        # released `out` was sized for reasoning_margin
                        # (protecting a possibly-still-forming tag), so
                        # pending[0] can be holding up to reasoning_margin
                        # chars of text that arrived AFTER the tag closed,
                        # not before it.
                        #
                        # buffer_all: drain pending[0] to EMPTY (not down
                        # to `margin`), and prepend all of it to `rest`.
                        # Every LATER write() call routes straight into buf
                        # via the buffer_all-and-not-in_reasoning fast path
                        # (bypassing pending[0]/the stop-margin machinery
                        # entirely -- stop-string truncation for buffered
                        # content happens once at the end, via
                        # _truncate_at_stop on the fully joined buf, not
                        # here). Leaving ANY residue in pending[0] at this
                        # point means it only gets flushed by the tail
                        # block after generation ends, landing AFTER every
                        # char the fast path already appended in the
                        # meantime -- a real, confirmed bug: "answer "
                        # left in pending[0] (its length happened to equal
                        # `margin` exactly, so the old margin-only drain
                        # skipped it) surfaced as "STOPHERE junkanswer "
                        # instead of "answer STOPHERE junk". has_tools=
                        # False does not hit this: it keeps using
                        # pending[0]+margin for its OWN stop-matching on
                        # live content, so only the buffer_all branch drains
                        # to zero here.
                        if buffer_all:
                            rest += pending[0]
                            pending[0] = ""
                        elif len(pending[0]) > margin:
                            drain = len(pending[0]) - margin
                            rest += pending[0][:drain]
                            pending[0] = pending[0][drain:]
                        if rest:
                            if buffer_all:
                                buf.append(rest)
                            else:
                                pieces.append((rest, False))
                    else:
                        pieces.append((out, True))
                else:
                    pieces.append((out, False))
                try:
                    for text, is_reasoning in pieces:
                        emit(text, reasoning=is_reasoning)
                except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                    # The client is gone (Cline cancelling a request is
                    # routine) or has stalled long enough to hit _pump's
                    # write timeout -- same outcome either way: nothing
                    # left to deliver. Swallowing this and carrying on means the
                    # server keeps decoding every remaining token for
                    # nobody, holding the global generation lock the whole
                    # time and blocking real requests behind it. Raising
                    # here unwinds speculative_generate -- which has no
                    # interruption hook of its own, it just prints -- out
                    # through _stream's try/finally (stdout restored,
                    # empty_cache called) and into do_POST's handler.
                    stopped[0] = True
                    raise _ClientGone()
            def flush(_s):
                pass

        drafter = STATE["drafter_obj"] if draft > 0 else None

        # See generate(): draft>0 no longer forces greedy -- speculative
        # sampling applies the request's temperature/top_k/top_p. The one
        # KV-mode -> kwargs mapping is in run_generate; nothing here
        # re-derives it. _request_params clamps max_output_tokens to
        # ctx_size and folds in the base GenParams for any None field.
        rp = _request_params(max_tokens, draft, temperature, top_k, top_p)

        # ---- CONTINUOUS-BATCHING SCHEDULER PATH -----------------------
        # When STATE["scheduler"] is set (qvb8_resident + non-TP, see
        # load()), route through it instead of the old STATE["lock"]-
        # serialised call below -- this is what actually lets two Cline
        # connections share one batched forward pass per decode step
        # rather than queueing behind each other. Reuses the SAME emit()/
        # _Shim/buf/pending machinery already built above (</think>
        # split, tool-call buffering, stop-string truncation) -- only the
        # SOURCE of the token text differs: instead of speculative_
        # generate() printing to a swapped sys.stdout under one global
        # lock, the scheduler's on_token callback (running on the
        # scheduler's OWN background thread, batching this request
        # together with whatever others are in the same round) pushes
        # token ids into a queue THIS thread blocks reading, decoded
        # incrementally the same way the non-batched path already does.
        if STATE["scheduler"] is not None:
            print(f"[serve] {_ts()} req #{req_id} arrived, "
                  f"prompt {ids.shape[1]} tok", file=sys.__stdout__, flush=True)
            ids_dev = ids
            out_q = STATE["scheduler"].submit(
                ids_dev, rp.max_output_tokens, rp.temperature or 0.0,
                rp.top_k or 0, rp.top_p or 1.0, req_id)
            _shim = _Shim()
            from qv_band.decode.spec_decode import (
                _IncStreamer, HEARTBEAT_INTERVAL_S as _HEARTBEAT_S)
            _streamer = _IncStreamer(STATE["tok"])
            st = None
            n_recv = 0
            t0 = time.time()
            try:
                while True:
                    # Ping on WIRE silence, not queue silence: while a long
                    # tool call is held back (parse_tool_calls() needs the
                    # whole text) tokens keep arriving here every few ms, so a
                    # get() timeout never fires, yet nothing reaches the
                    # socket. _wbuf_last is the time of the last real write.
                    # The timeout also covers a stalled engine (no tokens).
                    try:
                        item = out_q.get(timeout=1.0)
                    except queue.Empty:
                        item = None
                        # Nothing from the engine for a second (a long prefill, say): is the client still
                        # there? A cheap socket peek, so a stopped request frees its slot within ~1 s of
                        # leaving instead of after the 10 s ping interval.
                        if _peer_gone():
                            raise _ClientGone()
                    if time.monotonic() - _wbuf_last[0] >= _HEARTBEAT_S:
                        if _peer_gone():
                            raise _ClientGone()
                        _ping()
                    if item is None:
                        continue
                    if isinstance(item, Exception):
                        raise item
                    if hasattr(item, "stats"):   # _StreamDone sentinel
                        st = item.stats
                        break
                    n_recv += 1
                    piece = _streamer.feed([item])
                    if piece:
                        _shim.write(piece)
                        if stopped[0]:
                            # A stop string was hit: the answer is complete. Free the slot now
                            # instead of letting it generate to max_tokens for nobody.
                            STATE["scheduler"].cancel(req_id)
                            break
                if not stopped[0]:
                    tail = _streamer.finish()
                    if tail:
                        _shim.write(tail)
            except _ClientGone:
                STATE["scheduler"].cancel(req_id)
                print(f"[serve] {_ts()} req #{req_id} client disconnected after "
                     f"{time.time()-t0:.1f}s (scheduler path)",
                     file=sys.__stdout__, flush=True)
                return
            except Exception as e:
                STATE["scheduler"].cancel(req_id)
                traceback.print_exc()
                try:
                    _sse({"error": {"message": str(e), "type": type(e).__name__}})
                    _wbuf.append(b"data: [DONE]\n\n")
                    _pump(force=True)
                except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                    pass
                return
            dt = time.time() - t0
            out = st.emitted if st else n_recv
            STATE["served"] += 1
            STATE["tokens"] += out
            base_finish = ("stop" if stopped[0]
                           else "length" if out >= rp.max_output_tokens else "stop")
            if _gate is not None:
                if _gate.decided is None:
                    in_reasoning[0] = False                 # tiny output that never decided: plain content
                pending[0] = _gate.take() + pending[0]      # + any tail the gate held back (e.g. "\n</thi")
            try:
                if not buffer_all:
                    if not stopped[0] and pending[0]:
                        flushed, _ = _truncate_at_stop(pending[0], stops)
                        if flushed:
                            delta_key = "reasoning_content" if in_reasoning[0] else "content"
                            _sse({"id": f"chatcmpl-{created}", "object": "chat.completion.chunk",
                                 "created": created, "model": model_id,
                                 "choices": [{"index": 0, "delta": {delta_key: flushed}}]}
                                if chat else {"text": flushed})
                    finish = base_finish
                    if chat:
                        chunk = {"id": f"chatcmpl-{created}", "object": "chat.completion.chunk",
                                 "created": created, "model": model_id,
                                 "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                        if include_usage:
                            chunk["usage"] = {"prompt_tokens": int(ids.shape[1]),
                                              "completion_tokens": out,
                                              "total_tokens": int(ids.shape[1]) + out}
                        _sse(chunk)
                else:
                    joined, _stop_hit = _truncate_at_stop("".join(buf), stops)
                    if _stop_hit:
                        base_finish = "stop"
                    if not chat:
                        if joined:
                            _sse({"text": joined})
                        finish = base_finish
                    elif has_tools:
                        content, tool_calls = parse_tool_calls(joined)
                        if tool_calls:
                            delta = {"tool_calls": tool_calls}
                            finish = "tool_calls"
                        else:
                            delta = {"content": content or joined or None}
                            finish = base_finish
                    else:
                        delta = {"content": joined}
                        finish = base_finish
                    if chat:
                        chunk = {"id": f"chatcmpl-{created}", "object": "chat.completion.chunk",
                                 "created": created, "model": model_id,
                                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                        if include_usage:
                            chunk["usage"] = {"prompt_tokens": int(ids.shape[1]),
                                              "completion_tokens": out,
                                              "total_tokens": int(ids.shape[1]) + out}
                        _sse(chunk)
                _wbuf.append(b"data: [DONE]\n\n")
                _pump(force=True)
            except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError, _ClientGone):
                pass
            line = (f"[serve] {_ts()} req #{req_id} "
                   f"(thread={threading.current_thread().name}) "
                   f"streamed {out} tok in {dt:.1f}s "
                   f"({out/dt:.2f} tok/s) [scheduler]")
            if st and st.proposed:
                line += (f"  |  drafted {st.proposed}, accepted {st.accepted} "
                        f"({st.acceptance:.0%}), {st.emitted/max(1,st.steps):.2f} tok/step, "
                        f"{st.steps} steps: draft {st.t_draft:.1f}s verify {st.t_verify:.1f}s")
            print(line, file=sys.__stdout__, flush=True)
            return
        # ---- END SCHEDULER PATH ----------------------------------------

        # Printed BEFORE acquiring STATE["lock"] vs after would tell "queued"
        # from "actually running" apart -- this one fires the instant the
        # request arrives (before it waits its turn, if any), so two
        # requests logging "start" back-to-back with no "acquired the lock"
        # gap between them is exactly what a real overlap (a working
        # parallel scheduler) would look like, and what the current
        # STATE["lock"]-serialised path never produces (request 2's "start"
        # fires immediately, but its own prefill only begins once request
        # 1's full generation has released the lock -- watch the gap
        # between "start" and the first [prefill] line for each id).
        print(f"[serve] {_ts()} req #{req_id} arrived "
             f"(thread={threading.current_thread().name}), "
             f"prompt {ids.shape[1]} tok",
             file=sys.__stdout__, flush=True)
        with STATE["lock"]:
            if _TP_ACTIVE:
                _tp_dispatch("stream", dict(
                    text=text, max_tokens=rp.max_output_tokens, draft=rp.draft_k,
                    has_tools=has_tools, think=think, stops=stops,
                    temperature=rp.temperature, top_k=rp.top_k, top_p=rp.top_p))
            t0 = time.time()
            prefix_cache_in = STATE["prefix_cache"]
            # BUG FIX: spec_decode.py's [prefill] lines read this via
            # _REQ_ID.get() (a contextvar) to tag their log output with
            # the request they belong to -- added alongside this line,
            # but the .set() call to actually populate it was missing,
            # so every [prefill] line printed "req #0" regardless of the
            # real request, silently defeating the whole point of the
            # feature. Set it once per request, right where this request
            # starts genuinely running (inside the lock, not at arrival)
            # so it matches "which request is the model working on RIGHT
            # NOW" for a reader correlating [prefill] lines against
            # [serve] req #N arrived timestamps.
            from qv_band.decode.spec_decode import _REQ_ID
            _REQ_ID.set(req_id)
            # Captured INSIDE the lock (bug fix). It used to be read before the lock: a request that
            # arrived while another was generating saved THAT request's _Shim as its "original" stdout and, when
            # it finished, restored the dead shim process-wide -- every later print() then wrote into a closed
            # socket (log lines vanished; a print could raise inside another request's handler and drop it with
            # no trace: seen as a concurrent request that never logged "done"). Every swap happens under this
            # lock, so here sys.stdout is always the real stream.
            old = sys.stdout
            try:
                sys.stdout = _Shim()
                # abort_check: stop if the client disconnected OR live CUDA
                # memory reached --max-memory-gb. Polled every step (not
                # only when a token is emitted, which is all _Shim.write's
                # _ClientGone catches), and under TP speculative_generate
                # broadcasts the decision to every rank -- see its docstring.
                # A raised exception here (_ClientGone, a CUDA OOM, ...)
                # means speculative_generate's own return_cache path never
                # ran -- but the cache object it was mutating in place
                # (STATE["prefix_cache"].past, the SAME object just passed
                # in) may already reflect a partial run the OLD handle's
                # token_ids does not account for. Reusing that stale handle
                # next time would under-count what the cache actually
                # holds and duplicate tokens into it. Invalidate up front;
                # a successful run overwrites this with the fresh handle on
                # the next line, so the drop only sticks on that failure
                # path (via the except/finally below, since this line does
                # not execute again after an exception).
                STATE["prefix_cache"] = None
                kv_stats: dict = {}
                out, st, STATE["prefix_cache"] = run_generate(
                    bundle, drafter, ids, rp, stream=True,
                    abort_check=lambda: _peer_gone() or vram_abort_check(),
                    prefix_cache=prefix_cache_in, return_cache=True,
                    kv_stats=kv_stats, on_heartbeat=_ping)
            except _ClientGone:
                # Normal cancellation, not an error -- nothing to send, since
                # the socket that would carry it is what just went away.
                # Restore stdout first: the finally below has not run yet, so
                # a print() here would go to the _Shim and be swallowed as if
                # it were a generated token.
                sys.stdout = old
                print(f"[serve] {_ts()} client disconnected after "
                      f"{time.time()-t0:.1f}s, generation aborted", flush=True)
                return
            except Exception as e:
                # Any OTHER failure inside run_generate (a bad --kv-cache/
                # --kv-tau-threshold combination raising ValueError from
                # kv_band.py's codebook-overflow check, a CUDA error, ...)
                # used to propagate straight out of _stream() uncaught --
                # past do_POST's own try/except (which only wraps the
                # non-streaming path; _stream() returns directly from
                # do_POST's call site, outside that block) and up into
                # BaseHTTPRequestHandler's own handling, which the client
                # sees as a bare dropped/broken connection with no error
                # body at all. Confirmed: a real tau overflow
                # produced exactly this -- Cline got nothing usable, and
                # from its side that looks identical to the body-timeout
                # symptom this SSE path already works around elsewhere.
                # This must never take the whole server down (a threading
                # HTTP server survives one handler's uncaught exception
                # regardless, but "survives by accident" is not the same
                # as "the client gets told what happened") -- send it as a
                # clean SSE error object instead, same shape a well-behaved
                # OpenAI-compatible server uses, so Cline at least sees a
                # real error instead of a silent hang/drop.
                sys.stdout = old
                traceback.print_exc()
                try:
                    _sse({"error": {"message": str(e), "type": type(e).__name__}})
                    _wbuf.append(b"data: [DONE]\n\n")
                    _pump(force=True)
                except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                    pass
                return
            finally:
                sys.stdout = old
                _maybe_empty_cache(bundle)
            dt = time.time() - t0

        STATE["served"] += 1
        STATE["tokens"] += len(out)

        # Why generation ended, in OpenAI terms. `stopped[0]` is a
        # server-side stop-string hit; otherwise it is "length" iff the
        # decode loop ran out its budget, else "stop" (EOS). rp is the
        # resolved GenParams for this request -- max_output_tokens is the
        # real budget speculative_generate used.
        base_finish = ("stop" if stopped[0]
                       else "length" if len(out) >= rp.max_output_tokens
                       else "stop")
        # This tail block sends the closing finish_reason chunk + [DONE] --
        # the one write Cline actually depends on to know the response is
        # over ("Response stream ended without a finish reason" is exactly
        # what it prints when this write never happens). Generation is
        # already finished by this point, so there is no decode loop left
        # to protect by bailing out early on a disconnect check; re-running
        # one here only risks losing this write to a stale/racy read.
        # emit() calls _peer_gone() and raises _ClientGone on top of it --
        # that is right for mid-generation pieces (bail out of a still-
        # running decode loop), but wrong here, since it can skip this
        # closing write entirely and reproduce the exact "no finish_reason"
        # symptom on nothing more than a throttled liveness check flipping
        # true at the wrong instant. So the final flush/emit calls below
        # use _sse()+_pump() directly (no liveness check) wherever emit()
        # was previously used only for its wire-shape formatting, and
        # _ClientGone joins the except clause as a defensive catch-all for
        # a genuine disconnect -- same treatment as the pipe errors, since
        # by this point "client gone" and "nothing left to deliver" mean
        # the same thing.
        try:
            # Flush any held-back trailing window FIRST, regardless of
            # buffer_all -- it can still hold live-streamed reasoning text
            # that _Shim.write never got to emit (the pending[0] margin
            # withholds the last few chars of every write for stop-string
            # matching, see _Shim.write). This used to run only when
            # `not buffer_all`, which was correct back when buffer_all meant
            # "nothing was ever streamed live" -- now that has_tools still
            # streams reasoning_content incrementally, a trailing reasoning
            # fragment can be sitting in pending[0] even for a buffer_all
            # request, and skipping this flush silently dropped it (a
            # generation cut off mid-<think> with has_tools=True lost its
            # last few reasoning chars off the wire entirely).
            if _gate is not None:
                if _gate.decided is None:
                    in_reasoning[0] = False                 # tiny output that never decided: plain content
                pending[0] = _gate.take() + pending[0]      # + any tail the gate held back (e.g. "\n</thi")
            if not stopped[0] and pending[0]:
                flushed, _ = _truncate_at_stop(pending[0], stops)
                if flushed:
                    # Same </think> split as _Shim.write's incremental
                    # path -- the held-back trailing window can still
                    # contain the tag (or, if generation was cut off
                    # mid-thought with no </think> ever emitted, be
                    # entirely reasoning: in_reasoning[0] just stays
                    # True and every remaining byte goes out as
                    # reasoning_content, never promoted to content).
                    tail_pieces: list[tuple[str, bool]] = []
                    if in_reasoning[0]:
                        tag_at = flushed.find(_THINK_TAG)
                        if tag_at >= 0:
                            if tag_at:
                                tail_pieces.append((flushed[:tag_at], True))
                            rest = flushed[tag_at + len(_THINK_TAG):]
                            if rest:
                                if buffer_all:
                                    buf.append(rest)
                                else:
                                    tail_pieces.append((rest, False))
                        else:
                            tail_pieces.append((flushed, True))
                    elif buffer_all:
                        buf.append(flushed)
                    else:
                        tail_pieces.append((flushed, False))
                    for text, is_reasoning in tail_pieces:
                        delta_key = ("reasoning_content" if is_reasoning
                                    else "content")
                        _sse({"id": f"chatcmpl-{created}",
                              "object": "chat.completion.chunk",
                              "created": created, "model": model_id,
                              "choices": [{"index": 0,
                                           "delta": {delta_key: text}}]}
                             if chat else {"text": text})
            if not buffer_all:
                finish = base_finish
                if chat:
                    chunk = {"id": f"chatcmpl-{created}",
                             "object": "chat.completion.chunk",
                             "created": created, "model": model_id,
                             "choices": [{"index": 0, "delta": {},
                                        "finish_reason": finish}]}
                    if include_usage:
                        chunk["usage"] = {"prompt_tokens": int(ids.shape[1]),
                                          "completion_tokens": len(out),
                                          "total_tokens": int(ids.shape[1]) + len(out)}
                        chunk["timings"] = _timings(st, len(out), dt, int(ids.shape[1]))
                    _sse(chunk)
            else:
                # buf holds only POST-</think> content now (reasoning
                # streamed live above/in _Shim.write) -- no _strip_think
                # needed, unlike the old whole-response buffer this
                # replaced.
                joined, _ = _truncate_at_stop("".join(buf), stops)
                if not chat:
                    # /generate and /v1/completions have no tool-call
                    # contract; same {"text": ...} wire shape emit() sends.
                    if joined:
                        _sse({"text": joined})
                    finish = base_finish
                elif has_tools:
                    content, tool_calls = parse_tool_calls(joined)
                    if tool_calls:
                        delta = {"tool_calls": tool_calls}
                        finish = "tool_calls"
                    else:
                        delta = {"content": content or joined or None}
                        finish = base_finish
                else:
                    delta = {"content": joined}
                    finish = base_finish
                if chat:
                    chunk = {"id": f"chatcmpl-{created}",
                             "object": "chat.completion.chunk", "created": created,
                             "model": model_id,
                             "choices": [{"index": 0, "delta": delta,
                                        "finish_reason": finish}]}
                    if include_usage:
                        chunk["usage"] = {"prompt_tokens": int(ids.shape[1]),
                                          "completion_tokens": len(out),
                                          "total_tokens": int(ids.shape[1]) + len(out)}
                        chunk["timings"] = _timings(st, len(out), dt, int(ids.shape[1]))
                    _sse(chunk)
            _wbuf.append(b"data: [DONE]\n\n")
            _pump(force=True)
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError, _ClientGone):
            pass
        line = (f"[serve] {_ts()} req #{req_id} streamed {len(out)} tok in {dt:.1f}s "
                f"({len(out)/dt:.2f} tok/s)")
        _tm = _timings(st, len(out), dt, int(ids.shape[1]))
        if _timings_str(_tm):
            line += "  |  " + _timings_str(_tm)
        # Same acceptance-rate line chat.py prints -- _stream() never had
        # it, so a slow streaming request (the path Cline actually hits)
        # gave no way to tell "drafter barely accepting, mostly falling
        # back to plain decode" apart from any other slowdown. draft_k is
        # rp's resolved value (0 disables drafting; st.proposed is then 0
        # too, so the condition also covers that case without a separate
        # check).
        if rp.draft_k > 0 and st.proposed:
            line += (f"  |  drafted {st.proposed}, accepted {st.accepted} "
                     f"({st.acceptance:.0%}), {st.emitted/max(1,st.steps):.2f} tok/step, "
                        f"{st.steps} steps: draft {st.t_draft:.1f}s verify {st.t_verify:.1f}s")
        else:
            line += "  |  drafting: NONE (draft_k=0 or nothing proposed)"
        # Prompt size drives prefill cost, which this line otherwise hides
        # inside the same wall-clock total as decode -- a long prompt with
        # a normal tok/s decode rate can still look "slow overall" for a
        # reason that has nothing to do with drafting or the KV codec.
        # prefix_reused is an UPPER BOUND (the saved cache's whole token
        # count, not the actual matched length -- that needs cache_config,
        # internal to spec_decode.py) on how much of THIS prompt could have
        # skipped prefill by reusing the prior turn's cache; 0 means either
        # no prior cache existed or nothing could be shared.
        prefix_reused = len(prefix_cache_in.token_ids) if prefix_cache_in else 0
        line += f"  |  prompt {ids.shape[1]} tok, prefix_reused<={prefix_reused}"
        if kv_stats.get("ratio_vs_dense") is not None:
            line += (f"  |  KV {kv_stats['total_bytes']/2**20:.1f} MB "
                     f"({100*kv_stats['ratio_vs_dense']:.1f}% of dense)")
        # REAL BUG, confirmed: this used plain print()/sys.stdout, same
        # mistake spec_decode.py's [prefill] lines already had to fix (see
        # that fix's own comment). By the time THIS request's finally
        # block restores sys.stdout, that is correct FOR THIS REQUEST --
        # but under STATE["lock"] serialisation, the NEXT request's
        # _stream() call can already be running by the time a slow
        # client-write/logging path here actually executes, and that next
        # request has ALREADY swapped sys.stdout to ITS OWN _Shim. This
        # line then silently lands inside the SECOND request's response
        # stream instead of the console -- confirmed by the user seeing it
        # appear at the START of a later reply's thinking block, not at
        # the end of the request it actually describes. sys.__stdout__ is
        # never reassigned by any request's shim swap, so it is the only
        # stream a genuinely out-of-band diagnostic line can safely use.
        print(line, file=sys.__stdout__, flush=True)


def main():
    # Tensor-parallel launch (torchrun, WORLD_SIZE>1): silence every rank
    # but 0's stdout via redirect (not a builtins.print monkeypatch --
    # numba, imported lazily by qv_band's KV codec modules,
    # asserts the real print builtin is still reachable by identity at
    # builtins.print; see chat.py's identical guard for the fuller story).
    if _TP_ACTIVE and not _TP_IS_LEADER:
        sys.stdout = open(os.devnull, "w")

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
    ap.add_argument("--greedy", action="store_true",
                    help="plain greedy/deterministic decoding (== "
                         "--temperature 0).")
    ap.add_argument("--vision", choices=("on", "off"), default=os.environ.get("LLM_VISION", "on").lower(),
                    help="image input (image_url chat content). on (default): the vision tower is built lazily on "
                         "the first image request, so a text-only workload pays no memory for it; off: such "
                         "requests are refused with 400 and nothing vision-related is ever loaded. (Env: LLM_VISION.)")
    ap.add_argument("--no-scheduler", dest="scheduler", action="store_false",
                    default=True,
                    help="do not start the continuous-batching scheduler; "
                         "serve requests one at a time behind a lock. Slower "
                         "under concurrency; mainly for debugging and for "
                         "comparing against the batched path. (Env: "
                         "QV_NO_SCHEDULER=1.)")
    args = ap.parse_args()
    _set_scheduler(args.scheduler)

    args.temperature, args.top_k, args.top_p = resolve_sampling(
        args.think, args.greedy, args.temperature, args.top_k, args.top_p)

    # One GenParams from the launch flags -- llm_server.py holds it in
    # STATE["params"] and each request clones it (see _request_params).
    params = params_from_args(args)
    if _TP_ACTIVE and params.max_memory_gb:
        # Every rank checks its own card against the ceiling at each decode
        # step (spec_decode._over_vram_ceiling) -- a follower has no
        # abort_check of its own, and its card can be the fuller one.
        os.environ["QV_VRAM_CEILING_GB"] = str(params.max_memory_gb)
    STATE["vision"] = (args.vision != "off")
    # Bind the socket BEFORE loading the model: loading a 27B pack takes
    # minutes and ~24 GB of VRAM, and a port clash discovered afterwards
    # wastes all of that (the process then dies with "Address already in
    # use" only after the whole model is resident). Fail fast instead.
    # Only the leader binds: every rank of a torchrun launch shares the host,
    # so a follower binding first makes rank 0 fail with "Address in use".
    srv = None
    if _TP_IS_LEADER:
        try:
            srv = ThreadingHTTPServer((args.host, args.port), Handler)
        except OSError as exc:
            print(f"[serve] cannot bind {args.host}:{args.port}: {exc}",
                  file=sys.stderr, flush=True)
            print("[serve] another server is probably already running on that "
                  "port -- stop it, or pass -p <port>.", file=sys.stderr,
                  flush=True)
            raise SystemExit(1) from exc
    load(args.model, args.think, params)

    if not _TP_IS_LEADER:
        # Tensor-parallel non-leader rank: no HTTP socket, no port to bind
        # (every rank binding the same one would be nonsensical anyway) --
        # just follow rank 0's requests for the process lifetime. See
        # _tp_follower_loop's docstring for the full architecture.
        _tp_follower_loop()
        return

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
        # Tell any following ranks to exit their loop too -- otherwise
        # they sit forever on a broadcast this rank (now shutting down)
        # will never send, and torchrun has to SIGTERM them instead of a
        # clean exit.
        if _TP_ACTIVE:
            _tp_dispatch("stop", {})


if __name__ == "__main__":
    main()
