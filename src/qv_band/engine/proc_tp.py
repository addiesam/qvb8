"""Tensor Parallel over PROCESSES for any BatchTarget: torchrun, one process per card, NCCL all-reduce.

engine/tp_target.TPTarget runs the cards as threads of one process with an all-reduce through a thread barrier: simple,
but the cards' Python share one GIL and that all-reduce cannot sit inside a CUDA graph. Here every card is its own
process (torchrun --nproc-per-node N); the model's tp.all_reduce is NCCL (formats/qvb8/tp.TPGroup), so a static decode step
(engine/static_decode) captures the whole step -- all-reduces included -- as one graph per card.

    rank 0      the server: engine, HTTP, tokenizer. Its target is ProcTPTarget, the usual BatchTarget surface (+ a
                static stepper); every call is sent to the other ranks first (a small command over a gloo group), then run
                on its own card. Request states stay where they are: rank 0 hands the engine HANDLES, each rank keeps
                its own share of every state under the handle's id (freed on every rank when the engine drops it).
    rank > 0    serve_follower(): receive a command, run it on its card, forever.

A model adapter opts in by building its per-card target with tp=setup(device)'s TPGroup (see models/inkling/adapter.py):

    tp, ctrl = setup(device)
    local = MyTarget(..., tp=tp)
    if tp.rank:  serve_follower(local, ctrl)     # never returns
    return ProcTPTarget(local, tp, ctrl)
The machine-wide shared prefix pool (--prefix-store shared) works through export_state / import_state: every rank
exports its own share of the state and the followers send theirs to rank 0 (gloo, torch.save bytes: any state layout,
loaded on the host), so the pool holds {"ranks": [share of rank 0, share of rank 1, ...]}; an import sends each rank
only its own share and every rank rebuilds the state under the same handle.
"""
from __future__ import annotations

import io
import os
import weakref

import torch
import torch.distributed as dist


def setup(device):
    """The NCCL group of the cards (TPGroup) + a gloo group for the commands; call on EVERY rank, same order."""
    from qv_band.formats.qvb8.tp import init_distributed
    torch.cuda.set_device(torch.device(device))
    tp = init_distributed(device)
    ctrl = dist.new_group(backend="gloo")
    return tp, ctrl


def _cpu(x):
    return x.detach().cpu() if torch.is_tensor(x) else x


def _own(x):
    """x with every tensor in its own HOST storage (a pool entry's tensors are views of ONE arena, in several dtypes,
    which torch.save refuses): tensors copied to host RAM -- a share on the card goes straight to the host, never a
    second copy on the card (a long context's KV would double there) --; dicts, lists, tuples and dataclasses
    (e.g. kv.codec.BandedKV) rebuilt."""
    import dataclasses
    if torch.is_tensor(x):
        return x.detach().to("cpu", copy=True)
    if isinstance(x, dict):
        return {k: _own(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_own(v) for v in x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.replace(x, **{f.name: _own(getattr(x, f.name)) for f in dataclasses.fields(x) if f.init})
    return x


def _pack(tree) -> bytes:
    """A rank's state share as bytes (torch.save: tensors inside any object, copied off the card)."""
    b = io.BytesIO()
    torch.save(_own(tree), b)
    return b.getvalue()


def _unpack(data: bytes, device="cpu"):
    return torch.load(io.BytesIO(data), map_location=device, weights_only=False)


class _Handle:
    __slots__ = ("id", "__weakref__")


class ProcTPTarget(torch.nn.Module):
    def __init__(self, local, tp, ctrl):
        super().__init__()
        self.local, self.tp, self.ctrl = local, tp, ctrl
        self.world = tp.world
        self._states, self._next, self._dead = {}, 1, []
        for k in ("stop_ids", "dtype", "config", "_qv_chat_format", "_qv_eos_ids", "_qv_kv_desc"):
            if hasattr(local, k):
                setattr(self, k, getattr(local, k))
        self.m = getattr(local, "m", None)                   # the engine reads the card from target.m.device
        self.linears = getattr(local, "linears", None)
        self._qv_batch_target = True

    # -- commands -------------------------------------------------------------------------------------------------
    def _send(self, op, *args):
        dead, self._dead = self._dead, []          # swap FIRST: a handle freed while this message is pickled goes next time
        dist.broadcast_object_list([(op, args, dead)], src=0, group=self.ctrl)

    def _handle(self, state):
        h = _Handle()
        h.id, self._next = self._next, self._next + 1
        self._states[h.id] = state
        weakref.finalize(h, self._free, h.id)
        return h

    def _free(self, i):
        self._states.pop(i, None)
        self._dead.append(i)

    def quit(self):
        self._send("quit")

    # -- BatchTarget ---------------------------------------------------------------------------------------------
    def new_state(self):
        self._send("new", self._next)
        return self._handle(self.local.new_state())

    def clone_state(self, h):
        self._send("clone", h.id, self._next)
        return self._handle(self.local.clone_state(self._states[h.id]))

    def state_nbytes(self, h):
        return self.local.state_nbytes(self._states[h.id]) * self.world

    # -- the machine's shared prefix pool (engine/target_engine, --prefix-store shared) -----------------------------
    def export_state(self, h):
        self._send("export", h.id)
        own, err = None, None
        with _fatal_oom(0):
            try:                                          # the followers wait in the gather below: always reach it
                own = self.local.export_state(self._states[h.id])
            except torch.OutOfMemoryError:
                raise
            except Exception as e:                        # noqa: BLE001
                err = e
            got = [None] * self.world
            dist.gather_object(None, got, dst=0, group=self.ctrl)
        if err is not None:
            raise err
        if any(b is None for b in got[1:]):
            raise RuntimeError("proc-tp export: a follower card could not export its share (see its log)")
        return {"ranks": [own] + [_unpack(b) for b in got[1:]]}

    def import_state(self, tree, device):
        shares = tree["ranks"]
        if len(shares) != self.world:
            raise ValueError(f"a pool entry of {len(shares)} ranks, this target has {self.world}")
        packed = [None] + [_pack(x) for x in shares[1:]]  # before the command: nothing may fail between it and the scatter
        self._send("import", self._next)
        dist.scatter_object_list([None], packed, src=0, group=self.ctrl)
        with _fatal_oom(0):
            return self._handle(self.local.import_state(shares[0], device))

    def forward(self, rows):
        msg = [(r[0].id, list(r[1]), None if len(r) < 3 or r[2] is None else (list(map(int, r[2][0])), _cpu(r[2][1])))
               for r in rows]
        self._send("fwd", msg)
        with _fatal_oom(0):
            return self.local.forward([(self._states[r[0].id],) + tuple(r[1:]) for r in rows])

    # -- engine/spec.py protocol: rank 0 drafts (its own drafter), every card verifies and keeps --------------------
    @property
    def spec_propose(self):
        return self._spec_propose if getattr(self.local, "spec_propose", None) is not None else None

    @property
    def spec_max_draft(self):
        return self.local.spec_max_draft

    @property
    def static_verify(self):                         # engine/target_engine: speculative requests may take static slots
        return getattr(self.local, "static_verify", None)

    def _spec_propose(self, rows):
        return self.local.spec_propose([(self._states[h.id], last, sp) for h, last, sp in rows])

    def spec_verify(self, rows):
        self._send("spec_verify", [(h.id, list(toks)) for h, toks in rows])
        with _fatal_oom(0):
            return self.local.spec_verify([(self._states[h.id], toks) for h, toks in rows])

    def spec_keep(self, rows):
        self._send("spec_keep", [(h.id, int(k)) for h, k in rows])
        self.local.spec_keep([(self._states[h.id], k) for h, k in rows])

    # -- static decode: one stepper per card, driven together ----------------------------------------------------
    def static_stepper(self, reserve=None, slots=None, max_len=None):
        return _ProcStepper(self, reserve, slots, max_len)

    def prefill_peak(self, tokens):
        """The largest bytes one prefill of `tokens` tokens takes above what each card holds now (a scratch state)."""
        self._send("peak_reset")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        h = self.new_state()
        self.forward([(h, [0] * int(tokens))])
        torch.cuda.synchronize()
        own = torch.cuda.max_memory_allocated() - base
        del h
        self._send("peak")
        got = [None] * self.world
        dist.gather_object(None, got, dst=0, group=self.ctrl)
        torch.cuda.empty_cache()
        return max([own] + [int(x) for x in got[1:]])

    def free_bytes(self):
        """The tightest card's free bytes."""
        self._send("mem")
        torch.cuda.empty_cache()
        got = [None] * self.world
        dist.gather_object(None, got, dst=0, group=self.ctrl)
        return min([torch.cuda.mem_get_info()[0]] + [int(x) for x in got[1:]])


class _ProcStepper:
    """engine/static_decode.StaticStepper surface; every card replays its own graph of the same step."""

    def __init__(self, proxy, reserve=None, slots=None, max_len=None):
        from qv_band.engine.static_decode import StaticStepper
        self.p = proxy
        slots = int(slots or os.environ.get("QV_STATIC_SLOTS", "4"))
        max_len = int(max_len or os.environ.get("QV_STATIC_MAX_LEN", "32768"))
        self.local = StaticStepper(proxy.local, slots, max_len, reserve=reserve)
        proxy._send("st_init", self.local.slots, self.local.max_len, self.local.reserve)
        got = [None] * proxy.world
        dist.gather_object(None, got, dst=0, group=proxy.ctrl)
        fits = [(self.local.slots, self.local.max_len)] + [tuple(x) for x in got[1:]]
        agreed = (min(f[0] for f in fits), min(f[1] for f in fits))
        proxy._send("st_size", agreed)
        if agreed[0] == 0:                                # a card has no room at all: every card decodes eagerly
            if hasattr(proxy.local, "static_free"):
                proxy.local.static_free()
            torch.cuda.empty_cache()
            raise RuntimeError("static decode: a follower card has no room for the slots")
        if agreed != (self.local.slots, self.local.max_len):
            if hasattr(proxy.local, "static_free"):
                proxy.local.static_free()
            torch.cuda.empty_cache()
            self.local = StaticStepper(proxy.local, agreed[0], agreed[1], shrink=False, reserve=self.local.reserve)
        self.slots, self.max_len = self.local.slots, self.local.max_len

    @property
    def spec_margin(self):
        return self.local.spec_margin

    def warm(self):
        self.p._send("st_warm")
        with _fatal_oom(0):
            self.local.warm()

    def close(self):
        self.p._send("st_close")
        self.local.close()
        if getattr(self.p, "_qv_static_stepper", None) is self:
            self.p._qv_static_stepper = None

    def acquire(self):
        return self.local.acquire()

    def release(self, slot):
        self.local.release(slot)

    def load(self, slot, h):
        self.p._send("st_load", slot, h.id)
        with _fatal_oom(0):
            self.local.load(slot, self.p._states[h.id])

    def step(self, toks):
        self.p._send("st_step", dict(toks))
        with _fatal_oom(0):
            return self.local.step(toks)

    def spec(self, rows):
        """StaticStepper.spec with the other cards: they run the measure / verify / commit phases (the all-reduces)
        with rank 0's inputs; the proposal and the plan are rank 0's."""
        L, p = self.local, self.p
        with _fatal_oom(0):
            if not L.planner.vcost:
                p._send("sp_measure", min(rows))
                L._spec_measure(min(rows))
            order, drafts, dists = L._spec_plan(rows)
            R, inputs = L._spec_rows(order, rows, drafts)
            p._send("sp_verify", R, inputs)
            L._spec_load(inputs)
            L._spec_verify(R)
            out, keep = L._spec_accept(order, drafts, inputs, R, rows, dists)
            p._send("sp_commit", keep)
            L._spec_commit(keep)
        return out


class _fatal_oom:
    """Out of memory on ONE card in the middle of a command: the other cards wait in a collective this card never joins
    (and would read the next command out of step). Stop this process with the reason; torchrun then stops the others."""

    def __init__(self, rank):
        self.rank = rank

    def __enter__(self):
        return self

    def __exit__(self, et, e, tb):
        if et is not None and issubclass(et, torch.OutOfMemoryError):
            print(f"[proc-tp] card {self.rank} out of memory during a command ({e}); stopping: the cards would run "
                  f"out of step", flush=True)
            os._exit(1)
        return False


def serve_follower(local, ctrl):
    """Rank > 0: run rank 0's commands on this card until it quits (or dies)."""
    f = _Follower(local, ctrl)
    while True:
        msg = [None]
        dist.broadcast_object_list(msg, src=0, group=ctrl)
        op, args, dead = msg[0]
        for i in dead:
            f.states.pop(i, None)
        if op == "quit":
            return
        # an ordinary error (a request the model refuses) raises on rank 0 too, at the same point: rank 0's scheduler
        # fails that request and serves on, so this card must too -- crashing here would leave rank 0 sending to a dead
        # card. Out of memory is different (one card only): _fatal_oom stops the service.
        try:
            with _fatal_oom(f.rank):
                f.run(op, args)
        except Exception as e:                          # noqa: BLE001
            print(f"[proc-tp] card {f.rank}: {op} failed: {e!r}", flush=True)


class _Follower:
    """A follower card's side of every command. A command that ends in a collective answers even when its own work
    failed (None / 0), so rank 0 never waits for a card that gave up."""

    def __init__(self, local, ctrl):
        self.local, self.ctrl = local, ctrl
        self.dev = torch.cuda.current_device()
        self.rank = dist.get_rank()
        self.states, self.stepper, self.reserve = {}, None, None

    def run(self, op, args):
        from qv_band.engine.static_decode import StaticStepper
        local, ctrl, states = self.local, self.ctrl, self.states
        if op == "new":
            states[args[0]] = local.new_state()
        elif op == "clone":
            states[args[1]] = local.clone_state(states[args[0]])
        elif op == "fwd":
            local.forward([(states[i], toks) + (() if vis is None else ((vis[0], vis[1].to(self.dev)),))
                           for i, toks, vis in args[0]])
        elif op == "export":
            try:
                b = _pack(local.export_state(states[args[0]]))
            except torch.OutOfMemoryError:
                raise
            except Exception as e:                        # noqa: BLE001 -- rank 0 is waiting in the gather
                print(f"[proc-tp] export failed here: {e!r}", flush=True)
                b = None
            dist.gather_object(b, None, dst=0, group=ctrl)
        elif op == "import":
            got = [None]
            dist.scatter_object_list(got, None, src=0, group=ctrl)
            states[args[0]] = local.import_state(_unpack(got[0]), f"cuda:{self.dev}")
        elif op == "st_init":                             # rank 0's proposal: shrink to what fits here, answer
            try:
                self.stepper = StaticStepper(local, args[0], args[1], reserve=args[2])
                self.reserve = args[2]
                fit = (self.stepper.slots, self.stepper.max_len)
            except Exception as e:                        # noqa: BLE001 -- no size fits (or the init failed): agree on none
                print(f"[proc-tp] static slots: {e!r}", flush=True)
                self.stepper, fit = None, (0, 0)
            dist.gather_object(fit, None, dst=0, group=ctrl)
        elif op == "st_size":                             # the (slots, context) every card agreed on
            n, ml = args[0]
            if self.stepper is None or (self.stepper.slots, self.stepper.max_len) != (n, ml):
                if hasattr(local, "static_free"):
                    local.static_free()
                torch.cuda.empty_cache()
                self.stepper = StaticStepper(local, n, ml, shrink=False, reserve=self.reserve) if n else None
        elif op == "st_warm":
            self.stepper.warm()
        elif op == "st_close":
            self.stepper.close()
            self.stepper = None
        elif op == "peak_reset":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.peak_base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        elif op == "peak":
            torch.cuda.synchronize()
            dist.gather_object(torch.cuda.max_memory_allocated() - self.peak_base, None, dst=0, group=ctrl)
            torch.cuda.empty_cache()
        elif op == "mem":
            torch.cuda.empty_cache()
            dist.gather_object(torch.cuda.mem_get_info()[0], None, dst=0, group=ctrl)
        elif op == "st_load":
            self.stepper.load(args[0], states[args[1]])
        elif op == "st_step":
            self.stepper.step(args[0])
        elif op == "sp_measure":
            self.stepper._spec_measure(args[0])
        elif op == "sp_verify":
            self.stepper._spec_load(args[1])
            self.stepper._spec_verify(args[0])
        elif op == "sp_commit":
            self.stepper._spec_commit(args[0])
        elif op == "spec_verify":                         # rank 0's speculative verify (then "spec_keep")
            local.spec_verify([(states[i], toks) for i, toks in args[0]])
        elif op == "spec_keep":
            local.spec_keep([(states[i], k) for i, k in args[0]])
