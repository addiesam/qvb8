"""Decoder layers on another machine (design_doc/engine_architecture.md, section 6).

A remote stage runs a contiguous range of decoder layers [a, b) and keeps the cache of those layers. The main
process (embedding, the other layers, final norm, lm_head, drafter, scheduler) is the continuous-batching engine as
on one machine; this module adds the two ends of the link.

Main side, `RemoteStage`:
  - the cache entries of the remote layers are stateless placeholders, so every cache helper of the engine sees them
    as layers without state and works on the local layers only;
  - layer a's forward is replaced by one round trip: hidden states + the layer's inputs + the step layout go out,
    the hidden states after layer b-1 (and the drafter taps that live there) come back. Layers a+1..b-1 pass through;
  - every cache action of the engine on a slot is repeated on the stage: add, drop, crop, snapshot / restore of the
    recurrent state, ring rollback, slot export / import (prefix store). Actions without an answer are queued and
    travel with the next message, in order.

Stage side, `StageHost`: the same engine class, used for its model preparation and its arming; its cache holds
placeholders for the layers it does not own.

Failure rule: a lost or failing stage raises stage_link.StageLost in the engine step. Nothing is retried.
"""
from __future__ import annotations

import os
import sys
import threading

import torch

from qv_band.run.stage_link import Link, StageLost, accept, listen

_SCALARS = (bool, int, float, str)


def _log(msg: str) -> None:
    print(msg, file=sys.__stdout__, flush=True)


def _localize(past, own) -> None:
    """Keep the cache layers in `own`; every other layer becomes a stateless placeholder."""
    from qv_band.kv.banded_cache import _StatelessLayer
    if type(past).__name__ != "BandedKVCache":
        raise RuntimeError(f"remote stages: cache type {type(past).__name__} is not supported (BandedKVCache only)")
    for i in range(len(past.layers)):
        if i not in own:
            past.layers[i] = _StatelessLayer()


def _pack_kwargs(kw: dict, fields: dict, tensors: dict) -> None:
    """A decoder layer's keyword inputs -> message fields + tensors. The cache object stays behind."""
    scal, tup, none = {}, {}, []
    for k, v in kw.items():
        if k == "past_key_values":
            continue
        if v is None:
            none.append(k)
        elif isinstance(v, torch.Tensor):
            tensors[f"k:{k}"] = v
        elif isinstance(v, _SCALARS):
            scal[k] = v
        elif isinstance(v, (tuple, list)) and all(isinstance(x, torch.Tensor) for x in v):
            tup[k] = len(v)
            for i, x in enumerate(v):
                tensors[f"k:{k}:{i}"] = x
        else:
            raise TypeError(f"remote stage: layer input {k!r} of type {type(v).__name__} cannot cross the link")
    fields["kw"], fields["tup"], fields["none"] = scal, tup, none


def _unpack_kwargs(fields: dict, tensors: dict) -> dict:
    kw = dict(fields["kw"])
    for k in fields["none"]:
        kw[k] = None
    for k, n in fields["tup"].items():
        kw[k] = tuple(tensors[f"k:{k}:{i}"] for i in range(n))
    for name, t in tensors.items():
        if name.startswith("k:") and name.count(":") == 1:
            kw[name[2:]] = t
    return kw


# ---------------------------------------------------------------------------------------------------------------------
# main side
# ---------------------------------------------------------------------------------------------------------------------
class StagedState(list):
    """A slot state (_export_slot_state's list) whose remote layers are kept on their stage under an id."""
    remote: tuple = ()

    def __del__(self):
        for stage, sid in self.remote:
            stage.queue("free", sid)


class RemoteStage:
    def __init__(self, addr: str, a: int, b: int):
        self.addr, self.a, self.b = addr, a, b
        self.layers = range(a, b)
        self.link = None
        self._ops: list = []
        self._lock = threading.Lock()
        self._next_id = 0
        self.calls = self.wait_s = 0.0

    def connect(self, model) -> None:
        cfg = model.config.get_text_config(decoder=True)
        self.link = Link.connect(self.addr)
        f, _ = self.link.call("hello", {"layers": [self.a, self.b], "num_layers": len(model.model.layers),
                                        "hidden": cfg.hidden_size, "dtype": str(model.dtype).split(".")[1]})
        _log(f"[stage] {self.addr}: layers {self.a}..{self.b - 1} of {len(model.model.layers)} "
             f"on {f.get('device')} ({f.get('pack')})")

    def queue(self, name: str, *args) -> None:
        self._ops.append([name, *args])

    def call(self, op: str, fields: dict | None = None, tensors: dict | None = None, device=None):
        if self.link is None:
            raise StageLost(f"stage {self.addr} is lost: restart the stage worker, then this server")
        import time
        with self._lock:
            fields = dict(fields or {})
            fields["ops"], self._ops = self._ops, []
            t0 = time.perf_counter()
            try:
                out = self.link.call(op, fields, tensors, device)
            except StageLost:
                self.link.close()
                self.link = None          # every later call fails at once
                raise
            self.calls += 1
            self.wait_s += time.perf_counter() - t0
            return out

    # -- engine cache actions ----------------------------------------------------------------------------------------
    def reset(self, tau: float, residual: int) -> None:
        self._ops = []
        if self.link is None:
            return               # the stage was lost: the engine still builds, and every request fails with StageLost
        self.call("reset", {"tau": tau, "residual": residual})

    def export(self, bi: int):
        """-> state id on the stage, or None when a layer there cannot be exported."""
        self._next_id += 1
        f, _ = self.call("export", {"bi": bi, "id": self._next_id})
        return self._next_id if f["ok"] else None

    def import_(self, bi: int, sid: int) -> bool:
        return bool(self.call("import", {"bi": bi, "id": sid})[0]["ok"])

    def ring_plan(self, bi: int) -> str:
        """"ok" (a rollback is ready for slot bi), "none" (no recurrent layer here) or "fail"."""
        return self.call("ring_plan", {"bi": bi})[0]["plan"]

    # -- the forward -------------------------------------------------------------------------------------------------
    def forward(self, hidden, kw: dict):
        from qv_band.kv import step_ctx
        c = step_ctx.current()
        if c is None:
            raise RuntimeError("remote stage: a forward outside the continuous-batching engine "
                               "(remote stages are served by the scheduler only)")
        caps = sorted(li for li in c.capture_layers if li in self.layers) if c.capture is not None else []
        past = kw.get("past_key_values")
        ring_rows = getattr(past, "_ring_rows", None) if getattr(past, "_ring", None) is not None else None
        fields = {"segs": c.segs, "real_lens": c.real_lens, "cold_rows": c.cold_rows, "caps": caps,
                  "ring_rows": sorted(ring_rows) if ring_rows is not None else None}
        tensors = {"h": hidden}
        _pack_kwargs(kw, fields, tensors)
        _f, t = self.call("forward", fields, tensors, device=hidden.device)
        for li in caps:
            c.capture[li] = t[f"cap{li}"].to(c.capture_dev, non_blocking=True)
        return t["h"]


def attach(model, stages: list) -> None:
    """Route the layers of every stage in `stages` through its link. One call, at load."""
    if getattr(model, "_qv_stage_gate", None) is not None or os.environ.get("QV_PIPELINE", "").strip():
        raise RuntimeError("remote stages together with a multi-card --pipeline on the main machine: not built yet")
    layers = model.model.layers
    taken: set = set()
    for st in stages:
        if not (0 <= st.a < st.b <= len(layers)) or taken & set(st.layers):
            raise RuntimeError(f"stage {st.addr}: layers {st.a}:{st.b} out of range or overlapping another stage")
        if len(layers) - len(taken | set(st.layers)) < 1:
            raise RuntimeError("remote stages: the main machine must keep at least one decoder layer")
        taken |= set(st.layers)
        st.connect(model)
        layers[st.a].forward = (lambda h, _st=st, **kw: _st.forward(h, kw))
        for i in range(st.a + 1, st.b):
            layers[i].forward = (lambda h, **kw: h)
    model._qv_remote_stages = tuple(stages)
    model._qv_remote_layers = frozenset(taken)


def attach_from_env(model) -> bool:
    """QV_STAGES="HOST:PORT=a:b[,HOST:PORT=a:b]" (set by --pipeline HOST:PORT,local). True when stages were attached."""
    raw = (os.environ.get("QV_STAGES", "") or "").strip()
    if not raw:
        return False
    stages = []
    for item in raw.split(","):
        addr, _, rng = item.partition("=")
        a, _, b = rng.partition(":")
        if not (addr and a.isdigit() and b.isdigit()):
            raise RuntimeError(f"QV_STAGES={raw!r}: each stage as HOST:PORT=a:b")
        stages.append(RemoteStage(addr, int(a), int(b)))
    attach(model, stages)
    return True


# ---------------------------------------------------------------------------------------------------------------------
# stage side
# ---------------------------------------------------------------------------------------------------------------------
class StageHost:
    """Runs layers [a, b) of `model` for one main process at a time."""

    def __init__(self, model, tok, a: int, b: int, pack: str = ""):
        from qv_band.decode.spec_decode import ContinuousBatchEngine
        if not (0 <= a < b <= len(model.model.layers)):
            raise RuntimeError(f"stage layers {a}:{b} out of range (the model has {len(model.model.layers)})")
        self.model, self.a, self.b, self.pack = model, a, b, pack
        self.eng = ContinuousBatchEngine(model, tok)          # model preparation + arming, no drafter
        self.dev = next(model.model.layers[a].parameters()).device
        self.tau = self.residual = None
        self._clear()

    def _clear(self) -> None:
        self.eng.past = None
        self.states, self.snap, self.ring, self.plan = {}, None, None, None

    # -- cache actions ------------------------------------------------------------------------------------------------
    def _past(self):
        if self.eng.past is None:
            from qv_band.decode.spec_decode import _make_banded_cache
            if self.tau is None:
                raise RuntimeError("stage: cache action before reset")
            past = _make_banded_cache(self.model.config, tau=self.tau, residual_length=self.residual)
            _localize(past, range(self.a, self.b))
            self.eng.past = past
        return self.eng.past

    def _op(self, name: str, *args) -> None:
        from qv_band.decode import spec_decode as sd
        past = self._past()
        if name == "add_slot":
            cfg = self.model.config.get_text_config(decoder=True)
            heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
            head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
            like = torch.zeros(1, heads, 0, head_dim, dtype=self.model.dtype, device=self.dev)
            past.add_slot(like, like)
            if sd._linear_attention_state_ready(past):
                sd._grow_linear_attention_batch(past)
        elif name == "drop":
            past.drop_slot(args[0])
            if sd._linear_attention_state_ready(past):
                sd._drop_linear_attention_batch(past, args[0])
        elif name == "crop":
            past.crop(args[0], slot=args[1])
        elif name == "snapshot":
            self.snap = sd._snapshot_recurrent_state(past)
        elif name == "restore":
            sd._restore_recurrent_state_slot(past, self.snap, args[0])
        elif name == "ring_apply":
            sd._ring_apply(self.plan, args[0], args[1])
        elif name == "free":
            self.states.pop(args[0], None)
        else:
            raise RuntimeError(f"stage: unknown cache action {name!r}")

    # -- messages -----------------------------------------------------------------------------------------------------
    def handle(self, op: str, f: dict, t: dict):
        from qv_band.decode import spec_decode as sd
        for o in f.get("ops", ()):
            self._op(*o)
        if op == "forward":
            return self._forward(f, t)
        if op == "reset":
            self._clear()
            self.tau, self.residual = f["tau"], f["residual"]
            torch.cuda.empty_cache()
            return {}, None
        if op == "export":
            st = sd._export_slot_state(self._past(), f["bi"])
            if st is not None:
                self.states[f["id"]] = st
            return {"ok": st is not None}, None
        if op == "import":
            return {"ok": bool(sd._import_slot_state(self._past(), f["bi"], self.states[f["id"]]))}, None
        if op == "ring_plan":
            self.plan = sd._ring_plan(self._past(), self.ring, f["bi"]) if self.ring is not None else None
            return {"plan": "fail" if self.plan is None else "ok" if self.plan else "none"}, None
        if op == "hello":
            cfg = self.model.config.get_text_config(decoder=True)
            mine = {"layers": [self.a, self.b], "num_layers": len(self.model.model.layers),
                    "hidden": cfg.hidden_size, "dtype": str(self.model.dtype).split(".")[1]}
            theirs = {k: f.get(k) for k in mine}
            if theirs != mine:
                raise RuntimeError(f"stage and main do not agree: stage {mine}, main {theirs}")
            return {"device": str(self.dev), "pack": self.pack}, None
        raise RuntimeError(f"stage: unknown message {op!r}")

    def _forward(self, f: dict, t: dict):
        from qv_band.kv import step_ctx
        past, eng = self._past(), self.eng
        caps = f["caps"]
        if caps:
            eng._step_hooks(caps)
        kw = _unpack_kwargs(f, t)
        h = t["h"]
        self.ring = self.plan = None
        if f["ring_rows"] is not None:
            past._ring, past._ring_rows = {}, set(f["ring_rows"])
        try:
            with eng._armed(), step_ctx.active(
                    segs=[tuple(s) for s in f["segs"]] if f["segs"] is not None else None,
                    real_lens=f["real_lens"], cold_rows=f["cold_rows"], capture={} if caps else None,
                    capture_layers=caps, capture_dev=self.dev) as ctx:
                layers = self.model.model.layers
                for i in range(self.a, self.b):
                    h = layers[i](h, past_key_values=past, **kw)
        finally:
            if f["ring_rows"] is not None:
                self.ring, past._ring, past._ring_rows = past._ring, None, None
        out = {"h": h}
        for li in caps:
            out[f"cap{li}"] = ctx.capture[li]
        return {}, out

    def serve(self, port: int, host: str = "0.0.0.0") -> None:
        srv = listen(port, host)
        _log(f"[stage] layers {self.a}..{self.b - 1} on {self.dev}, listening on {host}:{port}")
        while True:
            link = accept(srv)
            _log(f"[stage] main connected from {link.peer}")
            try:
                while True:
                    op, f, t = link.recv(self.dev)
                    try:
                        with torch.no_grad():
                            rf, rt = self.handle(op, f, t)
                    except Exception as e:                # the main side raises StageLost with this text
                        import traceback
                        traceback.print_exc(file=sys.__stdout__)
                        link.send("error", {"text": f"{type(e).__name__}: {e}"})
                        continue
                    link.send("ok", rf, rt)
            except StageLost as e:
                _log(f"[stage] {e}; cache dropped, waiting for a main process")
            finally:
                link.close()
                self._clear()
                torch.cuda.empty_cache()
