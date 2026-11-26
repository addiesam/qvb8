"""Continuous batching for any model that implements BatchTarget (engine/adapter.py), with prompt-prefix reuse.

A model adapter whose model is not an HF module stack (Inkling: its own forward) returns a BatchTarget from
ModelAdapter.build_target(); llm_server's BatchScheduler then drives THIS engine instead of
spec_decode.ContinuousBatchEngine. The engine-facing surface is the same one BatchScheduler already uses:

    add_request(ids, max_new=, temperature=, top_k=, top_p=, req_id=, on_token=)
    step()            one forward over every live request: prompt chunks and one-token decodes in the same batch;
                      requests of a target that drafts by itself (engine/spec.py) take one speculative step
    has_work(), slots (each .req_id .bi .max_new .out_ids .done), finished[req_id] = (bi, out_ids, Stats),
    _drop(bi), cancel_check, over_ceiling, past (None), _pending_prompts, _prefix_store

Prefix reuse: a request's state is stored (_prefix_store, LRU, a byte budget) at the end of its prompt AND at
the prompt's last whole-BLOCK boundary (the router names prefixes by whole blocks, servers/pool_keys.BLOCK; the
boundary entry also survives an edit of the last message). A later prompt that EXTENDS a stored one token for
token starts from a copy of that state and prefills only the rest; an IDENTICAL resend (a retry) samples
from the stored last logits with no forward. Cline resends the whole conversation every
turn, so a turn costs its new text.

The target owns the maths; this file only schedules: no model code, no kernels.

--prefix-store shared (QV_PREFIX_STORE=shared): the store is the machine's pool, decode/shared_prefix.py, the same one
ContinuousBatchEngine uses -- shared by every server on the machine, sent between machines by servers/pool_xfer, listed
to the router by /v1/pool/state. The target turns its state into the pool's plain tree (export_state / import_state).
A restore copies the entry onto the card, so its pool leases are released at once.
"""
from __future__ import annotations

import os
import time

import torch


class _Slot:
    def __init__(self, req_id, ids, max_new, temperature, top_k, top_p, on_token, state, reused):
        from qv_band.decode.spec_decode import Stats
        self.req_id, self.max_new = req_id, int(max_new)
        self.temperature, self.top_k, self.top_p = float(temperature or 0.0), int(top_k or 0), float(top_p or 1.0)
        self.on_token, self.state = on_token, state
        self.prompt = list(ids)
        self.todo = self.prompt[reused:]                 # prompt tokens not yet through the model
        self.out_ids: list[int] = []
        self.done = False
        self.bi = -1
        self.stats = Stats()
        self.t0 = time.time()
        self.t_prefill_end = None
        self.reused = reused
        self.next_logits = None
        self.static_slot = None                          # engine/static_decode slot once the prompt is through


class TargetBatchEngine:
    def __init__(self, target, tok, *, prefill_chunk: int = 2048, prefix_budget_gb: float | None = None,
                 block: int = 1024, **_):
        self.t = target
        self.tok = tok
        self.prefill_chunk = int(prefill_chunk or 2048)
        self.block = int(block)                          # = servers/pool_keys.BLOCK (prefix keys are whole blocks)
        gb = prefix_budget_gb if prefix_budget_gb is not None else float(os.environ.get("QV_TARGET_PREFIX_GB", "8"))
        self.prefix_budget = int(gb * 2**30)
        self.slots: list[_Slot] = []
        self.finished: dict = {}
        self.cancel_check = None
        self.over_ceiling = None
        self.past = None
        self._prefix_store: list[dict] = []              # {"tokens": list[int], "state": ..., "bytes": int, "t": float}
        self.stop_ids = set(getattr(target, "stop_ids", ()) or ())
        self._pressure = False                           # at the VRAM ceiling this step: store no prefixes
        from qv_band.decode.shared_prefix import store_from_env
        self._shared = store_from_env()                  # --prefix-store shared: the machine's pool, else None
        if self._shared is not None:
            if not (hasattr(target, "export_state") and hasattr(target, "import_state")):
                raise SystemExit(f"--prefix-store shared: {type(target).__name__} has no export_state/import_state")
            self._shared.release_all()                   # a rebuilt engine: the old one's requests are gone
        devs = getattr(target, "devices", None)          # TPTarget: card 0 holds the logits
        self._device = devs[0] if devs else (getattr(getattr(target, "m", None), "device", None) or "cuda")
        # static decode (engine/static_decode): decoding requests in fixed slots, one CUDA graph per step.
        # A target opts in with static_init/static_load/static_step; QV_STATIC_DECODE=0 turns it off.
        self.static = None
        self._static_ok = ((hasattr(target, "static_init") or hasattr(target, "static_stepper"))
                           and os.environ.get("QV_STATIC_DECODE", "1") != "0")
        if self._static_ok:                              # sized NOW, on the idle card, graphs included, so that a prompt
            from qv_band.engine.static_decode import size_stepper     # still fits (lazily, a long first prompt left
            try:
                self.static = size_stepper(target, self.prefill_chunk)
                target._qv_static_stepper = self.static
            except Exception as e:                       # noqa: BLE001 -- the eager path keeps working
                print(f"[target] static decode off: {e!r}", flush=True)
                self.static = None
            if self.static is None:
                self._static_ok = False
        self._dtype = getattr(target, "dtype", None) or torch.bfloat16     # the pool casts 16-bit tensors to it
        from qv_band.engine.spec import SpecPlanner
        self._planner = SpecPlanner()                    # eager speculative steps (a target that drafts by itself)

    # -- BatchScheduler surface ----------------------------------------------------------------------
    @property
    def _pending_prompts(self):
        return {s.bi: torch.empty(len(s.todo)) for s in self.slots if s.todo}

    def has_work(self) -> bool:
        return any(not s.done for s in self.slots)

    def add_request(self, ids, *, max_new, temperature=0.0, top_k=0, top_p=1.0, req_id=0, on_token=None, **_):
        ids = ids.reshape(-1).tolist() if hasattr(ids, "reshape") else list(ids)
        if not ids:
            raise ValueError("empty prompt")
        state, reused, logits = self._restore_prefix(ids)
        s = _Slot(req_id, ids, max_new, temperature, top_k, top_p, on_token, state, reused)
        s.next_logits = logits                           # whole prompt restored: the first token needs no forward
        s.stats.n_prefill = len(s.todo)
        self.slots.append(s)
        self._reindex()
        print(f"[target] req #{req_id}: prompt {len(ids)} tok, prefix reused {reused}, to prefill {len(s.todo)}",
              flush=True)

    def _drop(self, bi: int) -> None:
        if 0 <= bi < len(self.slots):
            s = self.slots.pop(bi)
            s.done = True
            s.state = None
            if s.static_slot is not None and self.static is not None:
                self.static.release(s.static_slot)
                s.static_slot = None
            self._reindex()

    def step(self):
        self._ceiling_stop()
        live = [s for s in self.slots if not s.done]
        if not live:
            return []
        # rows: every decoding slot feeds its last token; prompt chunks share a per-step token budget
        emitted = []
        for s in live:                                    # whole prompt restored: sample from the stored logits
            if s.next_logits is not None:
                lg, s.next_logits = s.next_logits, None
                s.t_prefill_end = time.time()
                s.stats.t_prefill = s.t_prefill_end - s.t0
                self._emit(s, self._sample(lg, s), emitted)
                self._to_static(s)
        live = [s for s in live if not s.done]
        rows, owners, budget = [], [], self.prefill_chunk
        st_toks, st_owners, sp_owners, sp_static = {}, [], [], {}
        for s in live:
            if not s.todo and s.out_ids:
                if s.static_slot is not None and self._spec(s):   # speculative, on the slot buffers (verify graphs)
                    sp_static[s.static_slot] = s
                    continue
                if s.static_slot is not None:            # in a static slot: decoded by the graph below
                    st_toks[s.static_slot] = s.out_ids[-1]
                    st_owners.append(s)
                    continue
                if self._spec(s):                        # the target drafts by itself: speculative step below
                    sp_owners.append(s)
                    continue
                rows.append((s.state, [s.out_ids[-1]]))
                owners.append(s)
        for s in live:
            if s.todo and budget > 0:
                done_n = len(s.prompt) - len(s.todo)
                n = min(budget, len(s.todo))
                cut = len(s.prompt) // self.block * self.block     # stop at the last whole-block boundary once
                if done_n < cut < done_n + n and cut < len(s.prompt):
                    n = cut - done_n
                take = s.todo[:n]
                budget -= n
                rows.append((s.state, take))
                owners.append(s)
        if st_toks:
            lgs = self.static.step(st_toks)
            for s in st_owners:
                self._emit(s, self._sample(lgs[s.static_slot], s), emitted)
        if sp_static:
            res = self.static.spec({sl: (s.out_ids[-1], self._sp(s)) for sl, s in sp_static.items()})
            for sl, s in sp_static.items():
                for tok in res[sl]:
                    if s.done:
                        break
                    self._emit(s, tok, emitted)
        if sp_owners:                                     # eager: draft + one verify forward for all of them
            for s, toks in zip(sp_owners, self._spec_eager([(s.state, s.out_ids[-1], self._sp(s)) for s in sp_owners])):
                for tok in toks:
                    if s.done:
                        break
                    self._emit(s, tok, emitted)
        if not rows:
            return emitted
        logits = self.t.forward(rows)                     # [len(rows), vocab] fp32: last token of each row
        for i, s in enumerate(owners):
            n_in = len(rows[i][1])
            if s.todo:                                    # this row was (part of) the prompt
                s.todo = s.todo[n_in:]
                if s.todo:
                    done_n = len(s.prompt) - len(s.todo)
                    if done_n == len(s.prompt) // self.block * self.block and done_n > s.reused:
                        self._save_prefix(s, done_n)          # the last whole-block boundary (router key)
                    continue
                s.t_prefill_end = time.time()
                s.stats.t_prefill = s.t_prefill_end - s.t0
                self._save_prefix(s, logits=logits[i])
                self._emit(s, self._sample(logits[i], s), emitted)
                self._to_static(s)
                continue
            self._emit(s, self._sample(logits[i], s), emitted)
        return emitted

    def _emit(self, s, tok, emitted):
        s.out_ids.append(tok)
        s.stats.emitted = len(s.out_ids)
        if s.on_token is not None:
            s.on_token(tok)
        emitted.append((s.req_id, [tok]))
        if tok in self.stop_ids or len(s.out_ids) >= s.max_new:
            self._finish(s)

    # -- internals --------------------------------------------------------------------------------
    def _spec(self, s) -> bool:
        """A request on a target that drafts by itself (engine/spec.py protocol, e.g. Inkling's own drafter): greedy or
        speculative sampling."""
        return getattr(self.t, "spec_propose", None) is not None

    @staticmethod
    def _sp(s):
        return (s.temperature, s.top_k, s.top_p)

    def _spec_eager(self, rows):
        """One speculative step of eager requests [(state, last, sampling)] (engine/spec.eager_step)."""
        from qv_band.engine.spec import eager_step
        return eager_step(self.t, self._planner, rows)

    def _to_static(self, s) -> None:
        """After its prompt: move a still-running request into a static slot (its eager state is then dropped).
        Speculative requests go there when the target verifies on the slot buffers (static_verify), else stay eager."""
        if not self._static_ok or s.done or s.static_slot is not None or s.state is None:
            return
        if self.static is None:
            from qv_band.engine.static_decode import get_stepper
            try:                                         # the target's one stepper (also used by image chats)
                self.static = get_stepper(self.t)
            except Exception as e:                       # noqa: BLE001 -- the eager path keeps working
                print(f"[target] static decode off: {e!r}", flush=True)
                self._static_ok = False
                return
        spec = self._spec(s)
        if spec and getattr(self.t, "static_verify", None) is None:
            return                                       # the target verifies only eagerly
        if len(s.prompt) + s.max_new + (self.static.spec_margin if spec else 0) > self.static.max_len:
            return                                       # longer than the slot buffers: eager
        slot = self.static.acquire()
        if slot is None:
            return                                       # every slot busy: eager
        self.static.load(slot, s.state)
        s.static_slot, s.state = slot, None

    def _ceiling_stop(self) -> None:
        """At the --max-memory-gb ceiling (the scheduler's vram_abort_check): first give back the prefix store, then
        end the request admitted LAST with what it has (a normal end of stream). Without this the next allocation
        hits the allocator's hard cap, the step raises, and every live request fails together."""
        chk = self.over_ceiling
        self._pressure = bool(chk is not None and chk())
        if not self._pressure:
            return
        if self._prefix_store:                           # no new prefixes are stored while _pressure holds
            n, nb = len(self._prefix_store), sum(e["bytes"] for e in self._prefix_store)
            self._prefix_store = []
            torch.cuda.empty_cache()
            print(f"[target] VRAM ceiling reached: dropped {n} stored prefixes ({nb / 2**30:.2f} GiB)", flush=True)
            if not chk():
                return
        live = [s for s in self.slots if not s.done]
        if not live:
            return
        s = live[-1]
        s.stats.vram_stop = True
        print(f"[target] VRAM ceiling reached: request {s.req_id} stopped after {len(s.out_ids)} generated tokens; "
              f"{len(live) - 1} request(s) keep running", flush=True)
        self._finish(s)
        torch.cuda.empty_cache()

    def _reindex(self):
        for i, s in enumerate(self.slots):
            s.bi = i

    def _finish(self, s):
        s.stats.steps = len(s.out_ids)
        self.finished[s.req_id] = (s.bi, list(s.out_ids), s.stats)
        self._drop(self.slots.index(s))

    @staticmethod
    def _sample(lg, s) -> int:
        if s.temperature <= 0:
            return int(lg.argmax())
        lg = lg / s.temperature
        if 0 < s.top_k < lg.numel():
            v, _ = torch.topk(lg, s.top_k)
            lg = torch.where(lg < v[-1], torch.full_like(lg, float("-inf")), lg)
        p = torch.softmax(lg, -1)
        if s.top_p < 1.0:
            sp, si = torch.sort(p, descending=True)
            keep = sp.cumsum(-1) - sp < s.top_p
            p = torch.zeros_like(p).scatter_(0, si[keep], sp[keep])
        return int(torch.multinomial(p / p.sum(), 1))

    def _restore_prefix(self, ids):
        """-> (state, tokens reused, logits of the last reused token or None). A stored entry is usable when the
        prompt extends it, or equals it and the entry kept its last logits (an identical resend: no forward)."""
        best = None
        if self._shared is not None:                     # longest first; an entry that cannot serve (evicted, or the
            cand = [e for e in self._shared.entries()    # whole prompt without logits) falls back to the next shorter one
                    if len(e["tokens"]) <= len(ids) and ids[:len(e["tokens"])] == e["tokens"]]
            for e in sorted(cand, key=lambda e: -len(e["tokens"])):
                state, n, lg = self._restore_shared(e, ids)
                if n:
                    return state, n, lg
            return self.t.new_state(), 0, None
        for e in self._prefix_store:
            n = len(e["tokens"])
            ok = n < len(ids) or (n == len(ids) and e.get("logits") is not None)
            if ok and (best is None or n > len(best["tokens"])) and ids[:n] == e["tokens"]:
                best = e
        if best is None:
            return self.t.new_state(), 0, None
        best["t"] = time.time()
        n = len(best["tokens"])
        return self.t.clone_state(best["state"]), n, (best["logits"] if n == len(ids) else None)

    def _restore_shared(self, e, ids):
        got = self._shared.get(e["key"], self._device, self._dtype)
        n = len(e["tokens"])
        if got is None:                                  # evicted since the lookup
            return self.t.new_state(), 0, None
        tree, _, bids = got
        try:
            lg = tree.get("logits")
            if n == len(ids) and lg is None:             # a grid checkpoint of exactly this prompt: no logits kept
                return self.t.new_state(), 0, None
            state = self.t.import_state(tree["state"], self._device)
            lg = lg.to(self._device).float().clone() if (n == len(ids) and lg is not None) else None
        finally:
            self._shared.release(bids)                   # copied onto the card: the pool may evict it again
        return state, n, lg

    def _save_prefix(self, s, n=None, logits=None):
        """Store the state after the first n prompt tokens (default: the whole prompt, with its last logits)."""
        if self.prefix_budget <= 0 or self._pressure:
            return
        toks = list(s.prompt[:n] if n is not None else s.prompt)
        if self._shared is not None:                     # the machine's pool: copied to host RAM once, LRU by bytes
            self._shared.put(toks, {"state": self.t.export_state(s.state),
                                    "logits": None if logits is None else logits.detach()},
                             None, False, self._dtype)
            return
        st = self.t.clone_state(s.state)
        nb = self.t.state_nbytes(st)
        self._prefix_store = [e for e in self._prefix_store if e["tokens"] != toks]
        lg = None if logits is None else logits.detach().clone()
        self._prefix_store.append({"tokens": toks, "state": st, "logits": lg, "t": time.time(),
                                   "bytes": nb + (0 if lg is None else lg.numel() * lg.element_size())})
        while sum(e["bytes"] for e in self._prefix_store) > self.prefix_budget and len(self._prefix_store) > 1:
            self._prefix_store.remove(min(self._prefix_store, key=lambda e: e["t"]))
