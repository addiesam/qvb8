"""Tensor Parallel for any BatchTarget (engine/adapter.py): one target per card, run together as ONE target.

A model adapter builds one BatchTarget per card, each holding its share of the weights and calling
tp.all_reduce(...) (run/tp_dit.ThreadTP over a Hub: one Python thread per card, the cards' partial sums added in
the same order on every card) where its row-parallel outputs meet. TPTarget gives the engine the usual BatchTarget
surface, so engine/target_engine.TargetBatchEngine (continuous batching, prefix reuse, the shared prefix pool) serves
a TP model unchanged:

    state           one per card (each card's K/V heads), as a list
    forward(rows)   every card runs the same rows at once (its own states); card 0's logits are returned
    export/import   {"tp": [card 0's tree, card 1's tree, ...]}: a TP prefix restores only into the same split
    engine/spec.py  card 0 drafts (spec_propose); spec_verify runs on every card at once; spec_keep on every card

Every card computes the same logits (the replicated head after the last all-reduce), so card 0's are the answer.
"""
from __future__ import annotations

import os
import threading
import time

import torch


class TPTarget(torch.nn.Module):
    def __init__(self, ranks: list, devices: list, hub):
        super().__init__()
        self.ranks = torch.nn.ModuleList(ranks) if all(isinstance(r, torch.nn.Module) for r in ranks) else None
        self._ranks = list(ranks)
        self.devices = [torch.device(d) for d in devices]
        self.hub = hub
        self.world = len(self._ranks)
        r0 = self._ranks[0]
        self.stop_ids = set(getattr(r0, "stop_ids", ()) or ())
        self.dtype = getattr(r0, "dtype", None)
        self.config = getattr(r0, "config", None)
        self._qv_batch_target = True
        if getattr(r0, "spec_propose", None) is not None:   # engine/spec.py protocol: card 0 drafts
            self.spec_propose = lambda rows: r0.spec_propose([(st[0], last, sp) for st, last, sp in rows])
            self.spec_max_draft = r0.spec_max_draft
            self.spec_verify, self.spec_keep = self._spec_verify, self._spec_keep

    def prefill_peak(self, tokens):
        """The largest bytes one prefill of `tokens` tokens takes above what each card holds now (a scratch state)."""
        base = []
        for d in self.devices:
            torch.cuda.synchronize(d)
            torch.cuda.empty_cache()
            base.append(torch.cuda.memory_allocated(d))
            torch.cuda.reset_peak_memory_stats(d)
        st = self.new_state()
        self.forward([(st, [0] * int(tokens))])
        peak = max(torch.cuda.max_memory_allocated(d) - b for d, b in zip(self.devices, base))
        del st
        torch.cuda.empty_cache()
        return int(peak)

    def free_bytes(self):
        torch.cuda.empty_cache()
        return min(torch.cuda.mem_get_info(d)[0] for d in self.devices)

    def _parallel(self, work):
        """work(r) on every card at once (one thread per card: their all-reduces meet) -> card 0's result."""
        outs, errs = [None] * self.world, [None] * self.world

        def run(r):
            try:
                torch.cuda.set_device(self.devices[r])
                outs[r] = work(r)
                torch.cuda.current_stream(self.devices[r]).synchronize()
            except BaseException as e:                                   # noqa: BLE001
                errs[r] = e
                self.hub.bar.abort()                                     # the others leave their all-reduce
        ths = [threading.Thread(target=run, args=(r,), daemon=True) for r in range(1, self.world)]
        for t in ths:
            t.start()
        run(0)
        for t in ths:
            t.join()
        if any(e is not None for e in errs):
            self.hub.bar.reset()
            real = [e for e in errs if e is not None and not isinstance(e, threading.BrokenBarrierError)]
            raise (real or [e for e in errs if e is not None])[0]
        return outs[0]

    @torch.no_grad()
    def _spec_verify(self, rows):
        return self._parallel(lambda r: self._ranks[r].spec_verify([(st[r], toks) for st, toks in rows]))

    def _spec_keep(self, rows):
        for r, rk in enumerate(self._ranks):
            rk.spec_keep([(st[r], k) for st, k in rows])

    # -- state: one per card -----------------------------------------------------------------------
    def new_state(self):
        return [r.new_state() for r in self._ranks]

    def clone_state(self, state):
        return [r.clone_state(s) for r, s in zip(self._ranks, state)]

    def state_nbytes(self, state):
        return sum(r.state_nbytes(s) for r, s in zip(self._ranks, state))

    def export_state(self, state):
        return {"tp": [r.export_state(s) for r, s in zip(self._ranks, state)]}

    def import_state(self, tree, device=None):
        parts = tree["tp"]
        if len(parts) != self.world:
            raise ValueError(f"a prefix saved over {len(parts)} cards cannot restore over {self.world}")
        return [r.import_state(t, d) for r, t, d in zip(self._ranks, parts, self.devices)]

    # -- forward: every card at once -----------------------------------------------------------------
    @torch.no_grad()
    def forward(self, rows):
        t_fwd = time.perf_counter()
        outs, errs = [None] * self.world, [None] * self.world
        # image features: through host memory once (a card-to-card copy made on another card's stream read zeros on the V100:
        # the source was not finished yet; .cpu() waits for it) -- every card then copies its own
        rows = [(row[0], row[1], (row[2][0], row[2][1].detach().cpu())) if len(row) > 2 and row[2] is not None else row
                for row in rows]

        prof_now = (os.environ.get("QV_TP_CPROF") == "1" and len(rows) == 1 and len(rows[0][1]) == 1
                    and getattr(self, "_cp_n", 0) == 25)
        self._cp_n = getattr(self, "_cp_n", 0) + (1 if len(rows) == 1 and len(rows[0][1]) == 1 else 0)

        def run(r):
            try:
                if prof_now and r == 0:
                    import cProfile, pstats, io
                    cp = cProfile.Profile(); cp.enable()
                torch.cuda.set_device(self.devices[r])
                dev = self.devices[r]
                outs[r] = self._ranks[r].forward([(row[0][r], row[1]) + (
                    ((row[2][0], row[2][1].to(dev)),) if len(row) > 2 and row[2] is not None else ())
                    for row in rows])
                torch.cuda.current_stream(self.devices[r]).synchronize()
                if prof_now and r == 0:
                    cp.disable(); buf = io.StringIO()
                    pstats.Stats(cp, stream=buf).sort_stats("tottime").print_stats(28)
                    print("[tp-cprof] rank 0, one decode step:\n" + buf.getvalue()[:6000], flush=True)
            except BaseException as e:                                   # noqa: BLE001
                errs[r] = e
                self.hub.bar.abort()                                     # the others leave their all-reduce

        ths = [threading.Thread(target=run, args=(r,), daemon=True) for r in range(1, self.world)]
        for t in ths:
            t.start()
        run(0)
        for t in ths:
            t.join()
        if any(e is not None for e in errs):
            self.hub.bar.reset()
            real = [e for e in errs if e is not None and not isinstance(e, threading.BrokenBarrierError)]
            raise (real or [e for e in errs if e is not None])[0]
        if self.hub.prof:
            dt = time.perf_counter() - t_fwd
            self._pn = getattr(self, "_pn", 0) + 1
            self._pt = getattr(self, "_pt", 0.0) + dt
            if self._pn % 10 == 0 and len(rows) == 1:
                h = self.hub
                n = max(1, h.n_reduce - getattr(self, "_pr", 0))
                self._pr = h.n_reduce
                print(f"[tp-prof] step {dt*1e3:.0f} ms | all-reduces {n} | rank0: sync-before {h.t_comp[0]*1e3/10:.0f} ms, "
                      f"reduce total {h.t_all[0]*1e3/10:.0f} ms (of which barrier wait {h.t_wait[0]*1e3/10:.0f}) | rank1: "
                      f"sync-before {h.t_comp[1]*1e3/10:.0f}, reduce {h.t_all[1]*1e3/10:.0f} (wait {h.t_wait[1]*1e3/10:.0f}) "
                      f"via_host={h.via_host}", flush=True)
                h.t_all = [0.0] * h.world; h.t_wait = [0.0] * h.world; h.t_comp = [0.0] * h.world
        return outs[0]
