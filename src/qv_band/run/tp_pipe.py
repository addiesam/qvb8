"""Pipeline parallel (PP) across machines, tensor parallel (TP) inside each machine -- for SLOW links between machines.

TP across machines needs two all-reduces per layer: ~130 network round trips per token for a 64-layer model; on a
3 Mbit/s link that alone is seconds per token. Here each MACHINE is one pipeline stage holding a contiguous range of
layers (its cards split those layers by TP: NCCL, CUDA graphs as usual). The messages form a ring:

    driver (last stage) --CTRL--> stage 0 --FWD--> stage 1 --FWD--> ... --FWD--> last stage (driver)
    CTRL: header + the tokens to run (+ how many tokens of the previous step to keep: the commit rides along)
    FWD : header + tokens + hidden states [T, H] fp16 + the drafter's captured features from earlier stages

so a decode step costs (#stages) network messages instead of 2 x layers. Message sizes follow T (a small header
says how large the payload is): 10 KB of hidden state per token for H = 5120.

The generation loop (generate(), unchanged) runs on the LAST stage: the token argmax, the lm_head and most of
the drafter's capture layers are there, so speculative decoding (DFlash) costs no extra round trip -- every
network round trip verifies a whole block (several tokens per trip on a slow link). All cards of the last stage run
generate() in lock step (the lm_head is TP-split: a collective); only the stage leader talks to the network.

Streaming: prefill sends every chunk's CTRL at once; stage 0 works through them and streams each chunk's hidden
states on while it computes the next, so all stages and the link work at the same time.

Several users (--slots N): every stage keeps N KV caches (one StaticDecoder per slot, the weights shared) and every
message names its slot. One user's step crossing the network leaves the other machines idle; with N users in flight
each machine works on one user's step while the others' steps travel -- the pipeline stays full. The last stage runs
one generate() per active slot, in lock step on all its cards (Driver: the order of events comes from the leader).

Prefill: a message carries up to QV_TP_PP_CHUNK prompt tokens (default 64); each stage runs it as several graph-sized
pieces. Fewer messages: fewer network round trips per prompt.

Wire: payloads are compressed losslessly (fp16 bytes split into high / low planes, zlib level 1; QV_TP_PP_ZIP=0
disables it). The hidden states' high bytes (sign, exponent) compress well; nothing is rounded, output stays exact.

Messages between machines go over gloo (TCP, host memory: any NCCL / CUDA / GPU mix, any network); inside a
machine the stage leader hands them to its cards with NCCL broadcasts.

Launch on every machine (stage order = machine order of their first global rank; rank 0's machine is stage 0):
    QV_TP_PP_LAYERS=20,44 torchrun --nnodes 2 --node-rank i --nproc-per-node <cards> --master-addr <host of rank 0> \\
        -m qv_band.run.tp_pipe --model <pack on this machine> [--draft <drafter pack>]
Dense Qwen3.5-family models; greedy (exact: speculative output == plain output).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import zlib

import numpy as np

import torch
import torch.distributed as dist

from qv_band.formats.qvb8.tp import TPGroup, split_weighted

STOP, RUN, RESET = 0, 1, 2
CTRL, FWD = 0, 1
NH = 8                      # int64 header: kind, T, commit, msg (CTRL / FWD), feature layers, H, slot, wire bytes
ZIP = os.environ.get("QV_TP_PP_ZIP", "1") == "1"
CHUNK = int(os.environ.get("QV_TP_PP_CHUNK", "64"))


def _hdr(kind, T=0, c=0, msg=CTRL, nf=0, H=0, slot=0):
    return [kind, T, c, msg, nf, H, slot, 0]


def _zip(a: torch.Tensor) -> bytes | None:
    """Lossless: byte planes (every 2nd byte together: fp16 high bytes compress well), then zlib level 1.
    None when it does not pay (random-looking data)."""
    b = a.numpy()
    z = zlib.compress(np.ascontiguousarray(b.reshape(-1, 2).T).tobytes(), 1)
    return z if len(z) < 0.95 * len(b) else None


def _unzip(z: bytes, n: int) -> torch.Tensor:
    b = np.frombuffer(zlib.decompress(z), dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(b.reshape(2, n // 2).T).reshape(-1))


def _parts(T, sizes):
    """How a stage runs T tokens: the largest graph size that fits, again and again (every stage splits alike)."""
    out = []
    while T:
        s = max([x for x in sizes if x <= T] or [1])
        out.append(s)
        T -= s
    return out


def _size(h) -> int:
    """Payload bytes of a message: CTRL = tokens; FWD = tokens + hidden [T, H] + features [T, nf * H] (fp16)."""
    kind, T, _, msg, nf, H = h[:6]
    if kind != RUN:
        return 0
    return T * 8 if msg == CTRL else T * 8 + T * H * 2 * (1 + nf)


def _views(pay, T, H, nf):
    """tokens [T], hidden [1, T, H], features [T, nf * H] inside a FWD payload."""
    o = T * 8
    toks = pay[:o].view(torch.long)
    hid = pay[o:o + T * H * 2].view(torch.float16).view(1, T, H)
    o += T * H * 2
    feats = pay[o:o + T * nf * H * 2].view(torch.float16).view(T, nf * H)
    return toks, hid, feats


class Pipe:
    """Who is who: stages (one per machine), this rank's stage, its leader, its TP group inside the machine."""

    def __init__(self):
        dist.init_process_group("gloo")
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        hosts = [None] * self.world
        dist.all_gather_object(hosts, (os.environ.get("GROUP_RANK") or socket.gethostname(), self.rank))
        nodes = {}
        for h, r in hosts:
            nodes.setdefault(h, []).append(r)
        self.stages = sorted(nodes.values(), key=lambda rs: rs[0])
        self.n = len(self.stages)
        for s, rs in enumerate(self.stages):          # every rank takes part in creating every group
            g = dist.new_group(rs, backend="nccl") if len(rs) > 1 else None
            if self.rank in rs:
                self.stage, self.ranks, self.group = s, rs, g
                self.tp = TPGroup(rs.index(self.rank), len(rs), g)
        self.leader = self.ranks[0]
        self.is_leader = self.rank == self.leader
        self.first, self.last = self.stage == 0, self.stage == self.n - 1
        # the ring: stage s -> s + 1, the last stage -> stage 0
        self.next = self.stages[(self.stage + 1) % self.n][0]
        self.prev = self.stages[(self.stage - 1) % self.n][0]

    def layers(self, n_layers):
        """(lo, hi) of this stage. QV_TP_PP_LAYERS='20,44' (layer counts per stage), or QV_TP_PP_WEIGHTS='1,2',
        default: in proportion to the cards of each machine."""
        cnt = os.environ.get("QV_TP_PP_LAYERS")
        if cnt:
            c = [int(x) for x in cnt.split(",")]
            if len(c) != self.n or sum(c) != n_layers:
                raise ValueError(f"QV_TP_PP_LAYERS={cnt}: need {self.n} counts summing to {n_layers}")
            lo = sum(c[:self.stage])
            return lo, lo + c[self.stage]
        w = os.environ.get("QV_TP_PP_WEIGHTS")
        w = [float(x) for x in w.split(",")] if w else [len(rs) for rs in self.stages]
        return split_weighted(n_layers, w, self.stage)


class Link:
    """Variable-size messages: a header (NH int64), then a payload of _size(header) bytes.
    Network: gloo isend / recv of host buffers (a ring of buffers: a send never waits for the receiver).
    Machine: NCCL broadcasts of the same two pieces from the stage leader."""

    def __init__(self, p: Pipe, dev, t_max, H, ncap, ring=8):
        self.p, self.dev = p, dev
        maxb = t_max * 8 + t_max * H * 2 * (1 + ncap)
        self.hs = [torch.zeros(NH, dtype=torch.long) for _ in range(ring)]
        self.bs = [torch.zeros(maxb, dtype=torch.uint8) for _ in range(ring)]
        self.works = [[] for _ in range(ring)]
        self.k = 0
        self.rh = torch.zeros(NH, dtype=torch.long)
        self.rb = torch.zeros(maxb, dtype=torch.uint8)
        self.gh = torch.zeros(NH, dtype=torch.long, device=dev)
        self.gb = torch.zeros(maxb, dtype=torch.uint8, device=dev)
        self.raw = self.wire = 0                                      # payload bytes before / after compression

    def send(self, dst, h, payload=None):
        k = self.k = (self.k + 1) % len(self.hs)
        for w in self.works[k]:
            w.wait()
        hb, bb = self.hs[k], self.bs[k]
        h = list(h)
        n = _size(h)
        if n:
            bb[:n].copy_(payload[:n])
            z = _zip(bb[:n]) if ZIP and n >= 4096 else None
            if z is not None:
                h[7] = len(z)
                bb[:len(z)].copy_(torch.frombuffer(bytearray(z), dtype=torch.uint8))
            self.raw += n
            self.wire += h[7] or n
        hb.copy_(torch.tensor(h, dtype=torch.long))
        ws = [dist.isend(hb, dst)]
        if n:
            ws.append(dist.isend(bb[:h[7] or n], dst))
        self.works[k] = ws

    def recv(self, src):
        dist.recv(self.rh, src)
        h = self.rh.tolist()
        n, w = _size(h), h[7]
        if n:
            dist.recv(self.rb[:w or n], src)
            if w:
                self.gb[:n].copy_(_unzip(self.rb[:w].numpy().tobytes(), n))
            else:
                self.gb[:n].copy_(self.rb[:n])
            h[7] = 0
        return h

    def node(self, h=None):
        """Inside the machine: the leader's header + payload (already in self.gb) to its other cards."""
        p = self.p
        if p.tp.world > 1:
            if p.is_leader:
                self.gh.copy_(torch.tensor(h, dtype=torch.long))
            dist.broadcast(self.gh, p.leader, group=p.group)
            h = self.gh.tolist()
            n = _size(h)
            if n:
                dist.broadcast(self.gb[:n], p.leader, group=p.group)
        return h, self.gb[:_size(h)]

    def get(self, src):
        """A message from src (leader: network; its cards: broadcast)."""
        h = self.recv(src) if self.p.is_leader else None
        return self.node(h)

    def flush(self):
        for ws in self.works:
            for w in ws:
                w.wait()


class Stage:
    """This rank's part: its StaticDecoder per slot (its layers, its captured layers; the weights shared) and what it
    does with a message."""

    def __init__(self, p: Pipe, decs, lo, hi, caps, t_msg=None):
        decs = list(decs) if isinstance(decs, (list, tuple)) else [decs]
        self.p, self.decs, self.dec, self.lo, self.hi = p, decs, decs[0], lo, hi
        dec = decs[0]
        self.H = dec.h_in.shape[-1]
        self.t_msg = max(dec.t_max, t_msg or CHUNK)                   # tokens per message (prefill)
        self.sizes = sorted(dec.graphs) or list(range(1, dec.t_max + 1))
        self.caps = list(caps)                                        # global capture layers (drafter)
        self.nf_before = sum(1 for c in self.caps if c < lo)          # feature layers of earlier stages
        self.nf_here = sum(1 for c in self.caps if lo <= c < hi)
        # a capture layer that ends a stage IS the hidden state sent on: not sent twice (half the bytes per step)
        self.in_dup = (lo - 1) in self.caps and not p.first           # my input hidden = the feature of layer lo-1
        self.out_dup = (hi - 1) in self.caps and not p.last
        self.nf_out = self.nf_before + self.nf_here - int(self.out_dup)   # feature layers in my outgoing message
        self.link = Link(p, dec.dev, self.t_msg, self.H, len(self.caps))
        self.obuf = torch.zeros(self.t_msg * 8 + self.t_msg * self.H * 2 * (1 + len(self.caps)), dtype=torch.uint8,
                                device=dec.dev)
        self.ltok = torch.zeros(self.t_msg, dtype=torch.long, device=dec.dev)           # last stage: argmax per token
        self.lfeat = torch.zeros(self.t_msg, max(1, self.nf_here) * self.H, dtype=torch.float16, device=dec.dev)

    def parts(self, T):
        return _parts(T, self.sizes)

    def apply(self, h, pay):
        """Commit, then this stage's layers on the message (in graph-sized pieces, each committed but the last: its
        commit rides on the slot's next message). Returns (outgoing FWD payload, incoming features)."""
        kind, T, c, slot = h[0], h[1], h[2], h[6]
        dec = self.decs[slot]
        if c:
            dec.commit(c)
        if kind == RESET:
            dec.reset()
            return None, None
        if kind != RUN:
            return None, None
        toks = pay[:T * 8].view(torch.long)                           # CTRL and FWD both start with the tokens
        prior = hid = None
        if not self.p.first:
            _, hid, prior = _views(pay, T, self.H, h[4])
            if self.in_dup:                                           # the de-duplicated feature: the hidden itself
                prior = torch.cat([prior, hid[0]], dim=-1)
        o = None
        nh = self.nf_here - int(self.out_dup)
        if not self.p.last:
            o = self.obuf[:_size(_hdr(RUN, T, 0, FWD, self.nf_out, self.H))]
            t_o, hid_o, f_o = _views(o, T, self.H, self.nf_out)
            t_o.copy_(toks)
            if self.nf_before:
                f_o[:, :self.nf_before * self.H].copy_(prior)
        tl = toks.tolist() if self.p.first else None
        a = 0
        for s in self.parts(T):
            if a:
                dec.commit(prev)
            if dec.offload:
                dec.spill(s)
            if self.p.first:
                dec._embed(tl[a:a + s])
            else:
                dec.h_in[:, :s].copy_(hid[:, a:a + s])
            dec._forward(s)
            if not self.p.last:
                hid_o[:, a:a + s].copy_(dec.h_out[:, :s])
                if nh:
                    f_o[a:a + s, self.nf_before * self.H:].copy_(dec.feats[:s, :nh * self.H])
            else:
                self.ltok[a:a + s].copy_(dec.tok_out[:s])
                if self.nf_here:
                    self.lfeat[a:a + s].copy_(dec.feats[:s])
            a += s
            prev = s
        return o, prior


class PipeDecoder:
    """The StaticDecoder interface (run / commit / reset / prefill / feats) for generate(), for one slot, on every card
    of the LAST stage. CTRLs go to stage 0 (the stage leader sends); the answering FWD comes back through the Driver
    (several slots) or straight from the link (one slot, no Driver)."""

    def __init__(self, st: Stage, slot=0, drv=None):
        self.st, self.p, self.slot, self.drv = st, st.p, slot, drv
        self.dec = st.decs[slot]
        self.t_max, self.dev, self.H = self.dec.t_max, self.dec.dev, st.H
        self.offload, self.prefix = False, None
        self.graphs = self.dec.graphs
        self.feats = torch.zeros(st.t_msg, max(1, len(st.caps)) * self.H, dtype=torch.float16, device=self.dev)
        self.pending = 0
        self._first = None

    def _ctrl(self, kind, tokens=()):
        T = len(tokens)
        h = _hdr(kind, T, self.pending, CTRL, slot=self.slot)
        self.pending = 0
        if self.p.n == 1:                                             # one machine: no network
            pay = torch.tensor(list(tokens) or [0], dtype=torch.long, device=self.dev).view(torch.uint8)
            self._local().append((h, pay))
        elif self.p.is_leader:
            pay = torch.tensor(list(tokens) or [0], dtype=torch.long).view(torch.uint8)
            self.st.link.send(self.p.next, h, pay)
        if self.drv is not None:
            self.drv.inflight += 1

    def _local(self):
        st = self.st
        if not hasattr(st, "local"):
            st.local = []
        return st.local

    def _msg(self):
        if self.drv is not None:
            return self.drv.wait(self.slot)
        if self.p.n == 1:
            return self._local().pop(0)
        return self.st.link.get(self.p.prev)

    def _consume(self):
        """The FWD answering this slot's oldest CTRL: this stage's layers on it -> tokens, features."""
        h, pay = self._msg()
        _, prior = self.st.apply(h, pay)
        if h[0] != RUN:
            return h[0]
        T = h[1]
        nb = self.st.nf_before * self.H
        if nb:
            self.feats[:T, :nb].copy_(prior)
        if self.st.nf_here:
            self.feats[:T, nb:nb + self.st.nf_here * self.H].copy_(self.st.lfeat[:T])
        return self.st.ltok[:T].tolist()

    @torch.no_grad()
    def run(self, tokens):
        self._ctrl(RUN, tokens)
        return self._consume()

    def logits(self, T):
        """[T, vocab] logits of the last run (sampling): the last stage's decoder ran it as one piece."""
        return self.dec.logits(T)

    @property
    def first_logits(self):
        return self._first

    def commit(self, c):
        self.pending = int(c)                                         # rides on the next CTRL (no round trip)

    def reset(self):
        self.pending = 0
        self._ctrl(RESET)
        self._consume()

    def stop(self):
        self._ctrl(STOP)
        self._consume()
        self.st.link.flush()

    @torch.no_grad()
    def prefill(self, ids, drafter=None):
        """Streamed: every message's CTRL (up to t_msg tokens each) goes out at once, they come back in order."""
        msgs = [ids[i:i + self.st.t_msg] for i in range(0, len(ids), self.st.t_msg)]
        for m in msgs:
            self._ctrl(RUN, m)
            self.pending = self.st.parts(len(m))[-1]                  # the next message commits this one's last piece
        out, q0 = None, 0
        for m in msgs:
            out = self._consume()
            if drafter is not None:
                drafter.observe_features(q0, self.feats[:len(m)])
            q0 += len(m)
        self._first = self.dec.logits_out[self.st.parts(len(msgs[-1]))[-1] - 1].clone()
        return out[-1]                                                # pending = the last piece's commit


class Driver:
    """Several users on the last stage: one generate() thread per busy slot, run one at a time in the order of
    events -- a new job, or the next message off the ring (it names its slot). The leader picks the events and
    broadcasts them, so every card of the last stage runs the same GPU work (and collectives) in the same order."""

    def __init__(self, st: Stage, drafter, tp, eos, prefill="graph"):
        self.st, self.p, self.tp, self.eos, self.mode = st, st.p, tp, eos, prefill
        n = len(st.decs)
        self.pds = [PipeDecoder(st, i, self) for i in range(n)]
        self.drafters = [drafter] + [drafter.fork() for _ in range(n - 1)] if drafter is not None else [None] * n
        self.busy = [False] * n
        self.mail = [None] * n
        self.wake = [threading.Semaphore(0) for _ in range(n)]
        self.back = threading.Semaphore(0)
        self.inflight = 0
        self.dev = st.dec.dev
        self.ev = torch.zeros(4, dtype=torch.long, device=self.dev)

    # --- inside a slot's thread
    def wait(self, slot):
        self.back.release()                                           # hand the turn back to the driver
        self.wake[slot].acquire()
        return self.mail[slot]

    def _job(self, slot, ids, max_new, on_tokens, done, sp=(0.0, 0, 1.0)):
        from qv_band.run.tp_graph import generate
        torch.cuda.set_device(self.dev)
        try:
            with torch.no_grad():
                out, s = generate(self.pds[slot], ids, max_new, self.drafters[slot], self.tp, self.eos,
                                  on_tokens=on_tokens, prefill=self.mode, temperature=sp[0], top_k=int(sp[1]),
                                  top_p=sp[2])
            if done is not None:
                done(out, s)
        except Exception as e:                                        # noqa: BLE001
            print(f"[pp] slot {slot} failed: {e!r}", flush=True)
            if done is not None:
                done(None, {"error": repr(e)})
        finally:
            self.busy[slot] = False
            self.back.release()

    # --- the driver loop (main thread, every card of the last stage)
    def _bcast(self, ev):
        if self.tp is not None and self.tp.world > 1:
            if self.p.is_leader:
                self.ev.copy_(torch.tensor(ev, dtype=torch.long))
            dist.broadcast(self.ev, self.p.leader, group=self.p.group)
            ev = self.ev.tolist()
        return ev

    def _sp(self, job):
        """The job's (temperature, top_k, top_p) on every card of the stage (they decide whether a step samples)."""
        v = [float(job.get("temperature", 0.0)), float(job.get("top_k", 0)), float(job.get("top_p", 1.0))] if job \
            else [0.0, 0.0, 1.0]
        if self.tp is None or self.tp.world <= 1:
            return v
        t = torch.tensor(v, dtype=torch.float64, device=self.dev)
        dist.broadcast(t, self.p.leader, group=self.p.group)
        return t.tolist()

    def _ids(self, ids, n):
        if self.tp is None or self.tp.world <= 1:
            return ids
        t = torch.tensor(ids, dtype=torch.long, device=self.dev) if self.p.is_leader else \
            torch.empty(n, dtype=torch.long, device=self.dev)
        dist.broadcast(t, self.p.leader, group=self.p.group)
        return t.tolist()

    def serve(self, get_job):
        """get_job(block) -> {"ids", "max_new", "on_tokens", "done"} | None (nothing now) | "STOP" (leader only).
        Returns when the leader gets "STOP" and every slot is idle."""
        JOB, MSG, END = 1, 2, 0
        stopping = False
        while True:
            ev, job = None, None
            if self.p.is_leader:
                free = [i for i, b in enumerate(self.busy) if not b]
                if not stopping and free:
                    job = get_job(self.inflight == 0)
                    if job == "STOP":
                        stopping, job = True, None
                if job is not None:
                    ev = [JOB, free[0], len(job["ids"]), job["max_new"]]
                elif self.inflight:
                    ev = [MSG, 0, 0, 0]
                elif stopping or not free:
                    ev = [END, 0, 0, 0]
                else:
                    continue
            ev = self._bcast(ev)
            if ev[0] == END:
                return
            if ev[0] == JOB:
                slot = ev[1]
                ids = self._ids(job["ids"] if job else None, ev[2])
                sp = self._sp(job)
                self.busy[slot] = True
                on_t, done = (job["on_tokens"], job.get("done")) if job else (None, None)
                threading.Thread(target=self._job, args=(slot, ids, ev[3], on_t, done, sp), daemon=True).start()
                self.back.acquire()
                continue
            if self.p.n == 1:
                h, pay = self.st.local.pop(0)
            else:
                h, pay = self.st.link.get(self.p.prev)
            self.inflight -= 1
            slot = h[6]
            self.mail[slot] = (h, pay)
            self.wake[slot].release()
            self.back.acquire()

    def stop(self):
        pd = PipeDecoder(self.st, 0, None)
        pd.stop()


@torch.no_grad()
def serve(st: Stage):
    """Stages before the last: receive (CTRL on stage 0, FWD after), run, pass the FWD on -- until STOP."""
    p, L = st.p, st.link
    while True:
        h, pay = L.get(p.prev)
        kind, T = h[0], h[1]
        out, _ = st.apply(h, pay)
        if p.is_leader:
            L.send(p.next, _hdr(kind, T, h[2], FWD, st.nf_out, st.H, h[6]), out)
        if kind == STOP:
            L.flush()
            if p.is_leader and L.raw:
                print(f"[pp] stage {p.stage}: sent {L.raw / 1e6:.1f} MB as {L.wire / 1e6:.1f} MB on the wire "
                      f"({L.raw / max(1, L.wire):.2f}x)", flush=True)
            return


def build_stage(p: Pipe, model_path, device, draft=None, max_len=4096, block=8, slots=1, kv_window=None,
                verbose=True):
    """This machine's stage: the model's layers (lo..hi-1), one captured StaticDecoder per slot. -> (Stage, model, caps)"""
    from transformers import AutoConfig
    from qv_band.run.tp_qwen35 import build
    from qv_band.run.tp_graph import StaticDecoder
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    NL = getattr(cfg, "text_config", cfg).num_hidden_layers
    lo, hi = p.layers(NL)
    caps = []
    if draft:
        dc = json.load(open(os.path.join(draft, "config.json")))
        caps = list((dc.get("dflash_config") or {}).get("target_layer_ids") or [])
        block = (dc.get("dflash_config") or {}).get("block_size") or dc.get("block_size") or block
    if verbose:
        print(f"[pp] rank {p.rank}: stage {p.stage}/{p.n} layers {lo}..{hi - 1}, {p.tp.world} card(s) here, "
              f"{slots} slot(s)", flush=True)
    model, tcfg = build(model_path, p.tp, device, stage=(lo, hi), verbose=verbose and p.tp.rank == 0,
                        stage_embed=bool(draft) and p.last)
    decs = []
    for _ in range(slots):
        d = StaticDecoder(model, tcfg, device, max_len, t_max=block,
                          capture_layers=[c - lo for c in caps if lo <= c < hi], kv_window=kv_window)
        d.capture(sizes=(1, block))                          # inside a machine: NCCL only, graphs work
        decs.append(d)
    return Stage(p, decs, lo, hi, caps), model, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the pack on THIS machine (same pack on every machine)")
    ap.add_argument("--draft", default=None, help="DFlash drafter pack (runs on the last stage)")
    ap.add_argument("--prompt", action="append", default=None, help="repeat for several users")
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--block", type=int, default=8, help="prefill / verify graph size (drafter: its block size)")
    ap.add_argument("--slots", type=int, default=1, help="users in flight at once (same on every machine)")
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    prompts = a.prompt or ["Explain in three sentences why the sky is blue."]
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local}"
    torch.cuda.set_device(device)
    p = Pipe()
    st, model, caps = build_stage(p, a.model, device, a.draft, a.max_len, a.block, a.slots)
    dist.barrier()
    if not p.last:
        serve(st)
        dist.barrier()
        os._exit(0)
    drafter = None
    if a.draft:
        from qv_band.decode.dflash_draft import build_dflash_drafter
        drafter = build_dflash_drafter(model, a.draft, device=device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    eos = set(x for x in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")] if x is not None)
    tpg = p.tp if p.tp.world > 1 else None                       # drafts: the same block on every card here
    drv = Driver(st, drafter, tpg, eos)
    ids_all = [tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)).input_ids
               for q in prompts]

    def run_all(jobs_ids, new):
        res = [None] * len(jobs_ids)
        todo = list(enumerate(jobs_ids))

        def get_job(block):
            if not todo:
                return "STOP"
            i, ids = todo.pop(0)
            return {"ids": ids, "max_new": new, "on_tokens": None,
                    "done": (lambda i: lambda out, s: res.__setitem__(i, (out, s)))(i)}
        drv.serve(get_job)
        return res

    run_all(ids_all[:1], 12)                                     # warm (drafter graphs etc.)
    t0 = time.time()
    res = run_all(ids_all, a.new)
    dt = time.time() - t0
    if p.is_leader:
        n_tok = sum(len(r[0]) for r in res)
        print(f"[pp] {p.n} stages {p.stages}, {a.slots} slot(s): {len(prompts)} user(s), {n_tok} new tokens in "
              f"{dt:.2f}s = {n_tok / dt:.2f} tok/s total", flush=True)
        for q, ids, (out, s) in zip(prompts, ids_all, res):
            acc = f", accepted {s['accepted']}/{s['proposed']}" if drafter else ""
            print(f"[pp]  {len(ids)} prompt + {len(out)} new (prefill {s['t_prefill']:.2f}s, {s['steps']} steps{acc}): "
                  f"{tok.decode(out)!r}", flush=True)
        if a.dump:
            torch.save({"tokens": [r[0] for r in res]}, a.dump)
    drv.stop()
    if p.is_leader and st.link.raw:
        print(f"[pp] last stage: sent {st.link.raw / 1e6:.2f} MB as {st.link.wire / 1e6:.2f} MB", flush=True)
    dist.barrier()
    os._exit(0)


if __name__ == "__main__":
    main()
