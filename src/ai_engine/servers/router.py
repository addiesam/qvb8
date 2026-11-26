"""One endpoint in front of every model server: LLM replicas and job servers (image, video, audio) alike
(qv_band/design_doc/serving_topology.md). The router is the same for every model; what differs per kind of server is
only how a backend is scored.

    python -m ai_engine.servers.router --port 8000
        --backend v100c1=http://host-b:8080 --backend gb10=http://127.0.0.1:8080       (LLM servers)
        --backend h3c0=http://host-b:8101 --backend h3c1=http://host-b:8102        (job servers)
        --store pc=http://host-c:8082

A backend says what it is: an LLM server answers GET /v1/pool/state (model family, load, saved prefixes); a job server's
GET /health carries "routes" (the POST routes that submit work to it), "model", "queue", "busy", "last_job_seconds".

JOB SERVERS. POST <a job route> goes to the backend serving that route with the smallest wait: (queued + running jobs)
x its last job's duration, read live from the candidates at that moment, pick and forward as one step. The job id of
the answer is remembered: GET /v1/jobs/<id>, POST /v1/jobs/<id>/cancel, DELETE /v1/jobs/<id> and result files
(/videos/<id>.mp4, ...) reach the backend that holds the job (an id this router has not seen, e.g. after a restart, is
looked up on the backends). GET /v1/jobs merges the backends' lists. Two models on one route are told apart by the
request's "model". GET /ui/<backend or model>/... is that server's own page (its panel), proxied.

LLM SERVERS (the rest of this text).

MODELS. The backends may serve different models. Each server reports its model (GET /v1/pool/state "family": the source
model's name, e.g. Qwen3.8-27B); the backends of one model form a group, and a request is routed inside the group its
"model" field names (the family, or a backend's pack name; case does not matter). With one model up, any "model" value
is accepted, as a single server accepts any. With several, an unknown or missing "model" gets --default-model, or 404
with the list. GET /v1/models lists the models that are up.

    python -m ai_engine.servers.router \
        --backend v100c1=http://host-b:8080 --backend v100c0=http://host-b:8081 \
        --backend gb10=http://127.0.0.1:8080 \
        --rate v100c1=300/12 --rate v100c0=300/12 --rate gb10=500/15 --port 8000

ROUTING. For a request the router asks one replica for its prefix keys (POST /v1/pool/keys: that server's own prompt
builder and tokenizer, so the keys are the tokens the replicas will see), then scores every healthy replica r:

    T(r) = queued work on r      (its prompt tokens still to prefill / prefill rate + tokens still to decode / decode rate)
         + prefill of the part of this prompt that r does NOT hold   (n_tokens - cached_tokens(r)) / prefill rate
         + this request's decode (max_tokens / decode rate)

cached_tokens(r) = the longest prefix key of this request that r reports holding (GET /v1/pool/state, polled every
--poll seconds). The request goes to argmin T: the replica that already holds its prefix wins while its queue is
shorter than the prefill it saves elsewhere. Rates are tok/s per replica (--rate NAME=PREFILL/DECODE, measured).

PULLS BETWEEN MACHINES. A replica on another machine (another prefix pool) that holds a longer prefix of the request can
give it to r (POST r/v1/pool/pull, servers/pool_xfer.py): r's time then counts the transfer instead of that prefill,

    transfer = prefix tokens x bytes per token / link MB/s       (both learned from finished pulls; --net-mb-s to start)

and r is scored with the faster of the two (--pull cost: a pull happens only when the transfer is predicted to take
less time than the prefill it saves; on a 1 GbE link that is nearly always, on a slow test link rarely. --pull always
keeps the behaviour: pull whenever a longer prefix exists, for testing the transfer path). Donors are other replicas and --store prefix stores (a DRAM-only machine:
servers/prefix_store.py), which are polled like replicas but never get a request. Replicas must report
the same transfer signature (same packs, taus, checkpoint grid, dtype, drafter). A failed pull just means r prefills.

FAILURES. A replica that misses its state polls is marked down and gets no requests. A request whose replica fails
BEFORE any byte of the response reached the client is sent to the next best replica; once bytes have been forwarded,
a failure ends the response (no silent retry of a partly streamed answer). Every response carries X-QV-Replica.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

POST_ROUTES = ("/v1/chat/completions", "/v1/completions", "/generate")


class Replica:
    def __init__(self, name: str, url: str, prefill_tps: float, decode_tps: float, max_inflight: int = 0):
        u = urllib.parse.urlparse(url)
        self.name, self.url, self.host, self.port = name, url, u.hostname, u.port or 80
        self.prefill_tps, self.decode_tps = prefill_tps, decode_tps
        self.max_inflight = max_inflight      # 0 = no limit; 1 for a one-card V100 (it cannot prefill two long prompts)
        self.store_only = False               # a prefix store (--store): a pull donor, never gets a request
        self.up, self.state, self.keys, self.last_ok, self.inflight = False, {}, set(), 0.0, 0
        self.kind = None                      # "llm" | "job": what the backend said it is (None until it answered once)
        self.routes = ()                      # a job server's submit routes
        self.keys_by = {}                     # a prefix store: transfer signature -> its keys (it holds several models)
        self.family = None                    # the model this replica serves (its state's "family")
        # work this router sent and that has not finished yet: counted at once, not only when the next state poll shows
        # it (two requests 0.1 s apart both went to the same idle replica and one V100 cannot prefill two 10k prompts)
        # A reservation stops counting once a state poll taken AFTER the request was forwarded has arrived: from then on
        # the server's own load figures include it (counting both doubled every running request's work).
        self.reserved = []                    # [{"pre", "dec", "fwd": time forwarded or None}]
        self.state_time = 0.0                 # when the state now held was ASKED for
        self.lock = threading.Lock()

    def conn(self, timeout: float) -> http.client.HTTPConnection:
        from qv_band.common import tls
        return tls.http_connection(self.host, self.port, timeout)

    def queued_seconds(self) -> float:
        ld = (self.state or {}).get("load") or {}
        pre = ld.get("queued_prompt_tokens", 0) + ld.get("running_prefill_tokens", 0)
        dec = ld.get("queued_decode_tokens", 0) + ld.get("running_decode_tokens", 0)
        for x in list(self.reserved):
            if x["fwd"] is None or self.state_time <= x["fwd"]:      # not visible in the server's state yet
                pre, dec = pre + x["pre"], dec + x["dec"]
        return pre / self.prefill_tps + dec / self.decode_tps

    def cached_tokens(self, keys: list) -> int:
        best = 0
        for length, k in keys:
            if (length, k) in self.keys:
                best = length
        return best

    def score(self, n_tokens: int, keys: list, max_tokens: int) -> tuple:
        cached = self.cached_tokens(keys)
        t = self.queued_seconds() + (n_tokens - cached) / self.prefill_tps + max_tokens / self.decode_tps
        return t, cached


RETRYABLE_ERRORS = ("OutOfMemoryError", "CUDA out of memory", "StageUnavailable")


class AsyncJob:
    """An async job as the router submitted it: enough to run it again on another backend when the one holding it is lost
    (down, restarted without its queue: a 404) or failed through no fault of the request (out of memory on that card, a
    block stage gone). The client keeps the id it got first; the router answers it from wherever the job runs now."""
    __slots__ = ("route", "model", "raw", "ctype", "r", "bid", "tried", "moves")

    def __init__(self, route, model, raw, ctype, r, bid):
        self.route, self.model, self.raw, self.ctype, self.r, self.bid = route, model, raw, ctype, r, bid
        self.tried, self.moves = [], 0


class Router:
    def __init__(self, replicas: list, poll: float, down_after: float, default_max_tokens: int,
                 queue_timeout: float = 600.0, net_mb_s: float = 3.0, bytes_per_token: float = 41e3,
                 pull: str = "cost"):
        self.replicas, self.poll, self.down_after = replicas, poll, down_after
        self.net_mb_s = net_mb_s                     # link speed between two pools, MB/s (learned per pair from pulls)
        self.links = {}                              # (from pool, to pool) -> MB/s
        self.bytes_per_token = bytes_per_token       # a saved prefix's size per token (learned from pulls)
        self.queue_timeout = queue_timeout
        self.pull = pull                             # "cost" | "always"
        self.default_max_tokens = default_max_tokens
        self.default_model = None
        self.owner = {}                              # job id -> the backend that holds it
        self.jobs = {}                               # async job id (the one the client got) -> AsyncJob: resubmitted when lost
        self.panel = None                            # --panel: backend or model whose page is served at /
        self._rr = 0
        self.pick_lock = threading.Lock()

    started = 0.0
    MAX_ASYNC_JOBS = 1024                            # async job records kept (their request bodies only until they end)

    def remember_async(self, jid: str, route: str, model, raw: bytes, ctype: str, r) -> None:
        """Keep what is needed to run async job `jid` again elsewhere. Caller holds pick_lock."""
        self.jobs[jid] = AsyncJob(route, model, raw, ctype, r, jid)
        while len(self.jobs) > self.MAX_ASYNC_JOBS:
            self.jobs.pop(next(iter(self.jobs)))

    def job_target(self, jid: str):
        """(backend, its own id) of a job the client knows as `jid` (an async job moved elsewhere has a new id there)."""
        with self.pick_lock:
            a = self.jobs.get(jid)
        if a is not None:
            return a.r, a.bid
        r = self.job_owner(jid)
        return (r, jid) if r is not None else (None, jid)

    def start(self) -> None:
        self.started = time.time()
        for r in self.replicas:
            threading.Thread(target=self._poll_loop, args=(r,), daemon=True, name=f"poll-{r.name}").start()

    @staticmethod
    def fetch_state(r: Replica, timeout: float = 5.0) -> tuple:
        """("llm" | "job", state) of a backend, asked now. An LLM server (and a prefix store) answers /v1/pool/state; a
        job server's /health carries its routes. Raises when it answers neither."""
        c = r.conn(timeout=timeout)
        try:
            if r.kind != "job":
                c.request("GET", "/v1/pool/state")
                resp = c.getresponse()
                raw = resp.read()
                if resp.status == 200:
                    return "llm", json.loads(raw)
            c.request("GET", "/health")
            resp = c.getresponse()
            raw = resp.read()
            if resp.status == 200:
                st = json.loads(raw)
                if st.get("routes"):
                    return "job", st
            raise RuntimeError(f"neither an LLM server's /v1/pool/state nor a job server's /health with routes (HTTP {resp.status})")
        finally:
            c.close()

    def apply_state(self, r: Replica, kind: str, st: dict, asked: float | None = None) -> None:
        with r.lock:
            r.kind = kind
            r.state_time = time.time() if asked is None else asked
            if kind == "job":
                r.state, r.routes = st, tuple(st["routes"])
                r.family = st.get("model") or st.get("server")
                ready = st.get("status", "ok") == "ok"         # a job server still loading its model takes no jobs
            elif st.get("by_xfer") is not None:                 # a prefix store: keys per model
                r.keys_by = {sig: {(int(a), b) for a, b in v.get("prefixes", [])} for sig, v in st["by_xfer"].items()}
                r.state, r.keys = st, set().union(*r.keys_by.values()) if r.keys_by else set()
                ready = True
            else:
                r.state, r.keys = st, {(int(a), b) for a, b in st.get("prefixes", [])}
                r.family = st.get("family") or st.get("model")
                ready = True
            r.last_ok = time.time()
            if ready and not r.up:
                what = (f"job server {r.family}, routes {list(r.routes)}" if kind == "job" else
                        "prefix store" if r.store_only else f"{st.get('model')}, ctx {st.get('ctx_size')}")
                print(f"[router] {r.name} up ({what})", flush=True)
            r.up = ready

    def _poll_loop(self, r: Replica) -> None:
        while True:
            try:
                t_ask = time.time()
                self.apply_state(r, *self.fetch_state(r), asked=t_ask)
            except Exception as e:                  # noqa: BLE001 - a down replica is a normal state here
                if r.up and time.time() - r.last_ok > self.down_after:
                    print(f"[router] {r.name} down ({type(e).__name__}: {e})", flush=True)
                    r.up = False
            time.sleep(self.poll)

    # ---- job servers ---------------------------------------------------------------------------------------------
    def job_backends(self, route: str | None = None) -> list:
        return [r for r in self.replicas if r.up and r.kind == "job" and (route is None or route in r.routes)]

    @staticmethod
    def job_wait(r: Replica) -> float:
        """Seconds a new job would wait on r: the rest of the running job + the jobs waiting (queued + being submitted)
        x one job's time. One job's time = the last finished job's, or how long the running one has run when that is
        longer. A running job past every known duration (a server that has not finished a job yet, or a job longer
        than the last) is expected to run as long again: such a server is measured, not assumed fast."""
        st = r.state or {}
        last, running = float(st.get("last_job_seconds") or 0.0), float(st.get("running_seconds") or 0.0)
        per = max(last, running) or 1.0
        rest = 0.0
        if st.get("busy"):
            rest = last - running if last > running else (running or per)
        return (int(st.get("queue") or 0) + r.inflight) * per + rest

    def job_pick(self, route: str, model: str | None, exclude: tuple = ()):
        """(backend, None) for a new job on `route`, or (None, (http code, message)). Candidates are asked for their
        queue NOW (the 1 s poll is too old when two jobs arrive together). Caller holds pick_lock.
        exclude: backends that already failed this job (a retry goes elsewhere).
        A backend that has not answered its first poll yet is asked here too: a job sent right after the router came
        up otherwise only sees the card that answered first."""
        for r in self.replicas:
            if r.kind is None and r not in exclude:
                try:
                    t_ask = time.time()
                    self.apply_state(r, *self.fetch_state(r, timeout=2.0), asked=t_ask)
                except Exception:                   # noqa: BLE001 - still down: the poll keeps trying
                    pass
        cands = [r for r in self.job_backends(route) if r not in exclude]
        if not cands:
            known = sorted({rt for r in self.replicas if r.kind == "job" for rt in r.routes})
            return None, ((503, f"no server for {route} is up") if any(route in r.routes for r in self.replicas)
                          else (404, f"no such route: {route} (job routes here: {known})"))
        fams = sorted({r.family for r in cands})
        if len(fams) > 1:
            want = (model or "").strip().lower()
            pick = [f for f in fams if f and f.lower() == want]
            if not pick:
                return None, (404, f"{route} is served by several models; \"model\" must be one of {fams}")
            cands = [r for r in cands if r.family == pick[0]]
        for r in cands:
            try:
                t_ask = time.time()
                self.apply_state(r, *self.fetch_state(r, timeout=3.0), asked=t_ask)
            except Exception:                       # noqa: BLE001 - it keeps its last state; the poll marks it down
                pass
        cands = [r for r in cands if r.up] or cands
        # expected COMPLETION, not just the wait: the wait + how long a job takes on THAT backend (its last finished job).
        # By wait alone two idle backends tie and the first won every time -- on a V100 pair that was the throttled card
        # (684 s per Sonic clip there vs 132 s on the other). A backend with no finished job yet counts as the fastest
        # known, so it is tried; a slow one gets work only when it would still finish first.
        known = [t for t in (self.job_seconds(r) for r in cands) if t]
        floor = min(known) if known else 0.0
        return min(cands, key=lambda r: (self.job_wait(r) + (self.job_seconds(r) or floor), self.job_seconds(r) > 0,
                                         r.inflight)), None

    @staticmethod
    def job_seconds(r: Replica) -> float:
        """How long one job took on r (its last finished job), 0 when it has not finished one."""
        return float((r.state or {}).get("last_job_seconds") or 0.0)

    def job_owner(self, jid: str):
        """The backend that holds job `jid`: remembered from its submit, else asked for (a router restart)."""
        with self.pick_lock:
            r = self.owner.get(jid)
        if r is not None:
            return r
        for r in self.job_backends():
            try:
                c = r.conn(timeout=5)
                c.request("GET", f"/v1/jobs/{jid}")
                resp = c.getresponse()
                resp.read()
                c.close()
                if resp.status == 200:
                    with self.pick_lock:
                        self.owner[jid] = r
                    return r
            except Exception:                       # noqa: BLE001
                continue
        return None

    def healthy(self) -> list:
        return [r for r in self.replicas if r.up]

    def serving(self) -> list:
        """The LLM replicas that are up."""
        return [r for r in self.replicas if r.up and not r.store_only and r.kind == "llm"]

    def families(self) -> dict:
        """{model: [its replicas that are up]}."""
        out = {}
        for r in self.serving():
            out.setdefault(r.family or "?", []).append(r)
        return out

    def group_for(self, body: dict):
        """(model, its replicas) for a request, or (None, the models that are up) when its "model" names none of
        several."""
        fam = self.families()
        want = str(body.get("model") or "").strip().lower()
        for name, reps in fam.items():
            if want and (want == name.lower() or any(want == str((r.state or {}).get("model", "")).lower() for r in reps)):
                return name, reps
        if len(fam) == 1:
            return next(iter(fam.items()))
        if self.default_model and self.default_model in fam:
            return self.default_model, fam[self.default_model]
        return None, sorted(fam)

    def keys_for(self, body: dict, live: list) -> tuple:
        """(n_tokens, keys) from one healthy replica, round robin; (0, []) when none answers (route by load only)."""
        for i in range(len(live)):
            r = live[(self._rr + i) % len(live)]
            try:
                c = r.conn(timeout=10)
                c.request("POST", "/v1/pool/keys", json.dumps(body), {"Content-Type": "application/json"})
                resp = c.getresponse()
                d = json.loads(resp.read())
                c.close()
                if resp.status == 200:
                    self._rr += 1
                    return int(d["n_tokens"]), d["keys"]
            except Exception:                       # noqa: BLE001 - try the next replica
                continue
        return 0, []

    def ranked(self, body: dict, group: list | None = None, nk: tuple | None = None, why: dict | None = None) -> list:
        """[(replica, predicted seconds, cached tokens, prefill tokens, decode tokens, pull plan)] best first, over the
        replicas of the request's model (`group`; default: every serving replica) that are up. `nk`: (n_tokens, keys)
        asked for once by the caller (a request waiting for a free replica must not tokenize its prompt on a replica
        at every retry). `why` gets "too_long": True when replicas are up but the prompt fits none of them."""
        group = self.serving() if group is None else [r for r in group if r.up]
        if not group:
            return []
        live = group + [r for r in self.healthy() if r.store_only]      # the stores: pull donors
        n, keys = self.keys_for(body, group) if nk is None else nk
        max_tokens = int(body.get("max_tokens") or body.get("max_new_tokens") or self.default_max_tokens)
        out, fits = [], 0
        for r in live:
            if r.store_only:
                continue
            if n and r.state.get("ctx_size") and n + max_tokens > int(r.state["ctx_size"]):
                continue                            # cannot admit it
            fits += 1
            if r.max_inflight and r.inflight >= r.max_inflight:
                continue                            # full: the request waits for a free replica (Handler.do_POST)
            if r.state.get("accepting") is False:
                continue                            # its own waiting queue is full (server --max-queue): same
            t, cached = r.score(n, keys, max_tokens)
            best = (r, t, cached, n - cached, max_tokens, None)
            pl = self.pull_plan(r, keys, cached, live)
            if pl is not None:
                s, hexkey, c_s, t_x = pl
                tp = r.queued_seconds() + t_x + (n - c_s) / r.prefill_tps + max_tokens / r.decode_tps
                if self.pull == "always" or tp < t:  # "cost": only when the transfer beats the prefill it saves
                    best = (r, tp, c_s, n - c_s, max_tokens, (s, hexkey, c_s, t_x))
            out.append(best)
        if why is not None and not fits:
            why["too_long"] = True
        out.sort(key=lambda x: (x[1], x[0].inflight))
        return out

    def pull_plan(self, r: Replica, keys: list, cached: int, live: list):
        """(donor, prefix key hex, its tokens, predicted transfer seconds) when another machine's replica or prefix store
        holds a longer prefix of this request; else None. Always pulled when it exists (no cost test: the link here is
        a test link, ); the predicted transfer still counts in r's score, so a free replica that already
        holds the prefix is preferred over one that must pull it."""
        mine = (r.state or {}).get("pool")
        sig = (r.state or {}).get("xfer")
        if not mine or not sig or (r.state or {}).get("local_only"):
            return None                               # a split server's pool holds only its own layers: no pulls into it
        best = None
        for s in live:
            st = s.state or {}
            if s is r or not st.get("pool") or st["pool"] == mine or st.get("local_only"):
                continue                              # (a split server does not export its partial entries)
            if s.store_only:
                held = s.keys_by.get(sig, ())                 # a store keeps each model's prefixes apart
            elif st.get("xfer") == sig:
                held = s.keys
            else:
                continue
            c_s, hexkey = 0, None
            for length, k in keys:
                if (length, k) in held:
                    c_s, hexkey = length, k
            if c_s <= cached or best is not None and c_s <= best[2]:
                continue
            mb_s = self.links.get((st["pool"], mine), self.net_mb_s)
            t_x = c_s * self.bytes_per_token / (mb_s * 2**20)
            best = (s, hexkey, c_s, t_x)
        return best

    def learn_pull(self, s: Replica, r: Replica, res: dict) -> None:
        """A finished pull: its link speed and bytes per token (moving averages)."""
        if not res.get("bytes") or not res.get("seconds") or not res.get("n"):
            return
        key = ((s.state or {}).get("pool"), (r.state or {}).get("pool"))
        mb_s = res["bytes"] / 2**20 / res["seconds"]
        self.links[key] = 0.5 * self.links.get(key, mb_s) + 0.5 * mb_s
        if not res.get("reused"):                    # a whole prefix: its size per token
            self.bytes_per_token = 0.5 * self.bytes_per_token + 0.5 * res["bytes"] / res["n"]


def _reserve(cand: tuple, sign: int, res: dict | None = None) -> dict | None:
    """Count (sign +1) a routed request's work on its replica, or release it (-1, with the dict the +1 returned)."""
    r, _, _, pre, dec = cand[:5]
    with r.lock:
        r.inflight += sign
        if sign > 0:
            res = {"pre": pre, "dec": dec, "fwd": None}
            r.reserved.append(res)
            return res
        if res in r.reserved:
            r.reserved.remove(res)
    return None


class Handler(BaseHTTPRequestHandler):
    router: Router = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def _json(self, code: int, obj) -> None:
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path in ("", "/index.html") and self.router.panel:      # --panel NAME: that server's page is the front page
            r = next((x for x in self.router.replicas if x.up and self.router.panel in (x.name, x.family)), None)
            if r is None:
                return self._json(503, {"error": f"panel backend {self.router.panel!r} is not up"})
            return self._proxy(r, "GET", "/")
        if path in ("/health", "", "/v1/pool"):
            reps = [{"name": r.name, "url": r.url, "up": r.up, "inflight": r.inflight,
                     "queued_seconds": round(r.queued_seconds(), 2) if r.up else None,
                     "prefixes": len(r.keys), "model": (r.state or {}).get("model"),
                     "pool": (r.state or {}).get("pool"), "store": r.store_only, "family": r.family, "kind": r.kind,
                     "routes": list(r.routes), "queue": (r.state or {}).get("queue"), "busy": (r.state or {}).get("busy"),
                     "rate_tok_s": {"prefill": r.prefill_tps, "decode": r.decode_tps}} for r in self.router.replicas]
            # "starting" until every backend has answered once (or was found down): a launcher that waits for "ok"
            # then starts with every card known
            pending = any(r.kind is None and not r.up and time.time() - self.router.started < self.router.down_after
                          for r in self.router.replicas)
            status = "starting" if pending else "ok" if any(r["up"] for r in reps) else "no replica up"
            return self._json(503 if status == "no replica up" else 200, {"status": status, "replicas": reps})
        if path == "/v1/models":
            fam = dict(self.router.families())
            for r in self.router.job_backends():
                fam.setdefault(r.family or r.name, []).append(r)
            if not fam:
                return self._json(503, {"error": "no replica up"})
            return self._json(200, {"object": "list", "data": [
                {"id": name, "object": "model", "owned_by": "local", "replicas": [r.name for r in reps],
                 "routes": sorted({rt for r in reps for rt in r.routes}) or list(POST_ROUTES)}
                for name, reps in sorted(fam.items())]})
        return self._job_get(self.path.split("?", 1)[0])

    # ---- job servers: status, results, panels ----------------------------------------------------------------------
    def _ask(self, r: Replica, method: str, path: str, body: bytes | None = None, timeout: float = 600.0):
        """One request to r, nothing sent to the client yet; -> (status, content type, body) or None (unreachable)."""
        try:
            c = r.conn(timeout=timeout)
            hdr = {"Content-Type": self.headers.get("Content-Type") or "application/json"} if body is not None else {}
            c.request(method, path, body, hdr)
            resp = c.getresponse()
            data = resp.read()
            c.close()
        except Exception as e:                      # noqa: BLE001
            print(f"[router] backend {r.name} unreachable: {e}", flush=True)
            return None
        return resp.status, resp.getheader("Content-Type", "application/json"), data

    def _relay(self, r: Replica, status: int, ctype: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-QV-Replica", r.name)
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self, r: Replica, method: str, path: str, body: bytes | None = None, timeout: float = 600.0):
        """Forward one request to r and relay the answer; -> (status, body) or None when r could not be reached."""
        ans = self._ask(r, method, path, body, timeout)
        if ans is None:
            self._json(502, {"error": f"backend {r.name} unreachable"})
            return None
        self._relay(r, *ans)
        return ans[0], ans[2]

    def _job_get(self, path: str):
        rt = self.router
        if path == "/v1/jobs":
            data = []
            for r in rt.job_backends():
                try:
                    c = r.conn(timeout=5)
                    c.request("GET", "/v1/jobs")
                    resp = c.getresponse()
                    raw = resp.read()
                    c.close()
                    if resp.status == 200:
                        data += [dict(j, replica=r.name) for j in json.loads(raw)["data"]]
                except Exception:                   # noqa: BLE001
                    continue
            return self._json(200, {"data": sorted(data, key=lambda j: j.get("created", 0), reverse=True)[:50]})
        if path.startswith("/ui/"):                 # /ui/<backend or model>/...: that server's own page
            name, _, rest = path[4:].partition("/")
            r = next((x for x in rt.replicas if x.up and name in (x.name, x.family)), None)
            if r is None:
                return self._json(404, {"error": f"no backend or model {name!r} is up"})
            return self._proxy(r, "GET", "/" + rest + (("?" + self.path.split("?", 1)[1]) if "?" in self.path else ""))
        if path.startswith("/v1/jobs/") and path.count("/") == 3:
            return self._async_job_get(path.rsplit("/", 1)[-1])
        if path.startswith("/v1/jobs/") or "." in path.rsplit("/", 1)[-1]:      # a job, or a result file named <id>.<ext>
            jid = path.rsplit("/", 1)[-1]
            jid = jid.rsplit(".", 1)[0] if not path.startswith("/v1/jobs/") else jid
            r = rt.job_owner(jid)
            if r is None:
                return self._json(404, {"error": "no such job"})
            return self._proxy(r, "GET", self.path)
        return self._json(404, {"error": "not found"})

    def do_DELETE(self):
        path = self.path.split("?", 1)[0]
        if not path.startswith("/v1/jobs/"):
            return self._json(404, {"error": "not found"})
        r, bid = self.router.job_target(path.rsplit("/", 1)[-1])
        if r is None:
            return self._json(404, {"error": "no such job"})
        return self._proxy(r, "DELETE", f"/v1/jobs/{bid}")

    def _job_post(self, path: str):
        rt = self.router
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0:
            return self._json(400, {"error": "bad Content-Length"})
        raw = self.rfile.read(n)
        if path.startswith("/v1/jobs/") and path.endswith("/cancel"):
            r, bid = rt.job_target(path[len("/v1/jobs/"):-len("/cancel")])
            if r is None:
                return self._json(404, {"error": "no such job"})
            return self._proxy(r, "POST", f"/v1/jobs/{bid}/cancel", raw)
        model = None
        if "json" in (self.headers.get("Content-Type") or "json"):
            try:
                model = json.loads(raw or b"{}").get("model")
            except (ValueError, AttributeError):
                model = None
        tried, out, last = [], None, None
        while True:
            with rt.pick_lock:                       # pick + count as one step: two jobs arriving together spread
                r, err = rt.job_pick(path, model, exclude=tuple(tried))
                if r is not None:
                    with r.lock:
                        r.inflight += 1
            if r is None:
                if last is not None:                 # every other backend is gone too: the failure the client gets
                    self._relay(*last)
                    return
                return self._json(err[0], {"error": err[1]})
            try:
                ans = self._ask(r, "POST", self.path, raw, timeout=3600.0)
            finally:
                with r.lock:
                    r.inflight -= 1
            # A backend that could not START or FINISH the job through no fault of the request (unreachable, or 503
            # "retryable": a block stage went away) -- nothing of an answer has reached the client, so the job goes to
            # the next backend of the route. Anything else (200, 4xx, a real 500) is the answer.
            retry = ans is None
            if ans is not None and ans[0] == 503:
                try:
                    retry = bool(json.loads(ans[2]).get("retryable"))
                except (ValueError, AttributeError):
                    retry = False
            if not retry:
                self._relay(r, *ans)
                out = (ans[0], ans[2])
                break
            tried.append(r)
            last = (r, *ans) if ans is not None else (r, 502, "application/json",
                                                      json.dumps({"error": f"backend {r.name} unreachable"}).encode())
            print(f"[router] {path}: {r.name} could not run the job"
                  f"{' (' + str(ans[0]) + ')' if ans is not None else ' (unreachable)'}, trying another backend", flush=True)
        if out is not None and out[0] in (200, 202):
            try:
                jid = json.loads(out[1]).get("id")
                if jid:
                    with rt.pick_lock:
                        rt.owner[str(jid)] = r
                        while len(rt.owner) > 4096:
                            rt.owner.pop(next(iter(rt.owner)))
                        if out[0] == 202:            # async: the client polls; a lost job is run again elsewhere
                            rt.remember_async(str(jid), path, model, raw,
                                              self.headers.get("Content-Type") or "application/json", r)
            except (ValueError, AttributeError):
                pass
            print(f"[router] {path} -> {r.name} ({r.family})", flush=True)

    def _async_job_get(self, jid: str):
        """GET /v1/jobs/<jid>. An async job submitted through this router is followed: when its backend is down, does not
        know it any more (restarted: 404) or reports a retryable failure, it is submitted again to another backend of its
        route (never one that already lost it) and the client sees it queued again, under the same id."""
        rt = self.router
        with rt.pick_lock:
            a = rt.jobs.get(jid)
        if a is None:
            r = rt.job_owner(jid)
            if r is None:
                return self._json(404, {"error": "no such job"})
            return self._proxy(r, "GET", self.path)
        while True:
            ans = self._ask(a.r, "GET", f"/v1/jobs/{a.bid}", None, timeout=10.0)
            why = None
            if ans is None:
                if a.r.up:                           # a blip while the poll still sees it: answer that, do not move yet
                    return self._json(502, {"error": f"backend {a.r.name} unreachable", "id": jid})
                why = "its backend is down"
            elif ans[0] == 404:
                why = "its backend lost it (restarted)"
            elif ans[0] == 200:
                try:
                    j = json.loads(ans[2])
                except ValueError:
                    j = None
                if isinstance(j, dict):
                    if j.get("status") == "error" and any(e in str(j.get("error") or "") for e in RETRYABLE_ERRORS):
                        why = f"it failed there ({str(j.get('error'))[:80]})"
                    else:
                        if j.get("status") in ("done", "error", "cancelled"):
                            a.raw = b""              # finished: the request body is not needed any more
                        j["id"] = jid
                        if a.moves:
                            j["moved"] = a.moves
                        return self._relay(a.r, 200, "application/json", json.dumps(j).encode())
            if why is None or not a.raw:
                return self._relay(a.r, *ans) if ans is not None else self._json(502, {"error": "backend unreachable"})
            if not self._resubmit(a, jid, why):
                if ans is not None:
                    return self._relay(a.r, *ans)
                return self._json(502, {"error": f"job {jid}: {why}, and no other backend could take it", "id": jid})

    def _resubmit(self, a, jid: str, why: str) -> bool:
        """Submit async job `a` to another backend of its route; True when one took it."""
        rt = self.router
        a.tried.append(a.r)
        while True:
            with rt.pick_lock:
                r, _err = rt.job_pick(a.route, a.model, exclude=tuple(a.tried))
                if r is not None:
                    with r.lock:
                        r.inflight += 1
            if r is None:
                print(f"[router] job {jid}: {why}; no other backend for {a.route}", flush=True)
                return False
            try:
                c = r.conn(timeout=60.0)
                c.request("POST", a.route, a.raw, {"Content-Type": a.ctype})
                resp = c.getresponse()
                data = resp.read()
                c.close()
                ans = (resp.status, data)
            except Exception:                        # noqa: BLE001
                ans = None
            finally:
                with r.lock:
                    r.inflight -= 1
            bid = None
            if ans is not None and ans[0] in (200, 202):
                try:
                    bid = json.loads(ans[1]).get("id")
                except (ValueError, AttributeError):
                    bid = None
            if not bid:
                a.tried.append(r)
                continue
            with rt.pick_lock:
                a.r, a.bid, a.moves = r, str(bid), a.moves + 1
                rt.owner[str(bid)] = r
            print(f"[router] job {jid}: {why} -> submitted again to {r.name} as {bid}", flush=True)
            return True

    def _forward_get(self, r: Replica, path: str) -> None:
        c = r.conn(timeout=10)
        c.request("GET", path)
        resp = c.getresponse()
        b = resp.read()
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(b)))
        self.send_header("X-QV-Replica", r.name)
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in POST_ROUTES:
            return self._job_post(path)              # a job server's route, or 404 with the routes served here
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw or b"{}")
        except ValueError as e:
            return self._json(400, {"error": f"bad JSON: {e}"})
        model, group = self.router.group_for(body)
        if model is None:
            if not group:
                return self._json(503, {"error": "no replica up"})
            return self._json(404, {"error": f"model {body.get('model')!r} is not served here; \"model\" must be one of "
                                             f"{group} (or start the router with --default-model)"})
        deadline = time.time() + self.router.queue_timeout
        # the prompt's keys ONCE, outside the lock: it is a tokenization of the whole prompt on a replica (seconds for a
        # long one); under pick_lock and repeated every 0.2 s of waiting it stalled every other request's routing
        nk = self.router.keys_for(body, [r for r in group if r.up])
        why, res = {}, None
        while True:                          # every replica full: wait in the router (not in a replica that would OOM)
            with self.router.pick_lock:      # rank + reserve atomically: a concurrent request sees this choice
                ranked = self.router.ranked(body, group, nk, why)
                if ranked:
                    res = _reserve(ranked[0], +1)
            if ranked or why.get("too_long") or not any(r.up for r in group) or time.time() > deadline:
                break
            time.sleep(0.2)
        if not ranked and why.get("too_long"):
            return self._json(413, {"error": f"the prompt ({nk[0]} tokens) plus max_tokens exceeds the context window of "
                                             f"every replica of {model}"})
        if not ranked:
            busy = any(r.up for r in group)
            return self._json(429 if busy else 503, {"error": "every replica is busy, try again" if busy
                                                     else "no replica up (or none can admit this prompt)"})
        t0 = time.time()
        self._busy_retry = 0                 # handler objects serve one connection: reset per request
        held = ranked[0]                     # the reservation stays until the response ends (or moves on a retry)
        try:
            for cand in ranked:
                if cand is not held and cand[0].max_inflight and cand[0].inflight >= cand[0].max_inflight:
                    continue                 # full now: do not push a retry into it
                if cand is not held:
                    _reserve(held, -1, res)
                    res = _reserve(cand, +1)
                    held = cand
                r, t, cached = cand[0], cand[1], cand[2]
                if cand[5] is not None:
                    self._pull(r, cand[5])
                res["fwd"] = time.time()
                sent = self._forward_post(r, path, raw)
                if sent is None:             # failed (or refused: full) before any byte reached the client: next replica
                    print(f"[router] {r.name} {'is full' if getattr(self, '_busy_retry', 0) else 'failed before responding'}"
                          f"; trying the next replica", flush=True)
                    continue
                print(f"[router] {path} -> {r.name} (predicted {t:.1f}s, cached {cached} tok, "
                      f"{len(ranked)} candidates) done in {time.time() - t0:.1f}s", flush=True)
                return
            try:
                busy = getattr(self, "_busy_retry", 0)
                if busy:                     # every replica refused for lack of room: the client should come back
                    b = json.dumps({"error": "every replica is busy, try again", "retry_after": busy}).encode()
                    self.send_response(429)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(b)))
                    self.send_header("Retry-After", str(busy))
                    self.end_headers()
                    self.wfile.write(b)
                else:
                    self._json(502, {"error": "every replica failed"})
            except OSError:
                pass
        finally:
            _reserve(held, -1, res)

    def _pull(self, r: Replica, plan: tuple) -> None:
        """Have r copy the prefix from the donor's machine before the request goes in. A failure only costs the
        prefill it would have saved."""
        s, hexkey, c_s, t_x = plan
        t0 = time.time()
        try:
            c = r.conn(timeout=max(120.0, 4 * t_x))
            hdr = {"Content-Type": "application/json"}
            if os.environ.get("QV_POOL_TOKEN"):
                hdr["X-QV-Pool-Token"] = os.environ["QV_POOL_TOKEN"]
            c.request("POST", "/v1/pool/pull", json.dumps({"from": s.url, "key": hexkey}), hdr)
            resp = c.getresponse()
            res = json.loads(resp.read() or b"{}")
            c.close()
            if resp.status != 200:
                raise RuntimeError(res.get("error") or f"HTTP {resp.status}")
            self.router.learn_pull(s, r, res)
            print(f"[router] pulled {c_s} tokens {s.name} -> {r.name}: {res.get('bytes', 0) / 2**20:.0f} MB in "
                  f"{time.time() - t0:.1f}s (predicted {t_x:.1f}s)", flush=True)
        except Exception as e:                      # noqa: BLE001 - r prefills instead
            print(f"[router] pull {s.name} -> {r.name} failed ({e}); {r.name} prefills", flush=True)

    def _forward_post(self, r: Replica, path: str, raw: bytes):
        """Forward one request and stream the answer back. None = failed before anything was sent to the client
        (the caller may retry elsewhere); True = forwarded (complete, or cut off after bytes were sent)."""
        try:
            c = r.conn(timeout=3600)
            c.request("POST", path, raw, {"Content-Type": "application/json", "Content-Length": str(len(raw))})
            resp = c.getresponse()
        except Exception:                           # noqa: BLE001 - nothing sent yet
            return None
        if resp.status == 429 or (resp.status == 503 and resp.getheader("Retry-After")):
            # the replica has no room now (its admission control): nothing reached the client, try the next one
            self._busy_retry = max(getattr(self, "_busy_retry", 0), int(resp.getheader("Retry-After") or 5))
            resp.read()
            c.close()
            return None
        if resp.status >= 500:
            resp.read()
            c.close()
            return None
        self.send_response(resp.status)
        for h in ("Content-Type", "Cache-Control"):
            v = resp.getheader(h)
            if v:
                self.send_header(h, v)
        self.send_header("X-QV-Replica", r.name)
        n = resp.getheader("Content-Length")
        if n is not None:
            self.send_header("Content-Length", n)
            self.end_headers()
            self.wfile.write(resp.read())
        else:                                       # streamed (SSE): pass chunks through as they come
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except OSError:                         # client or replica gone mid-stream: end it, no retry
                pass
        c.close()
        return True


def _named(items, what):
    out = {}
    for it in items or ():
        name, sep, val = it.partition("=")
        if not sep or not name:
            raise SystemExit(f"--{what} NAME=VALUE, got {it!r}")
        out[name] = val
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ai_engine.servers.router", description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", action="append", required=True, metavar="NAME=URL", help="a model server of any kind: LLM or job server (repeat)")
    ap.add_argument("--rate", action="append", metavar="NAME=PREFILL/DECODE",
                    help="measured tok/s of a replica (default 300/12)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between state polls")
    ap.add_argument("--down-after", type=float, default=5.0, help="seconds without a state answer -> down")
    ap.add_argument("--default-max-tokens", type=int, default=1024, help="decode budget when a request gives none")
    ap.add_argument("--max-inflight", action="append", metavar="NAME=N",
                    help="at most N requests at a time on that replica (one-card V100: 1); others wait in the router")
    ap.add_argument("--queue-timeout", type=float, default=600.0, help="seconds a request may wait for a free replica")
    ap.add_argument("--store", action="append", metavar="NAME=URL",
                    help="a prefix store (DRAM-only machine, servers/prefix_store.py): pull donor, never gets requests")
    ap.add_argument("--panel", default=None, metavar="NAME",
                    help="serve this backend's (or model's) own page at / ; every server's page is at /ui/NAME/")
    ap.add_argument("--default-model", default=None, metavar="NAME",
                    help="the model for requests whose \"model\" names none of several served models (default: 404)")
    ap.add_argument("--pull", choices=("cost", "always"), default="cost",
                    help="cost: pull a prefix from another machine only when the transfer is predicted faster than the "
                         "prefill it saves (production); always: whenever a longer prefix exists (tests the transfer path)")
    ap.add_argument("--net-mb-s", type=float, default=3.0,
                    help="link speed between machines to start with, MB/s (learned from pulls)")
    a = ap.parse_args(argv)
    backends, rates = _named(a.backend, "backend"), _named(a.rate, "rate")
    caps = _named(a.max_inflight, "max-inflight")
    for n in list(rates) + list(caps):
        if n not in backends:
            raise SystemExit(f"--rate / --max-inflight {n}: no --backend of that name")
    reps = []
    for name, url in backends.items():
        pre, _, dec = rates.get(name, "300/12").partition("/")
        reps.append(Replica(name, url, float(pre), float(dec or 12), int(caps.get(name, 0))))
    for name, url in _named(a.store, "store").items():
        st = Replica(name, url, 1.0, 1.0)
        st.store_only = True
        reps.append(st)
    router = Router(reps, a.poll, a.down_after, a.default_max_tokens, queue_timeout=a.queue_timeout,
                    net_mb_s=a.net_mb_s, pull=a.pull)
    router.start()
    router.default_model, router.panel = a.default_model, a.panel
    Handler.router = router
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    from qv_band.common import tls as _tls
    _tls.wrap_server(srv)                 # TLS when the deployment sets QV_TLS_* (qv_band/common/tls.py)
    print(f"[router] {len(reps)} replicas: " + ", ".join(f"{r.name}={r.url} ({r.prefill_tps:g}/{r.decode_tps:g} tok/s)"
                                                         for r in reps) + f"; listening on {a.host}:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
