"""A DRAM-only prefix store: a machine without a GPU that keeps copies of the saved prompt prefixes of other servers
(qv_band/design_doc/replica_pool.md). Pure Python standard library (runs on the Windows 16 GB PC: no torch, no numpy).

    python prefix_store.py --mirror http://host-b:8080 --budget-gb 6 --port 8082

ONE store serves every model: an entry belongs to its source's transfer signature (the model's pack, taus, checkpoint grid
and drafter), so servers of different models mirror into the same process and share its one RAM budget.

MIRROR. Every --poll seconds it reads each source's GET /v1/pool/state and copies every prefix it does not hold yet
(GET /v1/pool/export, then /v1/pool/piece in PIECE-byte pieces over --streams connections, each checked against the
sender's blake2b; POST /v1/pool/release). Blobs are keyed by origin, so a later checkpoint of the same prompt only
fetches what is new. The source may then evict its own copy: this store still has it.

SERVE. It answers the same endpoints as an LLM server's pool (servers/pool_xfer.py): GET /v1/pool/state (its prefixes,
its own pool id and the sources' transfer signature), /v1/pool/export, /v1/pool/piece, POST /v1/pool/release. The
router lists it with --store NAME=URL: a pull donor that never gets a request.

AUTH. --token (or QV_POOL_TOKEN): sent to the sources and required from clients (X-QV-Pool-Token), the same token as
the servers and the router. Without it the store is open to the LAN (it holds prompts' token ids).

MEMORY. At most --budget-gb of blob bytes (copies in flight included), least recently used prefix evicted first (a
prefix being sent is kept). A prefix this store evicted, or that does not fit the budget at all, is NOT copied again
while its source still lists it (before 2026-11-14 sources holding more than the budget made the store evict and
re-copy the same prefixes for ever). Nothing is written to disk.

SOURCES. One thread per --mirror source: a slow link or a powered-off machine delays only its own copies.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import http.client
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

PIECE = 8 << 20
LEASE_S = 300.0                # after the last piece asked
TOKEN = ""                     # --token / QV_POOL_TOKEN


def _hdr() -> dict:
    return {"X-QV-Pool-Token": TOKEN} if TOKEN else {}


class Store:
    def __init__(self, budget: int, streams: int):
        self.budget, self.streams = budget, streams
        self.pool = uuid.uuid4().hex
        self.entries = {}                      # (transfer signature, hex key) -> {"L", "metas": [meta, ...], "used", "origins"}
        self.blobs = {}                        # origin -> bytearray
        self.leases = {}                       # lease -> (hex key, index of the prefix under it, expiry)
        self.pending = 0                       # bytes of copies in flight (allocated, not in self.blobs yet)
        self.skip = {}                         # (signature, hex key, n) -> why: evicted here / too large; not copied again
        self.listed = {}                       # source -> {(signature, hex key, n)} it listed at its last poll
        self.lock = threading.RLock()

    # ---- bookkeeping ---------------------------------------------------------------------------------------------
    def used(self) -> int:
        return sum(len(b) for b in self.blobs.values()) + self.pending

    def _in_use(self) -> set:
        now = time.time()
        for k in [k for k, (_, _, exp) in self.leases.items() if exp < now]:
            del self.leases[k]
        return {key for key, _, _ in self.leases.values()}

    def _drop_unused_blobs(self) -> None:
        live = {o for e in self.entries.values() for o in e["origins"]}
        for o in [o for o in self.blobs if o not in live]:
            del self.blobs[o]

    def _make_room(self, need: int) -> bool:
        """Evict least recently used prefixes (not one being sent) until `need` more bytes fit."""
        while self.used() + need > self.budget:
            busy = self._in_use()
            victims = sorted((e["used"], k) for k, e in self.entries.items() if k not in busy)
            if not victims:
                return False
            k = victims[0][1]
            print(f"[store] evicted the prefixes named {k[1][:12]} ({self.entries[k]['L']} tokens, {k[0]}, LRU)", flush=True)
            for m in self.entries[k]["metas"]:
                self.skip[(k[0], k[1], int(m["n"]))] = "evicted"
            del self.entries[k]
            self._drop_unused_blobs()
        return True

    # ---- mirror --------------------------------------------------------------------------------------------------
    def copy_from(self, base: str, sig: str, hexkey: str) -> None:
        """Copy the saved prefixes `base` exports under `hexkey` (a grid checkpoint and/or a turn's prompt end); `sig`
        is that server's transfer signature."""
        resp = _get_json(base, f"/v1/pool/export?key={hexkey}&xfer={sig}")
        metas = resp["entries"]
        ek = (sig, hexkey)
        try:
            with self.lock:
                if resp.get("xfer") != sig:
                    raise RuntimeError(f"{base} changed its transfer signature ({resp.get('xfer')} != {sig})")
                held = {int(m["n"]) for m in self.entries.get(ek, {}).get("metas", [])}
                new = [m for m in metas if int(m["n"]) not in held]
                todo, seen = [], set()
                for m in new:
                    for b in m["blobs"]:
                        if b["origin"] not in self.blobs and b["origin"] not in seen:
                            seen.add(b["origin"])
                            todo.append((m, b))
                need = sum(int(b["size"]) for _, b in todo)
                if new and (need > self.budget or not self._make_room(need)):
                    for m in new:              # do not ask for it again at every poll
                        self.skip[(sig, hexkey, int(m["n"]))] = "no room"
                    raise MemoryError(f"no room for {need >> 20} MB (budget {self.budget >> 20} MB); not asked for again")
                bufs = {b["origin"]: bytearray(int(b["size"])) for _, b in todo} if new else {}
                pend = sum(len(b) for b in bufs.values())
                self.pending += pend
            if not new:
                return
            t0 = time.time()
            local = threading.local()

            def fetch(job):
                m, b, off = job
                length = min(PIECE, int(b["size"]) - off)
                c = getattr(local, "c", None) or _conn(base)
                local.c = c
                c.request("GET", f"/v1/pool/piece?lease={m['lease']}&bid={b['bid']}&off={off}&len={length}",
                          headers=_hdr())
                r = c.getresponse()
                if r.status != 200:
                    raise RuntimeError(f"piece: HTTP {r.status} {r.read()[:200]!r}")
                view = memoryview(bufs[b["origin"]])[off:off + length]
                got = 0
                while got < length:
                    k = r.readinto(view[got:])
                    if not k:
                        raise RuntimeError("connection closed mid-piece")
                    got += k
                r.read()
                if hashlib.blake2b(view, digest_size=16).hexdigest() != r.getheader("X-QV-Hash"):
                    raise RuntimeError("piece hash mismatch")

            jobs = [(m, b, off) for m, b in todo for off in range(0, int(b["size"]), PIECE)]
            stop = threading.Event()

            def keep():                             # a waiting prefix's lease must not expire on a slow link
                while not stop.wait(60.0):
                    for m in metas:
                        try:
                            c = _conn(base)
                            c.request("GET", f"/v1/pool/piece?lease={m['lease']}&bid={m['blobs'][0]['bid']}&off=0&len=0",
                                      headers=_hdr())
                            c.getresponse().read()
                            c.close()
                        except Exception:           # noqa: BLE001
                            pass

            threading.Thread(target=keep, daemon=True).start()
            try:
                with ThreadPoolExecutor(max_workers=self.streams) as ex:
                    list(ex.map(fetch, jobs))
            finally:
                stop.set()
            with self.lock:
                self.pending -= pend
                pend = 0
                self.blobs.update(bufs)
                e = self.entries.setdefault(ek, {"L": int(new[0]["n"]) // 1024 * 1024, "metas": [],
                                                     "used": time.time(), "origins": set()})
                for m in new:
                    e["metas"].append({k: v for k, v in m.items() if k != "lease"})
                    e["origins"].update(b["origin"] for b in m["blobs"])
                e["used"] = time.time()
            dt = time.time() - t0
            print(f"[store] copied {len(new)} prefix(es) named {hexkey[:12]} (up to {max(int(m['n']) for m in new)} "
                  f"tokens) from {base}: {need / 2**20:.0f} MB in {dt:.1f}s ({need / 2**20 / max(dt, 1e-6):.1f} MB/s); "
                  f"holding {sum(len(x['metas']) for x in self.entries.values())} prefixes, "
                  f"{self.used() / 2**30:.2f} GiB", flush=True)
        finally:
            if locals().get("pend"):           # a failed copy: its buffers are gone
                with self.lock:
                    self.pending -= pend
            for m in metas:
                try:
                    c = _conn(base)
                    c.request("POST", "/v1/pool/release", json.dumps({"lease": m["lease"]}),
                              {"Content-Type": "application/json", **_hdr()})
                    c.getresponse().read()
                    c.close()
                except Exception:              # noqa: BLE001 - the source's lease expires anyway
                    pass

    def mirror_once(self, base: str) -> None:
        """One look at one source: copy what it holds and this store neither holds nor decided to skip."""
        st = _get_json(base, "/v1/pool/state", timeout=10.0)
        if not st.get("pool") or not st.get("xfer") or st.get("local_only"):
            return                             # no shared pool there, or a split server (partial entries)
        sig = st["xfer"]
        held = [(str(h), int(n)) for h, n in st.get("held", [])]
        with self.lock:                        # a skipped prefix no source lists any more: forget it
            self.listed[base] = {(sig, h, n) for h, n in held}
            anywhere = set().union(*self.listed.values())
            for k in [k for k in self.skip if k not in anywhere]:
                del self.skip[k]
        for hexkey, n in sorted(held, key=lambda x: x[1]):          # shorter first: re-use
            with self.lock:
                have = n in {int(m["n"]) for m in self.entries.get((sig, hexkey), {}).get("metas", [])}
                skipped = (sig, hexkey, n) in self.skip
            if not have and not skipped:
                self.copy_from(base, sig, hexkey)

    def mirror_loop(self, base: str, poll: float) -> None:
        last = None
        while True:
            try:
                self.mirror_once(base)
                last = None
            except Exception as e:             # noqa: BLE001 - a source down or a failed copy: try again later
                msg = f"{type(e).__name__}: {e}"
                if msg != last:                # once per change, not every poll
                    print(f"[store] {base}: {msg}", flush=True)
                    last = msg
            time.sleep(poll)


def _conn(base: str, timeout: float = 120.0) -> http.client.HTTPConnection:
    u = urlsplit(base)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def _get_json(base: str, path: str, timeout: float = 120.0):
    c = _conn(base, timeout)
    try:
        c.request("GET", path, headers=_hdr())
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"{path.split('?')[0]}: HTTP {r.status} {data[:200]!r}")
        return json.loads(data)
    finally:
        c.close()


class Handler(BaseHTTPRequestHandler):
    store: Store = None
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

    def _authorized(self) -> bool:
        if not TOKEN or hmac.compare_digest(self.headers.get("X-QV-Pool-Token", ""), TOKEN):
            return True
        self._json(403, {"error": "X-QV-Pool-Token missing or wrong"})
        return False

    def do_GET(self):
        st = self.store
        u = urlsplit(self.path)
        path = u.path.rstrip("/")
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if path in ("/v1/pool/export", "/v1/pool/piece") and not self._authorized():
            return
        if path in ("/v1/pool/state", "/health", ""):
            with st.lock:
                by = {}
                for (sig, k), e in st.entries.items():
                    v = by.setdefault(sig, {"prefixes": [], "held": []})
                    v["prefixes"].append([e["L"], k])
                    v["held"] += [[k, int(m["n"])] for m in e["metas"]]
                if q.get("xfer"):                # one model's view, in a server's own state format (--prefix-backup)
                    v = by.get(q["xfer"], {"prefixes": [], "held": []})
                    return self._json(200, {"pool": st.pool, "xfer": q["xfer"], "store": True, **v})
                return self._json(200, {"pool": st.pool, "store": True, "model": "prefix-store", "by_xfer": by,
                                        "load": None, "bytes": st.used(), "budget": st.budget})
        if path == "/v1/pool/export":
            with st.lock:
                ek = (q.get("xfer", ""), q.get("key", ""))
                e = st.entries.get(ek)
                if e is None:
                    return self._json(404, {"error": "no such prefix here (needs key and xfer: the asking server's transfer "
                                                     "signature)"})
                e["used"] = time.time()
                metas = []
                for i, m in enumerate(e["metas"]):     # one lease per prefix: blob ids are per source machine
                    lease = f"store:{uuid.uuid4().hex}"
                    st.leases[lease] = (ek, i, time.time() + LEASE_S)
                    metas.append(dict(m, lease=lease, pool=st.pool))
            return self._json(200, {"entries": metas, "xfer": ek[0]})
        if path == "/v1/pool/piece":
            with st.lock:
                lk = st.leases.get(q.get("lease", ""))
                e = st.entries.get(lk[0]) if lk else None
                blob = None
                if e is not None and lk[1] < len(e["metas"]):
                    st.leases[q["lease"]] = (lk[0], lk[1], time.time() + LEASE_S)   # alive while pieces keep coming
                    for b in e["metas"][lk[1]]["blobs"]:
                        if str(b["bid"]) == q.get("bid"):
                            blob = st.blobs.get(b["origin"])
            if blob is None:
                return self._json(404, {"error": "no such lease / blob"})
            off, length = int(q["off"]), int(q["len"])
            if off < 0 or length < 0 or off + length > len(blob):
                return self._json(400, {"error": "range outside the blob"})
            mv = memoryview(blob)[off:off + length]
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("X-QV-Hash", hashlib.blake2b(mv, digest_size=16).hexdigest())
            self.end_headers()
            self.wfile.write(mv)
            return
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if not self._authorized():
            return
        if self.path.rstrip("/") == "/v1/pool/release":
            try:
                lease = json.loads(raw or b"{}").get("lease", "")
            except ValueError:
                lease = ""
            with self.store.lock:
                self.store.leases.pop(lease, None)
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "not found"})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mirror", action="append", default=[], metavar="URL",
                    help="an LLM server whose saved prefixes to copy (repeat; one per machine is enough)")
    ap.add_argument("--budget-gb", type=float, default=6.0, help="RAM for prefixes, GiB")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between looks at the sources")
    ap.add_argument("--streams", type=int, default=4, help="connections per copy")
    ap.add_argument("--token", default=os.environ.get("QV_POOL_TOKEN", ""),
                    help="shared pool token (X-QV-Pool-Token), the same as the servers' QV_POOL_TOKEN")
    a = ap.parse_args(argv)
    global TOKEN
    TOKEN = a.token
    if not TOKEN:
        print("[store] WARNING: no --token: this store is open to the LAN (it holds prompts' token ids)", flush=True)
    store = Store(int(a.budget_gb * 2**30), a.streams)
    Handler.store = store
    for base in a.mirror:                      # one thread per source
        threading.Thread(target=store.mirror_loop, args=(base, a.poll), daemon=True, name=f"mirror-{base}").start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[store] prefix store on {a.host}:{a.port}, budget {a.budget_gb:g} GiB, mirroring {a.mirror or 'nothing'}",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
