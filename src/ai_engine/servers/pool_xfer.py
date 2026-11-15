"""Moving a saved prefix between machines (qv_band/design_doc/replica_pool.md; --prefix-store shared on both sides).

A COPY, not a move: a saved prefix never changes, so both machines keep it and each evicts by its own LRU.
Pull model, all over the servers' own HTTP port:

    receiver  POST /v1/pool/pull {"from": "http://sender:port", "key": "<prefix key hex>"}
      -> sender GET /v1/pool/export?key=...      lease the entry (XFER_LEASE_S), describe it: layout (JSON) + blobs
      -> sender GET /v1/pool/piece?lease&bid&off&len   PIECE bytes each, several connections at once, written
                                                 straight into the receiver's pool; X-QV-Hash (blake2b) checked
      -> sender POST /v1/pool/release {"lease"}
    the receiver publishes the entry only when every piece arrived and matched; any failure -> nothing published, the
    request is simply prefilled there (the router decides whether a pull is worth it).

AUTH. With QV_POOL_TOKEN set (the same on every server, the router and the prefix store), export / piece / release /
pull need the header X-QV-Pool-Token: a saved prefix holds a prompt's token ids, and pull makes this server fetch a URL.
Without it these endpoints are open to the LAN, like the completion endpoints (the server warns at start).

Both sides must have the same transfer signature (QV_PREFIX_XFER_SIG: packs by model + format family + tau, state tau,
KV tau, KV residual, checkpoint grid, dtype, drafter): checked before a byte moves.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
from urllib.parse import parse_qs, urlsplit

from ai_engine.servers.pool_keys import floor_key

STREAMS = int(os.environ.get("QV_PREFIX_XFER_STREAMS", "4"))
_KEY_CACHE: dict = {}          # (pool id, entry key) -> (whole-block length, its key hex, entry tokens)
_KEY_LOCK = __import__("threading").Lock()   # state polls, pulls and the backup watch run on different threads
TOKEN_HEADER = "X-QV-Pool-Token"


def _token() -> str:
    return os.environ.get("QV_POOL_TOKEN", "")


def _auth_headers() -> dict:
    return {TOKEN_HEADER: _token()} if _token() else {}


def _authorized(handler) -> bool:
    import hmac
    t = _token()
    if not t or hmac.compare_digest(handler.headers.get(TOKEN_HEADER, ""), t):
        return True
    handler._send(403, {"error": f"{TOKEN_HEADER} missing or wrong"})
    return False


def _store():
    from qv_band.decode.shared_prefix import store_from_env
    return store_from_env()


def signature() -> str:
    return os.environ.get("QV_PREFIX_XFER_SIG", "")


def state_extra() -> dict:
    """Fields for GET /v1/pool/state: which pool (one per machine) and the transfer signature."""
    st = _store()
    if st is None:
        return {}
    out = {"pool": st.uuid(), "xfer": signature()}
    if os.environ.get("QV_PREFIX_LOCAL_ONLY"):
        out["local_only"] = True       # a split model's server: partial entries, no export and no pull (the router skips it)
    return out


def keyed_entries(st) -> list:
    """[(entry key, whole-block length L, key hex, n tokens)] of this pool. A saved prefix of n tokens is named by its
    last whole 1024-token block (L = n rounded down), so several entries can share one name: a grid checkpoint at L and
    the prompt-end state of a turn a little past it."""
    pid, out, live = st.uuid(), [], set()
    entries = st.entries()
    with _KEY_LOCK:                # the cache was iterated here while another thread inserted (RuntimeError: dictionary
        for e in entries:          # changed size during iteration -> a failed state poll -> the replica marked down)
            k = _KEY_CACHE.get((pid, e["key"]))
            if k is None:
                fk = floor_key(e["tokens"])
                k = _KEY_CACHE[(pid, e["key"])] = (fk[0], fk[1], len(e["tokens"])) if fk else (0, None, len(e["tokens"]))
            live.add((pid, e["key"]))
            if k[1] is not None:
                out.append((e["key"],) + k)
        for ck in [ck for ck in _KEY_CACHE if ck[0] == pid and ck not in live]:     # evicted entries: forget their keys
            del _KEY_CACHE[ck]
    return out


def _entries_for(hexkey: str) -> list:
    """The entries to send for `hexkey`: the one ending exactly on its block (a grid checkpoint) and the longest (a
    turn's prompt end; the receiver restores whichever its prompt matches). Their KV segments are shared blobs, so the
    second costs little more than its state."""
    hits = sorted((n, k, L) for k, L, h, n in keyed_entries(_store()) if h == hexkey)
    if not hits:
        return []
    pick = [k for n, k, L in hits if n == L][:1]
    if hits[-1][1] not in pick:
        pick.append(hits[-1][1])
    return pick


# ---- sender ---------------------------------------------------------------------------------------------------------
def handle_get(handler, path: str) -> bool:
    """GET /v1/pool/export and /v1/pool/piece; True if handled."""
    u = urlsplit(path)
    if u.path.rstrip("/") not in ("/v1/pool/export", "/v1/pool/piece"):
        return False
    if not _authorized(handler):
        return True
    st = _store()
    if st is None:
        handler._send(404, {"error": "no shared prefix pool here (--prefix-store shared)"})
        return True
    if os.environ.get("QV_PREFIX_LOCAL_ONLY"):
        handler._send(404, {"error": "a split model's server: its prefixes are kept per machine (this one holds only "
                                     "its own layers) and are not exported"})
        return True
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    if u.path.rstrip("/") == "/v1/pool/export":
        if q.get("xfer") and q["xfer"] != signature():
            handler._send(404, {"error": f"this server's transfer signature is {signature()}, not {q['xfer']}"})
            return True
        if q.get("layers"):                          # one layer slice (stages of a chain)
            a, b = (int(x) for x in q["layers"].split(":"))
            keys = _entries_for(q.get("key", ""))
            try:
                m = st.export_layers(keys[-1], a, b, q.get("drafter") == "1") if keys else None
            except ValueError as e:
                handler._send(400, {"error": str(e)})
                return True
            if m is None:
                handler._send(404, {"error": "no such prefix here"})
                return True
            handler._send(200, {"entries": [m], "xfer": signature()})
            return True
        metas = [m for m in (st.export(k) for k in _entries_for(q.get("key", ""))) if m is not None]
        if not metas:
            handler._send(404, {"error": "no such prefix here"})
            return True
        handler._send(200, {"entries": metas, "xfer": signature()})
        return True
    try:
        mv = st.piece(q["lease"], int(q["bid"]), int(q["off"]), int(q["len"]))
    except (KeyError, ValueError) as e:
        handler._send(404, {"error": f"piece: {e}"})
        return True
    handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(len(mv)))
    handler.send_header("X-QV-Hash", hashlib.blake2b(mv, digest_size=16).hexdigest())
    handler.end_headers()
    handler.wfile.write(mv)
    return True


def handle_post(handler, route: str, body: dict) -> bool:
    """POST /v1/pool/release and /v1/pool/pull; True if handled."""
    if route not in ("/v1/pool/release", "/v1/pool/pull"):
        return False
    if not _authorized(handler):
        return True
    st = _store()
    if st is None:
        handler._send(404, {"error": "no shared prefix pool here (--prefix-store shared)"})
        return True
    if route == "/v1/pool/release":
        st.release_lease(str(body.get("lease", "")))
        handler._send(200, {"ok": True})
        return True
    if body.get("slices"):
        try:
            res = pull_slices(body["slices"], int(body.get("streams", STREAMS)))
        except Exception as e:                   # noqa: BLE001 - reported to the caller, who prefills instead
            print(f"[prefix] slice pull failed: {e!r}", flush=True)
            handler._send(502, {"error": f"slice pull failed: {e}"})
            return True
        handler._send(200, res)
        return True
    src = str(body.get("from", ""))
    if urlsplit(src).scheme != "http" or not urlsplit(src).hostname:
        handler._send(400, {"error": "from: an http://host:port URL"})
        return True
    try:
        res = pull(src, str(body["key"]), int(body.get("streams", STREAMS)))
    except Exception as e:                       # noqa: BLE001 - reported to the caller, who prefills instead
        print(f"[prefix] pull failed: {e!r}", flush=True)
        handler._send(502, {"error": f"pull failed: {e}"})
        return True
    handler._send(200, res)
    return True


# ---- receiver -------------------------------------------------------------------------------------------------------
class _KeepLeases:
    """While a transfer with several parts runs, touch every lease once a minute (a zero-length piece): parts wait their
    turn, and over a slow link a waiting part's lease would expire before its first piece (2026-11-13: two layer slices
    over Wi-Fi, the second slice's lease expired while the first one's 250 MB arrived)."""

    def __init__(self, items: list, every: float = 60.0):
        import threading
        self.items, self.every, self.stop = items, every, threading.Event()   # [(base, lease, bid)]
        self.t = threading.Thread(target=self._run, daemon=True, name="keep-leases")

    def _run(self):
        while not self.stop.wait(self.every):
            for base, lease, bid in self.items:
                try:
                    c = _conn(base)
                    c.request("GET", f"/v1/pool/piece?lease={lease}&bid={bid}&off=0&len=0", headers=_auth_headers())
                    c.getresponse().read()
                    c.close()
                except Exception:                    # noqa: BLE001 - the transfer itself reports a real failure
                    pass

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()

def _conn(base: str):
    u = urlsplit(base)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=120)


def _get_json(base: str, path: str):
    c = _conn(base)
    try:
        c.request("GET", path, headers=_auth_headers())
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"{path.split('?')[0]}: HTTP {r.status} {data[:200]!r}")
        return json.loads(data)
    finally:
        c.close()


def pull(base: str, hexkey: str, streams: int = STREAMS) -> dict:
    """Copy the saved prefixes named `hexkey` from the server or store at `base` into this machine's pool."""
    st = _store()
    if os.environ.get("QV_PREFIX_LOCAL_ONLY"):
        raise RuntimeError("a split model's server keeps only its own layers: a whole-model prefix cannot be pulled into it")
    resp = _get_json(base, f"/v1/pool/export?key={hexkey}&xfer={signature()}")   # a store holds several models: name ours
    metas = resp["entries"]
    total = {"bytes": 0, "seconds": 0.0, "reused": 0, "entries": 0, "n": 0}
    t0 = time.time()
    try:
        if resp.get("xfer") != signature():
            raise RuntimeError(f"transfer signature differs (sender {resp.get('xfer')!r}, here {signature()!r}): "
                               f"packs, taus, checkpoint grid and drafter must match")
        have = {(h, n) for _, _, h, n in keyed_entries(st)}
        keep = _KeepLeases([(base, m["lease"], m["blobs"][0]["bid"]) for m in metas if m["blobs"]]).__enter__()
        for meta in metas:
            if meta["pool"] == st.uuid() or (hexkey, int(meta["n"])) in have:
                continue                                 # this machine's own pool, or already here
            res = _pull_one(base, meta, streams)
            for k in ("bytes", "reused"):
                total[k] += res[k]
            total["entries"] += int(bool(res.get("entry")))
            total["n"] = max(total["n"], int(meta["n"]))
        total["seconds"] = time.time() - t0
        total["mb_s"] = round(total["bytes"] / 2**20 / max(total["seconds"], 1e-6), 1)
        if not total["entries"]:
            total["already"] = True
        else:
            print(f"[prefix] pulled {total['entries']} prefix(es) up to {total['n']} tokens from {base}: "
                  f"{total['bytes'] / 2**20:.0f} MB in {total['seconds']:.1f}s ({total['mb_s']} MB/s, "
                  f"{total['reused']} blobs already here)", flush=True)
        return total
    finally:
        if "keep" in locals():
            keep.__exit__()
        for meta in metas:
            try:
                c = _conn(base)
                c.request("POST", "/v1/pool/release", body=json.dumps({"lease": meta["lease"]}),
                          headers={"Content-Type": "application/json", **_auth_headers()})
                c.getresponse().read()
                c.close()
            except Exception:                        # noqa: BLE001 - the sender's lease expires anyway
                pass


def pull_slices(slices: list, streams: int = STREAMS) -> dict:
    """One entry here from layer slices on one or several machines: [{"from": url, "key": hex, "layers": [a, b],
    "drafter": bool}, ...] covering this server's layers (a whole-model replica pulling from the stages of a chain, or
    a stage pulling its slice of a whole-model prefix)."""
    st = _store()
    metas, bases = [], {}
    try:
        for sl in slices:
            src = str(sl["from"])
            if urlsplit(src).scheme != "http" or not urlsplit(src).hostname:
                raise ValueError("from: an http://host:port URL")
            a, b = (int(x) for x in sl["layers"])
            resp = _get_json(src, f"/v1/pool/export?key={sl['key']}&layers={a}:{b}&drafter={int(bool(sl.get('drafter')))}")
            if resp.get("xfer") != signature():
                raise RuntimeError(f"transfer signature differs at {src}")
            m = resp["entries"][0]
            metas.append(m)
            bases[m["lease"]] = src
        # this server runs every layer: an entry of only some layers could never be restored here and would shadow a
        # complete one (stage workers, which hold a layer range, do not have a pool yet)
        got = sorted(tuple(m["layers"]) for m in metas)
        if got[0][0] != 0 or got[-1][1] != int(metas[0]["of"]) or any(x[1] != y[0] for x, y in zip(got, got[1:])):
            raise ValueError(f"slices {got} do not cover layers 0:{metas[0]['of']}")
        conns = {}

        def fetch(meta, bid, off, length, view):
            import threading
            k = (bases[meta["lease"]], threading.get_ident())
            c = conns.get(k) or conns.setdefault(k, _conn(bases[meta["lease"]]))
            c.request("GET", f"/v1/pool/piece?lease={meta['lease']}&bid={bid}&off={off}&len={length}",
                      headers=_auth_headers())
            r = c.getresponse()
            if r.status != 200:
                raise RuntimeError(f"piece {bid}+{off}: HTTP {r.status} {r.read()[:200]!r}")
            want, got = r.getheader("X-QV-Hash"), 0
            while got < length:
                n = r.readinto(view[got:])
                if not n:
                    raise RuntimeError("connection closed mid-piece")
                got += n
            r.read()
            if hashlib.blake2b(view, digest_size=16).hexdigest() != want:
                raise RuntimeError(f"piece {bid}+{off}: hash mismatch")

        try:
            with _KeepLeases([(bases[m["lease"]], m["lease"], m["blobs"][0]["bid"]) for m in metas]):
                res = st.import_layers(metas, fetch, streams)
        finally:
            for c in conns.values():
                c.close()
        print(f"[prefix] assembled a {res['n']}-token prefix, layers {res['layers'][0]}:{res['layers'][1]}, from "
              f"{len(metas)} slice(s): {res['bytes'] / 2**20:.0f} MB in {res['seconds']:.1f}s", flush=True)
        return res
    finally:
        for m in metas:
            try:
                c = _conn(bases[m["lease"]])
                c.request("POST", "/v1/pool/release", body=json.dumps({"lease": m["lease"]}),
                          headers={"Content-Type": "application/json", **_auth_headers()})
                c.getresponse().read()
                c.close()
            except Exception:                        # noqa: BLE001 - the sender's lease expires anyway
                pass


def _pull_one(base: str, meta: dict, streams: int) -> dict:
    import threading
    lease = meta["lease"]
    local, conns, conns_lock = threading.local(), [], threading.Lock()

    def fetch(bid, off, length, view):
        c = getattr(local, "c", None)
        if c is None:
            c = local.c = _conn(base)                   # one keep-alive connection per worker thread
            with conns_lock:
                conns.append(c)
        c.request("GET", f"/v1/pool/piece?lease={lease}&bid={bid}&off={off}&len={length}", headers=_auth_headers())
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"piece {bid}+{off}: HTTP {r.status} {r.read()[:200]!r}")
        want = r.getheader("X-QV-Hash")
        got = 0
        while got < length:                             # straight into the pool, no intermediate buffer
            k = r.readinto(view[got:])
            if not k:
                raise RuntimeError(f"piece {bid}+{off}: connection closed after {got} of {length} bytes")
            got += k
        r.read()
        if hashlib.blake2b(view, digest_size=16).hexdigest() != want:
            raise RuntimeError(f"piece {bid}+{off}: hash mismatch")

    try:
        return _store().import_(meta, fetch, streams)
    finally:
        for c in conns:
            c.close()


# ---- backup store (the DRAM PC) ---------------------------------------------------------------------------------------
def start_backup_watch(url: str, every: float = 10.0) -> None:
    """--prefix-backup URL: every `every` s read the store's keys and mark the local entries it holds as backed, so this
    machine's (small) pool evicts those first. The store copies them itself (prefix_store.py --mirror)."""
    import threading

    def loop():
        warned = False
        while True:
            try:
                st = _store()
                held = {(h, int(n)) for h, n in _get_json(url, f"/v1/pool/state?xfer={signature()}").get("held", [])}
                local = [k for k, _, h, n in keyed_entries(st) if (h, n) in held]
                n = st.mark_backed(local)
                if n:
                    print(f"[prefix] backup {url}: {len(local)} of this pool's prefixes held there", flush=True)
                warned = False
            except Exception as e:                   # noqa: BLE001 - the store may be down; try again
                if not warned:
                    print(f"[prefix] backup {url} not reachable ({type(e).__name__}); retrying", flush=True)
                    warned = True
            time.sleep(every)

    threading.Thread(target=loop, daemon=True, name="prefix-backup").start()
