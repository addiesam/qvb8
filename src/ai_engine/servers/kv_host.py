"""KV host: keeps other machines' OLD KV blocks and computes the attention over them (qv_band/kv/remote_cold.py).

    python kv_host.py --port 8083 --budget-gb 8                    # CPU (numpy): the DRAM PC
    python -m ai_engine.servers.kv_host --device cuda:0 --budget-gb 20   # a GPU machine's RAM (the V100) as backup

A server whose own memory is full (QV_KV_REMOTE=http://host:8083 on it) ships each layer's oldest KV blocks here and,
per layer and step, sends its queries; this host answers (o, lse) over the blocks it holds. Nothing is written to disk.

  POST /v1/kv/put?sid&a&b     one block (wire format of remote_cold.encode); 507 when over --budget-gb
  POST /v1/kv/attend?sid      queries -> (o, lse) over every block of that session
  POST /v1/kv/free?sid        the session's blocks are dropped (the client frees a layer's session when it goes)
  GET  /health                sessions, bytes, budget
A session idle for --ttl seconds is dropped (a client that died). X-QV-Pool-Token is required when --token is set.

Backends: numpy (blocks decoded once to float32, BLAS matmuls per step) or a CUDA device. On a CUDA device the blocks
are decoded once and stay ON THE CARD as fp16 (--blocks gpu, the default: the budget is then GPU memory, and a step is
only matmuls); --blocks host keeps them as compact ids in pinned host RAM and decodes them on the card at every step
(slower by an order of magnitude; for a card too small to hold them).
Connections use TCP_NODELAY on both ends (a request is two small writes; Nagle + delayed ACK cost ~40 ms each).
"""
from __future__ import annotations

import argparse
import hmac
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

try:
    from qv_band.kv import remote_cold as rc
except ImportError:                                  # the DRAM PC: remote_cold.py next to this file
    import remote_cold as rc


class Host:
    def __init__(self, budget: int, device: str | None, ttl: float, resident: bool = True):
        self.budget, self.device, self.ttl = budget, device, ttl
        self.resident = bool(device) and resident        # decoded blocks stay on the card
        self.sessions = {}                           # sid -> {"blocks": [...], "used": t, "bytes": n}
        self.lock = threading.Lock()
        threading.Thread(target=self._reaper, daemon=True, name="kv-host-ttl").start()

    def used(self) -> int:
        return sum(s["bytes"] for s in self.sessions.values())

    def _reaper(self):
        while True:
            time.sleep(60)
            now = time.time()
            with self.lock:
                for sid in [k for k, s in self.sessions.items() if now - s["used"] > self.ttl]:
                    print(f"[kv-host] session {sid[:12]} idle > {self.ttl:.0f}s: dropped", flush=True)
                    del self.sessions[sid]

    def put(self, sid: str, a: int, arrays: dict) -> bool:
        blk = dict(arrays, a=a)
        if self.resident:
            import torch
            blk = rc.prepare_torch(blk, torch.device(self.device))
        elif self.device:
            import torch
            t = {"a": a}
            for k, v in arrays.items():
                x = torch.from_numpy(v.copy())
                t[k] = x.pin_memory() if k.startswith("ids_") else x
            t["nbytes"] = sum(v.nbytes for v in arrays.values())
            blk = t
        else:
            blk = rc.prepare_numpy(blk)
        with self.lock:
            old = self.sessions.get(sid)
            replaced = sum(x["nbytes"] for x in old["blocks"] if x["a"] == a) if old else 0     # a re-sent block frees its old copy
            if self.used() - replaced + blk["nbytes"] > self.budget:
                return False
            s = self.sessions.setdefault(sid, {"blocks": [], "used": time.time(), "bytes": 0})
            s["blocks"] = [b for b in s["blocks"] if b["a"] != a] + [blk]       # a re-sent block replaces itself
            s["bytes"] = sum(b["nbytes"] for b in s["blocks"])
            s["used"] = time.time()
        return True

    def attend(self, sid: str, q, q_pos0: int, scale: float):
        with self.lock:
            s = self.sessions.get(sid)
            if s is None:
                raise KeyError("no such session (expired or freed)")
            s["used"] = time.time()
            blocks = list(s["blocks"])
        if self.device:
            import torch
            fn = rc.attend_resident if self.resident else rc.attend_torch
            with torch.no_grad():
                o, lse = fn(torch.from_numpy(q.copy()), q_pos0, scale, blocks, torch.device(self.device))
            return o.cpu().numpy(), lse.cpu().numpy()
        return rc.attend_prepared(q.astype("float32"), q_pos0, scale, blocks)

    def free(self, sid: str) -> None:
        with self.lock:
            self.sessions.pop(sid, None)


class Handler(BaseHTTPRequestHandler):
    host: Host = None
    token = ""
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True               # answers are a header write + a body write: no 40 ms stall between

    def log_message(self, fmt, *a):
        pass

    def _reply(self, code: int, body: bytes, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlsplit(self.path).path.rstrip("/") in ("/health", ""):
            import json
            h = self.host
            with h.lock:
                body = {"sessions": len(h.sessions), "blocks": sum(len(s["blocks"]) for s in h.sessions.values()),
                        "bytes": h.used(), "budget": h.budget, "device": h.device or "cpu"}
            return self._reply(200, json.dumps(body).encode(), "application/json")
        return self._reply(404, b"not found", "text/plain")

    def do_POST(self):
        if self.token and not hmac.compare_digest(self.headers.get("X-QV-Pool-Token", ""), self.token):
            self.close_connection = True                 # the body is not read: this connection cannot be reused
            return self._reply(403, b"X-QV-Pool-Token missing or wrong", "text/plain")
        n = int(self.headers.get("Content-Length") or 0)
        if n > self.host.budget:                         # a block larger than the whole budget can never be kept
            self.close_connection = True
            return self._reply(413, b"body larger than the budget", "text/plain")
        raw = self.rfile.read(n)
        u = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        path, sid = u.path.rstrip("/"), q.get("sid", "")
        try:
            if path == "/v1/kv/put":
                arrays, _ = rc.decode(raw)
                ok = self.host.put(sid, int(q["a"]), arrays)
                return self._reply(200 if ok else 507, b"ok" if ok else b"over budget", "text/plain")
            if path == "/v1/kv/attend":
                arrays, meta = rc.decode(raw)
                o, lse = self.host.attend(sid, arrays["q"], int(meta["q_pos0"]), float(meta["scale"]))
                if meta.get("o") == "float16":           # the asker's width: half the bytes back
                    o = o.clip(-65504.0, 65504.0).astype("float16")
                return self._reply(200, rc.encode({"o": o, "lse": lse.astype("float32")}))
            if path == "/v1/kv/free":
                self.host.free(sid)
                return self._reply(200, b"ok", "text/plain")
        except KeyError as e:
            return self._reply(404, str(e).encode(), "text/plain")
        except Exception as e:                       # noqa: BLE001 - the client falls back / reports
            return self._reply(500, f"{type(e).__name__}: {e}".encode(), "text/plain")
        return self._reply(404, b"not found", "text/plain")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8083)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--budget-gb", type=float, default=8.0, help="RAM for blocks, GiB")
    ap.add_argument("--device", default=None, help="cuda:N: compute on that GPU (blocks stay in host RAM); default CPU")
    ap.add_argument("--blocks", choices=("gpu", "host"), default="gpu",
                    help="with --device: gpu = decoded blocks stay on the card as fp16 (the budget is GPU memory); "
                         "host = compact ids in pinned host RAM, decoded on the card at every step (much slower)")
    ap.add_argument("--matmul", choices=("tf32", "fp32"), default="tf32",
                    help="with --device: tf32 = tensor-core products with fp32 accumulation (Ampere and newer; keys, "
                         "values and the queries of a 16-bit model are 16-bit values, which TF32 holds exactly, so only "
                         "the attention weights are rounded, to 10 bits); fp32 = plain fp32 products")
    ap.add_argument("--ttl", type=float, default=3600.0, help="seconds a session may sit idle before it is dropped")
    ap.add_argument("--token", default=os.environ.get("QV_POOL_TOKEN", ""), help="X-QV-Pool-Token (QV_POOL_TOKEN)")
    a = ap.parse_args(argv)
    if a.device:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = a.matmul == "tf32"
    Handler.host = Host(int(a.budget_gb * 2**30), a.device, a.ttl, resident=a.blocks == "gpu")
    Handler.token = a.token
    if not a.token:
        print("[kv-host] WARNING: no --token: open to the LAN (it holds prompts' KV)", flush=True)
    print(f"[kv-host] on {a.host}:{a.port}, budget {a.budget_gb:g} GiB, compute on {a.device or 'CPU (numpy)'}"
          + (f", blocks {'resident on the card (fp16)' if a.blocks == 'gpu' else 'in host RAM (decoded per step)'}"
             if a.device else "") + (f", matmul {a.matmul}" if a.device else ""), flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True                    # an idle keep-alive client must not block shutdown
    srv.serve_forever()


if __name__ == "__main__":
    main()
