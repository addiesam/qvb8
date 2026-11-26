"""Transport between the stages of a pipeline over machines (design_doc/engine_architecture.md, section 6).

One persistent TCP connection per neighbour. A message is a small JSON header plus the raw bytes of its tensors:

    b"QVSL" | u32 header length | header JSON | tensor bytes, in header order

    header = {"op": str, "f": {small JSON fields}, "t": [[name, dtype, shape], ...]}

Nothing is pickled: a peer can only make this side build tensors of the dtypes listed in _DTYPES, of at most
_MAX_TENSOR bytes each.
Failure rule: any error on the connection raises StageLost. The caller stops the requests in flight; there is no
reconnect and no retry here.
Dead peers: TCP keep-alive (probes after 30 s of silence, every 10 s, 3 unanswered = lost) and a 60 s limit on unsent
data, so a neighbour that lost power or its cable ends in StageLost within about a minute instead of a step that never
returns. A neighbour that is only slow (a long prefill chunk) answers the probes and is waited for.
Who may connect is checked by the stage (decode/remote_stage.py: the "hello" message carries QV_POOL_TOKEN).

Link test (gate N0):
    python -m qv_band.run.stage_link serve --port 8081            # on the other machine
    python -m qv_band.run.stage_link test  HOST:8081              # here
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import time

import torch

MAGIC = b"QVSL"
_HEAD = struct.Struct("<4sI")
_MAX_HEADER = 1 << 20
_MAX_TENSOR = 16 << 30          # bytes of one received tensor (a 12288-token chunk of hidden states is 126 MB)
_DTYPES = {str(d).split(".")[1]: d for d in (
    torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
    torch.bool, torch.float64, torch.complex64)}      # float64 / complex64: rotary tables of image / video block stages


class StageLost(RuntimeError):
    """The connection to a neighbour stage is gone or sent something that is not a stage message."""


def parse_addr(addr: str) -> tuple[str, int]:
    host, _, port = addr.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"stage address must be HOST:PORT, got {addr!r}")
    return host, int(port)


class Link:
    """One connected neighbour. send() / recv() are not thread-safe: one caller at a time per direction."""

    def __init__(self, sock: socket.socket, peer: str):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt, val in (("TCP_KEEPIDLE", 30), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 3), ("TCP_USER_TIMEOUT", 60000)):
            if hasattr(socket, opt):                 # Linux; elsewhere the system's keep-alive defaults apply
                sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)
        self.sock, self.peer = sock, peer
        self.sent = self.received = 0          # bytes, for the link report

    @classmethod
    def connect(cls, addr: str, timeout: float = 10.0) -> "Link":
        host, port = parse_addr(addr)
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            from qv_band.common import tls
            sock = tls.client_socket(sock, host)           # TLS when the deployment sets QV_TLS_*
        except OSError as e:
            raise StageLost(f"stage {addr}: cannot connect ({e})") from e
        sock.settimeout(None)
        return cls(sock, addr)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def send(self, op: str, fields: dict | None = None, tensors: dict | None = None) -> None:
        metas, bufs = [], []
        for name, t in (tensors or {}).items():
            dt = str(t.dtype).split(".")[1]
            if dt not in _DTYPES:
                raise TypeError(f"stage link: tensor {name!r} has dtype {t.dtype}, which the link does not carry")
            c = t.detach().to("cpu").contiguous()
            metas.append([name, dt, list(c.shape)])
            if c.numel():
                bufs.append(memoryview(c.reshape(-1).view(torch.uint8).numpy()))
        head = json.dumps({"op": op, "f": fields or {}, "t": metas}, separators=(",", ":")).encode()
        try:
            self.sock.sendall(_HEAD.pack(MAGIC, len(head)) + head)
            for b in bufs:
                self.sock.sendall(b)
        except OSError as e:
            raise StageLost(f"stage {self.peer}: connection lost while sending {op!r} ({e})") from e
        self.sent += _HEAD.size + len(head) + sum(b.nbytes for b in bufs)

    def _read(self, n: int) -> bytearray:
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        try:
            while got < n:
                k = self.sock.recv_into(view[got:], n - got)
                if k == 0:
                    raise StageLost(f"stage {self.peer}: connection closed")
                got += k
        except OSError as e:
            raise StageLost(f"stage {self.peer}: connection lost while receiving ({e})") from e
        self.received += n
        return buf

    def recv(self, device=None) -> tuple[str, dict, dict]:
        """-> (op, fields, tensors). Tensors arrive on the CPU, or on `device` when given."""
        magic, n = _HEAD.unpack(bytes(self._read(_HEAD.size)))
        if magic != MAGIC or n > _MAX_HEADER:
            raise StageLost(f"stage {self.peer}: not a stage message")
        try:
            head = json.loads(bytes(self._read(n)))
            metas = [(str(name), str(dt), [int(s) for s in shape]) for name, dt, shape in head["t"]]
            op, fields = str(head["op"]), head["f"]
        except (ValueError, KeyError, TypeError) as e:
            raise StageLost(f"stage {self.peer}: not a stage message ({type(e).__name__})") from e
        tensors = {}
        for name, dt, shape in metas:
            if dt not in _DTYPES:
                raise StageLost(f"stage {self.peer}: unknown tensor dtype {dt!r}")
            dtype = _DTYPES[dt]
            count = 1
            for s in shape:
                count *= s
            if any(s < 0 for s in shape) or count * torch.empty((), dtype=dtype).element_size() > _MAX_TENSOR:
                raise StageLost(f"stage {self.peer}: tensor {name!r} of shape {shape} is not acceptable")
            if count == 0:
                t = torch.empty(shape, dtype=dtype)
            else:
                raw = self._read(count * torch.empty((), dtype=dtype).element_size())
                t = torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(shape)
            tensors[name] = t.to(device, non_blocking=True) if device is not None else t
        return op, fields, tensors

    def call(self, op: str, fields: dict | None = None, tensors: dict | None = None, device=None):
        """Send one message and wait for its reply. A reply with op "error" raises StageLost with the peer's text."""
        self.send(op, fields, tensors)
        rop, rf, rt = self.recv(device)
        if rop == "error":
            raise StageLost(f"stage {self.peer}: {op!r} failed there: {rf.get('text')}")
        return rf, rt


def listen(port: int, host: str = "0.0.0.0") -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    return srv


def accept(srv: socket.socket) -> Link:
    sock, peer = srv.accept()
    from qv_band.common import tls
    return Link(tls.server_socket(sock), f"{peer[0]}:{peer[1]}")


# ---------------------------------------------------------------------------------------------------------------------
# gate N0: what the link costs for one decode step and for one prefill chunk
# ---------------------------------------------------------------------------------------------------------------------
def _serve(port: int) -> None:
    srv = listen(port)
    print(f"[stage link] echo server on port {port}", flush=True)
    while True:
        link = accept(srv)
        print(f"[stage link] {link.peer} connected", flush=True)
        try:
            while True:
                op, f, t = link.recv()
                if op == "echo":
                    link.send("ok", {"bytes": sum(x.numel() * x.element_size() for x in t.values())})
                elif op == "back":
                    link.send("ok", None, t)
        except StageLost as e:
            print(f"[stage link] {e}", flush=True)
        finally:
            link.close()


def _test(addr: str, hidden: int, tokens: list[int]) -> None:
    link = Link.connect(addr)
    print(f"link to {addr}; hidden size {hidden}, fp16; one way = payload out, a few bytes back")
    print(f"{'tokens':>8} {'payload':>11} {'one way ms':>11} {'MB/s':>8} {'both ways ms':>13}")
    for n in tokens:
        x = torch.randn(1, n, hidden).to(torch.float16)
        size = x.numel() * 2
        reps = 200 if size < (1 << 22) else 5
        for op in ("echo", "back"):
            link.call(op, None, {"h": x})
        one, both = [], []
        for _ in range(reps):
            t0 = time.perf_counter(); link.call("echo", None, {"h": x}); one.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); link.call("back", None, {"h": x}); both.append(time.perf_counter() - t0)
        o, b = sorted(one)[len(one) // 2], sorted(both)[len(both) // 2]
        print(f"{n:>8} {size / 1e6:>9.2f}MB {o * 1e3:>11.3f} {size / 1e6 / o:>8.0f} {b * 1e3:>13.3f}", flush=True)
    link.close()


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="stage link test (gate N0)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, required=True)
    t = sub.add_parser("test"); t.add_argument("addr")
    t.add_argument("--hidden", type=int, default=5120, help="hidden size of the model (27B: 5120)")
    t.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 16, 1024, 12288])
    a = ap.parse_args(argv)
    _serve(a.port) if a.cmd == "serve" else _test(a.addr, a.hidden, a.tokens)


if __name__ == "__main__":
    main(sys.argv[1:])
