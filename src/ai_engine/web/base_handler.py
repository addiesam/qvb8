"""Standard HTTP handler for every hand-rolled server in this repo.

ai_engine' llm_server.py, embedding_server.py and reranker_server.py used to each
define an identical BaseHTTPRequestHandler subclass -- same
protocol_version, same CORS headers, same JSON envelope, same
Content-Length-based body-size guard, same OPTIONS/HEAD handling -- with
only the log line's tag and the request-size limit differing. That was
genuine duplicated logic (byte-identical apart from those two knobs), not
three servers that merely look alike, so it lives once here -- in `engine`,
not in ai_engine, because media_apps' image_gen_server.py needs the same
base and must not depend on a sibling app package for it. Both apps already
depend on engine, so this adds no new dependency edge.

Subclasses set class attributes to opt into or out of behaviour; every
default matches what ai_engine' three servers already did, so adopting this
base is zero-behavior-change for them:

    LOG_TAG          the bracketed prefix in QV_SERVE_LOG output
    MAX_BODY         the request-size limit in bytes, or a 0-arg callable
                     returning it (llm_server.py derives its limit from
                     --ctx-size at request time, so it needs the callable
                     form; most servers use a flat constant)
    ENABLE_CORS      default True. A server with its own bundled same-origin
                     UI and no browser cross-origin caller (image_gen_server)
                     can set this False to skip the CORS headers and
                     OPTIONS handling entirely -- do_OPTIONS then falls back
                     to BaseHTTPRequestHandler's stock 501.
    FORCE_CLOSE      default True (HTTP/1.1 + Connection: close per request
                     -- see the docstring on protocol_version below for why).
                     A server whose requests are single long-running calls
                     with no streaming/SSE risk (image_gen_server) can set
                     this False to allow keep-alive.

A server with request-shape needs the shared `_body()`/`_send()` helpers
don't cover (e.g. a custom multi-stage validate-then-drain-then-refuse
POST body reader) is free to keep its own do_POST/do_GET and use only the
CORS/close/log_message plumbing this base provides -- inheriting from
QVHTTPHandler does not require using every method on it.
"""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Callable

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _cors_allowlist() -> tuple[bool, set[str]]:
    """(allow_any, explicit origins) from QV_CORS_ORIGINS ("*" or a comma list)."""
    raw = os.environ.get("QV_CORS_ORIGINS", "").strip()
    if raw == "*":
        return True, set()
    return False, {o.strip().rstrip("/") for o in raw.split(",") if o.strip()}


class QVHTTPHandler(BaseHTTPRequestHandler):
    # HTTP/1.1 (for chunked SSE support in llm_server.py) but every response
    # forces Connection: close BY DEFAULT (see FORCE_CLOSE). Generation can
    # take tens of seconds to minutes; if any response along the way is
    # malformed (SSE never sends Content-Length, any path that raises after
    # headers went out, etc.) a kept-alive connection is left desynced and
    # the client's NEXT request on that same socket hangs forever waiting on
    # a stream the server already considers done. A fresh TCP connection per
    # request costs nothing at this request rate and removes the whole bug
    # class -- the default a server should only override with a specific
    # reason (image_gen_server.py's single long-running call has none of
    # this streaming risk).
    protocol_version = "HTTP/1.1"

    LOG_TAG = "engine"
    # Print one timestamped line when a request arrives (servers with no
    # request log of their own turn this on).
    LOG_RECEIVE = False
    # Socket timeout for reads/writes: a client that connects and then stalls (or trickles a
    # body) would otherwise pin a server thread forever. Waiting on a generation does not
    # count: the handler is not blocked on the socket while the model runs.
    timeout = int(os.environ.get("QV_SOCKET_TIMEOUT", "120"))
    MAX_BODY: int | "Callable[[], int]" = 50_000_000
    ENABLE_CORS = True
    FORCE_CLOSE = True

    def _end_close(self):
        if self.FORCE_CLOSE:
            self.send_header("Connection", "close")
            self.close_connection = True

    def log_message(self, fmt, *a):      # quieter than the default access log
        if os.environ.get("QV_SERVE_LOG"):
            print(f"[{self.LOG_TAG}] {self.address_string()} {fmt % a}", flush=True)

    # -- request screening --------------------------------------------
    def _origin_ok(self) -> bool:
        """A browser sends Origin on every cross-origin request. A page on another
        site must not be able to drive this server (CSRF), nor read its answers.
        No Origin (curl, SDKs, server-to-server) is always fine. Allowed: loopback
        origins (a local panel on another port) and anything in QV_CORS_ORIGINS
        ("*" allows every origin and restores the old wide-open behaviour)."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        allow_any, listed = _cors_allowlist()
        if allow_any or origin.rstrip("/") in listed:
            return True
        parts = urlsplit(origin)
        if parts.netloc and parts.netloc == (self.headers.get("Host") or ""):
            return True           # the page came from this very server (opened by its LAN address or name): same origin
        return (parts.hostname or "") in _LOOPBACK_HOSTS

    def _host_ok(self) -> bool:
        """DNS-rebinding guard: a server bound to loopback only answers requests whose
        Host header is a loopback name (a rebinding page reaches it as evil.example)."""
        try:
            bound = self.server.server_address[0]
        except Exception:
            return True
        if bound not in ("127.0.0.1", "::1", "localhost"):
            return True
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in {"localhost", "127.0.0.1", "::1"} or not host

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        if not self._host_ok():
            self.send_error(403, "Host not allowed for a loopback-bound server")
            return False
        if not self._origin_ok():
            self.send_error(403, "cross-origin request refused (set QV_CORS_ORIGINS to allow)")
            return False
        if self.LOG_RECEIVE:
            print(f"[{self.LOG_TAG}] {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"received {self.command} {self.path}", flush=True)
        return True

    def _cors_value(self) -> str:
        allow_any, _ = _cors_allowlist()
        origin = self.headers.get("Origin")
        return "*" if allow_any or not origin else origin

    # -- helpers ----------------------------------------------------
    def _send(self, code: int, obj, ctype="application/json"):
        raw = (obj if isinstance(obj, bytes)
               else json.dumps(obj, ensure_ascii=False).encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        if self.ENABLE_CORS:
            self.send_header("Access-Control-Allow-Origin", self._cors_value())
            self.send_header("Vary", "Origin")
        self._end_close()
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ValueError("invalid Content-Length header") from None
        if n < 0:
            # rfile.read(-1) would block until the client closes the socket.
            raise ValueError("invalid Content-Length header")
        if not n:
            return {}
        # Reject an oversized request from Content-Length ALONE, before
        # reading a single byte of the body -- a downstream check (e.g.
        # llm_server.py's --ctx-size/tokenizer check) may be more precise, but it
        # only fires AFTER the full body is read into memory and paid for.
        max_body = self.MAX_BODY() if callable(self.MAX_BODY) else self.MAX_BODY
        if n > max_body:
            raise ValueError(self._body_too_large_message(n, max_body))
        body = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    def _body_too_large_message(self, n: int, max_body: int) -> str:
        """Override to add context (e.g. llm_server.py names --ctx-size)."""
        return f"request body is {n} bytes, over the {max_body}-byte limit"

    def do_OPTIONS(self):
        if not self.ENABLE_CORS:
            return super().do_OPTIONS()
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self._cors_value())
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self._end_close()
        self.end_headers()

    def do_HEAD(self):
        # BaseHTTPRequestHandler 501s HEAD unless a handler exists; some
        # OpenAI-compatible clients probe with HEAD before the real request,
        # and a 501 there can surface as an instant "not found" in the
        # client UI without ever reaching the POST.
        self.send_response(200)
        if self.ENABLE_CORS:
            self.send_header("Access-Control-Allow-Origin", self._cors_value())
        self._end_close()
        self.end_headers()
