"""TLS for every link between machines: HTTP servers and their clients (router -> servers, prefix pool transfers, the
DRAM PC's prefix store and KV host) and the raw block / layer stage links. ONE place, used by ai_engine and qv_band.

On only when the environment names all three files (the deployment config / launchers set them):

    QV_TLS_CERT   this machine's certificate (PEM), signed by the cluster CA
    QV_TLS_KEY    its private key (PEM)
    QV_TLS_CA     the cluster CA certificate (PEM)

    QV_TLS_CLIENTS  who may connect to THIS server: "cluster" (default) = only holders of a cluster certificate (mutual TLS:
                    every machine-to-machine port); "any" = encrypted, but any client (the ports apps and browsers use, e.g.
                    the router's front port; protect those with QV_POOL_TOKEN / the app's own auth). Set per service by
                    the deployment config.

A client always verifies the server against the cluster CA, and presents its own cluster certificate. With
QV_TLS_CLIENTS=cluster a machine without a cluster certificate cannot connect at all (the pool token, QV_POOL_TOKEN,
still applies on top). Host names are not checked: the cluster's machines are addressed by LAN IP and every certificate comes from the one
private CA, which is what is verified. Nothing is automatic: a variable set to a missing file is an error, never a silent
fall back to plain TCP; with none of them set every link is plain TCP as before. Certificates: scripts/tls/make_certs.sh.
"""
from __future__ import annotations

import os
import ssl

_VARS = ("QV_TLS_CERT", "QV_TLS_KEY", "QV_TLS_CA")
_cache: dict = {}


def enabled() -> bool:
    """TLS is on when the three variables are set; some but not all of them is an error."""
    got = [v for v in _VARS if os.environ.get(v)]
    if not got:
        return False
    if len(got) != len(_VARS):
        raise RuntimeError(f"TLS: set all of {', '.join(_VARS)} or none (set: {', '.join(got)})")
    for v in _VARS:
        if not os.path.isfile(os.environ[v]):
            raise RuntimeError(f"TLS: {v}={os.environ[v]} does not exist")
    return True


def _ctx(server: bool) -> ssl.SSLContext:
    key = (server, tuple(os.environ.get(v) for v in _VARS), os.environ.get("QV_TLS_CLIENTS"))
    if key not in _cache:
        c = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if server else ssl.PROTOCOL_TLS_CLIENT)
        c.minimum_version = ssl.TLSVersion.TLSv1_3
        c.load_cert_chain(os.environ["QV_TLS_CERT"], os.environ["QV_TLS_KEY"])
        c.load_verify_locations(os.environ["QV_TLS_CA"])
        if server:
            who = os.environ.get("QV_TLS_CLIENTS", "cluster")
            if who not in ("cluster", "any"):
                raise RuntimeError(f"TLS: QV_TLS_CLIENTS must be cluster or any, not {who!r}")
            c.verify_mode = ssl.CERT_REQUIRED if who == "cluster" else ssl.CERT_NONE
        else:
            c.verify_mode = ssl.CERT_REQUIRED               # the server must hold a cluster certificate
        if not server:
            c.check_hostname = False                        # LAN IPs; the private CA is what is verified
        _cache[key] = c
    return _cache[key]


def server_context() -> ssl.SSLContext | None:
    return _ctx(True) if enabled() else None


def client_context() -> ssl.SSLContext | None:
    return _ctx(False) if enabled() else None


def wrap_server(httpd):
    """An http.server server: each accepted connection is wrapped. The handshake runs in the request's own thread (first
    read), not in the accept loop, so a slow or bad client cannot stall the server. Returns httpd."""
    ctx = server_context()
    if ctx is None:
        return httpd
    orig = httpd.get_request

    def get_request():
        sock, addr = orig()
        return ctx.wrap_socket(sock, server_side=True, do_handshake_on_connect=False), addr
    httpd.get_request = get_request
    orig_error = httpd.handle_error

    def handle_error(request, client_address):
        # a client without a cluster certificate (or plain HTTP on this port) fails the handshake in its own thread: one
        # line, not a traceback per attempt; any other error is the server's own and keeps its full report
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ssl.SSLError, ConnectionResetError)):
            print(f"[tls] {client_address[0]}:{client_address[1]} refused: {getattr(exc, 'reason', None) or exc}", flush=True)
            return
        orig_error(request, client_address)
    httpd.handle_error = handle_error
    httpd.qv_tls = True
    return httpd


def server_socket(sock):
    """An accepted raw socket (stage links)."""
    ctx = server_context()
    return sock if ctx is None else ctx.wrap_socket(sock, server_side=True)


def client_socket(sock, host: str):
    """A connected raw socket (stage links)."""
    ctx = client_context()
    return sock if ctx is None else ctx.wrap_socket(sock, server_hostname=host)


def http_connection(host: str, port: int, timeout: float):
    """http.client connection to another cluster server: HTTPS with the cluster certificate when TLS is on."""
    import http.client
    ctx = client_context()
    if ctx is None:
        return http.client.HTTPConnection(host, port, timeout=timeout)
    return http.client.HTTPSConnection(host, port, timeout=timeout, context=ctx)


def urlopen(req, timeout: float):
    """urllib.request.urlopen for cluster URLs: an http:// URL is sent as https:// with the cluster certificate when TLS
    is on (the deployment's URLs stay the same; the link is what changes)."""
    import urllib.request
    ctx = client_context()
    if ctx is None:
        return urllib.request.urlopen(req, timeout=timeout)
    if isinstance(req, str):
        req = req.replace("http://", "https://", 1) if req.startswith("http://") else req
    elif req.full_url.startswith("http://"):
        req.full_url = "https://" + req.full_url[len("http://"):]
    return urllib.request.urlopen(req, timeout=timeout, context=ctx)


def scheme() -> str:
    return "https" if enabled() else "http"
