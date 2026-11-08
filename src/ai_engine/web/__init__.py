"""Hand-rolled HTTP and MCP-stdio plumbing, shared by every server here.

There is no web framework in this project on purpose. What the servers need
is a JSON envelope, CORS, a body-size guard and correct Content-Length
handling -- and every one of those was re-implemented (slightly differently)
by each server until this package existed.

    base_handler.py  BaseHTTPRequestHandler subclass (was engine.http_base)
    mcp_stdio.py     JSON-RPC-over-stdio loop for MCP servers

WHY THIS PACKAGE IS CALLED `web` AND NOT `http`
-----------------------------------------------
Naming it `http` shadowed the standard library. Inside these modules
`from http.server import BaseHTTPRequestHandler` resolved to
ai_engine.http, which imports ai_engine.http.handler -- a circular import
that killed the module on first import. A subpackage name that collides
with a stdlib module that its own contents import is not worth the fight;
`web` collides with nothing.
"""
from ai_engine.web.base_handler import QVHTTPHandler    # noqa: F401

__all__ = ["QVHTTPHandler"]
