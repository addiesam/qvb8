"""Terminal chat for a running LLM server: it only sends OpenAI-style requests (POST /v1/chat/completions, streamed)
and prints the answer. No model is loaded here and nothing but the Python standard library is imported, so it works
with every model the server serves and from any machine that can reach it.

    python -m ai_engine.cli.chat_client                                  interactive, server http://127.0.0.1:8080
    python -m ai_engine.cli.chat_client "Explain entropy" --no-interactive
    python -m ai_engine.cli.chat_client --url http://host:50808 --no-think --max-tokens 2000

In the chat: /reset starts a new conversation, /exit (or Ctrl-D) leaves, Ctrl-C stops the current answer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DIM, OFF = "\033[2m", "\033[0m"


def _args():
    p = argparse.ArgumentParser(description="Terminal chat for a running LLM server (API client only).")
    p.add_argument("prompt", nargs="?", help="first message; without it the chat starts empty")
    p.add_argument("--url", default="http://127.0.0.1:8080", help="server address (default %(default)s)")
    p.add_argument("--no-interactive", action="store_true", help="answer the prompt and exit")
    p.add_argument("--system", help="system message")
    p.add_argument("--max-tokens", type=int, help="limit of the answer (default: the server's)")
    p.add_argument("--temperature", type=float, help="default: the server's")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--think", dest="think", action="store_true", default=None, help="ask for reasoning")
    g.add_argument("--no-think", dest="think", action="store_false", help="ask for no reasoning")
    p.add_argument("--hide-reasoning", action="store_true", help="do not print the reasoning text")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds without data before giving up")
    a = p.parse_args()
    if a.no_interactive and not a.prompt:
        p.error("--no-interactive needs a prompt")
    return a


def _ask(a, messages: list[dict]) -> str | None:
    """Stream one answer to the terminal. Returns the answer text, or None if the request failed."""
    body = {"messages": messages, "stream": True, "stream_options": {"include_usage": True}}
    if a.max_tokens is not None:
        body["max_tokens"] = a.max_tokens
    if a.temperature is not None:
        body["temperature"] = a.temperature
    if a.think is not None:
        body["think"] = a.think
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    color = sys.stdout.isatty()
    answer, in_reasoning, usage, finish = [], False, None, None
    t_send, t_first, t_last = time.perf_counter(), None, None      # the client's own clock: the server is not asked
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    print(f"\n[server error] {chunk['error']}", file=sys.stderr)
                    return None
                usage = chunk.get("usage") or usage
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    finish = ch.get("finish_reason") or finish
                    r, c = d.get("reasoning_content"), d.get("content")
                    if r or c or d.get("tool_calls"):
                        t_last = time.perf_counter()
                        t_first = t_first or t_last
                    if r and not a.hide_reasoning:
                        if not in_reasoning:
                            sys.stdout.write(DIM if color else "[reasoning] ")
                            in_reasoning = True
                        sys.stdout.write(r)
                    if c:
                        if in_reasoning:
                            sys.stdout.write((OFF if color else "") + "\n\n")
                            in_reasoning = False
                        sys.stdout.write(c)
                        answer.append(c)
                    if d.get("tool_calls"):
                        sys.stdout.write(f"\n[tool_calls] {json.dumps(d['tool_calls'])}")
                    sys.stdout.flush()
    except KeyboardInterrupt:
        print((OFF if color else "") + "\n[stopped]")
        return "".join(answer)
    except urllib.error.HTTPError as e:
        print(f"\n[server error] HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:2000]}", file=sys.stderr)
        return None
    except (urllib.error.URLError, OSError) as e:
        print(f"\n[no answer from {a.url}] {e}", file=sys.stderr)
        return None
    if in_reasoning and color:
        sys.stdout.write(OFF)
    note = []
    if usage:
        note.append(f"{usage.get('prompt_tokens')} prompt + {usage.get('completion_tokens')} answer tokens")
    if t_first is not None:
        # Measured here, so it includes the network and the server's queue. The answer rate counts from the first
        # text received to the last; the time before it is the prompt going through the model (plus any wait).
        note.append(f"first text after {t_first - t_send:.1f} s")
        n = (usage or {}).get("completion_tokens")
        if n and t_last - t_first > 0.2:
            note.append(f"{n / (t_last - t_first):.1f} tok/s")
    if finish and finish != "stop":
        note.append(f"finish: {finish}")
    print(f"\n{DIM if color else ''}[{', '.join(note)}]{OFF if color else ''}" if note else "")
    return "".join(answer)


def main() -> None:
    a = _args()
    try:
        with urllib.request.urlopen(a.url.rstrip("/") + "/health", timeout=10) as r:
            h = json.loads(r.read())
        print(f"[connected] {a.url}  model {h.get('model')}", file=sys.stderr)
    except (urllib.error.URLError, OSError, ValueError) as e:
        sys.exit(f"no server at {a.url} ({e}). Start it first, or pass --url.")
    base = [{"role": "system", "content": a.system}] if a.system else []
    messages = list(base)
    pending = a.prompt
    while True:
        if pending is None:
            try:
                pending = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not pending:
                pending = None
                continue
            if pending in ("/exit", "/quit"):
                return
            if pending == "/reset":
                messages, pending = list(base), None
                print("[new conversation]")
                continue
        messages.append({"role": "user", "content": pending})
        pending = None
        out = _ask(a, messages)
        if out is None:
            messages.pop()                    # the request failed: the message is not part of the conversation
            if a.no_interactive:
                sys.exit(1)
        else:
            messages.append({"role": "assistant", "content": out})
        if a.no_interactive:
            return


if __name__ == "__main__":
    main()
