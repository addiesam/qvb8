"""llm_server's streaming responses (split out of llm_server.py): the SSE wire, the per-reply stream state, and the
two streaming handlers (text: _stream, images: _vision_chat_stream) as a mixin of llm_server.Handler.

    _SSEOut          headers, coalesced writes with a send timeout, client-gone check, pings, chunk shape, [DONE]
    _StreamState     held-back window, </think> split, tool-call buffer, stop strings; finish() = the closing writes
    _ThinkSplitter   </think> split of the image path
    StreamMixin      _stream / _vision_chat_stream

Everything else these read lives in llm_server (STATE, MODEL_ID, helpers): reached as S.<name> at call time, so the
values the server sets at load are the ones used, and importing this module does not import llm_server.
"""
from __future__ import annotations

import json
import queue
import socket
import sys
import threading
import time
import traceback


class _LLMServer:
    """llm_server's module namespace, looked up on use (it imports this module, so not at import time)."""

    def __getattr__(self, name):
        mod = sys.modules.get("ai_engine.servers.llm_server")
        if mod is None:                                  # used on its own (a test): load the server module
            import importlib
            mod = importlib.import_module("ai_engine.servers.llm_server")
        return getattr(mod, name)


S = _LLMServer()

class _StreamState:
    """What one streamed reply holds between pieces (was _stream's closure lists + its _Shim): the held-back window
    for stop strings and a split </think> (pending), the post-</think> text of a tools request kept for
    parse_tool_calls (buf), whether the reply is still in its reasoning part, whether a stop string ended it, and the
    model's stream gate (a format that opens its own think block, e.g. Ripple / Inkling).
      write(piece)   one decoded piece (run_generate's on_text, or the scheduler path's token queue)
      finish(...)    the closing writes, the SAME for the scheduler and the single-request path
    emit(text, reasoning=) sends a live piece (it checks that the client is still there); finish() writes through
    the wire directly (no liveness check: a stale read there dropped the closing chunk -- Cline: "Response stream
    ended without a finish reason")."""
    TAG = "</think>"

    def __init__(self, *, think: bool, has_tools: bool, stops, gate, emit, wire, chat: bool):
        self.buffer_all = has_tools          # tools: post-</think> text waits for parse_tool_calls (see _stream)
        self.stops = stops
        self.margin = max((len(x) for x in stops), default=1) - 1
        self.reasoning_margin = max(self.margin, len(self.TAG) - 1) if think else self.margin
        self.in_reasoning = think
        self.gate, self.emit, self.wire, self.chat = gate, emit, wire, chat
        self.buf: list[str] = []
        self.pending = ""
        self.stopped = False

    def flush(self):
        pass

    def write(self, piece):
        if not piece or self.stopped:
            return
        if self.gate is not None:
            piece, _dec = self.gate.feed(piece)
            if _dec is False:
                self.in_reasoning = False
            if not piece:
                return
        if self.buffer_all and not self.in_reasoning:
            # Past </think> (or think=False, so in_reasoning started
            # False and there is no boundary to wait for) -- a
            # <tool_call> can appear anywhere in this remaining
            # text, so it must stay buffered for parse_tool_calls()
            # at the end.
            self.buf.append(piece)
            return
        self.pending += piece
        cut = min((self.pending.index(s) for s in self.stops if s in self.pending),
                 default=None)
        if cut is not None:
            self.stopped = True
            out = self.pending[:cut]
            self.pending = ""
        elif len(self.pending) > self.reasoning_margin:
            split = len(self.pending) - self.reasoning_margin
            # BUG FIX: a fixed-width self.margin (len(self.TAG)-1
            # trailing chars retained) is not enough on its own --
            # it protects the SUFFIX of self.pending from being
            # released while it could still grow into the tag, but
            # says nothing about the split point ITSELF landing
            # inside a partial tag match. Concretely: self.pending =
            # "...this. </think>" splits at len-7, releasing
            # "...this. <" (ending in the tag's OPENING '<') and
            # retaining "/think>" (7 chars) -- neither the released
            # prefix nor the retained suffix contains the full tag
            # any more, so out.find(self.TAG) below can never
            # find it and self.in_reasoning never flips. Confirmed:
            # this silently misclassified an ENTIRE response as
            # reasoning_content whenever the tag happened to land
            # on this exact boundary -- a real, pre-existing bug,
            # not new. Fix: while still in_reasoning, never let the
            # released slice's own tail be a partial prefix of the
            # tag -- pull the split point back to before that
            # partial match starts, so the tag stays whole and
            # detectable once enough of it has arrived.
            if self.in_reasoning:
                window = self.pending[:split]
                max_check = min(len(self.TAG) - 1, split)
                for i in range(max_check, 0, -1):
                    if window.endswith(self.TAG[:i]):
                        split -= i
                        break
            out = self.pending[:split]
            self.pending = self.pending[split:]
        else:
            return
        if not out:
            return
        # Split `out` at </think> if it closes inside this piece --
        # everything up to the tag is reasoning, the tag itself is
        # dropped (never shown to the client, same as before), the
        # remainder is real content. Only relevant while still
        # in_reasoning; once closed, every later write is plain
        # content and this whole branch is skipped for speed.
        pieces: list[tuple[str, bool]] = []  # (text, is_reasoning)
        if self.in_reasoning:
            tag_at = out.find(self.TAG)
            if tag_at >= 0:
                if tag_at:
                    pieces.append((out[:tag_at], True))
                rest = out[tag_at + len(self.TAG):]
                self.in_reasoning = False
                # self.pending can ALREADY hold post-tag content at
                # this exact moment -- the self.margin window that just
                # released `out` was sized for self.reasoning_margin
                # (protecting a possibly-still-forming tag), so
                # self.pending can be holding up to self.reasoning_margin
                # chars of text that arrived AFTER the tag closed,
                # not before it.
                #
                # self.buffer_all: drain self.pending to EMPTY (not down
                # to `self.margin`), and prepend all of it to `rest`.
                # Every LATER write() call routes straight into buf
                # via the self.buffer_all-and-not-in_reasoning fast path
                # (bypassing self.pending/the stop-self.margin machinery
                # entirely -- stop-string truncation for buffered
                # content happens once at the end, via
                # _truncate_at_stop on the fully joined buf, not
                # here). Leaving ANY residue in self.pending at this
                # point means it only gets flushed by the tail
                # block after generation ends, landing AFTER every
                # char the fast path already appended in the
                # meantime -- a real, confirmed bug: "answer "
                # left in self.pending (its length happened to equal
                # `self.margin` exactly, so the old self.margin-only drain
                # skipped it) surfaced as "STOPHERE junkanswer "
                # instead of "answer STOPHERE junk". has_tools=
                # False does not hit this: it keeps using
                # self.pending+self.margin for its OWN stop-matching on
                # live content, so only the self.buffer_all branch drains
                # to zero here.
                if self.buffer_all:
                    rest += self.pending
                    self.pending = ""
                elif len(self.pending) > self.margin:
                    drain = len(self.pending) - self.margin
                    rest += self.pending[:drain]
                    self.pending = self.pending[drain:]
                if rest:
                    if self.buffer_all:
                        self.buf.append(rest)
                    else:
                        pieces.append((rest, False))
            else:
                pieces.append((out, True))
        else:
            pieces.append((out, False))
        try:
            for text, is_reasoning in pieces:
                self.emit(text, reasoning=is_reasoning)
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            # The client is gone (Cline cancelling a request is
            # routine) or has stalled long enough to hit _pump's
            # write timeout -- same outcome either way: nothing
            # left to deliver. Swallowing this and carrying on means the
            # server keeps decoding every remaining token for
            # nobody, holding the global generation lock the whole
            # time and blocking real requests behind it. Raising
            # here unwinds speculative_generate -- which has no
            # interruption hook of its own, it just prints -- out
            # through _stream's try/finally (stdout restored,
            # empty_cache called) and into do_POST's handler.
            self.stopped = True
            raise S._ClientGone()

    def finish(self, n_out: int, max_out: int, n_prompt: int, include_usage: bool, timings=None) -> None:
        """Closing writes: the held-back window (split at </think>; post-tag text of a tools request joins buf), the
        buffered answer (tool calls parsed), the finish_reason chunk with usage, [DONE]. The scheduler path used to
        have its own copy that, for a tools request, dropped the held-back window (the reasoning's last characters
        and answer text that arrived with the closing tag)."""
        w, chat = self.wire, self.chat
        finish = "stop" if self.stopped else "length" if n_out >= max_out else "stop"
        if self.gate is not None:
            if self.gate.decided is None:
                self.in_reasoning = False                   # tiny output that never decided: plain content
            self.pending = self.gate.take() + self.pending  # + any tail the gate held back (e.g. "\n</thi")

        def usage(chunk):
            if include_usage:
                chunk["usage"] = {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                                  "total_tokens": n_prompt + n_out}
                if timings is not None:
                    chunk["timings"] = timings
            return chunk
        try:
            if not self.stopped and self.pending:
                flushed, _ = S._truncate_at_stop(self.pending, self.stops)
                self.pending = ""
                tail: list[tuple[str, bool]] = []
                if flushed and self.in_reasoning:           # cut off mid-thought: it all stays reasoning
                    at = flushed.find(self.TAG)
                    if at >= 0:
                        if at:
                            tail.append((flushed[:at], True))
                        rest = flushed[at + len(self.TAG):]
                        if rest:
                            if self.buffer_all:
                                self.buf.append(rest)
                            else:
                                tail.append((rest, False))
                    else:
                        tail.append((flushed, True))
                elif flushed and self.buffer_all:
                    self.buf.append(flushed)
                elif flushed:
                    tail.append((flushed, False))
                for text, is_reasoning in tail:
                    w.sse(w.chunk({"reasoning_content" if is_reasoning else "content": text})
                          if chat else {"text": text})
            if not self.buffer_all:
                if chat:
                    w.sse(usage(w.chunk({}, finish)))
            else:
                joined, hit = S._truncate_at_stop("".join(self.buf), self.stops)
                if hit:
                    finish = "stop"
                if not chat:                                # /generate, /v1/completions: no tool-call contract
                    if joined:
                        w.sse({"text": joined})
                else:
                    content, tool_calls = S.parse_tool_calls(joined)
                    if tool_calls:
                        delta, finish = {"tool_calls": tool_calls}, "tool_calls"
                    else:
                        delta = {"content": content or joined or None}
                    w.sse(usage(w.chunk(delta, finish)))
            w.done()
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError, S._ClientGone):
            pass

class _SSEOut:
    """The SSE wire of one streaming response, shared by the text (_stream) and image (_vision_chat_stream) paths:
    headers, coalesced writes bounded by a send timeout, a throttled client-gone check, keep-alive pings, the
    OpenAI chunk shape, [DONE] and the error event.
      writes   queued and flushed every FLUSH_S: a per-token write()+flush() pair serialises with the decode, and a
               client renders on its own frame budget anyway (SGLang / llama.cpp batch their writes too). Every
               socket write has a WRITE_TIMEOUT_S send timeout: the writing thread may hold the generation lock (or
               feed the scheduler), so a client that stopped reading must not stall it -- the write raises
               socket.timeout, which callers treat like a closed pipe.
      gone()   select() + recv(MSG_PEEK): two syscalls, so at most every PEER_POLL_S (force=True: now).
      ping()   an SSE comment line (":" -- every conformant client ignores it), written at once: during a long
               prefill or while a tool call is held back the client still sees bytes, so its body-read timeout
               (Cline: UND_ERR_BODY_TIMEOUT, then a resubmit queued behind the old request) does not fire.
               Best effort: a lost ping only lets that timeout fire sooner."""
    PEER_POLL_S, FLUSH_S, WRITE_TIMEOUT_S = 0.25, 0.05, 2.0

    def __init__(self, handler, created: int, model_id: str):
        self.h, self.created, self.model_id = handler, created, model_id
        self.cid = f"chatcmpl-{created}"
        self._buf: list[bytes] = []
        self.last_write = time.monotonic()              # of the last real socket write (pings use it)
        self._peer = [0.0, False]                       # [last check, result]

    def start(self) -> None:
        h = self.h
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Access-Control-Allow-Origin", h._cors_value())
        h.send_header("Vary", "Origin")
        h._end_close()
        h.end_headers()

    def gone(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._peer[0] < self.PEER_POLL_S:
            return self._peer[1]
        self._peer[0] = now
        g = self.h.peer_closed()                        # web/base_handler.py: the one check
        self._peer[1] = g
        return g

    def sse(self, obj) -> None:
        self._buf.append(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")

    def pump(self, force: bool = False) -> None:
        if not self._buf:
            return
        now = time.monotonic()
        if not force and now - self.last_write < self.FLUSH_S:
            return
        conn = self.h.connection
        conn.settimeout(self.WRITE_TIMEOUT_S)
        try:
            self.h.wfile.write(b"".join(self._buf))
            self.h.wfile.flush()
        finally:
            conn.settimeout(None)
        self._buf.clear()
        self.last_write = now

    def ping(self) -> None:
        self._buf.append(b": ping\n\n")
        try:
            self.pump(force=True)
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            pass                                        # the next gone() poll notices a closed client

    def chunk(self, delta: dict, finish=None, **extra) -> dict:
        """One chat.completion.chunk; finish_reason only on the closing one."""
        ch = {"index": 0, "delta": delta}
        if finish is not None:
            ch["finish_reason"] = finish
        out = {"id": self.cid, "object": "chat.completion.chunk", "created": self.created,
               "model": self.model_id, "choices": [ch]}
        out.update(extra)
        return out

    def done(self) -> None:
        self._buf.append(b"data: [DONE]\n\n")
        self.pump(force=True)

    def error(self, e, prefix: str = "") -> None:
        """The failure as an SSE error object + [DONE] (a client sees what happened, not a dropped stream)."""
        try:
            self.sse({"error": {"message": f"{prefix}{e}", "type": type(e).__name__}})
            self.done()
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            pass

class _ThinkSplitter:
    """Splits a token stream at the model's </think> into reasoning and answer
    text. The tag can arrive split across pieces ("</th" + "ink>"), so while in
    the reasoning part the last len(tag)-1 characters are held back until the
    next piece shows they are not the start of the tag."""
    CLOSE = "</think>"

    def __init__(self, in_reasoning: bool):
        self.reasoning = in_reasoning
        self._hold = ""

    def feed(self, piece: str) -> list[tuple[str, str]]:
        """-> [("reasoning_content" | "content", text), ...]"""
        self._hold += piece
        out: list[tuple[str, str]] = []
        while self.reasoning:
            if self.CLOSE in self._hold:
                head, _, self._hold = self._hold.partition(self.CLOSE)
                if head:
                    out.append(("reasoning_content", head))
                self.reasoning = False
                continue
            keep = len(self.CLOSE) - 1
            if len(self._hold) > keep:
                out.append(("reasoning_content", self._hold[:-keep]))
                self._hold = self._hold[-keep:]
            return out
        if self._hold:
            out.append(("content", self._hold))
            self._hold = ""
        return out

    def finish(self) -> list[tuple[str, str]]:
        """Whatever is still held when the stream ends (an unclosed think block
        stays reasoning)."""
        text, self._hold = self._hold, ""
        if not text:
            return []
        return [("reasoning_content" if self.reasoning else "content", text)]


class _GatedSplit:
    """_ThinkSplitter behind the model's stream gate (a chat format that writes its own channels, e.g. Ripple /
    Inkling: the gate turns them into reasoning, "</think>", answer). Without a format: _ThinkSplitter(think) as is.
    With one, the splitter starts in reasoning only when the gate says the reply opens with reasoning."""

    def __init__(self, fmt, think: bool):
        self.gate = fmt.new_stream_gate() if fmt is not None else None
        self.split = None if self.gate is not None else _ThinkSplitter(think)
        self._held = ""

    def feed(self, piece: str):
        if self.gate is None:
            return self.split.feed(piece)
        text, decided = self.gate.feed(piece)
        if self.split is None:
            self._held += text
            if decided is None:
                return []
            self.split = _ThinkSplitter(bool(decided))
            text, self._held = self._held, ""
        return self.split.feed(text)

    def finish(self):
        if self.gate is None:
            return self.split.finish()
        tail = self.gate.take()
        if self.split is None:
            self.split = _ThinkSplitter(bool(self.gate.decided))
            tail, self._held = self._held + tail, ""
        return self.split.feed(tail) + self.split.finish()


class StreamMixin:
    """The streaming handlers of llm_server.Handler (they use its socket / header helpers through self)."""

    def _vision_chat_stream(self, body, max_tokens, think, temperature, top_k, top_p,
                            req_id: int = 0):
        """SSE version of _vision_chat. The image route used to answer a streaming
        request with ONE JSON body after the whole generation (tower load + a
        thinking block of thousands of tokens), so a client saw no bytes for
        minutes and gave up or looked hung. Here headers and a keep-alive go out
        at once, tokens stream as they are sampled (reasoning_content before
        </think>, content after), and a client that goes away stops the run."""
        from ai_engine.generation import vision as _vision
        from qv_band.decode.spec_decode import _IncStreamer
        bundle = S.STATE["bundle"]
        has_tools = bool(body.get("tools"))
        stops = S._stop_list(body)
        margin = max((len(x) for x in stops), default=1) - 1
        include_usage = body.get("stream_options", {}).get("include_usage", True)
        out = _SSEOut(self, int(time.time()), S.MODEL_ID)     # the wire: shared with _stream
        out.start()

        def delta(key: str, text: str) -> None:
            if text:
                out.sse(out.chunk({key: text}))
                out.pump()

        split = _GatedSplit(S._chat_format(), bool(think))     # a model writing its own channels: its gate first
        st = {"sent": 0, "content": [], "n": 0}
        streamer = None
        t0 = time.time()
        # Same "arrived" line the text path prints -- without it a streaming
        # image request is invisible in the log until it finishes (or dies).
        print(f"[serve] {S._ts()} req #{req_id} arrived "
              f"(thread={threading.current_thread().name}), vision stream",
              file=sys.__stdout__, flush=True)

        def route_pieces(pieces) -> None:
            for key, text in pieces:
                if key == "reasoning_content":
                    delta(key, text)
                    continue
                st["content"].append(text)
                if has_tools:                           # needs the whole text for parse_tool_calls
                    continue
                full = "".join(st["content"])
                hit, _ = S._truncate_at_stop(full, stops)
                cut = min(max(len(full) - margin, st["sent"]), len(hit))
                delta("content", full[st["sent"]:cut])
                st["sent"] = cut

        try:
            out.ping()                                 # bytes on the wire before the slow parts
            with S.STATE["lock"], S._batch_paused():
                ad = _vision.adapter(bundle, S.STATE.get("drafter_obj"))            # first image request loads the tower
                out.ping()
                streamer = _IncStreamer(bundle.tok)

                def on_token(tid: int) -> None:
                    st["n"] += 1
                    route_pieces(split.feed(streamer.feed([tid])))

                r = ad.generate(
                    S._denormalise_tool_calls(body["messages"]), tools=body.get("tools"),
                    think=think, max_new=max_tokens,
                    temperature=0.0 if temperature is None else float(temperature),
                    top_k=20 if top_k is None else int(top_k),
                    top_p=0.95 if top_p is None else float(top_p),
                    on_token=on_token, should_stop=out.gone,
                    on_progress=out.ping,
                    prefill_chunk=getattr(S.STATE.get("params"), "prefill_chunk", None))
                S._maybe_empty_cache(bundle)
            if r.get("stopped"):
                print(f"[serve] {S._ts()} req #{req_id} vision stream: client gone after "
                      f"{time.time() - t0:.1f}s, {st['n']} tok",
                      file=sys.__stdout__, flush=True)
                return
            route_pieces(split.feed(streamer.finish()))  # a half-emitted codepoint held back
            route_pieces(split.finish())
            text, hit_stop = S._truncate_at_stop("".join(st["content"]), stops)
            finish = "stop" if hit_stop or r["completion_tokens"] < max_tokens else "length"
            if has_tools:
                content, tool_calls = S.parse_tool_calls(text)
                last = {"tool_calls": tool_calls} if tool_calls else {"content": content or text or None}
                finish = "tool_calls" if tool_calls else finish
            else:
                delta("content", text[st["sent"]:])
                last = {}
            chunk = out.chunk(last, finish)
            if include_usage:
                chunk["usage"] = {"prompt_tokens": r["prompt_tokens"],
                                  "completion_tokens": r["completion_tokens"],
                                  "total_tokens": r["prompt_tokens"] + r["completion_tokens"]}
                if r.get("timings"):
                    chunk["timings"] = r["timings"]
            out.sse(chunk)
            out.done()
            S.STATE["served"] += 1
            S.STATE["tokens"] += r["completion_tokens"]
            dt = time.time() - t0
            print(f"[serve] {S._ts()} req #{req_id} done: {st['n']} tok in {dt:.1f}s "
                  f"({st['n']/dt:.2f} tok/s), {r['prompt_tokens']} prompt tok, "
                  f"{r.get('images', 0)} image(s), vision stream",
                  file=sys.__stdout__, flush=True)
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            print(f"[serve] {S._ts()} req #{req_id} vision stream: connection lost "
                  f"after {time.time() - t0:.1f}s",
                  file=sys.__stdout__, flush=True)
        except Exception as e:
            print(f"[serve] {S._ts()} req #{req_id} vision failed: {type(e).__name__}: {e}",
                  file=sys.__stdout__, flush=True)
            traceback.print_exc()
            out.error(e, "vision: ")

    def _stream(self, text: str, max_tokens: int, draft: int, route: str,
               has_tools: bool = False, think: bool = False,
               stops: list[str] | None = None,
               temperature: float | None = None, top_k: int | None = None,
               top_p: float | None = None, req_id: int = 0,
               include_usage: bool = False):
        """Server-sent events.  At ~0.6 tok/s a long answer is minutes away,
        so streaming is the difference between usable and not.

        When the request carried "tools", a <tool_call> block can appear
        anywhere in the generated text and must never reach the client as
        raw content (Cline etc. would render the literal XML). There is no
        cheap way to know a token is/isn't inside a future tool call as it
        streams, so with has_tools this buffers the whole generation instead
        of emitting live deltas, then runs parse_tool_calls() once at the
        end -- same parser the non-streaming path uses. Requests without
        tools are unaffected and keep the original token-by-token path.

        Thinking (enable_thinking=True) is streamed incrementally too, but
        on a SEPARATE wire field: everything up to (not including) the
        closing </think> tag is sent as delta.reasoning_content, matching
        the vLLM/SGLang/DeepSeek convention Cline and other OpenAI-
        compatible clients already parse into their own collapsible
        "thinking" UI -- see reasoning_content in streaming docs. This
        replaces the older behaviour of buffering the whole response until
        </think> closed and discarding everything before it: that made a
        long reasoning block (this checkpoint's <think> routinely runs
        1200-5700+ tokens) look like a multi-minute hang with zero bytes
        on the wire, even though decode was running at its normal rate the
        whole time -- the client had nothing to show, not because nothing
        was happening. Streaming reasoning_content fixes the PERCEIVED
        latency; it does not change decode throughput itself.

        has_tools keeps the old whole-response buffer: a <tool_call> block
        can appear anywhere in the generated text and must never reach the
        client as raw content, and unlike the </think> boundary there is
        no cheap way to know a token is/isn't inside a future tool call as
        it streams, so this still buffers to run parse_tool_calls() once
        at the end. Requests without tools use the incremental path below
        regardless of `think` -- thinking and non-thinking content share
        the same split-by-</think> logic, just with reasoning_content
        empty (immediate switch to content) when </think> never appears
        (e.g. --no-think, or a non-thinking checkpoint).

        "stop" strings can span more than one streamed piece, so a small
        trailing window (the longest stop string, minus one char) is always
        held back before emitting; once a stop is found in the held buffer,
        emission cuts off there and the rest of the generation (which still
        runs to completion internally -- see _truncate_at_stop) is dropped
        rather than sent.
        """
        # Only the POST-</think> content needs buffering for has_tools: a
        # <tool_call> block is emitted by the checkpoint's own chat template
        # convention only in the answer that follows </think>, never inside
        # the reasoning block itself (the template puts <think> first,
        # always closed before any tool-call or plain-answer text begins).
        # So reasoning can stream incrementally exactly like the no-tools
        # path -- only content AFTER </think> needs to wait for
        # parse_tool_calls() to run once at the end. `buffer_all` now means
        # "buffer post-</think> content", not "buffer everything"; a
        # request with think=False and has_tools=True still buffers from
        # the first byte, since there is no </think> to wait for. (_StreamState.buffer_all)
        stops = stops or []
        created = int(time.time())
        chat = route == "/v1/chat/completions"
        sse_out = _SSEOut(self, created, S.MODEL_ID)      # the wire: headers, writes, pings (shared with vision)
        sse_out.start()

        bundle = S.STATE["bundle"]
        tok = bundle.tok          # model is reached via run_generate(bundle, ...)
        S._enter_home(bundle.model)
        ids = tok(text, return_tensors="pt").input_ids
        if bundle.use_gpu:
            ids = ids.to(bundle.gpu_dev)

        # The wire's operations (_SSEOut: throttled client-gone check, coalesced writes with a send timeout,
        # keep-alive pings). _ping is run_generate's on_heartbeat: it fires on a wall-clock timer during a long
        # chunked prefill and slow decode stretches alike.
        _peer_gone, _sse, _pump, _ping = sse_out.gone, sse_out.sse, sse_out.pump, sse_out.ping

        def emit(piece: str, reasoning: bool = False):
            if _peer_gone():
                raise S._ClientGone()
            if chat:
                delta_key = "reasoning_content" if reasoning else "content"
                obj = sse_out.chunk({delta_key: piece})
            else:
                # /v1/completions and /generate have no reasoning_content
                # slot in their wire shape -- reasoning text is dropped
                # there, same as this whole feature not existing pre-fix,
                # since only the chat-completions shape has a place to put
                # it and a client (Cline) actually parsing it.
                if reasoning:
                    return
                obj = {"text": piece}
            _sse(obj)
            _pump()

        # The generated text, piece by piece, goes to ss.write: from run_generate(on_text=) on the single-request
        # path, from the scheduler's token queue on the batched one.
        # A model whose output OPENS its own <think> (Ripple) decides reasoning vs content from its first
        # characters instead of assuming the prompt opened the block (Qwen). None = Qwen behaviour, unchanged.
        _fmt = S._chat_format()
        ss = _StreamState(think=think, has_tools=has_tools, stops=stops, emit=emit, wire=sse_out, chat=chat,
                          gate=_fmt.new_stream_gate() if (_fmt is not None and think) else None)

        drafter = S.STATE["drafter_obj"] if draft > 0 else None

        # See generate(): draft>0 no longer forces greedy -- speculative
        # sampling applies the request's temperature/top_k/top_p. The one
        # KV-mode -> kwargs mapping is in run_generate; nothing here
        # re-derives it. _request_params clamps max_output_tokens to
        # ctx_size and folds in the base GenParams for any None field.
        rp = S._request_params(max_tokens, draft, temperature, top_k, top_p)

        # ---- CONTINUOUS-BATCHING SCHEDULER PATH -----------------------
        # When STATE["scheduler"] is set (qvb8_resident, see
        # load()), route through it instead of the old STATE["lock"]-
        # serialised call below -- this is what actually lets two Cline
        # connections share one batched forward pass per decode step
        # rather than queueing behind each other. Reuses the SAME emit()/
        # _StreamState machinery already built above (</think>
        # split, tool-call buffering, stop-string truncation) -- only the
        # SOURCE of the token text differs: instead of speculative_
        # generate() printing to a swapped sys.stdout under one global
        # lock, the scheduler's on_token callback (running on the
        # scheduler's OWN background thread, batching this request
        # together with whatever others are in the same round) pushes
        # token ids into a queue THIS thread blocks reading, decoded
        # incrementally the same way the non-batched path already does.
        if S.STATE["scheduler"] is not None:
            print(f"[serve] {S._ts()} req #{req_id} arrived, "
                  f"prompt {ids.shape[1]} tok", file=sys.__stdout__, flush=True)
            ids_dev = ids
            out_q = S.STATE["scheduler"].submit(
                ids_dev, rp.max_output_tokens, rp.temperature or 0.0,
                rp.top_k or 0, rp.top_p or 1.0, req_id)
            _shim = ss
            from qv_band.decode.spec_decode import (
                _IncStreamer, HEARTBEAT_INTERVAL_S as _HEARTBEAT_S)
            _streamer = _IncStreamer(S.STATE["tok"])
            st = None
            n_recv = 0
            t0 = time.time()
            try:
                while True:
                    # Ping on WIRE silence, not queue silence: while a long
                    # tool call is held back (parse_tool_calls() needs the
                    # whole text) tokens keep arriving here every few ms, so a
                    # get() timeout never fires, yet nothing reaches the
                    # socket. sse_out.last_write is the time of the last real write.
                    # The timeout also covers a stalled engine (no tokens).
                    try:
                        item = out_q.get(timeout=1.0)
                    except queue.Empty:
                        item = None
                        # Nothing from the engine for a second (a long prefill, say): is the client still
                        # there? A cheap socket peek, so a stopped request frees its slot within ~1 s of
                        # leaving instead of after the 10 s ping interval.
                        if _peer_gone():
                            raise S._ClientGone()
                    if time.monotonic() - sse_out.last_write >= _HEARTBEAT_S:
                        if _peer_gone():
                            raise S._ClientGone()
                        _ping()
                    if item is None:
                        continue
                    if isinstance(item, Exception):
                        raise item
                    if hasattr(item, "stats"):   # _StreamDone sentinel
                        st = item.stats
                        break
                    n_recv += 1
                    piece = _streamer.feed([item])
                    if piece:
                        _shim.write(piece)
                        if ss.stopped:
                            # A stop string was hit: the answer is complete. Free the slot now
                            # instead of letting it generate to max_tokens for nobody.
                            S.STATE["scheduler"].cancel(req_id)
                            break
                if not ss.stopped:
                    tail = _streamer.finish()
                    if tail:
                        _shim.write(tail)
            except S._ClientGone:
                S.STATE["scheduler"].cancel(req_id)
                print(f"[serve] {S._ts()} req #{req_id} client disconnected after "
                     f"{time.time()-t0:.1f}s (scheduler path)",
                     file=sys.__stdout__, flush=True)
                return
            except Exception as e:
                S.STATE["scheduler"].cancel(req_id)
                traceback.print_exc()
                sse_out.error(e)
                return
            dt = time.time() - t0
            out = n_recv                 # every token sent (a retracted request's stats cover only its last run)
            S.STATE["served"] += 1
            S.STATE["tokens"] += out
            ss.finish(out, rp.max_output_tokens, int(ids.shape[1]), include_usage)
            line = (f"[serve] {S._ts()} req #{req_id} "
                   f"(thread={threading.current_thread().name}) "
                   f"streamed {out} tok in {dt:.1f}s "
                   f"({out/dt:.2f} tok/s) [scheduler]")
            if st and st.proposed:
                line += (f"  |  drafted {st.proposed}, accepted {st.accepted} "
                        f"({st.acceptance:.0%}), {st.emitted/max(1,st.steps):.2f} tok/step, "
                        f"{st.steps} steps: draft {st.t_draft:.1f}s verify {st.t_verify:.1f}s")
            print(line, file=sys.__stdout__, flush=True)
            return
        # ---- END SCHEDULER PATH ----------------------------------------

        # Printed BEFORE acquiring STATE["lock"] vs after would tell "queued"
        # from "actually running" apart -- this one fires the instant the
        # request arrives (before it waits its turn, if any), so two
        # requests logging "start" back-to-back with no "acquired the lock"
        # gap between them is exactly what a real overlap (a working
        # parallel scheduler) would look like, and what the current
        # STATE["lock"]-serialised path never produces (request 2's "start"
        # fires immediately, but its own prefill only begins once request
        # 1's full generation has released the lock -- watch the gap
        # between "start" and the first [prefill] line for each id).
        print(f"[serve] {S._ts()} req #{req_id} arrived "
             f"(thread={threading.current_thread().name}), "
             f"prompt {ids.shape[1]} tok",
             file=sys.__stdout__, flush=True)
        with S.STATE["lock"]:
            t0 = time.time()
            prefix_cache_in = S.STATE["prefix_cache"]
            # BUG FIX: spec_decode.py's [prefill] lines read this via
            # _REQ_ID.get() (a contextvar) to tag their log output with
            # the request they belong to -- added alongside this line,
            # but the .set() call to actually populate it was missing,
            # so every [prefill] line printed "req #0" regardless of the
            # real request, silently defeating the whole point of the
            # feature. Set it once per request, right where this request
            # starts genuinely running (inside the lock, not at arrival)
            # so it matches "which request is the model working on RIGHT
            # NOW" for a reader correlating [prefill] lines against
            # [serve] req #N arrived timestamps.
            from qv_band.decode.spec_decode import _REQ_ID
            _REQ_ID.set(req_id)
            # The generated text reaches ss.write through run_generate(on_text=...). (It used to be captured by
            # swapping the process-wide sys.stdout for a shim: fragile -- a swap read outside this lock once left
            # a dead request's shim as everyone's stdout, and any print() during a generation was taken for
            # generated text.)
            try:
                # abort_check: stop if the client disconnected OR live CUDA
                # memory reached --max-memory-gb. Polled every step (not
                # only when a token is emitted, which is all ss.write's
                # _ClientGone catches).
                # A raised exception here (_ClientGone, a CUDA OOM, ...)
                # means speculative_generate's own return_cache path never
                # ran -- but the cache object it was mutating in place
                # (STATE["prefix_cache"].past, the SAME object just passed
                # in) may already reflect a partial run the OLD handle's
                # token_ids does not account for. Reusing that stale handle
                # next time would under-count what the cache actually
                # holds and duplicate tokens into it. Invalidate up front;
                # a successful run overwrites this with the fresh handle on
                # the next line, so the drop only sticks on that failure
                # path (via the except/finally below, since this line does
                # not execute again after an exception).
                S.STATE["prefix_cache"] = None
                kv_stats: dict = {}
                out, st, S.STATE["prefix_cache"] = S.run_generate(
                    bundle, drafter, ids, rp, stream=True,
                    abort_check=lambda: _peer_gone() or S.vram_abort_check(),
                    prefix_cache=prefix_cache_in, return_cache=True,
                    kv_stats=kv_stats, on_heartbeat=_ping, on_text=ss.write)
            except S._ClientGone:
                # Normal cancellation, not an error -- nothing to send, since
                # the socket that would carry it is what just went away.
                print(f"[serve] {S._ts()} client disconnected after "
                      f"{time.time()-t0:.1f}s, generation aborted", flush=True)
                return
            except Exception as e:
                # Any OTHER failure inside run_generate (a bad --kv-cache/
                # --kv-tau-threshold combination raising ValueError from
                # kv_band.py's codebook-overflow check, a CUDA error, ...)
                # used to propagate straight out of _stream() uncaught --
                # past do_POST's own try/except (which only wraps the
                # non-streaming path; _stream() returns directly from
                # do_POST's call site, outside that block) and up into
                # BaseHTTPRequestHandler's own handling, which the client
                # sees as a bare dropped/broken connection with no error
                # body at all. Confirmed: a real tau overflow
                # produced exactly this -- Cline got nothing usable, and
                # from its side that looks identical to the body-timeout
                # symptom this SSE path already works around elsewhere.
                # This must never take the whole server down (a threading
                # HTTP server survives one handler's uncaught exception
                # regardless, but "survives by accident" is not the same
                # as "the client gets told what happened") -- send it as a
                # clean SSE error object instead, same shape a well-behaved
                # OpenAI-compatible server uses, so Cline at least sees a
                # real error instead of a silent hang/drop.
                traceback.print_exc()
                sse_out.error(e)
                return
            finally:
                S._maybe_empty_cache(bundle)
            dt = time.time() - t0

        S.STATE["served"] += 1
        S.STATE["tokens"] += len(out)

        # Why generation ended, in OpenAI terms. `ss.stopped` is a
        # server-side stop-string hit; otherwise it is "length" iff the
        # decode loop ran out its budget, else "stop" (EOS). rp is the
        # resolved GenParams for this request -- max_output_tokens is the
        # real budget speculative_generate used.
        ss.finish(len(out), rp.max_output_tokens, int(ids.shape[1]), include_usage,
                  S._timings(st, len(out), dt, int(ids.shape[1])))
        line = (f"[serve] {S._ts()} req #{req_id} streamed {len(out)} tok in {dt:.1f}s "
                f"({len(out)/dt:.2f} tok/s)")
        _tm = S._timings(st, len(out), dt, int(ids.shape[1]))
        if S._timings_str(_tm):
            line += "  |  " + S._timings_str(_tm)
        # Same acceptance-rate line chat.py prints -- _stream() never had
        # it, so a slow streaming request (the path Cline actually hits)
        # gave no way to tell "drafter barely accepting, mostly falling
        # back to plain decode" apart from any other slowdown. draft_k is
        # rp's resolved value (0 disables drafting; st.proposed is then 0
        # too, so the condition also covers that case without a separate
        # check).
        if rp.draft_k > 0 and st.proposed:
            line += (f"  |  drafted {st.proposed}, accepted {st.accepted} "
                     f"({st.acceptance:.0%}), {st.emitted/max(1,st.steps):.2f} tok/step, "
                        f"{st.steps} steps: draft {st.t_draft:.1f}s verify {st.t_verify:.1f}s")
        else:
            line += "  |  drafting: NONE (draft_k=0 or nothing proposed)"
        # Prompt size drives prefill cost, which this line otherwise hides
        # inside the same wall-clock total as decode -- a long prompt with
        # a normal tok/s decode rate can still look "slow overall" for a
        # reason that has nothing to do with drafting or the KV codec.
        # prefix_reused is an UPPER BOUND (the saved cache's whole token
        # count, not the actual matched length -- that needs cache_config,
        # internal to spec_decode.py) on how much of THIS prompt could have
        # skipped prefill by reusing the prior turn's cache; 0 means either
        # no prior cache existed or nothing could be shared.
        prefix_reused = len(prefix_cache_in.token_ids) if prefix_cache_in else 0
        line += f"  |  prompt {ids.shape[1]} tok, prefix_reused<={prefix_reused}"
        if kv_stats.get("ratio_vs_dense") is not None:
            line += (f"  |  KV {kv_stats['total_bytes']/2**20:.1f} MB "
                     f"({100*kv_stats['ratio_vs_dense']:.1f}% of dense)")
        print(line, file=sys.__stdout__, flush=True)
