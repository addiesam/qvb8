"""Inkling-2's chat output format, mapped onto the server's contract (see models/ripple/chat_format.py).

The model writes channels as special tokens:
    <|message_model|><|content_thinking|>REASONING<|end_message|>
    <|message_model|><|content_text|>ANSWER<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name": ..., "args": {...}}<|end_message|>
    <|content_model_end_sampling|>                                   (stop)

The server decodes with skip_special_tokens=True, which would erase every channel boundary. InklingTokenizer
(a thin wrapper of the checkpoint's tokenizer) decodes the channel tokens as one-character markers instead,
and InklingChatFormat turns the marked text into what the server expects:
    split / strip_reasoning      reasoning and answer of a finished text
    new_stream_gate()            streaming: reasoning text, then "</think>", then the answer (tool calls are
                                 kept as <tool_call>{json}</tool_call> for parse_tool_calls)
    parse_tool_calls             -> OpenAI tool_calls (arguments as a JSON string)
    prepare_messages             OpenAI tool_call arguments (JSON strings) -> objects, as the template requires
The wrapper also gives apply_chat_template a default reasoning_effort (the template's own default is 0.9).
"""
from __future__ import annotations

import json
import time

MSG, END = "\x1e", "\x03"                       # <|message_model|>, <|end_message|>
THINK, TEXT, TOOL = "\x02T", "\x02C", "\x02J"   # channel openers
_MARK = {"<|message_model|>": MSG, "<|end_message|>": END, "<|content_thinking|>": THINK,
         "<|content_text|>": TEXT, "<|content_invoke_tool_json|>": TOOL, "<|content_invoke_tool_text|>": TOOL}
THINK_END = "</think>"
STOP_TOKENS = ("<|content_model_end_sampling|>", "<|endoftext|>")


class InklingTokenizer:
    """Delegates everything to the checkpoint tokenizer; decode keeps the channel tokens as markers."""

    def __init__(self, tok, effort=None):
        object.__setattr__(self, "_tok", tok)
        object.__setattr__(self, "_effort", effort)
        object.__setattr__(self, "_mark_ids", {tok.convert_tokens_to_ids(t): m for t, m in _MARK.items()})

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __len__(self):
        return len(self._tok)

    def __call__(self, *a, **kw):
        return self._tok(*a, **kw)

    def decode(self, ids, skip_special_tokens=False, **kw):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        out, run = [], []
        for t in ids:
            m = self._mark_ids.get(int(t))
            if m is None:
                run.append(t)
                continue
            if run:
                out.append(self._tok.decode(run, skip_special_tokens=skip_special_tokens, **kw))
                run = []
            out.append(m)
        if run:
            out.append(self._tok.decode(run, skip_special_tokens=skip_special_tokens, **kw))
        return "".join(out)

    def apply_chat_template(self, conversation, *a, **kw):
        if self._effort is not None:
            kw.setdefault("reasoning_effort", self._effort)
        kw.pop("enable_thinking", None)                  # Inkling: thinking is set by reasoning_effort
        return self._tok.apply_chat_template(InklingChatFormat.prepare_messages(conversation), *a, **kw)


def _segments(text):
    """Marked text -> [(channel, name, body, closed)] in order; channel in {"think", "text", "tool", "raw"}."""
    segs, i, n = [], 0, len(text)
    while i < n:
        if text[i] == MSG:
            i += 1
            j = text.find("\x02", i)
            if j < 0:                                    # header not complete yet
                segs.append(("pending", text[i:], "", False))
                break
            name, kind = text[i:j], text[j:j + 2]
            i = j + 2
            k = text.find(END, i)
            body, closed = (text[i:], False) if k < 0 else (text[i:k], True)
            i = n if k < 0 else k + 1
            ch = {THINK: "think", TEXT: "text", TOOL: "tool"}.get(kind, "raw")
            segs.append((ch, name, body, closed))
        else:
            k = text.find(MSG, i)
            k = n if k < 0 else k
            body = text[i:k].replace(END, "")
            if body:
                segs.append(("raw", "", body, True))
            i = k
    return segs


def _tool_block(name, body):
    try:
        obj = json.loads(body)
        nm, args = obj.get("name", name), obj.get("args", obj.get("arguments", {}))
        body = json.dumps({"name": nm, "arguments": args}, ensure_ascii=False)
    except (ValueError, AttributeError):
        pass
    return f"<tool_call>{body}</tool_call>"


class StreamGate:
    """feed(marked piece) -> (text, decision). Output: reasoning text, then THINK_END, then the answer; tool
    calls as <tool_call>{json}</tool_call>. decision: None until the first channel is known, True when it is
    reasoning, False when the reply opens with an answer or a tool call (no THINK_END is ever written then)."""

    def __init__(self):
        self.decided = None
        self._buf = ""                 # everything received (channels are parsed from it as a whole)
        self._sent = 0                 # chars of the rendered output already returned

    def _render(self, final=False):
        out, closed_think = "", False
        for ch, name, body, closed in _segments(self._buf):
            if ch == "pending":
                break
            if ch == "think":
                if self.decided is None:
                    self.decided = True
                if not closed_think and self.decided:     # reasoning after the answer has begun is not streamed:
                    out += body                            # it would land in the answer (split() still keeps it)
                continue
            if self.decided is None:
                self.decided = False
            if self.decided and not closed_think:
                out += THINK_END
                closed_think = True
            if ch == "tool":
                if not closed and not final:
                    break                                 # a tool call is released whole
                out += _tool_block(name, body)
            else:
                out += body
        return out

    def feed(self, piece):
        self._buf += piece
        out = self._render()
        new, self._sent = out[self._sent:], len(out)
        return new, self.decided

    def take(self):
        out = self._render(final=True)
        new, self._sent = out[self._sent:], len(out)
        if self.decided is None:
            self.decided = False
        return new


class InklingChatFormat:
    name = "inkling"

    def split(self, text):
        """-> (reasoning or None, answer, truncated_inside_reasoning)."""
        reasoning, answer, truncated = [], [], False
        segs = _segments(text)
        for ch, name, body, closed in segs:
            if ch == "think":
                reasoning.append(body)
                truncated = not closed
            elif ch == "tool":
                answer.append(_tool_block(name, body))
                truncated = False
            elif ch in ("text", "raw"):
                answer.append(body)
                truncated = False
        if not reasoning:
            return None, "".join(answer), False
        return "".join(reasoning), "".join(answer), truncated and not answer

    def strip_reasoning(self, text):
        r, content, truncated = self.split(text)
        if truncated:
            return "[thinking was cut off before an answer was produced -- try again with a larger max_tokens]"
        return content.strip()

    def parse_tool_calls(self, text):
        """Text holding <tool_call>{"name", "arguments"}</tool_call> blocks (from the gate / split) or raw marked
        text -> (content, OpenAI tool_calls)."""
        if MSG in text or "\x02" in text:
            _, text, _ = self.split(text)
        calls, content, i = [], [], 0
        while True:
            a = text.find("<tool_call>", i)
            if a < 0:
                content.append(text[i:])
                break
            b = text.find("</tool_call>", a)
            if b < 0:
                content.append(text[i:])
                break
            content.append(text[i:a])
            try:
                obj = json.loads(text[a + len("<tool_call>"):b])
                args = obj.get("arguments", {})
                calls.append({"id": f"call_{len(calls)}_{int(time.time() * 1000)}", "type": "function",
                              "function": {"name": obj["name"], "arguments": args if isinstance(args, str)
                                           else json.dumps(args, ensure_ascii=False)}})
            except (ValueError, KeyError, TypeError, AttributeError):
                content.append(text[a:b + len("</tool_call>")])
            i = b + len("</tool_call>")
        return "".join(content).strip(), calls

    def new_stream_gate(self):
        return StreamGate()

    @staticmethod
    def prepare_messages(msgs):
        """OpenAI -> template: tool-call arguments become objects; developer -> system; None content -> ""."""
        out = []
        for m in msgs or []:
            m = dict(m)
            if m.get("role") == "developer":
                m["role"] = "system"
            if m.get("tool_calls"):
                tcs = []
                for tc in m["tool_calls"]:
                    tc = json.loads(json.dumps(tc))
                    fn = tc.get("function", {})
                    a = fn.get("arguments")
                    if isinstance(a, str):
                        try:
                            fn["arguments"] = json.loads(a) if a.strip() else {}
                        except ValueError:
                            fn["arguments"] = {"raw": a}
                    tcs.append(tc)
                m["tool_calls"] = tcs
            if m.get("content") is None:
                m["content"] = ""
            out.append(m)
        return out
