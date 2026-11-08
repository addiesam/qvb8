"""Ripple's chat output format (port of the checkpoint's inference/server/ripple_format.py, no SGLang).

The model generates the layout chat_template.jinja renders:

    <think>\\n{reasoning}\\n</think>\\n{content}
    <think>\\n{reasoning}\\n</think>\\n\\n<tool_call>\\n{"name": "f", "arguments": {...}}\\n</tool_call>

Differences from Qwen that the server must respect:
  * the generation prompt never opens <think> (enable_thinking is ignored by the template); output is reasoning
    ONLY when it starts with <think> -- the model decides per request;
  * tool calls are Hermes JSON inside <tool_call>, not Qwen's <function=...><parameter=...> XML;
  * text after the first tool call is dropped (as the reference parser does).

The server consults model._qv_chat_format (set by runtime.install) in parse_tool_calls, _strip_think and the
streaming writer; a model without it keeps the server's Qwen behaviour.
"""
from __future__ import annotations

import json
import re
import time

THINK_START, THINK_END = "<think>", "</think>"
_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _drop_newline(text: str, lead: bool) -> str:
    if lead:
        return text[1:] if text.startswith("\n") else text
    return text[:-1] if text.endswith("\n") else text


class StreamGate:
    """Decides from the first generated characters whether the output opens with <think>.

    feed(piece) -> (text_to_pass_on, decision) where decision is None while undecided, else True (reasoning;
    the <think> tag and its separator newline are removed) or False (plain content)."""

    # The separators the template puts around the reasoning ("<think>\n", "\n</think>\n") are not part of the
    # reasoning or the content (the reference parser drops them too), so the stream drops them as well:
    #   lead  -> drop one "\n" after <think>;   think -> drop one "\n" before </think> (held back until it is known);
    #   after -> drop one "\n" after </think>;  content -> pass through.
    _CLOSE = "\n" + THINK_END

    def __init__(self):
        self.decided: bool | None = None
        self._held = ""
        self._mode = None

    def feed(self, piece: str) -> tuple[str, bool | None]:
        if self.decided is None:
            self._held += piece
            if self._held.startswith(THINK_START):
                self.decided, self._mode = True, "lead"
                rest, self._held = self._held[len(THINK_START):], ""
                return self._run(rest), True
            if THINK_START.startswith(self._held):
                return "", None                       # could still become <think>
            self.decided, self._mode = False, "content"
            out, self._held = self._held, ""
            return out, False
        return self._run(piece), self.decided

    def _run(self, text: str) -> str:
        out = ""
        while text:
            if self._mode == "lead":
                text, self._mode = _drop_newline(text, lead=True), "think"
            elif self._mode == "think":
                buf = self._held + text
                text, self._held = "", ""
                i = buf.find(THINK_END)
                if i >= 0:
                    out += _drop_newline(buf[:i], lead=False) + THINK_END
                    text, self._mode = buf[i + len(THINK_END):], "after"
                    continue
                hold = next((n for n in range(min(len(buf), len(self._CLOSE) - 1), 0, -1)
                             if self._CLOSE.startswith(buf[-n:])), 0)
                out += buf[:len(buf) - hold]
                self._held = buf[len(buf) - hold:]
            elif self._mode == "after":
                text, self._mode = _drop_newline(text, lead=True), "content"
            else:
                out, text = out + text, ""
        return out

    def take(self) -> str:
        """Text still held at end of stream: an undecided short output, or the tail of an unclosed reasoning
        block held back while it could still turn into the closing tag."""
        out, self._held = self._held, ""
        if self.decided is None:
            self.decided = False
        return out


class RippleChatFormat:
    name = "ripple"

    def split(self, text: str) -> tuple[str | None, str, bool]:
        """-> (reasoning or None, content, truncated_inside_reasoning)."""
        if not text.startswith(THINK_START):
            return None, text, False
        body = _drop_newline(text[len(THINK_START):], lead=True)
        reasoning, end, content = body.partition(THINK_END)
        if not end:
            return reasoning, "", True
        return _drop_newline(reasoning, lead=False), _drop_newline(content, lead=True), False

    def strip_reasoning(self, text: str) -> str:
        _, content, truncated = self.split(text)
        if truncated:
            return ("[thinking was cut off before an answer was produced -- try again with a larger "
                    "max_tokens]")
        return content.strip()

    def parse_tool_calls(self, text: str) -> tuple[str, list[dict]]:
        """Hermes JSON calls. Content = text before the first call (minus the template's separator newline);
        anything after the first call is dropped, as in the reference parser. A block that is not valid JSON
        is left in the content."""
        first = text.find("<tool_call>")
        if first < 0:
            return text.strip(), []
        calls: list[dict] = []
        for body in _TOOL_RE.findall(text[first:]):
            try:
                obj = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                return text.strip(), []
            if not isinstance(obj, dict) or "name" not in obj:
                return text.strip(), []
            args = obj.get("arguments", {})
            calls.append({
                "id": f"call_{len(calls)}_{int(time.time() * 1000)}",
                "type": "function",
                "function": {"name": obj["name"],
                             "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)},
            })
        if not calls:
            return text.strip(), []
        return _drop_newline(text[:first], lead=False).strip(), calls

    def new_stream_gate(self) -> StreamGate:
        return StreamGate()

    @staticmethod
    def prepare_messages(msgs: list[dict]) -> list[dict]:
        """Ripple's chat template concatenates message.content as a string; OpenAI clients (Cline) may send it as a
        list of parts ([{"type": "text", "text": ...}, ...]), which made the template raise "can only concatenate
        str (not "list") to str" and the request fail. Text parts are joined with a newline; Ripple is served
        text-only, so a non-text part (image) becomes a visible marker instead of being dropped silently."""
        out = []
        for m in msgs:
            c = m.get("content")
            if isinstance(c, list):
                parts = []
                for p in c:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                        parts.append(str(p.get("text", "")))
                    elif isinstance(p, dict):
                        parts.append(f"[{p.get('type', 'non-text')} omitted: this model is served text-only]")
                m = {**m, "content": "\n".join(parts)}
            out.append(m)
        return out
