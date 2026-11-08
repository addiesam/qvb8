"""N-gram / prompt-lookup drafter: speculation with no draft model at all.

The speculative harness in spec_decode.py only needs something that proposes
k plausible next tokens; it does not care where they come from, because the
base model verifies every one and the emitted stream stays token-identical to
greedy decoding.  So the cheapest possible drafter is to look for a place
where the last few tokens already occurred, and propose whatever followed.

This costs no weights and no matmuls -- a dict lookup against the tokens
generated so far plus the prompt.  It cannot help on genuinely novel text,
but LLM output is full of exact repetition (names, quoted spans, code,
list scaffolding, the prompt's own wording echoed back), and every hit turns
one weight sweep into several tokens.

Matching is longest-context-first: try the last `max_ctx` tokens, then
shorter, and take the most recent occurrence, which tracks the current
context better than the first occurrence does.
"""
from __future__ import annotations

from collections import defaultdict


class NGramDrafter:
    """Propose continuations by finding where the recent context repeated."""

    def __init__(self, max_ctx: int = 3, min_ctx: int = 2):
        self.max_ctx = max_ctx
        # A 1-token context matches almost anywhere and is close to a random
        # guess; on this model a rejected draft can force a cache rebuild, so
        # the floor is 2 tokens of agreement before anything is proposed.
        self.min_ctx = max(2, min_ctx)
        self.tokens: list[int] = []
        # ctx tuple -> list of positions where it ends, most recent last
        self._idx: dict[tuple, list[int]] = defaultdict(list)
        self.hidden = None          # unused; kept for interface parity
        # Adaptive backoff: on text with no repetition every draft is wasted
        # work, so consecutive misses suppress drafting for a while.  One
        # probe every `_cooldown` steps is enough to notice when the text
        # becomes repetitive again (a list, a quoted span, a code block).
        self._misses = 0
        self._skip = 0
        self.miss_limit = 1
        self.cooldown = 32

    def reset(self, tokens: list[int]) -> None:
        self.tokens = list(tokens)
        self._idx.clear()
        for i in range(len(self.tokens)):
            self._index_at(i)

    def _index_at(self, i: int) -> None:
        for n in range(self.min_ctx, self.max_ctx + 1):
            if i + 1 >= n:
                self._idx[tuple(self.tokens[i + 1 - n:i + 1])].append(i)

    def append(self, tok: int) -> None:
        self.tokens.append(tok)
        self._index_at(len(self.tokens) - 1)

    def propose(self, last_token: int, k: int, temperature: float = 0.0,
                top_k: int = 0, top_p: float = 1.0) -> tuple[list[int], None]:
        """Return (up to k tokens, None): the n-gram drafter has no
        distribution, so its proposals are one-hot and it never supplies
        logits.  temperature/top_k/top_p are accepted (and ignored) so the
        caller can drive every drafter through the same propose() call.

        Longest context first, falling back to shorter ones: with cache
        rollback working a rejected draft costs only the wasted positions of
        one batched forward, so proposing on a weaker match is worth the risk.
        """
        if self._skip > 0:
            self._skip -= 1
            return [], None
        n_tok = len(self.tokens)
        # Longest match only.  A short match is close to a coin flip, and a
        # rejected draft can force a cache rebuild that costs more than the
        # tokens a lucky hit would have saved -- measured 0.39 tok/s with
        # loose matching against 0.63 for plain greedy on non-repetitive text.
        # Requiring the full context keeps drafting to places the text really
        # does repeat, where acceptance is ~90%.
        # Longest context only.  A rejection forces a full-prefix rebuild,
        # which costs far more than a hit saves, so precision beats recall
        # here: propose only where the text demonstrably repeats.
        for n in (self.max_ctx,):
            if n > n_tok:
                continue
            hits = self._idx.get(tuple(self.tokens[n_tok - n:]))
            if not hits:
                continue
            for pos in reversed(hits):
                if pos >= n_tok - 1:
                    continue
                out = self.tokens[pos + 1:pos + 1 + k]
                if out:
                    return out, None
        return [], None

    def feedback(self, n_accepted: int, n_proposed: int) -> None:
        """Told by the decoder how the last draft fared, to drive backoff."""
        if n_proposed == 0:
            return
        if n_accepted == 0:
            self._misses += 1
            if self._misses >= self.miss_limit:
                self._skip = self.cooldown
                self._misses = 0
        else:
            self._misses = 0
            self._skip = 0
