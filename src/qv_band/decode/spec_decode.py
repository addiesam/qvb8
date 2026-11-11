"""Self-speculative greedy decoding with the checkpoint's own MTP head.

Why this is the lever that matters on CPU.  Decoding is bound by weight
bandwidth: every token streams the whole 6.74 GB of packed weights, and this
machine delivers ~7.16 GB/s multithreaded, so ~1 tok/s is a ceiling no kernel
can beat.  Speculation changes the denominator instead of the rate -- the base
model verifies k+1 positions in ONE forward, reading the weights once for up
to k+1 emitted tokens.  Measured on this box:

    verify  1 token   1.45 s   ->  1.45 s/token
    verify  9 tokens  3.10 s   ->  0.34 s/token     4.2x if all accepted

The draft is cheap for the same reason: the MTP head is a single transformer
layer (205 MB packed) against the base model's 32 layers.

Correctness: at temperature 0 a draft token is kept only where it equals the
base model's own greedy argmax at that position, so the emitted stream is
token-identical to plain greedy decoding.  At temperature > 0 verification is
real speculative sampling -- each drafted token is accepted with probability
min(1, p_target/p_draft) and a rejection draws the emitted token from the
residual distribution max(0, p_target - p_draft), so the marginal distribution
of the emitted stream is exactly the base model's own sampling distribution
(Leviathan et al. 2023 / Chen et al. 2023).  A wrong draft costs throughput
but never changes the output distribution.

    python -m qv_band.decode.spec_decode "The capital of France is" --draft 8 -n 40
    python -m qv_band.decode.spec_decode "..." --draft 8 --compare
"""
from __future__ import annotations

import argparse
import contextvars
import math
import os
import types
import random
import sys
import threading
import time

import torch
from transformers import DynamicCache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# qv_band is a standalone library: the CLI's default model/pack paths come from
# the environment, not from a sibling project.  Callers that import
# speculative_generate() pass explicit paths and never touch these.
DEFAULT_MODEL = os.environ.get("QV_BAND_MODEL_PATH", "")
LLM_PACK_DIR = os.environ.get("QV_BAND_PACK_DIR") or None

# Purely a log-correlation tag (see llm_apps/serve.py's _next_req_id):
# lets an operator's console distinguish which HTTP request a given
# [prefill]/[serve] line belongs to, without threading a new parameter
# through run_generate -> speculative_generate's whole call chain. A
# contextvar (not a plain module global) so it is safe if this module is
# ever driven by more than one thread/request concurrently -- each
# caller's .set() is only visible within its own call stack, never
# clobbered by a different thread's concurrent request. Default 0 means
# "no id set" (a caller outside serve.py, e.g. the CLI __main__ below).
_REQ_ID: contextvars.ContextVar[int] = contextvars.ContextVar("_REQ_ID", default=0)

# Chunked prefill (llama.cpp's --ubatch-size / SGLang's chunked_prefill_size):
# a long prompt is fed through the model PREFILL_CHUNK tokens at a time
# instead of one giant forward, so peak VRAM for a prefill is bounded by
# one chunk's activations + whatever the cache has accumulated so far, not
# by the whole prompt at once. This is what actually prevents the OOM a
# long Cline conversation (or any large context) causes in ONE forward --
# --max-memory-gb's guards (memlimit.py) can only stop the PROCESS from
# exceeding a ceiling; they cannot make a single oversized forward fit.
# Chunking also lets the per-step abort_check (disconnect / VRAM poll)
# actually run DURING a long prefill instead of only after it completes.
# 512 matches llama.cpp server's default n_ubatch; override via env for
# tuning without a code change.
PREFILL_CHUNK = int(os.environ.get("QV_BAND_PREFILL_CHUNK", "512"))
# ContinuousBatchEngine, pure-attention models: pack each step into one unpadded row, and feed a long prompt
# one chunk per step so running requests keep decoding meanwhile (see _forward_packed / _prefill_admitted).
PACKED_ENABLED = os.environ.get("QV_BAND_PACKED", "1") != "0"
# opt-in: measured a net loss on the 65B (code 20.4 -> 17.6 tok/s, prose unchanged) -- full block wins
ADAPTIVE_K = os.environ.get("QV_BAND_ADAPTIVE_K", "0") == "1"
# Verify-token budget for a packed step (v1). A packed step concatenates every slot's [last + drafts]
# into ONE row, so the target's linears see M = total tokens. 16 is the largest M the tensor-core qvb8 kernel
# serves (~485 ms / 80 layers on Ripple); past it the bf16 path decodes the whole weight matrix (~3x the
# traffic). With several drafting slots each gets k = budget // n - 1 (2 slots: 7 each, 3: 4, ...).
# 0 = no budget (every slot drafts the engine's k). Opt-in: measured on the dev GPU (Ripple, 2 concurrent code
# requests, 300 tok each) aggregate 15.2 tok/s with the budget vs 16.5 without, so the default is off.
ROW_BUDGET = int(os.environ.get("QV_BAND_ROW_BUDGET", "0"))
_PROFILE_STEPS = [int(os.environ.get("QV_BAND_PROFILE_STEPS", "0"))]
_PROFILE_PREFILL = [int(os.environ.get("QV_BAND_PROFILE_PREFILL", "0"))]   # profile N prefill chunk forwards
# Prefill chunk while other requests are decoding: each decode step waits for one chunk, so keep it small
# enough for streaming to stay responsive (1024 tok ~ 3-4 s on the 65B) -- prefill stays compute bound.
INTERLEAVE_CHUNK = int(os.environ.get("QV_BAND_INTERLEAVE_CHUNK", "1024"))

# Minimum real wall-clock gap between on_heartbeat() calls -- see
# speculative_generate's docstring for on_heartbeat. Cline's own body-read
# timeout is undici's 300s default (confirmed: Cline sets bodyTimeout=0 for
# its own OpenAI-compatible provider path, but that override does not
# reliably apply to every provider path -- e.g. a local/custom endpoint --
# so the 300s default can still bite). 10s keeps well under any client timeout
# seen so far (the Cline BodyTimeoutError on a long buffered tool call); tune via
# the env var if a shorter client timeout is ever confirmed.
HEARTBEAT_INTERVAL_S = float(os.environ.get("QV_BAND_HEARTBEAT_INTERVAL_S", "10.0"))


class _IncStreamer:
    """Incremental detokeniser for stream=True. The naive approach does
    `tok.decode(out_ids)` EVERY step and prints the new suffix -- O(seq)
    tokenizer work per token, O(seq^2) over a generation, which is why a
    200-token reply visibly slows from ~3 tok/s to <1 as it grows. This
    keeps a small pending-id cache (HF TextIteratorStreamer's trick, also
    how llama.cpp's server streams): decode ONLY the cache, emit the
    newly-stable prefix, and flush the cache once a token clearly ends a
    word/codepoint so the cache never grows. Cost per token is O(1).

    "clearly ends a word" = the decoded cache ends in an ASCII space or a
    CJK char (which is always its own word) and not in U+FFFD (a
    half-emitted multi-byte codepoint a later token will complete).

    Module-level (not nested inside speculative_generate) so
    ContinuousBatchEngine and any other caller streaming MULTIPLE
    independent token sequences at once (llm_apps/serve.py's batch
    scheduler: one instance per request/slot, since each has its own
    pending-id cache) can reuse the exact same, already-tested
    incremental-decode logic instead of a hand-rolled duplicate.
    """
    def __init__(self, tok):
        self._tok = tok
        self._cache: list[int] = []
        self._printed = 0            # chars of decode(self._cache) emitted

    @staticmethod
    def _flushable(text: str) -> bool:
        if not text or text.endswith("�"):
            return False
        last = text[-1]
        return last == " " or ord(last) > 0x2000   # space or CJK/full-width

    def feed(self, new_ids: list[int]) -> str:
        if not new_ids:
            return ""
        self._cache.extend(new_ids)
        text = self._tok.decode(self._cache, skip_special_tokens=True)
        if text.endswith("�"):
            emit = text[self._printed:-1]          # hold the dangling half
            self._printed = len(text) - 1
            return emit
        emit = text[self._printed:]
        if self._flushable(text):
            self._cache = []
            self._printed = 0
        else:
            self._printed = len(text)
        return emit

    def finish(self) -> str:
        if not self._cache:
            return ""
        text = self._tok.decode(self._cache, skip_special_tokens=True)
        return text[self._printed:]                # emit whatever is held


class _Heartbeat:
    """Fire `callback` at most once every HEARTBEAT_INTERVAL_S of real
    wall-clock time, regardless of how often `.tick()` is called or from
    where -- decouples the ping cadence from chunk size, token count, or
    any other unit of work, since none of those reliably bound wall-clock
    time (a chunk or a decode step can take very different amounts of
    real time depending on GPU load). No-op if callback is None, so a
    caller that does not want heartbeats pays only the branch check."""
    __slots__ = ("_cb", "_last")

    def __init__(self, callback):
        self._cb = callback
        self._last = time.time()

    def tick(self) -> None:
        if self._cb is None:
            return
        now = time.time()
        if now - self._last >= HEARTBEAT_INTERVAL_S:
            self._last = now
            self._cb()

# MTP draft head context window (positions). The head is a full self-attention
# layer trained to attend over the preceding positions; drafting used to call
# it one row at a time, so it saw only itself. 0 disables the context and
# restores the old behaviour. See MTPHeadCUDA.forward_ctx for the measurements.
MTP_CTX_WINDOW = int(os.environ.get("QV_BAND_MTP_CTX", "256"))
# Confidence-gated draft depth for the context-enabled head (greedy only).
# Each further draft token costs ~10 ms (one MTP step + a wider verify) and an
# accepted token is worth ~40 ms, so a token is only worth drafting when the
# chance it is accepted is above ~25%. The head's confidence is well
# calibrated: along a chain, the PRODUCT of the per-token softmax maxima is
# accepted ~86% of the time above 0.7, ~47% at 0.5-0.7, ~29% at 0.3-0.5, ~16% at
# 0.2-0.3 and <= 8% below 0.2. Drafting stops at the first token whose
# cumulative confidence falls below this. 0 = always draft the full k.
MTP_MIN_CUM = float(os.environ.get("QV_BAND_MTP_MIN_CUM", "0.35"))
# Adaptive drafting in ContinuousBatchEngine, OFF by default. A slot whose
# smoothed per-token acceptance falls below SPEC_MIN_ACCEPT stops drafting
# for a while (backoff doubles per failed probe). It was meant to save the
# replay forward a rejected draft costs, but measured on the 27B at 6.6k
# context it LOST throughput: tokens/step fell 2.54 -> 1.83 and decode
# 5.4 -> 4.7 tok/s even though acceptance was ~77% overall, because
# per-step acceptance is bursty and each pause forfeits the ~2.5 tokens a
# step gains. Kept for experimentation only: QV_BAND_SPEC_MIN_ACCEPT=0.5.
SPEC_MIN_ACCEPT = float(os.environ.get("QV_BAND_SPEC_MIN_ACCEPT", "0"))
SPEC_EMA = 0.75
SPEC_BACKOFF_MIN = 8
SPEC_BACKOFF_MAX = 128


def _probs(logits: torch.Tensor, temperature: float, top_k: int,
           top_p: float) -> torch.Tensor:
    """The exact distribution _select() would draw from at temperature > 0:
    temperature-scaled, top-k then nucleus (top-p) filtered, softmaxed -- the
    same order transformers' own LogitsProcessor pipeline applies for
    TopKLogitsWarper -> TopPLogitsWarper.  Shared by _select(), _accept_prob()
    and _resample_residual() so the sampled token, the acceptance probability
    and the residual all come from identical mass functions."""
    scaled = logits / temperature
    if top_k > 0:
        # torch.sort, not torch.topk. On Intel XPU (Arc/Xe) torch.topk over a
        # vocab-sized 1-D tensor measured 58 SECONDS per call against 0.006 s
        # for a full sort of the same tensor -- a driver/kernel pathology, not
        # an algorithmic cost. topk dominated everything: 213 s of a 226 s
        # sampled run, versus 7 s for all 1992 weight-decode kernel calls.
        # A full sort is O(n log n) where top-k is O(n), but at these sizes
        # the constant factor is what matters, and CUDA is unaffected either
        # way (sort of 150 K floats is sub-millisecond on both).
        kth = torch.sort(scaled, descending=True).values[
            ..., min(top_k, scaled.shape[-1]) - 1, None]
        scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # keep the smallest prefix whose cumulative prob >= top_p; always
        # keep at least the top token even if its own prob already exceeds it
        remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits)
    return torch.softmax(scaled, dim=-1)


def _select(logits: torch.Tensor, temperature: float, top_k: int,
           top_p: float = 1.0) -> int:
    """argmax when temperature is 0 (the greedy path everything else here is
    built around); otherwise multinomial sampling, top-k then nucleus
    (top-p) filtered, temperature-scaled -- see _probs."""
    if temperature <= 0:
        return int(logits.argmax())
    return int(torch.multinomial(_probs(logits, temperature, top_k, top_p), 1))


def _accept_prob(p_target: torch.Tensor, p_draft: torch.Tensor | None,
                 token: int) -> float:
    """P(accept) = min(1, p_target(token) / p_draft(token)), the standard
    speculative-sampling acceptance identity (Leviathan et al. 2023 /
    Chen et al. 2023).

    p_draft=None means the drafter's proposal is a one-hot at `token`
    (n-gram, and MTP at temperature 0): the ratio then reduces to exactly
    p_target(token), so acceptance is a coin flip on the target model's own
    probability for that token."""
    if p_draft is None:
        return float(p_target[token])
    pd = float(p_draft[token])
    if pd <= 0.0:
        return 0.0
    return min(1.0, float(p_target[token]) / pd)


def _resample_residual(p_target: torch.Tensor, p_draft: torch.Tensor | None,
                       rejected: int) -> int:
    """Draw the token emitted after a rejection from the residual
    distribution max(0, p_target - p_draft), renormalised -- the standard
    speculative-sampling correction that keeps the marginal distribution
    exactly the target's.  With p_draft=None the residual is p_target with
    the rejected token's one-hot mass zeroed out.  Falls back to a plain
    target sample if all the mass was on `rejected` (top_k=1, or a draft
    that dominates the target)."""
    if p_draft is None:
        residual = p_target.clone()
        residual[rejected] = 0.0
    else:
        residual = torch.clamp(p_target - p_draft, min=0.0)
    total = float(residual.sum())
    if total <= 0.0 or not math.isfinite(total):
        return int(torch.multinomial(p_target, 1))
    return int(torch.multinomial(residual / total, 1))

def _topk_dist(logits: torch.Tensor, temperature: float, top_k: int, top_p: float):
    """_probs() for every row of `logits` at once, restricted to each row's top_k tokens
    (everything outside them has probability 0 in _probs too): (token ids, probabilities,
    count of logits >= the k-th). The count differs from top_k only on a tie at the k-th
    logit, where _probs keeps every tied token -- the caller then falls back to _probs."""
    vals, idx = torch.topk(logits, top_k, dim=-1)               # sorted, descending
    ties = (logits >= vals[:, -1:]).sum(-1)
    p = torch.softmax(vals.float() / temperature, dim=-1)
    if 0 < top_p < 1.0:
        p = p.masked_fill((p.cumsum(-1) - p) > top_p, 0.0)      # the nucleus, as in _probs
        p = p / p.sum(-1, keepdim=True)
    return idx, p, ties


def _verify_sampled(s, draft, dl, row_logits):
    """Speculative-sampling verification (the temperature > 0 branch of _verify) done on each
    position's top_k tokens and ONE device-to-host copy, or None when that is not exact.

    WHY: _verify ran _probs per drafted position -- two full-vocab (248k) sorts, several softmaxes
    and a GPU sync per accept test. Measured on the 27B that was ~55 ms of a ~260 ms decode step. A
    token outside the top_k has probability 0 under the target, so acceptance and the residual
    only ever involve the top_k tokens; this computes them once for all positions and decides on
    the host. Same distribution, same accept/residual rule."""
    k, V, n = s.top_k, row_logits.shape[-1], len(draft)
    if k <= 0 or k >= V:
        return None
    if dl is not None and any(x is not None for x in dl) and not getattr(s.drafter, "returns_probs", False):
        return None                                      # a draft that still needs _probs() of its logits
    idx, p, ties = _topk_dist(row_logits, s.temperature, k, s.top_p)
    pd = torch.zeros_like(p)                             # the draft's own probability on those tokens
    for i, x in enumerate(dl or ()):
        if x is not None and i < n:
            pd[i] = x[idx[i]].float()
    packed = torch.cat([idx.float(), p, pd, ties.float()[:, None]], dim=1).cpu()
    if bool((packed[:, -1] != k).any()):
        return None
    ids = packed[:, :k].long().tolist()
    tp, dp = packed[:, k:2 * k].tolist(), packed[:, 2 * k:3 * k].tolist()

    def draw(weights) -> int:
        return random.choices(range(k), weights=weights)[0]

    for i, d in enumerate(draft):
        pos = ids[i].index(d) if d in ids[i] else -1
        if pos < 0:
            accept = 0.0                                 # the target gives it no mass
        elif dl is None or dl[i] is None:
            accept = tp[i][pos]
        else:
            accept = 0.0 if dp[i][pos] <= 0.0 else min(1.0, tp[i][pos] / dp[i][pos])
        if random.random() < accept:
            continue
        if dl is None or dl[i] is None:
            residual = list(tp[i])
            if pos >= 0:
                residual[pos] = 0.0
        else:
            residual = [max(0.0, a - b) for a, b in zip(tp[i], dp[i])]
        total = sum(residual)
        weights = residual if total > 0.0 and math.isfinite(total) else tp[i]
        return i, ids[i][draw(weights)]
    return n, ids[n][draw(tp[n])]


# Whether a rejected draft can be undone exactly.
#
# It cannot, on this architecture.  crop() restores the linear-attention
# layers' conv WINDOW, but their recurrent state has already absorbed the
# rejected tokens and there is no inverse for that: measured, the logits after
# a crop differ from the un-speculated path by up to 6.0, which is enough to
# change the emitted token.  So any rejection forces the whole batch to be
# discarded and the cache rebuilt over the committed prefix -- seq[0] plus the
# accepted drafts, both of which were fed to the polluted cache -- and that
# rebuild is verified bit-exact against a clean prefill.  A rebuild wastes the
# step's forward, which is why the drafters are tuned to be conservative, but
# it never corrupts the stream.
#
# Acceptance itself is temperature-independent in the sense that the MTP head
# participates either way: at temperature 0 drafts are one-hot argmax tokens
# verified by exact match against the target's argmax; above 0 the head's real
# per-position distribution is used for the speculative-sampling acceptance
# min(1, p_target/p_draft) described in speculative_generate's docstring.
EXACT_ROLLBACK = False


class PrefixCache:
    """A KV cache handle a caller can save after one speculative_generate()
    call and pass back into the next, to skip re-prefilling whatever text
    is common to both calls -- the standard "prefix caching" a chat/agent
    loop wants (llama.cpp's server does this by default; SGLang's
    RadixAttention generalises it to a shared multi-request tree). This is
    the single-conversation version: one saved cache, reused by exact
    prefix match.

    WHY EXACT PREFIX ONLY, NOT "crop to the longest common prefix":
    the linear-attention (Gated DeltaNet) layers' recurrent/conv state is
    FIXED-SIZE and irreversible -- stock transformers'
    LinearAttentionCacheLayerMixin.crop() is a documented no-op for them
    (a documented no-op for them), so once a token has been fed through
    those layers there is no way to roll their state back to an earlier
    point, only the full-attention K/V can be cropped. A cache saved after
    processing prefix P can therefore only be reused by a NEW request whose
    own prefix is EXACTLY P (then the new request's remainder is appended);
    a request that diverges from P anywhere, or is merely a prefix OF P,
    cannot reuse it at all and must prefill fresh. This mirrors exactly how
    the existing decode-loop rollback (_try_crop) already
    treats these layers -- crop when possible, rebuild-from-scratch
    otherwise -- just applied at the whole-request granularity instead of
    per draft step.

    `token_ids`: the exact prompt+completion token ids the cache has
    consumed, in order -- the ONLY thing checked for reuse.
    `cache_config`: the (qvb8_resident, int8_kv,
    kv_tau, int8_residual_length) tuple the cache was built with. A reuse
    request whose kwargs differ is refused -- an int8 cache cannot serve a
    banded-cache request, so pretending config never matters would silently reuse
    the wrong storage form.
    """
    __slots__ = ("past", "token_ids", "cache_config")

    def __init__(self, past, token_ids: list[int], cache_config: tuple):
        self.past = past
        self.token_ids = token_ids
        self.cache_config = cache_config

    def reusable_prefix_len(self, new_ids: list[int], cache_config: tuple) -> int:
        """How many of THIS cache's tokens are a usable prefix of
        `new_ids` -- either len(self.token_ids) (full reuse, the new
        request continues exactly where this cache left off) or 0 (no
        reuse: config mismatch, or new_ids does not start with every
        token this cache has consumed). Never a partial number -- see the
        class docstring for why a partial match cannot be crop-reused."""
        if cache_config != self.cache_config:
            return 0
        n = len(self.token_ids)
        if n == 0 or n > len(new_ids) or new_ids[:n] != self.token_ids:
            return 0
        return n


def _mem_breakdown(past, drafter) -> str:
    """Allocated per GPU and what the long-context state holds (QV_MEM_DEBUG)."""
    GB = 1e9
    out = [f"cuda:{i} alloc {torch.cuda.memory_allocated(i)/GB:.2f}GB peak {torch.cuda.max_memory_allocated(i)/GB:.2f}GB"
           for i in range(torch.cuda.device_count())]
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)          # peak = since the previous breakdown (one chunk)
    packed = packed_gpu = meta_gpu = fused = resid = 0
    for L in getattr(past, "layers", None) or []:
        for segs in (getattr(L, "_banded_k", None) or []) + (getattr(L, "_banded_v", None) or []):
            for seg in segs:
                for b in seg:
                    packed += b.nbytes()
                    if b.packed.is_cuda:
                        packed_gpu += b.packed.numel()
                    for t in (b.min_s, b.tau_s):
                        if isinstance(t, torch.Tensor) and t.is_cuda:
                            meta_gpu += t.numel() * t.element_size()
        for f in getattr(L, "_fused_slots", None) or []:
            if f is not None and f.ki is not None:
                fused += 2 * f.ki.numel() * f.ki.element_size()
                fused += sum(t.numel() for t in (f.hk, f.hv) if t is not None)
        for r in (getattr(L, "_resid_k", None) or []) + (getattr(L, "_resid_v", None) or []):
            resid += r.numel() * r.element_size()
    d = getattr(drafter, "dflash", drafter)
    draft = sum(b.numel() * b.element_size() for b in (getattr(d, "_kb", None) or []) + (getattr(d, "_vb", None) or [])
                if b is not None)
    out.append(f"kv packed {packed/GB:.2f}GB (gpu {packed_gpu/GB:.2f} meta-gpu {meta_gpu/GB:.3f}) fused-ids {fused/GB:.2f}GB resid {resid/GB:.2f}GB "
               f"draft-kv {draft/GB:.2f}GB")
    if os.environ.get("QV_MEM_DEBUG") in ("2", "3"):  # largest live tensor groups on the main card
        out.append(_mem_groups())
    return "  ".join(out)


_MEM_SEEN: set | None = None


def _mem_groups() -> str:
    """QV_MEM_DEBUG=2: the largest live CUDA tensor groups. =3: only storages new since the previous call
    (the first call, made when the server is ready, sets the baseline)."""
    global _MEM_SEEN
    import collections
    import gc
    agg, seen = collections.Counter(), set()
    new_only = os.environ.get("QV_MEM_DEBUG") == "3"
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.is_cuda and o.device.index == 0:
                st = o.untyped_storage()
                p = st.data_ptr()
                if p in seen or (new_only and _MEM_SEEN is not None and p in _MEM_SEEN):
                    seen.add(p)
                    continue
                seen.add(p)
                agg[(tuple(o.shape), str(o.dtype))] += st.nbytes()
        except Exception:                          # noqa: BLE001
            pass
    first = new_only and _MEM_SEEN is None
    _MEM_SEEN = seen
    if first:
        return "baseline set"
    return (("new: " if new_only else "top: ") + f"{sum(agg.values())/1e6:.0f}MB in {len(agg)} groups; "
            + "; ".join(f"{k[0]} {k[1][6:]} {v/1e6:.0f}MB" for k, v in agg.most_common(12)))


def _rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return float("nan")


class Stats:
    def __init__(self):
        self.steps = self.proposed = self.accepted = self.emitted = 0
        self.rebuilds = 0
        self.t_draft = self.t_verify = 0.0
        self.t_prefill = 0.0         # seconds from request start to the first token's logits
        self.n_prefill = 0           # prompt tokens that actually went through the model (prefix-cache hits excluded)

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    def report(self) -> str:
        return (f"steps={self.steps}  proposed={self.proposed}  "
                f"accepted={self.accepted} ({self.acceptance:.1%})  "
                f"tokens/step={self.emitted/max(1,self.steps):.2f}  "
                f"cache rebuilds={self.rebuilds}\n"
                f"draft {self.t_draft:.1f}s  verify {self.t_verify:.1f}s")


class MTPDrafter:
    """The checkpoint's one-layer MTP head, used as a k-token proposer.

    RoPE inside the head is position-sensitive, so it needs the ABSOLUTE
    sequence position of each draft step, not a position counted from 0
    every call -- reference usage (mtp_cpu.SpeculativeDecoder.draft) passes
    `pos + i` where `pos` is the true position of `last_token`. `reset()`/
    `append()` (called by speculative_generate the same way it drives
    NGramDrafter, for prefix bookkeeping) double as that position counter
    here: `reset(tokens)` after prefill records len(tokens) as the position
    of the next token, and each emitted token advances it by one.
    """

    needs_hidden = True

    def __init__(self, head, embed, lm_head):
        self.head = head
        self.embed = embed
        self.lm_head = lm_head
        self.hidden = None            # base model's last hidden state
        self.pos = 0                  # absolute position of the next token
        # Restricted-vocabulary draft head (see set_vocab_subset): None = full.
        self.sub_head = None
        self.sub_ids = None
        # Head context (see enable_context): window length, the committed K/V
        # of the last `ctx_window` positions, and TRUE inputs not yet run.
        self.ctx_window = 0
        self.ctx_k = None
        self.ctx_v = None
        self._pending = []

    def enable_context(self, window: int = 256) -> bool:
        """Give the head its own attention context over the last `window`
        positions (built from the main model's hidden states). Returns False
        -- leaving the one-row-at-a-time behaviour -- for heads that cannot do
        it (heads without forward_ctx) or when window <= 0."""
        if window <= 0 or not hasattr(self.head, "forward_ctx"):
            return False
        self.ctx_window = int(window)
        return True

    def observe(self, first_pos: int, hiddens: torch.Tensor,
                next_tokens: list[int]) -> None:
        """Record TRUE head inputs: row i is the main model's hidden state at
        sequence position first_pos + i, and next_tokens[i] is the token that
        FOLLOWS that position. The drafter runs them (batched) at its next
        propose(), which also commits their K/V to the context."""
        if not self.ctx_window:
            return
        for i, t in enumerate(next_tokens):
            self._pending.append((first_pos + i, hiddens[i], int(t)))
        if len(self._pending) > self.ctx_window:
            self._pending = self._pending[-self.ctx_window:]

    def _commit_ctx(self, kn: torch.Tensor, vn: torch.Tensor) -> None:
        if self.ctx_k is None:
            self.ctx_k, self.ctx_v = kn, vn
        else:
            self.ctx_k = torch.cat([self.ctx_k, kn], dim=1)
            self.ctx_v = torch.cat([self.ctx_v, vn], dim=1)
        if self.ctx_k.shape[1] > self.ctx_window:
            self.ctx_k = self.ctx_k[:, -self.ctx_window:]
            self.ctx_v = self.ctx_v[:, -self.ctx_window:]

    def _propose_ctx(self, k: int, temperature: float, top_k: int, top_p: float,
                     forced) -> tuple[list[int], list | None]:
        """Draft with context: ONE batched head call over the pending TRUE rows
        (the last row is the depth-1 input), then one row per further depth,
        each attending to the committed context plus the draft-time entries
        (which are discarded -- only true entries ever enter the cache).
        `forced` tokens (e.g. from the n-gram drafter) are fed in place of the
        head's own pick for the first len(forced) depths, so the head's state
        stays consistent with the tokens that will actually be verified."""
        rows, self._pending = self._pending, []
        dev = rows[0][1].device
        H = torch.stack([r[1] for r in rows]).float()
        e = self.embed(torch.tensor([r[2] for r in rows], device=dev)).reshape(len(rows), -1).float()
        pos = torch.tensor([r[0] for r in rows], device=dev)
        past = (self.ctx_k, self.ctx_v) if self.ctx_k is not None else None
        out_h, (kn, vn) = self.head.forward_ctx(e, H, pos, past)
        self._commit_ctx(kn, vn)
        h = out_h[-1:]
        cur = rows[-1][0]
        tk = tv = None
        cum = 1.0
        forced = forced or []
        tokens: list[int] = []
        logits_out: list | None = [] if temperature > 0 else None
        for j in range(k):
            lg = None
            if j < len(forced):
                tid = int(forced[j])
            elif temperature <= 0 and self.sub_head is not None:
                tid = int(self.sub_ids[self.sub_head(h)[0].argmax()])
            else:
                lg = self.lm_head(h)[0]
                if temperature > 0:
                    tid = _select(lg, temperature, top_k, top_p)
                elif MTP_MIN_CUM > 0:
                    conf, tt = torch.softmax(lg.float(), dim=-1).max(0)
                    cum *= float(conf)
                    if cum < MTP_MIN_CUM:
                        break                    # not worth verifying this one
                    tid = int(tt)
                else:
                    tid = int(lg.argmax())
            tokens.append(tid)
            if temperature > 0:
                logits_out.append(lg)
            if j == k - 1:
                break
            e = self.embed(torch.tensor([tid], device=dev)).reshape(1, -1).float()
            pj = self.ctx_k, self.ctx_v
            if tk is not None:
                pj = (torch.cat([self.ctx_k, tk], dim=1), torch.cat([self.ctx_v, tv], dim=1))
            h, (kt, vt) = self.head.forward_ctx(
                e, h, torch.tensor([cur + 1 + j], device=dev), pj)
            tk = kt if tk is None else torch.cat([tk, kt], dim=1)
            tv = vt if tv is None else torch.cat([tv, vt], dim=1)
        return tokens, logits_out

    def set_vocab_subset(self, token_ids) -> bool:
        """Draft over `token_ids` only (greedy drafting; sampling keeps the
        full head). The lm_head pass is ~1 GB of packed weights per draft
        token -- ~5.6 ms on the 27B -- and the full model verifies every draft
        anyway, so a draft head over the most frequent tokens trades a little
        acceptance for a much cheaper draft. Output is unchanged. Returns
        False (leaving the full head) if the head cannot be sliced."""
        sub = getattr(self.lm_head, "row_subset", None)
        if sub is None:
            return False
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        try:
            self.sub_head = self.lm_head.row_subset(ids)
        except Exception:
            self.sub_head = None
            return False
        self.sub_ids = ids.to(self.lm_head.min_s.device)
        return True

    def reset(self, tokens: list[int]) -> None:
        self.pos = len(tokens)
        self.ctx_k = self.ctx_v = None
        self._pending = []

    def append(self, tok: int) -> None:
        self.pos += 1

    @torch.no_grad()
    def propose(self, last_token: int, k: int, temperature: float = 0.0,
                top_k: int = 0, top_p: float = 1.0, forced=None
                ) -> tuple[list[int], list[torch.Tensor] | None]:
        """Propose k tokens after `last_token`.

        Returns (tokens, logits).  temperature <= 0 drafts greedy and returns
        logits=None: the caller treats the draft as one-hot and verifies by
        exact match against the target's argmax (see speculative_generate).
        temperature > 0 drafts by SAMPLING each token from the head's own
        temperature/top-k/top-p filtered softmax and returns the raw
        per-position logits, so the caller can reconstruct the draft's true
        distribution p_draft and verify with the speculative-sampling
        identity min(1, p_target/p_draft).  Sampling from exactly the
        distribution we report as p_draft is what makes that identity valid
        -- an argmaxed token billed as a sample would have the wrong p_draft
        at its own position."""
        if self.ctx_window and self._pending:
            return self._propose_ctx(k, temperature, top_k, top_p, forced)
        if self.hidden is None:
            return [], None

        out: list[int] = []
        logits_out: list[torch.Tensor] | None = [] if temperature > 0 else None
        h, tid = self.hidden, last_token
        dev = h.device
        for i in range(k):
            e = self.embed(torch.tensor([tid], device=dev)).reshape(1, -1).float()
            h = self.head.forward(e, h.reshape(1, -1).float(),
                                  torch.tensor([self.pos + i], device=dev))
            if temperature <= 0 and self.sub_head is not None:
                tid = int(self.sub_ids[self.sub_head(h)[0].argmax()])
                lg = None
            else:
                lg = self.lm_head(h)[0]
                if temperature > 0:
                    tid = _select(lg, temperature, top_k, top_p)
                else:
                    tid = int(lg.argmax())
            if temperature > 0:
                logits_out.append(lg)
            out.append(tid)
        return out, logits_out


class NGramWithMTPDrafter:
    """N-gram lookup and the checkpoint's MTP head, used TOGETHER in one
    draft rather than as a fallback pair: n-gram fills what it can from
    demonstrated repetition (near-zero cost, ~90% acceptance where it
    fires -- see ngram_draft.NGramDrafter), and the MTP head fills
    whatever budget n-gram left unfilled (including the whole draft, on a
    context n-gram has never seen).

    Order matters here, not preference: MTP is a model call every step, so
    it only pays for the tokens n-gram did not already supply for free.
    n-gram's own miss-cooldown backoff (self.ngram._skip) still applies
    independently -- if n-gram is cooling off after a run of misses, this
    reduces to plain MTP-only drafting for that step, not a stall.

    `reset`/`append`/`feedback` fan out to both children so each keeps its
    own bookkeeping (n-gram's suffix index, MTP's absolute position
    counter) in sync with the real committed sequence, exactly as if it
    were the sole drafter.
    """

    needs_hidden = True

    def __init__(self, ngram, mtp):
        self.ngram = ngram
        self.mtp = mtp

    @property
    def hidden(self):
        return self.mtp.hidden

    @hidden.setter
    def hidden(self, value):
        self.mtp.hidden = value

    def enable_context(self, window: int = 256) -> bool:
        return self.mtp.enable_context(window)

    @property
    def ctx_window(self) -> int:
        return self.mtp.ctx_window

    def observe(self, first_pos: int, hiddens, next_tokens) -> None:
        self.mtp.observe(first_pos, hiddens, next_tokens)

    def reset(self, tokens: list[int]) -> None:
        self.ngram.reset(tokens)
        self.mtp.reset(tokens)

    def append(self, tok: int) -> None:
        self.ngram.append(tok)
        self.mtp.append(tok)

    def propose(self, last_token: int, k: int, temperature: float = 0.0,
               top_k: int = 0, top_p: float = 1.0
               ) -> tuple[list[int], list[torch.Tensor] | None]:
        out, logits = self.ngram.propose(last_token, k, temperature=temperature,
                                         top_k=top_k, top_p=top_p)
        # ngram is one-hot (no distribution): pad its slice of logits with
        # None placeholders so the combined list stays index-aligned with
        # `out` for the caller (speculative_generate reads draft_logits[i]
        # positionally). Only meaningful once MTP tokens are appended below,
        # since a temperature<=0 caller ignores logits regardless.
        if logits is None and temperature > 0:
            logits = [None] * len(out)
        remaining = k - len(out)
        if remaining <= 0:
            return out, logits
        if self.mtp.ctx_window and self.mtp._pending:
            # With head context the n-gram tokens are FORCED through the head
            # (so its state matches the tokens that will be verified) and the
            # head's own picks continue from there.
            toks, mtp_logits = self.mtp.propose(
                last_token, k, temperature=temperature, top_k=top_k,
                top_p=top_p, forced=out)
            if temperature > 0:
                logits = [None] * len(out) + list(mtp_logits[len(out):])
            return toks, logits
        # MTP continues from the last committed OR n-gram-proposed token, at
        # the absolute position that token actually occupies -- self.pos
        # (MTPDrafter.append() was not called for n-gram's tokens, since
        # they are not yet verified, so it must be offset here explicitly
        # rather than by mutating self.mtp.pos permanently.
        mtp_last = out[-1] if out else last_token
        saved_pos = self.mtp.pos
        self.mtp.pos = saved_pos + len(out)
        try:
            mtp_out, mtp_logits = self.mtp.propose(
                mtp_last, remaining, temperature=temperature,
                top_k=top_k, top_p=top_p)
        finally:
            self.mtp.pos = saved_pos
        out = out + mtp_out
        if temperature > 0:
            logits = (logits or []) + (mtp_logits or [])
        return out, logits

    def feedback(self, n_accepted: int, n_proposed: int) -> None:
        # n-gram's own backoff is driven by whether ITS tokens were accepted,
        # not the combined draft's outcome -- crediting it for MTP-sourced
        # acceptances (or blaming it for MTP-sourced rejections) would teach
        # it the wrong lesson about whether repetition is actually present.
        if hasattr(self.ngram, "feedback"):
            self.ngram.feedback(n_accepted, n_proposed)


def build_n_gram_with_mtp_drafter(model, pack_dir: str, model_dir: str, *, max_ctx: int = 8):
    """N-gram + MTP combined drafter (see NGramWithMTPDrafter) for the CPU path."""
    from qv_band.decode.ngram_draft import NGramDrafter
    ngram = NGramDrafter(max_ctx=max_ctx)
    mtp = build_drafter(model, pack_dir, model_dir)
    return NGramWithMTPDrafter(ngram, mtp)


def build_n_gram_with_mtp_drafter_cuda(model, pack_dir: str, model_dir: str, *,
                              max_ctx: int = 8, device: str = "cuda"):
    """N-gram + MTP combined drafter (see NGramWithMTPDrafter) for the CUDA path."""
    from qv_band.decode.ngram_draft import NGramDrafter
    ngram = NGramDrafter(max_ctx=max_ctx)
    mtp = build_drafter_cuda(model, pack_dir, model_dir, device=device)
    return NGramWithMTPDrafter(ngram, mtp)


def _max_safe_drop(past) -> int:
    """How many positions can be cropped before the conv window underflows.

    crop() slices the retained conv history as [-(drop+kernel) : -drop], so it
    needs drop + kernel <= history.

    ALWAYS RETURNS 0 on Qwen3.5/3.6/3.8 (and this stays true no matter what
    is done to the cache -- confirmed, not a config gap). `transformers`
    exposes activate_past_recording()/record_past to make LinearAttentionLayer
    retain full history instead of a fixed conv_kernel_size window, which
    _try_crop below would need to have any margin to crop from. But the
    Qwen3.5/3.6/3.8 conv1d forward (modeling_qwen3_5.py's
    Qwen3_5GatedDeltaNet.torch_forward) always shrinks its OWN new_conv_state
    to exactly conv_kernel_size before calling update_conv_state --
    `F.pad(mixed_qkv, (conv_kernel_size - mixed_qkv.shape[-1], 0))` goes
    NEGATIVE (i.e. crops, not pads) whenever the input is already >=
    conv_kernel_size wide, and the fast single-token decode path
    (causal_conv1d_update, an external CUDA kernel) is an in-place
    fixed-width ring-buffer update that does not consult record_past at all.
    So conv_states can never hold more than conv_kernel_size elements on
    this model family regardless of whether recording is armed -- there is
    no extra history for crop() to ever recover, so _try_crop always sees
    n_drop > 0 == _max_safe_drop() and returns False. Confirmed empirically:
    armed cs.shape stays pinned at exactly conv_kernel_size across every
    step (see session notes). _snapshot_recurrent_state/_restore_recurrent_
    state above are the real fix for this architecture (O(1) full-tensor
    swap instead of a partial crop) and do not depend on this being non-zero
    -- keep this function and _try_crop as a correct, harmless no-op for
    this model family, and as the actual working path on any future
    architecture whose conv1d forward doesn't pre-truncate this way.
    """
    worst = None
    for layer in getattr(past, "layers", []) or []:
        cs = getattr(layer, "conv_states", None)
        ks = getattr(layer, "conv_kernel_size", None)
        if not cs or not ks:
            continue
        for i in range(len(cs)):
            if cs[i] is None:
                continue
            room = int(cs[i].shape[-1]) - int(ks[i])
            worst = room if worst is None else min(worst, room)
    return worst if worst is not None else 0


def _try_crop(past, n_drop: int) -> bool:
    """Roll the cache back n_drop positions; False if this cache cannot.

    Preferred over rebuilding because crop is O(1) against a prefill's
    O(prefix).  The check is done up front rather than by catching the
    exception, since a partially applied crop would leave layers inconsistent.
    """
    if n_drop <= 0:
        return True
    if n_drop > _max_safe_drop(past):
        return False
    try:
        past.crop(-n_drop)
        return True
    except Exception:
        return False


def _snapshot_recurrent_state(past, to=None):
    """Clone every linear-attention layer's conv_states + recurrent_states,
    so a rejected speculative batch can be undone with a cheap restore
    instead of replaying the whole generated-so-far prefix.

    WHY this exists: crop() (above) rolls back a linear-attention layer's
    CONV window, restorable by slicing IF the layer ever retained more than
    conv_kernel_size positions -- on Qwen3.5/3.6/3.8 it never does (see
    _max_safe_drop's docstring: the model's own conv1d forward pre-truncates
    to conv_kernel_size before the cache ever sees it, so crop() always has
    zero margin here). The layer's RECURRENT state (the actual
    SSM/linear-attention accumulator, same fixed shape regardless of
    sequence length -- see
    transformers.cache_utils.LinearAttentionLayer.lazy_initialization,
    torch.zeros_like + in-place update, never grows) is mutated forward-only
    with no inverse operation crop() can express: there is no way to
    "un-accumulate" a linear-attention step from the state alone. That is
    exactly why the caller (speculative_generate) used to fall back to a
    full prefix replay on every rejection -- correct, but O(prefix length)
    and growing every step, confirmed via profiling to be the dominant cost
    of speculative decoding on this architecture (measured: 7 rebuilds in
    18 steps consuming 7.7s of an 8.6s generation, dwarfing both the ~0.9s
    of MTP drafting and the underlying per-token GEMV cost).

    The fix: since the recurrent+conv state is SMALL and FIXED-SIZE
    (measured ~76 MB total across 48 layers on a 27B model, clone+restore
    ~1.2ms + ~1.1ms combined -- orders of magnitude cheaper than even a
    SHORT replay, let alone one late in a long generation), just snapshot
    it before the speculative forward pass runs (which mutates it in
    place, same as any other forward) and restore it verbatim on
    rejection -- turning an O(prefix) replay into an O(1) memcpy.
    Correctness: restoring the exact pre-batch tensor values is bit-exact
    by construction (a plain clone/copy_ round-trip), not an approximation
    of what a "proper" partial rollback would compute.

    Returns an opaque snapshot object (a list of the tensors this layer
    type exposes) or None if `past` has no linear-attention layers, in
    which case the caller has nothing to restore and should keep using
    the existing crop/rebuild logic unchanged.
    """
    snaps = []
    found_any = False
    for layer in getattr(past, "layers", []) or []:
        cs = getattr(layer, "conv_states", None)
        rs = getattr(layer, "recurrent_states", None)
        if cs is None and rs is None:
            snaps.append(None)
            continue
        found_any = True
        # to=None: a GPU clone (the speculative-rollback snapshot, restored in place
        # microseconds later); to="cpu": straight into host memory, no GPU temporary
        # (the prompt anchor, taken at peak prefill memory on 16 GB cards).
        cp = (lambda t: t.clone()) if to is None else (lambda t: t.to(to))
        cs_snap = {i: (cp(t) if t is not None else None)
                  for i, t in (cs or {}).items()}
        rs_snap = {i: (cp(t) if t is not None else None)
                  for i, t in (rs or {}).items()}
        snaps.append((cs_snap, rs_snap))
    return snaps if found_any else None


_PIPE = {}


def _fused_ready(past) -> bool:
    """Every banded layer already takes the fused attention path (the only one
    a pipelined chunk may use): its BANDED history, not the token count, must
    reach _FUSED_MIN_BANDED -- after a prefix reuse up to residual_length
    tokens can still sit in the dense tail."""
    from qv_band.kv import banded_cache as bc
    ls = [L for L in getattr(past, "layers", []) if isinstance(L, bc.BandedKVLayer)]
    return bool(ls) and all(L.is_initialized and L._banded_len[0] >= bc._FUSED_MIN_BANDED
                            for L in ls)


def _pipe_prefill(model, past):
    """PipePrefill for a pipeline-placed model with the fused qvb8 path on, or
    None (QV_PIPE_PREFILL=0, one card, no fused attention / CUDA kernel)."""
    if os.environ.get("QV_PIPE_PREFILL", "1") == "0":
        return None
    pl = getattr(model, "_qv_placement", None)
    # the overlapped prefill has two stages: exactly two distinct cards (N-stage: unified_scheduler.md 4.3)
    if pl is None or len(pl.devices) != 2 or pl.devices[0] == pl.devices[1] or past is None:
        return None
    split = pl.bounds[0]
    from qv_band.kv import banded_cache
    from qv_band.kv.fused_attn import _cuda_ext
    if not banded_cache.FUSED_ENABLED or not isinstance(past, banded_cache.BandedKVCache):
        return None
    if _cuda_ext(torch.cuda.current_device()) is None:
        return None
    key = id(model)
    if key not in _PIPE:
        from qv_band.decode.pipe_prefill import PipePrefill
        try:
            _PIPE[key] = PipePrefill(model, split)
            print(f"[pipe-prefill] two-card overlapped prefill, split at layer {split}", flush=True)
        except Exception as e:                         # noqa: BLE001
            print(f"[pipe-prefill] off: {e!r}", flush=True)
            _PIPE[key] = None
    return _PIPE[key]


def _anchor_pos(tok, ids: list[int]) -> int:
    """Index of the last <|im_start|> in `ids` (0 if none / unknown)."""
    try:
        t = tok.convert_tokens_to_ids("<|im_start|>")
    except Exception:                                  # noqa: BLE001
        return 0
    if not isinstance(t, int) or t < 0 or t == getattr(tok, "unk_token_id", None):
        return 0
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == t:
            return i
    return 0


def _save_anchor(past, prompt_ids: list[int]) -> None:
    """Prompt-end anchor for agent-style reuse (Cline & co.).

    An agent's next request is the previous PROMPT + the previous reply as
    the chat template re-renders it (Qwen drops the <think> block, so it
    does not match what was generated) + new input. Exact-prefix reuse then
    always misses. The full-attention K/V can be cropped back to the prompt
    end; the linear-attention state cannot be rolled back -- so keep a copy
    of it (fixed size, ~0.15 GB on the 27B) in host RAM, taken right after
    prefill. QV_PROMPT_ANCHOR=0 disables."""
    if past is None or os.environ.get("QV_PROMPT_ANCHOR", "1") == "0":
        return
    snaps = _snapshot_recurrent_state(past, to="cpu")
    # Layer hook: a cache layer that cannot crop back arbitrarily far (e.g. Ripple's sliding-window layers keep only
    # window+margin keys) may provide anchor_snapshot()/anchor_restore(); its state is saved here and restored on
    # reuse instead of cropped. Layers without the hook (every Qwen/Nemotron layer) are handled exactly as before.
    lay = {i: L.anchor_snapshot() for i, L in enumerate(getattr(past, "layers", None) or [])
           if hasattr(L, "anchor_snapshot")}
    # A cache with neither recurrent state nor hooked layers (pure attention, e.g. Ripple on BandedKVCache: every
    # layer keeps its full history) is still reusable by crop alone, so it is anchored too.
    past._qv_anchor = (list(prompt_ids), snaps, lay)


def _reuse_from_anchor(prefix_cache, ids_list: list[int], cache_config) -> int:
    """Roll the saved cache back to its prompt-end anchor when the new request
    starts with that prompt; returns the reused length (0 = no reuse)."""
    past = prefix_cache.past
    anc = getattr(past, "_qv_anchor", None) if past is not None else None
    if anc is None or cache_config != prefix_cache.cache_config:
        return 0
    a_ids, snaps, lay = anc
    n = len(a_ids)
    if n < 2 or len(ids_list) <= n or ids_list[:n] != a_ids:
        return 0
    # Crop EVERY full-attention layer back to the anchor, whatever its cache type
    # (qvb8 banded, int8, dense): cropping only the banded layers left an int8 /
    # dense cache holding prompt + previous reply, and the new suffix then
    # attended to stale tokens. Linear-attention layers are skipped (their crop
    # raises unless past recording is on; their state is restored from the
    # snapshot below). A crop that fails part-way leaves the layers
    # inconsistent, so the saved cache is dropped and the request prefills fresh.
    try:
        for li, layer in enumerate(past.layers):
            if getattr(layer, "recurrent_states", None) is not None or getattr(layer, "conv_states", None) is not None:
                continue
            if li in lay:
                layer.anchor_restore(lay[li])      # layer hook (see _save_anchor): restore, not crop
                continue
            layer.crop(n)
    except Exception as e:
        print(f"[prefix] anchor reuse skipped, cache cannot crop to {n}: {e!r}",
              file=sys.__stdout__, flush=True)
        prefix_cache.past = None
        prefix_cache.token_ids = []
        return 0
    # Restore onto each live tensor's own card (pipeline placement spreads
    # the linear-attention layers over two GPUs).
    for layer, sn in zip(past.layers, snaps or []):
        if sn is None:
            continue
        cs, rs = getattr(layer, "conv_states", None), getattr(layer, "recurrent_states", None)
        for i, t in sn[0].items():
            if t is not None and cs is not None and cs.get(i) is not None:
                cs[i] = t.to(cs[i].device)
        for i, t in sn[1].items():
            if t is not None and rs is not None and rs.get(i) is not None:
                rs[i].copy_(t)
    prefix_cache.token_ids = list(a_ids)
    print(f"[prefix] reuse to prompt anchor: {n} tok kept, "
          f"{len(ids_list) - n} new", file=sys.__stdout__, flush=True)
    return n


def _restore_recurrent_state_slot(past, snaps, bi: int) -> None:
    """Like _restore_recurrent_state, but restores only batch ROW `bi` of
    every linear-attention layer's conv/recurrent state, leaving every
    other slot's state untouched -- the per-slot counterpart needed once
    a batched decode step can accept/reject a different number of draft
    tokens PER SLOT (SGLang/llama.cpp's standard continuous-batching
    shape: draft, one batched verify forward, then each slot accepts/
    rejects and rolls back independently).

    Safe because conv_states[state_idx] and recurrent_states[state_idx]
    are genuine (batch, ...) tensors (confirmed against this
    transformers version's LinearAttentionLayer.lazy_initialization:
    conv_states keeps conv_states.shape[:-1] as its leading shape,
    recurrent_states is torch.zeros_like(recurrent_states) -- both
    preserve the incoming batch axis verbatim), so restoring one row is
    a plain in-place slice-copy, not a shape-changing operation the way
    conv_states' whole-tensor REASSIGN in the batch=1 restore is.
    """
    if snaps is None:
        return
    for layer, snap in zip(getattr(past, "layers", []) or [], snaps):
        if snap is None:
            continue
        cs_snap, rs_snap = snap
        cs = getattr(layer, "conv_states", None)
        rs = getattr(layer, "recurrent_states", None)
        for i, t in cs_snap.items():
            if t is not None and cs is not None and cs.get(i) is not None:
                cs[i][bi:bi + 1].copy_(t[bi:bi + 1])
        for i, t in rs_snap.items():
            if t is not None and rs is not None and rs.get(i) is not None:
                rs[i][bi:bi + 1].copy_(t[bi:bi + 1])


def _restore_recurrent_state(past, snaps) -> None:
    """Undo everything a speculative forward did to `past`'s linear-attention
    layers, using a snapshot from _snapshot_recurrent_state.

    conv_states: reassigned (dict-slot swap), not copy_(). On this model
    family conv_states stays a fixed conv_kernel_size ring buffer (see
    _max_safe_drop's docstring), so copy_() would work today, but reassign
    anyway: it is the one form that stays correct if conv_states ever DOES
    change shape between the snapshot and the restore (a different model
    family, a future transformers version) -- copy_() would raise
    (RuntimeError: size mismatch) in that case instead of just working. A
    snapshot taken before the step is by definition the exact object slot to
    go back to, so swap the reference
    back in rather than trying to copy into the (differently-shaped, and
    generally larger) live tensor.

    recurrent_states: still copy_() in place. Unlike conv_states this is the
    fixed-size SSM/linear-attention accumulator (see docstring above) --
    genuinely constant shape regardless of history length or recording mode,
    so in-place mutation (matching the layers' own update_recurrent_state())
    is both correct and avoids an extra allocation per rejection.
    """
    if snaps is None:
        return
    for layer, snap in zip(getattr(past, "layers", []) or [], snaps):
        if snap is None:
            continue
        cs_snap, rs_snap = snap
        cs = getattr(layer, "conv_states", None)
        rs = getattr(layer, "recurrent_states", None)
        for i, t in cs_snap.items():
            if t is not None and cs is not None and cs.get(i) is not None:
                cs[i] = t
        for i, t in rs_snap.items():
            if t is not None and rs is not None and rs.get(i) is not None:
                rs[i].copy_(t)


class _ResultWithHidden:
    """Minimal stand-in for CausalLMOutputWithPast that also carries
    last_hidden_state, for callers that need the MTP drafter's input."""
    __slots__ = ("logits", "past_key_values", "last_hidden_state",
                 "layer_features")

    def __init__(self, logits, past_key_values, last_hidden_state,
                 layer_features=None):
        self.logits = logits
        self.past_key_values = past_key_values
        self.last_hidden_state = last_hidden_state
        self.layer_features = layer_features


def _resolve_final_norm(model):
    """Return the base model's final RMSNorm module.

    Standard HF causal-LM wrappers (Qwen, Llama, ...) expose it as
    ``model.model.norm``.  NemotronH names it ``norm_f`` instead.  This
    helper tries both so the pre-norm hidden-state capture hook works
    across architectures without caller-side special-casing.
    """
    base = getattr(model, "model", model)
    for attr in ("norm", "norm_f"):
        mod = getattr(base, attr, None)
        if mod is not None:
            return mod
    raise AttributeError(
        f"cannot find final norm on {type(base).__name__}; "
        f"tried 'norm' and 'norm_f'"
    )


def _forward_with_hidden(model, input_ids, past_key_values, want_hidden: bool,
                         logits_last_only: bool = False,
                         capture_layers: list[int] | None = None):
    """model(input_ids, ...) that also returns last_hidden_state when
    want_hidden is set. See speculative_generate's docstring for why plain
    model(...) cannot provide this: CausalLMOutputWithPast never forwards
    the tensor Qwen3_5ForCausalLM.forward() computes internally. Reproduces
    that forward's own logits computation (inner model + lm_head) exactly,
    so verification is unaffected -- this only additionally keeps the
    tensor the stock wrapper discards.

    `logits_last_only`: compute logits for ONLY the final position. The
    stock forward otherwise runs lm_head over every position -- a
    (1, seq_len, vocab) fp32 tensor, ~2.4 GB at seq_len=4000 with a 152k
    vocab -- even when the caller needs one row. The cache-rebuild replay
    (a full-prefix forward whose logits are unused; next_tok was already
    chosen from the pre-rebuild batch) sets this, which is what stops that
    path from allocating a multi-GB logits tensor per rejection during a
    long or degenerate generation. The verify step does NOT set it: it
    needs a logit row per drafted position, but its input is only
    1 + k tokens so full-sequence logits there are already tiny.

    The hidden state handed to the MTP drafter is captured PRE the base
    model's final `self.norm` (via a forward hook on that module, reading
    its INPUT not its output), not `outputs.last_hidden_state`
    (post-norm). Reasoning: MTP checkpoints (this one included: it ships
    its own learned `mtp.pre_fc_norm_hidden` norm, applied inside
    MTPHead.forward) are typically trained to consume the last decoder
    LAYER's raw output and normalise it themselves, so feeding them the
    base model's own already-normalised final hidden state would run that
    norm twice with two different learned scales.

    NOT CONFIRMED to fix draft quality: tried against this checkpoint's
    MTP head (still measured ~5-16% acceptance either way -- see
    spec_decode.py/mtp_cuda.py's other architecture-investigation notes;
    no reference implementation exists for this checkpoint's MTP head to
    verify any of these choices against). Kept because it is not WRONG on
    the same reasoning that motivated it and does not regress anything
    that worked before it (plain generation, n-gram/no-drafter paths never
    call this hook at all), not because it is known to help.
    `model.model.norm` is this model's final RMSNorm
    (Qwen3_5TextModel.forward's `hidden_states = self.norm(hidden_states)`,
    the very last line before returning `last_hidden_state`).

    `capture_layers` (only honoured with want_hidden): decoder-layer indices
    whose OUTPUTS are concatenated per position into `layer_features`
    (seq_len, len(capture_layers) * H) -- the context a multi-layer drafter
    (DFlashDrafter) is trained on; hidden_states[i + 1] in HF terms.
    """
    if not want_hidden:
        if logits_last_only:
            # logits_to_keep=1 -> stock forward slices hidden to the last
            # position BEFORE lm_head, so the (1, seq_len, vocab) tensor is
            # never built. res.logits is then (1, 1, vocab); callers that
            # use this flag only read the last row anyway.
            return model(input_ids, past_key_values=past_key_values,
                         use_cache=True, logits_to_keep=1)
        return model(input_ids, past_key_values=past_key_values, use_cache=True)
    pre_norm_hidden = None

    def _capture_pre_norm(_module, args, _kwargs):
        nonlocal pre_norm_hidden
        pre_norm_hidden = args[0]

    handles = [_resolve_final_norm(model).register_forward_pre_hook(_capture_pre_norm, with_kwargs=True)]
    layer_out: dict[int, torch.Tensor] = {}
    for li in capture_layers or ():
        def _capture_layer(_module, _args, out, li=li):     # tap layers may sit on other cards: gathered below (xfer)
            layer_out[li] = out[0] if isinstance(out, tuple) else out
        handles.append(model.model.layers[li].register_forward_hook(_capture_layer))
    try:
        outputs = model.model(input_ids, past_key_values=past_key_values, use_cache=True)
    finally:
        for h in handles:
            h.remove()
    # Under pipeline placement the captured layers live on different cards;
    # gather them on the main one (where the final hidden state and the
    # drafter are) -- through host memory, never peer-to-peer (common/xfer.py).
    from qv_band.common.xfer import to_device
    _fdev = outputs.last_hidden_state.device
    features = (torch.cat([to_device(layer_out[li], _fdev) for li in capture_layers], dim=-1)[0]
                if capture_layers else None)
    hidden = pre_norm_hidden if pre_norm_hidden is not None else outputs.last_hidden_state
    # lm_head over only the final position when the caller does not need the
    # full logit sequence -- same multi-GB saving as the logits_to_keep=1
    # path above, done by hand here because we call model.model(...) /
    # model.lm_head(...) ourselves. `hidden` (for the drafter) still spans
    # the whole sequence; only the logits projection is sliced.
    last_hs = outputs.last_hidden_state
    logits = model.lm_head(last_hs[:, -1:, :] if logits_last_only else last_hs)
    return _ResultWithHidden(logits, outputs.past_key_values, hidden, features)


def _advance_cache_only(model, input_ids, position_ids, past_key_values):
    """Run the decoder stack to advance the KV cache, computing no logits.

    This is all a prefill chunk or a recurrent-state replay needs: they
    exist only to push tokens through the cache. Running model.model (the
    decoder stack) rather than model (which ends in lm_head) is the
    correct forward for that job, so there is no flag to ask for the
    other one.

    It matters because lm_head is (hidden x vocab) and this family's
    vocab is 248320: one 1024-token chunk costs 2.6 TFLOP and a 485 MiB
    logits tensor that nothing indexes.
    """
    out = model.model(input_ids, attention_mask=None,
                      position_ids=position_ids,
                      past_key_values=past_key_values, use_cache=True)
    return _ResultWithHidden(None, out.past_key_values, None)


def _forward_batched_with_hidden(model, input_ids, attention_mask, position_ids,
                                 past_key_values, want_hidden: bool,
                                 keep: int | None = None,
                                 keep_hidden: int | None = None,
                                 capture_layers: list[int] | None = None):
    """_forward_with_hidden's batched counterpart: takes explicit
    attention_mask/position_ids, and the SAME pre-norm hidden-state
    capture trick so a needs_hidden=True drafter (MTPDrafter,
    NGramWithMTPDrafter) works inside ContinuousBatchEngine too -- see
    _forward_with_hidden's own docstring for why the hook reads
    model.model.norm's INPUT, not outputs.last_hidden_state.

    This path always produces logits, because its only caller is the
    decode/verify step, and verification indexes a logit ROW PER DRAFTED
    POSITION (_verify reads row_logits[i] for every i), not just the last
    one. A forward that needs no logits calls _advance_cache_only
    instead.

    `keep`: only the LAST `keep` positions of every row are returned (logits
    and hidden). The caller knows how many trailing positions it reads -- a
    decode row needs its k+1 verify positions, a just-admitted row only its
    final one -- and computing lm_head over a whole 4096-token prefill chunk
    (4096 x vocab logits, several GB) to read a single row is pure waste.
    None keeps every position.

    `capture_layers`: as in _forward_with_hidden, but `layer_features` is
    (batch, seq_len, F) and is NOT trimmed by `keep` -- a DFlash drafter needs
    every real position of an admitted prompt tail, not just the last few.
    """
    if not want_hidden:
        kw = {} if keep is None else {"logits_to_keep": keep}
        return model(input_ids, attention_mask=attention_mask,
                     position_ids=position_ids,
                     past_key_values=past_key_values, use_cache=True, **kw)
    pre_norm_hidden = None

    def _capture_pre_norm(_module, args, _kwargs):
        nonlocal pre_norm_hidden
        pre_norm_hidden = args[0]

    handles = [_resolve_final_norm(model).register_forward_pre_hook(_capture_pre_norm, with_kwargs=True)]
    layer_out: dict[int, torch.Tensor] = {}
    for li in capture_layers or ():
        def _capture_layer(_module, _args, out, li=li):     # tap layers may sit on other cards: gathered below (xfer)
            layer_out[li] = out[0] if isinstance(out, tuple) else out
        handles.append(model.model.layers[li].register_forward_hook(_capture_layer))
    try:
        outputs = model.model(input_ids, attention_mask=attention_mask,
                              position_ids=position_ids,
                              past_key_values=past_key_values, use_cache=True)
    finally:
        for h in handles:
            h.remove()
    features = (torch.cat([layer_out[li] for li in capture_layers], dim=-1)
                if capture_layers else None)
    hidden = pre_norm_hidden if pre_norm_hidden is not None else outputs.last_hidden_state
    last = outputs.last_hidden_state
    if keep is not None:
        # The hidden states are kept for MORE trailing positions than logits
        # when the draft head needs context (keep_hidden): logits cost a
        # vocab-sized lm_head row each, hidden rows are just H floats.
        hidden = hidden[:, -max(keep, keep_hidden or 0):]
        last = last[:, -keep:]
    logits = model.lm_head(last)
    return _ResultWithHidden(logits, outputs.past_key_values, hidden, features)



def _install_decode_graphs(model, layer_graphs: bool) -> None:
    """Dense models, opt-in QVB8_MLP_GRAPH=1 (the V100 launchers): replay each MLP as a CUDA graph at decode
    (moe_graph.install_mlp_graphs) and each linear-attention layer's input projections as one graph (proj_graph).
    layer_graphs: also each whole linear-attention decoder layer as ONE graph at verify (layer_graph) -- the
    one-request path only; the scheduler's packed steps re-split a row per slot inside that layer. Once per model
    (the layer graphs once more when a later caller asks for them); an optimisation only."""
    done = model.__dict__.setdefault("_qv_decode_graphs", set())
    try:
        if "mlp" not in done:
            done.add("mlp")
            from qv_band.formats.qvb8.moe_graph import install_mlp_graphs
            _n = install_mlp_graphs(model)
            if _n:
                print(f"[mlp-graph] {_n} MLPs replay as CUDA graphs at decode", flush=True)
                from qv_band.decode.proj_graph import install as _install_proj
                _np = _install_proj(model)
                if _np:
                    print(f"[proj-graph] {_np} projection groups replay as CUDA graphs at decode", flush=True)
                done.add("on")
        if layer_graphs and "on" in done and "layer" not in done:
            done.add("layer")
            from qv_band.decode.layer_graph import install as _install_layers
            _nl = _install_layers(model)
            if _nl:
                print(f"[layer-graph] {_nl} linear-attention layers replay as CUDA graphs at verify", flush=True)
    except Exception as _e:                       # noqa: BLE001 - optional accelerator
        print(f"[mlp-graph] not installed: {_e!r}", flush=True)

@torch.no_grad()
def speculative_generate(model, drafter, tok, ids, *, k: int = 8,
                         max_new: int = 40, stream: bool = True,
                         temperature: float = 0.0, top_k: int = 0,
                         top_p: float = 1.0, kv_tau: float | None = None,
                         kv_stats: dict | None = None,
                         int8_kv: bool = True,
                         int8_residual_length: int = 128,
                         qvb8_resident: bool = False,
                         skip_special_tokens: bool = True,
                         abort_check=None,
                         prefix_cache: "PrefixCache | None" = None,
                         return_cache: bool = False,
                         on_heartbeat=None,
                         prefill_chunk: int | None = None):
    """Greedy decode, drafting k tokens per step.  drafter=None -> plain greedy.

    `abort_check`: optional zero-arg callable, polled once per decode step
    (after a token is emitted). Returning True stops generation early,
    same as reaching max_new/eos, WITHOUT raising -- the caller gets back
    whatever was emitted so far, same return shape as a normal finish.
    Meant for a caller like serve.py's SSE handler to detect a client
    disconnect mid-stream (see its _peer_gone()).

    `on_heartbeat`: optional zero-arg callable, polled once per prefill
    chunk AND once per decode step, but only actually CALLED when
    HEARTBEAT_INTERVAL_S of wall-clock time has passed since the last
    call (see _Heartbeat) -- decoupled from chunk size or token count
    entirely, so it fires on a real time budget regardless of how big a
    chunk is or how slow/fast generation is running right now. A long
    prompt's chunked prefill can run for minutes producing zero streamed
    bytes -- confirmed to trip a client's own body-read timeout (Cline:
    "UND_ERR_BODY_TIMEOUT") well before the first token is ever emitted,
    which then cancels and resubmits, compounding delay on top of delay.
    Decode tokens are normally their own heartbeat (a client sees
    streamed content), but a slow or stalled step (heavy draft rejection,
    GPU contention) can still leave a real gap with nothing sent, so this
    covers decode too, not just prefill. A caller streaming over HTTP
    (serve.py's _stream) can use this to send an SSE keep-alive (e.g. a
    comment line) that resets the client's read timeout without emitting
    anything a client would render as content.

    temperature > 0 switches token selection to sampling (top-k then
    nucleus/top-p filtered) and ALSO switches verification from exact-match
    to real speculative sampling: each drafted token is accepted with
    probability min(1, p_target(token)/p_draft(token)) and a rejection draws
    the emitted token from the residual distribution max(0, p_target -
    p_draft), renormalised (see _accept_prob / _resample_residual).  p_draft
    is the drafter's own distribution: one-hot at the proposed token when
    the drafter has none (n-gram, and MTP at temperature 0), which reduces
    acceptance to a coin flip at p_target(token); the MTP head run with
    temperature > 0 returns its real per-position logits, so the true draft
    distribution is used there.  Either way the target model's sampling
    distribution is preserved exactly (Leviathan et al. 2023 / Chen et al.
    2023): unlike greedy verification, an accepted draft is not necessarily
    identical to what plain step-by-step sampling would have produced on a
    given random draw, but the marginal distribution over next tokens is the
    same either way.

    Note on the hybrid cache: the linear-attention layers hold a depthwise
    conv window (kernel 4), so the cache must never be cropped to fewer
    positions than that window needs.  Prompts shorter than the kernel are
    run without speculation for that reason.

    `kv_tau` (v7-KV, see kv_band.py) applies the certified magnitude/direction
    KV codec to the PREFILL cache, once, right after the prompt's forward
    pass -- the one moment the whole prompt population exists at once, which
    is what lets a per-request calibration carry the same certified bound an
    offline weight calibration gets. Calibration itself runs on the CPU
    deliberately: it is a sort/searchsorted pass, and doing it on-device
    would need a full-size fp32 temporary at exactly the moment the point is
    to relieve VRAM.

    IMPORTANT, so nobody reads a VRAM saving into this that is not there:
    the codec currently writes the RECONSTRUCTION back as a dense tensor, so
    the resident cache stays fp16 and this saves no memory today. What it
    does give is the true end-to-end quality cost of a given tau, measured
    on the real model and a real prompt. Turning the reported ratio into an
    actual saving needs an attention kernel that gathers from the banded
    form directly (the same thing qv_anchored_cuda does for weights) -- that
    kernel does not exist yet. `kv_stats`, if passed, receives the
    measured per-layer bytes/error dict.

    `prefix_cache` (see PrefixCache): a handle saved from a PRIOR call via
    return_cache=True. If `ids` starts with exactly that cache's consumed
    tokens (and the KV-cache kwargs match), the prompt's shared portion is
    NOT re-prefilled -- only the new suffix (`ids` past the shared prefix)
    runs through the model, same as llama.cpp's server reusing a slot's
    cache across turns of one conversation. A prefix_cache that does not
    match (different config, or `ids` diverges anywhere) is silently
    ignored and this falls back to a normal fresh prefill -- never raises,
    since "the cache did not apply" is routine (conversation reset, a
    branched retry) not an error.

    `return_cache`: when True, the return value gains a third element, a
    PrefixCache the caller may save and pass as `prefix_cache` on its next
    call. None if the run's own KV-cache class does not support this
    (bf16/DynamicCache without drafting has no cache object at all).

    `prefill_chunk`: tokens per chunked-prefill forward pass (see the
    module-level PREFILL_CHUNK docstring). None (default) uses
    PREFILL_CHUNK / QV_BAND_PREFILL_CHUNK; pass an explicit value to
    override per-call without touching the environment.
    """
    _prefill_chunk = prefill_chunk if prefill_chunk is not None else PREFILL_CHUNK
    st = Stats()
    out_ids: list[int] = []
    eos = tok.eos_token_id

    # Incremental detokeniser for stream=True -- see module-level
    # _IncStreamer's own docstring for the full "why" (O(1)/token vs the
    # old O(seq) tok.decode(out_ids) every step). speculative_generate_
    # batched and llm_apps/serve.py's scheduler branch construct their
    # own instance(s) directly (one per slot/request -- each stream needs
    # its own pending-id cache).

    # Hidden states are only needed by a drafter that consumes them (MTP); the
    # n-gram drafter does not.
    #
    # We call model.model(...) + model.lm_head(...) ourselves instead of
    # plain model(...) whenever want_hidden. Qwen3_5ForCausalLM.forward()
    # computes `hidden_states = outputs.last_hidden_state` from its inner
    # Qwen3_5TextModel internally, uses it to build logits, then returns a
    # CausalLMOutputWithPast that never forwards that tensor -- that class
    # only carries `hidden_states` (the per-layer tuple, populated only when
    # output_hidden_states=True), and Qwen3_5TextModel.forward does not wire
    # that flag through at all, so requesting it is also a dead end.
    # `res.last_hidden_state` therefore raised AttributeError on every call,
    # `propose()` never ran, and every request measured proposed=0 -- not a
    # head-math bug in the MTP weights themselves, a missing plumbing path to
    # get the base model's final hidden state out at all. _forward_with_hidden
    # below reproduces ForCausalLM.forward()'s own logits computation exactly
    # (same lm_head, same slice), so verification is unaffected; it just also
    # keeps the tensor the wrapper discards.
    want_hidden = getattr(drafter, "needs_hidden", False)
    # A multi-layer-context drafter (DFlashDrafter) additionally needs the
    # outputs of specific target layers at EVERY committed position: the
    # whole prompt (every prefill chunk, not just the last) and each step's
    # seq[0] + accepted drafts. They are taken from the forward that already
    # computed them -- the prefill chunk or the verify batch -- so no extra
    # target pass is ever run for the drafter.
    capture = (getattr(drafter, "capture_layers", None)
               if drafter is not None and k > 0 else None)
    draft_probs = bool(getattr(drafter, "returns_probs", False))

    # Rollback has to be armed on a FRESH cache, before the first forward.
    # The linear-attention layers keep only a kernel-sized conv window by
    # default and discard the rest as they go; record_past makes them retain
    # the full history so crop() can slice a correct window back out.  Arming
    # it after the prefill is too late -- the history is already gone, and the
    # later crop leaves a window shorter than the conv kernel.
    # INT8 KV cache (qv_band.kv.int8_kv_cache) is orthogonal to the QV-band
    # v7 weight packs: v7 governs how W is stored and replayed on the fly,
    # this governs how the runtime K/V population is stored. Both can be on at
    # once, and on the v7 path they usually should be -- v7 shrinks the
    # weights, this shrinks the cache, and the two costs dominate different
    # phases (weights are constant, cache grows with context).
    #
    # Unlike the v7-KV codec below, this is a REAL memory saving: the cache
    # holds int8 tensors, not a dense reconstruction. It quantizes only the
    # full_attention layers; linear-attention layers keep their stock class,
    # since their recurrent conv/SSM state has no sequence dimension to
    # quantize.
    # The tuple prefix_cache reuse is gated on -- a saved cache built with
    # different kwargs is a different storage form and must not be reused
    # (see PrefixCache's docstring).
    _cache_config = (qvb8_resident, int8_kv,
                     kv_tau, int8_residual_length)

    ids_list = ids[0].tolist()
    _reused_len = 0
    if prefix_cache is not None:
        _reused_len = prefix_cache.reusable_prefix_len(ids_list, _cache_config)
        # At least one token must still go through the model -- the forward
        # below needs real logits to pick the first emitted token from, and
        # a zero-length sequence has none. A cache that already covers ALL
        # of `ids_list` (the caller sent back exactly what was generated,
        # nothing new) reuses everything except the last token.
        if _reused_len >= len(ids_list):
            _reused_len = len(ids_list) - 1
        if not _reused_len:
            _reused_len = _reuse_from_anchor(prefix_cache, ids_list, _cache_config)
        if not _reused_len:
            # A miss can never be reused (exact prefix only, see PrefixCache),
            # yet the caller still holds the handle for this whole call: free
            # the old conversation's KV/recurrent state now, before the new
            # prefill, instead of carrying two contexts at once (~0.7 GB after
            # a 3k-token request on the 27B -- what a 16 GB card cannot spare).
            # Dropping the references is enough: the blocks go back to the
            # caching allocator and the new prefill reuses them.
            prefix_cache.past = None
            prefix_cache.token_ids = []

    past = None

    if _reused_len:
        # Exact-prefix hit: the saved cache already holds everything up to
        # _reused_len tokens (see PrefixCache -- this is the only case that
        # is ever reused, never a crop). Skip building a fresh cache AND
        # skip re-prefilling those tokens: only the new suffix goes through
        # the model below.
        past = prefix_cache.past
    elif qvb8_resident:
        # v7-KV held RESIDENT in banded form (qv_band/kv_cache_qvb8.py) --
        # unlike the kv_tau path below, which bands the prefill only to
        # measure it and writes a dense reconstruction straight back, this
        # keeps the compressed objects as the stored state between calls.
        # Measured 22.9% of dense on the 4B pack, against ~25% for int8.
        # Attention receives dense K/V from BandedKVCache.update() every
        # forward call -- no attention-implementation swap needed (see
        # kv_cache_qvb8.py's module docstring).
        from qv_band.kv.banded_cache import BandedKVCache
        from qv_band.kv.codec import KV_DEFAULT_TAU
        past = BandedKVCache(config=model.config,
                             tau=kv_tau or KV_DEFAULT_TAU,
                             residual_length=int8_residual_length)
    elif int8_kv:
        from qv_band.kv.int8_kv_cache import Int8QuantizedCache
        past = Int8QuantizedCache(config=model.config,
                                  residual_length=int8_residual_length)
    elif drafter is not None and k > 0:
        from transformers import DynamicCache
        past = DynamicCache(config=model.config)

    # Arm the ring-snapshot fast path for partial-accept rejection recovery
    # (see ring_snapshot.py's module docstring for the full "why" -- this
    # replaces the O(prefix) replay fallback for the 59%-of-rejections
    # partial-accept case with an O(1) splice, using data the batched
    # verify forward already computes). patch_model_for_ring_capture is
    # idempotent and a no-op on a non-hybrid model (returns 0, no
    # Qwen3_5GatedDeltaNet found) -- safe to call unconditionally here.
    _ring_available = False
    if past is not None and drafter is not None and k > 0:
        from qv_band.decode.ring_snapshot import patch_model_for_ring_capture
        _ring_available = patch_model_for_ring_capture(model) > 0

    if (past is not None and qvb8_resident
            and os.environ.get("QV_FUSED_ATTN", "1") != "0"):
        # Fused qvb8 decode attention (kv/fused_attn.py): attends the packed ids
        # directly instead of rebuilding the whole history every layer and step.
        # QV_FUSED_ATTN=0 keeps the dense reconstruction instead.
        try:
            from qv_band.kv.fused_attn import install as _install_fused
            _install_fused(model)
        except Exception as _e:                       # noqa: BLE001 - optional accelerator
            print(f"[fused-attn] not installed: {_e!r}", flush=True)

    try:                                  # sm_70: WMMA Gated DeltaNet prefill kernels
        from qv_band.kv.gdn_volta import install as _install_gdn
        _install_gdn()
    except Exception as _e:                       # noqa: BLE001 - optional accelerator
        print(f"[gdn] not installed: {_e!r}", flush=True)

    _install_decode_graphs(model, layer_graphs=True)

    _heartbeat = _Heartbeat(on_heartbeat)

    try:
        # A prefix-cache hit means `past` already holds everything up to
        # _reused_len tokens -- only forward the NEW suffix through the
        # model, same shape as decode-loop rebuild (spec_decode.py's own
        # replay path) but for the initial prompt instead of a rejection.
        prefill_ids = (ids[:, _reused_len:] if _reused_len else ids)
        _t_pf0 = time.perf_counter()

        # Chunked prefill (see PREFILL_CHUNK): feed the prompt through the
        # cache PREFILL_CHUNK tokens at a time instead of one forward over
        # the whole thing, so peak VRAM for a long prompt is bounded by one
        # chunk's activations, not the entire prompt's -- the same reason
        # llama.cpp (--ubatch-size) and SGLang (chunked_prefill_size) both
        # chunk prefill. Only the LAST chunk needs its logits (the first
        # generated token is picked from the final position); every
        # earlier chunk uses logits_last_only=True for the same reason the
        # cache-rebuild replay does (spec_decode.py's own
        # _forward_with_hidden docstring) -- it skips building a
        # (1, chunk_len, vocab) tensor nobody reads. abort_check is polled
        # BETWEEN chunks too, so a disconnect or the VRAM ceiling can stop
        # a long prefill partway instead of only after it finishes.
        n_prefill = prefill_ids.shape[-1]
        _prefill_aborted = False
        # Prompt anchor (see _save_anchor): snapshot the linear-attention state
        # at the start of the LAST chat turn header (<|im_start|>), i.e. the end
        # of the conversation history -- the point an agent's next request
        # shares with this one. A chunk boundary is placed there.
        _anchor_at = _anchor_pos(tok, ids[0].tolist()) - _reused_len
        # The anchor splits the prefill at that point: one extra forward. It pays
        # back only when the history it saves re-prefilling is longer than one
        # chunk (re-prefilling a shorter one costs that same single forward), so
        # short prompts keep the one-pass prefill (measured on the dev GPU: an anchored
        # 24-token prompt took 2.4 s to prefill instead of 1.2 s).
        if not _prefill_chunk < _anchor_at < n_prefill:
            _anchor_at = None
        if n_prefill <= _prefill_chunk and _anchor_at is None:
            res = _forward_with_hidden(model, prefill_ids, past, want_hidden,
                                       logits_last_only=True,
                                       capture_layers=capture)
            if capture:
                drafter.observe_features(_reused_len, res.layer_features)
        else:
            res = None
            _done = _reused_len   # how many of ids[0] the cache now holds
            # A long prompt chunked into many forward passes previously
            # printed NOTHING until the whole prefill finished -- from the
            # caller's side (serve.py's console, Cline's "thinking..."
            # spinner) that is indistinguishable from a hang, especially
            # since a very long prompt (a large agent system prompt + tool
            # schemas + file context, the exact shape Cline sends) can mean
            # dozens of chunks each doing a real 9B forward pass before
            # token one is ever emitted. Throttle to wall-clock (not every
            # chunk -- hundreds of chunks would spam) so there is always
            # visible progress without flooding the log.
            _chunk_t0 = time.time()
            _last_print = _chunk_t0
            _prof = os.environ.get("QV_PROFILE_PREFILL") == "1"
            if _prof:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
            _bounds = list(range(0, n_prefill, _prefill_chunk))
            if _anchor_at is not None and _anchor_at not in _bounds:
                _bounds = sorted(_bounds + [_anchor_at])
            _bounds.append(n_prefill)
            _chunks = list(zip(_bounds[:-1], _bounds[1:]))
            _pp = _pipe_prefill(model, past) if ids.shape[0] == 1 else None

            def _after(c0, c1, feats):
                """Per-chunk bookkeeping, in chunk order. True = abort."""
                nonlocal _done, _last_print
                is_last = c1 >= n_prefill
                if capture and feats is not None:
                    drafter.observe_features(_done, feats)
                _done += c1 - c0
                if _anchor_at is not None and c1 == _anchor_at:
                    _save_anchor(past, ids[0].tolist()[:_done])
                if is_last:
                    return False
                _heartbeat.tick()
                _now = time.time()
                if _now - _last_print >= 2.0:
                    # sys.__stdout__, NOT print()/sys.stdout: callers that
                    # stream generation swap sys.stdout for a shim that treats
                    # every write() as generated model text.
                    _chunk_done = _done - _reused_len   # this prefill's own progress
                    print(f"[prefill] req #{_REQ_ID.get()} "
                          f"(thread={threading.current_thread().name}) "
                          f"{_chunk_done}/{n_prefill} tok "
                          f"({100*_chunk_done/n_prefill:.0f}%) in "
                          f"{_now - _chunk_t0:.1f}s "
                          f"({_chunk_done/(_now - _chunk_t0):.0f} tok/s)",
                          file=sys.__stdout__, flush=True)
                    if os.environ.get("QV_MEM_DEBUG"):
                        print("[mem-debug] " + _mem_breakdown(past, drafter),
                              file=sys.__stdout__, flush=True)
                    _last_print = _now
                # abort_check between chunks: a disconnect or the VRAM ceiling can stop a long prefill partway.
                return abort_check is not None and bool(abort_check())

            ci = 0
            while ci < len(_chunks):
                c0, c1 = _chunks[ci]
                is_last = c1 >= n_prefill
                if _pp is not None and not is_last and _fused_ready(past):
                    # Two-card overlapped run (pipe_prefill.py): every chunk up to
                    # the last one or the prompt anchor, whichever comes first.
                    run, j = [], ci
                    while j < len(_chunks) and _chunks[j][1] < n_prefill:
                        run.append(_chunks[j])
                        j += 1
                        if _anchor_at is not None and run[-1][1] == _anchor_at:
                            break
                    if len(run) >= 2:
                        _stop = []
                        _pp.run(prefill_ids, past, _reused_len, run, capture,
                                lambda i, b0, b1, f: bool(_after(b0, b1, f) and not _stop.append(1)))
                        ci += len(run)
                        if _stop:
                            _prefill_aborted = True
                            break
                        continue
                _pq = None
                if _PROFILE_PREFILL[0] > 0 and ci % 4 == 0:      # QV_BAND_PROFILE_PREFILL=N: every 4th chunk, N times
                    _PROFILE_PREFILL[0] -= 1
                    torch.cuda.synchronize()
                    _pq = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                             torch.profiler.ProfilerActivity.CUDA],
                                                 record_shapes=True)
                    _pq.__enter__()
                    _pq_t = time.perf_counter()
                try:
                    res = _forward_with_hidden(
                        model, prefill_ids[:, c0:c1], past,
                        want_hidden and (is_last or bool(capture)),
                        logits_last_only=True, capture_layers=capture)
                except BaseException:
                    if _pq is not None:              # never leave the profiler running after a failed chunk
                        _pq.__exit__(None, None, None)
                    raise
                if _pq is not None:
                    torch.cuda.synchronize()
                    _pq.__exit__(None, None, None)
                    print(f"[profile-prefill] chunk {c1 - c0} tok at {c0}: "
                          f"{(time.perf_counter() - _pq_t) * 1e3:.0f} ms\n"
                          + _pq.key_averages(group_by_input_shape=True).table(
                              sort_by="self_cuda_time_total", row_limit=40, max_name_column_width=30,
                              max_shapes_column_width=70),
                          file=sys.__stdout__, flush=True)
                past = res.past_key_values
                ci += 1
                if _after(c0, c1, res.layer_features if capture else None):
                    _prefill_aborted = True
                    break
            if _prof:
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
            if not _prefill_aborted:
                # The last chunk deliberately skips the throttled print
                # above (is_last is always excluded there, since decode
                # starts right after with nothing left to wait for) -- but
                # that means the log's last visible number was whatever
                # the final throttled tick happened to land on (97%, 84%,
                # ...), never 100%, which reads as "did it actually
                # finish?" from outside. One unconditional line once the
                # loop is done removes that ambiguity for free.
                _now = time.time()
                _chunk_done = _done - _reused_len
                print(f"[prefill] req #{_REQ_ID.get()} "
                      f"(thread={threading.current_thread().name}) "
                      f"{_chunk_done}/{n_prefill} tok (100%) in "
                      f"{_now - _chunk_t0:.1f}s "
                      f"({_chunk_done/max(_now - _chunk_t0, 1e-9):.0f} tok/s)",
                      file=sys.__stdout__, flush=True)
        # Hand a CHUNKED prefill's cached blocks back to the driver before
        # decode. On a 16 GB card (V100) the allocator otherwise holds the last
        # free memory, and the decode-time Triton kernels (fused qvb8 KV
        # attention) fail at launch with "out of memory" -- no room for their
        # local memory. Only after a chunked prefill: a short prompt leaves
        # nothing worth returning, and an unconditional empty_cache per
        # request costs a device sync plus re-allocation every time.
        if n_prefill > _prefill_chunk and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if _prefill_aborted:
            # Stopped partway through a chunked prefill (disconnect / VRAM
            # ceiling) -- generation never started, so there is nothing to
            # return in out_ids. The cache DOES hold a clean, valid prefix
            # (chunk boundaries are token boundaries), so still offer it
            # back via return_cache: a caller whose next request repeats
            # this same prompt resumes from here instead of from zero.
            cache = (PrefixCache(past, ids[0].tolist()[:_done], _cache_config)
                    if (return_cache and past is not None) else None)
            if return_cache:
                return out_ids, st, cache
            return out_ids, st
        past = res.past_key_values

        # v7-KV: calibrate the prefill cache once, here -- the whole prompt's K/V
        # population exists at this instant and nothing has consumed it yet, so a
        # per-request calibration certifies the same |value - reconstruction| <=
        # tau bound an offline weight calibration would. Decode-step K/V produced
        # after this point are NOT banded (they are appended dense); banding them
        # would need the frozen prefill codebook plus an escape path, which is a
        # separate piece of work.
        if qvb8_resident:
            # Nothing to do here: BandedKVCache already banded the prefill
            # inside its own update(). Explicitly NOT calling
            # compress_cache_inplace, which would decode the banded store back
            # to dense and write it in -- undoing exactly what this mode is for.
            if kv_stats is not None:
                kv_stats.update(past.memory_report())
        elif int8_kv and kv_stats is not None:
            kv_stats.update(past.memory_report())

        prefix_ids = ids[0].tolist()
        if drafter is not None:
            if want_hidden:
                drafter.hidden = res.last_hidden_state[0, -1]
            if hasattr(drafter, "reset"):
                drafter.reset(ids[0].tolist())
        nxt = _select(res.logits[0, -1], temperature, top_k, top_p)
        st.t_prefill = time.perf_counter() - _t_pf0       # the pick above synchronises the GPU
        st.n_prefill = int(prefill_ids.shape[-1])
        out_ids.append(nxt)
        if drafter is not None and hasattr(drafter, "append"):
            drafter.append(nxt)
        st.emitted += 1
        # `prefix_ids` must track exactly the tokens the cache has consumed, so a
        # rebuild reproduces identical state.  The cache covers the prompt only at
        # this point; `nxt` has been emitted but not yet fed to the model, and the
        # loop below feeds out_ids[-1] as the first element of `seq`.
        #
        # Streaming: _IncStreamer does the CJK-safe incremental detokenise
        # in O(1) per token, replacing the old O(seq)-per-token
        # tok.decode(out_ids) that made a long reply slow down as it grew.
        # Created ONCE here (this block is the one-time prefill/first-token
        # path, not the decode loop below).
        _streamer = _IncStreamer(tok) if stream else None
        if _streamer is not None:
            piece = _streamer.feed([nxt])
            if piece:
                print(piece, end="", flush=True)

        _pd = int(os.environ.get("QV_PROFILE_DECODE", "0") or 0)   # profile steps 5..5+N
        _pd_step = 0
        _dp = None                    # QV_BAND_PROFILE_STEPS=N: profile N whole decode iterations of this loop
        while len(out_ids) < max_new and out_ids[-1] != eos:
            if _dp is not None:
                torch.cuda.synchronize()
                _dp.__exit__(None, None, None)
                if os.environ.get("QV_BAND_PROFILE_TRACE"):
                    _dp.export_chrome_trace(os.environ["QV_BAND_PROFILE_TRACE"] + f".{st.steps}.json")
                print(f"[profile-step] one decode iteration: {(time.perf_counter() - _dp_t) * 1e3:.0f} ms\n"
                      + _dp.key_averages().table(sort_by="self_cuda_time_total", row_limit=30,
                                                 max_name_column_width=50),
                      file=sys.__stdout__, flush=True)
                _dp = None
            if _PROFILE_STEPS[0] > 0 and st.steps >= 3:
                _PROFILE_STEPS[0] -= 1
                torch.cuda.synchronize()
                _dp = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                         torch.profiler.ProfilerActivity.CUDA],
                                             with_stack=bool(os.environ.get("QV_BAND_PROFILE_TRACE")))
                _dp.__enter__()
                _dp_t = time.perf_counter()
            if _pd:
                if _pd_step == 5:
                    torch.cuda.synchronize(); torch.cuda.profiler.start()
                elif _pd_step == 5 + _pd:
                    torch.cuda.synchronize(); torch.cuda.profiler.stop()
                _pd_step += 1
            st.steps += 1

            draft: list[int] = []
            draft_logits: list[torch.Tensor] | None = None
            if drafter is not None and k > 0:
                t0 = time.perf_counter()
                draft, draft_logits = drafter.propose(
                    out_ids[-1], k, temperature=temperature, top_k=top_k, top_p=top_p)
                st.t_draft += time.perf_counter() - t0
                st.proposed += len(draft)

            seq = [out_ids[-1]] + draft
            # Snapshot the linear-attention recurrent+conv state BEFORE this
            # step's speculative forward mutates it in place, so a rejection
            # can restore it directly instead of the full-prefix replay
            # below -- see _snapshot_recurrent_state's docstring for why
            # crop() alone cannot undo this (recurrent state has no inverse
            # operation) and why the snapshot is cheap enough to take on
            # every step regardless of whether this one ends up needing it
            # (~1.2ms measured on a 27B/48-linear-layer model -- negligible
            # against a token's own decode cost, let alone against skipping
            # a would-be O(prefix) rebuild).
            recur_snap = _snapshot_recurrent_state(past) if draft else None
            t0 = time.perf_counter()
            if _ring_available:
                past._qv_ring_capture = bool(draft)     # only the step that verifies a draft
            try:
                res = _forward_with_hidden(
                    model, torch.tensor([seq], dtype=torch.long, device=ids.device),
                    past, want_hidden, capture_layers=capture)
            finally:
                if _ring_available:
                    past._qv_ring_capture = False
            st.t_verify += time.perf_counter() - t0
            spec_past = res.past_key_values

            # Build the per-draft-token state ring from what the forward above
            # just captured (Qwen3_5GatedDeltaNet's patched forward stashes
            # q/k/v/g/beta onto each linear-attention layer when
            # past._qv_ring_capture is set -- see ring_snapshot.py). Cheap:
            # only the recurrent-scan kernel runs again per token, not a
            # second model forward. ring_built is False on a non-hybrid
            # model, or if this step had no draft -- either way the older
            # recur_snap/_try_crop/full-replay fallbacks below still apply.
            ring_built = False
            if draft and _ring_available:
                from qv_band.decode.ring_snapshot import build_ring
                ring_built = build_ring(spec_past, len(seq))

            if temperature <= 0:
                # Greedy verification, unchanged: a draft token is kept only where
                # it equals the base model's own argmax at that position, so the
                # emitted stream is token-identical to plain greedy decoding.
                greedy = res.logits[0].argmax(-1).tolist()
                n_acc = 0
                for i, d in enumerate(draft):
                    if greedy[i] != d:
                        break
                    n_acc += 1
                next_tok = greedy[n_acc]
            else:
                # Speculative sampling (Leviathan et al. 2023 / Chen et al. 2023):
                # accept each draft position with probability
                # min(1, p_target(token)/p_draft(token)) and, on the first
                # rejection, emit a token drawn from the residual distribution
                # max(0, p_target - p_draft), renormalised -- this preserves the
                # target model's sampling distribution exactly.  p_draft is one-hot
                # when the drafter has no distribution (n-gram, MTP at temperature
                # 0), which reduces acceptance to a coin flip at p_target(token);
                # an MTP drafter run with temperature > 0 returns its real
                # per-position logits, so the true draft distribution is used.
                n_acc = 0
                next_tok = None
                for i, d in enumerate(draft):
                    p_target = _probs(res.logits[0, i], temperature, top_k, top_p)
                    # draft_logits (the whole list) not being None does
                    # NOT mean every element is real: NGramWithMTPDrafter
                    # pads n-gram-sourced positions with a per-element
                    # None placeholder (see its propose()'s comment --
                    # n-gram has no distribution, only a one-hot pick),
                    # so a list that IS present can still hold None at
                    # position i. Checking only the outer list crashed
                    # here with `TypeError: unsupported operand type(s)
                    # for /: 'NoneType' and 'float'` inside _probs the
                    # first time a draft was ENTIRELY n-gram-sourced
                    # (remaining <= 0 in NGramWithMTPDrafter.propose, so
                    # MTP never even ran) -- _accept_prob/
                    # _resample_residual already correctly treat
                    # p_draft=None as "one-hot", they just never got the
                    # chance because _probs(None, ...) raised first.
                    _dl = draft_logits[i] if draft_logits is not None else None
                    if _dl is None:
                        p_draft = None
                    elif draft_probs:
                        # The drafter sampled from this exact
                        # distribution (DFlash2's selector over its
                        # top-k candidates); re-filtering it would make
                        # p_draft differ from the real proposal.
                        p_draft = _dl
                    else:
                        p_draft = _probs(_dl, temperature, top_k, top_p)
                    if random.random() < _accept_prob(p_target, p_draft, d):
                        n_acc += 1
                    else:
                        next_tok = _resample_residual(p_target, p_draft, d)
                        break
                if next_tok is None:
                    # Every drafted position was accepted (or there was no draft):
                    # emit the target's own next token at the position after the
                    # draft, sampled exactly as a plain step-by-step sample would
                    # be.
                    next_tok = _select(res.logits[0, n_acc], temperature, top_k, top_p)
            st.accepted += n_acc
            if capture:
                # Rows 0..n_acc of the verify batch are seq[0] + the accepted
                # drafts at positions len(prefix_ids)...; they saw exactly the
                # committed prefix (causal), so they are correct whatever
                # recovery path runs below. Taken before `res` is replaced.
                drafter.observe_features(len(prefix_ids),
                                         res.layer_features[:n_acc + 1])
            if drafter is not None and hasattr(drafter, "feedback"):
                drafter.feedback(n_acc, len(draft))

            # A rejected draft used to be unrecoverable on this architecture:
            # crop() restores the linear-attention layers' conv window but
            # not their recurrent state (measured logits off by up to 6.0
            # without it), and the cache object is mutated in place with no
            # pre-draft copy to restore FROM -- until now: recur_snap (taken
            # above, before this step's speculative forward) IS that
            # pre-draft copy. Profiling showed the old full-prefix rebuild
            # dominated speculative decoding's wall time on this
            # architecture (7 rebuilds consuming 7.7s of an 8.6s, 30-token
            # generation) -- growing O(prefix length) every step, since each
            # rebuild replayed the ENTIRE committed history from token 0.
            # Restoring the snapshot is O(1) in prefix length (a fixed ~76MB
            # clone/copy_ on a 27B/48-linear-layer model, ~1ms either way,
            # confirmed empirically), so only the CURRENT step's accepted
            # prefix needs replaying -- never the whole conversation.
            if draft and os.environ.get("QV_SPEC_DEBUG"):
                _kind = "full_reject" if n_acc == 0 else ("partial" if n_acc < len(draft) else "full_accept")
                print(f"[spec-debug] n_acc={n_acc}/{len(draft)} {_kind}", flush=True)
            if draft and n_acc < len(draft):
                st.rebuilds += 1
                t0 = time.perf_counter()
                # Fast path only when there's a recurrent-state snapshot to
                # restore AND the KV/conv-window crop back to the pre-step
                # length actually succeeds (checked up front by
                # _try_crop/_max_safe_drop, not caught via exception after
                # partial damage) -- either failing means "this cache is not
                # in a state the cheap path can recover from this step",
                # and we correctly fall through to the always-safe rebuild
                # rather than proceed with a stale/inconsistent cache.
                fast_path_ok = False
                ring_promoted = False
                if ring_built:
                    # Ring path: exact per-token state already computed by
                    # build_ring() above from data the batched verify forward
                    # itself produced -- promote slot n_acc (O(1) tensor
                    # swap), no replay needed at all (unlike the
                    # snapshot/crop path below): the accepted tokens' K/V
                    # and hidden states from the ORIGINAL batched forward
                    # (`res`) are already correct, only the rejected tail
                    # needs dropping. crop_full_attention_layers skips
                    # linear-attention layers entirely so it never hits the
                    # record_past gate that _try_crop always fails on this
                    # model family.
                    from qv_band.decode.ring_snapshot import promote_ring, crop_full_attention_layers, discard_ring
                    if promote_ring(spec_past, n_acc):
                        crop_full_attention_layers(spec_past, len(draft) - n_acc)
                        fast_path_ok = True
                        ring_promoted = True
                        past = spec_past
                        hidden_idx = n_acc
                        st.t_verify += time.perf_counter() - t0
                    else:
                        discard_ring(spec_past)
                # Engine hook for models WITHOUT recurrent state (pure attention, e.g. Ripple): a model may
                # provide model._qv_rollback(cache, n_drop) -> bool that drops the rejected tail from the
                # cache exactly. The accepted rows of the verify forward are already correct (causal), so
                # nothing is replayed -- O(1) instead of the full-prefix rebuild below. Models that do not
                # define the hook (Qwen, Nemotron) never enter this branch.
                _qv_rb = getattr(model, "_qv_rollback", None)
                if not fast_path_ok and recur_snap is None and _qv_rb is not None:
                    if _qv_rb(spec_past, len(draft) - n_acc):
                        fast_path_ok = True
                        ring_promoted = True        # "fully handled, no replay" (shared with the ring path)
                        past = spec_past
                        hidden_idx = n_acc
                        st.t_verify += time.perf_counter() - t0
                if not fast_path_ok and recur_snap is not None:
                    # Fast path: restore the linear-attention state to
                    # exactly what it was before this step's speculative
                    # forward polluted it, crop the full-attention layers'
                    # KV (and the linear layers' conv window, which DOES
                    # have a working crop -- see _max_safe_drop/_try_crop)
                    # back to the pre-step length, then replay only
                    # seq[0]+accepted drafts (never more than the draft
                    # length + 1 tokens) against the now-correct `past`, appending to
                    # the EXISTING cache instead of rebuilding one from
                    # scratch. Equivalent result to the old rebuild path
                    # (same tokens end up committed, same model state) at a
                    # cost that does not grow with how much has already
                    # been generated. Deliberately does NOT touch
                    # prefix_ids or drafter.pos/append here -- the shared
                    # bookkeeping below (prefix_ids.append/extend,
                    # drafter.append() inside the emit loop) already does
                    # exactly that for BOTH this branch and the accepted-
                    # in-full branch, using `seq[0]`/`draft[:n_acc]`/
                    # `next_tok` uniformly; duplicating it here (an earlier
                    # version of this patch did, with a `continue` to skip
                    # the shared block) skipped emitting `next_tok` to
                    # out_ids/the stream/the EOS check entirely -- caught
                    # before shipping, not a theoretical concern.
                    _restore_recurrent_state(past, recur_snap)
                    fast_path_ok = _try_crop(past, len(seq))  # undo this step's KV/conv growth
                    if not fast_path_ok:
                        # _try_crop can never succeed on this model family (its
                        # conv window has no margin, see _max_safe_drop), and
                        # this branch is the only one left when the ring path
                        # is unavailable (newer transformers moved the kernels
                        # the ring patch needs). The linear-attention layers
                        # were just restored EXACTLY from the snapshot, so only
                        # the full-attention KV still needs cropping. Without
                        # this every rejected draft re-ran the whole prefix.
                        try:
                            from qv_band.decode.ring_snapshot import crop_full_attention_layers
                            crop_full_attention_layers(past, len(seq))
                            fast_path_ok = True
                        except Exception:
                            fast_path_ok = False
                if ring_promoted:
                    # Already fully handled above: past/hidden_idx are set,
                    # no replay needed. Nothing to do here.
                    pass
                elif fast_path_ok:
                    spec_past = None
                    replay = [seq[0]] + draft[:n_acc]
                    res = _forward_with_hidden(
                        model, torch.tensor([replay], dtype=torch.long, device=ids.device),
                        past, want_hidden)
                    st.t_verify += time.perf_counter() - t0
                    past = res.past_key_values
                    hidden_idx = -1
                else:
                    # Fallback: this cache has no linear-attention layers to
                    # snapshot (e.g. a plain-attention model, or a cache
                    # class _snapshot_recurrent_state doesn't recognise) --
                    # keep the original full-rebuild behaviour, unchanged.
                    #
                    # Drop the polluted speculative cache and its forward's
                    # activations BEFORE building the replacement, so the old
                    # bytes are free for the rebuild's own forward instead of
                    # both being resident at once. On a long or degenerate
                    # generation this path runs every step; without the drop
                    # the allocator accumulates freed-but-cached blocks that
                    # fragment against each rebuild's slightly different sizes,
                    # which is a large part of why a runaway generation's VRAM
                    # climbed far past the KV cache's own footprint. empty_cache
                    # every 64th rebuild only -- it is a full device sync, too
                    # costly to call every step, but periodic trimming bounds
                    # the fragmentation.
                    spec_past = None
                    res = None
                    if torch.cuda.is_available() and st.rebuilds % 64 == 0:
                        torch.cuda.empty_cache()
                    # Rebuild with the SAME cache class the run started with -- falling
                    # back to a plain DynamicCache here would silently drop the int8
                    # cache mid-generation and quietly give back the memory saving.
                    if qvb8_resident:
                        from qv_band.kv.banded_cache import BandedKVCache
                        from qv_band.kv.codec import KV_DEFAULT_TAU
                        fresh = BandedKVCache(config=model.config,
                                              tau=kv_tau or KV_DEFAULT_TAU,
                                              residual_length=int8_residual_length)
                    elif int8_kv:
                        from qv_band.kv.int8_kv_cache import Int8QuantizedCache
                        fresh = Int8QuantizedCache(config=model.config,
                                                   residual_length=int8_residual_length)
                    else:
                        fresh = DynamicCache(config=model.config)
                    # Replay seq[0] AND the accepted drafts: both were fed to the
                    # polluted cache and both are part of the emitted stream, so the
                    # rebuilt cache must cover them too, or the model state and the
                    # token stream desynchronise.  `next_tok` was already chosen from
                    # the original batch's logits at the reject position
                    # (greedy[n_acc] for greedy, the residual draw for sampling) --
                    # its context is exactly this replay, so no recomputation is
                    # needed.  `res` now spans the replayed prefix, so the hidden
                    # state the drafter needs sits at its LAST position.
                    replay = prefix_ids + [seq[0]] + draft[:n_acc]
                    # Chunked like the prefill: this re-runs the WHOLE prefix, and
                    # one unchunked forward builds a (heads x len x len) attention
                    # matrix (3+ GB at a 3k-token prompt without flash-attn).
                    _rt = torch.tensor([replay], dtype=torch.long, device=ids.device)
                    _step = max(1, _prefill_chunk)
                    for _c0 in range(0, _rt.shape[-1], _step):
                        _last = _c0 + _step >= _rt.shape[-1]
                        res = _forward_with_hidden(
                            model, _rt[:, _c0:_c0 + _step], fresh,
                            want_hidden and _last, logits_last_only=True)
                        fresh = res.past_key_values
                    st.t_verify += time.perf_counter() - t0
                    past = res.past_key_values
                    hidden_idx = -1
            else:
                past = spec_past
                # The whole draft was accepted (or there was no draft): the cache
                # kept everything it saw, and the next token's prediction lives at
                # position n_acc == len(draft) of the speculative batch.
                hidden_idx = n_acc
                if ring_built:
                    # Nothing to promote -- the full batch was accepted, so
                    # spec_past is already exactly right. Drop the ring so it
                    # doesn't hold GPU memory until the next step's build_ring
                    # overwrites it anyway.
                    from qv_band.decode.ring_snapshot import discard_ring
                    discard_ring(spec_past)

            emit = draft[:n_acc] + [next_tok]
            stop = False
            _newly: list[int] = []
            for t in emit:
                if len(out_ids) >= max_new:
                    stop = True
                    break
                out_ids.append(t)
                _newly.append(t)
                if drafter is not None and hasattr(drafter, "append"):
                    drafter.append(t)
                st.emitted += 1
                if t == eos:
                    stop = True
                    break
            # Stream the whole batch of newly emitted tokens through
            # _IncStreamer together (a lone token can split a multi-byte CJK
            # codepoint; a whole draft batch landing at once more so).
            if _streamer is not None and _newly:
                piece = _streamer.feed(_newly)
                if piece:
                    print(piece, end="", flush=True)

            # Invariant: prefix_ids == exactly the tokens the cache has consumed.
            # This step fed seq[0] plus the accepted drafts.  The token in
            # `emit[-1]` (the next token) has been emitted but NOT fed to the
            # model -- the next iteration feeds it as seq[0] -- so it is
            # deliberately not counted here; counting it would make a later
            # rebuild replay it twice.
            prefix_ids.append(seq[0])
            prefix_ids.extend(draft[:n_acc])

            if drafter is not None and want_hidden:
                drafter.hidden = res.last_hidden_state[0, hidden_idx]

            _heartbeat.tick()

            if abort_check is not None and abort_check():
                stop = True
            if stop:
                break
        if _dp is not None:                  # the loop ended inside a profiled iteration: close it, or it stays on
            _dp.__exit__(None, None, None)
            _dp = None

        if _streamer is not None:
            # Flush whatever the withhold-trailing-U+FFFD logic held back.
            # If generation ended mid-codepoint (max_new can cut at any
            # token boundary) this prints the raw replacement char rather
            # than dropping the tail -- an honest truncation artifact beats
            # losing the last visible token.
            tail = _streamer.finish()
            if tail:
                print(tail, end="", flush=True)

    finally:
        # No attention-implementation swap to undo any more (the gather
        # path that needed one has been removed) -- kept as a no-op
        # try/finally rather than dedenting the whole prefill+decode body
        # above, which would be a large, purely mechanical diff for no
        # behavioural change.
        pass

    # Refresh kv_stats against the FINAL cache state before returning --
    # the only earlier update() call (right after prefill, above) measures
    # int8_kv before the residual buffer has had any chance to fill past
    # residual_length and flush to int8 (default 128 tokens; a prompt
    # shorter than that leaves the cache 100% fp16 at that snapshot no
    # matter how long the SUBSEQUENT generation runs), so the number shown
    # to the user was effectively always "roughly 100%, ~2.0 B/elem"
    # regardless of the real end-of-generation compression -- confirmed by
    # direct measurement: after 300 decode steps the SAME cache that
    # reported 100%/2.0 B/elem right after prefill was actually at 59.9%/
    # 1.2 B/elem. qvb8_resident does not have this specific problem (its
    # prefill is banded immediately, not threshold-gated), but decode-step
    # growth is unbanded there too (see the comment above), so refreshing
    # here is strictly more accurate for both paths, not just int8_kv's.
    if kv_stats is not None and past is not None and (int8_kv or qvb8_resident):
        kv_stats.update(past.memory_report())

    if not return_cache:
        return out_ids, st

    # `prefix_ids` is exactly what `past` has consumed (see its own
    # invariant comment in the loop above -- it always lags out_ids by the
    # one most-recently-emitted, not-yet-fed-back token). A caller that
    # appends this call's full reply to its next request's `ids` -- e.g. a
    # chat history turn -- will therefore always present that trailing
    # token again, so PrefixCache.reusable_prefix_len's exact-prefix check
    # against `prefix_ids` self-heals: the next call forwards just that one
    # token plus whatever is genuinely new. None if there is nothing to
    # save (no cache object at all -- e.g. bf16 with no drafter).
    cache = (PrefixCache(past, list(prefix_ids), _cache_config)
             if past is not None else None)
    return out_ids, st, cache


def _grow_linear_attention_batch(past) -> None:
    """Append ONE new, zero-initialised batch row to every linear-
    attention layer's conv_states/recurrent_states, so the cache's
    linear-attention layers agree with BandedKVCache.add_slot() having
    just grown the full_attention layers' batch axis by one slot.

    Called by ContinuousBatchEngine.add_request() on every mid-flight
    admission, right after BandedKVCache.add_slot().

    WHY THIS IS NEEDED: BandedKVCache.add_slot() (kv_cache_qvb8.py) only
    touches BandedKVLayer instances -- confirmed by a real crash before
    this existed (RuntimeError: Sizes of tensors must match except in
    dimension 2 ... inside Qwen3_5GatedDeltaNet.forward's
    torch.cat([conv_state, mixed_qkv], ...)), because the linear-
    attention layer's conv_state stayed at batch=1 while everything else
    moved to batch=2. There is no public "add one batch row" API on
    LinearAttentionLayer (checked: only lazy_initialization, which
    REPLACES the whole state, and batch_repeat_interleave/
    batch_select_indices, which reindex an EXISTING batch axis, not grow
    it) -- appending a zero row via torch.cat is the same shape
    lazy_initialization's own torch.zeros_like would produce for a fresh
    sequence, so a new slot's linear-attention state starts identical to
    what a brand-new single-sequence generation would see.
    """
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    for layer in getattr(past, "layers", []) or []:
        if not isinstance(layer, LinearAttentionCacheLayerMixin):
            continue
        cs = getattr(layer, "conv_states", None)
        rs = getattr(layer, "recurrent_states", None)
        for store in (cs, rs):
            for i, t in (store or {}).items():
                if t is None:
                    continue
                # t[:1] is EMPTY when the batch axis is already 0 (every
                # slot was dropped), so zeros_like(t[:1]) would append
                # nothing and leave the state at batch=0 while the next
                # forward runs at batch=1. Build the new row from the
                # tensor's own trailing shape instead, which is correct
                # for any current batch size including 0.
                zero_row = t.new_zeros((1,) + tuple(t.shape[1:]))
                store[i] = torch.cat([t, zero_row], dim=0)


def _linear_attention_state_ready(past) -> bool:
    """True once the linear-attention layers hold conv/recurrent tensors
    with a NON-EMPTY batch axis (they are created lazily by the first
    forward). Until then there is nothing to grow or shrink -- the first
    forward sizes the batch axis from its own input.

    The batch>0 part is load-bearing, not defensive. Dropping the last
    live slot leaves these tensors allocated but at batch=0, and a
    "tensors exist" test would then report ready for the NEXT admission.
    The grow would run against an empty batch axis while the model's own
    lazy_initialization -- which is what actually seeds a fresh
    sequence's state -- was skipped, so the forward arrived with
    conv_state at batch=0 against a batch=1 input: "Expected size 0 but
    got size 1" inside Qwen3_5GatedDeltaNet's torch.cat([conv_state,
    mixed_qkv]). Reporting not-ready at batch=0 hands that first slot
    back to lazy_initialization, which is the same path a cold engine
    takes."""
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    for layer in getattr(past, "layers", []) or []:
        if not isinstance(layer, LinearAttentionCacheLayerMixin):
            continue
        for store in (getattr(layer, "conv_states", None),
                      getattr(layer, "recurrent_states", None)):
            for t in (store or {}).values():
                if t is not None and t.shape[0] > 0:
                    return True
    return False



# ---------------------------------------------------------------------------
# Prefix reuse. An agent client resends its whole conversation every turn, and
# every request used to prefill it from scratch. After a prompt is prefilled up
# to PREFIX_TAIL tokens before its end, that slot's cache is saved; a later
# prompt that starts with exactly those tokens restores it and prefills only the
# rest. The checkpoint stops short of the end because the chat template renders
# the generation prefix ("<|im_start|>assistant\n<think>\n", or with an empty
# think block) differently when it becomes history, and the linear-attention
# state cannot be rolled back over those tokens.
# ---------------------------------------------------------------------------
PREFIX_MIN = 1024          # a shorter prompt is cheaper to prefill than to match
PREFIX_TAIL = 16           # tokens kept out of the checkpoint (see above)
PREFIX_KEEP = 12           # saved prefixes, most recently used first: a conversation each, plus branch points and the chunk boundaries of long prompts (~150 MB of linear state each)


def _bytes_of(obj, seen: set) -> int:
    """Device bytes held by the tensors reachable from `obj` (lists, tuples, dicts and the attributes of
    transformers / qv_band objects), each storage once. Containers are visited once too, so objects
    that refer to each other (cache segments, layers) cannot recurse forever."""
    if isinstance(obj, torch.Tensor):
        key = obj.untyped_storage().data_ptr() if obj.is_cuda else id(obj)
        if key in seen:
            return 0
        seen.add(key)
        return obj.untyped_storage().nbytes() if obj.is_cuda else 0
    if isinstance(obj, (int, float, str, bytes, bool, type(None))):
        return 0
    if ("obj", id(obj)) in seen:
        return 0
    seen.add(("obj", id(obj)))
    if isinstance(obj, dict):
        return sum(_bytes_of(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes_of(v, seen) for v in obj)
    if type(obj).__module__.split(".")[0] in ("transformers", "qv_band") and hasattr(obj, "__dict__"):
        return _bytes_of(list(vars(obj).values()), seen)       # a cache object and its layers
    return 0


def _seed_ngram_with_prompt(drafter, prompt_ids) -> None:
    """Let a drafter's n-gram lookup see the WHOLE prompt.

    A slot's step only carries the prompt TAIL: the head went through chunked
    prefill, or came from a restored prefix. reset(seq) therefore indexed just
    that tail, so on a 60k-token agent prompt with 2k new tokens the n-gram
    could not copy from the other 58k -- the file contents and earlier turns
    an edit quotes back, which is what it is for. (A prompt that fit in one
    step had no such gap, which is the behaviour the n-gram was tuned on.)"""
    ng = drafter if type(drafter).__name__ == "NGramDrafter" else getattr(drafter, "ngram", None)
    if ng is not None and prompt_ids is not None and len(prompt_ids) > len(ng.tokens):
        ng.reset(prompt_ids)


_UNSUPPORTED = object()


def _export_layer_slot(layer, bi: int):
    """Slot `bi` of ONE cache layer, as an entry of _export_slot_state's list, or _UNSUPPORTED."""
    from transformers.cache_utils import DynamicLayer, LinearAttentionCacheLayerMixin
    from qv_band.kv.banded_cache import BandedKVLayer
    if isinstance(layer, BandedKVLayer):
        return ("kv", layer.export_slot(bi))
    if isinstance(layer, LinearAttentionCacheLayerMixin) and not isinstance(layer, DynamicLayer):
        cs, rs = layer.conv_states.get(0), layer.recurrent_states.get(0)
        if cs is None or rs is None or cs.shape[0] <= bi or rs.shape[0] <= bi:
            return _UNSUPPORTED
        return ("lin", (cs[bi].clone(), rs[bi].clone()))
    if (getattr(layer, "conv_states", None) is None
            and getattr(layer, "recurrent_states", None) is None
            and getattr(layer, "keys", None) is None):
        # A stateless block (Nemotron-H's moe/mlp placeholder layers): it contributes nothing to a slot's state, so
        # record an empty entry instead of treating it as unsupported. A layer that DOES carry state but is not one
        # of the known types is unsupported.
        return ("none", None)
    return _UNSUPPORTED


def _export_slot_state(past, bi: int):
    """Slot `bi` of every cache layer, or None if some layer type is not supported."""
    out = []
    for layer in past.layers:
        e = _export_layer_slot(layer, bi)
        if e is _UNSUPPORTED:
            return None
        out.append(e)
    return out


def _import_slot_state(past, bi: int, state) -> bool:
    """Write a saved slot into the new, empty slot `bi`. All-or-nothing: False (nothing
    touched) when the layers do not line up or the linear-attention batch is in a state
    this cannot extend."""
    from qv_band.kv.banded_cache import BandedKVLayer
    layers = list(past.layers)
    if len(layers) != len(state):
        return False
    for layer, (kind, _) in zip(layers, state):
        if (kind == "kv") != isinstance(layer, BandedKVLayer):
            return False
    n_slots = next(len(l._resid_k) for l in layers if isinstance(l, BandedKVLayer))
    ready = _linear_attention_state_ready(past)
    if any(k == "lin" for k, _ in state) and not ready and n_slots != 1:
        return False             # other slots are waiting on a first forward that sizes the batch
    for layer, (kind, st) in zip(layers, state):
        if kind == "kv":
            layer.import_slot(bi, st)
            continue
        if kind == "none":
            continue             # stateless block: nothing to import
        conv, rec = st
        if ready:
            layer.conv_states[0][bi].copy_(conv)
            layer.recurrent_states[0][bi].copy_(rec)
        else:                    # no rows yet (first request, or the last slot just left): seed them
            layer.conv_states[0] = conv.unsqueeze(0).clone()
            layer.recurrent_states[0] = rec.unsqueeze(0).clone()
            layer.is_conv_states_initialized[0] = True
            layer.is_recurrent_states_initialized[0] = True
            layer.has_previous_state[0] = True
            layer.conv_kernel_size[0] = conv.shape[-1]
            if layer.device is None:
                layer.dtype, layer.device = conv.dtype, conv.device
    return True


def _drop_linear_attention_batch(past, bi: int) -> None:
    """Symmetric counterpart to _grow_linear_attention_batch: remove batch
    ROW `bi` from every linear-attention layer's conv_states/
    recurrent_states, keeping them in sync with BandedKVCache.drop_slot()
    having just shrunk the full_attention layers' batch axis by one slot.

    Every row AFTER bi shifts down by one index -- same contract
    BandedKVLayer.drop_slot() already documents (`del lst[bi]`), so a
    caller must remap its own slot->request tracking the same way for
    both calls together.
    """
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    for layer in getattr(past, "layers", []) or []:
        if not isinstance(layer, LinearAttentionCacheLayerMixin):
            continue
        cs = getattr(layer, "conv_states", None)
        rs = getattr(layer, "recurrent_states", None)
        emptied = False
        for store in (cs, rs):
            for i, t in (store or {}).items():
                if t is None:
                    continue
                store[i] = torch.cat([t[:bi], t[bi + 1:]], dim=0)
                if store[i].shape[0] == 0:
                    emptied = True
        if not emptied:
            continue
        # Dropping the LAST live slot leaves these tensors allocated at
        # batch=0. transformers gates re-seeding on three per-state
        # latches that a batch-axis edit does not touch:
        #   is_conv_states_initialized / is_recurrent_states_initialized
        #     gate lazy_initialization, so a batch=0 tensor would never
        #     be rebuilt for the next request;
        #   has_previous_state sends update_conv_state down its
        #     torch.cat([self.conv_states[i], conv_states], dim=-1)
        #     branch instead of the prefill branch -- the exact cat that
        #     raised "Expected size 0 but got size 1".
        # Clearing them returns the layer to its cold state, so the next
        # admission is initialised by the model itself rather than by us
        # guessing what a fresh sequence's conv/recurrent state is.
        for name in ("is_conv_states_initialized",
                     "is_recurrent_states_initialized",
                     "has_previous_state"):
            flags = getattr(layer, name, None)
            if isinstance(flags, dict):
                for i in flags:
                    flags[i] = False


class _BatchSlot:
    """One request's state inside ContinuousBatchEngine -- the
    batched counterpart of speculative_generate's local variables
    (out_ids, drafter, prefix_ids, ...), now one instance per slot instead
    of a single set of locals."""
    __slots__ = ("bi", "out_ids", "drafter", "max_new", "done",
                 "temperature", "top_k", "top_p", "_history_len",
                 "req_id", "stats", "on_token", "just_admitted",
                 "acc_ema", "spec_off", "spec_backoff", "prompt_ids", "prefix_hit", "branch", "_ckpt_at", "k_cur", "acc_len_ema",
                 "admit_no")

    def __init__(self, bi: int, drafter, max_new: int,
                 temperature: float, top_k: int, top_p: float,
                 req_id=None, on_token=None):
        self.bi = bi
        self.out_ids: list[int] = []
        self.drafter = drafter
        self.max_new = max_new
        self.done = False
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._history_len = 0
        self.req_id = req_id
        self.stats = Stats()
        self.on_token = on_token
        self.prompt_ids = None      # the full prompt as a list (to checkpoint / match prefixes)
        self.prefix_hit = 0         # prompt tokens restored from a saved prefix instead of prefilled
        self.k_cur = None           # adaptive draft length (packed engines), None = engine k
        self.acc_len_ema = 4.0      # accepted drafts per step, smoothed
        self._ckpt_at = 0           # slot length at the last chunk-boundary checkpoint
        self.branch = 0             # where this prompt leaves a saved one: save a checkpoint there too
        self.admit_no = 0           # admission order (engine counter): the VRAM-ceiling stop takes the newest
        # True for exactly the step a slot was admitted mid-batch on --
        # that step's forward is a genuine prefill for THIS slot (its
        # whole prompt, not [last_token]+draft), so the decode loop must
        # skip drafting for it and read its first token from the LAST
        # position of its own prompt, not treat the forward as a normal
        # verify step. Cleared right after that step is consumed.
        self.just_admitted = False
        # Adaptive drafting (see SPEC_MIN_ACCEPT): running per-token
        # acceptance, steps left to skip drafting, and the current backoff.
        self.acc_ema = 1.0
        self.spec_off = 0
        self.spec_backoff = SPEC_BACKOFF_MIN


def _find_attn_impl_holder(model):
    """Return whichever config object `model`'s decoder layers actually
    read `_attn_implementation` off at call time (see
    modeling_qwen3_5.py's `self.config._attn_implementation` lookup
    inside self_attn.forward) -- the object ContinuousBatchEngine
    temporarily flips to "qv_varlen" around its own forward call and
    restores immediately after. Checks the plausible attribute
    locations in the order this repo's own loaders populate them
    (run_inference.py sets it on the text_config specifically), falling
    back through get_text_config() for a config shape that nests it.
    Returns None if no holder with this attribute can be found (the
    caller then cannot use the varlen path and must fall back).
    """
    for holder in (
        getattr(getattr(model, "model", None), "config", None),
        getattr(model, "config", None),
    ):
        if holder is None:
            continue
        if hasattr(holder, "_attn_implementation"):
            return holder
        text_cfg = getattr(holder, "get_text_config", None)
        if callable(text_cfg):
            try:
                tc = holder.get_text_config(decoder=True)
            except TypeError:
                tc = holder.get_text_config()
            if hasattr(tc, "_attn_implementation"):
                return tc
    return None


_QV_VARLEN_REGISTERED = False

# Append-in-place dense K/V views for the ragged attention path
# (BandedKVLayer.ragged_kv_views). "0" restores the rebuild-per-step path.
_QV_DENSE_VIEWS = os.environ.get("QV_BAND_DENSE_VIEWS", "1")


def _sdpa_segment(q, k, v, dropout, scaling, window: int | None = None):
    """One sequence's causal attention via SDPA, for boxes without flash-attn.
    q: (1, lq, heads, dim); k, v: (1, lk, kv_heads, dim), lq <= lk. Returns
    (1, lq, heads, dim). The causal mask is bottom-right aligned like
    flash-attn's: query i sees keys j <= lk - lq + i; with `window`, also
    j > lk - lq + i - window (flash-attn's window_size=(window - 1, ...)).
    """
    lq, lk = q.shape[1], k.shape[1]
    mask = None
    if lq > 1 or (window is not None and lk > window):
        mask = torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril(lk - lq)
        if window is not None:
            mask &= torch.ones(lq, lk, dtype=torch.bool, device=q.device).triu(lk - lq - window + 1)
    o = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, dropout_p=dropout, scale=scaling,
        enable_gqa=q.shape[2] != k.shape[2])
    return o.transpose(1, 2)


# Fused qvb8 attention in packed steps (per slot): QV_FUSED_ATTN=0 turns it off here too.
_QV_FUSED_PACKED = os.environ.get("QV_FUSED_ATTN", "1") != "0"
_FLASH_OK: dict = {}


def _flash_ok(device) -> bool:
    """flash-attn usable on `device`: a CUDA card of sm_80 or newer with the library installed (V100 = sm_70 has no
    flash-attn kernels). QV_ATTN_FLASH=0 forces the no-flash path on any card (to test the sm_70 path elsewhere)."""
    if os.environ.get("QV_ATTN_FLASH", "1") == "0":
        return False
    key = str(device)
    if key not in _FLASH_OK:
        ok = False
        if torch.device(device).type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
            try:
                import flash_attn  # noqa: F401
                ok = True
            except ImportError:
                pass
        _FLASH_OK[key] = ok
    return _FLASH_OK[key]


def _qv_packed_attention(layer, query, segs, dropout, scaling):
    """Attention for a PACKED step: query is (1, heads, T, head_dim) holding every slot's new tokens back to
    back; segs = [(slot, offset, length)]. Each segment attends its own slot's history (already appended by
    update()), bottom-right causal, sliding window if the layer has one. No padded column exists anywhere.

    Kernel per GPU (fastest measured there):
      full-attention layer, decode / verify segment (<= 16 tokens from sm_80) with a banded history >= 1024: the
        fused qvb8 kernel straight from the slot's ids (fused_attn.segment_attention); the slot's dense view is
        released. Without flash-attn (sm_70, or QV_ATTN_FLASH=0) the fused kernel takes EVERY length, prefill
        chunks too (the sm_70 CUDA kernel tiles any query), so no dense view of the history is built.
      full-attention layer, anything else: flash-attn on the slot's dense view; without flash-attn, SDPA on it.
      sliding-window layer: the last window of the slot (O(window)); flash-attn with window_size, or SDPA with the
        same causal + window mask."""
    out = query.new_empty((1, query.shape[2], query.shape[1], query.shape[3]))
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    flash = _flash_ok(query.device)
    win = getattr(layer, "window", None)
    if win is not None:
        for bi, off, n in segs:
            q_i = query[:, :, off:off + n].transpose(1, 2)
            k_i, v_i = layer.window_kv_slot(bi, win - 1 + n, query.dtype, query.device)
            if flash:
                from flash_attn import flash_attn_func
                o_i = flash_attn_func(q_i, k_i.transpose(1, 2), v_i.transpose(1, 2), dropout_p=dropout,
                                      softmax_scale=scale, causal=True, window_size=(win - 1, win - 1))
            else:
                o_i = _sdpa_segment(q_i, k_i.transpose(1, 2), v_i.transpose(1, 2), dropout, scale, window=win)
            out[:, off:off + n] = o_i.to(out.dtype)
        return out, None
    if _QV_FUSED_PACKED and query.is_cuda:
        from qv_band.kv import banded_cache as _bc, fused_attn as _fa
        mq = _bc._fused_max_q(query.device) if flash else None
        rest = []
        for bi, off, n in segs:
            fk = None
            if (mq is None or n <= mq) and layer._banded_len[bi] >= _bc._FUSED_MIN_BANDED:
                fk = _fa.slot_fused(layer, bi)
            if fk is None:
                rest.append((bi, off, n))
                continue
            layer.release_sm_view(bi)
            out[:, off:off + n] = _fa.segment_attention(
                layer, fk, bi, query[:, :, off:off + n].transpose(1, 2), scale)
        if not rest:
            return out, None
        segs = rest
    views = None
    if _QV_DENSE_VIEWS != "0":
        views = layer.ragged_kv_views(query.dtype, [s for s, _, _ in segs])
    per_k = per_v = None
    for bi, off, n in segs:
        q_i = query[:, :, off:off + n].transpose(1, 2)
        if views is not None:
            k_i, v_i = views[bi]
        else:
            if per_k is None:
                per_k, per_v = layer.reconstruct_per_slot(query.dtype)
            k_i, v_i = per_k[bi].transpose(1, 2), per_v[bi].transpose(1, 2)
        if flash:
            from flash_attn import flash_attn_func
            o_i = flash_attn_func(q_i, k_i, v_i, dropout_p=dropout, softmax_scale=scale, causal=True)
        else:
            o_i = _sdpa_segment(q_i, k_i, v_i, dropout, scale)
        out[:, off:off + n] = o_i.to(out.dtype)
    return out, None


def _qv_varlen_attention_forward(module, query, key, value, attention_mask,
                                 dropout: float = 0.0, scaling: float | None = None,
                                 **kwargs):
    """Custom AttentionInterface entry, registered as "qv_varlen" --
    SGLang/vLLM-shaped ragged attention for BandedKVCache's full_attention
    layers, replacing the padded-dense-tensor + mask approach entirely.

    WHY THIS EXISTS (the real architectural fix, not another mask patch):
    every earlier fix in this session (4D causal mask, dict-mask for
    flash-attn, real_lens pad-stripping) was still built on top of ONE
    padded (batch, heads, seq_max, head_dim) K/V tensor per step --
    BandedKVLayer._pad_and_stack pads every slot's genuinely different
    history length up to a shared width, and every subsequent bug this
    session (mask-width formula, pad-corrupts-cache, per-layer crop
    desync, flash-attn 4D/2D mismatch) came from trying to describe that
    padding correctly to whichever kernel was running. SGLang/vLLM never
    have this class of bug because they never pad in the first place --
    each request's real tokens are concatenated flat and the kernel is
    told the boundaries via cu_seqlens, so there is no "fake" data a
    mask needs to hide.

    This function reproduces that shape for the ATTENTION CALL ONLY
    (query projection / RoPE / embeddings above this point still run on
    the model's own padded (batch, seq_max) input_ids -- rewriting those
    too would mean abandoning transformers' own decoder-layer forward
    entirely, a much larger change; see this function's calling context
    for the full scope decision). Concretely:
      - key/value: NOT read from the padded `key`/`value` arguments at
        all (those are BandedKVLayer.update()'s padded fallback, unused
        here) -- instead read straight from
        past_key_values.layers[layer_idx].reconstruct_per_slot(), the
        UNPADDED per-slot list update() always populates as a side
        effect (see kv_cache_qvb8.py's own docstring), then flattened.
      - query: the padded (batch, heads, seq_max, head_dim) tensor IS
        stripped of its own padding here, using
        past_key_values._step_real_lens (the same per-step real-length
        list ContinuousBatchEngine.step() already sets for update()'s
        pad-stripping, reused here for the query side).
      - flash_attn_varlen_func(q_flat, k_flat, v_flat, cu_seqlens_q,
        cu_seqlens_k, causal=True) computes each slot's attention over
        exactly its own real tokens, genuinely independent per slot
        (SDPA/flash attention was always row-independent -- the fix
        removes the PADDING each row carried, not any cross-row mixing,
        which was never the actual problem).
      - the flat output is scattered back into a (batch, seq_max, ...)
        tensor at the same padded positions query arrived at, since the
        decoder layer's own reshape/o_proj downstream still expects that
        shape -- only the attention math itself is ragged.
    """
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        flash_attn_varlen_func = None   # per-segment SDPA fallback below
    if flash_attn_varlen_func is not None and query.is_cuda and torch.cuda.get_device_capability(query.device)[0] < 8:
        flash_attn_varlen_func = None   # imports fine but "only supports Ampere or newer" (V100): SDPA per segment
    from qv_band.kv.banded_cache import BandedKVLayer

    # past_key_values is a named parameter of Qwen3_5Attention.forward,
    # not forwarded into attention_interface(...)'s own kwargs (only the
    # decoder layer's **kwargs passthrough is, which does not include
    # it -- confirmed by reading modeling_qwen3_5.py's self_attn.forward
    # directly: `attention_interface(self, query_states, key_states,
    # value_states, attention_mask, dropout=..., scaling=..., **kwargs)`
    # has no past_key_values in that call at all). ContinuousBatchEngine.
    # step() therefore stashes the active BandedKVCache directly on the
    # model object (`model._qv_active_past`) immediately before its
    # forward call, and clears it right after -- the same pattern
    # `_step_real_lens` already uses on the cache itself for the same
    # reason (no other channel exists to hand this function per-call
    # context through the stock model forward).
    # `is None`, not `or`: a cache object with no tokens yet is falsy, and
    # `or` would then silently swap in the stashed batch cache, so a foreign
    # caller's first forward would read and write another request's KV.
    past = kwargs.get("past_key_values")
    if past is None:
        past = getattr(module, "_qv_active_past", None)
    layer_idx = getattr(module, "layer_idx", None)
    if past is None or layer_idx is None:
        raise RuntimeError(
            "qv_varlen attention requires an active BandedKVCache "
            "(module._qv_active_past, set by ContinuousBatchEngine.step) "
            "and module.layer_idx -- got past=%r layer_idx=%r; this "
            "attention implementation is only valid for "
            "ContinuousBatchEngine's own forward calls." % (past, layer_idx))
    layer = past.layers[layer_idx]
    if not isinstance(layer, BandedKVLayer):
        raise RuntimeError(
            "qv_varlen attention: layer %d is not a BandedKVLayer -- this "
            "attention implementation is only valid for a BandedKVCache."
            % layer_idx)

    from qv_band.kv.step_ctx import current as _step_ctx
    _ctx = _step_ctx()                                  # this thread's engine step (qv_band/kv/step_ctx.py)
    segs = _ctx.segs if _ctx is not None else None
    if segs is not None:
        return _qv_packed_attention(layer, query, segs, dropout, scaling)

    # Sliding-window layer (BandedKVLayer.window, e.g. Ripple's local layers; Qwen has none): each slot attends
    # only its last window keys, so decode just that tail (O(window), not O(context)) and let flash-attn apply
    # the window with bottom-right causal alignment -- the same rule as the model's own window mask.
    win = getattr(layer, "window", None)
    if win is not None:
        if flash_attn_varlen_func is None:
            raise RuntimeError("qv_varlen: a sliding-window layer needs flash-attn (window_size)")
        from flash_attn import flash_attn_func
        batch, num_heads, seq_max, head_dim = query.shape
        real = (_ctx.real_lens if _ctx is not None else None) or [seq_max] * batch
        out = query.new_zeros((batch, seq_max, num_heads, head_dim))
        for bi in range(batch):
            rl = real[bi]
            if rl <= 0:
                continue
            k_i, v_i = layer.window_kv_slot(bi, win - 1 + rl, query.dtype, query.device)
            q_i = query[bi:bi + 1, :, seq_max - rl:, :].transpose(1, 2)
            o_i = flash_attn_func(q_i, k_i.transpose(1, 2), v_i.transpose(1, 2), dropout_p=dropout,
                                  softmax_scale=scaling, causal=True, window_size=(win - 1, win - 1))
            out[bi:bi + 1, seq_max - rl:] = o_i.to(out.dtype)
        return out, None

    # FAST PATH: attend straight off the per-slot append-in-place dense
    # views (BandedKVLayer.ragged_kv_views) -- no per-step history rebuild,
    # no cross-slot cat, no transpose. One kernel launch per active slot
    # instead of one varlen launch; slots per step are few, and each launch
    # now skips three history-sized copies.
    real_lens_fast = _ctx.real_lens if _ctx is not None else None
    if real_lens_fast is not None and _QV_DENSE_VIEWS != "0":
        rows = [bi for bi in range(query.shape[0]) if real_lens_fast[bi] > 0]
        views = layer.ragged_kv_views(query.dtype, rows) if rows else {}
        if views is not None:
            batch, num_heads, seq_max, head_dim = query.shape
            out = query.new_zeros((batch, seq_max, num_heads, head_dim))
            for bi in rows:
                rl_q = real_lens_fast[bi]
                k_i, v_i = views[bi]
                # (1, rl_q, heads, head_dim): real content is the LAST rl_q
                # columns (left-padding convention).
                q_i = query[bi:bi + 1, :, seq_max - rl_q:, :].transpose(1, 2)
                if flash_attn_varlen_func is not None:
                    from flash_attn import flash_attn_func
                    o_i = flash_attn_func(q_i, k_i, v_i, dropout_p=dropout,
                                          softmax_scale=scaling, causal=True)
                else:
                    o_i = _sdpa_segment(q_i, k_i, v_i, dropout, scaling)
                out[bi:bi + 1, seq_max - rl_q:] = o_i.to(out.dtype)
            return out, None

    # Decode this layer's banded store to ragged per-slot K/V for THIS
    # call only. Nothing keeps a reference to the result: it is consumed
    # by the kernel below and freed when this function returns, so the
    # resident cache stays banded (certified-bounded) rather than
    # carrying a full dense shadow copy of every slot's history.
    per_slot_k, per_slot_v = layer.reconstruct_per_slot(query.dtype)
    real_lens_q = _ctx.real_lens if _ctx is not None else None
    if real_lens_q is None:
        raise RuntimeError(
            "qv_varlen attention: no step real lengths -- ContinuousBatchEngine must run its forward inside "
            "step_ctx.active(real_lens=...) (query-side real lengths), the list update() itself reads.")
    batch, num_heads, seq_max, head_dim = query.shape
    _, num_kv_heads, _, _ = key.shape
    device, dtype = query.device, query.dtype
    # Slots contributing ZERO query tokens this call are left out of the
    # packing entirely rather than entered as empty segments: flash-attn
    # derives its per-sequence work from cu_seqlens, and a zero-length
    # entry corrupts that bookkeeping (confirmed by direct reproduction
    # -- "RuntimeError: shape '[4, 2, 4, 256]' is invalid for input of
    # size 2048" inside _flash_attn_varlen_forward). Such a slot has
    # nothing to compute anyway, and its output row stays zero, exactly
    # as the scatter-back below leaves it.
    q_parts, k_parts, v_parts = [], [], []
    cu_q, cu_k = [0], [0]
    packed_rows: list[int] = []
    for bi in range(batch):
        rl_q = real_lens_q[bi]
        if rl_q <= 0:
            continue
        # query is (batch, heads, seq_max, head_dim); real content is the
        # LAST rl_q columns (left-padding convention, see
        # _build_varlen_inputs -- the only input builder feeding this
        # path).
        q_parts.append(query[bi, :, seq_max - rl_q:, :].transpose(0, 1))
        k_bi = per_slot_k[bi][0].transpose(0, 1)   # (kv_len_i, kv_heads, head_dim)
        v_bi = per_slot_v[bi][0].transpose(0, 1)
        k_parts.append(k_bi)
        v_parts.append(v_bi)
        cu_q.append(cu_q[-1] + rl_q)
        cu_k.append(cu_k[-1] + k_bi.shape[0])
        packed_rows.append(bi)
    out = query.new_zeros((batch, seq_max, num_heads, head_dim))
    if not packed_rows:
        # Every row is a keepalive/zero-length query this step (can
        # happen transiently) -- nothing to compute either way.
        return out, None
    q_flat = torch.cat(q_parts, dim=0).to(dtype)
    k_flat = torch.cat(k_parts, dim=0).to(dtype)
    v_flat = torch.cat(v_parts, dim=0).to(dtype)
    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=device)
    n_packed = len(packed_rows)
    max_seqlen_q = max(cu_q[i + 1] - cu_q[i] for i in range(n_packed))
    max_seqlen_k = max(cu_k[i + 1] - cu_k[i] for i in range(n_packed))
    if flash_attn_varlen_func is not None:
        attn_flat = flash_attn_varlen_func(
            q_flat, k_flat, v_flat, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=dropout, softmax_scale=scaling, causal=True)
    else:
        # flash-attn not installed (e.g. aarch64/sm_121): run SDPA per
        # segment. Causal mask is bottom-right aligned like flash-attn's:
        # query i of an lq-long segment sees keys j <= lk - lq + i.
        segs = []
        for j in range(n_packed):
            segs.append(_sdpa_segment(
                q_flat[cu_q[j]:cu_q[j + 1]].unsqueeze(0),
                k_flat[cu_k[j]:cu_k[j + 1]].unsqueeze(0),
                v_flat[cu_k[j]:cu_k[j + 1]].unsqueeze(0),
                dropout, scaling)[0])
        attn_flat = torch.cat(segs, dim=0)
    # `packed_rows[j]` is the batch row that packed segment j came from
    # -- rows left out above keep their all-zero output.
    for j, bi in enumerate(packed_rows):
        out[bi, seq_max - real_lens_q[bi]:] = attn_flat[cu_q[j]:cu_q[j + 1]].to(out.dtype)
    return out, None
    # ----------------------------------------------------------------
# ---------------------------------------------------------------------------
# Draft rollback without a replay forward.
#
# A rejected draft used to cost a SECOND full model forward: the recurrent
# (Gated-DeltaNet) state has no partial rollback, so the engine restored the
# pre-step snapshot and re-ran the kept tokens through all 64 layers just to
# rebuild it -- ~230 ms on the 27B, about a quarter of decode time. That
# forward is unnecessary: attention is causal, so the inputs each linear layer
# saw for the kept prefix during the VERIFY forward are exactly what a replay
# would feed it. So the verify forward records, per linear layer, the scan's
# own inputs (post-conv q/k/v, g, beta), the pre-step recurrent state and the
# conv window; on rejection the state after the kept prefix is recomputed from
# those with the single-token recurrent kernel (a few small launches per layer)
# and written straight into the cache. No weights are read.
#
# Capture wraps two module-level functions the stock forward looks up by name
# at call time (torch_chunk_gated_delta_rule, causal_conv1d_fn), so the forward
# itself is not copied; transformers is pinned (see ai_engine/pyproject.toml).
# Anything that goes wrong -- a layer without a capture, a missing kernel --
# falls back to snapshot + replay. QV_BAND_RING=0 disables it.
RING_ENABLED = os.environ.get("QV_BAND_RING", "1") != "0"
# Longest scan (tokens per sequence) that runs on the single-kernel recurrent form instead of the chunked one.
SCAN_RECURRENT_MAX_T = 16
# Per-thread: the batch scheduler thread and an HTTP request thread can both be
# inside a forward; a shared global let one capture land in the other's context.
_RING_TLS = threading.local()


def _ring_ctx_get():
    return getattr(_RING_TLS, "ctx", None)


def _ring_ctx_set(ctx) -> None:
    _RING_TLS.ctx = ctx


_RING_PATCHED = False


def _install_ring_capture(model) -> bool:
    """Wrap the two scan entry points once. Returns False if this
    transformers build does not expose them the way the capture needs."""
    global _RING_PATCHED
    if _RING_PATCHED:
        return True
    lin = next((getattr(dl, "linear_attn", None)
                for dl in getattr(getattr(model, "model", None), "layers", []) or []
                if getattr(dl, "linear_attn", None) is not None), None)
    if lin is None:
        return False
    mod = sys.modules.get(type(lin).__module__)
    orig_chunk = getattr(mod, "torch_chunk_gated_delta_rule", None)
    orig_conv = getattr(mod, "causal_conv1d_fn", None)
    if orig_chunk is None or orig_conv is None:
        return False
    try:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    except Exception:
        return False

    def chunk(query, key, value, g=None, beta=None, initial_state=None, **kw):
        ctx = _ring_ctx_get()
        if ctx is not None and query.shape[1] > 1:
            # initial_state is the LIVE state tensor, overwritten in place by
            # update_recurrent_state at the end of this forward -- clone it.
            ctx["scan"] = (query, key, value, g, beta,
                           initial_state.clone() if initial_state is not None else None)
        if query.shape[1] <= SCAN_RECURRENT_MAX_T and kw.get("cu_seqlens") is None:
            # A verify step scans 8 tokens per request. The chunked kernel is built for long
            # sequences (64-token blocks): it runs ~10 small kernels and ~650 us of Python/launch
            # work per layer, which on the 35B-A3B was over half of a decode step's CPU time. The
            # sequential kernel is one launch, and is exactly what the rollback already uses.
            return fused_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=initial_state,
                output_final_state=kw.get("output_final_state", False),
                use_qk_l2norm_in_kernel=kw.get("use_qk_l2norm_in_kernel", False))
        return orig_chunk(query, key, value, g=g, beta=beta,
                          initial_state=initial_state, **kw)

    def conv(x, *a, **kw):
        ctx = _ring_ctx_get()
        if ctx is not None:
            ctx["win"] = x.clone()
        return orig_conv(x, *a, **kw)

    mod.torch_chunk_gated_delta_rule = chunk
    mod.causal_conv1d_fn = conv
    _RING_PATCHED = True
    return True


def _install_ring_capture_for(module) -> bool:
    """_install_ring_capture, but keyed on one linear-attention module (what the
    single-request path has) instead of the whole model."""
    class _M:                      # minimal stand-in exposing model.model.layers[*].linear_attn
        pass
    holder = _M(); holder.model = _M()
    holder.model.layers = [types.SimpleNamespace(linear_attn=module)]
    return _install_ring_capture(holder)


def _ring_call(orig_forward, module, cache_params, rows, rl, **call_kw):
    """orig_forward(...), recording this call's scan inputs into
    cache_params._ring[layer_idx] when the engine armed capture for the step
    and every row in the call carries a draft (a prompt being prefilled in the
    same step is not captured)."""
    store = getattr(cache_params, "_ring", None) if cache_params is not None else None
    draft_rows = getattr(cache_params, "_ring_rows", ()) if cache_params is not None else ()
    if (store is None or rl is None or rl <= 1
            or not all(r in draft_rows for r in rows)):
        return orig_forward(cache_params=cache_params, **call_kw)
    ctx = {}
    _ring_ctx_set(ctx)
    try:
        out = orig_forward(cache_params=cache_params, **call_kw)
    finally:
        _ring_ctx_set(None)
    if "scan" in ctx and "win" in ctx:
        store.setdefault(module.layer_idx, []).append(
            {"rows": list(rows), "scan": ctx["scan"], "win": ctx["win"]})
    return out


def _run_linear_attn_per_slot(module, hidden_states, cache_params, real_lens,
                              orig_forward, cold_rows=None, **kwargs):
    """Run a Qwen3_5GatedDeltaNet layer ONE SLOT AT A TIME, each at its
    own true (unpadded) length, and stitch the outputs back into the
    padded batch shape the rest of the decoder stack expects.
    WHY THIS IS NECESSARY (real, confirmed bug that nothing else fixes):
    this layer's conv1d + recurrent scan consume every timestep they are
    handed. A left-padded row's pad columns are therefore processed as
    genuine timesteps -- and a ZERO input is NOT a no-op for either a
    conv window or an SSM accumulator, it still advances and mixes into
    the state. Qwen3_5's own apply_mask_to_padding_states (which this
    module calls first thing) only ZEROES the pad positions' hidden
    states; it cannot stop them from being scanned. So masking can never
    fix pad corruption here, unlike on the full-attention side where a
    mask genuinely excludes keys from the softmax.
    Trying to undo it afterwards does not work either: a whole-row
    recurrent-state restore is the only rollback this architecture has
    (see _snapshot_recurrent_state), and on a step where the draft was
    fully ACCEPTED that restore would also discard the accepted tokens'
    own genuine effect on the state, desynchronising this slot's
    recurrent half from its full-attention half (confirmed by direct
    reproduction, both ways round: restoring on pad>0 degraded output
    whenever drafts were accepted, and not restoring let pad corruption
    through -- the two needs are irreconcilable while padding reaches
    this layer at all).
    The only correct fix is to never feed the pad columns in. Per-slot
    execution costs one small forward per row on the steps that actually
    have ragged lengths, and is skipped entirely when every row is the
    same length (the common case).
    `real_lens[bi]` is that slot's true token count for this call; the
    real tokens are the LAST real_lens[bi] columns (left-padding
    convention, see _build_varlen_inputs).
    """
    batch, seq_max, _ = hidden_states.shape
    layer = cache_params.layers[module.layer_idx] if cache_params is not None else None
    out = torch.zeros_like(hidden_states)
    # Rows sharing a true length (and warm/cold state) run TOGETHER as one
    # batched call: the module handles any batch at a fixed length, and a
    # typical mixed step (a few decode rows of width 1-3 plus maybe one
    # prompt) has far fewer distinct widths than rows. Only rows whose
    # lengths genuinely differ are separated.
    if layer is not None:
        # After the last request left, the state keeps its tensors with an EMPTY batch axis, and add_request does
        # not grow an empty state (_linear_attention_state_ready). Two or more requests of different lengths
        # admitted together then reach this function with fewer state rows than slots: index_select on a row that
        # does not exist is a device-side assert, which takes the whole CUDA context (every later request fails
        # until the server restarts). Give every slot its (zero = cold) row first.
        for store in (layer.conv_states, layer.recurrent_states):
            t = store.get(0) if store else None
            if t is not None and t.shape[0] < batch:
                store[0] = torch.cat([t, t.new_zeros((batch - t.shape[0],) + tuple(t.shape[1:]))], dim=0)
    groups: dict = {}
    for bi in range(batch):
        rl = real_lens[bi]
        if rl <= 0:
            continue
        cold = bool(cold_rows[bi]) if cold_rows is not None else None
        groups.setdefault((rl, cold), []).append(bi)
    for (rl, cold), rows in groups.items():
        idx = torch.tensor(rows, device=hidden_states.device)
        hs_i = hidden_states.index_select(0, idx)[:, seq_max - rl:, :]
        if layer is None:
            out[idx, seq_max - rl:, :] = orig_forward(
                hidden_states=hs_i, cache_params=None, attention_mask=None, **kwargs)
            continue
        rows_l = rows
        # Swap this group's rows of conv/recurrent state in as a
        # batch-of-g, run, then write the updated rows back. The module
        # reads conv_states[0]/recurrent_states[0] directly and writes via
        # update_conv_state/update_recurrent_state (an in-place copy_ that
        # requires a matching shape), so a plain row-slice view is not
        # enough -- the dict entries themselves must be temporarily
        # group-sized tensors.
        cs_full = layer.conv_states.get(0) if layer.conv_states else None
        rs_full = layer.recurrent_states.get(0) if layer.recurrent_states else None
        cs_rows = cs_full.index_select(0, idx) if cs_full is not None else None
        rs_rows = rs_full.index_select(0, idx) if rs_full is not None else None
        if cs_rows is not None:
            layer.conv_states[0] = cs_rows
        if rs_rows is not None:
            layer.recurrent_states[0] = rs_rows
        # has_previous_state drives the module's use_precomputed_states,
        # and it is per-LAYER while these rows are per-SLOT. On every
        # mid-flight admission the batch is genuinely heterogeneous: the
        # running rows have real conv state, the just-appended row is all
        # zeros and has never been prefilled. Without this, that new row
        # would take the use_precomputed_states branch and prepend its
        # own zero conv_state as though it were real left context.
        # Saved and restored per group so slots never inherit each other's.
        hps = getattr(layer, "has_previous_state", None)
        hps_saved = hps.get(0) if isinstance(hps, dict) else None
        if isinstance(hps, dict) and cold is not None:
            hps[0] = not cold
        try:
            out[idx, seq_max - rl:, :] = _ring_call(
                orig_forward, module, cache_params, rows_l, rl,
                hidden_states=hs_i, attention_mask=None, **kwargs)
        finally:
            if cs_full is not None:
                cs_full.index_copy_(0, idx, layer.conv_states[0])
                layer.conv_states[0] = cs_full
            if rs_full is not None:
                rs_full.index_copy_(0, idx, layer.recurrent_states[0])
                layer.recurrent_states[0] = rs_full
            # The first forward of a fresh cache with rows of DIFFERENT lengths (several requests admitted before
            # any step, prompts shorter than the chunk): this group's call just created the state, sized for its
            # own rows only. Give it the whole batch (the other rows zero = cold, as a fresh slot is), or the next
            # group indexes a row that does not exist ("srcIndex < srcSelectDimSize" device assert).
            for full, store in ((cs_full, layer.conv_states), (rs_full, layer.recurrent_states)):
                t = store.get(0) if (full is None and store) else None
                if t is not None and t.shape[0] == len(rows) < batch:
                    grown = t.new_zeros((batch,) + tuple(t.shape[1:]))
                    grown.index_copy_(0, idx, t)
                    store[0] = grown
            if isinstance(hps, dict) and hps_saved is not None:
                # Every row that ran here now HAS previous state, so the
                # layer ends this call warm regardless of how it started.
                hps[0] = True
    return out
def _run_mamba_per_slot(module, hidden_states, cache_params, real_lens,
                        orig_forward, **kwargs):
    """Run a Nemotron-H Mamba-2 mixer ONE SLOT AT A TIME, each at its own
    true (unpadded) length, and stitch the outputs back into the padded
    batch shape the rest of the decoder stack expects -- the mamba twin of
    _run_linear_attn_per_slot, which exists for exactly the same reason:
    the conv1d + SSM scan consumes every timestep it is handed, a
    left-padded row's pad columns are therefore processed as genuine
    timesteps, and no mask can stop that.

    `module` is the mixer (NemotronHMamba2Mixer); `orig_forward` is its
    RAW forward (layers/mamba2_hybrid._mamba2_forward, with the _Graphed fast
    path unwrapped by the installer), which reads and writes the state
    for module.layer_idx in place. The state tensors live in the owning
    layer's conv_states[0]/recurrent_states[0] dicts (the cache-level
    conv_states/ssm_states the model reads are views onto them), so
    swapping the LAYER's dict entry to this group's rows for the duration
    of the call -- and copying the updated rows back afterwards -- is the
    whole state management: the model never sees another slot's row, and
    a row with real_lens[bi] == 0 is not run at all (its state is
    untouched), exactly as the Qwen twin skips zero-length rows.
    """
    batch, seq_max, _ = hidden_states.shape
    li = module.layer_idx
    layer = cache_params.layers[li] if cache_params is not None else None
    out = torch.zeros_like(hidden_states)
    # Rows sharing a true length run TOGETHER as one batched call, exactly
    # as _run_linear_attn_per_slot groups them (the mixer handles any
    # batch at a fixed length).
    groups: dict = {}
    for bi in range(batch):
        rl = real_lens[bi]
        if rl <= 0:
            continue
        groups.setdefault(rl, []).append(bi)
    for rl, rows in groups.items():
        idx = torch.tensor(rows, device=hidden_states.device)
        hs_i = hidden_states.index_select(0, idx)[:, seq_max - rl:, :]
        if layer is None:
            out[idx, seq_max - rl:, :] = orig_forward(
                hs_i, cache_params=None, attention_mask=None, **kwargs)
            continue
        cs_full = layer.conv_states[0]
        rs_full = layer.recurrent_states[0]
        layer.conv_states[0] = cs_full.index_select(0, idx)
        layer.recurrent_states[0] = rs_full.index_select(0, idx)
        try:
            out_i = orig_forward(hs_i, cache_params=cache_params,
                                 attention_mask=None, **kwargs)
        finally:
            cs_full.index_copy_(0, idx, layer.conv_states[0])
            rs_full.index_copy_(0, idx, layer.recurrent_states[0])
            layer.conv_states[0] = cs_full
            layer.recurrent_states[0] = rs_full
        out[idx, seq_max - rl:, :] = out_i
    return out


def _run_linear_attn_packed(module, hidden_states, cache_params, segs, orig_forward, cold_rows, **kwargs):
    """Linear-attention layer on a PACKED step (one row, slots back to back; segs = [(slot, offset, length)]).
    The conv/recurrent state is per slot, so each segment is put back on its own slot row (left-padded, as
    _run_linear_attn_per_slot expects -- its pad columns are never scanned), run there, and the output is
    written back to the segment's place in the packed row."""
    n_rows = len(cold_rows) if cold_rows is not None else 1 + max(b for b, _, _ in segs)
    width = max(n for _, _, n in segs)
    padded = hidden_states.new_zeros((n_rows, width, hidden_states.shape[-1]))
    real = [0] * n_rows
    for bi, off, n in segs:
        padded[bi, width - n:] = hidden_states[0, off:off + n]
        real[bi] = n
    if len(set(real)) == 1:
        # every slot has the same length (e.g. a single slot): the stock call on the full batch, exactly as the
        # hook's uniform branch -- it is also what creates the state on the very first forward
        res = _ring_call(orig_forward, module, cache_params, range(n_rows), real[0],
                         hidden_states=padded, attention_mask=None, **kwargs)
    else:
        res = _run_linear_attn_per_slot(module, padded, cache_params, real, orig_forward,
                                        cold_rows=cold_rows, **kwargs)
    out = torch.empty_like(hidden_states)
    for bi, off, n in segs:
        out[0, off:off + n] = res[bi, width - n:]
    return out


def _install_linear_attn_per_slot_hooks(model):
    """Wrap every linear-attention layer's forward so that, on a step
    whose rows have DIFFERENT true lengths, each slot runs separately at
    its own length (see _run_linear_attn_per_slot for why). Covers both
    families this engine serves: Qwen3.5's GatedDeltaNet (dl.linear_attn,
    _run_linear_attn_per_slot) and Nemotron-H's Mamba-2 mixer (dl.mixer
    on a block_type == "mamba" block, _run_mamba_per_slot). Idempotent;
    the wrapper is a no-op passthrough whenever the caller has not set
    per-step real lengths or every row is already the same length, so
    non-batched callers keep the stock behaviour exactly.
    """
    for dl in getattr(getattr(model, "model", None), "layers", []) or []:
        lin = getattr(dl, "linear_attn", None)
        if lin is not None:
            if not getattr(lin, "_qv_per_slot_wrapped", False):
                orig_forward = lin.forward
                def _wrapped(hidden_states, cache_params=None, attention_mask=None,
                             _orig=orig_forward, _mod=lin, **kwargs):
                    from qv_band.kv.step_ctx import current as _step_ctx
                    _c = _step_ctx() if cache_params is not None else None
                    real_lens = _c.real_lens if _c is not None else None
                    cold = _c.cold_rows if _c is not None else None
                    segs = _c.segs if _c is not None else None
                    if segs is not None:
                        return _run_linear_attn_packed(_mod, hidden_states, cache_params, segs, _orig,
                                                       cold, **kwargs)
                    if (real_lens is None
                            or hidden_states.shape[0] != len(real_lens)
                            or len(set(real_lens)) == 1):
                        # No ragged rows this call -- stock path, byte-identical
                        # (plus verify-state capture when the engine armed it).
                        if (real_lens is not None and hidden_states.shape[0] == len(real_lens)
                                and real_lens[0] > 0):
                            return _ring_call(
                                _orig, _mod, cache_params, range(len(real_lens)), real_lens[0],
                                hidden_states=hidden_states, attention_mask=attention_mask,
                                **kwargs)
                        return _orig(hidden_states=hidden_states, cache_params=cache_params,
                                     attention_mask=attention_mask, **kwargs)
                    return _run_linear_attn_per_slot(
                        _mod, hidden_states, cache_params, real_lens, _orig,
                        cold_rows=cold, **kwargs)
                lin.forward = _wrapped
                lin._qv_per_slot_wrapped = True
            continue
        if getattr(dl, "block_type", None) != "mamba":
            continue
        mx = getattr(dl, "mixer", None)
        if mx is None or getattr(mx, "_qv_per_slot_wrapped", False):
            continue
        orig_forward = mx.forward
        # The _Graphed fast path (layers/mamba2_hybrid.install_fast_paths) keys its
        # graphs on the state tensors' addresses and is only safe for the
        # UNIFORM passthrough below; the per-group calls run the raw forward
        # directly so a ragged step never captures or replays a graph against
        # a swapped-in sub-batch state.
        raw_forward = (orig_forward.orig
                       if type(orig_forward).__name__ == "_Graphed"
                       else orig_forward)
        def _wrapped(hidden_states, cache_params=None, attention_mask=None,
                     _orig=orig_forward, _raw=raw_forward, _mod=mx, **kwargs):
            from qv_band.kv.step_ctx import current as _step_ctx
            _c = _step_ctx() if cache_params is not None else None
            real_lens = _c.real_lens if _c is not None else None
            if (real_lens is None
                    or hidden_states.shape[0] != len(real_lens)
                    or len(set(real_lens)) == 1):
                return _orig(hidden_states=hidden_states, cache_params=cache_params,
                             attention_mask=attention_mask, **kwargs)
            return _run_mamba_per_slot(
                _mod, hidden_states, cache_params, real_lens, _raw, **kwargs)
        mx.forward = _wrapped
        mx._qv_per_slot_wrapped = True

def _register_qv_varlen_attention() -> None:
    """Idempotent: registers "qv_varlen" with transformers' global
    AttentionInterface AND excludes it from ALL_MASK_ATTENTION_FUNCTIONS
    so create_causal_mask's own _preprocess_mask_arguments takes its
    documented "TGI/vLLM backends... we don't need a mask" early-exit
    (returns attention_mask=None, no dense/2D mask ever built) -- see
    _qv_varlen_attention_forward's own docstring for why building any
    mask at all was the root of every earlier bug this session found.
    """
    global _QV_VARLEN_REGISTERED
    if _QV_VARLEN_REGISTERED:
        return
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register("qv_varlen", _qv_varlen_attention_forward)
    _QV_VARLEN_REGISTERED = True
def _build_varlen_inputs(slots: list, seqs: list[list[int]], hist_lens: list[int],
                         device) -> tuple[torch.Tensor, torch.Tensor]:
    """input_ids + position_ids for one ragged step.
    Rows still share one rectangular `input_ids` tensor because the
    embedding / QKV projection / RoPE stages ahead of attention are
    per-token elementwise work that does not care about sequence
    boundaries -- packing them flat would buy nothing and would mean
    abandoning transformers' own decoder forward. Every stage that DOES
    care about boundaries is boundary-aware instead of pad-aware:
    attention runs ragged through cu_seqlens (_qv_varlen_attention_
    forward), the linear-attention scan runs per-slot at each row's true
    length (_run_linear_attn_per_slot), and the KV cache only commits
    each row's real tokens (real_lens, see BandedKVLayer.update). So the
    filler columns here are never read by anything -- no mask describes
    them and nothing consumes them.
    Real tokens are RIGHT-aligned in each row (filler on the left), and
    position_ids continue from that row's own history length so RoPE
    sees this step's tokens as contiguous with what came before them in
    that slot.
    """
    lens = [len(s) for s in seqs]
    width = max(lens) if lens else 0
    n = len(seqs)
    # Built on the host and moved once: this used to issue two device-side
    # writes (plus a tensor creation each) per row per step.
    ids_rows = [[0] * (width - len(seq)) + list(seq) for seq in seqs]
    pos_rows = [[0] * (width - len(seq)) + list(range(hist, hist + len(seq)))
                for seq, hist in zip(seqs, hist_lens)]
    both = torch.tensor([ids_rows, pos_rows], dtype=torch.long).to(
        device, non_blocking=True) if n else torch.zeros(2, 0, width, dtype=torch.long, device=device)
    return both[0], both[1]
def _ring_plan(past, ring: dict, bi: int):
    """The linear-attention layers of `past` and their capture of this step for slot `bi`: a list of
    (layer, (capture, row)), empty when the cache holds no recurrent state, None if a layer lacks a usable capture."""
    plan = []
    for li, layer in enumerate(past.layers):
        rs = getattr(layer, "recurrent_states", None)
        if rs is None or rs.get(0) is None:
            continue                      # full-attention layer
        hit = None
        for e in ring.get(li, ()):
            if bi in e["rows"]:
                hit = (e, e["rows"].index(bi))
                break
        if hit is None or hit[0]["scan"][5] is None:
            return None
        plan.append((layer, hit))
    return plan


def _ring_apply(plan, bi: int, t: int) -> None:
    """Rebuild slot `bi`'s state after `t` tokens of the step from a _ring_plan."""
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    for layer, (e, r) in plan:
        q, k, v, g, beta, init = e["scan"]
        K = layer.conv_states[0].shape[-1]
        layer.conv_states[0][bi].copy_(e["win"][r, :, t:t + K])
        _, st = fused_recurrent_gated_delta_rule(
            q[r:r + 1, :t], k[r:r + 1, :t], v[r:r + 1, :t],
            g=g[r:r + 1, :t], beta=beta[r:r + 1, :t],
            initial_state=init[r:r + 1], output_final_state=True,
            use_qk_l2norm_in_kernel=True)
        layer.recurrent_states[0][bi].copy_(st[0])


def _make_banded_cache(config, *, tau, residual_length):
    """The resident banded cache for `config`'s decoder. A Nemotron-H
    hybrid (config carries layers_block_type with mamba blocks) gets
    HybridBandedCache -- banded attention KV plus per-slot Mamba-2
    conv/SSM state and the model-level conv_states/ssm_states surface the
    shipped modeling code reads; everything else gets the plain
    BandedKVCache, unchanged."""
    from qv_band.kv.banded_cache import BandedKVCache, HybridBandedCache
    decoder = config.get_text_config(decoder=True)
    block_types = getattr(decoder, "layers_block_type", None)
    if block_types is not None and any(str(t) == "mamba" for t in block_types):
        return HybridBandedCache(config=config, tau=tau,
                                    residual_length=residual_length)
    return BandedKVCache(config=config, tau=tau,
                         residual_length=residual_length)


class ContinuousBatchEngine:
    """True step-level continuous batching (vLLM/SGLang shape): a
    persistent decode loop that a caller drives ONE STEP AT A TIME via
    step(), checking for newly-arrived requests and admitting them into
    the CURRENTLY-RUNNING batch before every step -- never "collect a
    fixed batch, run it to completion". A request that arrives while
    others are mid-generation is spliced in at the very next step, not
    queued behind the whole in-progress generation.
    WHY THIS EXISTS: serve.py's default path runs one request at a time
    behind a global lock. An earlier attempt batched requests into fixed
    "rounds" (collect for a short window, then run the whole round to
    completion), which still serialises anything that is not almost
    exactly simultaneous -- a late arrival waits out the entire
    in-progress round's generation, not just its own prefill. Here the
    round is gone: add_request()/step() let a request join or leave the
    live batch between ANY two decode steps, the same way vLLM
    re-evaluates batch composition every iteration and llama.cpp's
    server admits a slot as soon as one frees up.
    Mid-flight admission uses add_slot()/_grow_linear_attention_batch();
    finished slots leave via drop_slot()/_drop_linear_attention_batch().
    A newly admitted slot's prefill runs as part of the very next step(),
    alongside whatever decode work the running slots are doing -- so a
    single step legitimately mixes "whole prompt" and "one token" rows.
    NO PADDING IS EVER CONSUMED, matching SGLang: attention packs the
    batch flat and separates requests with cu_seqlens
    (_qv_varlen_attention_forward), the linear-attention conv/recurrent
    scan runs per-slot at each row's true length
    (_run_linear_attn_per_slot), and the KV cache commits only each
    row's real tokens (real_lens, see BandedKVLayer.update). No
    attention mask is built anywhere on this path.
    REQUIRES qvb8_resident (BandedKVCache) -- the per-slot banded store
    is what supplies ragged K/V directly.
    """
    def __init__(self, model, tok, drafter_factory=None, k: int = 8,
                kv_tau: float | None = None, int8_residual_length: int = 128,
                on_heartbeat=None, prefill_chunk: int | None = None):
        self.model = model
        self.tok = tok
        self.drafter_factory = drafter_factory
        self.k = k if drafter_factory is not None else 0
        self.kv_tau = kv_tau
        self.int8_residual_length = int8_residual_length
        # Per-instance chunk size, so a server's --prefill-chunk actually
        # reaches this path. Falls back to the module default when unset.
        # Measured on the 27B (6808-token prompt): 512 -> 324.7 tok/s,
        # 1024 -> 426.5, 2048 -> 459.9, so silently pinning every server
        # to the module default cost ~30% of prefill.
        self.prefill_chunk = (int(prefill_chunk) if prefill_chunk
                              else PREFILL_CHUNK)
        _probe = drafter_factory() if (self.k > 0 and drafter_factory) else None
        self.want_hidden = bool(getattr(_probe, "needs_hidden", False))
        # Target layers a DFlash drafter is fed from (see speculative_generate's
        # `capture`): every committed position of every slot -- prefill
        # chunks, the admitted tail, and each step's seq[0] + accepted drafts.
        self.capture = getattr(_probe, "capture_layers", None)
        eos = tok.eos_token_id
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        self.past = None            # BandedKVCache, built lazily on first admit
        # the V100's decode needs the MLP / projection CUDA graphs here too (CPU-bound launches); not the whole-layer graph
        _install_decode_graphs(model, layer_graphs=False)
        # Decoder layers on other machines (decode/remote_stage.py); empty on one machine. Every cache action below
        # is repeated on each stage for the layers it holds.
        self._stages = getattr(model, "_qv_remote_stages", ())
        self._remote_layers = getattr(model, "_qv_remote_layers", frozenset())
        if self._stages:
            from qv_band.kv.codec import KV_DEFAULT_TAU
            for _st in self._stages:
                _st.reset(kv_tau or KV_DEFAULT_TAU, int8_residual_length)
        self.slots: list[_BatchSlot | None] = []
        self._pending_prompts: dict[int, torch.Tensor] = {}   # bi -> prompt ids, this-step admits
        self._prefix_store: list[dict] = []        # saved prompt prefixes, most recently used first
        # set by the server: () -> set of req_ids whose client is gone; a chunked prefill stops between chunks for them
        self.cancel_check = None
        # set by the server: () -> True when a card is at the --max-memory-gb ceiling (per card under --pipeline).
        # Polled once per step(); see _ceiling_stop. None = no ceiling.
        self.over_ceiling = None
        self._admit_seq = 0
        # Populated by _drop() with (req_id, out_ids, Stats) the instant a
        # slot finishes and is removed from self.slots -- a caller
        # (BatchScheduler) that only checks "is this req_id still live"
        # after step() would otherwise have no way to retrieve that
        # request's final Stats, since the slot object itself is gone by
        # the time it notices. Keyed by req_id (stable); the caller pops
        # entries it has consumed.
        self.finished: dict = {}
        self._heartbeat = _Heartbeat(on_heartbeat)
        # Throttled decode-progress reporting (see _report_progress) --
        # without it a long generation prints nothing between admission
        # and completion, which is indistinguishable from a hang.
        self._last_progress = 0.0
        self._tokens_since_progress = 0
        # seconds in [draft, prep, verify, post], decode steps timed, verify tokens: since the last report
        self._phase = [0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.progress_interval_s = float(
            os.environ.get("QV_BAND_PROGRESS_INTERVAL_S", "10.0"))
        self._device = None
        # SGLang/vLLM-shaped ragged attention (see _qv_varlen_attention_
        # forward's own docstring for the full "why") -- replaces every
        # earlier padded-K/V + mask attempt this session made (4D causal
        # bias for sdpa, dict mask for flash-attn's own 2D convention,
        # both of which chased correctness bugs stemming from the SAME
        # root cause: padding rows to a shared width in the first place).
        # `_qv_text_config` is whichever config object the loaded model's
        # decoder layers actually read `_attn_implementation` from --
        # found once here, not re-derived per step, and step() flips it
        # to "qv_varlen" only for the duration of its own forward call,
        # restoring the ORIGINAL implementation immediately after, so any
        # OTHER caller sharing this same model object (e.g. serve.py's
        # old STATE["lock"] single-request path, when a request doesn't
        # route through this engine) is completely unaffected.
        _register_qv_varlen_attention()
        # Linear-attention layers cannot be fixed by masking (see
        # _run_linear_attn_per_slot's docstring) -- they must not be fed
        # pad columns at all, so wrap them to run per-slot on any step
        # whose rows have genuinely different lengths.
        _install_linear_attn_per_slot_hooks(model)
        # MoE models only (returns 0 and changes nothing for a dense model):
        # replay each MoE block as a CUDA graph at decode.
        try:
            from qv_band.formats.qvb8.moe_graph import install_moe_graphs, install_lean_norms
            install_moe_graphs(model)
            install_lean_norms(model)
        except Exception as _exc:          # an optimisation only: serve without it, but say so
            print(f"[engine] MoE CUDA-graph install skipped: {type(_exc).__name__}: {_exc}",
                  flush=True)
        self._ring_ok = RING_ENABLED and _install_ring_capture(model)
        self._qv_text_config = _find_attn_impl_holder(model)
        self._qv_orig_attn_impl = getattr(self._qv_text_config, "_attn_implementation", None) \
            if self._qv_text_config is not None else None
    def _advance_ragged(self, seqs: list[list[int]], hist_lens: list[int]) -> None:
        """Push `seqs` through the cache and compute NO logits.
        For prefill chunks and recurrent-state replay -- the forwards
        that exist only to advance the cache. See _advance_cache_only for
        why skipping lm_head here is the correct forward rather than an
        optional speed-up.
        """
        self._forward_ragged(seqs, hist_lens, want_hidden=False,
                             _logits=False)
    def _forward_ragged(self, seqs: list[list[int]], hist_lens: list[int],
                        want_hidden: bool, _logits: bool = True,
                        keep: int | None = None,
                        keep_hidden: int | None = None,
                        capture_layers: list[int] | None = None):
        """Run ONE ragged forward over `seqs` (one entry per live slot,
        possibly empty for a slot with nothing to do this call).
        This is the single place that arms the whole ragged path, so the
        step loop and the post-rollback replay cannot drift apart:
          - `_step_real_lens` on the cache and on every BandedKVLayer --
            read by update() to commit only each row's real tokens, and
            by _qv_varlen_attention_forward for the query side and by
            _run_linear_attn_per_slot for the recurrent side.
          - `_qv_active_past` on each full_attention submodule, the only
            channel through which the attention function can reach the
            live cache (see its docstring).
          - `_attn_implementation` switched to "qv_varlen" for the call
            and restored after, so any other caller sharing this model
            is unaffected.
        `attention_mask=None` throughout: nothing in this path consumes
        a mask (see _build_varlen_inputs).
        """
        if self._packed_ok():
            return self._forward_packed(seqs, hist_lens, want_hidden, _logits, keep, keep_hidden,
                                        capture_layers)
        real_lens = [len(s) for s in seqs]
        input_ids, position_ids = _build_varlen_inputs(
            [s for _, s in self._live()], seqs, hist_lens, self._device)
        # Every attention module (Qwen3.5: .self_attn; Nemotron-H: .mixer of an "attention" block) reaches the live
        # cache through _qv_active_past (it is not passed into attention_interface's kwargs): set by self._armed().
        # Rows whose linear-attention state is still COLD this step.
        # has_previous_state is a single per-LAYER flag, but a mid-flight
        # admission makes the batch heterogeneous: the running rows have
        # real conv/recurrent state while the just-appended row is all
        # zeros and has never been prefilled. With one flag reading True
        # for everyone, that new row's prefill takes the module's
        # use_precomputed_states branch and prepends its own zero
        # conv_state as though it were genuine left context.
        # hist_lens[row] == 0 is exactly "this row has no history yet";
        # _run_linear_attn_per_slot swaps the flag per row from this.
        # Verified reachable: a mid-flight admission really does present
        # cold_rows=[False, True] against has_previous_state={0: True}.
        # The residual error it removes is small enough that it did not
        # change sampled tokens in the cases measured -- it is a
        # correctness fix, not an output fix.
        from qv_band.kv import step_ctx
        with self._armed(), step_ctx.active(real_lens=real_lens, cold_rows=[h == 0 for h in hist_lens]):
            if not _logits:
                return _advance_cache_only(
                    self.model, input_ids, position_ids, self.past)
            return _forward_batched_with_hidden(
                self.model, input_ids, None, position_ids, self.past, want_hidden,
                keep=keep, keep_hidden=keep_hidden, capture_layers=capture_layers)

    def _armed(self):
        """Context: the model is switched to the engine's attention ("qv_varlen") and every attention module can reach
        the live cache (_qv_active_past). Reference-counted, so several forwards in flight at once (overlapped
        multi-card prefill, one thread per chunk) share one arming: the first entry sets it, the last exit restores."""
        from contextlib import contextmanager
        eng = self

        @contextmanager
        def _cm():
            if not hasattr(eng, "_arm_lock"):
                eng._arm_lock, eng._arm_depth = threading.Lock(), 0
            with eng._arm_lock:
                if eng._arm_depth == 0:
                    mods = [getattr(dl, "self_attn", None) or (
                        getattr(dl, "mixer", None) if getattr(dl, "block_type", None) == "attention" else None)
                        for dl in getattr(eng.model.model, "layers", [])]
                    eng._arm_mods = [m for m in mods if m is not None]
                    for m in eng._arm_mods:
                        m._qv_active_past = eng.past
                    if eng._qv_text_config is not None:
                        eng._qv_text_config._attn_implementation = "qv_varlen"
                eng._arm_depth += 1
            try:
                yield
            finally:
                with eng._arm_lock:
                    eng._arm_depth -= 1
                    if eng._arm_depth == 0:
                        if eng._qv_text_config is not None:
                            eng._qv_text_config._attn_implementation = eng._qv_orig_attn_impl
                        for m in eng._arm_mods:
                            m._qv_active_past = None
        return _cm()

    def _step_hooks(self, capture_layers):
        """Persistent forward hooks that write a drafter tap's output (and the final norm's input) into the CALLING
        thread's step context (qv_band/kv/step_ctx.py) -- not per-forward hooks, which would also fire for another
        chunk's forward in flight at the same time. Installed once per layer, idle outside a step that asks."""
        from qv_band.kv.step_ctx import current as _cur
        hooked = self.__dict__.setdefault("_hooked_taps", {})
        for li in capture_layers or ():
            if li not in hooked and li not in self._remote_layers:   # a remote tap comes back with the stage's reply
                def _tap(_m, _a, o, li=li):
                    c = _cur()
                    if c is not None and c.capture is not None and li in c.capture_layers:
                        # pipeline placement: a tap layer may sit on another card -- copied through host memory, never
                        # peer-to-peer (common/xfer.py): a direct card-to-card copy gave the drafter wrong taps on the two
                        # V100s (0% accepted, 2026-11-11)
                        from qv_band.common.xfer import to_device as _to_dev
                        c.capture[li] = _to_dev(o[0] if isinstance(o, tuple) else o, c.capture_dev)
                hooked[li] = self.model.model.layers[li].register_forward_hook(_tap)
        if "_norm" not in hooked:
            def _pre(_m, a, _k):
                c = _cur()
                if c is not None and c.want_pre_norm:
                    c.pre_norm = a[0]
            hooked["_norm"] = _resolve_final_norm(self.model).register_forward_pre_hook(_pre, with_kwargs=True)

    def _packed_ok(self) -> bool:
        """Packed steps need every cache layer to be a BandedKVLayer (pure attention, e.g. Ripple): a
        linear-attention / Mamba layer keeps per-ROW state, which a single packed row cannot carry."""
        p = getattr(self, "_packed", None)
        if p is None or p[0] is not self.past:
            from qv_band.kv.banded_cache import BandedKVLayer, _StatelessLayer
            # Attention layers must be banded; a linear-attention layer (Qwen3.5 GatedDeltaNet) is fine too -- its
            # hook re-splits a packed row per slot (_run_linear_attn_packed). Not covered: Mamba (Nemotron-H) and
            # MoE models (their decode CUDA graphs are keyed on the padded batch shape).
            cfg = self.model.config.get_text_config(decoder=True)
            moe = bool(getattr(cfg, "num_experts", 0) or getattr(cfg, "n_routed_experts", 0))
            lin_ok = all(hasattr(getattr(dl, "linear_attn", None), "_qv_per_slot_wrapped")
                         for dl in self.model.model.layers if getattr(dl, "linear_attn", None) is not None)
            ok = (PACKED_ENABLED and not moe and lin_ok and self.past is not None
                  and type(self.past).__name__ == "BandedKVCache" and len(self.past.layers) > 0
                  and all(isinstance(l, (BandedKVLayer, _StatelessLayer)) or hasattr(l, "recurrent_states")
                          for l in self.past.layers))   # _StatelessLayer: a layer held by a remote stage
            self._packed = p = (self.past, ok)
            print(f"[engine] packed steps (no padding, prefill interleaved with decode): "
                  f"{'ON' if ok else 'off (MoE / Mamba / unwrapped linear attention)'}"
                  f"{' -- attention: flash-attn + fused qvb8' if ok and _flash_ok(self._device) else ''}"
                  f"{' -- attention: fused qvb8 for every length + SDPA (no flash-attn)' if ok and not _flash_ok(self._device) else ''}",
                  file=sys.__stdout__, flush=True)
        return p[1]

    def _forward_packed(self, seqs, hist_lens, want_hidden, _logits, keep, keep_hidden, capture_layers):
        """_forward_ragged for pure-attention models with NO padding: every live slot's tokens are laid end to
        end in ONE row (SGLang/vLLM style). A 16-token verify row next to a 1024-token prefill chunk costs 1040
        tokens of compute, not 2 x 1024. Results are handed back in the padded (rows, width) layout the callers
        index (real tokens right-aligned), gathered only for the positions they read."""
        from qv_band.kv.banded_cache import BandedKVLayer
        live = self._live()
        segs, ids, pos = [], [], []
        for (bi, _), sq, h in zip(live, seqs, hist_lens):
            if sq:
                segs.append((bi, len(ids), len(sq)))
                ids.extend(sq)
                pos.extend(range(h, h + len(sq)))
        if not segs:
            return _ResultWithHidden(None, self.past, None)
        dev = self._device
        both = torch.tensor([ids, pos], dtype=torch.long).to(dev, non_blocking=True)
        input_ids, position_ids = both[0:1], both[1:2]
        from qv_band.kv import step_ctx
        if _logits:
            self._step_hooks(capture_layers)
        # the step layout lives in this thread's step context (several chunks can be in flight, one per thread);
        # linear-attention rows with no state yet start cold
        with self._armed(), step_ctx.active(segs=segs, cold_rows=[h == 0 for h in hist_lens],
                                            capture={} if (_logits and capture_layers) else None,
                                            capture_layers=capture_layers, capture_dev=dev) as ctx:
            if not _logits:
                self.model.model(input_ids, attention_mask=None, position_ids=position_ids,
                                 past_key_values=self.past, use_cache=True)
                return _ResultWithHidden(None, self.past, None)
            ctx.want_pre_norm = want_hidden
            out = self.model.model(input_ids, attention_mask=None, position_ids=position_ids,
                                   past_key_values=self.past, use_cache=True)
        pre_norm, layer_out = [ctx.pre_norm], ctx.capture
        # (rows, width) gather index, real tokens right-aligned; filler points at the row's first token
        width = max(len(s) for s in seqs)
        idx = torch.zeros(len(seqs), width, dtype=torch.long)
        r_of = {bi: r for r, (bi, _) in enumerate(live)}
        for bi, off, n in segs:
            idx[r_of[bi], :width - n] = off
            idx[r_of[bi], width - n:] = torch.arange(off, off + n)
        idx = idx.to(dev)
        nk = width if keep is None else min(keep, width)
        logits = self.model.lm_head(out.last_hidden_state[0][idx[:, -nk:]])
        hidden = None
        if want_hidden:
            hsrc = pre_norm[0] if pre_norm[0] is not None else out.last_hidden_state
            nh = width if keep is None else min(max(keep, keep_hidden or 0), width)
            hidden = hsrc[0][idx[:, -nh:]]
        feats = None
        if capture_layers:
            feats = torch.cat([layer_out[li] for li in capture_layers], dim=-1)[0][idx]
        return _ResultWithHidden(logits, self.past, hidden, feats)

    def _prefill_rounds_overlapped(self, gate, live, rows, work, heads, cuts, ch, done, totals, t0) -> bool:
        """The rounds of _prefill_admitted's loop, overlapped across the cards of a pipeline placement
        (decode/stage_gate.py): each round runs the model's own forward in its own thread holding a ticket; every
        card takes rounds strictly in order, so the KV is exactly what the sequential loop builds. At most one round
        per card is in flight. A prefix checkpoint after round c is assembled from per-card snapshots taken when
        round c leaves each card (the only moment that card's layers hold exactly the state after round c).
        Drafter features are observed and checkpoints saved here, in round order. Updates `done` per finished round.
        Returns True when a client cancelled (the caller parks the rest of the prompt)."""
        import functools
        from concurrent.futures import ThreadPoolExecutor
        from qv_band.decode import stage_gate as _sg
        slot = dict(live)
        hist = {bi: self._slot_len(bi) for bi, _ in live}
        ckpt_at = {bi: getattr(slot[bi], "_ckpt_at", 0) for bi in work}
        d = dict(done)
        plan = []                                    # (seqs, hist per row, saves after it, done after it, hist after it)
        while any(d[bi] < heads[bi] for bi in work):
            seqs = [[] for _ in live]
            for bi in work:
                if d[bi] < heads[bi]:
                    take = min(ch, heads[bi] - d[bi])
                    if d[bi] < cuts.get(bi, 0):
                        take = min(take, cuts[bi] - d[bi])
                    seqs[rows[bi]] = work[bi][0, d[bi]:d[bi] + take].tolist()
            hl = [hist[bi] for bi, _ in live]
            saves = []
            for bi in work:
                n = len(seqs[rows[bi]])
                d[bi] += n
                hist[bi] += n
                if cuts.get(bi) == d[bi]:
                    saves.append(bi)
                elif (d[bi] < heads[bi] and slot[bi].prompt_ids is not None
                      and hist[bi] - ckpt_at[bi] >= self.prefill_chunk):
                    ckpt_at[bi] = hist[bi]
                    saves.append(bi)
            plan.append((seqs, hl, saves, dict(d), dict(hist)))
        L = len(self.past.layers)
        base = gate.new_run(len(plan))
        snaps = {}

        def snap(c, s, bis):                         # inside gate.leave(s, base + c)
            lo, hi = gate.stage_layers[s]
            for bi in bis:
                arr = snaps[c][bi]
                for li in range(lo, hi):
                    arr[li] = _export_layer_slot(self.past.layers[li], bi)

        for c, (_, _, saves, _, _) in enumerate(plan):
            if saves:
                snaps[c] = {bi: [None] * L for bi in saves}
                for s in range(gate.n):
                    gate.on_leave[(s, base + c)] = functools.partial(snap, c, s, saves)
        failed = []

        @torch.no_grad()                             # grad mode is per thread
        def run(c):
            seqs, hl, _, _, _ = plan[c]
            with torch.cuda.device(self._device), _sg.as_ticket(base + c):
                if failed:
                    gate.release(base + c)
                    return None
                try:
                    if self.capture:
                        return self._forward_ragged(seqs, hl, True, keep=1, capture_layers=self.capture)
                    self._advance_ragged(seqs, hl)
                    return None
                except BaseException as e:
                    failed.append(e)
                    gate.release(base + c)
                    raise

        pool = ThreadPoolExecutor(max_workers=gate.n, thread_name_prefix="prefill-card")
        futs, nxt, completed, cancelled, last_print = {}, 0, 0, False, t0
        try:
            while completed < len(plan):
                while not cancelled and not failed and nxt < len(plan) and nxt - completed < gate.n:
                    futs[nxt] = pool.submit(run, nxt)
                    nxt += 1
                if completed >= nxt:
                    break
                c = completed
                res = futs.pop(c).result()           # re-raises the round's error
                seqs, hl, saves, dc, hc = plan[c]
                if res is not None:
                    for bi in work:
                        r, n = rows[bi], len(seqs[rows[bi]])
                        dr = slot[bi].drafter
                        if n and dr is not None:
                            dr.observe_features(hl[r], res.layer_features[r, -n:])
                    del res
                for bi in saves:
                    state = snaps[c][bi]
                    if all(e is not None and e is not _UNSUPPORTED for e in state):
                        slot[bi]._ckpt_at = hc[bi]
                        self._save_prefix(bi, slot[bi], branch=True, state=state, n=hc[bi])
                snaps.pop(c, None)
                done.update(dc)
                completed += 1
                now = time.time()
                if now - last_print >= 2.0:
                    tot = sum(done.values())
                    parts = " ".join(f"#{slot[bi].req_id}:{hc[bi]}/{hc[bi] + totals[bi] - done[bi]}" for bi in work)
                    print(f"[prefill] {parts} overlapped over {gate.n} cards -- {tot} tok in {now - t0:.1f}s "
                          f"({tot / max(now - t0, 1e-9):.0f} tok/s)", file=sys.__stdout__, flush=True)
                    last_print = now
                self._heartbeat.tick()
                if self.cancel_check is not None and (
                        self.cancel_check() & {slot[b].req_id for b in work}):
                    cancelled = True
        finally:
            pool.shutdown(wait=True)
            for c in range(len(plan)):
                for s in range(gate.n):
                    gate.on_leave.pop((s, base + c), None)
        return cancelled

    def _prefill_admitted(self) -> bool:
        """Consume all but the last PREFILL_CHUNK tokens of every
        newly-admitted slot's prompt, in chunks, before the shared step.
        WHY: a just-admitted slot contributes its WHOLE prompt to the
        next step's forward. For a real agent prompt (measured: 24093
        tokens from one Cline request) that is a single enormous
        forward -- peak activation memory scales with it, which is
        exactly what speculative_generate's own PREFILL_CHUNK loop
        exists to bound, and it runs for a long time with no output so
        the server looks hung. Feeding it in chunks bounds the peak to
        one chunk's activations and gives the operator visible progress.
        The FINAL chunk is deliberately left in `_pending_prompts` so
        the slot still joins the shared step normally -- that is where
        its first token is sampled from, alongside whatever decode work
        the already-running slots are doing. Only the part that carries
        no output is split off here.
        Chunks are fed one slot at a time: each is a different length,
        and batching them would reintroduce exactly the padding this
        engine exists to avoid. Admissions are rare relative to decode
        steps, so this costs little.
        """
        # Chunk ONE admitted slot at a time, and only while it is the
        # only slot that has not been prefilled yet.
        #
        # REAL CONFIRMED BUG this guards: a slot that has never taken
        # part in a forward has no row in the linear-attention state
        # (those tensors are created lazily, sized from the first
        # forward's batch). Letting it ride along as a zero-length row
        # in ANOTHER slot's chunk forward is what creates that state at
        # the wrong shape -- reproduced as "Sizes of tensors must match
        # except in dimension 2. Expected size 0 but got size 1" inside
        # Qwen3_5GatedDeltaNet's torch.cat([conv_state, mixed_qkv]) as
        # soon as two requests were admitted back-to-back before any
        # step. Chunking strictly one slot at a time keeps every forward
        # well-formed for every row that takes part in it.
        live = self._live()
        rows = {bi: r for r, (bi, _) in enumerate(live)}
        # Packed (pure-attention) engines feed ONE chunk round per step() and then let the running requests
        # take their decode step, so a long prompt no longer freezes everyone else for its whole prefill.
        # While someone is decoding the chunk is smaller (each of their steps waits for one chunk).
        self._prefilling = set()
        packed = self._packed_ok()
        decoding = any(not s.just_admitted for _, s in live)
        ch = min(INTERLEAVE_CHUNK, self.prefill_chunk) if (packed and decoding) else self.prefill_chunk

        def checkpoint(s) -> bool:
            # a prompt we can save a prefix of: long enough, and its not-yet-prefilled part
            # is the whole of what follows the restored prefix
            p = self._pending_prompts[s.bi]
            return (s.prompt_ids is not None and p.shape[-1] > PREFIX_TAIL
                    and s.prefix_hit + p.shape[-1] == len(s.prompt_ids))
        work = {bi: self._pending_prompts[bi] for bi, s in live
                if s.just_admitted and bi in self._pending_prompts
                and (self._pending_prompts[bi].shape[-1] > ch or checkpoint(s)
                     or s.branch > s.prefix_hit)}
        if not work:
            return False
        # Every admitted slot is chunked TOGETHER, in lockstep, rather
        # than one slot at a time or (worse) not at all when several
        # arrive at once. An earlier version bailed out entirely when
        # more than one slot was waiting, which meant two simultaneous
        # requests each fed their whole prompt as ONE forward -- the
        # exact peak-memory case chunking exists to prevent, hit in
        # precisely the situation that makes it worst.
        totals = {bi: t.shape[-1] for bi, t in work.items()}
        ckpt = {bi for bi, s in live if bi in work and checkpoint(s)}
        heads = {bi: totals[bi] - (PREFIX_TAIL if bi in ckpt else ch) for bi in work}
        # where each slot parts from a saved prompt, in this call's own coordinates
        cuts = {bi: s.branch - s.prefix_hit for bi, s in live
                if bi in work and s.branch > s.prefix_hit}
        for bi, c in cuts.items():
            heads[bi] = max(heads[bi], c)
        t0 = time.time()
        last_print = t0
        done = {bi: 0 for bi in work}
        gate = getattr(self.model, "_qv_stage_gate", None)
        if (gate is not None and gate.n > 1 and packed and not decoding
                and os.environ.get("QV_PIPE_PREFILL", "1") != "0"
                and len(self.past.layers) == gate.stage_layers[-1][1]):
            # Pipeline placement over several cards and nobody is decoding: run ALL the rounds now, overlapped
            # across the cards (round c + 1 on card i while round c is on card i + 1), not one round per step.
            if self._prefill_rounds_overlapped(gate, live, rows, work, heads, cuts, ch, done, totals, t0):
                for b in work:                       # a client is gone: park the rest, as the sequential loop does
                    sl = dict(live)[b]
                    self._pending_prompts[b] = work[b][:, done[b]:]
                    sl.prefix_hit += done[b]
                    sl._history_len = self._slot_len(b)
                return True
        while any(done[bi] < heads[bi] for bi in work):
            seqs = [[] for _ in live]
            for bi in work:
                if done[bi] < heads[bi]:
                    take = min(ch, heads[bi] - done[bi])
                    if done[bi] < cuts.get(bi, 0):
                        take = min(take, cuts[bi] - done[bi])
                    seqs[rows[bi]] = work[bi][0, done[bi]:done[bi] + take].tolist()
            hist = [self._slot_len(bi) for bi, _ in live]
            for (bi, sl), h in zip(live, hist):
                sl._history_len = h
            _pp = None
            if _PROFILE_PREFILL[0] > 0:
                _PROFILE_PREFILL[0] -= 1
                _pp = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                         torch.profiler.ProfilerActivity.CUDA])
                _pp.__enter__()
                _pp_t = time.perf_counter()
            try:
                if self.capture:
                    # A DFlash drafter needs these chunks' layer features too;
                    # logits are cut to one position, so this costs the hooks.
                    res = self._forward_ragged(seqs, hist, True, keep=1,
                                               capture_layers=self.capture)
                    for bi in work:
                        r, n = rows[bi], len(seqs[rows[bi]])
                        dr = dict(live)[bi].drafter
                        if n and dr is not None:
                            dr.observe_features(hist[r], res.layer_features[r, -n:])
                    del res
                else:
                    self._advance_ragged(seqs, hist)
            except BaseException:
                if _pp is not None:                  # never leave the profiler running after a failed chunk
                    _pp.__exit__(None, None, None)
                raise
            if _pp is not None:
                torch.cuda.synchronize()
                _pp.__exit__(None, None, None)
                print(f"[profile-prefill] chunk {sum(len(x) for x in seqs)} tok at hist {max(hist)}: "
                      f"{(time.perf_counter() - _pp_t) * 1e3:.0f} ms\n"
                      + _pp.key_averages().table(sort_by="cuda_time_total", row_limit=25),
                      file=sys.__stdout__, flush=True)
            for bi in work:
                done[bi] += len(seqs[rows[bi]])
                if cuts.get(bi) == done[bi]:
                    self._save_prefix(bi, dict(live)[bi], branch=True)
                elif done[bi] < heads[bi] and dict(live)[bi].prompt_ids is not None:
                    # A chunk boundary of a long prefill: if a later prompt rewrites something inside
                    # this prompt (an agent client replacing an old tool result), the nearest boundary
                    # before the rewrite is where it can restore, instead of prefilling from token 0.
                    # (At most one per --prefill-chunk tokens: interleaved chunks are smaller.)
                    sl = dict(live)[bi]
                    n_now = self._slot_len(bi)
                    if n_now - getattr(sl, "_ckpt_at", 0) >= self.prefill_chunk:
                        sl._ckpt_at = n_now
                        self._save_prefix(bi, sl, branch=True)
            now = time.time()
            if now - last_print >= 2.0:
                tot_done = sum(done.values())
                # Whole-prompt progress: a packed engine prefills one chunk per step(), so this call's own
                # done/totals only cover what was left when it started (the old line showed 4096/53698,
                # 4096/49602, ... -- shrinking). Cached tokens / (cached + still to prefill) always grows.
                cached = {bi: self._slot_len(bi) for bi in work}
                whole = {bi: cached[bi] + totals[bi] - done[bi] for bi in work}
                parts = " ".join(
                    f"#{dict(live)[bi].req_id}:{cached[bi]}/{whole[bi]}"
                    for bi in work)
                print(f"[prefill] {parts} "
                      f"(thread={threading.current_thread().name}) "
                      f"{100 * sum(cached.values()) / max(sum(whole.values()), 1):.0f}% -- "
                      f"{tot_done} tok in {now - t0:.1f}s ({tot_done / max(now - t0, 1e-9):.0f} tok/s)",
                      file=sys.__stdout__, flush=True)
                last_print = now
            self._heartbeat.tick()
            if self.cancel_check is not None and (
                    self.cancel_check() & {dict(live)[b].req_id for b in work}):
                # A client is gone: stop between chunks instead of finishing a prefill nobody will read.
                # Every slot keeps what it has consumed; the rest of its prompt is still pending and the
                # next step resumes it (the server drops the cancelled slots first).
                for b in work:
                    sl = dict(live)[b]
                    self._pending_prompts[b] = work[b][:, done[b]:]
                    sl.prefix_hit += done[b]
                    sl._history_len = self._slot_len(b)
                return True
            if packed and any(done[bi] < heads[bi] for bi in work):
                # One round per step: park the rest (exactly as the cancel path above) and let the step
                # run the decoding slots; the next step() continues this prefill where it stopped.
                for b in work:
                    sl = dict(live)[b]
                    self._pending_prompts[b] = work[b][:, done[b]:]
                    sl.prefix_hit += done[b]
                    sl._history_len = self._slot_len(b)
                    if done[b] < heads[b]:
                        self._prefilling.add(b)
                    elif b in ckpt:
                        self._save_prefix(b, sl)
                return False
        dt = time.time() - t0
        tot_done = sum(done.values())
        for bi in ckpt:
            self._save_prefix(bi, dict(live)[bi])
        for bi in work:
            # The remaining tail joins the ordinary shared step, which is
            # where this slot's first token gets sampled.
            self._pending_prompts[bi] = work[bi][:, heads[bi]:]
            dict(live)[bi]._history_len = self._slot_len(bi)
        parts = " ".join(f"#{dict(live)[bi].req_id}:{done[bi]}/{totals[bi]}"
                         for bi in work)
        print(f"[prefill] {parts} done "
              f"(thread={threading.current_thread().name}) "
              f"in {dt:.1f}s ({tot_done / max(dt, 1e-9):.0f} tok/s) -- "
              f"tails join the next batched step",
              file=sys.__stdout__, flush=True)
        return False
    def _report_progress(self) -> None:
        """One throttled line describing what the batch is doing right
        now -- which requests are live, how far each has generated, and
        the aggregate decode rate.
        Without this the server prints nothing between admitting a
        request and finishing it, so a long generation is
        indistinguishable from a hang from the operator's side. Throttled
        on wall-clock (QV_BAND_PROGRESS_INTERVAL_S, default 10s) so it
        stays readable no matter how fast decode runs.
        """
        now = time.time()
        if now - self._last_progress < self.progress_interval_s:
            return
        live = self._live()
        if not live:
            return
        if self._last_progress == 0.0:
            # First call: start the clock. Discard the tokens counted so
            # far too, otherwise the first reported rate divides tokens
            # produced before timing began by the interval since it did,
            # printing a number that is simply wrong.
            self._last_progress = now
            self._tokens_since_progress = 0
            return
        dt = now - self._last_progress
        made = self._tokens_since_progress
        ph, self._phase = self._phase, [0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mem = ""
        try:
            from qv_band.kv import banded_cache as _bc
            gib = 2.0 ** 30
            store = _bytes_of([e["state"] for e in self._prefix_store], set()) / gib
            pl = getattr(self.model, "_qv_placement", None)
            cards = sorted({torch.device(d).index for d in pl.devices}) if pl is not None else None
            if cards and len(cards) > 1:              # pipeline placement: every card, not just the current one
                vram = ", vram " + " ".join(
                    f"cuda:{c} {torch.cuda.memory_allocated(c) / gib:.1f}/{torch.cuda.memory_reserved(c) / gib:.1f}G"
                    for c in cards) + " alloc/reserved "
            else:
                vram = (f", vram {torch.cuda.memory_allocated() / gib:.1f}G alloc "
                        f"{torch.cuda.memory_reserved() / gib:.1f}G reserved ")
            mem = (vram +
                   f"(dense kv views {_bc._GLOBAL_SM_BYTES / gib:.2f}G joined {_bc._GLOBAL_JOINED_BYTES / gib:.2f}G "
                   f"prefix store {store:.2f}G x{len(self._prefix_store)})")
        except Exception:
            pass                                # reporting must never break generation
        timing = (f", per step ms: draft {1e3 * ph[0] / ph[4]:.0f} prep {1e3 * ph[1] / ph[4]:.0f} "
                  f"verify {1e3 * ph[2] / ph[4]:.0f} post {1e3 * ph[3] / ph[4]:.0f} "
                  f"(verify {1e3 * ph[8] / ph[4]:.0f} crop {1e3 * ph[9] / ph[4]:.0f} rollback {1e3 * ph[6] / ph[4]:.0f} "
                  f"observe {1e3 * ph[7] / ph[4]:.0f} emit {1e3 * ph[10] / ph[4]:.0f} replay {1e3 * ph[11] / ph[4]:.0f}), "
                  f"{ph[5] / ph[4]:.1f} tok verified, {made / ph[4]:.2f} kept") if ph[4] else ""
        self._last_progress = now
        self._tokens_since_progress = 0
        parts = [f"#{s.req_id}:{len(s.out_ids)}/{s.max_new}" for _, s in live]
        # KV-cache accounting is only computed HERE, inside the throttled
        # branch -- memory_report() walks every layer and every banded
        # segment, so calling it per step would be real overhead on a
        # long context. Once every progress_interval_s it is free.
        kv = ""
        try:
            m = self.past.memory_report()
            if m.get("total_bytes"):
                kv = (f", kv {m['total_bytes'] / 2**30:.2f}GiB"
                      f" ({m.get('ratio_vs_dense', 1.0) * 100:.0f}% of dense)")
        except Exception:
            # Reporting must never be able to break generation.
            pass
        print(f"[decode] {len(live)} live "
              f"(thread={threading.current_thread().name}) "
              f"{' '.join(parts)} -- {made / max(dt, 1e-9):.1f} tok/s "
              f"aggregate, ctx {max(self._slot_len(b) for b, _ in live)} tok{kv}{mem}{timing}",
              file=sys.__stdout__, flush=True)
    def _live(self) -> list[tuple[int, "_BatchSlot"]]:
        """(slot_index, slot) for every occupied slot, in slot order --
        the row order every ragged tensor in this class uses."""
        return [(i, s) for i, s in enumerate(self.slots) if s is not None]
    def _slot_len(self, bi: int) -> int:
        """Slot `bi`'s committed token count, read from the cache itself
        (the banded store is the single source of truth for it)."""
        cached = getattr(self, "_len_layer", None)
        if cached is None or cached[0] is not self.past:
            from qv_band.kv.banded_cache import BandedKVLayer
            cached = self._len_layer = (self.past, next(
                (l for l in self.past.layers if isinstance(l, BandedKVLayer)), None))
        layer = cached[1]
        return layer.get_seq_length(slot=bi) if layer is not None else 0
    def has_work(self) -> bool:
        return any(s is not None for s in self.slots)
    def add_request(self, ids: torch.Tensor, *, max_new: int,
                    temperature: float = 0.0, top_k: int = 0,
                    top_p: float = 1.0, req_id=None, on_token=None) -> int:
        """Admit a new request into the LIVE batch -- safe to call between
        any two step() calls, including while other slots are
        mid-generation. Returns the slot index. The new slot's own
        prefill runs as part of the NEXT step() call (see class
        docstring), not here -- admitting the KV-cache row must happen
        before that forward, but the forward itself is shared with
        whatever other slots are doing that same step, which is the
        whole point of continuous batching over a round model.
        """
        from qv_band.kv.banded_cache import BandedKVCache, KV_DEFAULT_TAU
        device = ids.device
        self._device = device
        drafter = self.drafter_factory() if self.drafter_factory else None
        if (drafter is not None and MTP_CTX_WINDOW > 0
                and hasattr(drafter, "enable_context")):
            drafter.enable_context(MTP_CTX_WINDOW)
        if self.past is None:
            self.past = _make_banded_cache(self.model.config,
                                           tau=self.kv_tau or KV_DEFAULT_TAU,
                                           residual_length=self.int8_residual_length)
            if self._stages:
                from qv_band.decode.remote_stage import _localize
                _localize(self.past, set(range(len(self.past.layers))) - self._remote_layers)
        # Find a free slot index: reuse a hole left by a dropped request
        # before growing the list, so bi stays dense (drop_slot's own
        # "every later index shifts down" contract, applied consistently).
        bi = None
        for i, s in enumerate(self.slots):
            if s is None:
                bi = i
                break
        key_like = ids.new_zeros((1, 1, 0, 1), dtype=self.model.dtype).to(device)
        # add_slot only needs heads/head_dim from key_like's SHAPE, not
        # these placeholder dims -- the real per-layer template comes
        # from BandedKVLayer.add_slot's own key_like.shape[1]/[3] read,
        # so build a shape-correct placeholder from the model's actual
        # attention config instead of a bogus (1,1,0,1).
        cfg = self.model.config.get_text_config(decoder=True)
        heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        key_like = torch.zeros(1, heads, 0, head_dim, dtype=self.model.dtype, device=device)
        val_like = key_like
        new_bi = self.past.add_slot(key_like, val_like)
        # Grow the linear-attention layers' batch axis to match -- but
        # ONLY once they actually hold state. Their tensors are created
        # lazily by the first forward, so before that there is no row to
        # clone and nothing to keep in sync (the first forward sizes
        # them from its own batch). Gating this on the BANDED layers'
        # is_initialized instead was a real bug: those are initialised
        # by add_slot() itself, so admitting a second request before any
        # step() had run took this branch while the linear state was
        # still absent, the grow silently did nothing, and the two layer
        # families ended up with different batch sizes -- surfacing as
        # "Sizes of tensors must match except in dimension 2. Expected
        # size 0 but got size 1" inside Qwen3_5GatedDeltaNet's own
        # torch.cat([conv_state, mixed_qkv]) on the next forward.
        if _linear_attention_state_ready(self.past):
            _grow_linear_attention_batch(self.past)
        for _st in self._stages:
            _st.queue("add_slot")
        if bi is None:
            bi = new_bi
            slot = _BatchSlot(bi, drafter, max_new, temperature, top_k, top_p,
                              req_id=req_id, on_token=on_token)
            self.slots.append(slot)
        else:
            # A reused hole's index must match what add_slot() just
            # assigned -- true whenever every slot is dense (no gaps
            # left mid-list), which drop_slot()'s own "shift every later
            # index down" contract guarantees as long as callers only
            # ever drop/add through this class (never index the raw
            # cache directly).
            slot = _BatchSlot(bi, drafter, max_new, temperature, top_k, top_p,
                              req_id=req_id, on_token=on_token)
            self.slots[bi] = slot
        slot.just_admitted = True
        self._admit_seq += 1
        slot.admit_no = self._admit_seq
        self._pending_prompts[bi] = ids
        self._restore_prefix(bi, slot, ids)
        return bi

    def _restore_prefix(self, bi: int, slot: "_BatchSlot", ids: torch.Tensor) -> None:
        """If a saved prefix is exactly the start of this prompt, load it into the slot and
        queue only the rest. Any doubt (unsupported drafter, layers that do not line up) just
        means the prompt is prefilled in full, as before."""
        n_ids = ids.shape[-1]
        if n_ids < PREFIX_MIN:
            return
        slot.prompt_ids = prompt = ids[0].tolist()
        best = None
        branch, closest = 0, (0, None)
        for e in self._prefix_store:
            n = len(e["tokens"])
            if n < n_ids and (best is None or n > len(best["tokens"])) and prompt[:n] == e["tokens"]:
                best = e
            lcp = len(os.path.commonprefix([prompt, e["tokens"]]))
            if lcp > closest[0]:
                closest = (lcp, e)
            if lcp < n and lcp > branch:
                branch = lcp                 # prompt and e agree up to here, then differ
        # A prompt that leaves a saved one early (a new task with the same system prompt, a
        # rewritten history) can only restore if a state exists exactly where they part. Save
        # one there while prefilling, so the next prompt that shares that start restores it.
        hit_n = len(best["tokens"]) if best is not None else 0
        if branch >= PREFIX_MIN and branch > hit_n and branch <= n_ids - PREFIX_TAIL:
            slot.branch = branch
        if best is None:
            if closest[1] is not None and closest[0] >= PREFIX_MIN:
                l = closest[0]
                print(f"[prefix] req #{slot.req_id}: no saved prefix matches; closest shares {l} of "
                      f"{n_ids} tokens, then new {self.tok.decode(prompt[l:l + 12])!r} vs saved "
                      f"{self.tok.decode(closest[1]['tokens'][l:l + 12])!r}"
                      + (f"; will save a checkpoint at {slot.branch}" if slot.branch else ""),
                      file=sys.__stdout__, flush=True)
            return
        dr = slot.drafter
        if dr is not None and not hasattr(dr, "import_state") and type(dr).__name__ != "NGramDrafter":
            return
        if not self._import_state(bi, best["state"]):
            return
        if dr is not None and best["drafter"] is not None:
            dr.import_state(best["drafter"])
        n = len(best["tokens"])
        self._pending_prompts[bi] = ids[:, n:]
        slot.prefix_hit = slot._history_len = slot._ckpt_at = n   # next chunk checkpoint counts from here
        self._prefix_store.remove(best)
        self._prefix_store.insert(0, best)
        print(f"[prefix] req #{slot.req_id}: restored {n} of {n_ids} prompt tokens, "
              f"prefilling {n_ids - n}", file=sys.__stdout__, flush=True)

    def _save_prefix(self, bi: int, slot: "_BatchSlot", branch: bool = False, state=None, n=None) -> None:
        """Keep slot `bi`'s cache as it stands (the prompt up to PREFIX_TAIL tokens
        short of its end) for a later prompt that starts the same way. `state` / `n`: a snapshot taken earlier
        (overlapped prefill: each card snapshots its layers when the round leaves it) and its length."""
        if n is None:
            n = self._slot_len(bi)
        dr = slot.drafter
        if slot.prompt_ids is None or n < PREFIX_MIN or n > len(slot.prompt_ids) - PREFIX_TAIL:
            return
        if dr is not None and not hasattr(dr, "export_state") and type(dr).__name__ != "NGramDrafter":
            return
        if state is None:
            state = self._export_state(bi)
        if state is None:
            return
        tokens = slot.prompt_ids[:n]
        # the same conversation moving forward supersedes its earlier prefix
        # (a branch point is not superseded: other prompts that part from this one there need it)
        self._prefix_store = [e for e in self._prefix_store
                              if e.get("branch") or not (len(e["tokens"]) <= n and tokens[:len(e["tokens"])] == e["tokens"])]
        if any(e["tokens"] == tokens for e in self._prefix_store):
            return
        self._prefix_store.insert(0, {"tokens": tokens, "state": state, "branch": branch,
                                      "drafter": dr.export_state() if hasattr(dr, "export_state") else None})
        self._trim_prefix_store()

    def _export_state(self, bi: int):
        """Slot `bi` of every cache layer, here and on every stage; None if some layer cannot be exported."""
        state = _export_slot_state(self.past, bi)
        if state is None or not self._stages:
            return state
        from qv_band.decode.remote_stage import StagedState
        state = StagedState(state)
        for st in self._stages:
            sid = st.export(bi)
            if sid is None:
                return None                    # the ids taken so far are freed with `state`
            state.remote += ((st, sid),)
        return state

    def _import_state(self, bi: int, state) -> bool:
        """Write a saved slot into the empty slot `bi`, here and on every stage. All-or-nothing."""
        remote = getattr(state, "remote", ())
        if len(remote) != len(self._stages):
            return False
        for n, (st, sid) in enumerate(remote):
            if not st.import_(bi, sid):
                if n:
                    raise RuntimeError("slot import: accepted by one stage and refused by another")
                return False
        if not _import_slot_state(self.past, bi, state):
            if remote:
                raise RuntimeError("slot import: accepted by the stages and refused by the main machine")
            return False
        return True

    def _trim_prefix_store(self) -> None:
        """Keep PREFIX_KEEP entries. Oldest first, but a branch point or chunk-boundary checkpoint goes
        before any end-of-prompt snapshot: a long prompt saves 6-8 of the former, and evicting plain
        oldest-first would let one long conversation push out the other's snapshot -- the entry that
        gives its next turn a full hit."""
        store = self._prefix_store
        while len(store) > PREFIX_KEEP:
            victim = next((i for i in range(len(store) - 1, -1, -1) if store[i].get("branch")), len(store) - 1)
            del store[victim]
    def _drop(self, bi: int) -> None:
        """Remove a finished slot from every layer of `past` AND shift
        every later slot's bookkeeping down by one, matching drop_slot's
        index-shift contract exactly."""
        finished_slot = self.slots[bi]
        if finished_slot is not None:
            key = finished_slot.req_id if finished_slot.req_id is not None else bi
            self.finished[key] = (finished_slot.req_id, finished_slot.out_ids,
                                  finished_slot.stats)
        self.past.drop_slot(bi)
        # Mirror of add_request's grow: only shrink the linear-attention
        # batch axis once it exists (see _linear_attention_state_ready).
        if _linear_attention_state_ready(self.past):
            _drop_linear_attention_batch(self.past, bi)
        for _st in self._stages:
            _st.queue("drop", bi)
        del self.slots[bi]
        for s in self.slots[bi:]:
            if s is not None:
                s.bi -= 1
        self._pending_prompts = {
            (k - 1 if k > bi else k): v for k, v in self._pending_prompts.items() if k != bi
        }
    @torch.no_grad()
    def step(self) -> list[tuple[int, list[int]]]:
        """Run ONE decode iteration over every live slot.
        Slots admitted since the last call are prefilled as part of this
        same forward -- that is what makes admission cost one step's
        latency instead of one generation's. Returns
        [(req_id, newly_emitted_tokens), ...]; streaming callers get
        tokens through on_token instead and can ignore it.
        Finished slots are dropped from the cache at the end of the
        step, freeing their row for the next add_request().
        SHAPE: this is the SGLang-style ragged path end to end. Nothing
        here builds an attention mask, and no padding is ever consumed:
        attention runs on flat cu_seqlens packing, the linear-attention
        scan runs per-slot at each row's true length, and the KV cache
        commits only each row's real tokens. See _build_varlen_inputs.
        """
        self._ceiling_stop()
        live = self._live()
        if not live:
            return []
        # A long prompt is fed through in PREFILL_CHUNK-sized pieces
        # rather than one giant forward -- see _prefill_admitted.
        if self._prefill_admitted():
            return []                   # interrupted by a cancel; the server drops it and the next step resumes
        live = self._live()
        if not live:
            return []
        # ---- Per-slot work for this step: a just-admitted slot
        # contributes its whole prompt, a running slot contributes
        # [last_token] + its own draft.
        _prof = None
        if _PROFILE_STEPS[0] > 0 and not any(s.just_admitted for _, s in live):
            # QV_BAND_PROFILE_STEPS=N: profile N decode steps (draft + verify forward), print, then off
            _PROFILE_STEPS[0] -= 1
            _prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA])
            _prof.__enter__()
        _t_a = time.perf_counter()
        seqs: list[list[int]] = []
        drafts: list[list[int]] = []
        draft_logits: list[list | None] = []
        admit_rows: set[int] = set()
        # slots still mid-prefill (packed engines: one chunk per step) sit this step out
        wait_rows = {row for row, (bi, _s) in enumerate(live) if bi in getattr(self, "_prefilling", ())}
        k_budget = None
        if ROW_BUDGET > 0 and self.k > 0 and self._packed_ok():
            n_draft = sum(1 for row, (bi, s) in enumerate(live)
                          if row not in wait_rows and not s.just_admitted and s.spec_off <= 0
                          and not s.done and s.drafter is not None)
            if n_draft > 1:
                k_budget = max(0, ROW_BUDGET // n_draft - 1)
        for row, (bi, s) in enumerate(live):
            if row in wait_rows:
                seqs.append([])
                drafts.append([])
                draft_logits.append(None)
                continue
            if s.just_admitted:
                seqs.append(self._pending_prompts.pop(bi)[0].tolist())
                drafts.append([])
                draft_logits.append(None)
                admit_rows.add(row)
                continue
            d, dl = [], None
            if s.spec_off > 0:
                s.spec_off -= 1          # backed off: plain decode this step
            elif not s.done and s.drafter is not None and self.k > 0:
                k_slot = s.k_cur or self.k
                if k_budget is not None:
                    k_slot = min(k_slot, k_budget)
                d, dl = s.drafter.propose(s.out_ids[-1], k_slot,
                                          temperature=s.temperature,
                                          top_k=s.top_k, top_p=s.top_p)
                s.stats.proposed += len(d)
            seqs.append(([s.out_ids[-1]] + d) if s.out_ids else [])
            drafts.append(d)
            draft_logits.append(dl)
        # Every row continues from whatever the cache already holds for it.
        # For a just-admitted slot that is 0 -- UNLESS its prompt was longer
        # than the prefill chunk: _prefill_admitted has then already pushed the
        # head of the prompt through, and this forward is only the TAIL. This
        # line used to hard-code 0 for admitted rows, which gave the tail
        # position ids starting at 0 instead of at the head's length and, via
        # _step_cold_rows (hist == 0 means "cold"), made the 48 linear-attention
        # layers discard the recurrent/conv state the head had built. Every
        # prompt longer than --prefill-chunk was affected: the first-token
        # distribution differed from a one-shot prefill by 0.001-0.07 nats of KL
        # (quantisation noise is ~0.003).
        if not any(seqs):
            self._heartbeat.tick()
            return []                   # only mid-prefill slots this step: their next chunk runs next step()
        _t_b = time.perf_counter()
        hist_lens = [self._slot_len(bi) for bi, s in live]
        for (bi, s), h in zip(live, hist_lens):
            s._history_len = h
        # Snapshot only to undo a REJECTED draft. Padding cannot reach
        # the recurrent state any more (_run_linear_attn_per_slot runs
        # each row at its true length), so there is nothing else a
        # snapshot would be protecting against.
        recur_snap = _snapshot_recurrent_state(self.past) if any(drafts) else None
        if recur_snap is not None:
            for _st in self._stages:
                _st.queue("snapshot")
        # A just-admitted row reads only its LAST position (first sampled
        # token); a decode row reads all of its k+1. Nothing else is needed,
        # so lm_head runs over max(decode widths, 1) trailing positions
        # instead of the whole (possibly 4096-token) prompt tail.
        keep = max([len(sq) for row, sq in enumerate(seqs)
                    if row not in admit_rows] + [1])
        ring = None
        if self._ring_ok and any(drafts):
            self.past._ring = ring = {}
            self.past._ring_rows = {bi for (bi, _s), d in zip(live, drafts) if d}
        # A just-admitted row hands its last MTP_CTX_WINDOW prompt positions'
        # hidden states to a context-enabled draft head.
        ctx_rows = [row for row in admit_rows
                    if getattr(live[row][1].drafter, "ctx_window", 0)]
        keep_hidden = (max(len(seqs[row]) for row in ctx_rows)
                       if ctx_rows else None)
        if keep_hidden is not None:
            keep_hidden = min(keep_hidden, MTP_CTX_WINDOW)
        _t_c = time.perf_counter()
        try:
            res = self._forward_ragged(seqs, hist_lens, self.want_hidden, keep=keep,
                                       keep_hidden=keep_hidden,
                                       capture_layers=self.capture)
            if _prof is not None:
                torch.cuda.synchronize()
        finally:
            if _prof is not None:
                _prof.__exit__(None, None, None)
                ka = _prof.key_averages()
                print(ka.table(sort_by="cuda_time_total", row_limit=25), file=sys.__stdout__, flush=True)
                print(ka.table(sort_by="cpu_time_total", row_limit=20), file=sys.__stdout__, flush=True)
            self.past._ring = None
            self.past._ring_rows = None
        widths = [len(sq) for sq in seqs]
        pad_of = [res.logits.shape[1] - w for w in widths]
        # Greedy rows all need argmax over their verify positions; do it once
        # for the whole batch and cross to the host once, instead of one
        # argmax + .tolist() sync per row.
        # Wait for the forward to finish HERE. A sampled row has no host read before _verify, so
        # without this the GPU's tail of the forward showed up as 'verify' inside the post phase.
        # It costs nothing: _verify would block on the same work a moment later.
        if res.logits.is_cuda:
            torch.cuda.synchronize()
        greedy_all = None
        if any(s.temperature <= 0 and row not in admit_rows and not s.done
               for row, (_bi, s) in enumerate(live)):
            greedy_all = res.logits.argmax(-1).tolist()
        _t_d = time.perf_counter()      # greedy_all's .tolist() above synced the GPU
        emitted: list[tuple[int, list[int]]] = []
        finished: list[int] = []
        # Slots whose recurrent state gets rolled back below must then be
        # re-advanced over the tokens they actually kept.
        replay: dict[int, list[int]] = {}
        for row, ((bi, s), draft, dl, pad, seq) in enumerate(
                zip(live, drafts, draft_logits, pad_of, seqs)):
            if row in wait_rows:
                continue
            if row in admit_rows:
                # This forward WAS this slot's prefill.
                s.just_admitted = False
                nxt = _select(res.logits[row, -1], s.temperature, s.top_k, s.top_p)
                # Seed the drafter with the prompt BEFORE emitting the
                # first token.
                #
                # REAL CONFIRMED BUG (order-of-operations): _emit calls
                # drafter.append(nxt), and NGramDrafter.reset() then does
                # `self.tokens = list(tokens)` -- wiping it. Emitting
                # first therefore appended the first token and
                # immediately threw it away, leaving the drafter's own
                # token history permanently one token short of the real
                # sequence and misaligned from then on. Only k>0 is
                # affected (no drafter, no bug), which is exactly what
                # the measurements showed: the engine's greedy output
                # matched speculative_generate exactly while its
                # drafting output did not -- violating the invariant
                # that speculative decoding at temperature 0 is
                # output-identical to greedy. speculative_generate does
                # reset() first for this reason; this now matches it.
                if self.capture and s.drafter is not None:
                    s.drafter.observe_features(
                        hist_lens[row], res.layer_features[row, -len(seq):])
                if s.drafter is not None and hasattr(s.drafter, "reset"):
                    s.drafter.reset(seq)
                    _seed_ngram_with_prompt(s.drafter, s.prompt_ids)
                if self.want_hidden and s.drafter is not None:
                    s.drafter.hidden = res.last_hidden_state[row, -1]
                    if getattr(s.drafter, "ctx_window", 0):
                        m = len(seq)
                        n = min(m, res.last_hidden_state.shape[1], MTP_CTX_WINDOW)
                        # Row r of the tail is sequence position hist + r; the
                        # token FOLLOWING position q is seq[q+1-hist] (the first
                        # generated token for the last one).
                        s.drafter.observe(hist_lens[row] + m - n,
                                          res.last_hidden_state[row, -n:],
                                          seq[m - n + 1:] + [nxt])
                self._emit(s, bi, [nxt], emitted, finished)
                continue
            if s.done:
                continue
            row_logits = res.logits[row, pad:]   # this row's real positions
            _tv = time.perf_counter()
            n_acc, next_tok = self._verify(
                s, draft, dl, row_logits,
                greedy_all[row][pad:] if greedy_all is not None else None)
            self._phase[8] += time.perf_counter() - _tv
            s.stats.accepted += n_acc
            s.stats.steps += 1
            if self.capture and s.drafter is not None:
                # seq[0] + accepted drafts, at positions hist..; taken from
                # this verify forward before any rollback (see
                # speculative_generate's matching observe).
                fpad = res.layer_features.shape[1] - len(seq)
                _to = time.perf_counter()
                s.drafter.observe_features(
                    hist_lens[row], res.layer_features[row, fpad:fpad + n_acc + 1])
                self._phase[7] += time.perf_counter() - _to
            if draft and SPEC_MIN_ACCEPT > 0:
                rate = n_acc / len(draft)
                s.acc_ema = SPEC_EMA * s.acc_ema + (1.0 - SPEC_EMA) * rate
                if s.acc_ema < SPEC_MIN_ACCEPT:
                    s.spec_off = s.spec_backoff
                    s.spec_backoff = min(s.spec_backoff * 2, SPEC_BACKOFF_MAX)
                    s.acc_ema = SPEC_MIN_ACCEPT      # a probe starts neutral
                elif rate == 1.0:
                    s.spec_backoff = SPEC_BACKOFF_MIN
            if draft and ADAPTIVE_K and self._packed_ok():
                # Pick the draft length that buys the most tokens per second. Verify cost by rows on the
                # 65B (the dev GPU): 1-4 rows ~375 ms (GEMV, bandwidth bound), 8 rows ~470, 16 rows ~507; the
                # draft itself costs the same at any length. A fully accepted draft says nothing about
                # how far it would have gone, so it counts as a few more so a short k can climb back.
                obs = n_acc + (4 if n_acc == len(draft) else 0)
                s.acc_len_ema = 0.6 * s.acc_len_ema + 0.4 * obs
                e = s.acc_len_ema
                s.k_cur = min(self.k, 3 if e < 2.0 else 7 if e < 4.0 else 15)
            if s.drafter is not None and hasattr(s.drafter, "feedback"):
                s.drafter.feedback(n_acc, len(draft))
            n_drop = len(draft) - n_acc
            if n_drop:
                s.stats.rebuilds += 1
                # Full-attention side: drop just the rejected tail.
                _tc = time.perf_counter()
                self.past.crop(-n_drop, slot=bi)
                for _st in self._stages:
                    _st.queue("crop", -n_drop, bi)
                self._phase[9] += time.perf_counter() - _tc
                if recur_snap is not None:
                    # Recurrent side has no partial rollback. Preferred:
                    # rebuild the state after seq[0] + the accepted drafts
                    # from what the verify forward recorded (no forward).
                    # Fallback: the whole row goes back to before this step,
                    # which also discards seq[0] and the accepted drafts, so
                    # they are replayed through the model to bring the two
                    # halves back into agreement (what speculative_generate
                    # does on its own fast path).
                    _tr = time.perf_counter()
                    _rolled = ring is not None and self._ring_rollback(ring, bi, 1 + n_acc)
                    self._phase[6] += time.perf_counter() - _tr
                    if not _rolled:
                        _restore_recurrent_state_slot(self.past, recur_snap, bi)
                        for _st in self._stages:
                            _st.queue("restore", bi)
                        replay[bi] = [seq[0]] + draft[:n_acc]
            _te = time.perf_counter()
            self._emit(s, bi, draft[:n_acc] + [next_tok], emitted, finished)
            self._phase[10] += time.perf_counter() - _te
            if self.want_hidden and s.drafter is not None and not s.done:
                # Position n_acc of this row is the one that produced
                # next_tok, so its hidden state is exactly what the
                # drafter needs to propose from -- on a REJECTED step as
                # much as on a fully accepted one. This forward already
                # computed it; the rejected tail sitting after it is
                # simply not read.
                #
                # Skipping the update on rejection (the old
                # n_acc == len(draft) guard) was not the "draft quality
                # only" cost it was documented as -- it was a death
                # spiral. A stale hidden makes the next proposal worse,
                # which makes rejection more likely, which keeps hidden
                # stale. Measured on the 27B, per-step (drafted,
                # accepted) went (2,2) (2,2) (2,2) and then (2,0)
                # forever from the first rejection onward: 8/64 accepted
                # where the single-request path (which always refreshes
                # hidden, see speculative_generate's hidden_idx) got
                # 17/46 on the same prompt.
                hpad = res.last_hidden_state.shape[1] - len(seq)
                pos = hpad + n_acc
                if 0 <= pos < res.last_hidden_state.shape[1]:
                    s.drafter.hidden = res.last_hidden_state[row, pos]
                    if getattr(s.drafter, "ctx_window", 0):
                        # TRUE head inputs for every position this step
                        # confirmed: row i is position hist + i, followed by
                        # draft[i] (accepted) or the verified next token.
                        s.drafter.observe(
                            hist_lens[row],
                            res.last_hidden_state[row, hpad:hpad + n_acc + 1],
                            draft[:n_acc] + [next_tok])
        # A slot that finished this step is about to be dropped, so
        # re-advancing its recurrent state would be pure waste (and
        # would mutate a row that is on its way out).
        for bi in finished:
            replay.pop(bi, None)
        _tp = time.perf_counter()
        self._replay_accepted(replay)
        self._phase[11] += time.perf_counter() - _tp
        self._tokens_since_progress += sum(len(t) for _, t in emitted)
        if (admit_rows and torch.cuda.is_available()
                and torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > 2 ** 30):
            # A prefill's activations (several GB at --prefill-chunk 12288) stay reserved by the
            # allocator long after they are free; hand them back now that the prompt is in the cache.
            # Only when there is a lot to give back: a tiny prompt would just re-allocate next time.
            torch.cuda.empty_cache()
        if not admit_rows:
            ph = self._phase
            ph[0] += _t_b - _t_a; ph[1] += _t_c - _t_b; ph[2] += _t_d - _t_c
            ph[3] += time.perf_counter() - _t_d; ph[4] += 1; ph[5] += sum(len(sq) for sq in seqs)
        self._report_progress()
        self._heartbeat.tick()
        # Highest index first so the earlier ones stay valid as each
        # removal shifts later slots down.
        for bi in sorted(finished, reverse=True):
            self._drop(bi)
        return emitted
    def _ceiling_stop(self) -> None:
        """The scheduler's form of the single path's abort-on-VRAM-ceiling. At the --max-memory-gb ceiling the
        request admitted LAST ends here with what it has produced (its client gets a normal end of stream); the
        requests admitted before it keep running. One request per step, so the poll sees the freed memory before
        it takes another. Without this the next allocation hits the allocator's hard cap, the step raises, and
        every live request fails together."""
        chk = self.over_ceiling
        if chk is None or not chk():
            return
        live = self._live()
        if not live:
            return
        bi, s = max(live, key=lambda x: x[1].admit_no)
        s.stats.vram_stop = True
        ctx = self._slot_len(bi)
        self._drop(bi)
        print(f"[engine] VRAM ceiling reached: request {s.req_id} stopped after {len(s.out_ids)} generated tokens "
              f"({ctx} tokens of context freed); {len(self._live())} request(s) keep running", flush=True)

    def _ring_rollback(self, ring: dict, bi: int, t: int) -> bool:
        """Put slot `bi`'s linear-attention state where it would be after
        `t` tokens of this step's verify input (seq[0] + t-1 accepted
        drafts), using what that forward recorded. All-or-nothing: False
        (nothing touched, here or on a stage) if any linear layer lacks a usable capture."""
        plan = _ring_plan(self.past, ring, bi)
        if plan is None:
            return False
        got = [st.ring_plan(bi) for st in self._stages]
        if "fail" in got or not (plan or "ok" in got):
            return False
        _ring_apply(plan, bi, t)
        for st, g in zip(self._stages, got):
            if g == "ok":
                st.queue("ring_apply", bi, t)
        return True

    def _verify(self, s, draft, dl, row_logits, greedy=None) -> tuple[int, int]:
        """Accept/reject this slot's draft and pick its next token --
        greedy exact-match at temperature 0, speculative sampling above
        it (identical to speculative_generate's own verification)."""
        if s.temperature <= 0:
            if greedy is None:
                greedy = row_logits.argmax(-1).tolist()
            n_acc = 0
            for i, d in enumerate(draft):
                if greedy[i] != d:
                    break
                n_acc += 1
            return n_acc, greedy[n_acc]
        fast = _verify_sampled(s, draft, dl, row_logits)
        if fast is not None:
            return fast
        n_acc = 0
        for i, d in enumerate(draft):
            p_target = _probs(row_logits[i], s.temperature, s.top_k, s.top_p)
            _dl = dl[i] if dl is not None else None
            if _dl is None:
                p_draft = None
            elif getattr(s.drafter, "returns_probs", False):
                p_draft = _dl            # already the proposal distribution
            else:
                p_draft = _probs(_dl, s.temperature, s.top_k, s.top_p)
            if random.random() < _accept_prob(p_target, p_draft, d):
                n_acc += 1
            else:
                return n_acc, _resample_residual(p_target, p_draft, d)
        return n_acc, _select(row_logits[n_acc], s.temperature, s.top_k, s.top_p)
    def _emit(self, s, bi: int, tokens: list[int], emitted: list, finished: list) -> None:
        """Commit `tokens` to this slot, stopping at max_new/EOS."""
        newly: list[int] = []
        for t in tokens:
            if len(s.out_ids) >= s.max_new:
                s.done = True
                break
            s.out_ids.append(t)
            newly.append(t)
            if s.drafter is not None and hasattr(s.drafter, "append"):
                s.drafter.append(t)
            if s.on_token is not None:
                s.on_token(t)
            s.stats.emitted += 1
            if t in self.eos_ids:
                s.done = True
                break
        if newly:
            emitted.append((s.req_id if s.req_id is not None else bi, newly))
        if s.done and bi not in finished:
            finished.append(bi)
    def _replay_accepted(self, replay: dict[int, list[int]]) -> None:
        """Re-advance the recurrent state of every rolled-back slot over
        the tokens it kept, in ONE ragged forward.
        Every live slot must take part (the cache carries a row per slot
        and a forward has to present all of them), so slots with nothing
        to replay contribute an empty sequence -- real_lens reports 0 for
        them, so update() commits nothing and the per-slot linear scan
        skips their row entirely.
        The full-attention side already holds these tokens and is not
        being rolled back here, so the copy this replay appends to it is
        removed afterwards; only the recurrent state needed the replay.
        Logits are discarded -- each slot's next token was already chosen
        from the verify pass whose context this reproduces.
        """
        replay = {bi: toks for bi, toks in replay.items() if toks}
        if not replay:
            return
        live = self._live()
        seqs = [replay.get(bi, []) for bi, _ in live]
        if not any(seqs):
            return
        hist = [max(self._slot_len(bi) - len(replay.get(bi, [])), 0)
                for bi, _ in live]
        for (bi, s), h in zip(live, hist):
            s._history_len = h
        self._advance_ragged(seqs, hist)
        for bi, toks in replay.items():
            self.past.crop(-len(toks), slot=bi)
            for _st in self._stages:
                _st.queue("crop", -len(toks), bi)
        # Restore each slot's true length for the next step.
        for bi, s in self._live():
            s._history_len = self._slot_len(bi)
def _mtp_head_kwargs(config) -> dict:
    """Read the MTP head's shape/rope config off the checkpoint's own
    Qwen3_5TextConfig instead of assuming the 9B's numbers.
    MTPHead/MTPHeadCUDA both defaulted to hidden_size=4096, num_heads=16,
    head_dim=256 (the 9B's text_config), which build_drafter()/
    build_drafter_cuda() never overrode. That happens to be exactly right
    for the 9B pack (hidden_size 4096) and silently wrong for any other
    size -- e.g. the 4B pack's hidden_size is 2560, so every reshape inside
    MTPHead.forward assumed a tensor 1.6x too wide, and the head produced
    pure noise (proposed tokens with no relation to the true continuation)
    rather than a clean shape-mismatch error, because every intermediate
    op is `.view`/`.reshape` on already-flat tensors with no shape
    assertions. Reading these off `config` makes the head correct for any
    checkpoint size, matching how attention_kwargs is threaded elsewhere.
    """
    rope = getattr(config, "rope_parameters", {}) or {}
    return dict(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_eps=config.rms_norm_eps,
        rope_theta=rope.get("rope_theta", getattr(config, "rope_theta", 1e7)),
        partial_rotary_factor=getattr(config, "partial_rotary_factor",
                                      rope.get("partial_rotary_factor", 0.25)),
        attn_output_gate=getattr(config, "attn_output_gate", True),
        # MoE MTP head only (see MTPHeadCUDA.forward's is_moe branch);
        # ignored by a dense checkpoint's MTPHead/MTPHeadCUDA. Read off
        # config rather than hardcoded, same reasoning as every other
        # value here -- a MoE checkpoint with a different top_k would
        # otherwise silently route to the wrong number of experts.
        num_experts_per_tok=getattr(config, "num_experts_per_tok", 8),
    )
def build_drafter(model, pack_dir: str, model_dir: str):
    """Load the MTP head and bind it to the base model's embedding + lm_head."""
    from qv_band.decode.mtp_cpu import load_mtp, MTPHead
    w = load_mtp(pack_dir, model_dir)
    head = MTPHead(w, **_mtp_head_kwargs(model.config))
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    print(f"MTP head: {w.packed_bytes/1e6:.0f} MB packed")
    return MTPDrafter(head, embed, lm_head)
def build_drafter_cuda(model, pack_dir: str, model_dir: str, device: str = "cuda"):
    """CUDA counterpart of build_drafter(): MTP head on GPU-resident packs."""
    from qv_band.decode.mtp_cuda import load_mtp_cuda, MTPHeadCUDA
    w = load_mtp_cuda(pack_dir, model_dir, device=device)
    head = MTPHeadCUDA(w, device=device, **_mtp_head_kwargs(model.config))
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    print(f"MTP head (cuda): loaded on {device}")
    return MTPDrafter(head, embed, lm_head)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*")
    ap.add_argument("--draft", type=int, default=0,
                    help="draft tokens per step; 0 (default) = plain greedy")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=40)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pack-dir")
    ap.add_argument("--drafter", choices=["n-gram", "mtp", "dspark", "dflash", "none"],
                    default="n-gram",
                    help="n-gram = prompt-lookup (no weights, works today); "
                         "mtp = the checkpoint's draft head, one pass per "
                         "token; dspark = DSpark-style two-pass semi-"
                         "autoregressive block draft built on the same head, "
                         "with confidence-scheduled verification; dflash = a "
                         "separately trained DFlash/DFlash 2 block drafter "
                         "(needs --draft-model)")
    ap.add_argument("--draft-model",
                    help="DFlash draft model dir (with --drafter dflash)")
    ap.add_argument("--draft-pack",
                    help="qvb8 pack of --draft-model (optional)")
    ap.add_argument("--ctx", type=int, default=3,
                    help="n-gram context length to match on")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature; 0 (default) = greedy, and "
                         "verification is exact-match against the argmax")
    ap.add_argument("--top-k", type=int, default=0,
                    help="top-k filter for sampling (0 = no filter)")
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="nucleus filter for sampling (1.0 = no filter)")
    ap.add_argument("--compare", action="store_true",
                    help="also run a non-speculative control and report the "
                         "speedup (streams match only at temperature 0)")
    args = ap.parse_args()
    if args.drafter == "dflash" and not args.draft_model:
        ap.error("--drafter dflash needs --draft-model")
    text = " ".join(args.prompt).strip()
    if not text:
        ap.error("give some text")
    pack = args.pack_dir or LLM_PACK_DIR
    from qv_band.run.run_inference_cpu import load_qv_model_cpu
    bk = "cpu (torch)"
    print(f"backend: {bk}   pack: {pack}   drafter: {args.drafter}")
    model, tok = load_qv_model_cpu(args.model, pack)
    if args.drafter == "mtp":
        drafter = build_drafter(model, pack, args.model)
    elif args.drafter == "dspark":
        from qv_band.decode.dspark_draft import build_dspark_drafter
        drafter = build_dspark_drafter(model, pack, args.model)
    elif args.drafter == "dflash":
        from qv_band.decode.dflash_draft import build_dflash_drafter
        drafter = build_dflash_drafter(
            model, args.draft_model, pack_dir=args.draft_pack,
            device=str(next(model.parameters()).device))
    elif args.drafter == "n-gram":
        from qv_band.decode.ngram_draft import NGramDrafter
        drafter = NGramDrafter(max_ctx=args.ctx)
    else:
        drafter = None
    print(f"loaded, RSS {_rss_gb():.1f} GB")
    s = text
    if args.chat:
        s = tok.apply_chat_template([{"role": "user", "content": text}],
                                    tokenize=False, add_generation_prompt=True)
    ids = tok(s, return_tensors="pt").input_ids
    print(f"\n> {text}\n" + "=" * 64)
    t0 = time.time()
    out, st = speculative_generate(model, drafter, tok, ids,
                                   k=args.draft, max_new=args.max_new_tokens,
                                   temperature=args.temperature,
                                   top_k=args.top_k, top_p=args.top_p)
    dt = time.time() - t0
    print("\n" + "=" * 64)
    print(f"{args.drafter} k={args.draft}: {len(out)} tokens in {dt:.1f}s = "
          f"{len(out)/dt:.3f} tok/s")
    print(st.report())
    print(f"RSS {_rss_gb():.1f} GB")
    if args.compare:
        print("\n" + "-" * 64 + "\nplain non-speculative control ...")
        # A fresh drafter (and hence a fresh cache) per run: the speculative
        # pass leaves its cache cropped, and reusing it desynchronises the
        # recurrent layers' conv window.
        t0 = time.time()
        out2, _ = speculative_generate(model, None, tok, ids.clone(), k=0,
                                       max_new=args.max_new_tokens, stream=False,
                                       temperature=args.temperature,
                                       top_k=args.top_k, top_p=args.top_p)
        dt2 = time.time() - t0
        print(f"control: {len(out2)} tokens in {dt2:.1f}s = {len(out2)/dt2:.3f} tok/s")
        same = out[:min(len(out), len(out2))] == out2[:min(len(out), len(out2))]
        note = ("streams match in distribution only (stochastic sampling)"
                if args.temperature > 0
                else f"streams identical: {'YES' if same else 'NO'}")
        print(f"\nspeedup {(len(out)/dt)/(len(out2)/dt2):.2f}x   {note}")
if __name__ == "__main__":
    main()