"""Speculative decoding for targets that draft by themselves: the draft plan (which drafts to verify), shared by the
eager path (engine/target_engine) and the static-slot path (engine/static_decode.StaticStepper.spec).

THE PROTOCOL (a BatchTarget, engine/adapter.py, opts in by having these):
  eager
    spec_propose(rows [(state, last token, (temperature, top_k, top_p))])
        -> [(drafts [k], each draft's chance [k], distributions [k][vocab] or None)]   the target's own drafter
                                                    (a sampled row draws its drafts and returns their distributions)
    spec_verify(rows [(state, [last] + drafts)]) -> logits [sum of lengths, vocab] fp32 of EVERY row token; the states
                                                    hold all of them until spec_keep
    spec_keep(rows [(state, keep)])                 keep the first `keep` tokens of each state's last verify (the rest
                                                    rolled back; the drafter follows)
    spec_max_draft                                  drafts per row at most
  static slots (StaticStepper): st_vtok / st_vslot / st_vj / st_vrow / st_vstart / st_acc / st_vlogits buffers and
    static_verify(R), static_commit(); a target with a drafter also st_pslot / st_panchor / st_draft / st_dprob /
    st_ddist and static_propose(n, sp). Tensor Parallel cards other than the first only verify (no drafter).
Acceptance (verify_row): temperature 0 greedy (the target's own greedy text); above it speculative sampling (the
target's own distribution).

THE PLAN: draft j of a row survives with the product of the drafter's first j probabilities; the draft rows of ONE
verify go to the most likely drafts of any row while they raise the expected tokens per second of the whole step
(measured costs: a verify of R rows, a proposal of n rows). A step where no draft paid is followed by SKIP steps that
do not propose at all, then the planner tries again.
"""
from __future__ import annotations

import math
import random

import torch


class SpecPlanner:
    """Draft lengths for one engine: measured costs (verify of R rows, proposal of n rows) + each request's chance its
    j-th draft survives: the drafter's probabilities when it gives them (plan), else that request's measured
    acceptance per position (plan_k: the drafter's feedback). A request whose drafts do not pay skips SKIP proposals
    (plain steps), then is tried again."""
    SKIP = 3
    PRIOR = 0.6                                           # a new request's chance a draft survives, per position
    EXPLORE = 16                                          # draft-free decisions in a row before one forced full draft

    def __init__(self):
        self.vcost: dict[int, float] = {}                 # verify rows -> seconds
        self.pcost: dict[int, float] = {}                 # proposing rows -> seconds
        self.skip: dict = {}                              # request key -> plain steps left before a proposal
        self.rates: dict = {}                             # request key -> [P(draft j accepted | j-1 accepted)]
        self.idle: dict = {}                              # request key -> draft-free decisions in a row

    def observe(self, table: dict, key: int, seconds: float, exact: bool = False) -> None:
        """A measured cost (exact: a graph timed once). Real steps: a lower time replaces it, a higher one moves it
        slowly (a warm-up outlier must not stop drafting for good; a longer context raises it over time)."""
        old = table.get(key)
        if exact or old is None or seconds < old:
            table[key] = seconds
        else:
            table[key] = 0.8 * old + 0.2 * seconds

    def feedback(self, key, n_acc: int, n_prop: int) -> None:
        """A request's verify result: of n_prop drafts the first n_acc were accepted."""
        r = self.rates.setdefault(key, [])
        for j in range(min(n_prop, n_acc + 1)):          # position j was tested; it passed when j < n_acc
            while len(r) <= j:
                r.append(self.PRIOR)
            r[j] = 0.8 * r[j] + 0.2 * (1.0 if j < n_acc else 0.0)

    def forget(self, key) -> None:
        self.rates.pop(key, None)
        self.skip.pop(key, None)
        self.idle.pop(key, None)

    def verify_cost(self, R: int):
        """Seconds of a verify of R rows: measured; else the line through the smallest and largest measured; with one
        measured size, that cost (a guess that does not favour fewer rows: the plan then tries and measures)."""
        c = self.vcost
        if R in c:
            return c[R]
        if not c:
            return None
        ks = sorted(c)
        if len(ks) == 1:
            return c[ks[0]]
        a, b = ks[0], ks[-1]
        return max(c[a] + (c[b] - c[a]) / (b - a) * (R - a), 1e-6)

    def want_proposal(self, key=None) -> bool:
        n = self.skip.get(key, 0)
        if n > 0:
            self.skip[key] = n - 1
            return False
        return True

    def plan(self, probs: list, kmax: int, keys=None, whole: bool = False) -> list:
        """probs [rows][k] (each draft's chance to be accepted given the earlier ones) -> drafts to verify per row, for
        ONE verify of all rows. No verify costs measured yet: every draft (that measures them). whole: a row verifies
        all its kmax drafts or none (graphs of fixed width)."""
        n = len(probs)
        keys = list(keys) if keys is not None else [None] * n
        cand = []                                         # (expected tokens, rows, row): most tokens per row first
        for i, pr in enumerate(probs):
            surv, tot = 1.0, 0.0
            for pj in pr[:kmax]:
                surv *= pj
                tot += surv
                if not whole:
                    cand.append((surv, 1, i))
            if whole and pr:
                cand.append((tot, min(len(pr), kmax), i))
        if not self.vcost:                                # nothing measured: every draft (measures a full verify)
            k = [min(len(p), kmax) for p in probs]
        elif n not in self.vcost:                         # the plain step not measured yet: one (no skip after it)
            return [0] * n
        else:
            cand.sort(key=lambda x: -x[0] / x[1])
            pc = self.pcost.get(n, 0.0)
            best, m_best, tokens, rows = n / self.verify_cost(n), 0, float(n), n
            for m, (sv, w, _) in enumerate(cand, 1):
                tokens += sv
                rows += w
                r = tokens / (self.verify_cost(rows) + pc)
                if r > best:
                    best, m_best = r, m
            k = [0] * n
            for _, w, i in cand[:m_best]:
                k[i] += w
        if not any(k):                                    # nothing paid: the rows skip proposals for a while
            for key in set(keys):
                self.skip[key] = self.SKIP
        return k

    def plan_k(self, rows: list, whole: bool = False) -> list:
        """rows [(request key, max drafts)] -> drafts to propose per row, BEFORE proposing (a drafter whose draft
        length costs, or that gives no probabilities): from each request's measured acceptance per position.
        whole: all of a row's drafts or none (see plan). A request with EXPLORE draft-free steps in a row drafts once
        in full (costs and acceptance are only measured by drafting)."""
        out = [0] * len(rows)
        live = []
        for i, (key, kmax) in enumerate(rows):
            if kmax <= 0:
                continue
            if self.idle.get(key, 0) >= self.EXPLORE:
                self.idle[key], self.skip[key], out[i] = 0, 0, kmax
            elif self.want_proposal(key):
                live.append((i, key, kmax))
            else:
                self.idle[key] = self.idle.get(key, 0) + 1
        if live:
            probs = [(self.rates.get(key, []) + [self.PRIOR] * kmax)[:kmax] for _, key, kmax in live]
            ks = self.plan(probs, max(kmax for _, _, kmax in live), [key for _, key, _ in live], whole)
            for (i, key, kmax), k in zip(live, ks):
                out[i] = k
                if k:
                    self.idle[key] = 0
                    continue
                if key in self.rates:                     # a plain step: its estimate drifts back toward the prior,
                    self.rates[key] = [0.9 * r + 0.1 * self.PRIOR for r in self.rates[key]]   # so it tries again
                self.idle[key] = self.idle.get(key, 0) + 1
        return out


def accept(drafts: list, target_next: list) -> int:
    """Greedy acceptance: drafts d[0..] vs the target's token after each input -> accepted count a (then the target's
    own token target_next[a] follows)."""
    a = 0
    while a < len(drafts) and drafts[a] == target_next[a]:
        a += 1
    return a


def eager_step(target, planner: SpecPlanner, rows: list) -> list:
    """One speculative step of eager requests [(state, last token, (temperature, top_k, top_p))] -> tokens emitted
    per row: the planner's drafts, ONE verify forward of all rows, acceptance (greedy or speculative sampling), the
    rest rolled back (spec_keep). Costs measured on the way."""
    import time
    n = len(rows)
    drafts, dists = [[] for _ in rows], [None] * n
    if planner.want_proposal():
        t0 = time.perf_counter()
        props = target.spec_propose(rows)
        planner.observe(planner.pcost, n, time.perf_counter() - t0)
        k = planner.plan([p for _, p, _ in props], target.spec_max_draft)
        drafts = [d[:ki] for (d, _, _), ki in zip(props, k)]
        dists = [None if q is None else q[:ki] for (_, _, q), ki in zip(props, k)]
    vrows = [(st, [last] + d) for (st, last, _), d in zip(rows, drafts)]
    t0 = time.perf_counter()
    logits = target.spec_verify(vrows)
    greedy = logits.argmax(-1).tolist()                   # (the host read waits for the verify)
    planner.observe(planner.vcost, sum(len(toks) for _, toks in vrows), time.perf_counter() - t0)
    out, keeps, r0 = [], [], 0
    for (st, toks), d, q, (_, _, sp) in zip(vrows, drafts, dists, rows):
        R = len(toks)
        a, nxt = verify_row(d, q, logits[r0:r0 + R], sp[0], sp[1], sp[2], True, greedy[r0:r0 + R])
        out.append(d[:a] + [nxt])
        keeps.append((st, a + 1))
        r0 += R
    target.spec_keep(keeps)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# acceptance: greedy exact match at temperature 0, speculative sampling above it (Leviathan et al. 2023, Chen et al.
# 2023) -- the one implementation every engine uses (engine/target_engine, engine/static_decode, decode/spec_decode).
def sampling_probs(logits: torch.Tensor, temperature: float, top_k: int,
           top_p: float) -> torch.Tensor:
    """The exact distribution sample_token() would draw from at temperature > 0:
    temperature-scaled, top-k then nucleus (top-p) filtered, softmaxed -- the
    same order transformers' own LogitsProcessor pipeline applies for
    TopKLogitsWarper -> TopPLogitsWarper.  Shared by sample_token(), accept_prob()
    and resample_residual() so the sampled token, the acceptance probability
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


def sample_token(logits: torch.Tensor, temperature: float, top_k: int,
           top_p: float = 1.0) -> int:
    """argmax when temperature is 0 (the greedy path everything else here is
    built around); otherwise multinomial sampling, top-k then nucleus
    (top-p) filtered, temperature-scaled -- see sampling_probs."""
    if temperature <= 0:
        return int(logits.argmax())
    return int(torch.multinomial(sampling_probs(logits, temperature, top_k, top_p), 1))


def accept_prob(p_target: torch.Tensor, p_draft: torch.Tensor | None,
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


def resample_residual(p_target: torch.Tensor, p_draft: torch.Tensor | None,
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


def topk_dist(logits: torch.Tensor, temperature: float, top_k: int, top_p: float):
    """sampling_probs() for every row of `logits` at once, restricted to each row's top_k tokens
    (everything outside them has probability 0 in sampling_probs too): (token ids, probabilities,
    count of logits >= the k-th). The count differs from top_k only on a tie at the k-th
    logit, where sampling_probs keeps every tied token -- the caller then falls back to it."""
    vals, idx = torch.topk(logits, top_k, dim=-1)               # sorted, descending
    ties = (logits >= vals[:, -1:]).sum(-1)
    p = torch.softmax(vals.float() / temperature, dim=-1)
    if 0 < top_p < 1.0:
        p = p.masked_fill((p.cumsum(-1) - p) > top_p, 0.0)      # the nucleus, as in sampling_probs
        p = p / p.sum(-1, keepdim=True)
    return idx, p, ties


def _verify_sampled(draft, dl, row_logits, temperature, top_k, top_p, returns_probs):
    """Speculative-sampling verification (the temperature > 0 branch of verify_row) done on each
    position's top_k tokens and ONE device-to-host copy, or None when that is not exact.

    WHY: per drafted position sampling_probs costs -- two full-vocab (248k) sorts, several softmaxes
    and a GPU sync per accept test. Measured on the 27B that was ~55 ms of a ~260 ms decode step. A
    token outside the top_k has probability 0 under the target, so acceptance and the residual
    only ever involve the top_k tokens; this computes them once for all positions and decides on
    the host. Same distribution, same accept/residual rule."""
    k, V, n = top_k, row_logits.shape[-1], len(draft)
    if k <= 0 or k >= V:
        return None
    if dl is not None and any(x is not None for x in dl) and not returns_probs:
        return None                                      # a draft that still needs sampling_probs() of its logits
    idx, p, ties = topk_dist(row_logits, temperature, k, top_p)
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


def verify_row(draft, dl, row_logits, temperature, top_k=0, top_p=1.0, returns_probs=False, greedy=None):
    """Accept / reject one row's drafts and pick its next token -> (accepted count, next token). draft [k] tokens,
    dl the drafter's per-draft logits or probabilities (returns_probs) or None, row_logits [k+1, vocab] the target's
    logits after [last] + draft; greedy: the target's argmax per position when already on the host."""
    if temperature <= 0:
        g = greedy if greedy is not None else row_logits.argmax(-1).tolist()
        a = accept(draft, g)
        return a, g[a]
    fast = _verify_sampled(draft, dl, row_logits, temperature, top_k, top_p, returns_probs)
    if fast is not None:
        return fast
    for i, d in enumerate(draft):
        p_target = sampling_probs(row_logits[i], temperature, top_k, top_p)
        _dl = dl[i] if dl is not None else None
        p_draft = None if _dl is None else (_dl if returns_probs else sampling_probs(_dl, temperature, top_k, top_p))
        if random.random() >= accept_prob(p_target, p_draft, d):
            return i, resample_residual(p_target, p_draft, d)
    return len(draft), sample_token(row_logits[len(draft)], temperature, top_k, top_p)
