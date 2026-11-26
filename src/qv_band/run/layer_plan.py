"""THE layer placement of every model (LLM decoder layers, DiT / text-encoder blocks, ...): which contiguous layers go
on which card, and which stream. Model independent: it sees per-layer bytes and per-card capacities, nothing else.

  plan_layers(sizes, caps, stream=True) -> (counts per card, first streamed layer)
    everything fits the cards together: a contiguous split balanced by capacity (capacity_split), nothing streams;
    else, with stream=True: the cards are filled front to back (stream_caps) and the rest streams; stream=False raises.
Callers: run/placement.py (LLM pipeline over cards), ai_engine pipelines/common/block_pipeline.py (block stacks)."""
from __future__ import annotations


def balanced_split(per_layer: list[int], n_cards: int, main_extra: int) -> list[int]:
    """First layer of cards 2..N so that card loads are as equal as possible; the last (main) card also carries
    `main_extra` bytes. After each card the target is re-derived from what is left, so the error does not pile up
    on the main card. Every card gets at least one layer."""
    L = len(per_layer)
    if n_cards > L:
        raise RuntimeError(f"{n_cards} cards for {L} layers: at most one card per layer")
    bounds, i = [], 0
    left = sum(per_layer) + main_extra
    for card in range(n_cards - 1):
        target = left / (n_cards - card)
        acc = per_layer[i]
        j = i + 1
        # take layer j while that brings the card nearer its target, keeping a layer for every later card
        while j < L - (n_cards - 1 - card) and abs(acc + per_layer[j] - target) < abs(acc - target):
            acc += per_layer[j]
            j += 1
        bounds.append(j)
        left -= acc
        i = j
    return bounds


def capacity_split(per_layer: list[int], caps: list[int], main_extra: int) -> list[int]:
    """balanced_split with a capacity per card (bytes it can take): each card's target is its SHARE of what is left,
    cap_c / sum(caps of the cards still open). Equal caps give exactly balanced_split. Raises when the weights do not
    fit the capacities in total, naming both (github issue #1: the balanced split put 1/3 of a 65B on a 10 GB card
    next to a 24 GB one and ran out of memory)."""
    L, n = len(per_layer), len(caps)
    if n > L:
        raise RuntimeError(f"{n} cards for {L} layers: at most one card per layer")
    need = sum(per_layer) + main_extra
    if need > sum(caps):
        raise RuntimeError(f"pipeline: the weights need {need / 2**30:.1f} GiB, the cards have "
                           f"{' + '.join(f'{c / 2**30:.1f}' for c in caps)} GiB free for them (after the per-card "
                           f"reserve): free memory on the cards, add a card, or lower --pipeline-reserve-mb")
    bounds, i, left = [], 0, need
    for card in range(n - 1):
        target = left * caps[card] / sum(caps[card:])
        acc = per_layer[i]
        j = i + 1
        while j < L - (n - 1 - card) and abs(acc + per_layer[j] - target) < abs(acc - target):
            acc += per_layer[j]
            j += 1
        while acc > caps[card] and j > i + 1:           # never over a card's capacity: give layers back
            j -= 1
            acc -= per_layer[j]
        bounds.append(j)
        left -= acc
        i = j
    def loads(bs):
        edges = [0] + bs + [L]
        return [sum(per_layer[a:b]) + (main_extra if c == n - 1 else 0) for c, (a, b) in enumerate(zip(edges, edges[1:]))]

    for _ in range(L * n):
        ld = loads(bounds)
        over = [c for c in range(n) if ld[c] > caps[c]]
        if not over:
            break
        o, edges = over[-1], [0] + bounds + [L]
        moved = False
        for c in range(o - 1, -1, -1):                  # layers only move to an EARLIER card (the order is fixed)
            if edges[c + 2] - edges[c + 1] >= 2 and ld[c] + per_layer[bounds[c]] <= caps[c]:
                bounds[c] += 1
                moved = True
                break
        if not moved:
            break
    ld = loads(bounds)
    if any(ld[c] > caps[c] for c in range(n)):
        raise RuntimeError("pipeline: no split of contiguous layers fits the cards: loads "
                           + ", ".join(f"{x / 2**30:.1f}" for x in ld) + " GiB against "
                           + ", ".join(f"{c / 2**30:.1f}" for c in caps) + " GiB free (after the per-card reserve; the "
                           "last card also holds embedding / lm_head / drafter). Free memory on the cards, change "
                           "their order, add a card, or set --pipeline-split by hand")
    return bounds


def plan_layers(sizes: list[int], caps: list[int], stream: bool = True, stream_caps: list[int] | None = None,
                main_extra: int = 0):
    """(counts per card, first streamed layer). `caps`: bytes each card takes when nothing streams; `stream_caps`: the
    same with the stream window kept free (default: caps). The last card also holds `main_extra` (capacity_split)."""
    L, n = len(sizes), len(caps)
    if L == 0:
        return [0] * n, 0
    if n == 1:
        if sizes and sum(sizes) + main_extra <= caps[0]:
            return [L], L
    elif L >= n and sum(sizes) + main_extra <= sum(caps):
        try:
            bounds = capacity_split(sizes, caps, main_extra)
            edges = [0] + bounds + [L]
            return [b - a for a, b in zip(edges, edges[1:])], L
        except RuntimeError:
            pass                                       # no contiguous split fits: fill and stream below
    if not stream:
        capacity_split(sizes, caps, main_extra)        # raises with the reason
    counts, i = [], 0
    for c in (stream_caps or caps):
        acc, k = 0, 0
        while i + k < L and acc + sizes[i + k] <= c:
            acc += sizes[i + k]; k += 1
        counts.append(k); i += k
    return counts, i


