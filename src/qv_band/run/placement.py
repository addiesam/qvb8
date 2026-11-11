"""Pipeline placement of an LLM over any number of cards (N >= 2), one process. Set with the server flags
--pipeline / --pipeline-split (ai_engine/cli/pipeline_flags.py), which reach the loader as:

    QV_PIPELINE=cuda:0,cuda:1,...,cuda:N-1    the cards in layer order; the LAST one is the main card (embedding,
                                              final norm, lm_head, MTP / vision, the drafter, request tensors)
    QV_PIPELINE_SPLIT=s1,...,sN-1             optional: the first decoder layer of cards 2..N. Without it the split
                                              balances weight bytes (pack manifest + the dense safetensors): every
                                              card carries about the same bytes, the main card counting its
                                              non-layer weights too. Same pack + same cards -> same split.

Each card holds a contiguous range of decoder layers; run_inference._install_pipeline_hooks moves a layer's inputs
to its card and makes that card current while the layer runs, so a model forward (and the scheduler's) is
device-transparent. A card may appear twice only to SIMULATE more cards on one GPU (tests): it changes nothing.

A model spread over machines (decode/remote_stage.py): each process loads only its own part.

    QV_LAYERS_HERE=a:b        this process holds decoder layers a..b-1 and everything that is not a decoder layer
                              (the main machine; "a:" = to the last layer)
    QV_LAYERS_HERE=a:b,only   decoder layers a..b-1 and nothing else (a stage worker)

The other weights are never read: their modules stay empty. With QV_PIPELINE the balanced split counts only the
weights held here.
"""
from __future__ import annotations

import json
import os
import re
import struct
from bisect import bisect_right

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8,
             "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def requested() -> list[str] | None:
    """The card list from QV_PIPELINE, or None (one card)."""
    raw = (os.environ.get("QV_PIPELINE", "") or "").strip()
    if not raw:
        if (os.environ.get("QV_PIPELINE_SPLIT") or "").strip():
            raise RuntimeError("--pipeline-split (QV_PIPELINE_SPLIT) needs --pipeline (QV_PIPELINE): the cards")
        return None
    devs = [d.strip() for d in raw.split(",") if d.strip()]
    bad = [d for d in devs if not re.fullmatch(r"cuda:\d+", d)]
    if bad:
        raise RuntimeError(f"QV_PIPELINE={raw!r}: every card as cuda:N (got {bad})")
    if len(devs) < 2:
        raise RuntimeError(f"QV_PIPELINE={raw!r}: needs at least two cards (one card = leave it unset)")
    return devs


def _layer_of(name: str):
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def layers_here():
    """(a, b or None, only) from QV_LAYERS_HERE, or None when this process holds the whole model."""
    raw = (os.environ.get("QV_LAYERS_HERE", "") or "").strip()
    if not raw:
        return None
    rng, _, tail = raw.partition(",")
    a, sep, b = rng.partition(":")
    if not (sep and a.isdigit() and (b == "" or b.isdigit()) and tail in ("", "only")):
        raise RuntimeError(f"QV_LAYERS_HERE={raw!r}: a:b or a:b,only")
    return int(a), (int(b) if b else None), tail == "only"


def held_fn():
    """name -> True when the weight / module of that name is loaded by this process."""
    here = layers_here()
    if here is None:
        return lambda name: True
    a, b, only = here

    def held(name: str) -> bool:
        li = None if (name.startswith("mtp.") or ".visual." in name) else _layer_of(name)
        if li is None:
            return not only
        return a <= li and (b is None or li < b)
    return held


def weight_bytes(model_dir: str, pack_dir: str | None, n_layers: int) -> tuple[list[int], int]:
    """(bytes per decoder layer, bytes of everything else) as the model will hold them: packed tensors at their
    pack size (manifest), the rest at their dense size (safetensors headers, no tensor read)."""
    per = [0] * n_layers
    other = 0
    packed = set()
    src = pack_dir or model_dir
    man_path = os.path.join(pack_dir, "manifest.json") if pack_dir else None
    if man_path and os.path.isfile(man_path):
        with open(man_path) as fh:
            tensors = json.load(fh).get("tensors") or {}
        for name, info in tensors.items():
            base = name.split("#", 1)[0]
            nb = int(info.get("bytes") or 0)
            if not nb and info.get("pack"):
                f = os.path.join(pack_dir, info["pack"])
                nb = os.path.getsize(f) if os.path.exists(f) else 0
            li = _layer_of(base)
            if li is not None and li < n_layers:
                per[li] += nb
            else:
                other += nb
            packed.add(base)
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".safetensors"):
            continue
        with open(os.path.join(src, fn), "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            hdr = json.loads(fh.read(hlen))
        for name, info in hdr.items():
            if name == "__metadata__" or name in packed or name.split("#", 1)[0] in packed:
                continue
            nb = _DT_BYTES.get(info.get("dtype"), 2)
            for s in info.get("shape", []):
                nb *= int(s)
            li = _layer_of(name)
            if li is not None and li < n_layers:
                per[li] += nb
            else:
                other += nb
    return per, other


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


class Placement:
    def __init__(self, devices: list[str], bounds: list[int], n_layers: int):
        if len(bounds) != len(devices) - 1 or any(b <= a for a, b in zip([0] + bounds, bounds + [n_layers])):
            raise RuntimeError(f"pipeline split {bounds} for {len(devices)} cards / {n_layers} layers: needs "
                               f"{len(devices) - 1} increasing layer indices in 1..{n_layers - 1}")
        self.devices, self.bounds, self.n_layers = devices, bounds, n_layers
        self.main = devices[-1]

    def card_of_layer(self, i: int) -> str:
        return self.devices[bisect_right(self.bounds, i)]

    def place(self, name: str) -> str:
        """The card a parameter / buffer / module of this name belongs on."""
        if name.startswith("mtp.") or ".visual." in name:
            return self.main
        li = _layer_of(name)
        return self.card_of_layer(li) if li is not None and li < self.n_layers else self.main

    def describe(self) -> str:
        edges = [0] + self.bounds + [self.n_layers]
        parts = [f"layers {a}..{b - 1} on {d}" for d, a, b in zip(self.devices, edges, edges[1:])]
        return "; ".join(parts) + f"; embed/norm/lm_head/drafter on {self.main}"


def build(devices: list[str], model_dir: str, pack_dir: str | None, n_layers: int) -> Placement:
    raw = (os.environ.get("QV_PIPELINE_SPLIT", "") or "").strip()
    if raw:
        try:
            bounds = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            raise RuntimeError(f"QV_PIPELINE_SPLIT={raw!r}: comma-separated layer indices") from None
        how = "QV_PIPELINE_SPLIT"
    else:
        per, other = weight_bytes(model_dir, pack_dir, n_layers)
        here = layers_here()
        if here is not None:                     # a model over machines: count what this process holds
            a, b, only = here
            per = [nb if a <= i and (b is None or i < b) else 0 for i, nb in enumerate(per)]
            other = 0 if only else other
        bounds = balanced_split(per, len(devices), other)
        how = "balanced by weight bytes" + (" held here" if here is not None else "")
    p = Placement(devices, bounds, n_layers)
    p.how = how
    return p
