"""Prefix keys shared by the LLM server and the router (qv_band/design_doc/replica_pool.md, section 3.3).

A prompt's token ids are cut into BLOCK-token blocks; the key of block i is a hash of (key of block i-1, the ids of
block i), so one key names one exact prefix of BLOCK * (i + 1) tokens. A server lists the keys of the prefixes its
engine can restore; the router asks one server for a request's keys and finds, per replica, the longest key it holds.
Checkpoints sit on chunk boundaries (multiples of the prefill chunk: 1024 or 12288), all multiples of BLOCK.
"""
from __future__ import annotations

import hashlib
import struct

BLOCK = 1024


def block_keys(ids) -> list:
    """[[length, hex key], ...] for every full BLOCK of `ids` (a list of ints)."""
    out, prev = [], b""
    n = len(ids) // BLOCK * BLOCK
    for a in range(0, n, BLOCK):
        h = hashlib.blake2b(prev + struct.pack(f"<{BLOCK}i", *ids[a:a + BLOCK]), digest_size=16).digest()
        out.append([a + BLOCK, h.hex()])
        prev = h
    return out


def prefix_key(ids) -> list | None:
    """[length, hex key] of the whole of `ids`, or None when its length is not a multiple of BLOCK."""
    if not ids or len(ids) % BLOCK:
        return None
    return block_keys(ids)[-1]


def floor_key(ids) -> list | None:
    """[length, hex key] of the longest whole-BLOCK prefix of `ids` (a saved prefix of any length is named by its last
    full block: a conversation turn's prompt-end state becomes transferable too), or None under one BLOCK."""
    n = len(ids) // BLOCK * BLOCK
    return block_keys(ids[:n])[-1] if n else None
