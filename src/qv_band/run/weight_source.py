"""Where each sub-model's weights come from, chosen when the engine starts -- ONE setting for every model, one card or
several DP servers sharing a machine.

QV_PLACE="transformer=dram:shared,text_encoder=cache:shared,vae=vram"     (launchers: --place ...)

  sub-model   the pack sub-folder (transformer, text_encoder, vae, ...; "*" = every sub-model not named)
  tier        vram   on the card as far as it fits (only what cannot fit streams)
              dram   streamed blocks stay pinned in host RAM (budget QV_STREAM_HOST_GB; default: all of them)
              cache  streamed from the files, which stay in the page cache (RAM the kernel reclaims; read-only)
              disk   streamed from the files, dropped from the page cache after each read (least RAM)
  share       shared one host copy for every server of this machine (dram: /dev/shm, RAM; cache: the page cache)
              own    each server keeps its own copy (default)
A sub-model not named keeps the launcher's env (QV_STREAM_FROM, QV_STREAM_HOST_GB): own copies, pages dropped.
Nothing here writes to a disk."""
from __future__ import annotations

import os

TIERS = ("vram", "dram", "cache", "disk")


def table() -> dict:
    out = {}
    for item in os.environ.get("QV_PLACE", "").replace(";", ",").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        sub, val = item.split("=", 1)
        tier, _, share = val.partition(":")
        tier, share = tier.strip().lower(), (share.strip().lower() or "own")
        if tier not in TIERS:
            raise ValueError(f"QV_PLACE: {sub}={val}: tier must be one of {TIERS}")
        if share not in ("shared", "own"):
            raise ValueError(f"QV_PLACE: {sub}={val}: share must be shared or own")
        out[sub.strip()] = (tier, share == "shared")
    return out


def of(name_or_path):
    """(tier, shared) for a sub-model name or any file path inside its pack folder; None when not set."""
    if not name_or_path:
        return None
    t = table()
    if not t:
        return None
    s = str(name_or_path)
    if s in t:
        return t[s]
    parts = set(s.split(os.sep)) | set(os.path.realpath(s).split(os.sep))
    for sub, v in t.items():
        if sub != "*" and sub in parts:
            return v
    return t.get("*")


def tier(name_or_path):
    v = of(name_or_path)
    return v[0] if v else None


def keeps_page_cache(path: str) -> bool:
    v = of(path)
    if v is not None:
        return v[0] == "cache"                        # dram / vram hold their own copies: the pages would double RAM
    return False


def shared_host(path) -> bool:
    v = of(path)
    if v is not None:
        return v[1] and v[0] == "dram"
    return False


def host_gb(name_or_path):
    """Pinned host budget (GB) for this sub-model's streamed blocks: None = none (stream from the files)."""
    t = tier(name_or_path)
    env = os.environ.get("QV_STREAM_HOST_GB")
    if t is None:
        return float(env) if env else None
    if t == "dram":
        # dram = the streamed blocks pinned in host RAM: a positive QV_STREAM_HOST_GB limits it, 0 / unset = all of them
        # (a single-card launcher exports its HOST_GB, 0 = "no buffer", which must not turn a dram placement into the disk)
        return float(env) if env and float(env) > 0 else float("inf")
    return None


def from_disk(name_or_path, default: bool) -> bool:
    """Stream from the files (True) or from pinned host RAM (False)."""
    t = tier(name_or_path)
    if t in ("cache", "disk"):
        return True
    if t == "dram":
        return False
    return default


def describe() -> str:
    t = table()
    return ", ".join(f"{k}={v[0]}{':shared' if v[1] else ''}" for k, v in t.items()) or "(launcher env)"
