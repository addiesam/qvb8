"""The two escape tables of a pack: the fp16 table and the bf16 table -- READER side.

A pack certified for both precisions (restore "fp16+bf16") stores ONE escape list in each pack file: the escapes the
fp16 certificate needs, plus the positions the bf16 check added. That full list is the bf16 table. The fp16 table (the
subset an fp16 / fp32 decode needs) is shipped with the pack in escape_tables.json, next to manifest.json.

    fp16 table  -- used by the GEMV, fp16 decode, MoE expert kernels       (from escape_tables.json)
    bf16 table  -- used only by the paths that round the weight to bf16   (the pack file's own list)

At load NOTHING is computed and nothing is chosen: the file is read and validated against the pack (same escape count,
every listed position an escape of the pack); a mismatch is an error. An fp16+bf16 pack without the file has no fp16
table: a bf16 run uses the pack's full list (exact values, correct for every path); an fp16 run is refused
(require_fp16_tables at start-up, and the modules' forward). escape_tables.json comes with the pack on Hugging Face.

File format (version 2): {"format": "qvb8-escape-tables", "version": 2, "tensors": {"<pack file>": {"tensor": name,
"restore": "fp16+bf16", "n_bf16": N, "n_fp16": n, "fp16": [flat idx...] OR "bf16_only": [flat idx...]}}} -- the fp16
table is the pack's list restricted to "fp16", or without "bf16_only" (the shorter one is stored). fp16-only tensors:
counts only.
"""
from __future__ import annotations

import json as _json
import os as _os

import numpy as np

TABLES_FILE = "escape_tables.json"
_CACHE: dict = {}


def fp16_list_for(path: str | None):
    """The fp16 escape list of pack file `path` from its folder's escape_tables.json, or None (no file / no entry)."""
    if not path:
        return None
    d = _os.path.abspath(_os.path.dirname(path))
    if d not in _CACHE:
        f = _os.path.join(d, TABLES_FILE)
        try:
            if _os.path.exists(f):
                with open(f) as fh:
                    data = _json.load(fh)
                if data.get("version") != 2:
                    raise EscapeTableError(f"{f}: version {data.get('version')}, this reader knows 2; "
                                           f"{_rebuild_hint(d)}")
                _CACHE[d] = data["tensors"]
            else:
                _CACHE[d] = {}
        except (OSError, ValueError, KeyError) as ex:
            raise EscapeTableError(f"{f}: unreadable ({ex}); {_rebuild_hint(d)}") from None
    e = _CACHE[d].get(_os.path.basename(path))
    if e is None or e.get("restore") != "fp16+bf16":
        return None
    if "fp16" in e:
        return ("fp16", np.asarray(e["fp16"], np.int64), int(e["n_bf16"]), int(e["n_fp16"]))
    return ("bf16_only", np.asarray(e["bf16_only"], np.int64), int(e["n_bf16"]), int(e["n_fp16"]))


class EscapeTableError(RuntimeError):
    pass


def _rebuild_hint(where: str) -> str:
    d = _os.path.dirname(where) if where and not _os.path.isdir(where) else where
    return (f"download escape_tables.json from the pack's Hugging Face repo into {d or 'the pack folder'} "
            f"(next to manifest.json)")


def select_fp16(full_idx: np.ndarray, fp16_list, where: str = "") -> np.ndarray:
    """Keep-mask over the pack's full escape list for the fp16 table given in escape_tables.json. Validation only:
    the file must match the pack (same escape count, every listed position in the pack's list), else an error."""
    kind, lst, n_bf16, n_fp16 = fp16_list
    full_idx = np.asarray(full_idx, np.int64)
    if n_bf16 != len(full_idx):
        raise EscapeTableError(f"{where}: {TABLES_FILE} says {n_bf16} escapes, the pack has {len(full_idx)} -- the file "
                               f"belongs to another pack; {_rebuild_hint(where)}")
    hit = np.isin(full_idx, lst)
    if int(hit.sum()) != len(lst):
        raise EscapeTableError(f"{where}: {TABLES_FILE} lists {len(lst) - int(hit.sum())} positions that are not "
                               f"escapes of this pack; {_rebuild_hint(where)}")
    keep = hit if kind == "fp16" else ~hit
    if int(keep.sum()) != n_fp16:
        raise EscapeTableError(f"{where}: {TABLES_FILE} fp16 table has {int(keep.sum())} entries, n_fp16 says {n_fp16}; "
                               f"{_rebuild_hint(where)}")
    return keep


def fp16_missing_error(where: str) -> EscapeTableError:
    return EscapeTableError(f"{where}: fp16 run on an fp16+bf16 pack without {TABLES_FILE} (the fp16 escape table); "
                            f"{_rebuild_hint(where)}")


def require_fp16_tables(pack_dir: str) -> None:
    """For an fp16 run: every fp16+bf16 tensor of the pack at `pack_dir` (and its component folders) must have its fp16
    table in escape_tables.json. Raises EscapeTableError otherwise. Reads the manifests only."""
    man_path = _os.path.join(pack_dir, "manifest.json")
    if not _os.path.exists(man_path):
        for c in sorted(_os.listdir(pack_dir)) if _os.path.isdir(pack_dir) else []:
            if _os.path.isfile(_os.path.join(pack_dir, c, "manifest.json")):
                require_fp16_tables(_os.path.join(pack_dir, c))
        return
    with open(man_path) as fh:
        man = _json.load(fh)
    for comp in man.get("components", []) or []:
        require_fp16_tables(_os.path.join(pack_dir, comp))
    tensors = man.get("tensors")
    if not isinstance(tensors, dict):
        return
    need = [i["pack"] for i in tensors.values() if i["restore"] == "fp16+bf16"]
    if not need:
        return
    f = _os.path.join(pack_dir, TABLES_FILE)
    have = {}
    if _os.path.exists(f):
        with open(f) as fh:
            have = _json.load(fh).get("tensors", {})
    missing = [p for p in need if p not in have]
    if missing:
        raise EscapeTableError(f"{pack_dir}: fp16 run, but {len(missing)} fp16+bf16 tensors have no fp16 escape table "
                               f"({TABLES_FILE} {'missing' if not have else 'incomplete'}); {_rebuild_hint(pack_dir)}")
