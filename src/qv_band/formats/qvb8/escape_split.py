"""The two escape tables of a pack: the fp16 table and the bf16 table.

A pack certified for both precisions (restore "fp16+bf16") stores ONE escape list on disk: the escapes the fp16
certificate needs, plus the positions the bf16 check added (a weight whose bf16 rounding of the band value is further
than tau from the original). The second group is useless to an fp16 decode -- the band value at those positions is
already within tau in fp16 -- and only costs work (and, for kernels that refuse any escape, the fast path).

The pack has two tables:

    fp16 table  -- every escape an fp16 / fp32 decode needs    (used by the GEMV, fp16 decode, MoE expert kernels)
    bf16 table  -- the full list, as on disk                    (used only by the paths that round the weight to bf16)

The fp16 table is written to escape_tables.json next to manifest.json when the pack is calibrated (or by this module's
CLI for an existing pack -- no re-calibration). An escape leaves the fp16 table only when the writer proves it
unnecessary there, with the same arithmetic as the certificate:

    |fp16(band) - w| <= tau   and   |band - w| <= tau          (band = fma32(id, tau_s, min_s), w = the escape value)

At load nothing is computed and nothing is chosen: the file is read and validated against the pack (mismatch = error).
An fp16 run of an fp16+bf16 pack without the file is an error; a bf16 run uses the pack's full list.

Only an escape whose value is the ORIGINAL weight may move (the verbatim esc32 pool): an fp8-coded escape holds an
approximation, and its band id is 0. Callers that know the pools pass only esc32 entries; a caller that has a merged
list (qvb8r) relies on the magnitude rule below, which excludes every fp8-coded escape: those have |w_orig| > bound and
|w_fp8 - w_orig| <= tau, so |w_fp8| > bound - tau.
"""
from __future__ import annotations

import numpy as np

FP16_MANTISSA_BITS = 10


def candidate_limit(tau: float) -> np.float32:
    """|w| at or below which an escape can be a bf16-only one. Base (fp16) escapes are |w| > tau * 2^10
    (pack.tail_boundary); 2 tau below that keeps every fp8-coded escape out, with margin."""
    return np.float32(float(tau) * (1 << FP16_MANTISSA_BITS) - 2.0 * float(tau))


def fp16_needed(vals: np.ndarray, band_lo: np.ndarray, band_hi: np.ndarray, tau: float) -> np.ndarray:
    """Boolean mask: True where the escape must stay in the fp16 table.

    The true fp32 band value lies in [band_lo, band_hi] (equal when it is known exactly). |x - w| is convex in x and
    fp16() is monotone, so checking both ends bounds every value in between."""
    w = np.asarray(vals, np.float32)
    t = np.float32(tau)
    ok = np.abs(w) <= candidate_limit(tau)
    for b in (band_lo, band_hi):
        b = np.asarray(b, np.float32)
        ok &= np.abs(b - w) <= t
        ok &= np.abs(b.astype(np.float16).astype(np.float32) - w) <= t
    return ~ok          # NaN / inf compare False -> stays needed


def ids_at(buf: np.ndarray, nbits: np.ndarray, bit_off: np.ndarray, flat: np.ndarray, P: int) -> np.ndarray:
    """The packed band ids at flat positions q*P + c (pack_bits_rows layout: rows back to back, LSB-first, a row's ids
    all nbits[q] wide). Reads 3 bytes per id: at most 14 bits + a 7-bit shift. The payload carries 8 pad bytes."""
    flat = np.asarray(flat, np.int64)
    if flat.size == 0:
        return np.zeros(0, np.int64)
    q, c = flat // P, flat % P
    nb = np.asarray(nbits, np.int64)[q]
    pos = np.asarray(bit_off, np.int64)[q] + c * nb
    byte, sh = pos >> 3, pos & 7
    b = np.asarray(buf, np.uint8)
    w = b[byte].astype(np.int64) | (b[byte + 1].astype(np.int64) << 8) | (b[byte + 2].astype(np.int64) << 16)
    return (w >> sh) & ((np.int64(1) << nb) - 1)


def split_esc32(esc32_idx, esc32_vals, *, buf, nbits, bit_off, min_s, tau_s, P: int, tau: float):
    """(keep_mask) for the verbatim pool of a QVBD pack: True = needed by fp16. The band value is exact (ids read from
    the payload, fma32 as the kernel)."""
    from .decode import fma32
    idx = np.asarray(esc32_idx, np.int64)
    if idx.size == 0:
        return np.zeros(0, bool)
    q = idx // P
    ids = ids_at(buf, nbits, bit_off, idx, P).astype(np.float32)
    band = fma32(ids, np.asarray(tau_s, np.float32)[q], np.asarray(min_s, np.float32)[q])
    return fp16_needed(esc32_vals, band, band, tau)


def split_merged_delta(esc_val, esc_delta, tau: float):
    """(keep_mask) for a qvb8r escape list: value w and delta = fp32(w - band). band' = w - delta is within
    |delta| 2^-23 + ulp(band') of the true band (one rounding in each subtraction); the check runs on that interval."""
    w = np.asarray(esc_val, np.float32)
    d = np.asarray(esc_delta, np.float32)
    if w.size == 0:
        return np.zeros(0, bool)
    band = (w - d).astype(np.float32)
    # x2: the interval ends are themselves rounded to fp32; doubling covers that with room to spare
    eps = (np.float32(2.0) * (np.abs(d) * np.float32(2.0 ** -23)
                              + np.spacing(np.abs(band)).astype(np.float32))).astype(np.float32)
    return fp16_needed(w, (band - eps).astype(np.float32), (band + eps).astype(np.float32), tau)


# ---------------------------------------------------------------------------------------------------------------------
# escape_tables.json: the fp16 table of every tensor, written next to manifest.json
#
#   {"format": "qvb8-escape-tables", "version": 2,
#    "tables": {...what each table is...},
#    "tensors": {"<pack file>": {"tensor": name, "restore": "fp16+bf16", "n_bf16": N, "n_fp16": n,
#                                "fp16": [flat idx...]  OR  "bf16_only": [flat idx...]}}}
#
# The bf16 table is the pack file's own escape list (nothing to store). The fp16 table is that list, restricted to
# "fp16" or with "bf16_only" removed -- whichever list is shorter is stored, and the key says which. An fp16-only
# tensor ("restore": "fp16") has one table: counts only, no list. The
# packer writes this file at the end of a calibration (pack_model.py, qvb8r/pack.py); for an existing pack:
#     python -m qv_band.formats.qvb8.escape_split <pack_dir>
# At load NOTHING is computed or chosen: the fp16 table is read from this file and validated against the pack (same
# escape count, the list a subset of the pack's list); a mismatch is an error. An fp16+bf16 pack without the file has
# no fp16 table: a bf16 run uses the pack's full list (the bf16 table, exact values -- correct for every path); an
# fp16 run is refused (require_fp16_tables, and the modules' forward), with the command that builds the file.
# ---------------------------------------------------------------------------------------------------------------------
import json as _json
import os as _os
import warnings as _warnings

TABLES_FILE = "escape_tables.json"
_TABLES_DOC = {
    "bf16": "the escape list stored in the pack file (all escapes); used only by decodes that round the weight to bf16",
    "fp16": "used by every fp16 / fp32 decode: the pack's list restricted to 'fp16', or without 'bf16_only' (the shorter "
            "of the two is stored). A bf16-only escape is one where the band value already holds |fp16(band) - w| <= tau "
            "and |band - w| <= tau (checked when this file was written). fp16-only tensors: one table, counts only",
}


def _tables_of_file(path: str):
    """(restore, full escape idx, fp16 escape idx) of one pack file, computed with the load-time check."""
    if path.endswith(".qvb8r"):
        from qv_band.formats.qvb8r import container as QF
        d = QF.read(path)
        cert = d.get("cert", {})
        full = np.asarray(d["esc_idx"], np.int64)
        need = np.ones(len(full), bool)
        if cert.get("restore") == "fp16+bf16" and len(full) and cert.get("tau"):
            need = split_merged_delta(d["esc_val"], d["esc_delta"], float(cert["tau"]))
        return cert.get("restore", "fp16"), full, np.sort(full[need])
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:4] == b"QVBD":
        from .pack_container import compute_fp16_table
        return compute_fp16_table(blob)
    from .pack import unpack_tensor
    full = np.asarray(unpack_tensor(blob)[8], np.int64)
    return "fp16", full, full


def write_tables(pack_dir: str, verbose: bool = True) -> str | None:
    """Write <pack_dir>/escape_tables.json (and one per component folder of a component-layout pack)."""
    man_path = _os.path.join(pack_dir, "manifest.json")
    if not _os.path.exists(man_path):
        # a multi-component pack without a root manifest (MiniMax-H3: transformer/, vae/, ...): each component folder
        wrote = [write_tables(_os.path.join(pack_dir, c), verbose) for c in sorted(_os.listdir(pack_dir))
                 if _os.path.isfile(_os.path.join(pack_dir, c, "manifest.json"))]
        return next((w for w in wrote if w), None)
    with open(man_path) as fh:
        man = _json.load(fh)
    if man.get("format") == "qvb8r" and man.get("qvb8r_from"):
        if _os.path.isfile(_os.path.join(man["qvb8r_from"], "manifest.json")):
            return copy_tables_from_qvb8(man["qvb8r_from"], pack_dir, verbose)
        if verbose:                                          # source pack gone: compute from this qvb8r pack's files
            print(f"[escape-tables] {pack_dir}: QVB8 source {man['qvb8r_from']} not found, computing from the qvb8r "
                  f"files", flush=True)
    comp_out = [write_tables(_os.path.join(pack_dir, comp), verbose) for comp in man.get("components", []) or []]
    tensors = man.get("tensors")
    if man.get("format") == "qvb8_lora":            # a LoRA pack (tensors = a count): one .qvband per packed A / B matrix
        tensors = {f[:-len(".qvband")]: {"pack": f} for f in sorted(_os.listdir(pack_dir)) if f.endswith(".qvband")}
    if not isinstance(tensors, dict):
        return next((w for w in comp_out if w), None)
    entries, nb, nf = {}, 0, 0
    for name, info in tensors.items():
        f = _os.path.join(pack_dir, info["pack"])
        if not _os.path.exists(f):
            continue
        restore, full, f16 = _tables_of_file(f)
        e = {"tensor": name, "restore": restore, "n_bf16": int(len(full)), "n_fp16": int(len(f16))}
        if restore == "fp16+bf16":
            only = np.setdiff1d(full, f16)
            if len(f16) <= len(only):
                e["fp16"] = f16.tolist()
            else:
                e["bf16_only"] = only.tolist()
        entries[info["pack"]] = e
        nb += len(full); nf += len(f16)
    if not entries:
        return next((w for w in comp_out if w), None)
    out = _os.path.join(pack_dir, TABLES_FILE)
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        _json.dump({"format": "qvb8-escape-tables", "version": 2, "tables": _TABLES_DOC, "tensors": entries}, fh)
    _os.replace(tmp, out)
    _CACHE.pop(_os.path.abspath(pack_dir), None)
    if verbose:
        print(f"[escape-tables] {out}: {len(entries)} tensors, escapes bf16 table {nb}, fp16 table {nf}", flush=True)
    return out


def copy_tables_from_qvb8(src_dir: str, out_dir: str, verbose: bool = True) -> str | None:
    """escape_tables.json of a qvb8r pack = its QVB8 source's, re-keyed to the qvb8r file names. The transcode keeps
    every escape (same flat positions, same values), so the tables are the same; copying keeps them identical and uses
    the exact QVB8 check. Each entry's n_bf16 is checked against the qvb8r file."""
    src_tables = _os.path.join(src_dir, TABLES_FILE)
    if not _os.path.exists(src_tables):
        write_tables(src_dir, verbose)                      # the source is a QVB8 pack: build its file first
    if not _os.path.exists(src_tables):
        return None
    with open(src_tables) as fh:
        by_tensor = {e["tensor"]: e for e in _json.load(fh)["tensors"].values()}
    with open(_os.path.join(out_dir, "manifest.json")) as fh:
        man = _json.load(fh)
    entries = {}
    for name, info in (man.get("tensors") or {}).items():
        e = by_tensor.get(name)
        if e is None:
            raise EscapeTableError(f"{src_tables}: no entry for {name}; rebuild it")
        f = _os.path.join(out_dir, info["pack"])
        if f.endswith(".qvb8r"):
            from qv_band.formats.qvb8r import container as QF
            n = len(QF.read(f)["esc_idx"])
            if n != e["n_bf16"]:
                raise EscapeTableError(f"{f}: {n} escapes, its QVB8 source {e['n_bf16']} -- not a transcode of {src_dir}")
        entries[info["pack"]] = dict(e)
    out = _os.path.join(out_dir, TABLES_FILE)
    with open(out + ".tmp", "w") as fh:
        _json.dump({"format": "qvb8-escape-tables", "version": 2, "tables": _TABLES_DOC, "tensors": entries,
                    "copied_from": src_tables}, fh)
    _os.replace(out + ".tmp", out)
    _CACHE.pop(_os.path.abspath(out_dir), None)
    if verbose:
        print(f"[escape-tables] {out}: {len(entries)} tensors, copied from {src_tables}", flush=True)
    return out


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
    return f"build it (no re-calibration): python -m qv_band.formats.qvb8.escape_split {d or '<pack_dir>'}"


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


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m qv_band.formats.qvb8.escape_split <pack_dir> [<pack_dir> ...]")
    for p in sys.argv[1:]:
        if write_tables(p) is None:
            print(f"[escape-tables] {p}: no QVB8 tensors (no manifest.json, or not a linear pack)", flush=True)
