"""THE weight loader: one model-independent way to load every packed weight of every model (LLM, DiT, text encoder,
VAE, audio, ...), whole or as a Tensor Parallel share, onto a card, into host RAM, or left in its file.

  load_weight(src, rows=, cols=, device=, bias=)   one weight -> its module (the bare module, no parallel wrapper)
  load_parallel(fn, items)                         many weights at once: one weight per worker thread, results in order
WHERE a weight goes is the caller's placement (single card, PP stage card, TP share, streamed tier: run/weight_source.py);
HOW it is loaded is always this:
  * .qvb8r file / QVBandLinearV8R with a file: cut IN PLACE from the rANS stream (formats/qvb8r/cut.py: decoder
    states at the cut, no re-encoding). Any output rows x any input-column range (column start a multiple of 64,
    width a multiple of 128: every kernel's P). Outside that: the re-encode fallback (formats/qvb8r/tp_slice.py).
  * .qvband / .qvb8 file or QVBandLinearV8: fixed 8-bit ids, a plain slice (formats/qvb8/tp.py).
  rows: [(r0, r1), ...] output rows (column-parallel: q/k/v, FFN up); cols: (c0, c1) input columns (row-parallel:
  o_proj / to_out, FFN down). The bias stays with the caller (a row-parallel bias is added after the all-reduce).
  None for a module type it does not know (the caller keeps its own path, e.g. dense nn.Linear).

Pre-cut TP share files (any model, any TP path -- they all cut through load_weight):
  QV_TP_SHARD_WRITE=DIR   every share this process cuts is also saved to DIR/<file>.<rows|cols>.pt (start the TP server
                          once with it: each card writes its own shares)
  QV_TP_SHARD_DIR=DIR     a share found there is read as it is (no cut); a share file whose source file changed (size /
                          mtime) stops the load with the reason -- rewrite the shares (nothing is rebuilt silently)

load_parallel(fn, items) runs the loads on QV_LOAD_WORKERS threads (default: min(8, CPUs)). File reads, the
cut's CPU walk (GIL released) and the card copies of different linears overlap; results come back in order, so
the model is assembled exactly as a serial load would."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import torch


def _rows_list(rows):
    if rows is None:
        return None
    if isinstance(rows, tuple) and len(rows) == 2 and not isinstance(rows[0], (tuple, list)):
        return [rows]
    return [tuple(r) for r in rows]


def _share_name(path, rows, cols) -> str:
    tag = ("r" + "_".join(f"{a}-{b}" for a, b in rows)) if rows is not None else f"c{int(cols[0])}-{int(cols[1])}"
    return f"{os.path.basename(path)}.{tag}.pt"


def _stamp(path):
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)


def load_weight(src, rows=None, cols=None, device="cuda", bias=None):
    """src: a weight file (any registered format) or a loaded module. Whole (rows = cols = None) or a share."""
    rows = _rows_list(rows)
    share = isinstance(src, str) and (rows is not None or cols is not None) and str(device) not in ("ref", "disk")
    rd, wr = os.environ.get("QV_TP_SHARD_DIR"), os.environ.get("QV_TP_SHARD_WRITE")
    if share and rd:
        f = os.path.join(rd, _share_name(src, rows, cols))
        if os.path.isfile(f):
            got = torch.load(f, map_location=device, weights_only=False, mmap=True)   # mapped, not copied into RAM
            if tuple(got["stamp"]) != _stamp(src):
                raise RuntimeError(f"weight_load: share {f} was cut from another version of {src}: rewrite the shares "
                                   f"(QV_TP_SHARD_WRITE)")
            mod = got["module"]
            try:                                    # its pages are on the card now: out of the page cache (the RAM cap counts it)
                fd = os.open(f, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except (OSError, AttributeError):
                pass
            if bias is not None and getattr(mod, "bias", None) is None and hasattr(mod, "bias"):
                mod.bias = torch.nn.Parameter(bias.detach().to(device), requires_grad=False)
            return mod
    mod = _load_weight(src, rows, cols, device, bias)
    if share and wr and mod is not None:
        os.makedirs(wr, exist_ok=True)
        f = os.path.join(wr, _share_name(src, rows, cols))
        tmp = f + f".tmp{os.getpid()}"
        torch.save({"stamp": _stamp(src), "module": mod}, tmp)
        os.replace(tmp, f)
    return mod


def _load_weight(src, rows, cols, device, bias):
    from qv_band.formats.qvb8r.module import QVBandLinearV8R
    if isinstance(src, str):
        if rows is None and cols is None:                           # whole: the file's own reader
            from qv_band.common.formats import get_format
            ext = os.path.splitext(src)[1].lstrip(".")
            return get_format(ext).linear.from_packfile(src, bias=bias, device=device)
        if src.endswith(".qvb8r"):
            from qv_band.formats.qvb8r.cut import can_cut
            hdr = QVBandLinearV8R.from_packfile(src, device="ref")          # header only
            Q, P = hdr.out_features, hdr.in_features
            if (rows == [(0, Q)] and cols is None) or (rows is None and tuple(cols) == (0, P)):
                return QVBandLinearV8R.from_packfile(src, device=device, bias=bias)
            if can_cut(P, cols) is None:
                return QVBandLinearV8R.from_packfile(src, device=device, rows=rows, cols=cols)
            src = hdr                                               # fallback below (re-encode)
        else:
            from qv_band.formats.qvb8r.to_qvb8 import load_linear_any
            src = load_linear_any(src, device)
    if isinstance(src, QVBandLinearV8R):
        from qv_band.formats.qvb8r.tp_slice import slice_qvb8r
        local, why = slice_qvb8r(src, rows=rows, cols=None if cols is None else tuple(cols), device=device)
        if local is None:
            raise RuntimeError(f"weight_load: cannot cut {getattr(src, 'path', '?')}: {why}")
        return local
    if hasattr(src, "ids") and hasattr(src, "nbits"):               # QVB8
        from qv_band.formats.qvb8.tp import slice_cols, slice_rows
        if cols is not None and rows is not None:
            raise ValueError("weight_load: QVB8 share is rows or columns, not both")
        if cols is not None:
            if tuple(cols) == (0, src.in_features):
                return src.to(device)
            return slice_cols(src, int(cols[0]), int(cols[1]), device)
        if rows is None or rows == [(0, src.out_features)]:
            return src.to(device)
        return slice_rows(src, rows, device)
    return None


_WARM = set()


def warm(devices):
    """Build / load the kernel extensions once per card before threads use them (no compile race)."""
    from qv_band.formats.qvb8r.module import _load_ext
    for d in devices:
        d = torch.device(d)
        if d.type == "cuda" and d not in _WARM:
            with torch.cuda.device(d):
                _load_ext()
            _WARM.add(d)


def _avail_gb() -> float:
    """Host RAM available now (MemAvailable), GB; a cgroup limit (the launchers' RAM cap) when it is lower."""
    gb = 1e9
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                gb = int(line.split()[1]) / 1e6
    except OSError:
        pass
    try:                                   # cgroup v2: this process's own group (a systemd scope with MemoryMax)
        rel = open("/proc/self/cgroup").read().strip().split("::", 1)[1]
        d = "/sys/fs/cgroup" + rel
        mx = open(os.path.join(d, "memory.max")).read().strip()
        if mx != "max":
            gb = min(gb, (int(mx) - int(open(os.path.join(d, "memory.current")).read())) / 1e9)
    except (OSError, ValueError, IndexError):
        pass
    return gb


def workers() -> int:
    """QV_LOAD_WORKERS, default min(8, CPUs), and never more than ~2 GB of free host RAM per thread (a thread holds one
    table file and its share while it works)."""
    n = int(os.environ.get("QV_LOAD_WORKERS", str(min(8, os.cpu_count() or 1))))
    return max(1, min(n, int(_avail_gb() // 2)))


def on_card(device):
    """Context: `device` current while a weight for it is built (a kernel launched by its loader runs on ITS card, not
    on whatever card the worker thread had); a no-op for host / file targets."""
    import contextlib
    d = None if device in (None, "ref", "disk") else torch.device(device)
    return torch.cuda.device(d) if d is not None and d.type == "cuda" else contextlib.nullcontext()


def load_parallel(fn, items, n=None):
    """[fn(x) for x in items] on n threads, in order. Each thread keeps the caller's current card and grad mode."""
    items = list(items)
    n = n or workers()
    if n <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    dev = torch.cuda.current_device() if torch.cuda.is_available() else None
    grad = torch.is_grad_enabled()

    def call(x):
        if dev is not None:
            torch.cuda.set_device(dev)
        with torch.set_grad_enabled(grad):
            return fn(x)
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(call, items))
