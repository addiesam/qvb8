"""ONE host (DRAM) copy of a weight for every process of this machine: the OWNER loads it, the ATTACHED ones map it.

A sub-model placed dram:shared (QV_PLACE, run/weight_source.py) keeps its pinned host buffers in /dev/shm (RAM, tmpfs:
nothing reaches a disk):
  owner     the first process that needs a weight: reads its file, cuts its share, writes the buffers to /dev/shm and
            then a small meta file (shapes, dtypes, the module's table layout). It deletes what it wrote when it exits.
  attached  every later process (the other DP server, a restarted one): finds the meta file and maps the buffers -- the
            file is never read. Its mapping stays valid after the owner exits (tmpfs frees the pages at the last unmap).
A copy whose owner died without cleaning up (killed) is removed by the next process that starts (sweep()).
Names: /dev/shm/qv_<hash of weight file + its size/mtime + the share's cut>.<buffer>  and  .meta"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
import weakref

import torch

DIR = "/dev/shm"
_MINE: set[str] = set()                 # files this process wrote (the owner's copies)
_LOCK = threading.Lock()
_SWEPT = False


def _base(path: str, cut) -> str:
    try:
        st = os.stat(path); ver = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        ver = ""
    return os.path.join(DIR, "qv_" + hashlib.sha1(f"{os.path.realpath(path)}|{ver}|{cut}".encode()).hexdigest()[:24])


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def sweep():
    """Remove copies whose owner is gone (killed before it could clean up). Once per process."""
    global _SWEPT
    if _SWEPT:
        return
    _SWEPT = True
    try:
        names = os.listdir(DIR)
    except OSError:
        return
    for n in names:
        if n.startswith("qv_owner_") and n.endswith(".list"):
            try:
                pid = int(n[len("qv_owner_"):-len(".list")])
            except ValueError:
                continue
            if not _alive(pid):
                try:
                    for m in open(os.path.join(DIR, n)).read().split() + [os.path.join(DIR, n)]:
                        try:
                            os.remove(m)
                        except OSError:
                            pass
                except OSError:
                    pass
            continue
        if n.startswith("qv_") and n.endswith(".meta"):
            p = os.path.join(DIR, n)
            try:
                owner = int(json.load(open(p)).get("owner", 0))
            except (OSError, ValueError):
                continue
            if owner and not _alive(owner):
                stem = p[:-len(".meta")]
                for m in names:
                    if os.path.join(DIR, m).startswith(stem + "."):
                        try:
                            os.remove(os.path.join(DIR, m))
                        except OSError:
                            pass


_WATCH = None


def _owner_list() -> str:
    return os.path.join(DIR, f"qv_owner_{os.getpid()}.list")


def _watch():
    """The owner's copies must go however it ends (SIGTERM from a launcher's stop, a kill): a tiny watcher process waits
    for this process to exit and removes every file listed in its owner list. Started once, at the first publish."""
    global _WATCH
    if _WATCH is not None:
        return
    import subprocess, sys
    code = ("import os,sys,time\n"
            "pid,lst=int(sys.argv[1]),sys.argv[2]\n"
            "while True:\n"
            "    try: os.kill(pid,0)\n"
            "    except ProcessLookupError: break\n"
            "    except PermissionError: pass\n"
            "    time.sleep(1)\n"
            "try: names=open(lst).read().split()\n"
            "except OSError: names=[]\n"
            "for n in names+[lst]:\n"
            "    try: os.remove(n)\n"
            "    except OSError: pass\n")
    _WATCH = subprocess.Popen([sys.executable, "-c", code, str(os.getpid()), _owner_list()],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True, close_fds=True)


def _own(fname: str):
    with _LOCK:
        _MINE.add(fname)
        with open(_owner_list(), "a") as f:
            f.write(fname + "\n")


@atexit.register
def _cleanup():
    """The owner leaves: its copies go (processes still mapping them keep working)."""
    for p in list(_MINE) + [_owner_list()]:
        try:
            os.remove(p)
        except OSError:
            pass
    _MINE.clear()


def _map(fname: str, nbytes: int, dtype, shape) -> torch.Tensor:
    """The mapped copy, page-locked for this process's card (unlocked when the returned tensor goes)."""
    base = torch.from_file(fname, shared=True, size=nbytes, dtype=torch.uint8)
    out = base.view(dtype).view(shape)
    if torch.cuda.is_available() and nbytes:
        from qv_band.swap.block_stream import _unregister
        cr = torch.cuda.cudart()
        if int(cr.cudaHostRegister(base.data_ptr(), nbytes, 0)) != 0:
            cr.cudaGetLastError()
        else:
            weakref.finalize(out, _unregister, base)
    return out


def cut_key(rows, cols) -> str:
    return json.dumps([None if rows is None else [list(map(int, r)) for r in rows],
                       None if cols is None else list(map(int, cols))])


def attach(path: str, cut):
    """(buffers {name: pinned tensor}, attrs) mapped from the owner's copy, or None when there is none (be the owner)."""
    sweep()
    meta = _base(path, cut) + ".meta"
    try:
        m = json.load(open(meta))
    except (OSError, ValueError):
        return None
    bufs = {}
    for n, (dt, shape, nbytes) in m["buffers"].items():
        if dt is None:
            bufs[n] = None
            continue
        if nbytes == 0:
            bufs[n] = torch.empty(tuple(shape), dtype=getattr(torch, dt))
            continue
        fname = _base(path, cut) + "." + n
        if not os.path.exists(fname) or os.path.getsize(fname) != nbytes:
            return None
        bufs[n] = _map(fname, nbytes, getattr(torch, dt), tuple(shape))
    return bufs, m.get("attrs", {})


def claim(path: str, cut):
    """Before loading a weight: None = this process is its owner (load it, then publish); else the attached copy (the
    owner that holds the claim is loading it now: wait for its meta). A claim whose owner died is taken over."""
    import time
    sweep()
    base = _base(path, cut)
    got = attach(path, cut)
    if got is not None:
        return got
    lock = base + ".lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
            _watch(); _own(lock)
            return None
        except FileExistsError:
            try:
                owner = int(open(lock).read() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and not _alive(owner):
                try:
                    os.remove(lock)
                except OSError:
                    pass
                continue
            for _ in range(6000):                    # the owner is loading it: its meta comes last
                got = attach(path, cut)
                if got is not None:
                    return got
                if not os.path.exists(lock):
                    break
                time.sleep(0.1)
            got = attach(path, cut)
            if got is not None:
                return got
    return None                                      # load it ourselves (own copy if another claim is stuck)


def publish(mod, path: str, cut, attrs: dict):
    """Owner: write every buffer of `mod` to /dev/shm, swap in the pinned mapped copies, then the meta file (last: an
    attached process only sees a complete copy). Another process publishing the same weight at the same moment: the
    first meta wins, both copies are equal."""
    sweep()
    _watch()
    base = _base(path, cut)
    meta_buf = {}
    for n, b in list(mod._buffers.items()):
        if b is None:
            meta_buf[n] = (None, None, 0)
            continue
        b = b.detach().to("cpu").contiguous()
        nbytes = b.numel() * b.element_size()
        meta_buf[n] = (str(b.dtype).replace("torch.", ""), list(b.shape), nbytes)
        if nbytes == 0:                         # an empty buffer: no file, rebuilt from its shape
            continue
        fname = base + "." + n
        tmp = f"{fname}.{os.getpid()}.tmp"
        t = torch.from_file(tmp, shared=True, size=nbytes, dtype=torch.uint8)
        t.view(b.dtype).view(b.shape).copy_(b)
        del t
        os.replace(tmp, fname)
        _own(fname)
        mod._buffers[n] = _map(fname, nbytes, b.dtype, b.shape)
    tmp = f"{base}.meta.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({"owner": os.getpid(), "buffers": meta_buf, "attrs": attrs}, f)
    os.replace(tmp, base + ".meta")
    _own(base + ".meta")


if __name__ == "__main__":          # python -m qv_band.swap.shared_host sweep : launchers run it after a stop
    import sys
    if sys.argv[1:] == ["sweep"]:
        sweep()
        left = [n for n in os.listdir(DIR) if n.startswith("qv_")]
        print(f"[shared_host] /dev/shm: {len(left)} shared weight files left (owners still running)")
