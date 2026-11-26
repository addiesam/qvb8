"""Saved prompt prefixes in ONE host-RAM pool per machine, shared by every server process on it (--prefix-store shared).

Why: two one-card servers on the same box (V100 card 0 and card 1) each kept their own prefixes in pinned host RAM, so a
long document used on both cards was held twice (~9.5 GB at 94k tokens). Here the bytes live once in a POSIX shared
memory file (/dev/shm: RAM only, nothing goes to disk), registered with CUDA in each process so the GPU copies straight
from it, and any server restores a prefix another one computed.

Layout (all files in /dev/shm, names from `tag`: the pack, its tau, the KV cache, the chunk grid -> servers that could
produce different bytes never share):
  <tag>.arena   the pool. Created by the launcher (fallocate) so its pages count against the launcher's memory, not one
                server's RAM cap; created here if missing.
  <tag>.index   the table of contents (pickle), rewritten under the lock; small.
  <tag>.lock    flock() for the index.

A saved prefix (an "entry") = its token ids + a layout (the engine's slot state with every tensor replaced by a TensorRef)
+ blobs. A blob is one contiguous range of the pool holding many tensors; the KV cache's packed segments are immutable, so
a later checkpoint of the same prompt re-uses the blobs of the earlier one (refcounted) instead of copying them again.

Readers get VIEWS of the pool (no copy) and hold a lease on every blob they use until their request leaves (release());
eviction only frees blobs that no entry and no lease holds. Leases and half-written entries of a process that died are
dropped by the next process that takes the lock.
"""
from __future__ import annotations

import fcntl
import mmap
import os
import pickle
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

SHM_DIR = "/dev/shm"
ALIGN = 256
_EPOCHS: dict = {}


def _start_time(pid: int):
    """The kernel's start time of process `pid` (clock ticks since boot, /proc/<pid>/stat field 22), or None when no
    such process exists. pid + start time names ONE process for the life of this boot: a pid handed to a new process
    has another start time."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            return int(f.read().rsplit(b")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _epoch() -> str:
    """This process as the index names it: "<pid>:<start time>". Looked up per pid, so a forked child is itself and
    not its parent."""
    pid = os.getpid()
    e = _EPOCHS.get(pid)
    if e is None:
        e = _EPOCHS[pid] = f"{pid}:{_start_time(pid)}"
    return e


@dataclass(frozen=True)
class TensorRef:
    bid: int
    off: int              # bytes from the start of the blob
    shape: tuple
    dtype: torch.dtype
    small: bool           # min_s / tau_s of a KV segment: copied to the card in one go on a restore


def _align(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def _alive(epoch: str) -> bool:
    """Is the process that wrote `epoch` still running? Exact: the pid must exist AND have the start time recorded.
    No timeout on purpose: a lease says "this process reads these bytes through a view", and a live process that is
    merely slow must keep them. Transfers to other machines have their own expiring leases (XFER_LEASE_S)."""
    pid, _, start = epoch.partition(":")
    now = _start_time(int(pid))
    return now is not None and str(now) == start


def pack_signature(pack: str) -> tuple:
    """What makes two packs interchangeable for saved prefixes, read from manifest.json only (fast): the source model
    (hf_model folder name), the format FAMILY (first 4 characters: qvb8 and qvb8r are the same numbers, the earlier format is not;
    the older qvb8 packs say "qvband") and tau. Not the pack's path: the same pack sits in different folders on
    different machines."""
    import json
    try:
        with open(os.path.join(pack, "manifest.json")) as f:
            m = json.load(f)
    except (OSError, ValueError):
        return ("path", os.path.realpath(pack))
    fmt = str(m.get("format", ""))
    fam = "qvb8" if fmt == "qvband" else fmt[:4]
    return (os.path.basename(str(m.get("hf_model", "")).rstrip("/")), fam, m.get("tau"))


def config_tag(*parts) -> str:
    """The pool's name from everything that decides a saved slot's bytes (packs, taus, chunk grid, dtype, cards, drafter):
    servers that differ in any of them use different pools."""
    import hashlib
    raw = repr(tuple(os.path.realpath(p) if isinstance(p, str) and os.path.isdir(p) else p for p in parts))
    return "qv_prefix_" + hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()


class SharedPrefixStore:
    def __init__(self, tag: str, size_bytes: int, config: str = ""):
        """tag: the pool's file name (the launcher's choice, shared by the servers it starts on this machine). config:
        a fingerprint of everything that decides a saved slot's bytes (config_tag); the first server writes it into
        the pool and a server with a different one is refused (ValueError) instead of restoring foreign state."""
        self.tag, self.config = tag, config
        self._tlock, self._tdepth = threading.RLock(), 0
        self.arena_path = os.path.join(SHM_DIR, tag + ".arena")
        self.index_path = os.path.join(SHM_DIR, tag + ".index")
        self.lock_path = os.path.join(SHM_DIR, tag + ".lock")
        old_mask = os.umask(0o077)
        try:
            self._lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            with self._locked():
                if not os.path.exists(self.arena_path):
                    print(f"[prefix] shared pool {self.arena_path} not found: creating {size_bytes / 2**30:.1f} GiB "
                          f"(its RAM counts against THIS server; the launcher normally creates it)",
                          file=sys.__stdout__, flush=True)
                    fd = os.open(self.arena_path, os.O_RDWR | os.O_CREAT, 0o600)
                    os.posix_fallocate(fd, 0, size_bytes)
                    os.close(fd)
        finally:
            os.umask(old_mask)
        fd = os.open(self.arena_path, os.O_RDWR)
        self.size = os.fstat(fd).st_size
        self._mm = mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        self.base = torch.frombuffer(self._mm, dtype=torch.uint8)
        self.pinned = False
        if torch.cuda.is_available():
            # cudaHostRegisterPortable (1): every CUDA context of this process may DMA from it
            err = torch.cuda.cudart().cudaHostRegister(self.base.data_ptr(), self.size, 1)
            self.pinned = int(err) == 0
            if not self.pinned:
                torch.cuda.cudart().cudaGetLastError()       # clear it: a sticky error would fail the next CUDA call
                print(f"[prefix] shared pool: cudaHostRegister failed ({err}); restores copy through pageable memory",
                      file=sys.__stdout__, flush=True)
        self._own = weakref.WeakKeyDictionary()      # live KV segment object -> (pool uuid, bid, its skeleton)
        self._leases = {}                            # bid -> count held by this process
        self._tok_cache = {}                         # entry key -> token list
        with self._locked():
            self._save(self._load())                 # claims an empty pool / refuses another configuration now
        print(f"[prefix] shared pool {self.arena_path}: {self.size / 2**30:.1f} GiB, "
              f"{'pinned' if self.pinned else 'NOT pinned'}", file=sys.__stdout__, flush=True)

    def close(self) -> None:
        """Unregister and unmap (the next mapping at this address must not find it still registered with CUDA)."""
        if getattr(self, "pinned", False):
            torch.cuda.cudart().cudaHostUnregister(self.base.data_ptr())
            self.pinned = False
        self.base = None
        mm, self._mm = getattr(self, "_mm", None), None
        if mm is not None:
            try:
                mm.close()
            except BufferError:                  # views still exported: the mapping goes with them
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:                        # noqa: BLE001 - interpreter shutdown
            pass

    # ---- index -----------------------------------------------------------------------------------------------------
    @contextmanager
    def _locked(self):
        """Other processes: flock. Other threads of THIS process: a thread lock first -- flock does not exclude threads
        that share one open file (4 piece requests at once from the prefix store wrote the index together:
        FileNotFoundError on the rename, and a lost update would drop leases or entries)."""
        with self._tlock:
            if self._tdepth == 0:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            self._tdepth += 1
            try:
                yield
            finally:
                self._tdepth -= 1
                if self._tdepth == 0:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _load(self) -> dict:
        try:
            with open(self.index_path, "rb") as f:
                idx = pickle.load(f)
        except (FileNotFoundError, EOFError):
            idx = None
        if not idx or idx.get("size") != self.size:
            import uuid
            idx = {"size": self.size, "blobs": {}, "entries": {}, "next_bid": 1, "next_key": 1, "leases": {},
                   "config": self.config, "uuid": uuid.uuid4().hex}
        if "uuid" not in idx:
            import uuid
            idx["uuid"] = uuid.uuid4().hex
        if idx["uuid"] != getattr(self, "_seen_uuid", None):
            # another pool under this name (its files were removed and created again while this process ran): entry
            # keys start at 1 again, so the tokens remembered for a key belong to the old pool
            self._seen_uuid = idx["uuid"]
            getattr(self, "_tok_cache", {}).clear()
        if idx.get("fmt") != INDEX_FORMAT:
            self._reap(idx)
            if idx["entries"] and any(idx["leases"].values()):
                raise ValueError(f"shared prefix pool {self.arena_path} was written by another version of this code "
                                 f"and is in use: stop the servers using it first")
            if idx["entries"]:
                print(f"[prefix] shared pool {self.arena_path}: written by another version of this code -- emptied",
                      file=sys.__stdout__, flush=True)
            idx.update(blobs={}, entries={}, leases={}, lease_exp={}, fmt=INDEX_FORMAT)
        if idx.get("config") != self.config:
            if not idx["entries"] and not idx["leases"]:
                idx["config"] = self.config           # empty pool: the new configuration takes it over
            else:
                raise ValueError(f"shared prefix pool {self.arena_path} holds prefixes of another configuration "
                                 f"({idx.get('config')} vs this server's {self.config}): use another "
                                 f"--prefix-shared-name, or stop the servers using it")
        self._reap(idx)
        return idx

    def _save(self, idx: dict) -> None:
        tmp = self.index_path + f".{os.getpid()}.{threading.get_ident()}"
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
            pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.index_path)

    @staticmethod
    def _reap(idx: dict) -> None:
        """Drop the leases and half-written blobs of processes that are gone, and transfer leases past their expiry."""
        exp = idx.setdefault("lease_exp", {})
        now = time.time()
        for ep in [ep for ep in idx["leases"]
                   if (exp.get(ep, 0.0) < now if ep.startswith("xfer:") else not _alive(ep))]:
            del idx["leases"][ep]
            exp.pop(ep, None)
        for bid in [b for b, v in idx["blobs"].items() if v.get("writing") and not _alive(v["writing"])]:
            del idx["blobs"][bid]
        held = SharedPrefixStore._held(idx)
        for bid in [b for b, v in idx["blobs"].items()                  # orphans: no entry, no lease, not being written
                    if v["refs"] <= 0 and not v.get("writing") and b not in held]:
            del idx["blobs"][bid]

    @staticmethod
    def _held(idx: dict) -> set:
        return {b for le in idx["leases"].values() for b, c in le.items() if c > 0}

    def _alloc(self, idx: dict, nbytes: int, protect: set):
        """First fit in the pool; evicts least recently used entries (branch / grid checkpoints first) until it fits.
        -> offset or None."""
        def fit(blobs):
            pos = 0
            for a, b in sorted((v["off"], v["off"] + v["size"]) for v in blobs):
                if a - pos >= nbytes:
                    return pos
                pos = max(pos, _align(b))
            return pos if self.size - pos >= nbytes else None

        pos = fit(idx["blobs"].values())
        if pos is not None:
            return pos
        held = self._held(idx) | protect
        stay = set(held) | {b for b, v in idx["blobs"].items() if v.get("writing")}
        for e in idx["entries"].values():
            if any(b in held for b in e["blobs"]):
                stay.update(e["blobs"])
        if nbytes > self.size or fit([v for b, v in idx["blobs"].items() if b in stay]) is None:
            return None
        while True:
            pos = fit(idx["blobs"].values())
            if pos is not None:
                return pos
            if not self._evict_one(idx, protect):
                return None

    def _evict_one(self, idx: dict, protect: set) -> bool:
        held = self._held(idx) | protect
        # a prefix a backup store already holds (--prefix-backup: the DRAM PC) goes first: losing it here costs a pull,
        # not a prefill; then branch / grid checkpoints before prompt-end snapshots; then least recently used
        ents = sorted(idx["entries"].items(),
                      key=lambda kv: (not kv[1].get("backed"), not kv[1]["branch"], kv[1]["used"]))
        for key, e in ents:
            if any(b in held for b in e["blobs"]):
                continue                              # in use by a running request somewhere
            self._unref(idx, key)
            return True
        return False

    def _unref(self, idx: dict, key) -> None:
        e = idx["entries"].pop(key)
        for b in e["blobs"]:
            bl = idx["blobs"].get(b)
            if bl is not None:
                bl["refs"] -= 1
                if bl["refs"] <= 0 and b not in self._held(idx):
                    del idx["blobs"][b]
        self._tok_cache.pop(key, None)

    # ---- views -----------------------------------------------------------------------------------------------------
    def _view(self, blob_off: int, ref: TensorRef) -> torch.Tensor:
        n = 1
        for s in ref.shape:
            n *= s
        nb = n * torch.empty((), dtype=ref.dtype).element_size()
        a = blob_off + ref.off
        return self.base[a:a + nb].view(ref.dtype).view(ref.shape)

    # ---- save ------------------------------------------------------------------------------------------------------
    def put(self, tokens: list, state, drafter, branch: bool, dtype=None) -> bool:
        """Copy one saved slot into the pool (once per machine: a prefix another server already saved is skipped).
        Segments already in the pool (an earlier checkpoint of this prompt, or a restored prefix) are referenced, not
        copied. False when it does not fit."""
        from qv_band.kv.codec import BandedKV
        from qv_band.decode.spec_decode import PackedState
        n = len(tokens)
        with self._locked():
            idx = self._load()
            for key, e in idx["entries"].items():
                if e["n"] == n and self._tokens(idx, key) == tokens:
                    e["used"] = time.time()
                    self._save(idx)
                    return True
            live = {b for b in idx["blobs"]}
            pool_id = idx["uuid"]
        # 1) walk the state: a skeleton with TensorRefs; new tensors listed for the new blob
        new_big, new_small, reused = [], [], set()
        state_dev = {}

        def tref(t, small):
            t = t.detach()
            lst = new_small if small else new_big
            lst.append(t)
            return ("new", small, len(lst) - 1, tuple(t.shape), t.dtype)

        def bk_ref(bk):
            own = self._own.get(bk)
            if own is not None and own[0] == pool_id and own[1] in live:    # same pool (an index reset renumbers blobs)
                reused.add(own[1])
                return own[2]
            return BandedKV(packed=tref(bk.packed, False), min_s=tref(bk.min_s, True), tau_s=tref(bk.tau_s, True),
                            bits=bk.bits, orig_shape=bk.orig_shape, orig_dtype=bk.orig_dtype, tau=bk.tau)

        def walk(o):
            if isinstance(o, BandedKV):
                return bk_ref(o)
            if isinstance(o, PackedState):
                p = PackedState.__new__(PackedState)
                p.__dict__.update(o.__dict__)
                p.heads = [walk(h) for h in o.heads]
                return p
            if isinstance(o, torch.Tensor):
                return tref(o, False)
            if isinstance(o, dict):
                return {k: walk(v) for k, v in o.items()}
            if isinstance(o, list):
                return [walk(v) for v in o]
            if isinstance(o, tuple):
                return tuple(walk(v) for v in o)
            if isinstance(o, torch.device):
                state_dev[str(o)] = o
                return o
            return o

        skel = walk({"state": state, "drafter": drafter})
        tok = torch.tensor(tokens, dtype=torch.int32)
        new_big.append(tok)
        tok_i = len(new_big) - 1
        small_bytes = sum(_align(t.numel() * t.element_size()) for t in new_small)
        big_bytes = sum(_align(t.numel() * t.element_size()) for t in new_big)
        total = small_bytes + big_bytes
        # 2) reserve the blob (under the lock), copy outside it, then publish
        with self._locked():
            idx = self._load()
            if any(b not in idx["blobs"] for b in reused):
                return False                        # a re-used blob was evicted meanwhile: skip this save
            off = self._alloc(idx, total, protect=reused)
            if off is None:
                self._save(idx)
                print(f"[prefix] shared pool full: {total / 2**20:.0f} MB checkpoint not kept", file=sys.__stdout__,
                      flush=True)
                return False
            bid = idx["next_bid"]
            idx["next_bid"] += 1
            idx["blobs"][bid] = {"off": off, "size": total, "refs": 0, "writing": _epoch(), "small": small_bytes}
            for b in reused:                         # keep them until the entry holds them
                idx["leases"].setdefault(_epoch(), {})
                idx["leases"][_epoch()][b] = idx["leases"][_epoch()].get(b, 0) + 1
            self._save(idx)
        refs_small, refs_big = [], []
        pos = 0
        for t in new_small:
            refs_small.append(TensorRef(bid, pos, tuple(t.shape), t.dtype, True))
            pos += _align(t.numel() * t.element_size())
        for t in new_big:
            refs_big.append(TensorRef(bid, pos, tuple(t.shape), t.dtype, False))
            pos += _align(t.numel() * t.element_size())
        try:
            for t, r in zip(new_small + new_big, refs_small + refs_big):
                if t.numel():
                    self._view(off, r).copy_(t.reshape(r.shape))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except BaseException:
            # the copy failed (e.g. a CUDA error): give the reserved range and the leases on the re-used blobs back --
            # this process stays alive, so nobody would ever reap them (the range was lost and the re-used blobs could
            # never be evicted again)
            with self._locked():
                idx = self._load()
                idx["blobs"].pop(bid, None)
                self._unlease(idx, reused)
                self._save(idx)
            raise

        def fix(o):                                 # ("new", small, i, ...) -> TensorRef
            if isinstance(o, tuple) and len(o) == 5 and o[0] == "new":
                return (refs_small if o[1] else refs_big)[o[2]]
            if isinstance(o, BandedKV):
                return BandedKV(packed=fix(o.packed), min_s=fix(o.min_s), tau_s=fix(o.tau_s), bits=o.bits,
                                orig_shape=o.orig_shape, orig_dtype=o.orig_dtype, tau=o.tau)
            if isinstance(o, PackedState):
                o.heads = [fix(h) for h in o.heads]
                return o
            if isinstance(o, dict):
                return {k: fix(v) for k, v in o.items()}
            if isinstance(o, list):
                return [fix(v) for v in o]
            if isinstance(o, tuple):
                return tuple(fix(v) for v in o)
            return o

        skel = fix(skel)
        # remember which live segments now have a copy in the pool: the next checkpoint of this prompt re-uses it
        self._remember(state, skel["state"], pool_id)
        layout = pickle.dumps(skel, protocol=pickle.HIGHEST_PROTOCOL)
        with self._locked():
            idx = self._load()
            bl = idx["blobs"].get(bid)
            le = idx["leases"].get(_epoch(), {})
            for b in reused:
                le[b] = le.get(b, 0) - 1
                if le[b] <= 0:
                    le.pop(b, None)
            if bl is None:
                self._save(idx)
                return False
            if any(e["n"] == n and self._tokens(idx, k) == tokens for k, e in idx["entries"].items()):
                del idx["blobs"][bid]                # the other card saved the same prefix meanwhile: keep one
                self._save(idx)
                return True
            loff = self._alloc(idx, _align(len(layout)), protect=reused | {bid})
            if loff is None:
                del idx["blobs"][bid]
                self._save(idx)
                return False
            lbid = idx["next_bid"]
            idx["next_bid"] += 1
            idx["blobs"][lbid] = {"off": loff, "size": _align(len(layout)), "refs": 0}
            self.base[loff:loff + len(layout)].copy_(torch.frombuffer(bytearray(layout), dtype=torch.uint8))
            bl.pop("writing", None)
            blobs = sorted(reused | {bid, lbid})
            for b in blobs:
                idx["blobs"][b]["refs"] += 1
            key = idx["next_key"]
            idx["next_key"] += 1
            idx["entries"][key] = {"n": n, "tokens": refs_big[tok_i], "layout": (lbid, len(layout)), "blobs": blobs,
                                   "branch": branch, "used": time.time(), "by": _epoch(),
                                   "dtype": _dt_name(dtype) if dtype is not None else None}
            self._save(idx)
        self._tok_cache[key] = list(tokens)
        return True

    def _remember(self, live_state, skel_state, pool_id: str) -> None:
        from qv_band.kv.codec import BandedKV

        def pair(a, b):
            if isinstance(a, BandedKV) and isinstance(b, BandedKV):
                if isinstance(b.packed, TensorRef):
                    self._own[a] = (pool_id, b.packed.bid, b)
                return
            if isinstance(a, dict) and isinstance(b, dict):
                for k in a:
                    if k in b:
                        pair(a[k], b[k])
            elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
                for x, y in zip(a, b):
                    pair(x, y)

        pair(live_state, skel_state)

    # ---- find / restore --------------------------------------------------------------------------------------------
    def _tokens(self, idx: dict, key) -> list:
        t = self._tok_cache.get(key)
        if t is None:
            e = idx["entries"][key]
            b = idx["blobs"][e["tokens"].bid]
            t = self._view(b["off"], e["tokens"]).tolist()
            self._tok_cache[key] = t
        return t

    def entries(self) -> list:
        """[{"key", "tokens", "branch"}, ...] most recently used first."""
        with self._locked():
            idx = self._load()
            out = [{"key": k, "tokens": self._tokens(idx, k), "branch": e["branch"]}
                   for k, e in sorted(idx["entries"].items(), key=lambda kv: -kv[1]["used"])]
            live = {e["key"] for e in out}           # under the lock: state polls and the scheduler both come here
            for k in [k for k in self._tok_cache if k not in live]:
                self._tok_cache.pop(k, None)
        return out

    def get(self, key, dev, dtype=None):
        """-> (state, drafter, leased blob ids) with every tensor a VIEW of the pool (KV segment scales copied to `dev`),
        leased by this process until release(); None if the entry is gone."""
        from qv_band.kv.codec import BandedKV
        from qv_band.decode.spec_decode import PackedState
        with self._locked():
            idx = self._load()
            e = idx["entries"].get(key)
            if e is None:
                return None
            e["used"] = time.time()
            le = idx["leases"].setdefault(_epoch(), {})
            for b in e["blobs"]:
                le[b] = le.get(b, 0) + 1
            blobs = {b: dict(idx["blobs"][b]) for b in e["blobs"]}
            lbid, ln = e["layout"]
            pool_id = idx["uuid"]
            # saved by a machine running another model dtype (fp16 the dev GPU / bf16 V100): its dense 16-bit tensors (KV
            # residual, conv state, drafter context) are cast to ours below; qvb8 ids and fp32 states are dtype-free
            cast = (dtype if dtype is not None and e.get("dtype") and e["dtype"] != _dt_name(dtype) else None)
            self._save(idx)
        for b in e["blobs"]:
            self._leases[b] = self._leases.get(b, 0) + 1
        small_dev = {}
        for b, bl in blobs.items():                 # the KV segments' scales: one copy per blob to the card
            if bl.get("small"):
                small_dev[b] = self.base[bl["off"]:bl["off"] + bl["small"]].to(dev, non_blocking=self.pinned)

        def view(r: TensorRef):
            if cast is not None and not r.small and r.dtype in (torch.float16, torch.bfloat16) and r.dtype != cast:
                v = self._view(blobs[r.bid]["off"], r)
                if cast == torch.float16:
                    return v.float().clamp_(-65504.0, 65504.0).to(cast)     # bf16 range > fp16's
                return v.to(cast)
            if r.small and r.bid in small_dev:
                n = 1
                for s in r.shape:
                    n *= s
                nb = n * torch.empty((), dtype=r.dtype).element_size()
                return small_dev[r.bid][r.off:r.off + nb].view(r.dtype).view(r.shape)
            return self._view(blobs[r.bid]["off"], r)

        def walk(o):
            if isinstance(o, TensorRef):
                return view(o)
            if isinstance(o, BandedKV):
                od = cast if cast is not None and o.orig_dtype in (torch.float16, torch.bfloat16) else o.orig_dtype
                return BandedKV(packed=walk(o.packed), min_s=walk(o.min_s), tau_s=walk(o.tau_s), bits=o.bits,
                                orig_shape=o.orig_shape, orig_dtype=od, tau=o.tau)
            if isinstance(o, PackedState):
                p = PackedState.__new__(PackedState)
                p.__dict__.update(o.__dict__)
                p.heads = [walk(h) for h in o.heads]
                if cast is not None and p.dtype in (torch.float16, torch.bfloat16):
                    p.dtype = cast
                return p
            if isinstance(o, dict):
                return {k: walk(v) for k, v in o.items()}
            if isinstance(o, list):
                return [walk(v) for v in o]
            if isinstance(o, tuple):
                return tuple(walk(v) for v in o)
            return o

        lo = blobs[lbid]["off"]
        skel = pickle.loads(self._mm[lo:lo + ln])
        out = walk(skel)
        self._remember(out["state"], skel["state"], pool_id)  # a later checkpoint of the restored request re-uses them
        return out["state"], out["drafter"], list(e["blobs"])

    def release(self, bids) -> None:
        """A request that restored from the pool is gone: its blobs may be evicted again."""
        if not bids:
            return
        with self._locked():
            idx = self._load()
            le = idx["leases"].get(_epoch(), {})
            for b in bids:
                le[b] = le.get(b, 0) - 1
                if le[b] <= 0:
                    le.pop(b, None)
                    bl = idx["blobs"].get(b)
                    if bl is not None and bl["refs"] <= 0 and b not in self._held(idx):
                        del idx["blobs"][b]
                self._leases[b] = self._leases.get(b, 0) - 1
                if self._leases[b] <= 0:
                    self._leases.pop(b, None)
            self._save(idx)

    def adopt(self, live_bks, already=()) -> tuple:
        """The running request's own KV segments that the pool now holds (a checkpoint just copied them): -> ([(segment,
        pool view of its packed bytes)], newly leased blob ids). The caller points each segment at the view and frees
        its private copy, so a long request's older KV sits in host RAM once (, 166k tokens on one V100:
        12.1 GB private pinned + the pool's copy). `already`: blob ids this request leased before (not leased again)."""
        with self._locked():
            idx = self._load()
            pid = idx["uuid"]
            out, bids = [], set()
            for bk in live_bks:
                own = self._own.get(bk)
                if own is None or own[0] != pid or own[1] not in idx["blobs"]:
                    continue
                r = own[2].packed
                if not isinstance(r, TensorRef):
                    continue
                out.append((bk, self._view(idx["blobs"][r.bid]["off"], r)))
                bids.add(r.bid)
            new = sorted(bids - set(already))
            if new:
                le = idx["leases"].setdefault(_epoch(), {})
                for b in new:
                    le[b] = le.get(b, 0) + 1
                self._save(idx)
        for b in new:
            self._leases[b] = self._leases.get(b, 0) + 1
        return out, new

    def drop(self, key) -> bool:
        """Forget entry `key` (its blobs go unless another entry or a lease still holds them). False if it is gone."""
        with self._locked():
            idx = self._load()
            if key not in idx["entries"]:
                return False
            self._unref(idx, key)
            self._save(idx)
        return True

    def mark_backed(self, keys) -> int:
        """Entries `keys` are held by a backup store too (evicted first here). -> how many changed."""
        keys, n = set(keys), 0
        with self._locked():
            idx = self._load()
            for k, e in idx["entries"].items():
                b = k in keys
                if bool(e.get("backed")) != b:
                    e["backed"], n = b, n + 1
            if n:
                self._save(idx)
        return n

    def release_all(self) -> None:
        """Every lease this process holds (an engine was rebuilt: its requests are gone)."""
        held = [b for b, c in self._leases.items() for _ in range(c)]
        self.release(held)

    def stats(self) -> dict:
        with self._locked():
            idx = self._load()
            return {"entries": len(idx["entries"]), "blobs": len(idx["blobs"]),
                    "used_bytes": sum(v["size"] for v in idx["blobs"].values()), "size_bytes": self.size,
                    "leased_blobs": len(self._held(idx))}


_STORE = None


def store_from_env():
    """The process's pool (--prefix-store shared: QV_PREFIX_SHARED_NAME / _GB / _TAG, set by the server), or
    None. One per process: an engine rebuilt after a failure re-uses it (release_all drops the old engine's leases)."""
    global _STORE
    tag = os.environ.get("QV_PREFIX_SHARED_NAME", "").strip()
    cfg = os.environ.get("QV_PREFIX_SHARED_TAG", "").strip()
    if os.environ.get("QV_PREFIX_STORE", "gpu") != "shared" or not tag:
        return None
    if _STORE is None or _STORE.tag != tag:
        _STORE = SharedPrefixStore(tag, int(float(os.environ.get("QV_PREFIX_SHARED_GB", "12")) * 2**30), cfg)
    return _STORE


# ---- moving a saved prefix to another machine ------------------------------------------------------------------------
# The layout travels as JSON with type tags (never pickle: a pickle from the network could run code). Blobs travel as
# raw bytes, piece by piece over several connections, straight into the receiver's pool; each piece is checked against
# the sender's blake2b. A blob the receiver already holds (same origin: an earlier checkpoint of the same prompt sent
# before) is not sent again.
INDEX_FORMAT = 4              # layout of index and entries; a pool written by other code is reset, never misread
                              # (4: processes are named pid:start time)
XFER_LEASE_S = 300.0          # a sender keeps an exported prefix from eviction this long after the last piece asked
PIECE = 8 << 20               # bytes per request


def _dt_name(dt) -> str:
    return str(dt).replace("torch.", "")


def layout_to_json(o):
    from qv_band.kv.codec import BandedKV
    from qv_band.decode.spec_decode import PackedState
    if isinstance(o, TensorRef):
        return {"T": [o.bid, o.off, list(o.shape), _dt_name(o.dtype), o.small]}
    if isinstance(o, BandedKV):
        return {"B": [layout_to_json(o.packed), layout_to_json(o.min_s), layout_to_json(o.tau_s), o.bits,
                      list(o.orig_shape), _dt_name(o.orig_dtype), o.tau]}
    if isinstance(o, PackedState):
        return {"P": [[layout_to_json(h) for h in o.heads], list(o.shape), _dt_name(o.dtype), list(o.bits), o.escaped]}
    if isinstance(o, torch.device):
        return {"D": str(o)}
    if isinstance(o, torch.dtype):
        return {"y": _dt_name(o)}
    if isinstance(o, tuple):
        return {"t": [layout_to_json(v) for v in o]}
    if isinstance(o, list):
        return [layout_to_json(v) for v in o]
    if isinstance(o, dict):
        return {"d": [[layout_to_json(k), layout_to_json(v)] for k, v in o.items()]}
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    raise TypeError(f"cannot send a {type(o).__name__} in a prefix layout")


def _dtype(name: str):
    dt = getattr(torch, str(name), None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"not a dtype: {name!r}")
    return dt


def layout_from_json(o, bid_map: dict):
    """The inverse of layout_to_json, with every blob id mapped to the receiver's (bid_map: sender bid -> local bid)."""
    from qv_band.kv.codec import BandedKV
    from qv_band.decode.spec_decode import PackedState
    if isinstance(o, list):
        return [layout_from_json(v, bid_map) for v in o]
    if not isinstance(o, dict):
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        raise ValueError("bad layout")
    if len(o) != 1:
        raise ValueError("bad layout node")
    (k, v), = o.items()
    if k == "T":
        bid, off, shape, dt, small = v
        return TensorRef(bid_map[int(bid)], int(off), tuple(int(x) for x in shape), _dtype(dt), bool(small))
    if k == "B":
        packed, mn, st, bits, shape, dt, tau = v
        return BandedKV(packed=layout_from_json(packed, bid_map), min_s=layout_from_json(mn, bid_map),
                        tau_s=layout_from_json(st, bid_map), bits=int(bits), orig_shape=tuple(int(x) for x in shape),
                        orig_dtype=_dtype(dt), tau=float(tau))
    if k == "P":
        heads, shape, dt, bits, escaped = v
        p = PackedState.__new__(PackedState)
        p.heads = [layout_from_json(h, bid_map) for h in heads]
        p.shape, p.dtype, p.bits, p.escaped = tuple(int(x) for x in shape), _dtype(dt), [int(b) for b in bits], int(escaped)
        return p
    if k == "D":
        return torch.device(str(v))
    if k == "y":
        return _dtype(v)
    if k == "t":
        return tuple(layout_from_json(x, bid_map) for x in v)
    if k == "d":
        return {layout_from_json(a, bid_map): layout_from_json(b, bid_map) for a, b in v}
    raise ValueError(f"bad layout tag {k!r}")


def _export(self, key) -> dict | None:
    """Sender: lease entry `key` for a transfer and describe it -> {"lease", "pool", "n", "branch", "layout" (JSON),
    "blobs": [{"bid", "size", "small", "origin"}]}; None if it is gone."""
    import uuid
    with self._locked():
        idx = self._load()
        e = idx["entries"].get(key)
        if e is None:
            return None
        lease = f"xfer:{uuid.uuid4().hex}"
        idx["leases"][lease] = {b: 1 for b in e["blobs"]}
        idx.setdefault("lease_exp", {})[lease] = time.time() + XFER_LEASE_S
        lbid, ln = e["layout"]
        blobs = [{"bid": b, "size": idx["blobs"][b]["size"], "small": idx["blobs"][b].get("small", 0),
                  "origin": idx["blobs"][b].get("origin") or f"{idx['uuid']}:{b}"}
                 for b in e["blobs"] if b != lbid]
        lo = idx["blobs"][lbid]["off"]
        skel = pickle.loads(self._mm[lo:lo + ln])
        meta = {"lease": lease, "pool": idx["uuid"], "n": e["n"], "branch": e["branch"], "dtype": e.get("dtype"),
                "layout": {"skel": layout_to_json(skel), "tokens": layout_to_json(e["tokens"])}, "blobs": blobs}
        self._save(idx)
    return meta


def _piece(self, lease: str, bid: int, off: int, length: int) -> memoryview:
    """Sender: bytes [off, off + length) of blob `bid`, which `lease` must hold."""
    with self._locked():
        idx = self._load()
        if bid not in idx["leases"].get(lease, {}):
            raise KeyError("no such lease / blob (expired?)")
        idx["lease_exp"][lease] = time.time() + XFER_LEASE_S          # alive while pieces keep coming (slow links)
        self._save(idx)
        bl = idx["blobs"][bid]
        if off < 0 or length < 0 or off + length > bl["size"]:
            raise ValueError("range outside the blob")
        a = bl["off"] + off
    return memoryview(self._mm)[a:a + length]


def _release_lease(self, lease: str) -> None:
    with self._locked():
        idx = self._load()
        if idx["leases"].pop(lease, None) is not None:
            idx.get("lease_exp", {}).pop(lease, None)
            self._reap(idx)
            self._save(idx)


def _import(self, meta: dict, fetch, streams: int = 4) -> dict:
    """Receiver: copy the entry `meta` describes (from _export on another machine) into this pool. fetch(bid, off,
    length, view) fills `view` with the sender's bytes (and checks them). Blobs already here (same origin) are re-used.
    -> {"bytes", "seconds", "reused", "entry": bool}. Raises on any failure (nothing is published then)."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    n = int(meta["n"])
    with self._locked():
        idx = self._load()
        by_origin = {v.get("origin") or f"{idx['uuid']}:{b}": b for b, v in idx["blobs"].items() if not v.get("writing")}
        bid_map, todo, reused = {}, [], []
        for bl in meta["blobs"]:
            if not 0 <= int(bl["size"]) <= self.size or not 0 <= int(bl.get("small", 0)) <= int(bl["size"]):
                raise ValueError("bad blob size in a prefix description")
            local = by_origin.get(bl["origin"])
            if local is not None:
                bid_map[int(bl["bid"])] = local
                reused.append(local)
        le = idx["leases"].setdefault(_epoch(), {})
        for b in reused:                               # keep them until the entry holds them
            le[b] = le.get(b, 0) + 1
        for bl in meta["blobs"]:
            if int(bl["bid"]) in bid_map:
                continue
            size = int(bl["size"])
            off = self._alloc(idx, size, protect=set(reused) | {b for b, _ in todo})
            if off is None:
                for b, _ in todo:
                    idx["blobs"].pop(b, None)
                self._unlease(idx, reused)
                self._save(idx)
                raise MemoryError(f"pool full: cannot take a {size >> 20} MB blob")
            b = idx["next_bid"]
            idx["next_bid"] += 1
            idx["blobs"][b] = {"off": off, "size": size, "refs": 0, "writing": _epoch(), "small": int(bl["small"]),
                               "origin": bl["origin"]}
            bid_map[int(bl["bid"])] = b
            todo.append((b, bl))
        self._save(idx)
    nbytes = sum(int(bl["size"]) for _, bl in todo)
    try:
        jobs = [(b, int(bl["bid"]), o, min(PIECE, int(bl["size"]) - o), idx["blobs"][b]["off"])
                for b, bl in todo for o in range(0, int(bl["size"]), PIECE)]
        mv = memoryview(self._mm)
        with ThreadPoolExecutor(max_workers=max(1, streams)) as ex:      # several connections at once, in blob order
            list(ex.map(lambda j: fetch(j[1], j[2], j[3], mv[j[4] + j[2]:j[4] + j[2] + j[3]]), jobs))
        skel = layout_from_json(meta["layout"]["skel"], bid_map)
        tokens_ref = layout_from_json(meta["layout"]["tokens"], bid_map)
        if not isinstance(tokens_ref, TensorRef):
            raise ValueError("bad token reference in a prefix description")
        sizes = {bid_map[int(bl["bid"])]: int(bl["size"]) for bl in meta["blobs"]}
        for r in _refs_in(skel, [tokens_ref]):       # a reference past its blob would read another prefix's bytes
            if r.off < 0 or any(x < 0 for x in r.shape) or r.off + _ref_bytes(r) > sizes[r.bid]:
                raise ValueError("a tensor of the prefix description lies outside its blob")
        layout = pickle.dumps(skel, protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException:
        with self._locked():
            idx = self._load()
            for b, _ in todo:
                idx["blobs"].pop(b, None)
            self._unlease(idx, reused)
            self._save(idx)
        raise
    with self._locked():
        idx = self._load()
        self._unlease(idx, reused)
        if any(b not in idx["blobs"] for b in list(bid_map.values())):
            for b, _ in todo:                        # do not leave the fetched blobs behind as "being written"
                idx["blobs"].pop(b, None)
            self._save(idx)
            raise RuntimeError("a blob vanished during the transfer")
        tok_bl = idx["blobs"][tokens_ref.bid]
        toks = self._view(tok_bl["off"], tokens_ref).tolist()
        if any(e["n"] == n and self._tokens(idx, k) == toks for k, e in idx["entries"].items()):
            for b, _ in todo:                        # arrived meanwhile (another pull, or computed here): keep one
                idx["blobs"].pop(b, None)
            self._save(idx)
            return {"bytes": nbytes, "seconds": time.time() - t0, "reused": len(reused), "entry": False}
        loff = self._alloc(idx, _align(len(layout)), protect=set(bid_map.values()))
        if loff is None:
            for b, _ in todo:
                idx["blobs"].pop(b, None)
            self._save(idx)
            raise MemoryError("pool full (layout)")
        lbid = idx["next_bid"]
        idx["next_bid"] += 1
        idx["blobs"][lbid] = {"off": loff, "size": _align(len(layout)), "refs": 0}
        self.base[loff:loff + len(layout)].copy_(torch.frombuffer(bytearray(layout), dtype=torch.uint8))
        for b, _ in todo:
            idx["blobs"][b].pop("writing", None)
        blobs = sorted(set(bid_map.values()) | {lbid})
        for b in blobs:
            idx["blobs"][b]["refs"] += 1
        key = idx["next_key"]
        idx["next_key"] += 1
        idx["entries"][key] = {"n": n, "tokens": tokens_ref, "layout": (lbid, len(layout)), "blobs": blobs,
                               "branch": bool(meta.get("branch")), "used": time.time(), "by": "import:" + meta["pool"],
                               "dtype": str(meta["dtype"]) if meta.get("dtype") else None}
        self._save(idx)
    return {"bytes": nbytes, "seconds": time.time() - t0, "reused": len(reused), "entry": True}


def _unlease(self, idx: dict, bids) -> None:
    le = idx["leases"].get(_epoch(), {})
    for b in bids:
        le[b] = le.get(b, 0) - 1
        if le[b] <= 0:
            le.pop(b, None)


def _key_for(self, n: int, tokens: list):
    """The entry holding exactly `tokens` (length n), or None."""
    with self._locked():
        idx = self._load()
        for k, e in idx["entries"].items():
            if e["n"] == n and self._tokens(idx, k) == tokens:
                return k
    return None


def _uuid(self) -> str:
    with self._locked():
        return self._load()["uuid"]


def _refs_in(o, out: list) -> list:
    """Every TensorRef inside a layout object, in walk order."""
    from qv_band.kv.codec import BandedKV
    from qv_band.decode.spec_decode import PackedState
    if isinstance(o, TensorRef):
        out.append(o)
    elif isinstance(o, BandedKV):
        for t in (o.packed, o.min_s, o.tau_s):
            _refs_in(t, out)
    elif isinstance(o, PackedState):
        for h in o.heads:
            _refs_in(h, out)
    elif isinstance(o, dict):
        for v in o.values():
            _refs_in(v, out)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _refs_in(v, out)
    return out


def _ref_bytes(r: TensorRef) -> int:
    n = 1
    for x in r.shape:
        n *= x
    return n * torch.empty((), dtype=r.dtype).element_size()


def _export_layers(self, key, a: int, b: int, drafter: bool) -> dict | None:
    """Sender, layer slice: layers [a, b) of entry `key` (and the drafter context when `drafter`: the stage that holds
    the last layers), as byte RANGES of the blobs (one checkpoint's blob holds every layer). A stage of a chain sends its
    layers; a whole-model replica sends the slice a stage asks for. -> meta like export() plus "layers" and "ranges"."""
    import uuid
    with self._locked():
        idx = self._load()
        e = idx["entries"].get(key)
        if e is None:
            return None
        lbid, ln = e["layout"]
        lo = idx["blobs"][lbid]["off"]
        skel = pickle.loads(self._mm[lo:lo + ln])
        if not (0 <= a < b <= len(skel["state"])):
            raise ValueError(f"layers {a}:{b} outside 0:{len(skel['state'])}")
        part = {"state": skel["state"][a:b], "drafter": skel["drafter"] if drafter else None}
        refs = _refs_in(part, [])
        bids = sorted({r.bid for r in refs})
        lease = f"xfer:{uuid.uuid4().hex}"
        idx["leases"][lease] = {x: 1 for x in bids}
        idx.setdefault("lease_exp", {})[lease] = time.time() + XFER_LEASE_S
        ranges = sorted({(r.bid, r.off, _ref_bytes(r), r.small) for r in refs})
        meta = {"lease": lease, "pool": idx["uuid"], "n": e["n"], "branch": e["branch"], "dtype": e.get("dtype"),
                "layers": [a, b], "of": len(skel["state"]),
                "layout": {"skel": layout_to_json(part), "tokens": layout_to_json(e["tokens"])},
                "blobs": [{"bid": e["tokens"].bid, "size": idx["blobs"][e["tokens"].bid]["size"], "small": 0,
                           "origin": idx["blobs"][e["tokens"].bid].get("origin") or f"{idx['uuid']}:{e['tokens'].bid}"}],
                "ranges": [list(r) for r in ranges]}
        idx["leases"][lease][e["tokens"].bid] = 1
        self._save(idx)
    return meta


def _import_layers(self, metas: list, fetch, streams: int = 4) -> dict:
    """Receiver: one entry from layer slices (metas of _export_layers, possibly from several machines) that together
    cover `layers` (the receiving server's own range: all layers for a whole-model replica, its range for a stage).
    fetch(meta, bid, off, length, view) fills view. The slices' byte ranges land in ONE new blob here (small scales
    first); the layout's references are renumbered. Nothing is published unless every range arrived."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    metas = sorted(metas, key=lambda m: m["layers"][0])
    lo_l, hi_l = metas[0]["layers"][0], metas[-1]["layers"][1]
    for m1, m2 in zip(metas, metas[1:]):
        if m1["layers"][1] != m2["layers"][0] or m1["n"] != m2["n"]:
            raise ValueError("layer slices must be contiguous and of the same prefix")
    jobs, small_sz, big_sz = [], 0, 0
    for mi, m in enumerate(metas):
        for bid, off, nb, small in m["ranges"]:
            if int(off) < 0 or not 0 <= int(nb) <= self.size:
                raise ValueError("bad byte range in a layer-slice description")
            jobs.append([mi, int(bid), int(off), int(nb), bool(small), 0])
    for j in jobs:                                           # small scales first: one copy to the card on a restore
        if j[4]:
            j[5] = small_sz
            small_sz += _align(j[3])
    for j in jobs:
        if not j[4]:
            j[5] = small_sz + big_sz
            big_sz += _align(j[3])
    tok_meta = metas[-1]
    tr = tok_meta["layout"]["tokens"]["T"]                   # [bid, off, shape, dtype, small]: only this range is fetched
    tok_bid, tok_src = int(tr[0]), int(tr[1])
    tok_nb = int(np.prod(tr[2])) * torch.empty((), dtype=_dtype(tr[3])).element_size() if tr[2] else 0
    total = small_sz + big_sz
    with self._locked():
        idx = self._load()
        off = self._alloc(idx, total + _align(tok_nb), protect=set())
        if off is None:
            self._save(idx)
            raise MemoryError(f"pool full: cannot take {total >> 20} MB of layer slices")
        bid = idx["next_bid"]
        idx["next_bid"] += 1
        idx["blobs"][bid] = {"off": off, "size": total + _align(tok_nb), "refs": 0, "writing": _epoch(),
                             "small": small_sz}
        self._save(idx)
    mv = memoryview(self._mm)
    tok_off = total
    try:
        spans = []                                           # [meta index, bid, src off, src end, [(src off, nb, dst)]]
        for j in sorted(jobs, key=lambda j: (j[0], j[1], j[2])):
            mi, b_, o_, nb, _, d_ = j
            sp = spans[-1] if spans else None
            if sp and sp[0] == mi and sp[1] == b_ and o_ - sp[3] <= ALIGN and o_ + nb - sp[2] <= PIECE:
                sp[3] = max(sp[3], o_ + nb)
                sp[4].append((o_, nb, off + d_))
            else:
                spans.append([mi, b_, o_, o_ + nb, [(o_, nb, off + d_)]])

        def get_span(sp):
            mi, b_, s0, s1, parts = sp
            buf = bytearray(s1 - s0)
            fetch(metas[mi], b_, s0, s1 - s0, memoryview(buf))
            for o_, nb, dst in parts:
                mv[dst:dst + nb] = buf[o_ - s0:o_ - s0 + nb]

        with ThreadPoolExecutor(max_workers=max(1, streams)) as ex:
            list(ex.map(get_span, spans))
        fetch(tok_meta, tok_bid, tok_src, tok_nb, mv[off + tok_off:off + tok_off + tok_nb])
        remap = {(mi, b_, o_): (bid, j5) for mi, b_, o_, _, _, j5 in jobs}
        state, drafter = [], None
        for mi, m in enumerate(metas):
            part = _layout_from_json_ranges(m["layout"]["skel"], lambda r, mi=mi: TensorRef(
                bid, remap[(mi, r[0], r[1])][1], r[2], r[3], r[4]))
            state.extend(part["state"])
            if part.get("drafter") is not None:
                drafter = part["drafter"]
        tokens_ref = TensorRef(bid, tok_off, tuple(int(x) for x in tr[2]), _dtype(tr[3]), False)
        layout = pickle.dumps({"state": state, "drafter": drafter}, protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException:
        with self._locked():
            idx = self._load()
            idx["blobs"].pop(bid, None)
            self._save(idx)
        raise
    with self._locked():
        idx = self._load()
        n = int(tok_meta["n"])
        toks = self._view(off, tokens_ref).tolist()
        if any(e["n"] == n and e.get("layers", [lo_l, hi_l]) == [lo_l, hi_l] and self._tokens(idx, k) == toks
               for k, e in idx["entries"].items()):
            idx["blobs"].pop(bid, None)              # already here (another pull, or computed meanwhile): keep one
            self._save(idx)
            return {"bytes": total, "seconds": time.time() - t0, "entry": False, "layers": [lo_l, hi_l], "n": n}
        loff = self._alloc(idx, _align(len(layout)), protect={bid})
        if loff is None or bid not in idx["blobs"]:
            idx["blobs"].pop(bid, None)
            self._save(idx)
            raise MemoryError("pool full (layout)")
        lbid = idx["next_bid"]
        idx["next_bid"] += 1
        idx["blobs"][lbid] = {"off": loff, "size": _align(len(layout)), "refs": 1}
        self.base[loff:loff + len(layout)].copy_(torch.frombuffer(bytearray(layout), dtype=torch.uint8))
        idx["blobs"][bid].pop("writing", None)
        idx["blobs"][bid]["refs"] = 1
        key = idx["next_key"]
        idx["next_key"] += 1
        idx["entries"][key] = {"n": n, "tokens": tokens_ref, "layout": (lbid, len(layout)), "blobs": [bid, lbid],
                               "branch": bool(tok_meta.get("branch")), "used": time.time(),
                               "by": "slices:" + ",".join(sorted({m["pool"] for m in metas})),
                               "dtype": tok_meta.get("dtype"), "layers": [lo_l, hi_l]}
        self._save(idx)
    return {"bytes": total, "seconds": time.time() - t0, "entry": True, "layers": [lo_l, hi_l], "n": n,
            "of": int(tok_meta["of"])}


def _layout_from_json_ranges(o, mk_ref):
    """layout_from_json with every TensorRef built by mk_ref([bid, off, shape, dtype, small]) (slice imports)."""
    if isinstance(o, dict) and len(o) == 1 and "T" in o:
        bid, off, shape, dt, small = o["T"]
        return mk_ref([int(bid), int(off), tuple(int(x) for x in shape), _dtype(dt), bool(small)])
    if isinstance(o, dict) and len(o) == 1 and "B" in o:
        from qv_band.kv.codec import BandedKV
        packed, mn, st, bits, shape, dt, tau = o["B"]
        return BandedKV(packed=_layout_from_json_ranges(packed, mk_ref), min_s=_layout_from_json_ranges(mn, mk_ref),
                        tau_s=_layout_from_json_ranges(st, mk_ref), bits=int(bits),
                        orig_shape=tuple(int(x) for x in shape), orig_dtype=_dtype(dt), tau=float(tau))
    if isinstance(o, dict) and len(o) == 1 and "P" in o:
        from qv_band.decode.spec_decode import PackedState
        heads, shape, dt, bits, escaped = o["P"]
        p = PackedState.__new__(PackedState)
        p.heads = [_layout_from_json_ranges(h, mk_ref) for h in heads]
        p.shape, p.dtype, p.bits, p.escaped = tuple(int(x) for x in shape), _dtype(dt), [int(x) for x in bits], int(escaped)
        return p
    if isinstance(o, dict) and len(o) == 1 and "t" in o:
        return tuple(_layout_from_json_ranges(x, mk_ref) for x in o["t"])
    if isinstance(o, dict) and len(o) == 1 and "d" in o:
        return {_layout_from_json_ranges(k, mk_ref): _layout_from_json_ranges(v, mk_ref) for k, v in o["d"]}
    if isinstance(o, list):
        return [_layout_from_json_ranges(x, mk_ref) for x in o]
    return layout_from_json(o, {})


SharedPrefixStore.export_layers = _export_layers
SharedPrefixStore.import_layers = _import_layers
SharedPrefixStore.export = _export
SharedPrefixStore.piece = _piece
SharedPrefixStore.release_lease = _release_lease
SharedPrefixStore.import_ = _import
SharedPrefixStore._unlease = _unlease
SharedPrefixStore.key_for = _key_for
SharedPrefixStore.uuid = _uuid
