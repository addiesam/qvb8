"""Build-on-first-use for every native extension of every pack format: no setup step on any machine.

A format declares its native code by calling one of these where it first needs it:

    load_cuda_ext(name, sources, cflags=, cuda_cflags=, ldflags=)   CUDA (+ optional host .cpp), for the current GPU
    load_sycl_ext(name, sources, cflags=)                            Intel XPU (SYCL .sycl sources, via torch)
    load_c_ext(name, sources, cflags=, ldflags=, machines=)          plain CPython C extension (CPU kernels)

Each compiles with THIS machine's toolchain and torch, and caches the result under `<qv_band>/.ext_cache/`, keyed on
source bytes + flags + target (GPU arch / XPU / CPU machine) + torch version + compiler. So:

  * the first run on a machine, or after a source change (a new tree copied over), compiles once (announced on stdout); later
    starts import the cached build;
  * machines sharing a checkout keep separate builds; nothing stale or ABI-mismatched can be loaded.

A new format (qvb9, ...) needs nothing beyond calling these from its loader -- no setup.py, no build.py.

CUDA toolkit: the newest under $CUDA_HOME, /usr/local/cuda, /usr/local/cuda-* that can still target the GPU (CUDA 13
dropped sm_70/72: a V100 gets the newest 12.x). Concurrent first starts (torchrun ranks) serialise on torch's
build-directory lock (CUDA / SYCL) or a host+pid lock file (C), which a later start clears if its builder
died.
"""
from __future__ import annotations

import functools
import glob
import hashlib
import importlib.util
import os
import platform
import re
import subprocess
import sys

_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".ext_cache")
_LOADED: dict = {}


_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def _deps(sources) -> list[str]:
    """Every file the build reads from the tree: the sources plus, recursively, each `#include "..."` (resolved
    next to the including file). `<...>` system / torch headers are covered by the toolkit and torch versions in
    the key. Exactly the files that matter: a header edit rebuilds, an unrelated kernel's header does not."""
    seen, todo = set(), [os.path.abspath(s) for s in sources]
    while todo:
        f = todo.pop()
        if f in seen or not os.path.isfile(f):
            continue
        seen.add(f)
        with open(f, "r", errors="replace") as fh:
            text = fh.read()
        todo.extend(os.path.normpath(os.path.join(os.path.dirname(f), inc)) for inc in _INCLUDE.findall(text))
    return sorted(seen)


def _key(name: str, sources, *extra) -> str:
    """Hash of the build inputs: source + included file contents (by path relative to the first source's
    directory, so two checkouts at different paths share the key), flags, target, toolchain and torch."""
    root = os.path.dirname(os.path.abspath(sources[0]))
    h = hashlib.sha1()
    for f in _deps(sources):
        h.update(os.path.relpath(f, root).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    h.update(repr(extra).encode())
    return h.hexdigest()[:12]


def _announce(name: str, target: str, how: str) -> None:
    print(f"[qv_band] compiling {name} for {target} with {how} "
          f"(first run on this machine or after a source update)", flush=True)


def _venv_on_path(*more: str) -> None:
    """torch's builder needs ninja, which lives in the venv's bin/ (not always on PATH)."""
    os.environ["PATH"] = os.pathsep.join([os.path.dirname(sys.executable), *more, os.environ.get("PATH", "")])


# -- CUDA ------------------------------------------------------------------------------------------------------------
def _nvcc_version(home: str) -> tuple[int, int] | None:
    nvcc = os.path.join(home, "bin", "nvcc")
    if not os.path.isfile(nvcc):
        return None
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


@functools.lru_cache(maxsize=None)      # one nvcc probe per arch per process, not per extension
def _toolkit_for(cc: int) -> tuple[str, tuple[int, int]]:
    """CUDA_HOME that can compile for compute capability `cc` (e.g. 70, 121). Among those, one with the same major
    version as torch's own CUDA wins (torch refuses to build an extension with a different major), newest first."""
    cands = [os.environ.get("CUDA_HOME", ""), "/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda-*"), reverse=True)
    found = []
    for h in dict.fromkeys(c for c in cands if c):
        v = _nvcc_version(h)
        if v is not None:
            found.append((v, h))
    ok = [(v, h) for v, h in found if not (cc < 75 and v[0] >= 13)]
    if not ok:
        raise RuntimeError(f"no CUDA toolkit can compile for sm_{cc} (found {[(h, v) for v, h in found]}); "
                           f"sm_70/72 need CUDA <= 12.x")
    import torch
    tmaj = int((torch.version.cuda or "0").split(".")[0])
    v, h = max(ok, key=lambda vh: (vh[0][0] == tmaj, vh[0]))
    return h, v


def load_cuda_ext(name: str, sources: list[str], *, cflags=(), cuda_cflags=(), ldflags=(), device=None):
    """Extension `name` built from `sources` for the GPU `device` (default: current), built here if needed.
    Cheap to call per launch: after the first call per card it is a dict lookup, so callers need not (and must
    not, with cards of different archs in one process) keep their own single global copy."""
    import torch
    idx = None if device is None else torch.device(device).index
    dev = torch.cuda.current_device() if idx is None else idx      # "cuda" (no index) = the current card, not card 0
    hit = _LOADED.get((name, "dev", dev))                          # hot path: no hashing once a card has its build
    if hit is not None:
        return hit
    maj, mnr = torch.cuda.get_device_capability(dev)
    cc = maj * 10 + mnr
    arch = f"{cc}a" if maj >= 9 else f"{cc}"           # sm_90a / 100a / 120a / 121a: arch-specific features
    if (name, arch) in _LOADED:
        _LOADED[(name, "dev", dev)] = _LOADED[(name, arch)]
        return _LOADED[(name, arch)]
    home, ver = _toolkit_for(cc)
    bdir = os.path.join(_CACHE, f"{name}-sm{arch}-{_key(name, sources, cflags, cuda_cflags, ldflags, arch, torch.__version__, ver)}")
    os.makedirs(bdir, exist_ok=True)
    if not glob.glob(os.path.join(bdir, f"{name}*.so")):
        _announce(name, f"sm_{arch}", f"CUDA {ver[0]}.{ver[1]}")
    _venv_on_path(os.path.join(home, "bin"))
    os.environ["CUDA_HOME"] = home
    import torch.utils.cpp_extension as ce
    ce.CUDA_HOME = home
    mod = ce.load(name=name, sources=list(sources), build_directory=bdir, verbose=False,
                  extra_cflags=list(cflags),
                  extra_cuda_cflags=list(cuda_cflags) + [f"-gencode=arch=compute_{arch},code=sm_{arch}",
                                                         f"-gencode=arch=compute_{cc},code=compute_{cc}"],
                  extra_ldflags=list(ldflags))
    _LOADED[(name, arch)] = _LOADED[(name, "dev", dev)] = mod
    return mod


# -- Intel XPU (SYCL) ------------------------------------------------------------------------------------------------
def load_sycl_ext(name: str, sources: list[str], *, cflags=(), sycl_cflags=()):
    """SYCL extension for the XPU, compiled by torch's builder with the oneAPI compiler (icpx) on PATH."""
    if ("sycl", name) in _LOADED:
        return _LOADED[("sycl", name)]
    import torch
    bdir = os.path.join(_CACHE, f"{name}-xpu-{_key(name, sources, cflags, sycl_cflags, torch.__version__)}")
    os.makedirs(bdir, exist_ok=True)
    if not glob.glob(os.path.join(bdir, f"{name}*.so")) and not glob.glob(os.path.join(bdir, f"{name}*.pyd")):
        _announce(name, "XPU", "the oneAPI SYCL compiler")
    _venv_on_path()
    import torch.utils.cpp_extension as ce
    mod = ce.load(name=name, sources=list(sources), build_directory=bdir, verbose=False,
                  extra_cflags=list(cflags), extra_sycl_cflags=list(sycl_cflags))
    _LOADED[("sycl", name)] = mod
    return mod


class NotForThisMachine(ImportError):
    """load_c_ext: the kernel has no build for this CPU architecture (expected, not a failure)."""


# -- build lock (C builds; torch's builder locks CUDA / SYCL itself) ---------------------------------------------
def _lock_take(lock: str) -> bool:
    """Create `lock` holding "host pid"; False if another builder holds it."""
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(f"{platform.node()} {os.getpid()}")
    return True


def _lock_owner_dead(lock: str) -> bool:
    """True when the lock's builder is gone: a pid on THIS host that no longer exists (killed / OOM), or -- for a
    builder on another host sharing the checkout, whose pid cannot be checked -- no progress for an hour."""
    import time
    try:
        host, pid = open(lock).read().split()
        if host == platform.node():
            try:
                os.kill(int(pid), 0)
                return False
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
        return time.time() - os.path.getmtime(lock) > 3600
    except (OSError, ValueError):
        return False


def _lock_wait(lock: str, done, name: str) -> None:
    """Wait for another process's build of `name`; take over (clear the lock) if that builder died."""
    import time
    said = False
    while not done() and os.path.exists(lock):
        if _lock_owner_dead(lock):
            print(f"[qv_band] {name}: the previous build was interrupted -- building again", flush=True)
            try:
                os.remove(lock)
            except OSError:
                pass
            return
        if not said:
            try:
                owner = open(lock).read().strip()
            except OSError:
                owner = "?"
            print(f"[qv_band] waiting for {name} being compiled by {owner}", flush=True)
            said = True
        time.sleep(0.5)


# -- plain C (CPU kernels) -------------------------------------------------------------------------------------------
def load_c_ext(name: str, sources: list[str], *, cflags=(), ldflags=(), machines=None):
    """A CPython C extension (module init PyInit_<name>) compiled with this platform's C compiler (gcc / clang /
    MSVC, via setuptools). `machines`: platform.machine() values it can be built for (e.g. x86_64 for AVX2 code);
    elsewhere ImportError, which callers treat as "this kernel is not available here"."""
    if ("c", name) in _LOADED:
        return _LOADED[("c", name)]
    mach = platform.machine().lower()
    if machines is not None and mach not in machines:
        raise NotForThisMachine(f"{name}: no build for {mach} (needs one of {sorted(machines)})")
    bdir = os.path.join(_CACHE, f"{name}-{mach}-{_key(name, sources, cflags, ldflags, sys.version, sys.platform)}")
    os.makedirs(bdir, exist_ok=True)
    found = lambda: glob.glob(os.path.join(bdir, f"{name}*.so")) + glob.glob(os.path.join(bdir, f"{name}*.pyd"))
    built = found()
    if not built:
        # One builder at a time (torchrun ranks start together). The module is linked in a private stage dir and
        # moved into bdir with os.replace, so found() only ever sees a complete file; the others wait for it.
        lock = os.path.join(bdir, "lock")
        if _lock_take(lock):
            try:
                _announce(name, mach, "the platform C compiler")
                from setuptools import Distribution, Extension
                stage = os.path.join(bdir, f"stage-{os.getpid()}")
                dist = Distribution({"name": name, "ext_modules": [Extension(name, list(sources),
                                     extra_compile_args=list(cflags), extra_link_args=list(ldflags))]})
                cmd = dist.get_command_obj("build_ext")
                cmd.build_lib = stage
                cmd.build_temp = os.path.join(stage, "tmp")
                cmd.ensure_finalized()
                cmd.run()
                for f in glob.glob(os.path.join(stage, f"{name}*.so")) + glob.glob(os.path.join(stage, f"{name}*.pyd")):
                    os.replace(f, os.path.join(bdir, os.path.basename(f)))
            finally:
                try:
                    os.remove(lock)
                except OSError:
                    pass
        else:
            _lock_wait(lock, found, name)
            if not found():                               # the builder failed or died: build here
                return load_c_ext(name, sources, cflags=cflags, ldflags=ldflags, machines=machines)
        built = found()
        if not built:
            raise ImportError(f"{name}: build produced no module in {bdir}")
    spec = importlib.util.spec_from_file_location(name, built[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LOADED[("c", name)] = mod
    return mod
