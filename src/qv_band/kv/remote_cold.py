"""Cold KV on another machine, with the attention over it computed THERE (qv_band/design_doc/serving_topology.md §5).

Why: a machine whose own memory is full (the the dev GPU holding a ~100 GB model in its 128 GB unified memory) has no host RAM
to offload old KV into, and old KV is read at EVERY step, so it cannot be shipped back per step either. A KV host keeps
the old blocks and computes the partial attention over them; per layer and step only the queries go out and (o, lse)
comes back. The main machine merges that with its own part by log-sum-exp: the same algebra as the local cold tier.

Pieces (this file has no torch at import, so the DRAM PC runs it with numpy only):
  wire format   encode()/decode(): JSON header + raw arrays (no pickle)
  block         the ids of tokens [a, b) of one layer and slot, built from the packed segments (exact 16-bit ids), with
                each covered segment's per-channel (min, step): value = min + id * step
  attention     attend_numpy() / attend_torch(): exact softmax attention of queries over blocks, -> (o, lse)
  client        RemoteCold (torch side): background shipping, one attend request per layer call, free on drop
The server is ai_engine/servers/kv_host.py.
"""
from __future__ import annotations

import json
import struct

import numpy as np

_DT = {"uint8": np.uint8, "int16": np.int16, "uint16": np.uint16, "int32": np.int32, "int64": np.int64,
       "float16": np.float16, "float32": np.float32}


# ---- wire format ------------------------------------------------------------------------------------------------------
def encode(arrays: dict, meta: dict | None = None) -> bytes:
    """{name: numpy array} + meta -> bytes: 4-byte header length, JSON header, then each array's raw bytes."""
    names, parts = [], []
    for k, a in arrays.items():
        a = np.ascontiguousarray(a)
        names.append([k, str(a.dtype), list(a.shape)])
        parts.append(a.tobytes())
    head = json.dumps({"arrays": names, "meta": meta or {}}).encode()
    return struct.pack("<I", len(head)) + head + b"".join(parts)


def decode(buf) -> tuple:
    """bytes -> ({name: numpy array (read-only views of buf)}, meta)."""
    mv = memoryview(buf)
    (n,) = struct.unpack("<I", mv[:4])
    head = json.loads(bytes(mv[4:4 + n]))
    off, out = 4 + n, {}
    for name, dt, shape in head["arrays"]:
        t = _DT[dt]
        cnt = int(np.prod(shape)) if shape else 1
        nb = cnt * np.dtype(t).itemsize
        out[name] = np.frombuffer(mv[off:off + nb], dtype=t).reshape(shape)
        off += nb
    return out, head["meta"]


# ---- attention (numpy: the DRAM PC) -----------------------------------------------------------------------------------
def prepare_numpy(blk: dict) -> dict:
    """A received block -> keys / values as float32 (h, B, d), decoded once (value = min + id * step), laid out for
    batched BLAS matmuls: a CPU host then does only matmuls per step (numpy einsum is single-threaded)."""
    out = {"a": int(blk["a"])}
    seg = blk["row_seg"]
    for side in ("k", "v"):
        ids = blk["ids_" + side].astype(np.uint16).astype(np.float32)               # (B, h, d)
        x = blk["min_" + side][seg] + ids * blk["step_" + side][seg]
        out[side] = np.ascontiguousarray(x.transpose(1, 0, 2))                       # (h, B, d)
    out["nbytes"] = out["k"].nbytes + out["v"].nbytes
    return out


def attend_prepared(q: np.ndarray, q_pos0: int, scale: float, blocks: list) -> tuple:
    """q (MQ, Hq, D) float32 at absolute positions q_pos0.. over prepared blocks; keys after a query's position are
    not seen. -> o (MQ, Hq, D) float32, lse (MQ, Hq) float32 (-inf where no key is visible)."""
    mq, hq, d = q.shape
    m = np.full((mq, hq), -np.inf, np.float32)
    l = np.zeros((mq, hq), np.float32)
    acc = np.zeros((mq, hq, d), np.float32)
    qpos = q_pos0 + np.arange(mq)
    for blk in blocks:
        k, v = blk["k"], blk["v"]                                                    # (h, B, d)
        h, b, _ = k.shape
        g = hq // h
        kpos = blk["a"] + np.arange(b)
        mask = kpos[None, :] > qpos[:, None]                                         # (MQ, B)
        if mask.all():
            continue
        qh = q.reshape(mq, h, g, d).transpose(1, 0, 2, 3).reshape(h, mq * g, d)      # (h, MQ*g, d)
        s = np.matmul(qh, k.transpose(0, 2, 1)) * scale                               # (h, MQ*g, B)  BLAS
        s = s.reshape(h, mq, g, b)
        s[np.broadcast_to(mask[None, :, None, :], s.shape)] = -np.inf
        mb = s.max(axis=-1).transpose(1, 0, 2).reshape(mq, hq)
        mnew = np.maximum(m, mb)
        safe = np.where(np.isinf(mnew), 0.0, mnew).astype(np.float32)
        p = np.exp(s - safe.reshape(mq, h, g).transpose(1, 0, 2)[..., None])         # (h, MQ, g, B)
        a_old = np.where(np.isinf(m), 0.0, np.exp(m - safe)).astype(np.float32)
        l = l * a_old + p.sum(axis=-1).transpose(1, 0, 2).reshape(mq, hq)
        pv = np.matmul(p.reshape(h, mq * g, b), v)                                   # (h, MQ*g, d)  BLAS
        acc = acc * a_old[..., None] + pv.reshape(h, mq, g, d).transpose(1, 0, 2, 3).reshape(mq, hq, d)
        m = mnew
    o = acc / np.maximum(l, 1e-30)[..., None]
    lse = np.where(l > 0, np.where(np.isinf(m), 0.0, m) + np.log(np.maximum(l, 1e-30)), -np.inf).astype(np.float32)
    return o.astype(np.float32), lse


def attend_numpy(q: np.ndarray, q_pos0: int, scale: float, blocks: list) -> tuple:
    """attend_prepared over raw (as received) blocks."""
    return attend_prepared(q, q_pos0, scale, [prepare_numpy(b) for b in blocks])


def prepare_torch(blk: dict, dev) -> dict:
    """A received block -> keys / values DECODED ONCE, resident on the GPU as fp16 (h, B, d): the same bytes as the
    compact ids (2 per element), and a step then costs only its matmuls. Before every attend call copied
    the ids of every block from host RAM to the card and decoded them again (4 blocks of 8192 tokens: 134 MB moved
    and decoded per layer per step: 26 ms for a 16-row decode step, 146 ms for a 512-row prefill chunk)."""
    import torch
    out = {"a": int(blk["a"])}
    seg = torch.as_tensor(np.ascontiguousarray(blk["row_seg"])).to(dev).long()
    for side in ("k", "v"):
        ids = torch.as_tensor(np.ascontiguousarray(blk["ids_" + side])).to(dev).to(torch.int32) & 0xFFFF
        mn = torch.as_tensor(np.ascontiguousarray(blk["min_" + side])).to(dev)
        st = torch.as_tensor(np.ascontiguousarray(blk["step_" + side])).to(dev)
        x = mn[seg] + ids.float() * st[seg]                                          # (B, h, d) fp32, exact decode
        out[side] = x.clamp_(-65504.0, 65504.0).to(torch.float16).permute(1, 0, 2).contiguous()   # (h, B, d)
        del ids, x
    out["nbytes"] = out["k"].numel() * 2 + out["v"].numel() * 2
    return out


def attend_resident(q, q_pos0: int, scale: float, blocks: list, dev):
    """attend_torch over blocks prepared by prepare_torch (resident fp16 keys / values): batched matmuls per block,
    scores and sums in fp32. q (MQ, Hq, D) any float dtype -> o (MQ, Hq, D) fp32, lse (MQ, Hq) fp32."""
    import torch
    mq, hq, d = q.shape
    q = q.to(dev, torch.float32)
    m = torch.full((mq, hq), float("-inf"), device=dev)
    l = torch.zeros((mq, hq), device=dev)
    acc = torch.zeros((mq, hq, d), device=dev)
    qpos = q_pos0 + torch.arange(mq, device=dev)
    for blk in blocks:
        k, v = blk["k"], blk["v"]                                                    # (h, B, d) fp16
        h, b, _ = k.shape
        g = hq // h
        last = int(blk["a"]) + b - 1
        if int(blk["a"]) > q_pos0 + mq - 1:
            continue                                                                 # every key is after every query
        qh = q.reshape(mq, h, g, d).permute(1, 0, 2, 3).reshape(h, mq * g, d)        # (h, MQ*g, d)
        s = torch.matmul(qh, k.float().transpose(1, 2)) * scale                      # (h, MQ*g, B)
        s = s.reshape(h, mq, g, b)
        if last > q_pos0:                                                            # some keys lie after some queries
            kpos = int(blk["a"]) + torch.arange(b, device=dev)
            s = s.masked_fill((kpos[None, :] > qpos[:, None])[None, :, None, :], float("-inf"))
        mb = s.amax(dim=-1).permute(1, 0, 2).reshape(mq, hq)
        mnew = torch.maximum(m, mb)
        safe = torch.where(torch.isinf(mnew), torch.zeros_like(mnew), mnew)
        p = torch.exp(s - safe.reshape(mq, h, g).permute(1, 0, 2)[..., None])        # (h, MQ, g, B)
        a_old = torch.where(torch.isinf(m), torch.zeros_like(m), torch.exp(m - safe))
        l = l * a_old + p.sum(dim=-1).permute(1, 0, 2).reshape(mq, hq)
        pv = torch.matmul(p.reshape(h, mq * g, b), v.float())                        # (h, MQ*g, d)
        acc = acc * a_old[..., None] + pv.reshape(h, mq, g, d).permute(1, 0, 2, 3).reshape(mq, hq, d)
        m = mnew
    o = acc / l.clamp_min(1e-30)[..., None]
    lse = torch.where(l > 0, torch.where(torch.isinf(m), torch.zeros_like(m), m) + torch.log(l.clamp_min(1e-30)),
                      torch.full_like(l, float("-inf")))
    return o, lse


def attend_torch(q, q_pos0: int, scale: float, blocks: list, dev):
    """attend_numpy on a GPU (the V100 as a KV host): the same algebra; blocks hold torch tensors (pinned host)."""
    import torch
    mq, hq, d = q.shape
    q = q.to(dev, torch.float32)
    m = torch.full((mq, hq), float("-inf"), device=dev)
    l = torch.zeros((mq, hq), device=dev)
    acc = torch.zeros((mq, hq, d), device=dev)
    qpos = q_pos0 + torch.arange(mq, device=dev)
    for blk in blocks:
        seg = blk["row_seg"].to(dev, non_blocking=True).long()
        kv = []
        for side in ("k", "v"):
            ids = blk["ids_" + side].to(dev, non_blocking=True).to(torch.int32) & 0xFFFF
            kv.append(blk["min_" + side].to(dev)[seg] + ids.float() * blk["step_" + side].to(dev)[seg])
        k, v = kv
        b, h, _ = k.shape
        g = hq // h
        kpos = int(blk["a"]) + torch.arange(b, device=dev)
        mask = kpos[None, :] > qpos[:, None]
        if bool(mask.all()):
            continue
        s = torch.einsum("qhgd,bhd->qhgb", q.reshape(mq, h, g, d), k) * scale
        s = s.masked_fill(mask[:, None, None, :], float("-inf"))
        mnew = torch.maximum(m, s.amax(dim=-1).reshape(mq, hq))
        safe = torch.where(torch.isinf(mnew), torch.zeros_like(mnew), mnew)
        p = torch.exp(s - safe.reshape(mq, h, g)[..., None])
        a_old = torch.where(torch.isinf(m), torch.zeros_like(m), torch.exp(m - safe))
        l = l * a_old + p.sum(dim=-1).reshape(mq, hq)
        acc = acc * a_old[..., None] + torch.einsum("qhgb,bhd->qhgd", p, v).reshape(mq, hq, d)
        m = mnew
    o = acc / l.clamp_min(1e-30)[..., None]
    lse = torch.where(l > 0, torch.where(torch.isinf(m), torch.zeros_like(m), m) + torch.log(l.clamp_min(1e-30)),
                      torch.full_like(l, float("-inf")))
    return o, lse


# ---- building a block on the main machine (torch) ---------------------------------------------------------------------
def build_block(sk: list, sv: list, lens: list, a: int, b: int, dev) -> dict:
    """Tokens [a, b) of one layer and slot from its packed segments (lists over segments of per-head BandedKV) ->
    numpy arrays for the wire: exact ids (uint16, as int16 bit patterns) and each covered segment's (min, step)."""
    import torch
    from .codec import _unpack_dev
    h = len(sk[0])
    d = sk[0][0].orig_shape[1]
    ids = {s: torch.empty((b - a, h, d), dtype=torch.int16, device=dev) for s in ("k", "v")}
    row_seg = torch.empty(b - a, dtype=torch.int32)
    mins = {"k": [], "v": []}
    steps = {"k": [], "v": []}
    pos, j = 0, 0
    for si, sl in enumerate(lens):
        s0, s1 = max(pos, a), min(pos + sl, b)
        if s0 < s1:
            for side, store in (("k", sk), ("v", sv)):
                mn, st = [], []
                for hi in range(h):
                    bk = store[si][hi]
                    full = _unpack_dev(bk.packed.to(dev), bk.bits, sl * d).reshape(sl, d)
                    ids[side][s0 - a:s1 - a, hi] = full[s0 - pos:s1 - pos].to(torch.int32).to(torch.int16)
                    mn.append(bk.min_s.to(dev, torch.float32))
                    st.append(bk.tau_s.to(dev, torch.float32))
                mins[side].append(torch.stack(mn))
                steps[side].append(torch.stack(st))
            row_seg[s0 - a:s1 - a] = j
            j += 1
        pos += sl
        if pos >= b:
            break
    out = {"row_seg": row_seg.numpy()}
    for side in ("k", "v"):
        out["ids_" + side] = ids[side].cpu().numpy()
        out["min_" + side] = torch.stack(mins[side]).cpu().numpy()
        out["step_" + side] = torch.stack(steps[side]).cpu().numpy()
    return out


# ---- client (torch side) ----------------------------------------------------------------------------------------------
class RemoteCold:
    """One KV host (URL). Thread-safe; one keep-alive connection per calling thread."""

    def __init__(self, url: str, token: str = ""):
        import queue
        import threading
        from urllib.parse import urlsplit
        u = urlsplit(url)
        self.url, self.host, self.port, self.token = url, u.hostname, u.port or 80, token
        self._tls = threading.local()
        self._q = queue.Queue()
        threading.Thread(target=self._ship_loop, daemon=True, name="kv-host-ship").start()

    def _conn(self):
        import http.client
        c = getattr(self._tls, "c", None)
        if c is None:
            import socket
            from qv_band.common import tls as _qtls
            c = self._tls.c = _qtls.http_connection(self.host, self.port, 600)
            c.connect()
            # http.client sends the headers and the body as two writes: with Nagle's algorithm the second waits for
            # the ACK of the first, which the host delays (~40 ms) -- per layer, per step
            c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return c

    def _post(self, path: str, body: bytes) -> bytes:
        hdr = {"Content-Type": "application/octet-stream", "Content-Length": str(len(body))}
        if self.token:
            hdr["X-QV-Pool-Token"] = self.token
        for attempt in (0, 1):                       # a keep-alive connection the host closed: reconnect once
            c = self._conn()
            try:
                c.request("POST", path, body, hdr)
                r = c.getresponse()
                data = r.read()
                if r.status != 200:
                    raise RuntimeError(f"kv host {path.split('?')[0]}: HTTP {r.status} {data[:200]!r}")
                return data
            except (ConnectionError, OSError) as e:
                c.close()
                self._tls.c = None
                if attempt:
                    raise RuntimeError(f"kv host {self.url}: {e}") from e
        raise AssertionError

    # shipping: blocks go out in the background, in order; `done` is called once the host holds the block
    def ship(self, sid: str, a: int, b: int, arrays: dict, done) -> None:
        self._q.put((sid, a, b, arrays, done))

    def _ship_loop(self):
        while True:
            item = self._q.get()
            sid, a, b, arrays, done = item
            try:
                self._post(f"/v1/kv/put?sid={sid}&a={a}&b={b}", encode(arrays))
                done(True)
            except Exception as e:                  # noqa: BLE001 - the block stays local; report once per block
                print(f"[kv-host] shipping block {a}-{b} failed ({e}); it stays in local memory", flush=True)
                done(False)
            del item, sid, a, b, arrays, done

    def attend(self, sid: str, q_np: np.ndarray, q_pos0: int, scale: float) -> tuple:
        """q as the caller holds it (float16 from a 16-bit model: half the bytes on the link); the answer's o comes
        back in the same width (a 512-row prefill chunk of 16 heads x 256 is 8 MB in fp32), lse in fp32."""
        data = self._post(f"/v1/kv/attend?sid={sid}",
                          encode({"q": q_np}, {"q_pos0": int(q_pos0), "scale": float(scale), "o": str(q_np.dtype)}))
        arr, _ = decode(data)
        return arr["o"], arr["lse"]

    def free(self, sid: str) -> None:
        try:
            self._post(f"/v1/kv/free?sid={sid}", b"")
        except Exception:                           # noqa: BLE001 - the host drops idle sessions itself
            pass


def landed_callback(owner, fn):
    """A ship() callback that holds `owner` (a layer's state) WEAKLY: fn(owner, ok) runs only while owner is alive, and a
    queued or finished shipment never keeps it -- its finalizer frees the host session when the layer's state goes."""
    import weakref
    ref = weakref.ref(owner)

    def done(ok):
        o = ref()
        if o is not None:
            fn(o, ok)
    return done


_CLIENT = None


def client_from_env():
    """QV_KV_REMOTE=http://host:port (and QV_POOL_TOKEN): the KV host for cold blocks, or None."""
    import os
    global _CLIENT
    url = os.environ.get("QV_KV_REMOTE", "").strip()
    if not url:
        return None
    if _CLIENT is None or _CLIENT.url != url:
        _CLIENT = RemoteCold(url, os.environ.get("QV_POOL_TOKEN", ""))
    return _CLIENT
