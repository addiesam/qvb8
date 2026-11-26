"""Tensor Parallel (TP) Qwen3.5 / Qwen3.8 (hybrid Gated-DeltaNet + gated attention), QVB8 packs.

Stage 3 of design_doc/tensor_parallel.md: the whole model split by heads over n cards, one
user, plain HF forward (no scheduler, no CUDA graphs yet).

    torchrun --nproc-per-node 2 -m qv_band.run.tp_qwen35 --model PACK --prompt "..." --new 32
    python -m qv_band.run.tp_qwen35 --model PACK ...            # world 1: the same code on one card

Each card holds 1/n of every linear in VRAM:
  column (output rows): attention q (12 of 24 heads, q|gate interleaved per head), k, v (2 of 4
      KV heads); DeltaNet in_proj_qkv (q, k, v head ranges), z, b, a; MLP gate, up; lm_head (vocab,
      gathered).
  row (input columns), one all-reduce each: attention o_proj, DeltaNet out_proj, MLP down.
  sliced dense: conv1d channels, A_log, dt_bias. Replicated: norms, embedding (host RAM).
Each table is read on the CPU, cut, and only this card's share is moved to the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
import torch.nn as nn
from safetensors import safe_open

from qv_band.formats.qvb8.embedding import QVBandEmbeddingV8
from qv_band.formats.qvb8.module import QVBandLinearV8
from qv_band.formats.qvb8r.to_qvb8 import load_linear_any
from qv_band.formats.qvb8.tp import (ColumnParallelLinear, RowParallelLinear, TPGroup, split_even,
                                     split_weighted, split_ceil)

PFX = "model.language_model."


def _hf_name(pack_key: str) -> str | None:
    if pack_key.startswith(PFX):
        return "model." + pack_key[len(PFX):]
    if pack_key.startswith("lm_head."):
        return pack_key
    return None                                          # vision tower etc.: not used here


def _get(model: nn.Module, dotted: str):
    m = model
    for p in dotted.split("."):
        m = getattr(m, p)
    return m


def _set(model: nn.Module, dotted: str, value):
    parent, _, leaf = dotted.rpartition(".")
    setattr(_get(model, parent) if parent else model, leaf, value)


class Plan:
    """Which rows / columns / channels / heads of every tensor this rank keeps."""

    def __init__(self, cfg, tp: TPGroup, weights=None):
        """weights: one per rank (e.g. (3, 1)); None = even. Cuts follow head boundaries:
        attention by KV head (its query heads follow), DeltaNet by key head (its value heads
        follow), MLP by 128-wide blocks; lm_head stays even (gathered)."""
        w, r = tp.world, tp.rank
        weights = tuple(weights) if weights else (1,) * w
        if len(weights) != w:
            raise ValueError(f"{len(weights)} weights for {w} cards")
        self.tp, self.weights = tp, weights
        hd = cfg.head_dim
        nq, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        g, gv = nq // nkv, nv // nk                      # query heads per KV head, value heads per key head
        k0, k1 = split_weighted(nkv, weights, r)
        q0, q1 = k0 * g, k1 * g
        a0, a1 = split_weighted(nk, weights, r)
        v0, v1 = a0 * gv, a1 * gv
        self.nq, self.nkv, self.nk, self.nv = q1 - q0, k1 - k0, a1 - a0, v1 - v0
        self.attn_q = [(q0 * 2 * hd, q1 * 2 * hd)]       # per head: [q(hd) | gate(hd)]
        self.attn_kv = [(k0 * hd, k1 * hd)]
        self.attn_o = (q0 * hd, q1 * hd)
        K, V = nk * dk, nv * dv                          # v head j uses k head j // gv: blocks line up
        self.gdn_qkv = [(a0 * dk, a1 * dk), (K + a0 * dk, K + a1 * dk), (2 * K + v0 * dv, 2 * K + v1 * dv)]
        self.gdn_z = [(v0 * dv, v1 * dv)]
        self.gdn_ba = [(v0, v1)]
        self.gdn_heads = (v0, v1)
        self.gdn_out = (v0 * dv, v1 * dv)
        self.key_dim, self.value_dim = (a1 - a0) * dk, (v1 - v0) * dv
        self.moe = int(getattr(cfg, "num_experts", 0) or 0)     # MoE: experts by whole expert (tp_moe.py)
        if not self.moe:
            m = split_weighted(cfg.intermediate_size, weights, r, unit=128)
            self.mlp_rows, self.mlp_down = [m], m
        else:
            self.mlp_rows = self.mlp_down = None
        self.vocab = [split_ceil(cfg.vocab_size, w, r)]   # gathered: any world size

    def linear(self, name: str):
        """('col', rows) | ('row', (c0, c1)) | ('col_gather', rows) for a linear's HF name."""
        leaf = name.rsplit(".", 2)[-2] if name.endswith(".weight") else name.rsplit(".", 1)[-1]
        if name.startswith("lm_head"):
            return "col_gather", self.vocab
        if ".self_attn." in name:
            return {"q_proj": ("col", self.attn_q), "k_proj": ("col", self.attn_kv),
                    "v_proj": ("col", self.attn_kv), "o_proj": ("row", self.attn_o)}[leaf]
        if ".linear_attn." in name:
            return {"in_proj_qkv": ("col", self.gdn_qkv), "in_proj_z": ("col", self.gdn_z),
                    "in_proj_b": ("col", self.gdn_ba), "in_proj_a": ("col", self.gdn_ba),
                    "out_proj": ("row", self.gdn_out)}[leaf]
        if ".mlp." in name:
            return {"gate_proj": ("col", self.mlp_rows), "up_proj": ("col", self.mlp_rows),
                    "down_proj": ("row", self.mlp_down)}[leaf]
        raise KeyError(name)

    def dense(self, name: str, t: torch.Tensor) -> torch.Tensor:
        """Cut a dense (safetensors) tensor: DeltaNet conv channels and per-head params."""
        if ".linear_attn.conv1d." in name:
            return torch.cat([t[a:b] for a, b in self.gdn_qkv], 0)
        if name.endswith(".linear_attn.A_log") or name.endswith(".linear_attn.dt_bias"):
            a, b = self.gdn_heads
            return t[a:b]
        return t


EMB_PLACE = os.environ.get("QV_TP_EMB", "vram")       # vram (default): a copy on every card; ram: one per machine


def _shared_embedding(path, dtype, tp, device=None):
    """The (replicated) embedding. Default (QV_TP_EMB=vram): the packed table on each card, rows looked up there
    (fastest: no host round trip; ~1 GB per card for a 248k x 5120 QVB8 table). QV_TP_EMB=ram: ONE copy in host
    RAM per machine (card 0 decodes the pack into /dev/shm -- RAM, never disk; every card maps the same pages)."""
    emb = None
    if EMB_PLACE == "vram" and device is not None and not path.endswith(".qvb8r"):
        return QVBandEmbeddingV8.from_packfile(path, out_dtype=dtype, device=device)
    shm = "/dev/shm/qvemb_" + hashlib.sha1(f"{os.path.realpath(path)}|{os.path.getmtime(path)}|{dtype}".encode()
                                           ).hexdigest()[:16]
    if path.endswith(".qvb8r"):                   # rANS-coded embedding: its own CPU reader, one copy per card
        from qv_band.formats.qvb8r.module import QVBandEmbeddingV8R
        return QVBandEmbeddingV8R.from_packfile(path, out_dtype=dtype, device="cpu")
    # one copy per MACHINE: its first card writes it (several machines: each has its own /dev/shm)
    local, local_world = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("LOCAL_WORLD_SIZE", tp.world))
    if tp.world <= 1:
        return QVBandEmbeddingV8.from_packfile(path, out_dtype=dtype, device="cpu")
    if local_world <= 1:                          # alone on this machine: nothing to share; still meet the barriers
        torch.distributed.barrier(group=tp.group)
        emb = QVBandEmbeddingV8.from_packfile(path, out_dtype=dtype, device="cpu")
        torch.distributed.barrier(group=tp.group)
        return emb
    if local == 0:
        emb = QVBandEmbeddingV8.from_packfile(path, out_dtype=dtype, device="cpu")
        meta = {}
        for name, b in emb.named_buffers():
            f = f"{shm}.{name}"
            nb = b.numel() * b.element_size()
            m = torch.from_file(f, shared=True, size=max(nb, 1), dtype=torch.uint8)
            if nb:
                m.copy_(b.contiguous().view(-1).view(torch.uint8))
            meta[name] = (str(b.dtype).split(".")[-1], list(b.shape))
        meta = {"buffers": meta, "Q": emb.num_embeddings, "P": emb.embedding_dim}
        with open(shm + ".json", "w") as fh:
            json.dump(meta, fh)
    torch.distributed.barrier(group=tp.group)
    meta = json.load(open(shm + ".json"))
    if emb is None:
        emb = QVBandEmbeddingV8.__new__(QVBandEmbeddingV8)
        torch.nn.Module.__init__(emb)
        emb.num_embeddings, emb.embedding_dim, emb._out_dtype = meta["Q"], meta["P"], dtype
    for name, (dt, shape) in meta["buffers"].items():
        dt = getattr(torch, dt)
        n = 1
        for x in shape:
            n *= x
        nb = n * torch.empty((), dtype=dt).element_size()
        m = torch.from_file(f"{shm}.{name}", shared=True, size=max(nb, 1), dtype=torch.uint8)
        t = m[:nb].view(dt).view(shape) if nb else torch.empty(shape, dtype=dt)
        emb.register_buffer(name, t, persistent=False)     # card 0's private copy is freed here
    torch.distributed.barrier(group=tp.group)
    return emb


def _plain_conv1d(M):
    """The causal_conv1d wheel has no sm_70 build: use plain torch (same math as HF's fallback)."""
    import torch.nn.functional as F
    from transformers.activations import ACT2FN

    def update(h, conv_state, weight, bias=None, activation=None):
        _, hs, s = h.shape
        n = conv_state.shape[-1]
        hn = torch.cat([conv_state, h], dim=-1).to(weight.dtype)
        conv_state.copy_(hn[:, :, -n:])
        out = F.conv1d(hn, weight.unsqueeze(1), bias, padding=0, groups=hs)[:, :, -s:]
        return (ACT2FN[activation](out) if activation else out).to(h.dtype)

    def fn(h, weight, bias=None, activation=None, **kw):
        _, hs, s = h.shape
        out = F.conv1d(h.to(weight.dtype), weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1,
                       groups=hs)[:, :, :s]
        return (ACT2FN[activation](out) if activation else out).to(h.dtype)

    M.causal_conv1d_update, M.causal_conv1d_fn = update, fn


def _linear_share(path, kind, sel, tp, device):
    """This card's share of one linear, wrapped for its parallel kind (run/weight_load.py: every format)."""
    from qv_band.run.weight_load import load_weight as share_local
    # every format through the one loader (.qvb8r: cut in place; .qvband: a plain slice) -- so pre-cut share files
    # (QV_TP_SHARD_WRITE / QV_TP_SHARD_DIR) serve every pack
    if kind == "row":
        return RowParallelLinear(share_local(path, cols=sel, device=device), tp, _in_features(path))
    local = share_local(path, rows=sel, device=device)
    return ColumnParallelLinear(local, tp, _out_features(path), gather=(kind == "col_gather"))


def _hdr(path):
    from qv_band.formats.qvb8r import container as QF
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    import json as _j
    import numpy as _np
    hl = int(_np.frombuffer(head, _np.uint32, 1, 8)[0])
    return _j.loads(head[12:12 + hl]) if hl <= len(head) - 12 else QF.read(path)


_MAN = {}


def _qp(path):
    """(Q, P) of a weight file: the .qvb8r header, else the pack manifest beside it (cached per pack)."""
    if path.endswith(".qvb8r"):
        h = _hdr(path)
        return int(h["Q"]), int(h["P"])
    import json as _j
    d, base = os.path.dirname(path), os.path.basename(path)
    if d not in _MAN:
        _MAN[d] = {e["pack"]: (int(e["Q"]), int(e["P"])) for e in
                   _j.load(open(os.path.join(d, "manifest.json")))["tensors"].values() if "Q" in e and "P" in e}
    return _MAN[d][base]


def _in_features(path):
    return _qp(path)[1]


def _out_features(path):
    return _qp(path)[0]


def load_config(pack: str):
    """The pack's transformers config. A sensenova_loop pack (remote-code model_type) is a Qwen3.5 hybrid with a loop: its
    text_config IS a Qwen3.5 text config, so it loads as one; the raw dict is kept in cfg._raw (loop_settings, mhc)."""
    import json
    from transformers import AutoConfig
    raw = json.load(open(os.path.join(pack, "config.json")))
    if raw.get("model_type") == "sensenova_loop":
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        keep = ("text_config", "vision_config", "image_token_id", "video_token_id", "vision_start_token_id",
                "vision_end_token_id", "tie_word_embeddings")
        cfg = Qwen3_5Config(**{k: v for k, v in raw.items() if k in keep})
        cfg._raw = raw
        return cfg
    return AutoConfig.from_pretrained(pack, local_files_only=True)


def install_adapter(model, pack: str, cfg) -> bool:
    """Run the model adapter's install() on a TP-built model when the pack has one that rebuilds the layers (SenseNova's loop:
    unrolled execution slots + mHC streams). The layers are already this card's shares; their weights are shared by the slots."""
    raw = getattr(cfg, "_raw", None)
    if not raw or raw.get("model_type") != "sensenova_loop":
        return False
    import types
    from qv_band.models.sensenova.adapter import ADAPTER
    model._qv_tp = True
    ADAPTER.install(model, model_dir=pack, config=types.SimpleNamespace(**raw), tok=None)
    return True


def family(cfg):
    """(modeling module, CausalLM class) for a Qwen3.5-family config: dense or MoE (same hybrid layer stack)."""
    t = getattr(cfg, "text_config", cfg)
    if "moe" in (getattr(t, "model_type", "") or "") or getattr(t, "num_experts", 0):
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
        return M, M.Qwen3_5MoeForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    return M, M.Qwen3_5ForCausalLM


def _stage_keep(name: str, stage, n_layers: int, embed_too: bool = False) -> bool:
    """Pipeline stage (lo, hi): does this rank load tensor `name`? Layers lo..hi-1; the embedding only on the
    first stage (lo == 0), the final norm and lm_head only on the last (hi == n_layers)."""
    if stage is None:
        return True
    lo, hi = stage
    if name.startswith("model.layers."):
        i = int(name.split(".")[2])
        return lo <= i < hi
    if "embed_tokens" in name:
        return lo == 0 or embed_too
    if name.startswith("lm_head") or name.startswith("model.norm."):
        return hi == n_layers
    return True


def build(pack: str, tp: TPGroup, device: str, dtype=torch.float16, verbose=True, weights=None, stage=None,
          stage_embed: bool = False):
    """stage=(lo, hi): pipeline parallel -- only layers lo..hi-1 are built on this rank (the others become
    nn.Identity placeholders); see run/tp_pipe.py."""
    from transformers import AutoConfig

    if torch.cuda.get_device_capability(device)[0] < 8:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M0
        _plain_conv1d(M0)
        try:
            from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M1
            _plain_conv1d(M1)
        except Exception:                      # noqa: BLE001
            pass
        try:                                   # V100: fla's chunked gated-delta Triton kernels -> WMMA CUDA versions
            from qv_band.kv.gdn_volta import install as _gdn_volta
            with torch.cuda.device(device):
                _gdn_volta()
        except Exception as e:                 # noqa: BLE001 -- prefill still runs on fla
            print(f"[tp] gdn_volta not installed: {e!r}", flush=True)

    cfg = load_config(pack)
    tcfg = getattr(cfg, "text_config", cfg)
    M, CausalLM = family(cfg)
    tcfg._attn_implementation = "sdpa"
    tcfg.torch_dtype = dtype
    plan = Plan(tcfg, tp, weights)
    if verbose:
        mlp = (f"{plan.moe} experts (whole experts per card)" if plan.moe else
               f"MLP {plan.mlp_down[1] - plan.mlp_down[0]} of {tcfg.intermediate_size}")
        print(f"[tp{tp.rank}] share: {plan.nkv} KV heads, {plan.nk} DeltaNet key heads, {mlp}", flush=True)
    with torch.device("meta"):
        model = CausalLM(tcfg)
    NL = tcfg.num_hidden_layers
    if stage is not None:
        if plan.moe:
            raise NotImplementedError("pipeline stages: dense models only for now")
        lo, hi = stage
        for i in range(NL):
            if not lo <= i < hi:
                model.model.layers[i] = nn.Identity()
        if lo != 0 and not stage_embed:                  # stage_embed: the last stage's drafter embeds too
            model.model.embed_tokens = nn.Identity()
        if hi != NL:
            model.model.norm = nn.Identity()
            model.lm_head = nn.Identity()

    # per-module head counts (HF reshapes use -1 for the rest)
    for layer in model.model.layers:
        g = getattr(layer, "linear_attn", None)
        if g is not None:
            g.num_k_heads, g.num_v_heads = plan.nk, plan.nv
            g.key_dim, g.value_dim = plan.key_dim, plan.value_dim
            g.conv_dim = 2 * plan.key_dim + plan.value_dim

    man = json.load(open(os.path.join(pack, "manifest.json")))["tensors"]
    t0, n, dev_bytes = time.time(), 0, 0
    jobs = []                                         # (module name, file, kind, selection): loaded in parallel below
    for key, ent in man.items():
        name = _hf_name(key)
        if name is None:
            continue
        if plan.moe and ".mlp." in name:                 # MoE blocks: tp_moe.build_moe_blocks below
            continue
        if not _stage_keep(name, stage, NL, stage_embed):
            continue
        mod_name = name[:-len(".weight")]
        path = os.path.join(pack, ent["pack"])
        if mod_name.endswith("embed_tokens"):
            _set(model, mod_name, _shared_embedding(path, dtype, tp, device))
            continue
        kind, sel = plan.linear(name)
        jobs.append((mod_name, path, kind, sel))
    from qv_band.run.weight_load import load_parallel as run_parallel, warm, workers
    warm([device])
    # QVB8R is cut in place from the stream; every share is loaded on QV_LOAD_WORKERS threads, set in order
    mods = run_parallel(lambda j: _linear_share(j[1], j[2], j[3], tp, device), jobs)
    for (mod_name, _, _, _), m in zip(jobs, mods):
        dev_bytes += sum(b.numel() * b.element_size() for b in m.buffers() if b is not None)
        _set(model, mod_name, m)
        n += 1
    if verbose:
        print(f"[tp{tp.rank}] {n} linears, {dev_bytes / 1e9:.2f} GB on card, {time.time() - t0:.0f}s "
              f"({workers()} loader threads)", flush=True)

    if plan.moe:
        from qv_band.run.tp_moe import build_moe_blocks
        build_moe_blocks(model, pack, man, tcfg, tp, device, weights, prefix=PFX, verbose=verbose)

    with safe_open(os.path.join(pack, "model.safetensors"), "pt") as f:
        for key in f.keys():
            name = _hf_name(key)
            if name is None or not _stage_keep(name, stage, NL, stage_embed):
                continue
            try:
                parent, _, leaf = name.rpartition(".")
                owner = _get(model, parent)
            except AttributeError:
                continue
            if not isinstance(getattr(owner, leaf, None), torch.Tensor):
                continue
            t = plan.dense(name, f.get_tensor(key))
            is_float = t.is_floating_point() and not name.endswith("A_log")
            t = t.to(device=device, dtype=dtype if is_float else t.dtype)
            setattr(owner, leaf, nn.Parameter(t, requires_grad=False))

    with torch.device(device):
        model.model.rotary_emb = type(model.model.rotary_emb)(tcfg)
    left = [k for k, v in list(model.named_parameters()) + list(model.named_buffers()) if v.is_meta]
    if left:
        raise RuntimeError(f"still on meta: {left[:10]} ... ({len(left)})")
    model.eval()
    if verbose:
        print(f"[tp{tp.rank}] loaded {n} linears in {time.time() - t0:.0f}s; "
              f"VRAM {torch.cuda.memory_allocated(device) / 1e9:.2f} GB", flush=True)
    install_adapter(model, pack, cfg)                  # SenseNova: unrolled loop slots + mHC streams (no-op for other packs)
    return model, tcfg


@torch.no_grad()
def greedy(model, tok, prompt: str, new: int, device: str, tp: TPGroup):
    from transformers import DynamicCache
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    cache = DynamicCache(config=model.config)
    out, first_logits = [], None
    torch.cuda.synchronize(device)
    t0 = time.time()
    x = ids
    t_first = None
    for i in range(new):
        logits = model(input_ids=x, past_key_values=cache, use_cache=True,
                       logits_to_keep=1).logits[:, -1].float()
        if first_logits is None:
            first_logits = logits.clone()
        nxt = logits.argmax(-1, keepdim=True)
        if tp.world > 1:                                 # every rank must feed the same token
            torch.distributed.broadcast(nxt, 0, group=tp.group)
        out.append(int(nxt))
        x = nxt
        if i == 0:
            torch.cuda.synchronize(device)
            t_first = time.time()
    torch.cuda.synchronize(device)
    dt = time.time() - t_first
    return out, first_logits, ids.shape[1], t_first - t0, (new - 1) / dt if new > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--tp-weights", default=None, help="share per card, e.g. 3,1 (default even)")
    ap.add_argument("--dump", default=None, help="save tokens + first-step logits (rank 0)")
    a = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        torch.distributed.init_process_group(
            "nccl", device_id=torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"))
    tp = TPGroup.from_dist()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local}"
    torch.cuda.set_device(device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    weights = [float(x) for x in a.tp_weights.split(",")] if a.tp_weights else None
    model, _ = build(a.model, tp, device, weights=weights)
    if tp.world > 1:
        torch.distributed.barrier()
    greedy(model, tok, "Hi", 4, device, tp)              # warm-up
    out, logits, n_in, t_prefill, tps = greedy(model, tok, a.prompt, a.new, device, tp)
    if tp.rank == 0:
        print(f"[tp] world {tp.world}: prompt {n_in} tokens, prefill {t_prefill:.2f}s, "
              f"decode {tps:.2f} tok/s, peak VRAM {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        print("[tp] text:", repr(tok.decode(out)))
        if a.dump:
            torch.save({"tokens": out, "logits": logits.cpu()}, a.dump)
    if tp.world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
