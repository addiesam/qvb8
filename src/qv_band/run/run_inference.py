"""Run a text prompt through the real model with EVERY weight loaded from a
pre-calibrated QV-band pack directory (.qvband), HF-style, and report
peak runtime VRAM + tokens/s.

This is the INFERENCE half of the split pipeline (calibration/pack is done
separately by `python -m qv_band.run_calibrate_model`).  Following
src_tau_comp/run_hf_adapter.py:

  1. build the model skeleton on `meta` (no weights allocated)
  2. for every tensor in the pack-dir manifest, swap the owning module for
     the linear/embedding class the manifest's "format" names (looked up in
     qv_band.common.formats), loaded straight from its pack file
  3. stream all remaining (dense) tensors from safetensors into VRAM
  4. generate, reporting weights-resident and transient-peak VRAM + tok/s

  PYTHONPATH=. python -m qv_band.run.run_inference \
      --model /mnt/backup/models/Qwen3.5-0.8B \
      --pack-dir /mnt/data/models/Qwen3.5-9B_qvb8_fp16_1e-3 \
      --prompt "Explain entropy in one sentence." --max-new-tokens 128
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn

from qv_band.formats.qvb8.module import QVBandLinearV8
from qv_band.formats.qvb8.embedding import QVBandEmbeddingV8
from qv_band.common.errors import UncertifiableTensor
from qv_band.run.common import tensor_key_to_module, _resolve, make_resolve_key
# A routed MoE expert's manifest key is "<base_tensor>#expert<e>" (e.g.
# "model.language_model.layers.0.mlp.experts.gate_up_proj#expert003"),
# addressing one [out, in] slice of the checkpoint's 3-D [E, out, in] stack.
# Both this module's swap loop and the calibration drivers that write these
# keys (qvb8.calibration.pack_model)
# need the same separator; qv_band.common.expert_key holds the one
# definition, and importing it here does not create a calibration
# dependency -- it is a plain string constant, not calibration logic.
from qv_band.common.expert_key import EXPERT_KEY_SEP as _EXPERT_KEY_SEP


def _parse_expert_key(tensor_key: str):
    """Split "<base>#expert<idx>" -> (base_key, expert_idx) or (tensor_key, None)."""
    if _EXPERT_KEY_SEP not in tensor_key:
        return tensor_key, None
    base, _, idx_str = tensor_key.partition(_EXPERT_KEY_SEP)
    return base, int(idx_str)


def _backend(device: str):
    """torch.cuda or torch.xpu, whichever owns `device` ('cuda'/'cuda:N' or
    'xpu'/'xpu:N'). Both expose the same memory-stats/sync API surface used
    below, so callers can stay device-agnostic."""
    return torch.xpu if device.startswith("xpu") else torch.cuda


def _bind_packed_weight_forward(owner: nn.Module, packed: "QVBandLinearV8") -> None:
    """Rebind `owner`'s forward to route its matmul through `packed` instead
    of `owner.weight` (a bare, now-stale meta nn.Parameter this swap loop
    never allocates real data for -- see the "F.linear(x, self.weight)"
    fallback branch above that calls this).

    Deliberately specific to Qwen3_5MoeTopKRouter's exact forward (the only
    module shape this fallback branch currently matches: a bare 2-D
    nn.Parameter named `weight`, no bias -- the MoE gate/router). A generic
    monkeypatch of every possible "read self.weight" call site is not
    attempted; if a future model surfaces a DIFFERENT such module, this
    function should gain a shape/attribute check and its own branch rather
    than silently reusing router math for something that is not a router.
    """
    if not (hasattr(owner, "top_k") and hasattr(owner, "num_experts")
            and hasattr(owner, "hidden_dim")):
        raise TypeError(
            f"_bind_packed_weight_forward: {type(owner).__name__} does not "
            f"look like Qwen3_5MoeTopKRouter (missing top_k/num_experts/"
            f"hidden_dim) -- this fallback only knows the router's forward; "
            f"add a shape check + dedicated branch for this module type "
            f"instead of assuming router semantics.")
    owner.add_module("_qv_packed_weight", packed)
    # Remove the stale meta Parameter so nothing can accidentally read
    # garbage through it (e.g. a state_dict() call, or code that checks
    # `hasattr(owner, "weight")` for a different reason than this forward).
    if "weight" in owner._parameters:
        del owner._parameters["weight"]

    def _router_forward(self, hidden_states):
        import torch.nn.functional as F
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self._qv_packed_weight(hidden_states)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices

    owner.forward = _router_forward.__get__(owner, type(owner))




def _setattr_dotted(model: nn.Module, dotted: str, value):
    parent = _resolve(model, dotted.rsplit(".", 1)[0]) if "." in dotted else model
    setattr(parent, dotted.rsplit(".", 1)[-1], value)


def _load_dense_from_source(manifest: dict, tensor_key: str, module: str,
                            model: nn.Module, device: str) -> bool:
    """Load ONE tensor by its checkpoint key straight from
    manifest['hf_model'] and set it into `model` at `module`.weight --
    the per-tensor equivalent of _repair_pack_dir_missing_dense, used
    when a SPECIFIC tensor is discovered to need dense fallback mid-load
    (e.g. UncertifiableTensor at load time) rather than the whole
    model.safetensors bundle being absent. Without this, a tensor
    refused after packing believed it succeeded (so
    _bundle_dense_tensors excluded it from the dense bundle) has no
    fallback data anywhere on disk and is left on `meta` forever --
    the exact crash this function exists to prevent. Returns True if
    the tensor was found and loaded, False if the source checkpoint
    does not have it (caller should then treat this as a hard failure,
    not silently continue past a still-meta parameter).
    """
    base_dir = manifest.get("hf_model")
    if not base_dir or not os.path.isdir(base_dir):
        return False
    from safetensors import safe_open
    from accelerate.utils import set_module_tensor_to_device
    idx_path = os.path.join(base_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        weight_map = json.load(open(idx_path))["weight_map"]
        if tensor_key not in weight_map:
            return False
        shard = weight_map[tensor_key]
    else:
        shard = None
        for p in os.listdir(base_dir):
            if not p.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(base_dir, p), framework="pt", device="cpu") as f:
                if tensor_key in f.keys():
                    shard = p
                    break
        if shard is None:
            return False
    with safe_open(os.path.join(base_dir, shard), framework="pt", device="cpu") as f:
        value = f.get_tensor(tensor_key)
    set_module_tensor_to_device(model, f"{module}.weight", device,
                                value=value.to(device))
    return True


def _repair_pack_dir_missing_dense(pack_dir: str, manifest: dict,
                                   skip_weights: set, resolve_key) -> None:
    """Write pack_dir/model.safetensors from manifest['hf_model'] if absent.

    A pack dir is supposed to ship its own model.safetensors holding
    every tensor the .qvband packs do NOT cover -- layernorm weight, linear-attn
    dt_bias/A_log/conv1d.weight, and anything else that isn't a plain
    Linear/Embedding matmul weight. Some packs were built without copying it
    (see manifest['hf_model'] -- the dense checkpoint the pack was calibrated
    from is always recorded there, so it never has to be re-typed by hand).
    Runs once per broken pack dir: after this, the file is on disk and every
    future load of this pack finds it immediately via the normal
    has_dense_here check, no repair step re-entered.

    `skip_weights` (module-dotted names, already computed by the swap loop
    above) is the source of truth for "actually packed", NOT
    manifest['tensors'].keys() -- the manifest can list a key as packed
    (e.g. linear-attn conv1d.weight, from an older pack-format attempt) even
    though the loader has no QVBand class for that module type and silently
    `continue`s past it without swapping or adding it to skip_weights. A
    repair that trusted the manifest's claim would omit that tensor from the
    rebuilt file too, leaving it stuck on `meta` in every future load -- the
    exact failure this function exists to fix, just moved one level down.
    """
    base_dir = manifest.get("hf_model")
    if not base_dir or not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"{pack_dir}/model.safetensors is missing and manifest.json's "
            f"'hf_model' ({base_dir!r}) is not a usable directory -- cannot "
            f"recover the un-packed tensors this pack needs.")

    from safetensors import safe_open
    from safetensors.torch import save_file

    idx_path = os.path.join(base_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        weight_map = json.load(open(idx_path))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = [p for p in os.listdir(base_dir) if p.endswith(".safetensors")]

    print(f">>> {pack_dir} has no model.safetensors -- rebuilding it once "
          f"from {base_dir} (only tensors actually swapped by a QVBand "
          f"module are skipped; everything else, including any manifest "
          f"entry the loader could not apply, is copied) ...", flush=True)
    t0 = time.time()
    tensors = {}
    for shard in shards:
        with safe_open(os.path.join(base_dir, shard), framework="pt",
                       device="cpu") as f:
            for key in f.keys():
                tgt = resolve_key(key)
                if tgt is not None and tgt in skip_weights:
                    continue
                tensors[key] = f.get_tensor(key)
    tmp_path = os.path.join(pack_dir, ".model.safetensors.tmp")
    save_file(tensors, tmp_path)
    os.replace(tmp_path, os.path.join(pack_dir, "model.safetensors"))
    print(f"    wrote model.safetensors: {len(tensors)} tensors, "
          f"{time.time() - t0:.0f}s -- {pack_dir} is now self-contained",
          flush=True)


def load_qv_model(model_dir: str, pack_dir: Optional[str], *,
                  no_compress: bool = False, device: str = "cuda"):
    """Build the model skeleton on meta, swap QV-band v7 anchored packs,
    stream the remaining dense tensors, and return (model, tokenizer).
    Extracted from main() so other harnesses (speculative decode, benchmarks)
    can reuse the exact deployed loading path."""
    class _A:  # minimal arg shim for the body below
        pass
    args = _A()
    args.model = model_dir
    args.pack_dir = pack_dir
    # pack_dir=None (config.py's LLM_PACK_DIR in "normal" mode) means dense.
    args.no_compress = no_compress or pack_dir is None
    args.device = device
    return _load_qv_model_impl(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir (local)")
    ap.add_argument("--pack-dir", help="calibrated QV-band pack dir "
                    "(manifest.json + *.qvband -- the manifest's "
                    "'format' picks the readers)")
    ap.add_argument("--prompt", default="Explain entropy in one sentence.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--no-compress", action="store_true",
                    help="stream everything dense (native dtype) -- VRAM/speed baseline")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for sampling (implies --do-sample)")
    ap.add_argument("--do-sample", action="store_true",
                    help="sample instead of greedy decoding")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--draft-model", default="auto",
                    help="DFlash / DFlash 2 draft model dir to decode "
                         "speculatively with (lossless -- the target verifies "
                         "every draft token). Default 'auto': the model's own "
                         "drafter if one is found (dflash_draft."
                         "find_dflash_draft), else plain decoding")
    ap.add_argument("--no-draft", action="store_true",
                    help="plain decoding, no speculative drafter")
    ap.add_argument("--draft-pack", default=None,
                    help="qvb8 pack of --draft-model (pack_model.py output): "
                         "load the drafter's linears packed")
    ap.add_argument("--spec-k", type=int, default=7,
                    help="draft tokens per verify step with --draft-model "
                         "(the drafter may propose fewer)")
    args = ap.parse_args()

    if not args.no_compress and not args.pack_dir:
        ap.error("--pack-dir is required unless --no-compress")
    if args.no_draft:
        args.draft_model = None
    elif args.draft_model == "auto":
        from qv_band.decode.dflash_draft import find_dflash_draft
        args.draft_model = find_dflash_draft(args.pack_dir or args.model)
        args.draft_auto = True       # found by name: a bad fit degrades, not dies
        print(f">>> drafter: {args.draft_model or 'none found -- plain decoding'}",
              flush=True)
    if args.draft_pack and not args.draft_model:
        ap.error("--draft-pack needs a draft model (the pack holds only the "
                 "draft's linears; norms and codebooks come from the checkpoint)")
    model, tok = _load_qv_model_impl(args)
    _generate_and_report(model, tok, args)


class _HostEmbedding(nn.Embedding):
    """A dense embedding table kept in host RAM, output moved to the input's device.

    A lookup is a gather of a few rows, so it gains nothing from VRAM, while a
    248k x 5120 bf16 table costs 2.5 GB on EVERY tensor-parallel rank (it is
    replicated, not sharded) -- the margin that decides whether a 27B model
    fits two 16 GB cards.
    """

    def forward(self, input_ids):
        out = nn.functional.embedding(input_ids.to(self.weight.device),
                                      self.weight)
        return out.to(input_ids.device)


from qv_band.common.xfer import to_device  # noqa: E402


def _install_pipeline_hooks(model, place) -> int:
    """Make each decoder layer run on the card its weights live on.

    Each layer's forward is wrapped: its tensor inputs (hidden states, the
    shared rotary cos/sin, masks, positions -- cache objects are left alone,
    their per-layer tensors already live with the layer) move to its card, and
    that card is CURRENT for the duration (`with torch.cuda.device`): the qvb8
    extension and the Triton kernels launch on the current device's stream,
    not on their tensors' device. The context manager restores the previous
    card even when the layer raises (an OOM, an abort) -- a forward post-hook
    does not run then, and the process used to stay on the first card for
    every later request. Inputs every layer shares are copied once per model
    forward: the copies are dropped when the next forward starts, so they never
    pin a previous request's masks / hidden states. Returns the layers wrapped."""
    layers = model.model.layers
    shared: dict = {}

    def _move(x, dev):
        if isinstance(x, torch.Tensor):
            if x.device == dev:
                return x
            key = (id(x), x._version, dev)
            hit = shared.get(key)
            if hit is None or hit[0] is not x:
                if len(shared) > 64:          # bound for callers that run layers without a model forward
                    shared.clear()
                hit = shared[key] = (x, to_device(x, dev))
            return hit[1]
        if isinstance(x, tuple):
            return tuple(_move(v, dev) for v in x)
        if isinstance(x, list):
            return [_move(v, dev) for v in x]
        return x

    n = 0
    for i, layer in enumerate(layers):
        dev = torch.device(place(f"model.layers.{i}.weight"))
        orig = layer.forward

        def fwd(*args, _orig=orig, _dev=dev, **kwargs):
            with torch.cuda.device(_dev):
                return _orig(*(_move(a, _dev) for a in args), **{k: _move(v, _dev) for k, v in kwargs.items()})

        layer.forward = fwd
        n += 1
    # A new model forward: the previous one's cross-card copies are dead.
    model.model.register_forward_pre_hook(lambda mod, args: shared.clear())
    # The final norm runs on the main card: bring the last hidden state home.
    norm = getattr(model.model, "norm", None)
    if norm is not None:
        ndev = torch.device(place("model.norm.weight"))
        norm.register_forward_pre_hook(
            lambda mod, args: tuple(_move(a, ndev) for a in args))
    return n


_HOST_EMBED_MIN_ROWS = 65536
_HOST_EMBED_ENABLED = os.environ.get("QV_HOST_EMBED", "0") == "1"


def _dequant_fp8_block(w, key, f, weight_map, model_dir, device, block):
    """float8_e4m3fn weight [R, C] and its [ceil(R/b0), ceil(C/b1)] `<key>_scale_inv` -> bfloat16 w * scale."""
    from safetensors import safe_open
    skey = key + "_scale_inv"
    if skey in set(f.keys()):
        scale = f.get_tensor(skey)
    else:
        with safe_open(os.path.join(model_dir, weight_map[skey]), framework="pt", device=device) as g:
            scale = g.get_tensor(skey)
    b0, b1 = int(block[0]), int(block[1])
    s = scale.float().repeat_interleave(b0, dim=0).repeat_interleave(b1, dim=1)[: w.shape[0], : w.shape[1]]
    return (w.float() * s).to(torch.bfloat16)


def _fp8_fused_experts_for_layer(model_dir, wm, layer, device, block, prefix="model.language_model.layers"):
    """The official FP8 MoE checkpoint stores every expert as separate FP8 matrices (experts.E.gate_proj / up_proj / down_proj, each with a
    block scale); the model holds them fused. Dequantise one layer's experts and stack them: gate_up_proj [E, 2I, H] (gate rows, then up
    rows) and down_proj [E, H, I], in bf16."""
    import re
    from safetensors import safe_open
    pre = f"{prefix}.{layer}.mlp.experts"
    es = sorted({int(m.group(1)) for k in wm for m in [re.match(re.escape(pre) + r"\.(\d+)\.gate_proj\.weight$", k)] if m})
    shard = os.path.join(model_dir, wm[f"{pre}.{es[0]}.gate_proj.weight"])
    gu = dn = None
    with safe_open(shard, framework="pt", device=device) as f:
        for e in es:
            g, u, d = (_dequant_fp8_block(f.get_tensor(f"{pre}.{e}.{p}.weight"), f"{pre}.{e}.{p}.weight", f, wm, model_dir, device, block)
                       for p in ("gate_proj", "up_proj", "down_proj"))
            if gu is None:
                gu = torch.empty((len(es), g.shape[0] + u.shape[0], g.shape[1]), dtype=torch.bfloat16, device=device)
                dn = torch.empty((len(es),) + tuple(d.shape), dtype=torch.bfloat16, device=device)
            gu[e, :g.shape[0]] = g; gu[e, g.shape[0]:] = u; dn[e] = d
    return gu, dn


def _stream_fp8_experts(model, model_dir, wm, device, block, resolve_key):
    """Fused experts for every layer of an official FP8 MoE checkpoint (see _fp8_fused_experts_for_layer); returns the number of layers."""
    import re
    from accelerate.utils import set_module_tensor_to_device
    layers = sorted({int(m.group(1)) for k in wm for m in [re.match(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight$", k)] if m})
    for L in layers:
        gu, dn = _fp8_fused_experts_for_layer(model_dir, wm, L, device, block)
        for suffix, val in (("gate_up_proj", gu), ("down_proj", dn)):
            tgt = resolve_key(f"model.language_model.layers.{L}.mlp.experts.{suffix}")
            if tgt is None:
                raise RuntimeError(f"no module parameter for layer {L} experts.{suffix}")
            set_module_tensor_to_device(model, tgt, device, value=val)
        del gu, dn
    return len(layers)


def _nemotron_eos_ids(model_dir, config, tok) -> set:
    """Every id that ends a reply: tokenizer, wrapper and text config, generation_config.json."""
    ids = set()
    def add(v):
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(x, int):
                ids.add(x)
    add(tok.eos_token_id); add(getattr(config, "eos_token_id", None))
    add(getattr(config.llm_config, "eos_token_id", None))
    try:
        add(json.load(open(os.path.join(model_dir, "generation_config.json"))).get("eos_token_id"))
    except OSError:
        pass
    return ids


def _load_qv_model_impl(args):
    device = args.device
    if not (torch.cuda.is_available() or
            (hasattr(torch, "xpu") and torch.xpu.is_available())):
        raise RuntimeError("inference requires a CUDA or XPU GPU")
    backend = _backend(device)

    # Tensor-parallel mode: this process was launched under torchrun (RANK/
    # WORLD_SIZE/LOCAL_RANK set, WORLD_SIZE > 1). Every big Linear then loads
    # only its 1/world row-slice of the pack (QVBandShardedLinearV8) instead
    # of the whole thing, all-gathering back to the full activation inside
    # its own forward() -- see qv_band/inference/sharded.py's docstring for
    # why row-sharding (not accelerate's whole-layer offload) is required on
    # this hybrid linear-attention model. Lets a pack too big for one GPU's
    # VRAM (e.g. 27B's 19+ GB packs on a 16 GB V100) run split across GPUs
    # through the SAME load_qv_model / chat.py entry point single-GPU
    # callers already use -- just launched differently.
    tp_world = int(os.environ.get("WORLD_SIZE", "1"))
    tp_rank = int(os.environ.get("RANK", "0"))
    tp_enabled = tp_world > 1
    if tp_enabled:
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if backend is torch.cuda:
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        if not dist.is_initialized():
            # No collective timeout: an idle follower rank waits in a broadcast for
            # the next request, and the 10-minute default kills the server.
            import datetime
            dist.init_process_group("nccl" if backend is torch.cuda else "gloo",
                                    timeout=datetime.timedelta(days=365))

    # Pipeline placement (QV_PIPELINE_SPLIT=k, one process, no torchrun):
    # decoder layers 0..k-1 live WHOLE on QV_PIPELINE_FIRST (default cuda:0),
    # everything else -- layers k.., embedding, final norm, lm_head, and the
    # caller's drafter and request tensors -- on `device`, the main card. For
    # a pack too big for one card but with no need to shard each layer: no
    # per-linear collectives (tensor parallel issues ~500 a decode step, the
    # step is CPU-bound), no ranks to keep in lockstep, and the first card
    # holds weights only. _install_pipeline_hooks moves each layer's inputs
    # to its card and makes that card current while it runs.
    main_device = device
    pipe_split = int(os.environ.get("QV_PIPELINE_SPLIT", "0") or 0)
    pipe_first = None
    if pipe_split > 0:
        if tp_enabled:
            raise RuntimeError("QV_PIPELINE_SPLIT is for a single-process launch, not torchrun")
        pipe_first = os.environ.get("QV_PIPELINE_FIRST", "cuda:0")
        _md = torch.device(main_device)
        if _md.index is None:                  # "cuda" -> the current card, explicitly
            main_device = device = f"{_md.type}:{backend.current_device()}"
        if torch.device(pipe_first) == torch.device(main_device):
            raise RuntimeError(f"QV_PIPELINE_FIRST={pipe_first} is the main device too; "
                               f"pass --device for the OTHER card")
        if backend is torch.cuda:
            torch.cuda.set_device(torch.device(main_device))
        print(f">>> pipeline: layers 0..{pipe_split - 1} on {pipe_first}, "
              f"layers {pipe_split}.. + embed/norm/lm_head on {main_device}", flush=True)

    import re
    _layer_re = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

    def _place(name: str) -> str:
        """The card a parameter/buffer/module of this name belongs on, always with
        an index: a bare "cuda" (the single-card default) never compares equal to
        the cuda:N a tensor reports, which used to send every weight on a host
        round trip at load (to_device / the final placement pass)."""
        if pipe_first is None or name.startswith("mtp.") or ".visual." in name:
            d = main_device
        else:
            m = _layer_re.search(name)
            d = pipe_first if m and int(m.group(1)) < pipe_split else main_device
        if d == "cuda" and torch.cuda.is_available():
            d = f"cuda:{torch.cuda.current_device()}"
        return d

    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                        trust_remote_code=True)

    # 1) skeleton on meta -- no weights anywhere.
    print(">>> building model skeleton on meta ...", flush=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True,
                                        trust_remote_code=True)
    nemotron = False
    text_cfg = config.llm_config if nemotron else getattr(config, "text_config", config)
    _attn_impl = os.environ.get("QV_ATTN_IMPL", "").strip()
    if not _attn_impl:
        from transformers.utils import is_flash_attn_2_available
        _attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    text_cfg._attn_implementation = _attn_impl
    if _attn_impl == "sdpa":                      # V100: tensor-core causal prefill kernel (qv_band/attention); no-op elsewhere
        from qv_band.attention import install_llm_prefill
        if install_llm_prefill():
            print(">>> V100 prefill attention kernel installed", flush=True)
    print(f">>> attention impl: {_attn_impl}"
          f"{' (QV_ATTN_IMPL override)' if os.environ.get('QV_ATTN_IMPL', '').strip() else ''}",
          flush=True)
    with init_empty_weights():
        if nemotron:
            model, text_cfg = build_text_skeleton(args.model, config, _attn_impl)
        else:
            model = AutoModelForCausalLM.from_config(text_cfg, trust_remote_code=True)
    model_dtype = getattr(text_cfg, "torch_dtype", None) or torch.bfloat16
    # QV_MODEL_DTYPE=float16: run a bf16 checkpoint in fp16 -- for GPUs with no
    # bf16 hardware (V100, sm_70), where every bf16 op is emulated. An fp16
    # pack's weights already restore to fp16; the risk is activation overflow
    # (fp16 tops out at 65504), so compare outputs against bf16 first.
    _dt_override = os.environ.get("QV_MODEL_DTYPE", "").strip().lower()
    if _dt_override:
        _dtypes = {"float16": torch.float16, "fp16": torch.float16,
                   "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
        if _dt_override not in _dtypes:
            raise ValueError(f"QV_MODEL_DTYPE={_dt_override!r}: expected one of {sorted(_dtypes)}")
        model_dtype = _dtypes[_dt_override]
        text_cfg.torch_dtype = model_dtype
        print(f">>> model dtype: {model_dtype} (QV_MODEL_DTYPE override)", flush=True)
    model = model.to(model_dtype).eval()
    tied = bool(getattr(text_cfg, "tie_word_embeddings", False))

    meta_keys = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
    _resolve_key = make_resolve_key(meta_keys)

    # 2) swap modules from the pack-dir manifest BEFORE allocating dense weights.
    #    tau=0 packs are never materialised densely -> the VRAM win is real.
    skip_weights: set[str] = set()
    tau_used = None
    # Read unconditionally below (tied lm_head handling) regardless of
    # --no-compress -- must exist even on the dense path, or that read is an
    # UnboundLocalError (found via a real --no-compress load, not a
    # hypothetical: every no_compress caller crashed here before this fix).
    embed_packfile = None
    if not args.no_compress:
        manifest = json.load(open(os.path.join(args.pack_dir, "manifest.json")))
        tau_used = manifest.get("tau")
        tensors = manifest["tensors"]
        from qv_band.common.formats import get_format, require
        spec = get_format(manifest.get("format"))
        fmt = spec.name
        lin_cls, emb_cls = spec.linear, spec.embedding
        print(f">>> swapping {len(tensors)} modules from {args.pack_dir} "
              f"(tau={tau_used}, format={fmt}) ...", flush=True)
        failed = manifest.get("failed") or {}
        if failed:
            print(f"WARNING: {len(failed)} tensors FAILED to pack in this "
                  f"directory and will stream DENSE instead (not compressed): "
                  f"{list(failed)[:10]}{'...' if len(failed) > 10 else ''}",
                  flush=True)
        # Routed MoE expert stacks (Qwen3_5MoeExperts-shaped modules) are not
        # 1 tensor <-> 1 module like every other manifest entry -- every
        # expert of ONE projection (gate_up_proj or down_proj) packs
        # separately (see run_fullmodel_v7_deploy.py's EXPERT_KEY_SEP), so
        # they must be grouped by (block, projection) and swapped as one
        # QVBandExpertsV8 BEFORE the per-tensor loop below runs (which
        # would otherwise try to resolve each "<key>#expert<e>" as its own
        # module path and silently skip every one via the AttributeError
        # guard, leaving that block's experts unswapped and never streamed
        # dense either -- run_fullmodel_v7_deploy.py never bundles a fully-
        # packed stack's tensor into model.safetensors, so a skipped expert
        # stack is missing weights entirely, not merely slow).
        expert_by_block: dict[str, dict[str, dict[int, str]]] = {}
        for tkey in list(tensors):
            base, eidx = _parse_expert_key(tkey)
            if eidx is None:
                continue
            module = tensor_key_to_module(base)
            block_path, _, proj_name = module.rpartition(".")
            expert_by_block.setdefault(block_path, {}).setdefault(
                proj_name, {})[eidx] = os.path.join(args.pack_dir, tensors[tkey]["pack"])

        for block_path, projs in expert_by_block.items():
            try:
                owner_block = _resolve(model, block_path)
            except AttributeError:
                if not block_path.startswith("model."):
                    try:
                        owner_block = _resolve(model, "model." + block_path)
                        block_path = "model." + block_path
                    except AttributeError:
                        print(f"[qv] skipping experts at '{block_path}': "
                              f"module not present in this model", flush=True)
                        continue
                else:
                    print(f"[qv] skipping experts at '{block_path}': module "
                          f"not present in this model", flush=True)
                    continue
            num_experts = getattr(owner_block, "num_experts", None)
            if num_experts is None:
                print(f"[qv] skipping experts at '{block_path}': no "
                      f"`num_experts` attribute (not a Qwen3_5MoeExperts-"
                      f"shaped module) -- leaving on meta", flush=True)
                continue
            gate_up_packs = projs.get("gate_up_proj", {})
            down_packs = projs.get("down_proj", {})
            if len(gate_up_packs) != num_experts or len(down_packs) != num_experts:
                # run_fullmodel_v7_deploy.py's dense bundle is all-or-nothing
                # per stack (see its _bundle_dense_tensors docstring): a
                # partially-packed stack was bundled fully dense there, so
                # the RIGHT thing here is to also leave it alone -- it will
                # be streamed dense by step 3 below, exactly like any other
                # tensor this loader does not swap.
                print(f"[qv] '{block_path}': only "
                      f"{len(gate_up_packs)}/{num_experts} gate_up + "
                      f"{len(down_packs)}/{num_experts} down packed -- "
                      f"streaming this stack dense instead", flush=True)
                continue
            if spec.experts is None:
                # This format has no expert-stack reader. Streaming the
                # stack dense is the SAME thing this loop already does for
                # a partially-packed stack above, so the model still loads;
                # it just does not get this stack compressed.
                print(f"[qv] '{block_path}': format {fmt} has no expert "
                      f"reader -- streaming this stack dense", flush=True)
                continue
            replacement = spec.experts.from_packfiles(
                gate_up_packs, down_packs, num_experts=num_experts,
                act_fn=getattr(owner_block, "act_fn", None),
                device=_place(block_path),     # pipeline placement: this layer's card, not the main one
                tau=tau_used)
            _setattr_dotted(model, block_path, replacement)
            skip_weights.add(f"{block_path}.gate_up_proj")
            skip_weights.add(f"{block_path}.down_proj")
        backend.empty_cache()

        embed_module_name = None
        # Overlaps the NEXT tensor's disk read with the CURRENT tensor's
        # GPU decode/certify work -- profiling the real 27B fp8 load
        # found I/O is 24% of from_packfile's own time, the rest GPU
        # compute a background thread cannot speed up (see prefetch.py's
        # module docstring). Same path list and order this loop's own
        # `tensors.items()` filter produces, computed once up front.
        from qv_band.common.prefetch import PackfilePrefetcher
        _prefetch_paths = [
            os.path.join(args.pack_dir, info["pack"])
            for tkey, info in tensors.items()
            if _parse_expert_key(tkey)[1] is None]
        _prefetcher = PackfilePrefetcher(_prefetch_paths)
        for tkey, info in tensors.items():
            base, eidx = _parse_expert_key(tkey)
            if eidx is not None:
                continue  # handled above, as a whole stack per block
            module = tensor_key_to_module(tkey)
            packfile = os.path.join(args.pack_dir, info["pack"])
            _blob = _prefetcher.get(packfile)
            try:
                owner = _resolve(model, module)
            except AttributeError:
                # Some checkpoints' tensor keys omit the leading 'model.'
                # the skeleton's own module tree uses (confirmed on real
                # checkpoints: Qwen3-Embedding-0.6B/4B ship keys like
                # 'layers.0...'/'embed_tokens...' with no prefix at all,
                # while Qwen3-Reranker-0.6B/4B -- same qwen3 architecture,
                # same AutoModelForCausalLM skeleton -- DO keep it).
                # tensor_key_to_module only strips a stray
                # 'language_model.' segment, never ADDS a missing 'model.'
                # prefix, so every tensor here failed to resolve and this
                # branch silently treated the WHOLE MODEL as "not in the
                # text-only skeleton" -- the entire checkpoint stayed on
                # meta with zero real weights, compressed or dense, since
                # the dense-stream fallback (step 3 below) hits the
                # identical mismatch. Try the prefixed path before giving
                # up for real.
                if module.startswith("model."):
                    continue
                try:
                    owner = _resolve(model, "model." + module)
                    module = "model." + module
                except AttributeError:
                    # genuinely not in the text-only skeleton (e.g. vision
                    # tower) -- leave it to stream dense.
                    continue
            # Pipeline placement: every reader below loads onto `device`, so
            # point it at this module's card (restored after the loop).
            device = _place(module)
            if isinstance(owner, nn.Embedding):
                # A tensor's OWN pack extension decides its reader, not the
                # run-wide manifest["format"] -- qvband packs are read the
                # same way regardless of which restore target calibrated
                # them (the calc method is a tag in the pack's own header,
                # not a separate format), so this only matters when a
                # tensor's pack extension genuinely differs from the run's
                # (e.g. a pack dir shipping one tensor in another container).
                tensor_fmt = os.path.splitext(info["pack"])[1].lstrip(".")
                tensor_spec = get_format(tensor_fmt) if tensor_fmt != fmt else spec
                tensor_emb_cls = tensor_spec.embedding
                if tensor_emb_cls is None:
                    # This format has no embedding reader (e.g. fp8n:
                    # a gather is not a GEMM, so native FP8 packs the
                    # linears only). Streaming dense is the same handling
                    # this loop gives everything else it does not swap.
                    print(f"[qv] '{module}': format {tensor_fmt} has no "
                          f"embedding reader -- streaming it dense", flush=True)
                    continue
                if tensor_emb_cls is QVBandEmbeddingV8:
                    # QV_HOST_EMBED=1 (TP or pipeline): the packed table stays in
                    # host RAM; a lookup decodes a few rows on the CPU (QVBand-
                    # EmbeddingV8.forward returns them on the input's device).
                    host = _HOST_EMBED_ENABLED and (tp_enabled or pipe_split > 0)
                    try:
                        repl = tensor_emb_cls.from_packfile(
                            packfile, out_dtype=model_dtype,
                            device="cpu" if host else device,
                            tau=tau_used, blob=_blob)
                        if host:
                            repl._qv_host = True
                            print(f"[qv] '{module}': qvb8 embedding kept in host RAM", flush=True)
                        _setattr_dotted(model, module, repl)
                        embed_module_name = module
                        embed_packfile = packfile
                    except NotImplementedError as e:
                        # QVBandEmbeddingV8 raises this when the tensor's
                        # OWN restore type (fp8/nvfp4) has no embedding
                        # reader -- a per-tensor decision only knowable
                        # after peeking the pack's own header, unlike the
                        # format-level "tensor_emb_cls is None" case
                        # above. That format-level case's `continue`
                        # works because such a tensor was NEVER excluded
                        # from the dense bundle in the first place (the
                        # packer itself never attempted it) -- but a
                        # tensor whose OWN restore type packed
                        # successfully IS excluded from model.safetensors
                        # by the packer, believing the pack covers it, so
                        # a bare `continue` here leaves it on `meta`
                        # forever. Same fallback as UncertifiableTensor
                        # below: read it straight from the source
                        # checkpoint.
                        print(f"[qv] '{module}': {e} -- loading dense from "
                              f"source instead", flush=True)
                        if not _load_dense_from_source(manifest, tkey, module,
                                                       model, device):
                            raise RuntimeError(
                                f"'{module}' has no embedding reader for its "
                                f"restore type and its source checkpoint "
                                f"({manifest.get('hf_model')}) does not have "
                                f"'{tkey}' either -- no way to recover real "
                                f"data for this tensor.") from e
                    except UncertifiableTensor as e:
                        # This tensor packed successfully -- _bundle_dense_
                        # tensors excluded it from model.safetensors,
                        # believing the pack covered it -- but now refuses
                        # at LOAD time, so the dense bundle has no fallback
                        # for it. Load it straight from the original
                        # checkpoint instead of leaving it on `meta`.
                        print(f"[qv] '{module}': cannot be certified at its "
                              f"stored precision -- loading dense from "
                              f"source instead ({e})", flush=True)
                        if not _load_dense_from_source(manifest, tkey, module,
                                                       model, device):
                            raise RuntimeError(
                                f"'{module}' refused at load time and its "
                                f"source checkpoint ({manifest.get('hf_model')}) "
                                f"does not have '{tkey}' either -- no way to "
                                f"recover real data for this tensor.") from e
                else:
                    repl = tensor_emb_cls.from_packfile(
                        packfile, out_dtype=model_dtype, device=device)
                    _setattr_dotted(model, module, repl)
                    embed_module_name = module
                    embed_packfile = packfile
            elif isinstance(owner, nn.Linear):
                bias = None if owner.bias is None or owner.bias.is_meta else owner.bias
                # Same per-tensor extension rule as the embedding branch
                # above: a Linear packed at a different restore than the
                # run-wide format (currently only embed_tokens/lm_head can
                # be, when tied and owned by an nn.Linear rather than an
                # nn.Embedding) must be read with ITS format's class.
                tensor_fmt = os.path.splitext(info["pack"])[1].lstrip(".")
                tensor_spec = get_format(tensor_fmt) if tensor_fmt != fmt else spec
                tensor_lin_cls = tensor_spec.linear
                if tp_enabled:
                    sharded_cls = require(tensor_spec, "sharded_linear")
                    repl = sharded_cls.from_packfile(
                        packfile, rank=tp_rank, world=tp_world, bias=bias,
                        device=device)
                else:
                    if tensor_lin_cls is QVBandLinearV8:
                        try:
                            repl = tensor_lin_cls.from_packfile(
                                packfile, bias=bias, device=device,
                                tau=tau_used, blob=_blob)
                        except UncertifiableTensor as e:
                            # See the embedding branch above for why this
                            # tensor has no dense-bundle fallback and must
                            # be re-read from the source checkpoint here.
                            print(f"[qv] '{module}': cannot be certified at "
                                  f"its stored precision -- loading dense "
                                  f"from source instead ({e})", flush=True)
                            if not _load_dense_from_source(
                                    manifest, tkey, module, model, device):
                                raise RuntimeError(
                                    f"'{module}' refused at load time and "
                                    f"its source checkpoint "
                                    f"({manifest.get('hf_model')}) does not "
                                    f"have '{tkey}' either -- no way to "
                                    f"recover real data for this tensor.") from e
                            skip_weights.add(module + ".weight")
                            continue
                    else:
                        repl = tensor_lin_cls.from_packfile(
                            packfile, bias=bias, device=device)
                _setattr_dotted(model, module, repl)
            elif (isinstance(getattr(owner, "weight", None), torch.nn.Parameter)
                  and getattr(owner, "weight").dim() == 2
                  and getattr(owner, "bias", "MISSING") in (None, "MISSING")):
                # A module whose forward does F.linear(x, self.weight) via a
                # bare 2-D nn.Parameter -- neither nn.Linear nor nn.Embedding
                # -- e.g. Qwen3_5MoeTopKRouter (the MoE gate/router): its
                # `self.weight = nn.Parameter(torch.zeros(E, hidden))` is
                # what run_fullmodel_v7_deploy.py calibrated (it's a plain
                # 2-D tensor, eligible same as any nn.Linear weight), but
                # the isinstance checks above never matched its OWNING
                # module, so it fell through this whole loop untouched --
                # left on `meta` (never allocated real data by either the
                # swap here or the dense stream in step 3, since
                # _bundle_dense_tensors correctly excludes it as "already
                # packed" per the manifest) and fed NaN/garbage router
                # logits into every MoE block's expert selection. Rebind
                # the module's forward to route its matmul through the
                # packed linear instead of replacing the module (keeps
                # top_k/num_experts/hidden_dim and any other owner state
                # intact) -- same pattern as fuse_tau_mlps in the sibling
                # src_tau_comp attempt.
                if type(owner).__name__ == "NemotronHTopkRouter":
                    # sigmoid router run in fp32 (small: n_experts x hidden). Decode the pack ONCE to the dense
                    # restored weight (identity through the packed GEMM = the certified bf16 values exactly) and
                    # let the model's own forward use it, rather than binding a packed matmul to it.
                    packed = lin_cls.from_packfile(packfile, bias=None, device=device, tau=tau_used, blob=_blob)
                    n_in = owner.weight.shape[1]
                    dense = packed(torch.eye(n_in, dtype=torch.bfloat16, device=device)).T.contiguous()
                    owner.weight = nn.Parameter(dense, requires_grad=False)
                    del packed
                elif lin_cls is QVBandLinearV8:
                    try:
                        packed = lin_cls.from_packfile(
                            packfile, bias=None, device=device, tau=tau_used,
                            blob=_blob)
                        _bind_packed_weight_forward(owner, packed)
                    except UncertifiableTensor as e:
                        # See the embedding branch above for why this
                        # tensor has no dense-bundle fallback and must be
                        # re-read from the source checkpoint here --
                        # `module` is this router's own dotted path, and
                        # owner.weight is exactly model.<module>.weight,
                        # so the usual "{module}.weight" target applies
                        # unchanged.
                        print(f"[qv] '{module}': cannot be certified at its "
                              f"stored precision -- loading dense from "
                              f"source instead ({e})", flush=True)
                        if not _load_dense_from_source(
                                manifest, tkey, module, model, device):
                            raise RuntimeError(
                                f"'{module}' refused at load time and its "
                                f"source checkpoint "
                                f"({manifest.get('hf_model')}) does not "
                                f"have '{tkey}' either -- no way to recover "
                                f"real data for this tensor.") from e
                else:
                    packed = lin_cls.from_packfile(packfile, bias=None, device=device)
                    _bind_packed_weight_forward(owner, packed)
            else:
                continue
            skip_weights.add(module + ".weight")
        device = main_device
        backend.empty_cache()

    if tp_enabled and not args.no_compress and not os.environ.get("QV_NO_FUSE"):
        from qv_band.formats.qvb8.backends_experts import fuse_sharded_siblings
        _nf = fuse_sharded_siblings(model)
        if _nf:
            print(f">>> fused {_nf} groups of sibling projections "
                  f"(one kernel + one collective each)", flush=True)
        backend.empty_cache()

    # 3) stream remaining (dense) tensors from safetensors, skipping swapped ones.
    idx_path = os.path.join(args.model, "model.safetensors.index.json")
    has_dense_here = os.path.exists(idx_path) or any(
        p.endswith(".safetensors") for p in os.listdir(args.model))
    if not has_dense_here and not args.no_compress:
        # A pack dir is meant to be self-contained (see config.py's
        # LLM_PACK_DIR comment): the .qvband files cover every Linear/Embedding,
        # but 1-D tensors (layernorm weight, linear-attn dt_bias/A_log/conv1d,
        # ...) are never packed and were always meant to ship alongside as a
        # plain model.safetensors -- built once, here, on whatever run first
        # finds it missing, so the pack dir heals itself instead of silently
        # leaving those tensors on `meta` (which crashes deep inside the
        # embedding forward with a confusing "Cannot copy out of meta tensor"
        # instead of a message that points at the actual cause).
        _repair_pack_dir_missing_dense(args.pack_dir, manifest,
                                       skip_weights, _resolve_key)
        has_dense_here = os.path.exists(
            os.path.join(args.pack_dir, "model.safetensors"))
    if os.path.exists(idx_path):
        weight_map = json.load(open(idx_path))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = [p for p in os.listdir(args.model) if p.endswith(".safetensors")]

    # official FP8 checkpoint: float8_e4m3fn weights with a per-block inverse scale; dequantised to the model
    # dtype while streaming (weight-only, no activation quantisation)
    weight_map_fp8 = json.load(open(idx_path))["weight_map"] if os.path.exists(idx_path) else {}
    fp8_block = (getattr(config, "quantization_config", None) or {}).get("weight_block_size") or [128, 128]
    if isinstance(getattr(config, "quantization_config", None), dict) is False and hasattr(
            getattr(config, "quantization_config", None), "weight_block_size"):
        fp8_block = config.quantization_config.weight_block_size or [128, 128]

    print(">>> streaming remaining dense weights ...", flush=True)
    t0 = time.time()
    n_loaded = n_skipped = 0
    for shard in shards:
        with safe_open(os.path.join(args.model, shard), framework="pt", device=device) as f:
            for ckpt_key in f.keys():
                tgt = _resolve_key(ckpt_key)
                if tgt is None:
                    continue
                if tgt in skip_weights:
                    n_skipped += 1
                    continue
                tgt_device = _place(tgt)
                # Host-resident embedding is opt-in (QV_HOST_EMBED=1): by default
                # every weight lives on the GPU, nothing is served from host RAM.
                if ((tp_enabled or pipe_split > 0) and _HOST_EMBED_ENABLED
                        and tgt.endswith(".weight")):
                    try:
                        emb = _resolve(model, tgt[:-len(".weight")])
                    except AttributeError:
                        emb = None
                    if (type(emb) is nn.Embedding
                            and emb.num_embeddings >= _HOST_EMBED_MIN_ROWS):
                        emb.__class__ = _HostEmbedding
                        tgt_device = "cpu"
                        print(f"[qv] '{tgt}': dense, kept in host RAM "
                              f"(tensor-parallel: saves VRAM on every rank)",
                              flush=True)
                value = f.get_tensor(ckpt_key)
                if value.dtype == torch.float8_e4m3fn:
                    value = _dequant_fp8_block(value, ckpt_key, f, weight_map_fp8, args.model, device,
                                               fp8_block)
                set_module_tensor_to_device(model, tgt, tgt_device,
                                            value=to_device(value, tgt_device)
                                            if tgt_device != "cpu" else value)
                n_loaded += 1
    print(f"    loaded {n_loaded} dense tensors, skipped {n_skipped} swapped "
          f"in {time.time()-t0:.0f}s", flush=True)
    if weight_map_fp8 and any(k.endswith(".mlp.experts.0.gate_proj.weight") for k in weight_map_fp8):
        t1 = time.time()
        n_l = _stream_fp8_experts(model, args.model, weight_map_fp8, device, fp8_block, _resolve_key)
        print(f"    stacked the per-expert FP8 weights of {n_l} layers (dequantised) in {time.time()-t1:.0f}s", flush=True)

    # Tied lm_head: the checkpoint omits lm_head.weight because it reuses
    # the embedding's. The embedding is now a packed module with no .weight
    # parameter, so HF's tie_weights() cannot find its source -- the head is
    # built from the embedding's own pack file instead, through the format
    # registry. Reading the pack a second time costs its bytes again; the
    # alternative, sharing the module, is not available for every format.
    if tied:
        lm_head = getattr(model, "lm_head", None)
        if embed_packfile is not None and isinstance(lm_head, nn.Linear):
            # embed_packfile's own extension decides the reader here too --
            # it may be a different format than the run-wide `fmt` if the
            # embedding and the rest of the model were packed separately,
            # and lin_cls is the run-wide one.
            embed_fmt = os.path.splitext(embed_packfile)[1].lstrip(".")
            embed_lin_cls = (get_format(embed_fmt).linear
                             if embed_fmt != fmt else lin_cls)
            if embed_lin_cls is QVBandLinearV8:
                model.lm_head = embed_lin_cls.from_packfile(
                    embed_packfile, bias=None, device=device, tau=tau_used)
            else:
                model.lm_head = embed_lin_cls.from_packfile(
                    embed_packfile, bias=None, device=device)
            print(f"    tied lm_head -> {embed_lin_cls.__name__} from embed pack",
                  flush=True)
        else:
            model.tie_weights()

    # re-init any still-meta buffers (rotary inv_freq etc.).
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            parent_path, _, leaf = name.rpartition(".")
            owner = _resolve(model, parent_path) if parent_path else model
            if hasattr(owner, "rope_init_fn") and leaf == "inv_freq":
                inv_freq, _ = owner.rope_init_fn(owner.config, device)
                set_module_tensor_to_device(model, name, device, value=inv_freq)
            else:
                set_module_tensor_to_device(model, name, device,
                                            value=torch.zeros(buf.shape, dtype=buf.dtype,
                                                              device=device))
    # Every remaining tensor onto ITS card. (This compared p.device.type --
    # "cuda" -- against a device string like "cuda:0", which never matched,
    # so it re-set every tensor; harmless on one card, but under pipeline
    # placement it would pull every layer back onto the main card.)
    def _want(name: str) -> torch.device:
        return torch.device(_place(name))       # _place always names an index
    for name, buf in list(model.named_buffers()):
        if getattr(_resolve(model, name.rpartition(".")[0]), "_qv_host", False):
            continue                       # deliberately host-resident
        want = _want(name)
        if not buf.is_meta and buf.device != want:
            set_module_tensor_to_device(model, name, want, value=to_device(buf, want))
    for name, p in list(model.named_parameters()):
        if isinstance(_resolve(model, name.rpartition(".")[0]), _HostEmbedding):
            continue                       # deliberately host-resident
        want = _want(name)
        if not p.is_meta and p.device != want:
            set_module_tensor_to_device(model, name, want, value=to_device(p.data, want))
    # the pack format's own post-load step (registry hook; qvb8r fuses same-input
    # linears into one launch, each group on its members' card) -- no format names here
    if not args.no_compress and spec.finalize is not None:
        print(f">>> {fmt} finalize: {spec.finalize(model)}", flush=True)
    if pipe_first is not None:
        _install_pipeline_hooks(model, _place)

    gc.collect()
    backend.empty_cache()
    if nemotron:
        model._qv_nemotron = True
        model._qv_eos_ids = _nemotron_eos_ids(args.model, config, tok)
        from qv_band.run.nemotron_h import install_fast_paths
        print(f">>> nemotron fast paths: {install_fast_paths(model)}", flush=True)
    model._qv_vram_weights = backend.memory_allocated() / 1e6
    model._qv_tau_used = tau_used
    model._qv_dtype = model_dtype
    return model, tok


def _generate_and_report(model, tok, args):
    device = args.device
    backend = _backend(device)
    vram_weights = model._qv_vram_weights
    tau_used = model._qv_tau_used
    model_dtype = model._qv_dtype

    # 4) generate.
    text = args.prompt
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                       tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    in_len = inputs["input_ids"].shape[1]
    pad = getattr(model.config, "eos_token_id", None)

    do_sample = args.do_sample or (args.seed is not None)
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=do_sample,
                      pad_token_id=pad)
    if do_sample:
        gen_kwargs.update(temperature=args.temperature,
                          top_p=args.top_p, top_k=args.top_k)

    if getattr(args, "draft_model", None):
        if _spec_generate_and_report(model, tok, args, inputs["input_ids"], do_sample):
            return

    with torch.no_grad():       # warmup so first timed token isn't paying JIT init
        model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=pad)
    backend.synchronize()

    backend.reset_peak_memory_stats()
    print(">>> generating ...", flush=True)
    # Seed right before the timed generate so sampling is reproducible.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if hasattr(backend, "manual_seed_all"):
            backend.manual_seed_all(args.seed)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    backend.synchronize()
    dt = time.time() - t0

    new_tokens = out.shape[1] - in_len
    peak = backend.max_memory_allocated() / 1e6

    print("\n=== OUTPUT ===")
    print(tok.decode(out[0, in_len:], skip_special_tokens=True))
    print("\n=== STATS ===")
    mode = f"dense {model_dtype}" if args.no_compress else f"QV-band tau={tau_used}"
    print(f"  mode            : {mode}")
    if do_sample:
        print(f"  sampling        : seed={args.seed} temp={args.temperature} "
              f"top_p={args.top_p} top_k={args.top_k}")
    else:
        print(f"  sampling        : greedy (deterministic)")
    print(f"  new tokens      : {new_tokens}")
    print(f"  time            : {dt:.2f}s")
    print(f"  throughput      : {new_tokens/dt:.2f} tok/s")
    print(f"  VRAM (weights)  : {vram_weights:.0f} MB  (resident after load)")
    print(f"  VRAM (peak)     : {peak:.0f} MB  (transient peak during generate)")


def _spec_generate_and_report(model, tok, args, input_ids, do_sample):
    """--draft-model: speculative decoding with a DFlash drafter through
    spec_decode.speculative_generate (the same engine the server uses), on the
    already-loaded packed target. Plain (non-int8) KV cache, so the target
    sees exactly what model.generate would give it."""
    from qv_band.decode.dflash_draft import build_dflash_drafter
    from qv_band.decode.spec_decode import speculative_generate
    device = args.device
    backend = _backend(device)
    try:
        drafter = build_dflash_drafter(model, args.draft_model, device=device,
                                        pack_dir=getattr(args, "draft_pack", None))
    except Exception as e:
        # An explicit --draft-model that does not fit is the caller's error;
        # one picked by name must not take a working model down with it.
        if not getattr(args, "draft_auto", False):
            raise
        print(f">>> drafter {args.draft_model} unusable ({e}); plain decoding",
              flush=True)
        return False
    temp = args.temperature if do_sample else 0.0
    kw = dict(k=args.spec_k, stream=False, temperature=temp,
              top_k=args.top_k if do_sample else 0,
              top_p=args.top_p if do_sample else 1.0, int8_kv=False)

    speculative_generate(model, drafter, tok, input_ids, max_new=8, **kw)  # warmup
    backend.synchronize()
    backend.reset_peak_memory_stats()
    print(">>> generating (DFlash speculative) ...", flush=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if hasattr(backend, "manual_seed_all"):
            backend.manual_seed_all(args.seed)
    t0 = time.time()
    out, st = speculative_generate(model, drafter, tok, input_ids,
                                   max_new=args.max_new_tokens, **kw)
    backend.synchronize()
    dt = time.time() - t0
    peak = backend.max_memory_allocated() / 1e6

    print("\n=== OUTPUT ===")
    print(tok.decode(out, skip_special_tokens=True))
    print("\n=== STATS ===")
    print(f"  mode            : QV-band tau={model._qv_tau_used} + "
          f"DFlash{drafter.version} (k={args.spec_k})")
    print(f"  sampling        : " + (f"seed={args.seed} temp={args.temperature} "
          f"top_p={args.top_p} top_k={args.top_k}" if do_sample
          else "greedy (deterministic, identical to plain decoding)"))
    print(f"  new tokens      : {len(out)}")
    print(f"  time            : {dt:.2f}s")
    print(f"  throughput      : {len(out)/dt:.2f} tok/s")
    print(f"  accept length   : {(len(out) - 1)/max(st.steps, 1):.2f} tok/verify "
          f"({st.accepted}/{st.proposed} drafts accepted)")
    print(f"  {st.report()}")
    print(f"  VRAM (weights)  : {model._qv_vram_weights:.0f} MB target "
          f"+ {drafter.param_bytes/1e6:.0f} MB draft"
          + (" (qvb8)" if getattr(args, "draft_pack", None) else ""))
    print(f"  VRAM (peak)     : {peak:.0f} MB  (transient peak during generate)")
    return True


if __name__ == "__main__":
    main()
