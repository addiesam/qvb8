"""Image input for the served Qwen3.5 / 3.6 LLM (image_url chat content).

WHY A SEPARATE PATH. qv_band builds the TEXT half only (AutoModelForCausalLM
from text_config) and speculative_generate takes token ids, so neither the
vision tower nor image embeddings exist on the normal route. This module adds
them beside the already-loaded text model instead of changing either:

  1. the vision tower (model.visual.*) is built from the pack's vision_config
     and loaded DENSE from the source checkpoint (manifest "hf_model") -- it is
     small (~0.5B) and the pack's qvband vision linears would need their own
     reader wiring for no real gain;
  2. the processor expands each image into <|image_pad|> tokens and pixels;
  3. image features replace those placeholders in the text model's input
     embeddings, and multimodal RoPE positions come from the upstream
     get_rope_index (borrowed, not reimplemented);
  4. a plain prefill + decode loop runs the SAME text model (its QVB8 linears
     and HF cache) -- no drafter. It has its own small prefix store (see
     VisionAdapter._prefix_lookup): an agent client resends the whole
     conversation, image included, every turn, and without it each turn
     prefilled all of it again.

Requests without images never reach this module.
"""
from __future__ import annotations

import base64
import io
import json
import copy
import hashlib
import os
import sys
import time

_ADAPTER = None

from qv_band.decode.spec_decode import PREFIX_MIN, PREFIX_TAIL, _bytes_of  # noqa: E402


def has_images(messages) -> bool:
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list) and any(
                isinstance(p, dict) and p.get("type") in ("image_url", "image")
                for p in c):
            return True
    return False


def _load_image(part: dict):
    from PIL import Image
    if part.get("type") == "image" and part.get("image"):
        src = part["image"]
    else:
        src = (part.get("image_url") or {})
        src = src.get("url") if isinstance(src, dict) else src
    if not isinstance(src, str):
        raise ValueError("image_url must be a data: URL or base64 string")
    # URLs and paths are refused (no fetching, no file reads). Checked BEFORE
    # stripping a data: header, and never on "/": raw base64 JPEG starts "/9j/".
    if src.startswith(("http://", "https://", "file:")):
        raise ValueError("only inline base64 / data: URL images are accepted")
    if src.startswith("data:"):
        src = src.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(src))).convert("RGB")


def split_messages(messages):
    """OpenAI content lists -> (template messages with {"type":"image"}, PIL images)."""
    out, images = [], []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if not isinstance(p, dict):
                    continue
                if p.get("type") in ("image_url", "image"):
                    images.append(_load_image(p))
                    parts.append({"type": "image"})
                elif p.get("type") == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
            m = dict(m, content=parts)
        out.append(m)
    return out, images


def _patch_conv_cls():
    import torch

    class PatchConvAsLinear(torch.nn.Module):
        """The patch-embed Conv3d (kernel == stride, one output per patch) as
        the packed [out, C*T*P*P] matrix it is stored as: conv over a whole
        patch == linear over the flattened patch, in the same C,T,P,P order."""

        def __init__(self, lin, conv, dtype):
            super().__init__()
            self.lin = lin
            self.out_channels = conv.out_channels
            # PatchEmbed.forward reads proj.weight.dtype to cast its input.
            self.register_buffer("weight", torch.empty(0, dtype=dtype), persistent=False)

        def forward(self, x):
            n = x.shape[0]
            return self.lin(x.reshape(n, -1)).view(n, self.out_channels, 1, 1, 1)
    return PatchConvAsLinear


class _OnHost:
    """A card tensor kept in pinned host RAM by _to_host: the copy and the card it came from."""
    __slots__ = ("t", "dev")

    def __init__(self, t, dev):
        self.t, self.dev = t, dev


def _to_host(obj, memo):
    """A copy of `obj` (a cache: lists / dicts / tuples / objects holding tensors) whose card tensors are in pinned host
    RAM (_OnHost), copied one at a time: no second copy of the cache on the card."""
    import torch
    k = id(obj)
    if k in memo:
        return memo[k]
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            h = torch.empty(obj.shape, dtype=obj.dtype, pin_memory=True)
            h.copy_(obj)
            out = _OnHost(h, obj.device)
        else:
            out = obj.clone()
    elif isinstance(obj, list):
        out = memo[k] = []
        out.extend(_to_host(v, memo) for v in obj)
        return out
    elif isinstance(obj, tuple):
        out = tuple(_to_host(v, memo) for v in obj)
    elif isinstance(obj, dict):
        out = memo[k] = {}
        out.update({kk: _to_host(v, memo) for kk, v in obj.items()})
        return out
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        out = memo[k] = copy.copy(obj)
        for kk, v in vars(obj).items():
            setattr(out, kk, _to_host(v, memo))
        return out
    else:
        out = copy.deepcopy(obj)
    memo[k] = out
    return out


def _host_bytes(obj, seen) -> int:
    """Pinned host bytes of the _OnHost tensors reachable from a _to_host copy."""
    if isinstance(obj, _OnHost):
        return obj.t.untyped_storage().nbytes()
    if id(obj) in seen or isinstance(obj, (int, float, str, bytes, bool, type(None))):
        return 0
    seen.add(id(obj))
    if isinstance(obj, dict):
        return sum(_host_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_host_bytes(v, seen) for v in obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return sum(_host_bytes(v, seen) for v in vars(obj).values())
    return 0


def _from_host(obj, memo=None):
    """The inverse of _to_host: a fresh cache with every _OnHost back on its card (the saved copy stays as it is)."""
    memo = {} if memo is None else memo
    k = id(obj)
    if k in memo:
        return memo[k]
    if isinstance(obj, _OnHost):
        out = obj.t.to(obj.dev, non_blocking=True)
    elif isinstance(obj, list):
        out = memo[k] = []
        out.extend(_from_host(v, memo) for v in obj)
        return out
    elif isinstance(obj, tuple):
        out = tuple(_from_host(v, memo) for v in obj)
    elif isinstance(obj, dict):
        out = memo[k] = {}
        out.update({kk: _from_host(v, memo) for kk, v in obj.items()})
        return out
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        out = memo[k] = copy.copy(obj)
        for kk, v in vars(obj).items():
            setattr(out, kk, _from_host(v, memo))
        return out
    else:
        out = copy.deepcopy(obj)
    memo[k] = out
    return out


class VisionAdapter:
    def __init__(self, bundle, verbose: bool = True):
        import torch
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from transformers import AutoConfig

        t0 = time.time()
        self.bundle = bundle
        pack = bundle.pack
        self.config = AutoConfig.from_pretrained(pack, local_files_only=True,
                                                 trust_remote_code=True)
        if getattr(self.config, "vision_config", None) is None:
            raise RuntimeError(f"{pack} has no vision_config: this model cannot take images")
        # The vision tower is DENSE in the pack (its dense bundle, at the source dtype): calibration never quantizes
        # it (pack_model.py), and qv_band.formats.qvb8.vision_dense brings older packs in line. The pack alone is
        # enough -- no source checkpoint is read.
        source = pack
        # Image processor only (PIL backend): the stock Qwen3VLProcessor also
        # builds a video processor that imports torchvision, which is not in
        # the serving venv (same reason as pipelines/image_edit._processor).
        # Image tokens are expanded here instead of by the processor.
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
            Qwen2VLImageProcessorPil)
        pp = pack if os.path.isfile(os.path.join(pack, "preprocessor_config.json")) else source
        max_px = int(os.environ.get("LLM_VISION_MAX_PIXELS", str(1024 * 1024)))
        self.image_processor = Qwen2VLImageProcessorPil.from_pretrained(
            pp, size={"shortest_edge": 65536, "longest_edge": max_px})

        # Vision tower class + rope-index owner for this architecture.
        mt = self.config.model_type
        # a model whose config CLASS derives from Qwen3.5's (SenseNova-Loop: SenseNovaLoopConfig(Qwen3_5Config)) has
        # the Qwen3.5 tower and rope index
        bases = {c.__name__ for c in type(self.config).__mro__}
        if mt != "qwen3_5_moe" and "Qwen3_5MoeConfig" in bases:
            mt = "qwen3_5_moe"
        elif mt not in ("qwen3_5", "qwen3_5_moe") and "Qwen3_5Config" in bases:
            mt = "qwen3_5"
        if mt == "qwen3_5_moe":
            from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as mod
            vis_cls, owner_cls = mod.Qwen3_5MoeVisionModel, mod.Qwen3_5MoeModel
        elif mt == "qwen3_5":
            from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
            vis_cls, owner_cls = mod.Qwen3_5VisionModel, mod.Qwen3_5Model
        else:
            raise RuntimeError(f"vision not wired for model_type {mt!r}")
        self._owner_cls = owner_cls

        dev = bundle.gpu_dev or "cpu"
        vcfg = self.config.vision_config
        vcfg._attn_implementation = "sdpa"
        with init_empty_weights():
            visual = vis_cls(vcfg)
        dtype = torch.bfloat16
        visual = self._load_dense(visual, pack, dtype, safe_open)
        self.weights_from = f"{pack} (dense)"
        # Non-persistent buffers (rotary inv_freq) were built on meta: rebuild.
        for name, mod_ in visual.named_modules():
            if hasattr(mod_, "inv_freq") and mod_.inv_freq.is_meta:
                dim = mod_.inv_freq.shape[0] * 2
                theta = getattr(mod_, "theta", 10000.0)
                mod_.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        # --vision-tower host (LLM_VISION_TOWER=host): the tower lives in pinned host RAM and visits the card only while it
        # encodes a picture -- a 16 GB card holding the whole 27B has no room left for it beside a long prompt
        self.tower_host = os.environ.get("LLM_VISION_TOWER", "gpu") == "host" and str(dev) != "cpu"
        # --vision-tower cpu: the tower RUNS on the host (fp32) and only its image embeddings go to the card -- a 16 GB
        # card holding the whole 27B has not even room for the tower's visit (measured V100: out of memory at 340 MB)
        self.tower_cpu = os.environ.get("LLM_VISION_TOWER", "gpu") == "cpu"
        if self.tower_cpu:
            self.visual = visual.to("cpu", torch.float32).eval()
        elif self.tower_host:
            self.visual = visual.to("cpu").eval()
            # one pinned host copy per tensor; _tower_to swaps which copy the module points at
            self._host_tensors = [(t_, t_.data.pin_memory()) for t_ in list(self.visual.parameters()) + list(self.visual.buffers())]
            for t_, h_ in self._host_tensors:
                t_.data = h_
        else:
            self.visual = visual.to(dev).eval()
        self.dev = dev
        if verbose:
            n = sum(p.numel() for p in self.visual.parameters()) / 1e6
            print(f"[vision] tower {n:.0f}M dense params from {self.weights_from} "
                  f"({time.time() - t0:.1f}s)"
                  + ("; kept in pinned host RAM, on the card only while encoding" if self.tower_host else "")
                  + ("; runs on the CPU, only its embeddings go to the card" if self.tower_cpu else ""), flush=True)

    @staticmethod
    def _visual_tensors(sd_keys):
        return {k[len("model.visual."):]: k for k in sd_keys if k.startswith("model.visual.")}

    def _load_dense(self, visual, source, dtype, safe_open):
        """Every vision tensor dense, from the pack's dense bundle."""
        idx = os.path.join(source, "model.safetensors.index.json")
        files = (sorted(set(json.load(open(idx))["weight_map"].values()))
                 if os.path.isfile(idx) else ["model.safetensors"])
        sd = {}
        for fn in files:
            with safe_open(os.path.join(source, fn), framework="pt", device="cpu") as f:
                for k in f.keys():
                    for pre in ("model.visual.", "visual."):
                        if k.startswith(pre):
                            sd[k[len(pre):]] = f.get_tensor(k)
        if not sd:
            raise RuntimeError(f"no visual.* weights in {source}")
        missing = set(dict(visual.named_parameters())) - set(sd)
        if missing:
            raise RuntimeError(f"{source}: the vision tower is not dense in this pack ({len(missing)} tensors missing, "
                               f"e.g. {sorted(missing)[:3]}): python -m qv_band.formats.qvb8.vision_dense --pack {source}")
        visual.load_state_dict({k: v.to(dtype) for k, v in sd.items()},
                               strict=False, assign=True)
        return visual

    # Saved prompt prefixes, most recently used first. Each holds a DENSE HF cache (~1.7 GB for the
    # 27B at 24k tokens, 3x that at 80k), so they are bounded by size, not only by count: the text
    # route keeps compressed ones, these are not.
    _prefixes = None
    _PREFIX_KEEP = 2
    _PREFIX_BYTES = 6 * 2 ** 30

    def _prefix_lookup(self, key, hashes, runs):
        """(cache, tokens already in it, pictures it covers) for the longest saved prefix
        that starts this prompt, else (None, 0, 0). The cache is a copy: generating mutates it."""
        best = None
        if len(key) >= PREFIX_MIN:
            for e in self._prefixes or []:
                n = e["n"]
                if (n < len(key) and (best is None or n > best["n"]) and key[:n] == e["tokens"]
                        and hashes[:len(e["hashes"])] == e["hashes"]):
                    best = e
        if best is None:
            return None, 0, 0
        self._prefixes.remove(best)
        self._prefixes.insert(0, best)
        print(f"[prefix] vision: restored {best['n']} of {len(key)} prompt tokens, "
              f"prefilling {len(key) - best['n']}", file=sys.__stdout__, flush=True)
        if best.get("host"):
            return _from_host(best["past"]), best["n"], len(best["hashes"])
        return copy.deepcopy(best["past"]), best["n"], len(best["hashes"])

    def _prefix_save(self, key, hashes, runs, n, past) -> None:
        """Keep `past` (holding exactly key[:n]) for a later prompt that starts the same way."""
        if self._prefixes is None:
            self._prefixes = []
        tokens = key[:n]
        n_cov = sum(1 for _, r1 in runs if r1 <= n)
        # the same conversation moving forward supersedes its earlier prefix
        self._prefixes = [e for e in self._prefixes
                          if not (e["n"] <= n and tokens[:e["n"]] == e["tokens"]
                                  and hashes[:len(e["hashes"])] == e["hashes"])]
        host = os.environ.get("QV_PREFIX_STORE", "gpu") == "host"
        # --prefix-store host: copied straight from the live cache into pinned host RAM, never a second copy on the card
        snap = _to_host(past, {}) if host else copy.deepcopy(past)
        size = _host_bytes(snap, set()) if host else _bytes_of(snap, set())   # the same budget for host RAM
        if size > self._PREFIX_BYTES:
            return                                   # one snapshot alone would exceed the budget
        self._prefixes.insert(0, {"tokens": tokens, "hashes": hashes[:n_cov], "n": n,
                                  "past": snap, "bytes": size, "host": host})
        del self._prefixes[self._PREFIX_KEEP:]
        while sum(e["bytes"] for e in self._prefixes) > self._PREFIX_BYTES:
            self._prefixes.pop()                     # drop the least recently used

    # -- upstream helpers, borrowed with a minimal `self`
    def _rope_index(self, input_ids, mm_types, grid_thw):
        owner = self._owner_cls.__new__(self._owner_cls)
        object.__setattr__(owner, "config", self.config)
        return self._owner_cls.get_rope_index(owner, input_ids, mm_types,
                                              image_grid_thw=grid_thw)

    def prepare(self, messages, tok, embed, dev, *, tools=None, think=False):
        """An image chat for a server that runs the language model itself (the TP service): (prompt ids, the input
        embeddings [n, H] on `dev` with every picture's vision features in place, M-RoPE positions [3, n] (cpu),
        rope delta). `embed`: the language model's input embedding. Same template, processor, tower and rope index as
        generate()."""
        import torch
        msgs, images = split_messages(messages)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       tools=tools or None, enable_thinking=bool(think))
        px = self.image_processor(images=images, return_tensors="pt")
        grid = px["image_grid_thw"]
        merge = self.image_processor.merge_size ** 2
        pad = "<|image_pad|>"
        pieces = text.split(pad)
        if len(pieces) - 1 != len(images):
            raise ValueError(f"template has {len(pieces) - 1} image slots for {len(images)} images")
        text = pieces[0] + "".join(pad * int(g.prod() // merge) + rest for g, rest in zip(grid, pieces[1:]))
        ids = tok(text, return_tensors="pt").input_ids
        img_id = self.config.image_token_id
        with torch.no_grad():
            emb = embed(ids.to(dev))                                   # [1, n, H]
            pv = px["pixel_values"]
            if self.tower_cpu:
                feats = self.visual(pv.float(), grid_thw=grid, return_dict=True).pooler_output
            else:
                if self.tower_host:
                    for t_, h_ in self._host_tensors:
                        t_.data = h_.to(self.dev, non_blocking=True)
                try:
                    feats = self.visual(pv.to(self.dev, torch.bfloat16), grid_thw=grid.to(self.dev),
                                        return_dict=True).pooler_output
                finally:
                    if self.tower_host:
                        for t_, h_ in self._host_tensors:
                            t_.data = h_
                        torch.cuda.empty_cache()
            emb = emb.to(dev)
            mask = (ids.to(dev) == img_id).unsqueeze(-1).expand_as(emb)
            emb = emb.masked_scatter(mask, feats.to(dev, emb.dtype))
            pos, delta = self._rope_index(ids, (ids == img_id).int(), grid)
        return ids[0].tolist(), emb[0], pos[:, 0].cpu(), int(delta.reshape(-1)[0])

    def generate(self, messages, *, tools=None, think=False, max_new=1024,
                 temperature=0.0, top_k=20, top_p=0.95, on_token=None,
                 should_stop=None, on_progress=None, prefill_chunk=None) -> dict:
        """on_token(id): called for every generated token as it is sampled.
        should_stop(): polled after every prefill chunk and token (client gone).
        on_progress(): called after every prefill chunk (keep-alive hook).
        prefill_chunk: bound on tokens per prefill forward, so a long prompt does
        not need one prompt-sized forward (memory and an unresponsive server)."""
        import torch
        tok = self.bundle.tok
        msgs, images = split_messages(messages)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       tools=tools or None, enable_thinking=bool(think))
        px = self.image_processor(images=images, return_tensors="pt")
        grid_cpu = px["image_grid_thw"]
        merge = self.image_processor.merge_size ** 2
        pad = "<|image_pad|>"
        pieces = text.split(pad)
        if len(pieces) - 1 != len(images):
            raise ValueError(f"template has {len(pieces) - 1} image slots for {len(images)} images")
        text = pieces[0] + "".join(pad * int(g.prod() // merge) + rest
                                   for g, rest in zip(grid_cpu, pieces[1:]))
        ids = tok(text, return_tensors="pt").input_ids.to(self.dev)
        grid = grid_cpu.to(self.dev)
        img_id = self.config.image_token_id
        mm = (ids == img_id).int()
        enc = px
        key = ids[0].tolist()
        n_in = len(key)
        # Per-image identity and token run: two prompts with the same token ids but a
        # different picture must not share a prefix.
        patches = [int(g.prod()) for g in grid_cpu]
        pv_cpu = enc["pixel_values"]
        pad_at = (ids[0] == img_id).nonzero().flatten().tolist()
        hashes, runs, p0, q0 = [], [], 0, 0
        for n_p, g in zip(patches, grid_cpu):
            # identity = the pixels AND their grid: the same bytes laid out as a different grid get different positions
            hashes.append(hashlib.blake2b(pv_cpu[p0:p0 + n_p].numpy().tobytes() + repr(g.tolist()).encode(),
                                          digest_size=16).digest())
            runs.append((pad_at[q0], pad_at[q0] + n_p // merge))   # pad tokens of one image are contiguous
            p0 += n_p
            q0 += n_p // merge

        model = self.bundle.model
        t0 = time.time()
        with torch.no_grad():  # NOT inference_mode: qv_band MoE graph buffers created here are reused by the text route
            past, start, n_cov = self._prefix_lookup(key, hashes, runs)
            # The prefix already holds the pictures it covers: encode only the rest.
            emb = model.get_input_embeddings()(ids[:, start:])
            if n_cov < len(hashes):
                p_cov = sum(patches[:n_cov])
                pv = pv_cpu[p_cov:].float() if self.tower_cpu else pv_cpu[p_cov:].to(self.dev, torch.bfloat16)
                if self.tower_host:
                    for t_, h_ in self._host_tensors:
                        t_.data = h_.to(self.dev, non_blocking=True)
                try:
                    if self.tower_cpu:
                        feats = self.visual(pv, grid_thw=grid[n_cov:].cpu(), return_dict=True).pooler_output.to(self.dev)
                    else:
                        feats = self.visual(pv, grid_thw=grid[n_cov:], return_dict=True).pooler_output
                finally:
                    if self.tower_host:
                        for t_, h_ in self._host_tensors:
                            t_.data = h_                # the card copies are freed with this
                        torch.cuda.empty_cache()
                mask = (ids[:, start:] == img_id).unsqueeze(-1).expand_as(emb)
                emb = emb.masked_scatter(mask, feats.to(emb.dtype))
            pos, delta = self._rope_index(ids, mm, grid)

            eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
            if tok.eos_token_id is not None:
                eos.add(tok.eos_token_id)
            out_ids = []
            stopped = False
            chunk = int(prefill_chunk) if prefill_chunk and int(prefill_chunk) > 0 else n_in
            # Save point: PREFIX_TAIL short of the end (the generation prefix renders
            # differently once it is history, and the linear-attention state cannot be
            # rolled back), moved back to the start of a picture it would split.
            ckpt = n_in - PREFIX_TAIL
            for r0, r1 in runs:
                if r0 < ckpt < r1:
                    ckpt = r0
            save = ckpt >= PREFIX_MIN and ckpt > start
            o = None
            i = start
            while i < n_in:
                j = min(i + chunk, n_in)
                if save and i < ckpt < j:
                    j = ckpt
                o = model(inputs_embeds=emb[:, i - start:j - start], position_ids=pos[:, :, i:j],
                          past_key_values=past, use_cache=True, logits_to_keep=1)
                past = o.past_key_values
                if save and j == ckpt:
                    self._prefix_save(key, hashes, runs, ckpt, past)
                i = j
                if on_progress is not None:
                    on_progress()
                if should_stop is not None and should_stop():
                    stopped = True
                    break
            for step in range(0 if stopped else max_new):
                logits = o.logits[0, -1].float()
                nxt = self._sample(logits, temperature, top_k, top_p)
                if nxt in eos:
                    break
                out_ids.append(nxt)
                if on_token is not None:
                    on_token(nxt)
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                if step + 1 >= max_new:
                    break
                t_ids = torch.tensor([[nxt]], device=self.dev)
                step_emb = model.get_input_embeddings()(t_ids)
                p = ids.shape[1] + len(out_ids) - 1 + delta.item()
                step_pos = torch.full((3, 1, 1), p, dtype=torch.long, device=self.dev)
                o = model(inputs_embeds=step_emb, position_ids=step_pos,
                          past_key_values=past, use_cache=True, logits_to_keep=1)
                past = o.past_key_values
        dt = time.time() - t0
        return {"text": tok.decode(out_ids, skip_special_tokens=True),
                "prompt_tokens": int(ids.shape[1]), "completion_tokens": len(out_ids),
                "seconds": round(dt, 2), "images": len(images), "stopped": stopped}

    @staticmethod
    def _sample(logits, temperature, top_k, top_p) -> int:
        import torch
        if not temperature or temperature <= 0:
            return int(logits.argmax())
        logits = logits / temperature
        if top_k and top_k > 0:
            v, _ = torch.topk(logits, min(int(top_k), logits.numel()))
            logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, -1)
        if top_p and 0 < top_p < 1:
            sp, si = torch.sort(probs, descending=True)
            keep = sp.cumsum(-1) - sp <= top_p
            probs = torch.zeros_like(probs).scatter(0, si[keep], sp[keep])
            probs /= probs.sum()
        return int(torch.multinomial(probs, 1))


def adapter(bundle, drafter=None):
    """Lazily build (once) the vision adapter for the served bundle. A model whose adapter names one gets that
    (e.g. vision_cradio.py: C-RADIO tower, same generate() contract); `drafter` is the server's speculative drafter,
    used by that adapter's decode loop."""
    global _ADAPTER
    if _ADAPTER is None or _ADAPTER.bundle is not bundle:
        _ma = getattr(bundle.model, "_qv_adapter", None)
        if _ma is not None and _ma.vision_adapter:      # the model's adapter names another image-input kind
            import importlib
            mod, cls = _ma.vision_adapter.split(":")
            _ADAPTER = getattr(importlib.import_module(mod), cls)(bundle, drafter)
        else:
            _ADAPTER = VisionAdapter(bundle)
    return _ADAPTER


__all__ = ["has_images", "split_messages", "VisionAdapter", "adapter"]
