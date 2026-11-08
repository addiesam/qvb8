"""DFlash block-diffusion drafters for the speculative harness in spec_decode.py:
DFlash 2 (Inco AI, "DFlash 2: Keep Drafting Parallel", 2026) and the original
DFlash (Chen et al., ICML 2026).

Unlike the MTP/DSpark drafters, this is a separately trained draft model
(e.g. incoai/Qwen3.8-27B-DFlash2, ~3.8 GB bf16) that runs alongside the packed
target:

  * context  -- the outputs of target decoder layers `target_layer_ids`
    (5, 19, 33, 47, 61 on the 27B) are concatenated per position, projected by
    `fc` + `hidden_norm`, and become K/V context for every draft layer. The
    context K/V depends only on that projected feature (not on the draft
    layer's own input), so it is computed ONCE per committed position, in
    observe_features(), and kept for the sliding window.
  * block    -- [last_token, mask x (block_size-1)] at positions start..start+7
    run through the 5 draft layers non-causally (is_causal=false, sliding
    window 2048) in ONE pass. Each layer wraps attention and MLP with a
    grouped dynamic two-tap causal conv over the block (prepare before /
    finish after), which is what keeps the block tail from decaying.
  * selector -- the target lm_head scores the block; a low-rank
    predecessor/successor bilinear selector then walks one coherent path
    through the top-16 candidates at each position.

Embedding and lm_head are the TARGET's (packed) modules -- the checkpoint
ships neither. Math is a direct port of the reference implementation
(github.com/z-lab/dflash, dflash/model.py: DFlash2DraftModel,
GroupedDynamicCausalConv, CandidateSelector) and keeps its dtype behaviour
(bf16 weights, fp32 RMSNorm statistics).

Correctness is the harness's job, as with every drafter: greedy verification
is exact-match, so output is token-identical to plain greedy decoding. At
temperature > 0 propose() returns the selector's real proposal distribution
(full-vocab probabilities, mass on the 16 candidates) and sets
`returns_probs`, so spec_decode uses it as p_draft directly instead of
re-deriving one from logits.

    ./inference_qvb8.sh /mnt/data/models/Qwen3.8-27B_qvb8_1e-3_bf16 \\
        --draft-model /mnt/data/models_raw/incoai/Qwen3.8-27B-DFlash2

Measured on the dev GPU (one prompt each, 256 tokens, greedy; identical output to
plain decoding on the bf16 packs):

    target pack                 drafter                      plain   DFlash
    Qwen3.8-27B qvb8 bf16       incoai/...-DFlash2 (v2)       2.7    8.5 tok/s
    Qwen3.6-35B-A3B qvb8 fp16   incoai/...-DFlash2 (v2)      30.4   65.8 tok/s
    Qwen3.5-9B qvb8 bf16        z-lab/Qwen3.5-9B-DFlash (v1) 21.3   36.7 tok/s

A qvb8 pack of the drafter (--draft-pack) cuts its weights to ~52-62% with
unchanged acceptance; draft steps get slower on the 27B/9B (packed GEMMs at
M=8..16), so dense stays the default.
"""
from __future__ import annotations

import copy
import glob
import json
import os

import torch
import torch.nn.functional as F


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3RMSNorm: fp32 statistics, weight applied in the input dtype."""
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def _lin(x: torch.Tensor, w) -> torch.Tensor:
    """x @ w.T for a dense weight, or the packed module's own forward."""
    return F.linear(x, w) if isinstance(w, torch.Tensor) else w(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    """Reference _grouped_dynamic_convolve, verbatim: a per-group two-tap
    causal conv over the block whose taps are a learned base kernel plus a
    per-position dynamic kernel. Zero-padded at the block start, so it needs
    no state across blocks."""
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(
            blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)


class _DynConv:
    def __init__(self, base_kernel, kernel_projection, kernel_size, group_size):
        self.base = base_kernel                 # (2, kernel_size, H)
        self.proj = kernel_projection           # (2*kernel_size*groups, H)
        self.kernel_size = kernel_size
        self.group_size = group_size

    def prepare(self, x):
        groups = x.shape[-1] // self.group_size
        dyn = _lin(x, self.proj).view(*x.shape[:-1], 2, self.kernel_size, groups)
        return (_grouped_dynamic_convolve(x, dyn[..., 0, :, :], self.base[0], self.group_size),
                dyn[..., 1, :, :])

    def finish(self, x, dyn):
        return _grouped_dynamic_convolve(x, dyn, self.base[1], self.group_size)


class DFlashDrafter:
    """Drafter interface as spec_decode expects it (needs_hidden, hidden,
    reset, append, propose, feedback), plus the two hooks that are specific
    to a multi-layer-context drafter:

      capture_layers           -- target decoder layer indices whose OUTPUTS
                                  spec_decode concatenates per position;
      observe_features(p, f)   -- f[i] is that feature for absolute position
                                  p + i. Entries at >= p are dropped first, so
                                  a new request (p == 0), a reused prefix
                                  cache, or a rollback all stay consistent.

    Also drives DFlash v1 checkpoints (architectures DFlashDraftModel, e.g.
    z-lab/Qwen3.5-9B-DFlash): the same backbone without the dynamic convs or
    the candidate selector (tokens come straight from the draft logits), and
    with the reference's per-layer attention rule -- when config is_causal is
    unset, sliding layers are causal and full layers bidirectional.
    """

    needs_hidden = True          # routes spec_decode through the hooked forward
    returns_probs = True         # temperature>0 propose() yields probabilities

    def __init__(self, draft_dir: str, embed, lm_head, *, device="cuda",
                 dtype=torch.bfloat16, pack_dir: str | None = None):
        self._draft_dir = draft_dir
        cfg = json.load(open(os.path.join(draft_dir, "config.json")))
        dc = cfg.get("dflash_config", {})
        archs = cfg.get("architectures") or []
        if "DFlash2DraftModel" in archs:
            self.version = 2
        elif "DFlashDraftModel" in archs:
            self.version = 1
        else:
            raise ValueError(f"{draft_dir} is not a DFlash draft model "
                             f"(architectures={archs})")
        self.embed = embed
        self.lm_head = lm_head
        self.device = torch.device(device)
        self.dtype = dtype
        self.H = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim", self.H // self.n_heads)
        self.eps = cfg["rms_norm_eps"]
        self.n_layers = cfg["num_hidden_layers"]
        # Per-layer attention rule, as the reference Qwen3DFlashAttention:
        # a sliding layer has the window, and is causal unless config
        # is_causal says otherwise; a full layer sees everything.
        types = cfg.get("layer_types") or ["full_attention"] * self.n_layers
        sw = cfg.get("sliding_window") if cfg.get("use_sliding_window", True) else None
        forced = cfg.get("is_causal")
        self.layer_attn = []                   # (is_causal, window or None)
        for t in types:
            win = sw if t == "sliding_attention" else None
            causal = (t == "sliding_attention") if forced is None else bool(forced)
            self.layer_attn.append((causal, win))
        # Context positions older than the widest window are never visible
        # again. A full-attention layer needs the whole context (no trim).
        wins = [w for _, w in self.layer_attn]
        self.ctx_keep = None if any(w is None for w in wins) else max(wins)
        self._ctx_cap = int(os.environ.get("QV_DRAFT_CTX", "0") or 0) or None
        self.block_size = int(dc.get("block_size", 16))
        self.mask_token_id = int(dc["mask_token_id"])
        self.capture_layers = list(dc["target_layer_ids"])
        self.top_k = int(dc["selector_top_k"]) if self.version == 2 else 0
        rope = cfg.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope_type {rope.get('rope_type')}")
        theta = float(rope.get("rope_theta", cfg.get("rope_theta", 1e6)))
        self.inv_freq = 1.0 / (theta ** (
            torch.arange(0, self.hd, 2, dtype=torch.float32, device=self.device) / self.hd))
        for key in ("input_embedding_scale", "output_multiplier", "final_logit_softcapping"):
            if dc.get(key, cfg.get(key)) not in (None, 1.0, 0, 0.0):
                raise NotImplementedError(f"DFlash config {key}={dc.get(key, cfg.get(key))}")

        from safetensors.torch import load_file
        w = load_file(os.path.join(draft_dir, "model.safetensors"), device=str(self.device))
        w = {k: v.to(dtype) for k, v in w.items()}
        self.param_bytes = sum(v.numel() * v.element_size() for v in w.values())
        if pack_dir is None and self._is_pack_dir(draft_dir):
            # --draft-model pointed at a qvb8 PACK of the drafter (its model.safetensors holds only the
            # dense remainder: norms, conv kernels, codebooks). That is a complete drafter by itself --
            # the packed linears come from the same directory -- so use it as its own pack.
            pack_dir = draft_dir
        if pack_dir is not None:
            w = self._swap_packed(w, pack_dir)
        self.fc = w["fc.weight"]
        self.hidden_norm = w["hidden_norm.weight"]
        self.norm = w["norm.weight"]
        self.fc_in = (self.fc.shape[-1] if isinstance(self.fc, torch.Tensor)
                      else self.fc.in_features)
        if self.version == 2:
            ks, gs = int(dc["conv_kernel_size"]), int(dc["conv_group_size"])
        self.layers = []
        for i in range(self.n_layers):
            p = f"layers.{i}."
            L = {n: w[p + n + ".weight"] for n in (
                "input_layernorm", "post_attention_layernorm",
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
            L["attn_conv"] = L["mlp_conv"] = None
            if self.version == 2:
                L["attn_conv"] = _DynConv(w[p + "attention_conv.base_kernel"],
                                          w[p + "attention_conv.kernel_projection.weight"], ks, gs)
                L["mlp_conv"] = _DynConv(w[p + "mlp_conv.base_kernel"],
                                         w[p + "mlp_conv.kernel_projection.weight"], ks, gs)
            self.layers.append(L)
        if self.version == 2:
            self.sel_pred = w["candidate_selector.predecessor_codebook"]
            self.sel_succ = w["candidate_selector.successor_codebook"]
            self.sel_proj = w["candidate_selector.hidden_projection.weight"]
            # The two (vocab, r) codebooks (~0.25 GB each dense) stay in host RAM (pinned with a GPU): propose() reads
            # only the candidate rows from them -- exact, a few hundred KB per step.
            host = lambda t: t.cpu().pin_memory() if torch.cuda.is_available() else t.cpu()
            self.sel_pred, self.sel_succ = host(self.sel_pred), host(self.sel_succ)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del w

        self._target_dtype = None   # learnt from the target embedding's output
        self._graph_shared = {}     # device codebooks / cached mask embedding for dflash_graph (shared by forks)
        self._reset_state()

    def _reset_state(self) -> None:
        self.hidden = None       # set by spec_decode (needs_hidden); unused
        self.pos = 0             # absolute position of the NEXT token
        # Context K/V live in preallocated per-layer buffers (1, n_kv, cap, hd),
        # RoPE applied; the valid keys are [_off, _off + _n). Appending writes in
        # place and a draft block is written into the scratch room after the
        # context, so neither copies the context (torch.cat did, every step:
        # a transient second copy of an O(context) cache).
        self._kb = [None] * self.n_layers
        self._vb = [None] * self.n_layers
        self._off = 0
        self._n = 0
        self._ctx_start = 0      # absolute position of the first valid key
        self._ctx_end = 0        # one past the last observed position
        self.blocks = 0
        self._graph = None       # dflash_graph.DraftGraph, created lazily; False = capture failed, stay eager

    @property
    def _ctx_k(self):
        return [None if b is None or not self._n else b[:, :, self._off:self._off + self._n]
                for b in self._kb]

    @property
    def _ctx_v(self):
        return [None if b is None or not self._n else b[:, :, self._off:self._off + self._n]
                for b in self._vb]

    def _set_ctx(self, ks, vs) -> None:
        self._kb = [None] * self.n_layers
        self._vb = [None] * self.n_layers
        self._off = 0
        self._n = 0 if ks[0] is None else ks[0].shape[2]
        if self._n:
            self._reserve(0)
            for i in range(self.n_layers):
                self._kb[i][:, :, :self._n] = ks[i]
                self._vb[i][:, :, :self._n] = vs[i]

    def _reserve(self, extra: int) -> None:
        """Room for `extra` more keys plus one draft block after the context."""
        need = self._n + extra + self.block_size
        cap = 0 if self._kb[0] is None else self._kb[0].shape[2]
        if self._off + need <= cap:
            return
        if need <= cap and self._off:            # trimmed front: slide back in place
            for bufs in (self._kb, self._vb):
                for b in bufs:
                    b[:, :, :self._n] = b[:, :, self._off:self._off + self._n].clone()
            self._off = 0
            return
        new = max(need + need // 2, need + 1024)
        if self._ctx_cap:
            new = min(new, max(need, self._ctx_cap + 2 * 1024))
        for bufs in (self._kb, self._vb):
            for i in range(self.n_layers):        # one layer at a time: small peak
                # zeros, not empty: dflash_graph attends the whole buffer under a mask, and a masked key with
                # garbage (NaN/Inf) V would still poison the output (0 * NaN)
                b = torch.zeros(1, self.n_kv, new, self.hd, dtype=self.dtype, device=self.device)
                if bufs[i] is not None and self._n:
                    b[:, :, :self._n] = bufs[i][:, :, self._off:self._off + self._n]
                bufs[i] = b
        self._off = 0

    def export_state(self) -> dict:
        """The drafter's per-sequence context (bounded by the sliding window), so
        a sequence resumed from a saved prefix can draft without re-observing it."""
        c = lambda ts: [None if t is None else t.clone() for t in ts]
        return {"pos": self.pos, "ctx_k": c(self._ctx_k), "ctx_v": c(self._ctx_v),
                "ctx_start": self._ctx_start, "ctx_end": self._ctx_end, "blocks": self.blocks}

    def import_state(self, st: dict) -> None:
        self.pos, self._ctx_start, self._ctx_end, self.blocks = (
            st["pos"], st["ctx_start"], st["ctx_end"], st["blocks"])
        self._set_ctx(st["ctx_k"], st["ctx_v"])

    def fork(self) -> "DFlashDrafter":
        """A drafter for another sequence that SHARES this one's weights
        (gigabytes) and has its own, empty context. ContinuousBatchEngine
        needs one drafter per slot; building each from disk would load the
        draft model once per concurrent request."""
        d = copy.copy(self)
        d._reset_state()
        return d

    # Gathered by row, not multiplied: a packed module cannot serve them.
    _DENSE_ONLY = ("candidate_selector.predecessor_codebook",
                   "candidate_selector.successor_codebook")

    @staticmethod
    def _is_pack_dir(d: str) -> bool:
        try:
            man = json.load(open(os.path.join(d, "manifest.json")))
        except (OSError, ValueError):
            return False
        return bool(man.get("tensors")) and not man.get("component_layout")

    def _swap_packed(self, w: dict, pack_dir: str) -> dict:
        """Replace every tensor a qvb8 pack of this checkpoint covers with its
        packed linear module; the rest (norms, conv base kernels, codebooks,
        anything the packer refused) stays the dense tensor already loaded."""
        from qv_band.common.formats import get_format
        man = json.load(open(os.path.join(pack_dir, "manifest.json")))
        get_format(man.get("format"))                 # an unknown pack format fails here, loudly
        if man.get("component_layout"):
            raise ValueError(f"{pack_dir} is a per-component pack, not a draft model pack")
        n = 0
        self.param_bytes = 0
        # Each tensor is read by its OWN file extension, as the model loader does: a qvb8r pack keeps a
        # few tensors as .qvband.
        spec_of = lambda path: get_format(os.path.splitext(path)[1].lstrip("."))
        # Tensor parallel: every rank runs the drafter in lockstep (same
        # inputs, rank 0's draft broadcast), so its linears can be row-sharded
        # like the target's -- each rank holds 1/world of the drafter instead
        # of a full replica (2.5 GB -> 1.25 GB per card on the 27B's DFlash2).
        # A format without a sharded reader (qvb8r) loads a full replica on
        # every rank instead -- what every format did before sharding existed.
        import torch.distributed as dist
        tp_world = dist.get_world_size() if dist.is_initialized() else 1
        replicated: list = []
        if tp_world > 1:
            tp_rank = dist.get_rank()

            def _load_linear(path):
                spec = spec_of(path)
                if spec.sharded_linear is None:
                    if not replicated:
                        print(f"DFlash2: format {spec.name} has no sharded reader -- "
                              f"full drafter replica on every rank", flush=True)
                    replicated.append(path)
                    return spec.linear.from_packfile(path, device=str(self.device))
                return spec.sharded_linear.from_packfile(
                    path, rank=tp_rank, world=tp_world, device=str(self.device))
        else:
            def _load_linear(path):
                return spec_of(path).linear.from_packfile(path, device=str(self.device))
        for name, info in man.get("tensors", {}).items():
            if name in self._DENSE_ONLY:
                if name not in w and pack_dir == self._draft_dir:
                    # A pack used as its own checkpoint: the drafter indexes this tensor like a table, so it
                    # must be dense. Decode it by pushing an identity through the packed layer (exactly the
                    # decoded weight: every output is one weight times 1.0).
                    lin = spec_of(info["pack"]).linear.from_packfile(os.path.join(pack_dir, info["pack"]), device=str(self.device))
                    eye = torch.eye(info["P"], dtype=torch.float16, device=self.device)
                    w[name] = lin(eye).t().contiguous().to(self.dtype)
                    del lin, eye
                continue
            # A pack of a DIFFERENT checkpoint must not load silently: it
            # would give plausible-looking but wrong drafts.
            # (a pack used as its own checkpoint has no dense tensor for what it packed: nothing to compare)
            standalone = pack_dir == self._draft_dir
            if name in w and tuple(w[name].shape) != (info["Q"], info["P"]) or (name not in w and not standalone):
                raise ValueError(
                    f"{pack_dir} does not match this draft model: {name} is "
                    f"{(info.get('Q'), info.get('P'))} in the pack, "
                    f"{tuple(w[name].shape) if name in w else 'absent'} here "
                    f"(pack source: {man.get('hf_model')})")
            path = os.path.join(pack_dir, info["pack"])
            w[name] = _load_linear(path)
            self.param_bytes += info["bytes"] if (tp_world == 1 or path in replicated) else info["bytes"] // tp_world
            n += 1
        self.param_bytes += sum(v.numel() * v.element_size()
                                for v in w.values() if isinstance(v, torch.Tensor))
        print(f"DFlash2: {n} tensors from {man.get('format')} pack {pack_dir}", flush=True)
        return w

    # -- rope ---------------------------------------------------------------
    def _cos_sin(self, positions: torch.Tensor):
        f = positions.float()[:, None] * self.inv_freq[None]
        emb = torch.cat((f, f), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _rope(x, cos, sin):                 # x: (1, heads, n, hd)
        return x * cos + _rotate_half(x) * sin

    def _kv(self, L, x, cos, sin):
        n = x.shape[0]
        k = _lin(x, L["self_attn.k_proj"]).view(n, self.n_kv, self.hd)
        v = _lin(x, L["self_attn.v_proj"]).view(n, self.n_kv, self.hd)
        k = _rms(k, L["self_attn.k_norm"], self.eps).transpose(0, 1)[None]
        return self._rope(k, cos, sin), v.transpose(0, 1)[None]

    # -- context ------------------------------------------------------------
    @torch.no_grad()
    def observe_features(self, first_pos: int, feats: torch.Tensor) -> None:
        if first_pos < self._ctx_end:          # rollback / new request
            keep = max(0, first_pos - self._ctx_start)
            if keep == 0:
                self._off = self._n = 0
                self._ctx_start = first_pos
            else:
                self._n = min(self._n, keep)
            self._ctx_end = first_pos
        elif first_pos > self._ctx_end:        # gap: older context is unusable
            self._off = self._n = 0
            self._ctx_start = self._ctx_end = first_pos
        n = feats.shape[0]
        if n == 0:
            return
        th = _rms(_lin(feats.to(self.device, self.dtype), self.fc),
                  self.hidden_norm, self.eps)
        self._last_th = th[-1]
        self._last_feats = feats[-1]
        cos, sin = self._cos_sin(torch.arange(first_pos, first_pos + n, device=self.device))
        self._reserve(n)
        a, b = self._off + self._n, self._off + self._n + n
        for i, L in enumerate(self.layers):
            k, v = self._kv(L, th, cos, sin)
            self._kb[i][:, :, a:b] = k
            self._vb[i][:, :, a:b] = v
        self._n += n
        self._ctx_end = first_pos + n
        # Keys further than the sliding window from every future query are
        # never visible again; drop them so the context stays bounded.
        # QV_DRAFT_CTX caps it further (any layer type): the drafter then
        # attends only the most recent keys -- fewer accepted drafts on very
        # long prompts, never a wrong token (the target verifies every one).
        keep = min(x for x in (self.ctx_keep, self._ctx_cap) if x) if (
            self.ctx_keep or self._ctx_cap) else None
        if keep is not None:
            drop = self._n - keep
            if drop > 0:
                self._off += drop
                self._n -= drop
                self._ctx_start += drop

    # -- harness bookkeeping -------------------------------------------------
    def reset(self, tokens: list[int]) -> None:
        # The harness observes the prompt's features BEFORE reset(), so the
        # context end is the prompt's absolute length. Prefer it to
        # len(tokens): ContinuousBatchEngine hands reset() only the prompt
        # TAIL that went through its shared step after chunked prefill.
        self.pos = self._ctx_end or len(tokens)

    def append(self, tok: int) -> None:
        self.pos += 1

    def feedback(self, n_accepted: int, n_proposed: int) -> None:
        pass

    # -- drafting -----------------------------------------------------------
    def _attend(self, q, k, v, mask, layer_idx):
        # GQA by folding each kv head's query group into the query rows: (1, H, bs, d) -> (1, kvh, g*bs, d).
        # enable_gqa on SDPA's math backend (V100: no flash) copies K and V out to all H heads first
        # (2 x 34 MB at a 4096-token context) -- here K and V are read in place.
        b, H, bs, d = q.shape
        kvh = k.shape[1]
        if kvh == H:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        g = H // kvh
        if mask is not None:
            mask = mask.expand(b, 1, bs, mask.shape[-1]).repeat(1, 1, g, 1)
        o = F.scaled_dot_product_attention(q.reshape(b, kvh, g * bs, d), k, v, attn_mask=mask)
        return o.reshape(b, H, bs, d)   # SDPA with a mask can return a non-viewable layout

    @torch.no_grad()
    def _block_hidden(self, last_token: int, start: int) -> torch.Tensor:
        bs = self.block_size
        ids = torch.full((bs,), self.mask_token_id, dtype=torch.long, device=self.device)
        ids[0] = last_token
        h = self.embed(ids)
        self._target_dtype = h.dtype      # what the target's lm_head consumes
        h = h.to(self.dtype)                                     # (bs, H)
        q_pos = torch.arange(start, start + bs, device=self.device)
        cos, sin = self._cos_sin(q_pos)
        k_pos = torch.cat([torch.arange(self._ctx_start, self._ctx_end, device=self.device), q_pos])
        d = q_pos[:, None] - k_pos[None, :]     # query minus key position
        masks = {}
        for causal, win in set(self.layer_attn):
            m = torch.ones_like(d, dtype=torch.bool)
            if causal:
                m &= d >= 0
            if win is not None:
                m &= d < win
                if not causal:
                    m &= -d < win
            masks[(causal, win)] = None if bool(m.all()) else m[None, None]
        for i, L in enumerate(self.layers):
            mask = masks[self.layer_attn[i]]
            res = h
            x = _rms(h, L["input_layernorm"], self.eps)
            dyn = None
            if L["attn_conv"] is not None:
                x, dyn = L["attn_conv"].prepare(x[None])
                x = x[0]
            q = _lin(x, L["self_attn.q_proj"]).view(bs, self.n_heads, self.hd)
            q = self._rope(_rms(q, L["self_attn.q_norm"], self.eps).transpose(0, 1)[None], cos, sin)
            k, v = self._kv(L, x, cos, sin)
            if self._n:                            # block keys go in the scratch room
                self._reserve(0)
                a0 = self._off + self._n
                self._kb[i][:, :, a0:a0 + bs] = k
                self._vb[i][:, :, a0:a0 + bs] = v
                k = self._kb[i][:, :, self._off:a0 + bs]
                v = self._vb[i][:, :, self._off:a0 + bs]
            a = self._attend(q, k, v, mask, i)
            a = _lin(a[0].transpose(0, 1).reshape(bs, -1), L["self_attn.o_proj"])
            if dyn is not None:
                a = L["attn_conv"].finish(a[None], dyn)[0]
            h = res + a
            res = h
            x = _rms(h, L["post_attention_layernorm"], self.eps)
            dyn = None
            if L["mlp_conv"] is not None:
                x, dyn = L["mlp_conv"].prepare(x[None])
                x = x[0]
            x = _lin(F.silu(_lin(x, L["mlp.gate_proj"])) * _lin(x, L["mlp.up_proj"]),
                         L["mlp.down_proj"])
            if dyn is not None:
                x = L["mlp_conv"].finish(x[None], dyn)[0]
            h = res + x
        return _rms(h, self.norm, self.eps)[1:]                  # (bs-1, H)

    @torch.no_grad()
    def propose(self, last_token: int, k: int, temperature: float = 0.0,
                top_k: int = 0, top_p: float = 1.0):
        """Draft block_size-1 tokens in one pass and return the first k.

        The whole block is always drafted, even for k < block_size-1: the
        draft layers are bidirectional over the block, so a shorter block
        would change the earlier positions too, away from what was trained.
        """
        if k <= 0:
            return [], None
        start = self.pos - 1                    # position of last_token
        if self._ctx_end != start:
            # Context out of step with the stream (e.g. the harness skipped
            # a feature capture). Never guess positions: no draft this step.
            return [], None
        if temperature <= 0 and self._graph is not False:
            if self._graph is None:
                from qv_band.decode.dflash_graph import DraftGraph, graph_enabled
                self._graph = DraftGraph(self) if graph_enabled(self) else False
            if self._graph:
                toks = self._graph.propose(last_token, start)
                if toks is not None:
                    self.blocks += 1
                    return toks[:k], None
        hid = self._block_hidden(last_token, start)
        logits = self.lm_head(hid.to(self._target_dtype))       # (bs-1, V)
        if self.version == 1:
            return self._propose_v1(logits.float(), k, temperature, top_k, top_p)
        unary, cand = torch.topk(logits.float(), self.top_k, dim=-1, sorted=False)
        hp = _lin(hid, self.sel_proj).float()                # (bs-1, r)
        cc = cand.cpu()                                          # gather only the candidate rows of the host codebooks
        succ = self.sel_succ[cc].to(self.device, non_blocking=True).float()      # (bs-1, K, r)
        psub = self.sel_pred[cc].to(self.device, non_blocking=True)              # (bs-1, K, r)
        prow = self.sel_pred[last_token].to(self.device, non_blocking=True)
        path, q_rows = [], []
        for j in range(hid.shape[0]):
            scores = unary[j] + succ[j] @ (prow.float() * hp[j])
            if temperature > 0:
                q = torch.softmax(scores / temperature, dim=-1)
                idx = torch.multinomial(q, 1)[0]
                q_rows.append(q)
            else:
                idx = scores.argmax()
            prow = psub[j, idx]
            path.append(cand[j, idx])
        self.blocks += 1
        n = min(k, len(path))
        tokens = torch.stack(path[:n]).tolist()
        if temperature <= 0:
            return tokens, None
        V = logits.shape[-1]
        probs = [torch.zeros(V, device=self.device).scatter_(0, cand[j], q_rows[j])
                 for j in range(n)]
        return tokens, probs

    def _propose_v1(self, logits, k, temperature, top_k, top_p):
        """DFlash v1: argmax of the draft logits, or a sample from the same
        filtered distribution the harness samples the target with (so the
        returned probabilities ARE the proposal distribution)."""
        n = min(k, logits.shape[0])
        self.blocks += 1
        if temperature <= 0:
            return logits[:n].argmax(-1).tolist(), None
        from qv_band.decode.spec_decode import _probs
        probs = [_probs(logits[j], temperature, top_k, top_p) for j in range(n)]
        return [int(torch.multinomial(p, 1)) for p in probs], probs


class NGramWithDFlashDrafter:
    """N-gram lookup and a DFlash block drafter in ONE draft (the DFlash counterpart of NGramWithMTPDrafter).

    n-gram goes first: where the text demonstrably repeats it costs nothing and is accepted ~90% of the time. If it
    fills the whole budget, the DFlash forward is skipped. Otherwise DFlash drafts its block and its tokens at the
    positions n-gram did not fill are appended (a short n-gram hit is extended, no hit = plain DFlash). DFlash's
    block is bidirectional, so a block is never shortened -- only its tail is spliced after the n-gram tokens.

    Both children see every committed token (reset/append) and the target features (observe_features go to DFlash
    only). The n-gram's miss back-off is driven by ITS tokens only, as in NGramWithMTPDrafter. At temperature > 0 the
    n-gram positions carry None (a point mass); the DFlash positions carry its real proposal probabilities."""

    needs_hidden = True
    returns_probs = True

    def __init__(self, ngram, dflash, min_ngram: int = 1):
        self.ngram, self.dflash, self.min_ngram = ngram, dflash, max(1, int(min_ngram))
        self._ng_len = 0
        self.ngram_steps = self.steps = 0
        self.hidden = None

    capture_layers = property(lambda self: self.dflash.capture_layers)
    block_size = property(lambda self: self.dflash.block_size)
    param_bytes = property(lambda self: self.dflash.param_bytes)
    version = property(lambda self: self.dflash.version)

    def observe_features(self, first_pos, feats):
        self.dflash.observe_features(first_pos, feats)

    def reset(self, tokens):
        self.ngram.reset(tokens)
        self.dflash.reset(tokens)

    def append(self, tok):
        self.ngram.append(tok)
        self.dflash.append(tok)

    def propose(self, last_token, k, temperature=0.0, top_k=0, top_p=1.0):
        self.steps += 1
        ng, _ = self.ngram.propose(last_token, k)
        if len(ng) < self.min_ngram:
            ng = []
        self._ng_len = len(ng)
        if ng:
            self.ngram_steps += 1
        if len(ng) >= k:
            return ng[:k], ([None] * k if temperature > 0 else None)
        toks, probs = self.dflash.propose(last_token, k, temperature, top_k, top_p)
        if not ng:
            return toks, probs
        n = len(ng)
        tail = toks[n:k]
        out = ng + tail
        if probs is None:
            return out, ([None] * n + [None] * len(tail) if temperature > 0 else None)
        return out, [None] * n + list(probs[n:k])

    def feedback(self, n_accepted, n_proposed):
        if self._ng_len and hasattr(self.ngram, "feedback"):
            self.ngram.feedback(min(n_accepted, self._ng_len), self._ng_len)
        self.dflash.feedback(n_accepted, n_proposed)

    def fork(self):
        from qv_band.decode.ngram_draft import NGramDrafter
        ng = NGramDrafter(max_ctx=self.ngram.max_ctx, min_ctx=self.ngram.min_ctx)
        return NGramWithDFlashDrafter(ng, self.dflash.fork(), self.min_ngram)

    def export_state(self):
        return {"dflash": self.dflash.export_state(), "tokens": list(self.ngram.tokens)}

    def import_state(self, st):
        self.dflash.import_state(st["dflash"])
        self.ngram.reset(st["tokens"])


# Where find_dflash_draft looks; QV_BAND_DRAFT_ROOTS (":"-separated) overrides.
# /mnt/backup/models holds the raw HF checkpoints that were moved off the
# boot disk -- without it a moved model silently loses its drafter.
DRAFT_ROOTS = ("/mnt/data/models_raw", "/mnt/data/models",
               "/mnt/backup/models", "/mnt/backup/models_raw")


def target_name(model_dir: str) -> str:
    """The checkpoint name a drafter is published under, e.g.
    "Qwen3.8-27B" for the pack /mnt/data/models/Qwen3.8-27B_qvb8_1e-3_bf16:
    the basename of the pack manifest's hf_model, else of `model_dir`
    itself with any "_qvb..." suffix removed."""
    man = os.path.join(model_dir, "manifest.json")
    if os.path.exists(man):
        hf = json.load(open(man)).get("hf_model")
        if hf:
            return os.path.basename(os.path.normpath(hf))
    return os.path.basename(os.path.normpath(model_dir)).split("_qvb", 1)[0]


def find_dflash_draft(model_dir: str, roots=None) -> str | None:
    """The DFlash draft model for the target at `model_dir`, or None.

    Looks for "<name>-DFlash2" then "<name>-DFlash" (dense HF checkpoints:
    config.json + model.safetensors) directly under each root or one
    organisation level down (models_raw/incoai/Qwen3.8-27B-DFlash2), so
    downloaded drafters are found with no configuration. DFlash 2 wins over
    v1. Whether it really fits the target is checked by build_dflash_drafter.
    """
    if roots is None:
        env = os.environ.get("QV_BAND_DRAFT_ROOTS")
        roots = env.split(":") if env else DRAFT_ROOTS
    name = target_name(model_dir)
    for suffix in ("-DFlash2", "-DFlash"):
        for root in roots:
            for cand in sorted(glob.glob(os.path.join(root, name + suffix))
                               + glob.glob(os.path.join(root, "*", name + suffix))):
                if (os.path.isfile(os.path.join(cand, "config.json"))
                        and os.path.isfile(os.path.join(cand, "model.safetensors"))):
                    return cand
    return None


def find_dflash_pack_for(draft_dir: str, roots=None) -> str | None:
    """The qvb8 pack of a DRAFT checkpoint at `draft_dir`, or None.

    The counterpart of find_dflash_draft for the explicit-checkpoint flow:
    `draft_dir` is the dense DFlash directory itself (e.g.
    /mnt/backup/models/incoai/Qwen3.8-27B-DFlash2), and the pack is published
    beside it as "<basename>_qvb8_*" (possibly one organisation level down,
    like the dense drafter itself). The drafter's own basename is used, so a
    drafter that is not the target's default sibling is packed correctly too.
    """
    if roots is None:
        env = os.environ.get("QV_BAND_DRAFT_ROOTS")
        roots = env.split(":") if env else DRAFT_ROOTS
    name = os.path.basename(os.path.normpath(draft_dir))
    for root in roots:
        pat = os.path.join(root, name + "_qvb8_*")
        for cand in sorted(glob.glob(pat)
                           + glob.glob(os.path.join(root, "*", name + "_qvb8_*"))):
            man = os.path.join(cand, "manifest.json")
            if not os.path.isfile(man):
                continue
            try:
                m = json.load(open(man))
            except Exception:
                continue
            if m.get("format") != "qvband" or m.get("component_layout"):
                continue
            return cand
    return None


def suggest_dflash_draft(draft_dir: str, roots=None) -> str | None:
    """The DFlash checkpoint that most likely WAS meant by `draft_dir`.

    A mistyped path is the common failure ("...-DFlash" for the published
    "...-DFlash2", or the wrong organisation folder). Rather than only saying
    "not a directory", name the checkpoint that is actually on disk.

    The rules, in order:
      1. the same basename with the other DFlash suffix ("-DFlash" <-> "-DFlash2");
      2. any DFlash checkpoint whose basename contains the wanted basename
         (the wanted name with the suffix stripped), i.e. a near miss.

    Returns None when nothing plausible is on disk.
    """
    if roots is None:
        env = os.environ.get("QV_BAND_DRAFT_ROOTS")
        roots = env.split(":") if env else DRAFT_ROOTS
    base = os.path.basename(os.path.normpath(draft_dir))

    def _is_draft(cand: str) -> bool:
        return (os.path.isfile(os.path.join(cand, "config.json"))
                and os.path.isfile(os.path.join(cand, "model.safetensors")))

    def _all_draft_dirs():
        for root in roots:
            for cand in sorted(glob.glob(os.path.join(root, "*DFlash*"))
                               + glob.glob(os.path.join(root, "*", "*DFlash*"))):
                if os.path.isdir(cand) and _is_draft(cand):
                    yield cand

    # 1. the same name with the other suffix ("-DFlash" <-> "-DFlash2")
    for suffix in ("-DFlash2", "-DFlash"):
        if base.endswith("-DFlash2"):
            other = base[: -len("-DFlash2")] + suffix
        elif base.endswith("-DFlash"):
            other = base[: -len("-DFlash")] + suffix
        else:
            continue
        for cand in _all_draft_dirs():
            if os.path.basename(cand) == other:
                return cand

    # 2. a near miss: any DFlash dir whose name CONTAINS the wanted stem
    stem = base
    for suffix in ("-DFlash2", "-DFlash"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem:
        for cand in _all_draft_dirs():
            if stem in os.path.basename(cand):
                return cand
    return None


def build_dflash_drafter(model, draft_dir: str, device: str = "cuda",
                         pack_dir: str | None = None):
    """Bind a DFlash (v1 or v2) checkpoint to a loaded target -- its embedding
    and lm_head -- after checking the checkpoint was trained for this target.
    `pack_dir`: a qvb8 pack of `draft_dir` (pack_model.py) for the linears."""
    n_target = model.config.num_hidden_layers
    cfg = json.load(open(os.path.join(draft_dir, "config.json")))
    if "LiLiCorrDraftModel" in (cfg.get("architectures") or []):      # Nemotron-3.5-Bolt LiLiCorr drafter
        from qv_band.decode.lilicorr_draft import LiLiCorrDrafter
        cls = LiLiCorrDrafter
    else:
        cls = DFlashDrafter
    d = cls(draft_dir, model.get_input_embeddings(), model.get_output_embeddings(),
            device=device, pack_dir=pack_dir)
    if cfg.get("num_target_layers", n_target) != n_target or max(d.capture_layers) >= n_target:
        raise ValueError(f"{draft_dir} was trained for a {cfg.get('num_target_layers')}-layer "
                         f"target, this model has {n_target} layers")
    if d.fc_in != len(d.capture_layers) * model.config.hidden_size:
        raise ValueError(f"{draft_dir} expects {d.fc_in // len(d.capture_layers)}-wide "
                         f"target hidden states, this model has {model.config.hidden_size}")
    if cfg.get("vocab_size") not in (None, model.config.vocab_size):
        raise ValueError(f"vocab mismatch: draft {cfg.get('vocab_size')} vs "
                         f"target {model.config.vocab_size}")
    print(f"{'LiLiCorr' if d.version == 3 else 'DFlash%d' % d.version} drafter: {d.param_bytes/1e9:.2f} GB on {device}, block "
          f"{d.block_size}, target layers {d.capture_layers}", flush=True)
    return d


# Names from when only DFlash 2 was supported.
DFlash2Drafter = DFlashDrafter
build_dflash2_drafter = build_dflash_drafter


def build_ngram_dflash_drafter(model, draft_dir: str, device: str = "cuda",
                               pack_dir: str | None = None, max_ctx: int = 4, min_ngram: int = 1):
    """DFlash (v1 / v2 / LiLiCorr) drafter wrapped with an n-gram lookup of `max_ctx` tokens of context."""
    from qv_band.decode.ngram_draft import NGramDrafter
    d = build_dflash_drafter(model, draft_dir, device=device, pack_dir=pack_dir)
    return NGramWithDFlashDrafter(NGramDrafter(max_ctx=max_ctx), d, min_ngram)
