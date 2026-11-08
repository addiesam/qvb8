"""Per-draft-token recurrent-state ring, llama.cpp-PR-#22400-style.

WHY this exists: on a hybrid linear-attention model (Qwen3.5/3.6/3.8's Gated
DeltaNet layers), a speculative-decoding draft batch is verified with ONE
batched forward over all k drafted tokens. That forward's chunked scan
kernel (`chunk_gated_delta_rule`, see fla.ops.gated_delta_rule.chunk) only
ever returns the FINAL recurrent state after all k tokens -- there is no
public API to get "the state after just the first n_acc of them", which is
exactly what's needed on a PARTIAL accept (0 < n_acc < k). Measured on this
project's real traffic: 59% of rejection events are partial accepts, and
every one of them was falling back to a full O(prefix) replay for lack of
anything cheaper (see spec_decode.py's rebuild path).

Rejected two cheaper alternatives before building this (see conversation /
session notes): (a) k sequential single-token model forwards to get
per-token states "for free" -- measured 2.66x the batched verify's own
cost, paid on EVERY step regardless of whether that step needs it, which
nets WORSE than the status quo at this project's measured accept rate; (b)
patching the Triton chunk kernel itself to emit intermediate states --
higher risk, version-fragile (transformers/fla upgrade breaks it silently),
ruled out in favour of this instead.

The approach actually used, matching what llama.cpp's PR #22400/#28123
does: after the ONE batched verify forward already ran (for its logits --
that cost is unavoidable and already paid), re-derive the k intermediate
recurrent states from data that forward ALREADY computed as local
variables inside Qwen3_5GatedDeltaNet.forward -- q/k/v/g/beta per position,
the inputs to the chunked scan -- by feeding them one token at a time to
`fused_recurrent_gated_delta_rule` (the SAME fused single-step kernel the
model itself already uses for seq_len==1 decode, see modeling_qwen3_5.py
line ~528). This is cheap: it is ONLY the recurrent-scan math (a handful of
small matmuls per layer per token, no conv1d/attention/MLP/projections --
those are already correctly computed once by the shared batched forward),
not a second model forward.

Cost paid: k small recurrent-kernel calls per linear-attention layer, per
speculative step (regardless of whether that step is a full accept, so
this IS overhead on top of every step, same as the snapshot/restore this
replaces) -- but the recurrent kernel alone is a tiny fraction of a full
forward's cost (no attention/MLP/projections), unlike alternative (a)
above which was a FULL extra forward per token and measured to lose.
"""
from __future__ import annotations

import types

import torch

_PATCHED_MARK = "_qv_ring_patched"


def _patched_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """Drop-in replacement for Qwen3_5GatedDeltaNet.forward that additionally
    stashes the per-position (query, key, value, g, beta) tensors fed to the
    chunked scan onto `cache_params.layers[self.layer_idx]` as `_qv_ring_qkvgb`,
    whenever `cache_params` carries a `_qv_ring_capture = True` flag. Byte-
    for-byte identical to the original otherwise -- this only ADDS an
    attribute write, it does not change any tensor computation or control
    flow the original had.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
    import torch.nn.functional as F

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

    if use_precomputed_states:
        conv_state = cache_params.layers[self.layer_idx].conv_states[0]
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states and seq_len == 1:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation,
        )
    else:
        if use_precomputed_states:
            mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
        # Captured BEFORE the conv runs and BEFORE mixed_qkv gets reassigned
        # by it below -- this is the (batch, channels, prev_conv_len+seq_len)
        # raw window update_conv_state derives conv_states from (same tensor
        # `new_conv_state` below is padded/sliced from). Needed by the ring
        # builder to reconstruct conv_states at intermediate prefix lengths;
        # unconditional (not gated on _qv_ring_capture) because seq_len==1
        # decode never reaches this branch and the extra reference here is
        # cheap (no copy, same storage) -- freed with everything else in
        # this frame if capture wasn't armed and nothing reads it.
        _pre_conv_window = mixed_qkv
        if cache_params is not None:
            new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.update_conv_state(new_conv_state, self.layer_idx)
        # causal_conv1d is an optional fast CUDA kernel. Some valid
        # Transformers/Qwen3.5 combinations do not expose it; fall back to
        # the framework's equivalent convolution path instead of crashing.
        causal_conv1d_fn = getattr(self, 'causal_conv1d_fn', None)
        if causal_conv1d_fn is not None:
            mixed_qkv = causal_conv1d_fn(
                x=mixed_qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
                activation=self.activation, seq_idx=kwargs.get("seq_idx"),
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
        if use_precomputed_states:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    # --- the only addition: stash the scan's own inputs for the caller to
    # replay through the cheap single-step kernel afterwards.
    if cache_params is not None and getattr(cache_params, "_qv_ring_capture", False) and seq_len > 1:
        cache_params.layers[self.layer_idx]._qv_ring_qkvgb = (
            query.detach(), key.detach(), value.detach(), g.detach(), beta.detach(),
        )
        # .clone() (not .detach()) matters here: update_recurrent_state
        # below does an IN-PLACE self.recurrent_states[state_idx].copy_(...)
        # "to preserve the static address for cudagraphs" (see
        # transformers/cache_utils.py's LinearAttentionLayer.
        # update_recurrent_state), so `recurrent_state` is the SAME tensor
        # object that gets overwritten by the end of this very forward call
        # -- .detach() alone still shares that storage and would silently
        # become the POST-step state by the time build_ring reads it. Real
        # bug, caught by test_ring_e2e.py's recurrent_states diff (~4-10,
        # not the ~1e-3 the underlying kernel math itself produces).
        cache_params.layers[self.layer_idx]._qv_ring_initial_state = (
            recurrent_state.clone() if use_precomputed_states else None
        )
        # Slicing _pre_conv_window per intermediate prefix length gives the
        # exact conv_states an actual forward through only i of the k draft
        # tokens would have produced -- no kernel replay needed, it's
        # already sitting in this tensor from the one batched forward.
        # (seq_len > 1 is required to even reach this point, so the
        # seq_len==1 branch above -- which never defines _pre_conv_window --
        # cannot land here.)
        cache_params.layers[self.layer_idx]._qv_ring_conv_before = _pre_conv_window.detach()
        cache_params.layers[self.layer_idx]._qv_ring_conv_kernel_size = self.conv_kernel_size

    if use_precomputed_states and seq_len == 1:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=recurrent_state,
            output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state if use_precomputed_states else None,
            output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs.get("cu_seq_lens_q"),
        )

    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    core_attn_out = self.out_proj(core_attn_out)
    return core_attn_out


def patch_model_for_ring_capture(model) -> int:
    """Rebind every Qwen3_5GatedDeltaNet instance's forward to the capturing
    version. Idempotent -- safe to call more than once (checks a marker
    attribute per module, and only imports/patches the class if a real
    instance is found, so this is a no-op cost on a non-hybrid model).
    Returns how many layers were patched (0 on a non-hybrid model, meaning
    the ring-snapshot path is simply unavailable and callers should fall
    back to full replay unchanged).
    """
    n = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        # The ring-capture replacement targets the legacy Qwen3.5 kernel
        # interface. Newer Transformers releases moved these to module-level
        # FLA functions; in that case leave the native forward untouched so
        # QV Band falls back to its ordinary cache path instead of replacing
        # a working implementation with an incompatible one.
        if not all(hasattr(module, name) for name in (
            'chunk_gated_delta_rule', 'recurrent_gated_delta_rule',
            'causal_conv1d_update')):
            # transformers >= 5: the scan/conv are module-level functions, so
            # the legacy forward copy above cannot be used. Capture them the
            # way the batch engine does instead (see _patch_modern).
            if _patch_modern(module):
                n += 1
            continue
        if getattr(module, _PATCHED_MARK, False):
            n += 1
            continue
        module.forward = types.MethodType(_patched_forward, module)
        setattr(module, _PATCHED_MARK, True)
        n += 1
    return n


_MODERN_MARK = "_qv_ring_modern_patched"


def _patch_modern(module) -> bool:
    """transformers >= 5 capture: wrap the module's forward so a verify-width
    call made while capture is armed records the scan inputs and conv window that
    _install_ring_capture's module-level wrappers see, and stash them on the
    cache layer as `_qv_ring_modern`. Nothing is computed here; promote_ring
    replays only the accepted prefix, and only when a draft was rejected.
    Same capture the batch engine uses (spec_decode._ring_call)."""
    import os
    if os.environ.get("QV_BAND_RING", "1") == "0":
        return False
    if getattr(module, _MODERN_MARK, False):
        return True
    from qv_band.decode import spec_decode as sd
    if not sd._install_ring_capture_for(module):
        return False
    orig_forward = module.forward

    def forward(hidden_states, cache_params=None, attention_mask=None,
                _orig=orig_forward, _mod=module, **kwargs):
        n_tok = hidden_states.shape[1]
        if (cache_params is None or not getattr(cache_params, "_qv_ring_capture", False)
                or n_tok <= 1):
            return _orig(hidden_states=hidden_states, cache_params=cache_params,
                         attention_mask=attention_mask, **kwargs)
        layer = cache_params.layers[_mod.layer_idx]
        layer._qv_ring_modern = None            # never promote a previous step's capture
        ctx = {}
        sd._ring_ctx_set(ctx)
        try:
            out = _orig(hidden_states=hidden_states, cache_params=cache_params,
                        attention_mask=attention_mask, **kwargs)
        finally:
            sd._ring_ctx_set(None)
        if "scan" in ctx and "win" in ctx and ctx["scan"][5] is not None:
            layer._qv_ring_modern = (ctx["scan"], ctx["win"], n_tok)
        return out

    module.forward = forward
    setattr(module, _MODERN_MARK, True)
    return True


def _modern_layers(past):
    """[(layer, (scan, win, n_tok))] for every linear-attention layer, or None
    if any of them has no usable capture for this step."""
    out = []
    for layer in getattr(past, "layers", []) or []:
        rs = getattr(layer, "recurrent_states", None)
        if rs is None or rs.get(0) is None:
            continue                            # full-attention layer
        cap = getattr(layer, "_qv_ring_modern", None)
        if cap is None:
            return None
        out.append((layer, cap))
    return out or None


@torch.no_grad()
def build_ring(past, n_seq_tok: int) -> bool:
    """After a captured batched verify forward, compute exact per-token
    recurrent states for prefix lengths 0..n_seq_tok by replaying the
    captured (query, key, value, g, beta) through the cheap single-step
    kernel, one token at a time, per linear-attention layer.

    `n_seq_tok` is the length of the WHOLE verify-forward input
    (spec_decode.py's `seq = [out_ids[-1]] + draft`, so n_seq_tok ==
    len(draft) + 1), NOT the draft length alone -- ring[i] is the state
    after applying the first i tokens of `seq`, so ring[0] is the pre-step
    state (before even seq[0]) and ring[n_acc + 1] (not ring[n_acc]!) is
    the state after seq[0] plus the first n_acc accepted drafts. Passing
    len(draft) here instead of len(seq) silently truncates the ring by one
    slot short and was a real off-by-one bug caught by an end-to-end
    correctness test (ring-promoted state vs a true sequential replay
    diverged: conv_states max diff ~36, not the ~1e-3 bf16-noise level the
    underlying kernel math itself produces -- see test_ring_math.py /
    test_ring_e2e.py in session notes) before this parameter was renamed
    from n_drop_max to make the caller's job harder to get wrong again.

    Stores the ring on each layer as `_qv_ring` (a list of (conv_state,
    recurrent_state) snapshots indexed by number of seq-tokens applied,
    0..n_seq_tok). Returns True if at least one linear-attention layer got
    a ring built (the normal case on a hybrid model whose capture was
    armed), False if none did (e.g. a non-hybrid model, or capture wasn't
    armed for this call) -- callers should treat False as "no ring
    available, use the existing snapshot/restore or full-replay path
    instead," not as an error. promote_ring (below) is itself defensive
    per-layer regardless of this return value.
    """
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    mod = _modern_layers(past)
    if mod is not None:
        # transformers >= 5 capture: states are derived lazily in promote_ring.
        ok = all(cap[2] == n_seq_tok for _, cap in mod)
        if not ok:
            discard_ring(past)
        return ok

    built_any = False
    for layer in getattr(past, "layers", []) or []:
        qkvgb = getattr(layer, "_qv_ring_qkvgb", None)
        if qkvgb is None:
            continue  # not a linear-attention layer, or nothing captured this step
        built_any = True
        query, key, value, g, beta = qkvgb
        init_state = getattr(layer, "_qv_ring_initial_state", None)
        conv_before = getattr(layer, "_qv_ring_conv_before", None)
        conv_ks = getattr(layer, "_qv_ring_conv_kernel_size", None)
        n_tok = query.shape[1]
        # n_seq_tok must exactly match what was captured -- both are
        # derived from the SAME `seq` tensor the caller fed to the model
        # (spec_decode.py: n_seq_tok=len(seq), n_tok=query.shape[1] from
        # that same forward), so any mismatch means a caller bug (e.g. a
        # future refactor computing n_seq_tok from something other than
        # the actual forwarded sequence) rather than a normal runtime
        # condition. A silent min() here would instead truncate the ring
        # and promote_ring would splice the wrong slot without any error --
        # exactly the class of bug the len(draft) vs len(seq) off-by-one
        # (see build_ring's docstring) was, so fail loudly instead of
        # repeating that mistake in a new form.
        assert n_seq_tok == n_tok, (
            f"ring_snapshot.build_ring: n_seq_tok={n_seq_tok} != captured "
            f"query length {n_tok} -- caller passed a length that doesn't "
            f"match the forward that was actually captured"
        )
        n_max = n_tok

        # conv_before is (batch, channels, prev_conv_len + n_tok) along dim=-1
        # (see the capture site's comment). Prefix i draft tokens applied
        # means the conv window as of position i is the last conv_ks
        # elements of conv_before[..., : prev_len + i], where prev_len is
        # whatever came before this step's new tokens.
        prev_len = conv_before.shape[-1] - n_tok if conv_before is not None else 0

        # ring[0] (the pre-step state) is never actually promoted --
        # promote_ring always looks up ring[n_acc + 1] with n_acc >= 0, so
        # index 0 is unreachable in the hot path. Left as None/state rather
        # than cloning conv_states[0] into it (a real per-step GPU clone
        # that would never be read) -- kept as an explicit slot anyway so
        # the ring's indexing (i == seq-tokens-applied) stays simple and
        # doesn't need an off-by-one relative to the loop below.
        ring = [None] * (n_max + 1)
        state = init_state
        ring[0] = (None, state)
        for i in range(n_max):
            _, state = fused_recurrent_gated_delta_rule(
                query[:, i:i + 1], key[:, i:i + 1], value[:, i:i + 1],
                g=g[:, i:i + 1], beta=beta[:, i:i + 1],
                initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if conv_before is not None:
                window_end = prev_len + i + 1
                window = conv_before[..., max(0, window_end - conv_ks):window_end]
                if window.shape[-1] < conv_ks:
                    window = torch.nn.functional.pad(window, (conv_ks - window.shape[-1], 0))
                conv_i = window.contiguous()
            else:
                conv_i = None
            ring[i + 1] = (conv_i, state)
        layer._qv_ring = ring
        del layer._qv_ring_qkvgb, layer._qv_ring_initial_state
        if conv_before is not None:
            del layer._qv_ring_conv_before, layer._qv_ring_conv_kernel_size
    return built_any


def promote_ring(past, n_acc: int) -> bool:
    """Splice the ring slot for `n_acc` accepted drafts into the live cache
    in place of whatever the batched verify forward left there (which
    reflects ALL drafted tokens, not just the n_acc accepted ones). O(1) --
    no replay. Returns False (leaving `past` untouched) if any
    linear-attention layer has no ring for this step, so the caller can
    fall back safely.

    Takes `n_acc` (accepted DRAFT count), not a raw ring index: internally
    promotes ring[n_acc + 1], since the ring is indexed by seq-tokens-
    applied (build_ring's `seq = [out_ids[-1]] + draft`) and seq[0] is
    always applied regardless of n_acc. Callers should never need to know
    about that +1 -- see build_ring's docstring for the bug this caused
    when the offset was the caller's responsibility.
    """
    idx = n_acc + 1
    layers = getattr(past, "layers", []) or []
    mod = _modern_layers(past)
    if mod is not None:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        for layer, ((q, k, v, g, beta, init), win, n_tok) in mod:
            if idx > n_tok:
                return False
        for layer, ((q, k, v, g, beta, init), win, n_tok) in mod:
            K = layer.conv_states[0].shape[-1]
            layer.conv_states[0].copy_(win[:, :, idx:idx + K])
            _, st = fused_recurrent_gated_delta_rule(
                q[:, :idx], k[:, :idx], v[:, :idx], g=g[:, :idx], beta=beta[:, :idx],
                initial_state=init, output_final_state=True,
                use_qk_l2norm_in_kernel=True)
            layer.recurrent_states[0].copy_(st)
            layer._qv_ring_modern = None
        return True
    for layer in layers:
        ring = getattr(layer, "_qv_ring", None)
        if ring is None:
            continue
        if idx >= len(ring):
            return False
    for layer in layers:
        ring = getattr(layer, "_qv_ring", None)
        if ring is None:
            continue
        conv_i, state_i = ring[idx]
        layer.recurrent_states[0] = state_i
        if conv_i is not None:
            layer.conv_states[0] = conv_i
        del layer._qv_ring
    return True


def crop_full_attention_layers(past, n_drop: int) -> None:
    """Crop only the full-attention (real KV) layers back by n_drop
    positions, skipping every linear-attention layer entirely.

    Cache.crop() (the composite/whole-cache version) calls .crop() on EVERY
    layer unconditionally, including linear-attention ones -- whose crop()
    (transformers.cache_utils.LinearAttentionCacheLayerMixin.crop, see
    cache_utils.py ~line 912) raises RuntimeError unless
    activate_past_recording() was called first (`record_past`). We don't
    want that: ring-promote (promote_ring, above) already put the correct
    conv_states/recurrent_states in place directly, so there is nothing
    left for crop() to do on those layers, and arming record_past would
    only add the OOM-risking full-history retention this project
    deliberately avoids (see project memory / _max_safe_drop's docstring --
    it wouldn't even help, since this model's own conv1d forward truncates
    to conv_kernel_size regardless of record_past). So this crops ONLY
    layers that are not a LinearAttentionCacheLayerMixin instance --
    DynamicLayer/Int8QuantizedLayer/etc, whose crop() has no such gate.

    NOTE this assumes every layer is EITHER pure linear-attention (conv/SSM
    state only, no keys/values) OR pure full-attention (keys/values only,
    no conv/SSM state) -- true for every model this project currently
    targets (Qwen3.5/3.6/3.8's layer_types are always a plain
    linear_attention/full_attention split, verified against both the 9B
    and 27B packs in use). transformers also defines
    LinearAttentionAndFullAttentionLayer (layer_type "hybrid": SAME layer
    carries both a conv/SSM state AND real keys/values, dual-inherits
    LinearAttentionLayer + DynamicLayer) for models that interleave the two
    within one layer -- that class IS a LinearAttentionCacheLayerMixin, so
    the isinstance skip below would silently leave its keys/values
    uncropped if this were ever pointed at such a model. Asserting instead
    of silently mishandling it.
    """
    from transformers.cache_utils import DynamicLayer, LinearAttentionCacheLayerMixin

    for layer in getattr(past, "layers", []) or []:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            assert not isinstance(layer, DynamicLayer), (
                f"crop_full_attention_layers: {type(layer).__name__} carries "
                "both linear-attention state and real KV (e.g. a 'hybrid' "
                "layer_type) -- the ring-promote fast path does not support "
                "this layer type yet, it would leave its KV uncropped"
            )
            continue
        layer.crop(-n_drop)


def discard_ring(past) -> None:
    """Drop any ring state left on `past` without promoting it (e.g. this
    step's fast path wasn't taken for some other reason). Safe no-op if
    nothing was captured."""
    for layer in getattr(past, "layers", []) or []:
        if getattr(layer, "_qv_ring_modern", None) is not None:
            layer._qv_ring_modern = None
        for attr in ("_qv_ring", "_qv_ring_qkvgb", "_qv_ring_initial_state",
                     "_qv_ring_conv_before", "_qv_ring_conv_kernel_size"):
            if hasattr(layer, attr):
                delattr(layer, attr)
