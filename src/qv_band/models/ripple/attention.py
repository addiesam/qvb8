"""Ripple attention forward, routed through the engine the same way Qwen's is.

The shipped RicursiveAttention.forward calls F.scaled_dot_product_attention directly, which bypasses the
engine (no qvb8 fused attention, no flash-attn) and has two masking bugs with a KV cache:
  1. global layers: is_causal=True with q_len > 1 on top of a cache = top-left causal alignment, so a
     speculative verify / chunked prefill sees almost no context;
  2. sliding layers: keys are numbered 0..kv_len-1 while queries use absolute positions, which is wrong once
     the window cache has dropped old keys (context > 1024 tokens) -- plain decoding included.
This forward keeps every Ripple-specific op (value residual, per-head q/k norm, RoPE on sliding layers only,
NoPE global layers, sigmoid output gate) and hands the attention itself to transformers' attention registry
(ALL_ATTENTION_FUNCTIONS[config._attn_implementation]) exactly like Qwen's modeling code. That gives:
  * flash-attn with bottom-right causal alignment and window_size=(W-1, W-1) -> both bugs gone by construction;
  * GQA without repeat_kv copies;
  * the qvb8 fused attention (kv/fused_attn.install wraps that registry entry) on qvb8 (BandedKVLayer) layers.
"""
from __future__ import annotations

import sys

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from qv_band.models.ripple import fused as _fused


def ripple_attention_forward(self, hidden_states, attention_mask=None, position_ids=None,
                             past_key_value=None, output_attentions=False, use_cache=False,
                             cache_position=None, value_residual=None, **kwargs):
    bsz, q_len, _ = hidden_states.shape
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    value_states = hidden_states
    if value_residual is not None:
        blend = torch.sigmoid(self.value_res_lambda[0]).to(dtype=hidden_states.dtype)
        value_states = torch.lerp(value_residual.to(dtype=hidden_states.dtype), hidden_states, blend)
    v = self.v_proj(value_states)

    q = self.q_norm(q.view(bsz, q_len, self.num_heads, self.head_dim))
    k = self.k_norm(k.view(bsz, q_len, self.num_kv_heads, self.head_dim))
    v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    if self._qv_sliding and _fused.enabled() and q.is_cuda:
        # RoPE on local layers only; global layers are NoPE. Fused fp32 RoPE (models/ripple/fused.py) on the
        # (B, T, H, D) layout, before the transpose: one pass instead of ~10 fp32 copies / muls / adds.
        cos, sin = self.rotary_emb(v, position_ids)
        q = _fused.rope_bthd(q, cos, sin).transpose(1, 2)
        k = _fused.rope_bthd(k, cos, sin).transpose(1, 2)
    else:
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        if self._qv_sliding:                               # RoPE on local layers only; global layers are NoPE
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = self._qv_apply_rope(q, k, cos, sin)

    if past_key_value is not None:
        k, v = past_key_value.update(k, v, self.layer_idx)

    attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]   # looked up per call: fused wrapper
    out, _ = attn_fn(self, q, k, v, None, dropout=0.0, scaling=self.head_dim ** -0.5,
                     sliding_window=self._qv_window, is_causal=True)
    out = out.reshape(bsz, q_len, -1)                      # (B, T, H, D) -> (B, T, H*D)

    if self.config.use_gated_attention:
        gate = self.g_proj(hidden_states)
        if _fused.enabled() and out.is_cuda and gate.dtype == out.dtype and gate.shape == out.shape:
            out = _fused.gate_sigmoid(out, gate)           # sigmoid in fp32 either way (gate_full_precision)
        else:
            if self.config.gate_full_precision:
                gate = gate.float()
            out = out * torch.sigmoid(gate).to(out.dtype)
    return self.o_proj(out), None, past_key_value


def install_attention(model, layer_types: list[str], window: int) -> int:
    """Bind ripple_attention_forward to every attention module. layer_types: the ORIGINAL per-layer types."""
    n = 0
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        sliding = layer_types[i] == "sliding_attention"
        attn._qv_sliding = sliding
        attn._qv_window = window if sliding else None
        attn.is_causal = True
        attn._qv_apply_rope = sys.modules[type(attn).__module__].apply_rotary_pos_emb
        attn.forward = ripple_attention_forward.__get__(attn)
        n += 1
    return n
