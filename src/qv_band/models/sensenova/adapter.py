"""SenseNova-Loop adapter (model_type sensenova_loop): a Qwen3.5 hybrid (Gated-DeltaNet + gated attention, 32 layers)
whose layers [start, end) run `eval_count` times with shared weights, with mHC (multi-stream hyper-connections) in
that range. Reference: the checkpoint's modeling_sensenova_loop.py.

How it runs on the engine: the skeleton is the plain Qwen3.5 text model the engine already serves (32 physical layers,
qvb8 weights). install() then
  1. loads the fp32 parameters the model keeps in fp32 (mHC routers, A_log, dt_bias) from the checkpoint/pack's dense
     tensors -- the skeleton cast A_log / dt_bias to the model dtype, and has no routers at all;
  2. turns layers [start, end) into MHCDecoderLayer (the reference forward: enter streams, router -> branch,
     attention / DeltaNet, merge, router -> branch, MLP, merge, leave);
  3. UNROLLS the execution plan into the module list: 0..start-1, (start..end-1) x eval_count, end..N-1 -- 56 entries
     for the 9B. A repeated visit is a shallow clone of its layer: every weight, norm, router and the MLP are the SAME
     objects (no extra memory), only the attention / DeltaNet module is a fresh shell carrying its own layer_idx = the
     execution slot. config.num_hidden_layers / layer_types follow the plan.
To the rest of the engine this is a 56-layer Qwen3.5: BandedKVCache (qvb8 KV), the continuous-batching scheduler,
prefix reuse / the shared pool, speculative rollback and the MTP drafter all index caches by layer_idx, and each visit
gets its own cache slot -- exactly the reference's per-visit caches (its _VisitCache).
Hidden states between mHC layers are the flattened streams [.., streams * hidden]; the last mHC layer averages them,
so everything outside the range (and the final norm, lm_head, drafters) sees the plain hidden size.
"""
from __future__ import annotations

import copy
import glob
import os

import torch
from torch import nn

from qv_band.engine.adapter import ModelAdapter, register


# ----------------------------------------------------------------------------------------- the router
class MHCRouter(nn.Module):
    """mHC read / write / mix weights of one sub-block; fp32 parameters, fp32 maths (reference: MHCRouter). The sigmoids
    and the Sinkhorn run as the framework's layers/fused_sinkhorn kind (one launch, bit-identical)."""

    def __init__(self, weight, bias, settings):
        super().__init__()
        self.iterations = int(settings["iterations"])
        self.pre_scale = float(settings["pre_scale"])
        self.weight = nn.Parameter(weight.float(), requires_grad=False)
        self.bias = nn.Parameter(bias.float(), requires_grad=False)

    def _apply(self, fn, recurse=True):
        # model.to(dtype) after install must not cast the router: keep fp32, follow the device
        def keep_fp32(t):
            r = fn(t)
            return r.float() if t.is_floating_point() and r.dtype != torch.float32 else r
        return super()._apply(keep_fp32, recurse=recurse)

    def forward(self, residual):
        from qv_band.layers.fused_sinkhorn import mhc_route
        with torch.autocast(device_type=residual.device.type, enabled=False):
            flat = residual.flatten(-2).float()
            normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
            pre, post, mix = mhc_route(torch.nn.functional.linear(normalized, self.weight, self.bias), self.iterations)
            branch = torch.matmul(pre.to(residual.dtype).unsqueeze(-2), residual).squeeze(-2)
            if self.pre_scale != 1.0:
                branch = branch * self.pre_scale
        return branch, post, mix

    @staticmethod
    def merge(residual, branch, post, mix):
        """Mix the residual streams and add the branch output to each stream."""
        with torch.autocast(device_type=residual.device.type, enabled=False):
            mixed = torch.matmul(mix.to(residual.dtype), residual)
            return mixed + post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)


def _qwen35_layer_class():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    return Qwen3_5DecoderLayer


def _make_mhc_class():
    base = _qwen35_layer_class()

    class MHCDecoderLayer(base):
        """Reference MHCDecoderLayer.forward; _mhc_physical / _mhc_settings / mhc_attn / mhc_ffn set by install."""

        def forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                    past_key_values=None, **kwargs):
            st = self._mhc_settings
            if self._mhc_physical == st["start"]:            # enter: copy the hidden state into every stream
                residual = hidden_states.unsqueeze(-2).expand(*hidden_states.shape[:-1], st["streams"],
                                                              hidden_states.shape[-1]).contiguous()
            else:
                residual = hidden_states.unflatten(-1, (st["streams"], -1))
            branch, post, mix = self.mhc_attn(residual)
            branch = self.input_layernorm(branch)
            if self.block_type == "linear_attention":
                branch = self.linear_attn(hidden_states=branch, cache_params=past_key_values,
                                          attention_mask=attention_mask, **kwargs)
            else:
                branch, _ = self.self_attn(hidden_states=branch, position_embeddings=position_embeddings,
                                           attention_mask=attention_mask, position_ids=position_ids,
                                           past_key_values=past_key_values, **kwargs)
            residual = MHCRouter.merge(residual, branch, post, mix)
            branch, post, mix = self.mhc_ffn(residual)
            branch = self.mlp(self.post_attention_layernorm(branch))
            out = MHCRouter.merge(residual, branch, post, mix)
            # leave: the last mHC layer averages the streams; inside the range they travel flattened
            return out.mean(-2) if self._mhc_physical == st["end"] - 1 else out.flatten(-2)

    return MHCDecoderLayer


# ----------------------------------------------------------------------------------------- plan
def execution_plan(loop: dict, n_layers: int) -> list[int]:
    """Physical layer per execution slot (reference: execution_plan; fixed-count checkpoints only)."""
    start, end, count = int(loop["start"]), int(loop["end"]), int(loop["eval_count"])
    if not (0 <= start < end <= n_layers and count >= 1) or list(loop.get("counts", [count])) != [count]:
        raise ValueError(f"sensenova: unsupported loop_settings {loop}")
    return list(range(start)) + list(range(start, end)) * count + list(range(end, n_layers))


def _shell(mod: nn.Module) -> nn.Module:
    """A new module object sharing every parameter, buffer and child of `mod` (its own attribute dict)."""
    new = copy.copy(mod)
    new.__dict__ = dict(mod.__dict__)
    for k in ("_parameters", "_buffers", "_modules", "_forward_hooks", "_forward_pre_hooks"):
        if k in mod.__dict__:
            new.__dict__[k] = type(mod.__dict__[k])(mod.__dict__[k])
    return new


def _visit(layer: nn.Module, slot: int, clone: bool) -> nn.Module:
    """The layer as execution slot `slot`: a clone (shared weights) for a repeated visit; its token mixer's layer_idx
    is the slot, so each visit has its own cache layer."""
    out = _shell(layer) if clone else layer
    for name in ("self_attn", "linear_attn"):
        mixer = out._modules.get(name)
        if mixer is not None:
            if clone:
                mixer = _shell(mixer)
                out._modules[name] = mixer
            mixer.layer_idx = slot
    out._qv_slot = slot
    return out


def _fp32_tensors(model_dir: str, wanted) -> dict:
    """The fp32 source tensors named in `wanted` (suffix match on checkpoint names), from the dir's safetensors."""
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if any(k.endswith(w) for w in wanted(k)):
                    out[k] = h.get_tensor(k)
    return out


# ----------------------------------------------------------------------------------------- adapter
class SenseNovaLoopAdapter(ModelAdapter):
    name = "sensenova"

    def matches(self, config) -> bool:
        return getattr(config, "model_type", None) == "sensenova_loop"

    def execution_layer_types(self, raw_config):
        tc = raw_config.get("text_config") or raw_config
        types = tc["layer_types"]
        return [types[p] for p in execution_plan(raw_config["loop_settings"], len(types))]

    def install(self, model, *, model_dir, config, tok) -> None:
        loop = getattr(config, "loop_settings", None)
        mhc = getattr(config, "mhc", None)
        if not isinstance(loop, dict):
            raise RuntimeError("sensenova: config.json has no loop_settings")
        text = model.model                                   # Qwen3_5TextModel
        cfg = model.config
        layers = list(text.layers)
        n = len(layers)
        if any("forward" in L.__dict__ for L in layers):
            raise RuntimeError("sensenova: a decoder layer's forward is already wrapped: install() must run before the "
                               "pipeline hooks (run_inference does)")
        plan = execution_plan(loop, n)
        dev = next(p.device for p in text.parameters() if p.device.type != "meta")
        # 1) fp32 parameters: A_log / dt_bias of every DeltaNet layer, the mHC routers
        pre = "model.language_model.layers."
        src = _fp32_tensors(model_dir, lambda k: (".linear_attn.A_log", ".linear_attn.dt_bias",
                                                    ".mhc_attn.weight", ".mhc_attn.bias",
                                                    ".mhc_ffn.weight", ".mhc_ffn.bias") if k.startswith(pre) else ())
        n_fp32 = 0
        for i, L in enumerate(layers):
            la = getattr(L, "linear_attn", None)
            if la is not None and not getattr(model, "_qv_tp", False):      # TP: the card already holds its own share of A_log / dt_bias
                for p in ("A_log", "dt_bias"):
                    t = src.get(f"{pre}{i}.linear_attn.{p}")
                    if t is None:
                        raise RuntimeError(f"sensenova: {pre}{i}.linear_attn.{p} not found in {model_dir}")
                    ldev = next((q.device for q in L.parameters() if q.device.type != "meta"), dev)     # the layer's own card (pipeline)
                    la._parameters[p] = nn.Parameter(t.float().to(ldev), requires_grad=False)
                    n_fp32 += 1
        # 2) mHC layers
        n_mhc = 0
        if mhc is not None:
            st = dict(mhc)
            bounds = (0, int(loop["start"]), int(loop["end"]), n)          # the reference's phase rule
            if not any(lo <= int(st["start"]) < int(st["end"]) <= hi for lo, hi in zip(bounds, bounds[1:])):
                raise RuntimeError("sensenova: an mHC range that crosses a loop boundary is not supported")
            from qv_band.layers.fused_sinkhorn import STREAMS
            if int(st["streams"]) != STREAMS or st.get("fixed_identity"):
                raise RuntimeError(f"sensenova: mHC {st}: only {STREAMS} streams with a learned mix are supported "
                                   f"(layers/fused_sinkhorn)")
            cls = _make_mhc_class()
            for i in range(int(st["start"]), int(st["end"])):
                L = layers[i]
                L.__class__ = cls
                L._mhc_physical, L._mhc_settings = i, st
                for r in ("mhc_attn", "mhc_ffn"):
                    w, b = src.get(f"{pre}{i}.{r}.weight"), src.get(f"{pre}{i}.{r}.bias")
                    if w is None or b is None:
                        raise RuntimeError(f"sensenova: {pre}{i}.{r} not found in {model_dir}")
                    ldev = next((q.device for q in L.parameters() if q.device.type != "meta"), dev)
                    L._modules[r] = MHCRouter(w.to(ldev), b.to(ldev), st)
                    n_fp32 += 2
                n_mhc += 1
        # 3) unroll the plan: one module per execution slot (repeats share every weight)
        seen = set()
        unrolled = []
        for slot, phys in enumerate(plan):
            unrolled.append(_visit(layers[phys], slot, clone=phys in seen))
            seen.add(phys)
        types = list(cfg.layer_types)
        text.layers = nn.ModuleList(unrolled)
        cfg.layer_types = [types[p] for p in plan]
        cfg.num_hidden_layers = len(plan)
        cfg._qv_physical_layers = n
        cfg._qv_execution_plan = plan
        model._qv_execution_plan = plan
        n_full = sum(t == "full_attention" for t in cfg.layer_types)
        print(f"sensenova runtime: {n} physical layers -> {len(plan)} execution slots "
              f"(layers {loop['start']}-{int(loop['end']) - 1} x{loop['eval_count']}, shared weights; "
              f"{n_full} full-attention + {len(plan) - n_full} DeltaNet cache slots); "
              f"mHC {n_mhc} layers ({mhc['streams'] if mhc else 0} streams, Sinkhorn {mhc['iterations'] if mhc else 0}); "
              f"{n_fp32} fp32 tensors from the checkpoint", flush=True)


ADAPTER = register(SenseNovaLoopAdapter())
