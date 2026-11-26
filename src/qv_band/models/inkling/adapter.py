"""Inkling-2 (model_type inkling2_loop) adapter: the model runs its own forward (model.py), no HF skeleton.

  forward    models/inkling/model.py (24 layers, layers 6-17 looped 3x, block / message residuals)
  attention  models/inkling/attention.py (relative bias + sinks + sliding window; O(L) memory)
  weights    qvb8 linears (QVBandLinearV8), routed experts in the "qvbq" format (NVFP4 / MXFP8 bit-exact)
  serving    engine/target_engine.TargetBatchEngine (continuous batching, prompt-prefix reuse) via build_target
  chat       models/inkling/chat_format.py (channel tokens -> reasoning / answer / tool calls)

Matches on config.json's model_type, so the engine's own transformers need not know the architecture.

Cards (env, set by the launchers):
  QV_TP_DEVICES      e.g. "cuda:0,cuda:1": Tensor Parallel over those cards (engine/tp_target.TPTarget; attention
                     by heads, experts by ranges). Unset: the one --device.
  QV_EXPERTS         resident (default) | host: the expert stack in page-locked host RAM, QV_EXPERT_CACHE_GB of it
                     (default 8) cached per card -- a card too small for the experts (one 16 GB V100).
  --dtype            bf16 | fp16 (V100: fp16, no bf16 tensor cores); default by the card.
  --drafter dspark   (QV_DRAFTER / QV_DRAFT, set by the engine): the checkpoint's own DSpark drafter (dspark.py) for
                     greedy requests (engine/spec.py), up to --draft drafts per step (<= 7); every topology: one card,
                     Tensor Parallel over threads or processes (card 0 drafts, every card verifies).
"""
from __future__ import annotations

import json
import os

import torch

from qv_band.engine.adapter import ModelAdapter, ServeDefaults, register


class InklingTarget(torch.nn.Module):
    """BatchTarget over InklingModel. An nn.Module only so the engine's module walks (bf16 GEMM switch) see the
    qvb8 linears; it has no forward of its own beyond the BatchTarget one."""

    def __init__(self, model_dir, device="cuda", kv_cache="bf16", kv_tau=0.1, dtype=None, tp=None,
                 experts="resident", expert_cache_gb=8.0):
        super().__init__()
        from qv_band.formats.qvb8.module import QVBandLinearV8
        from qv_band.models.inkling.model import InklingModel
        with torch.cuda.device(torch.device(device)):
            self.m = InklingModel(model_dir, device=device, kv_cache=kv_cache, kv_tau=kv_tau, dtype=dtype, tp=tp,
                                  experts=experts, expert_cache_gb=expert_cache_gb)
        lins = []
        for L in self.m.layers:
            lins += [v for v in vars(L).values() if isinstance(v, QVBandLinearV8)]
        if isinstance(self.m.lm_head, QVBandLinearV8):
            lins.append(self.m.lm_head)
        self.linears = torch.nn.ModuleList(lins)
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        tc = cfg["text_config"]
        from types import SimpleNamespace
        self.config = SimpleNamespace(model_type=cfg.get("model_type"), max_position_embeddings=tc.get("model_max_length"),
                                      text_config=SimpleNamespace(max_position_embeddings=tc.get("model_max_length")))
        self.dtype = self.m.dt
        self.stop_ids = set()
        self._qv_batch_target = True

    def new_state(self):
        return self.m.new_state()

    def clone_state(self, state):
        return self.m.clone_state(state)

    def state_nbytes(self, state):
        return self.m.state_nbytes(state)

    def export_state(self, state):
        return self.m.export_state(state)

    def import_state(self, tree, device):
        return self.m.import_state(tree, device)

    def forward(self, rows):
        return self.m.forward(rows)

    # static decode (engine/static_decode.StaticStepper): the model's slot buffers
    def static_init(self, slots, max_len):
        self.m.static_init(slots, max_len)
        self.st_tok, self.st_active, self.st_pos, self.st_logits = self.m.st_tok, self.m.st_active, self.m.st_pos, self.m.st_logits
        m = self.m
        if m.spec_width:                          # speculative verify on the slots (engine/spec.py protocol)
            self.st_vtok, self.st_vslot, self.st_vj, self.st_vrow = m.st_vtok, m.st_vslot, m.st_vj, m.st_vrow
            self.st_vstart, self.st_acc, self.st_vlogits = m.st_vstart, m.st_acc, m.st_vlogits
            self.static_verify, self.static_commit = m.static_verify, m.static_commit
        if m.dspark is not None:                  # the drafting card: the proposal too
            d = m.dspark
            self.st_pslot, self.st_panchor, self.st_draft, self.st_dprob = d.st_pslot, d.st_panchor, d.st_draft, d.st_dprob
            self.st_ddist = d.st_ddist
            self.static_propose, self.static_max_draft = m.static_propose, m.spec_max_draft
        self.static_capacity = self.m.static_capacity
        # experts streamed from host RAM (QV_EXPERTS=host) are fetched with host syncs: not capturable
        self.static_graphable = not str(self.m.expert_mode).startswith("host")

    def static_free(self):
        self.m.static_free()
        for n in ("st_tok", "st_active", "st_pos", "st_logits", "st_vtok", "st_vslot", "st_vj", "st_vrow", "st_vstart",
                  "st_acc", "st_vlogits", "st_pslot", "st_panchor", "st_draft", "st_dprob", "st_ddist",
                  "static_verify", "static_commit", "static_propose", "static_max_draft"):
            self.__dict__.pop(n, None)

    def static_load(self, slot, state):
        self.m.static_load(slot, state)

    def static_bytes(self, slots, max_len):
        return self.m.static_bytes(slots, max_len)

    def enable_spec(self, width, drafter_pack=None, max_draft=7):
        """engine/spec.py protocol on this card: verify (+ the checkpoint's own drafter when drafter_pack is given)."""
        m = self.m
        if drafter_pack is not None:
            m.enable_dspark(drafter_pack, max_draft)
            self.spec_propose, self.spec_max_draft = m.spec_propose, m.spec_max_draft
        else:
            m.enable_spec(width)
        self.spec_verify, self.spec_keep = m.spec_verify, m.spec_keep

    def static_step(self):
        self.m.static_step()


class InklingAdapter(ModelAdapter):
    name = "inkling"
    vision_adapter = "qv_band.models.inkling.vision:InklingVision"   # image chats: own tower + own forward

    def matches(self, config) -> bool:
        return getattr(config, "model_type", None) == "inkling2_loop"

    def serve_defaults(self) -> ServeDefaults:
        return ServeDefaults(kv_cache="qvb8", prefill_chunk=2048, drafter="", draft_k=0)

    def build_target(self, model_dir, *, device="cuda", dtype=None, kv_cache="qvb8", kv_tau=0.1):
        dts = {None: None, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
               "float16": torch.float16}
        if dtype not in dts:
            raise ValueError(f"Inkling: --dtype bf16 or fp16 (got {dtype})")
        dt = dts[dtype]
        devs = [d.strip() for d in os.environ.get("QV_TP_DEVICES", "").split(",") if d.strip()] or [device]
        experts = os.environ.get("QV_EXPERTS", "resident")
        cache_gb = float(os.environ.get("QV_EXPERT_CACHE_GB", "8"))
        if dt == torch.bfloat16 and any(torch.cuda.get_device_capability(torch.device(d))[0] < 8 for d in devs):
            raise ValueError("Inkling: --dtype bf16 needs sm >= 80; use --dtype fp16 on this card")
        from qv_band.models.inkling.chat_format import STOP_TOKENS, InklingChatFormat, InklingTokenizer
        try:                                             # the tokenizer class itself: no model config is read
            from transformers import TokenizersBackend as _Tok
        except ImportError:
            from transformers import PreTrainedTokenizerFast as _Tok
        base = _Tok.from_pretrained(model_dir, local_files_only=True)
        tok = InklingTokenizer(base, effort=os.environ.get("QV_INKLING_EFFORT") or None)
        if kv_cache not in ("bf16", "qvb8"):
            raise ValueError(f"Inkling: --kv-cache bf16 or qvb8 (got {kv_cache})")
        proc = int(os.environ.get("WORLD_SIZE", "1")) > 1      # torchrun: Tensor Parallel over processes (NCCL)
        dspark = os.environ.get("QV_DRAFTER") == "dspark"
        k = int(os.environ.get("QV_DRAFT", "7"))
        if dspark:                                       # the verify width: the drafter's block (its checkpoint config)
            cfg = json.load(open(os.path.join(model_dir, "config.json")))
            mc = cfg.get("mtp_config") or cfg.get("text_config", {}).get("mtp_config")
            if not mc:
                raise ValueError("Inkling: --drafter dspark: this checkpoint has no mtp_config (no bundled drafter)")
            width = int(mc["block_size"])
        if proc:
            from qv_band.engine.proc_tp import ProcTPTarget, serve_follower, setup
            dev = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
            tp, ctrl = setup(dev)
            local = InklingTarget(model_dir, device=dev, kv_cache=kv_cache, kv_tau=kv_tau, dtype=dt, tp=tp,
                                  experts=experts, expert_cache_gb=cache_gb)
            if dspark:                                   # card 0 drafts (its own drafter); the others only verify
                local.enable_spec(width, model_dir if tp.rank == 0 else None, k)
            if tp.rank != 0:                                   # a follower card: rank 0's commands, forever
                serve_follower(local, ctrl)
                os._exit(0)
            t = ProcTPTarget(local, tp, ctrl)
            m0, linears, devs = local.m, local.linears, [dev] * tp.world
        elif len(devs) == 1:
            t = InklingTarget(model_dir, device=devs[0], kv_cache=kv_cache, kv_tau=kv_tau, dtype=dt,
                              experts=experts, expert_cache_gb=cache_gb)
            m0, linears = t.m, t.linears
            if dspark:                                   # --drafter dspark: the checkpoint's own drafter
                t.enable_spec(width, model_dir, k)
        else:                                            # Tensor Parallel: one share per card, run as one target
            from qv_band.engine.tp_target import TPTarget
            from qv_band.run.tp_dit import Hub, ThreadTP
            hub = Hub(devs)
            ranks = [InklingTarget(model_dir, device=d, kv_cache=kv_cache, kv_tau=kv_tau, dtype=dt,
                                   tp=ThreadTP(hub, r), experts=experts, expert_cache_gb=cache_gb)
                     for r, d in enumerate(devs)]
            if dspark:                                   # card 0 drafts; every card verifies
                for r, rt in enumerate(ranks):
                    rt.enable_spec(width, model_dir if r == 0 else None, k)
            t = TPTarget(ranks, devs, hub)
            m0 = ranks[0].m
            linears = [lin for r in ranks for lin in r.linears]
        t.stop_ids = {base.convert_tokens_to_ids(s) for s in STOP_TOKENS}
        t._qv_chat_format = InklingChatFormat()
        t._qv_adapter = self                             # generation/vision.adapter() reads its vision_adapter
        t._qv_eos_ids = set(t.stop_ids)
        dn = str(m0.dt).replace("torch.", "")
        t._qv_kv_desc = (f"qvb8 on the {sum(not L.sliding for L in m0.layers)} full-attention layers, tau {kv_tau:g}; "
                         f"sliding layers {dn} window" if kv_cache == "qvb8" else dn)
        if proc:
            cards = (f"TP over {t.world} processes (heads {m0.cfg['head_range']} of {m0.cfg['heads']} on card 0, "
                     f"expert parallel, NCCL all-reduce, static decode graphs)")
        else:
            cards = devs[0] if len(devs) == 1 else (f"TP over {', '.join(devs)} (heads {m0.cfg['head_range']} of "
                                                   f"{m0.cfg['heads']} on card 0, experts split, "
                                                   f"all-reduce {'through host memory' if t.hub.via_host else 'card to card'})")
        print(f">>> inkling target: {len(linears)} qvb8 linears, qvbq experts {m0.expert_mode}, {dn}, {cards}, "
              f"own forward, KV {kv_cache}{f' tau {kv_tau:g}' if kv_cache == 'qvb8' else ''} "
              f"(effort default {tok._effort or 'template 0.9'})", flush=True)
        return t, tok


ADAPTER = register(InklingAdapter())
