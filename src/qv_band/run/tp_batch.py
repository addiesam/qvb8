"""Batched decoding for a TP service: B conversations share every decode step (several users at once).

The single-conversation path (tp_graph.StaticDecoder + DFlash) is the fastest for ONE user; with several users
it runs them one after another. Here B sequence SLOTS are decoded together: one CUDA graph runs one token for every
slot, so each weight is read once for B tokens (decode is bound by reading the weights: B users cost little more
than one). Each slot has its own KV (static, [B, nkv, L, hd]), DeltaNet conv window and recurrent state, and
position. A new request is prefilled into a free slot (chunked HF forward, as prefill_hf) while the others wait,
then joins the shared step. Free slots run a dummy token (their state is overwritten at the next prefill).

Every card runs the same calls in the same order (card 0 broadcasts each prefill and each step's tokens).
v1: greedy, no drafter in batch mode, context per slot = L (no host offload).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from qv_band.engine.static_decode import StaticQKV

from qv_band.run.tp_gdn_fused import gdn_fused

PACK_TAU = float(os.environ.get("QV_STATE_TAU", "0") or 0) or None


class BatchDecoder:
    def __init__(self, model, tcfg, device, B, max_len):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        self.M, self.model, self.cfg, self.dev, self.B, self.L = M, model, tcfg, device, B, max_len
        dt = torch.float16
        self.layers = list(model.model.layers)
        if any(getattr(l, "_mhc_settings", None) for l in self.layers):
            raise ValueError("mHC layers (SenseNova): the batched TP decoder does not run them (USERS=1 / tp_graph only)")
        self.state = []
        for layer in self.layers:
            g = getattr(layer, "linear_attn", None)
            if g is not None:
                conv = torch.zeros(B, g.conv_dim, g.conv_kernel_size, dtype=dt, device=device)
                rec = torch.zeros(B, g.num_v_heads, g.head_k_dim, g.head_v_dim, dtype=torch.float32, device=device)
                self.state.append(("gdn", conv, rec))
            else:
                a = layer.self_attn
                nkv = a.k_proj.out_features // a.head_dim
                # qvb8 KV for every slot (engine/static_decode.StaticQKV)
                self.state.append(("attn", StaticQKV(B, nkv, max_len, a.head_dim, dt, device, extra=0)))
        H = tcfg.hidden_size
        self.h_in = torch.zeros(B, 1, H, dtype=dt, device=device)
        self.pos = torch.zeros(B, dtype=torch.long, device=device)
        self.tok_out = torch.zeros(B, dtype=torch.long, device=device)
        self.logits_out = None                    # [B, vocab] of the last step (sampling; allocated by the warm-up)
        self.first_logits = {}                    # slot -> the prompt's last logits (sampling the first token)
        self.bidx = torch.arange(B, device=device)
        self.graph = None
        if os.environ.get("QV_ROW_DETERMINISTIC", "1") == "1":
            from qv_band.run.tp_rownorm import install
            install(model)

    # ------------------------------------------------------------------ one token for every slot
    def _gdn(self, m, x, conv, rec):
        B = self.B
        mixed = m.in_proj_qkv(x).transpose(1, 2)                           # [B, C, 1]
        z = m.in_proj_z(x).reshape(B, 1, -1, m.head_v_dim)
        b, a = m.in_proj_b(x), m.in_proj_a(x)
        w = torch.cat([conv, mixed.to(conv.dtype)], dim=-1)                # [B, C, n + 1]
        out = F.conv1d(w, m.conv1d.weight, m.conv1d.bias, groups=w.shape[1])[:, :, -1:]
        conv.copy_(w[:, :, 1:])
        mixed = F.silu(out).to(x.dtype).transpose(1, 2)                    # [B, 1, C]
        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(B, -1, m.head_k_dim)
        k = k.reshape(B, -1, m.head_k_dim)
        v = v.reshape(B, -1, m.head_v_dim)
        beta = b.sigmoid().reshape(B, -1)
        g = (-m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)).reshape(B, -1)
        r = m.num_v_heads // m.num_k_heads
        if r > 1:
            q, k = q.repeat_interleave(r, dim=1), k.repeat_interleave(r, dim=1)
        Hv = v.shape[1]
        S = rec.view(B * Hv, m.head_k_dim, m.head_v_dim)
        o = gdn_fused(q.float().reshape(1, B * Hv, -1).contiguous(), k.float().reshape(1, B * Hv, -1).contiguous(),
                      v.float().reshape(1, B * Hv, -1).contiguous(), g.reshape(1, -1).contiguous(),
                      beta.float().reshape(1, -1).contiguous(), S, S.view(1, *S.shape), PACK_TAU)
        o = m.norm(o.to(x.dtype).reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(B, 1, -1)
        return m.out_proj(o)

    def _attn(self, m, x, kv, cos, sin):
        B, hd = self.B, m.head_dim
        q, gate = torch.chunk(m.q_proj(x).view(B, 1, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(B, 1, -1)
        q = m.q_norm(q).transpose(1, 2)
        k = m.k_norm(m.k_proj(x).view(B, 1, -1, hd)).transpose(1, 2)
        v = m.v_proj(x).view(B, 1, -1, hd).transpose(1, 2)
        q, k = self.M.apply_rotary_pos_emb(q, k, cos, sin)
        kv.write(k, v, self.pos)
        o, _ = kv.attn(q, (self.pos - kv.lc).clamp(min=0).view(B, 1) + 1, m.scaling)
        kv.set_len(self.pos + 1)
        return m.o_proj(o.to(x.dtype).transpose(1, 2).reshape(B, 1, -1) * torch.sigmoid(gate))

    def _step(self):
        h = self.h_in
        cos, sin = self.model.model.rotary_emb(h, self.pos.view(1, self.B, 1).expand(3, self.B, 1))
        for layer, st in zip(self.layers, self.state):
            x = layer.input_layernorm(h)
            if st[0] == "gdn":
                o = self._gdn(layer.linear_attn, x, st[1], st[2])
            else:
                o = self._attn(layer.self_attn, x, st[1], cos, sin)
            h = h + o
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = self.model.lm_head(self.model.model.norm(h))
        self.tok_out.copy_(logits[:, -1].argmax(-1))
        if self.logits_out is None:
            self.logits_out = torch.empty(self.B, logits.shape[-1], dtype=torch.float32, device=self.dev)
        self.logits_out.copy_(logits[:, -1])
        self.pos.add_(1).clamp_(max=self.L - 1)                             # free slots must stay in range

    @torch.no_grad()
    def capture(self, warm=2):
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(warm):
                self._step()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            self._step()
        self.graph = g
        torch.cuda.synchronize(self.dev)
        self.pos.zero_()

    @torch.no_grad()
    def step(self, tokens):
        """tokens: one per slot (any id for a free slot). Returns the next token of every slot."""
        e = self.model.model.embed_tokens(torch.tensor([tokens], device=self.dev).view(self.B, 1))
        self.h_in.copy_(e.to(self.dev, non_blocking=True))
        if self.graph is not None:
            self.graph.replay()
        else:
            self._step()
        return self.tok_out.tolist()

    # ------------------------------------------------------------------ a new conversation into slot b
    @torch.no_grad()
    def prefill_slot(self, b, ids, chunk=int(os.environ.get("QV_TP_PREFILL_CHUNK", "512"))):
        from transformers import DynamicCache
        n = len(ids)
        assert n < self.L - 1, f"prompt of {n} tokens: slot context is {self.L}"
        cache = DynamicCache(config=self.model.config)
        t = torch.tensor([ids], device=self.dev)
        last = None
        for p in range(0, n, chunk):
            out = self.model.model(input_ids=t[:, p:p + chunk], past_key_values=cache, use_cache=True)
            last = out.last_hidden_state[:, -1:]
        lg = self.model.lm_head(last)[0, -1]
        self.first_logits[b] = lg.float()
        tok = int(lg.argmax())
        for li, st in enumerate(self.state):
            L = cache.layers[li]
            if st[0] == "gdn":
                st[1][b].copy_(L.conv_states[0][0])
                st[2][b].copy_(L.recurrent_states[0][0].float())
            else:
                st[1].load(b, L.keys[:, :, :n], L.values[:, :, :n])
        del cache
        self.pos[b] = n
        return tok


def serve_rank0(bd, jobs, eos, tp, device, state, log=print):
    from qv_band.engine.spec import sample_token
    """Card 0's scheduler: admit queued jobs into free slots (prefill), then one shared step for all active slots.
    job: {"ids", "max_new", "out": queue} -- tokens are put on job["out"] as lists, None at the end."""
    import queue as _q
    B = bd.B
    slots = [None] * B
    hdr = torch.zeros(3, dtype=torch.long, device=device)

    def bcast(cmd, a=0, b=0, payload=None):
        if tp.world > 1:
            hdr.copy_(torch.tensor([cmd, a, b]))
            torch.distributed.broadcast(hdr, 0, group=tp.group)
            if payload is not None:
                torch.distributed.broadcast(payload, 0, group=tp.group)

    def finish(i):
        slots[i]["job"]["out"].put(None)
        state["served"] += 1
        slots[i] = None

    while True:
        active = [i for i in range(B) if slots[i] is not None]
        for i in range(B):
            if slots[i] is not None:
                continue
            try:
                job = jobs.get(block=not active and not any(slots))
            except _q.Empty:
                break
            state["queue"] -= 1
            ids = job["ids"]
            bcast(2, i, len(ids), torch.tensor(ids, dtype=torch.long, device=device))
            tok = bd.prefill_slot(i, ids)
            if job.get("temperature", 0) > 0:                 # sampled: card 0 picks; it broadcasts the tokens
                tok = sample_token(bd.first_logits[i], job["temperature"], job.get("top_k", 0), job.get("top_p", 1.0))
            job["out"].put([tok])
            slots[i] = {"job": job, "n": 1, "last": tok}
            if tok in eos or job["max_new"] <= 1:
                finish(i)
            active = [j for j in range(B) if slots[j] is not None]
        active = [i for i in range(B) if slots[i] is not None]
        if not active:
            continue
        toks = [slots[i]["last"] if slots[i] is not None else 0 for i in range(B)]
        bcast(3, 0, 0, torch.tensor(toks, dtype=torch.long, device=device))
        nxt = bd.step(toks)
        state["tokens"] += len(active)
        for i in active:
            s = slots[i]
            j = s["job"]
            t = (sample_token(bd.logits_out[i], j["temperature"], j.get("top_k", 0), j.get("top_p", 1.0))
                 if j.get("temperature", 0) > 0 else nxt[i])
            s["job"]["out"].put([t])
            s["n"] += 1
            s["last"] = t
            if t in eos or s["n"] >= s["job"]["max_new"]:
                finish(i)
        state["busy"] = any(slots)


def serve_follower(bd, tp, device):
    hdr = torch.zeros(3, dtype=torch.long, device=device)
    while True:
        torch.distributed.broadcast(hdr, 0, group=tp.group)
        cmd, a, b = hdr.tolist()
        if cmd == 0:
            return
        if cmd == 2:
            ids = torch.empty(b, dtype=torch.long, device=device)
            torch.distributed.broadcast(ids, 0, group=tp.group)
            bd.prefill_slot(a, ids.tolist())
        elif cmd == 3:
            toks = torch.empty(bd.B, dtype=torch.long, device=device)
            torch.distributed.broadcast(toks, 0, group=tp.group)
            bd.step(toks.tolist())
