"""Several users at once WITH DFlash on a TP service: B sequence slots, each drafts its own block, ONE verify for all.

Every step: each drafting slot proposes T-1 tokens (its own DFlash context); all B slots then run T tokens through ONE
CUDA graph ([B, T] tokens: each weight read once for B*T rows); each slot keeps what its target agrees with
(n_accepted + 1 tokens) and commits only that: KV positions past it are masked and later overwritten, each DeltaNet
layer restores the state after its kept token from the per-token history, the conv window from the window history.
Output is exactly greedy decoding per slot (verified drafts only).

A slot that does not draft this step (no drafter, its acceptance paused, or just admitted) still runs in the shared
step with its last token followed by padding, and keeps 1 token. Free slots run padding (overwritten at the next
prefill). New requests are prefilled into a free slot with the chunked HF forward (as tp_graph.prefill_hf), which also
feeds that slot's drafter the prompt's target features.

Every card runs the same calls in the same order: card 0 broadcasts each admission, which slots draft, and the drafts
(the split drafter's collectives need every card to draft the same slots). Adaptive drafting per slot as tp_graph."""
from __future__ import annotations

import os

import time

import torch
import torch.nn.functional as F

from qv_band.engine.static_decode import StaticQKV

from qv_band.run.tp_gdn_fused import gdn_fused
from qv_band.engine.spec import SpecPlanner
from qv_band.run.tp_graph import tap_features

PACK_TAU = float(os.environ.get("QV_STATE_TAU", "0") or 0) or None


class SpecBatchDecoder:
    def __init__(self, model, tcfg, device, B, max_len, T, capture_layers=()):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        self.M, self.model, self.cfg, self.dev = M, model, tcfg, device
        self.B, self.L, self.T = B, max_len, T
        self.caps = list(capture_layers)
        dt = torch.float16
        self.layers = list(model.model.layers)
        if any(getattr(l, "_mhc_settings", None) for l in self.layers):
            raise ValueError("mHC layers (SenseNova): the batched TP decoder does not run them (USERS=1 / tp_graph only)")
        self.state = []
        for layer in self.layers:
            g = getattr(layer, "linear_attn", None)
            if g is not None:
                n = g.conv_kernel_size
                conv = torch.zeros(B, g.conv_dim, n, dtype=dt, device=device)
                hn = torch.zeros(B, g.conv_dim, n + T, dtype=dt, device=device)           # window history
                rec = torch.zeros(B, g.num_v_heads, g.head_k_dim, g.head_v_dim, dtype=torch.float32, device=device)
                hist = torch.zeros(T, B * g.num_v_heads, g.head_k_dim, g.head_v_dim, dtype=torch.float32,
                                   device=device)                                         # state after each token
                self.state.append(("gdn", conv, rec, hn, hist))
            else:
                a = layer.self_attn
                nkv = a.k_proj.out_features // a.head_dim
                # qvb8 KV for every slot (engine/static_decode.StaticQKV)
                self.state.append(("attn", StaticQKV(B, nkv, max_len, a.head_dim, dt, device, extra=T)))
        H = tcfg.hidden_size
        self.h_in = torch.zeros(B, T, H, dtype=dt, device=device)
        self.pos = torch.zeros(B, dtype=torch.long, device=device)
        self.keep = torch.ones(B, dtype=torch.long, device=device)
        self.tok_out = torch.zeros(B, T, dtype=torch.long, device=device)
        self.logits_out = None                    # [B, T, vocab] of the last run (sampling; allocated by the warm-up)
        self.first_logits = {}                    # slot -> the prompt's last logits (sampling the first token)
        self.feats = torch.zeros(B, T, max(1, len(self.caps)) * H, dtype=dt, device=device)
        self.bidx = torch.arange(B, device=device)
        self.tar = torch.arange(T, device=device)
        self.graphs, self.commit_graph = {}, None
        if os.environ.get("QV_ROW_DETERMINISTIC", "1") == "1":       # bit-identical rows at any batch / T
            from qv_band.run.tp_rownorm import install
            install(model)

    # ------------------------------------------------------------------ T tokens for every slot
    def _gdn(self, m, x, T, conv, rec, hn, hist):
        B = self.B
        mixed = m.in_proj_qkv(x).transpose(1, 2)                           # [B, C, T]
        z = m.in_proj_z(x).reshape(B, T, -1, m.head_v_dim)
        b, a = m.in_proj_b(x), m.in_proj_a(x)                              # [B, T, Hv]
        n = conv.shape[-1]
        w = torch.cat([conv, mixed.to(conv.dtype)], dim=-1)                # [B, C, n + T]
        hn[:, :, :n + T].copy_(w)
        out = F.conv1d(w, m.conv1d.weight, m.conv1d.bias, groups=w.shape[1])[:, :, -T:]
        mixed = F.silu(out).to(x.dtype).transpose(1, 2)                    # [B, T, C]
        q, k, v = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(B, T, -1, m.head_k_dim)
        k = k.reshape(B, T, -1, m.head_k_dim)
        v = v.reshape(B, T, -1, m.head_v_dim)
        beta = b.sigmoid()
        g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
        r = m.num_v_heads // m.num_k_heads
        if r > 1:
            q, k = q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2)
        Hv = v.shape[2]
        tb = lambda t: t.float().transpose(0, 1).reshape(T, B * Hv, *t.shape[3:]).contiguous()   # [T, B*Hv, ...]
        o = gdn_fused(tb(q), tb(k), tb(v), g.float().transpose(0, 1).reshape(T, B * Hv).contiguous(),
                      beta.float().transpose(0, 1).reshape(T, B * Hv).contiguous(),
                      rec.view(B * Hv, m.head_k_dim, m.head_v_dim), hist[:T], PACK_TAU)       # [T, B*Hv, dv]
        o = o.to(x.dtype).view(T, B, Hv, m.head_v_dim).transpose(0, 1)                         # [B, T, Hv, dv]
        o = m.norm(o.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(B, T, -1)
        return m.out_proj(o)

    def _attn(self, m, x, T, kv, cos, sin):
        B, hd = self.B, m.head_dim
        q, gate = torch.chunk(m.q_proj(x).view(B, T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = m.q_norm(q).transpose(1, 2)
        k = m.k_norm(m.k_proj(x).view(B, T, -1, hd)).transpose(1, 2)
        v = m.v_proj(x).view(B, T, -1, hd).transpose(1, 2)
        q, k = self.M.apply_rotary_pos_emb(q, k, cos, sin)
        kv.write(k, v, self.pos)
        o, _ = kv.attn(q, (self.pos - kv.lc).clamp(min=0).view(B, 1) + self.tar[None, :T] + 1, m.scaling)
        return m.o_proj(o.to(x.dtype).transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate))

    def _step(self, T):
        B = self.B
        h = self.h_in[:, :T]
        posT = (self.pos[:, None] + self.tar[None, :T]).clamp(max=self.L - 1)        # [B, T]
        cos, sin = self.model.model.rotary_emb(h, posT.view(1, B, T).expand(3, B, T))
        caps = {}
        for i, (layer, st) in enumerate(zip(self.layers, self.state)):
            x = layer.input_layernorm(h)
            if st[0] == "gdn":
                o = self._gdn(layer.linear_attn, x, T, *st[1:])
            else:
                o = self._attn(layer.self_attn, x, T, st[1], cos, sin)
            h = h + o
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            if i in self.caps:
                caps[i] = h
        hn = self.model.model.norm(h)
        if len(self.layers) in self.caps:                    # drafter source "final_postnorm"
            caps[len(self.layers)] = hn
        if self.caps:
            self.feats[:, :T].copy_(torch.cat([caps[i] for i in self.caps], dim=-1))
        logits = self.model.lm_head(hn)
        self.tok_out[:, :T].copy_(logits.argmax(-1))
        if self.logits_out is None:
            self.logits_out = torch.empty(self.B, self.T, logits.shape[-1], dtype=torch.float32, device=self.dev)
        self.logits_out[:, :T].copy_(logits)

    def _commit(self):
        c = self.keep                                                      # [B], tokens kept per slot (1..T)
        B = self.B
        for st in self.state:
            if st[0] == "gdn":
                conv, rec, hn, hist = st[1:]
                n = conv.shape[-1]
                Hv = rec.shape[1]
                rec.copy_(hist.view(hist.shape[0], B, Hv, *rec.shape[2:])[c - 1, self.bidx])
                idx = (c[:, None] + torch.arange(n, device=self.dev)[None, :])[:, None, :].expand(B, hn.shape[1], n)
                conv.copy_(torch.gather(hn, 2, idx))
        self.pos.add_(c).clamp_(max=self.L - 1 - self.T)                  # free slots must stay in range
        for st in self.state:                                              # qvb8 KV: c keys kept, a full tail sealed
            if st[0] == "attn":
                st[1].set_len(self.pos)

    @torch.no_grad()
    def capture(self, warm=2):
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        pool = None
        for T in sorted({1, self.T}):
            with torch.cuda.stream(s):
                for _ in range(warm):
                    self._step(T)
            s.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s, pool=pool):
                self._step(T)
            pool = g.pool()
            self.graphs[T] = g
        with torch.cuda.stream(s):
            self._commit()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s, pool=pool):
            self._commit()
        self.commit_graph = g
        torch.cuda.synchronize(self.dev)
        self.pos.zero_()

    @torch.no_grad()
    def run(self, tokens, T):
        """tokens: [B][T] ids (padding for slots that do not draft / are free). Returns tok_out [B][T]."""
        e = self.model.model.embed_tokens(torch.tensor(tokens, device=self.dev).view(self.B, T))
        self.h_in[:, :T].copy_(e.to(self.dev, non_blocking=True))
        g = self.graphs.get(T)
        if g is not None:
            g.replay()
        else:
            self._step(T)
        return self.tok_out[:, :T].tolist()

    @torch.no_grad()
    def commit(self, keep):
        self.keep.copy_(torch.tensor(keep, dtype=torch.long, device=self.dev))
        if self.commit_graph is not None:
            self.commit_graph.replay()
        else:
            self._commit()

    # ------------------------------------------------------------------ a new conversation into slot b
    @torch.no_grad()
    def prefill_slot(self, b, ids, drafter=None, chunk=int(os.environ.get("QV_TP_PREFILL_CHUNK", "512"))):
        from transformers import DynamicCache
        n = len(ids)
        assert n < self.L - 1 - self.T, f"prompt of {n} tokens: slot context is {self.L}"
        cache = DynamicCache(config=self.model.config)
        t = torch.tensor([ids], device=self.dev)
        last = None
        if drafter is not None:
            drafter._reset_state()
        for p in range(0, n, chunk):
            out = self.model.model(input_ids=t[:, p:p + chunk], past_key_values=cache, use_cache=True,
                                   output_hidden_states=drafter is not None and bool(self.caps))
            last = out.last_hidden_state[:, -1:]
            if drafter is not None and self.caps:
                drafter.observe_features(p, tap_features(out, self.caps))
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
        if drafter is not None:
            drafter.reset(ids + [tok])
            drafter.pos = n + 1
        return tok


def _spec_step(bd, drafters, slots, active, draft_mask, tp, device, sp=None):
    """One shared step on every card: draft (slots in draft_mask), broadcast the drafts, verify all, commit.
    sp (card 0): per slot (temperature, top_k, top_p) or None (greedy). Card 0 decides what each slot keeps (greedy,
    or speculative sampling: engine/spec.verify_row) and broadcasts it. Returns per slot (new tokens, n_accepted,
    n_proposed), the same on every card."""
    from qv_band.engine.spec import accept, verify_row
    B, T, k = bd.B, bd.T, bd.T - 1
    last = slots["last"]
    sp = sp or [None] * B
    drafts, dls = [[] for _ in range(B)], [None] * B
    for b in range(B):
        if draft_mask[b]:
            d, dls[b] = drafters[b].propose(last[b], k, *(sp[b] or ()))
            d = list(d)[:k]
            drafts[b] = d if len(d) == k else []
    if tp.world > 1:                                                  # card 0's drafts on every card
        buf = torch.full((B, k + 1), -1, dtype=torch.long, device=device)
        for b in range(B):
            buf[b, 0] = len(drafts[b])
            if drafts[b]:
                buf[b, 1:] = torch.tensor(drafts[b], device=device)
        torch.distributed.broadcast(buf, 0, group=tp.group)
        drafts = [buf[b, 1:1 + int(buf[b, 0])].tolist() for b in range(B)]
    Tr = T if any(drafts) else 1
    toks = [[last[b]] + (drafts[b] if Tr > 1 and drafts[b] else [last[b]] * (Tr - 1)) for b in range(B)]
    tgt = bd.run(toks, Tr)
    res, keep, nxt = [None] * B, [1] * B, [0] * B
    for b in range(B):
        if sp[b] is not None and sp[b][0] > 0:
            n_acc, nxt[b] = verify_row(drafts[b], dls[b] if drafts[b] else None, bd.logits_out[b, :len(drafts[b]) + 1],
                                       *sp[b], getattr(drafters[b], "returns_probs", False))
        else:
            n_acc = accept(drafts[b], tgt[b])
            nxt[b] = tgt[b][n_acc]
        keep[b] = n_acc + 1
    if tp.world > 1:                                                  # card 0's decision on every card
        dec_ = torch.tensor(keep + nxt, dtype=torch.long, device=device)
        torch.distributed.broadcast(dec_, 0, group=tp.group)
        d_ = dec_.tolist()
        keep, nxt = d_[:B], d_[B:]
    for b in range(B):
        n_acc = keep[b] - 1
        res[b] = (drafts[b][:n_acc] + [nxt[b]], n_acc, len(drafts[b]))
    bd.commit(keep)
    for b in range(B):
        if active[b] and drafters[b] is not None:
            d = drafters[b]
            d.observe_features(d._ctx_end, bd.feats[b, :keep[b]])
            d.pos = d._ctx_end + 1
    return res


def serve_rank0(bd, drafters, jobs, eos, tp, device, state):
    """Card 0: admit queued jobs into free slots (prefill), then one shared draft+verify step for all slots.
    job: {"ids", "max_new", "out": queue} -- tokens are put on job["out"] as lists, None at the end."""
    import queue as _q
    B = bd.B
    slots = [None] * B
    plan = SpecPlanner()                                  # card 0 decides who drafts (engine/spec.py); the others follow
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
        for i in range(B):
            if slots[i] is not None:
                continue
            try:
                job = jobs.get(block=not any(slots))
            except _q.Empty:
                break
            state["queue"] -= 1
            ids = job["ids"]
            bcast(2, i, len(ids), torch.tensor(ids, dtype=torch.long, device=device))
            tok = bd.prefill_slot(i, ids, drafters[i])
            if job.get("temperature", 0) > 0:                 # sampled: card 0 picks (the next step carries it)
                from qv_band.engine.spec import sample_token
                tok = sample_token(bd.first_logits[i], job["temperature"], job.get("top_k", 0), job.get("top_p", 1.0))
            plan.forget(i)
            job["out"].put([tok])
            slots[i] = {"job": job, "n": 1, "last": tok}
            if tok in eos or job["max_new"] <= 1:
                finish(i)
        active = [slots[i] is not None for i in range(B)]
        if not any(active):
            continue
        cand = [b for b in range(B) if active[b] and drafters[b] is not None]
        ks = dict(zip(cand, plan.plan_k([(b, bd.T - 1) for b in cand], whole=True)))   # the graphs: all drafts or none
        mask = [ks.get(b, 0) > 0 for b in range(B)]
        last = {"last": [slots[b]["last"] if slots[b] is not None else 0 for b in range(B)]}
        bcast(3, 0, 0, torch.tensor(mask + active + last["last"], dtype=torch.long, device=device))
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        sp = [(slots[b]["job"]["temperature"], slots[b]["job"].get("top_k", 0), slots[b]["job"].get("top_p", 1.0))
              if slots[b] is not None and slots[b]["job"].get("temperature", 0) > 0 else None for b in range(B)]
        res = _spec_step(bd, drafters, last, active, mask, tp, device, sp)
        drafted = any(res[b][2] for b in range(B) if active[b])
        plan.observe(plan.vcost, B * (bd.T if drafted else 1), time.perf_counter() - t0)
        for b in range(B):
            if not active[b]:
                continue
            new, n_acc, n_prop = res[b]
            if n_prop:
                plan.feedback(b, n_acc, n_prop)
            s = slots[b]
            cut = next((j for j, t in enumerate(new) if t in eos), None)
            if cut is not None:
                new = new[:cut + 1]
            new = new[:max(0, s["job"]["max_new"] - s["n"])]
            state["tokens"] += len(new)
            if new:
                s["job"]["out"].put(new)
                s["n"] += len(new)
                s["last"] = new[-1]
            if (new and new[-1] in eos) or s["n"] >= s["job"]["max_new"]:
                finish(b)
        state["busy"] = any(slots)


def serve_follower(bd, drafters, tp, device):
    B = bd.B
    hdr = torch.zeros(3, dtype=torch.long, device=device)
    last = [0] * B
    while True:
        torch.distributed.broadcast(hdr, 0, group=tp.group)
        cmd, a, b = hdr.tolist()
        if cmd == 0:
            return
        if cmd == 2:
            ids = torch.empty(b, dtype=torch.long, device=device)
            torch.distributed.broadcast(ids, 0, group=tp.group)
            last[a] = bd.prefill_slot(a, ids.tolist(), drafters[a])
        elif cmd == 3:
            m = torch.empty(3 * B, dtype=torch.long, device=device)
            torch.distributed.broadcast(m, 0, group=tp.group)
            m = m.tolist()
            mask, active, last = [bool(x) for x in m[:B]], [bool(x) for x in m[B:2 * B]], m[2 * B:]
            res = _spec_step(bd, drafters, {"last": last}, active, mask, tp, device)
            for i in range(B):
                if active[i]:
                    last[i] = res[i][0][-1]
