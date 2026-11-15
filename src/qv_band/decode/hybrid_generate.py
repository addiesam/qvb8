"""Decode loop kind: Mamba-2 hybrid stack; used by Nemotron-H (Mamba-2 / attention / LatentMoE), plain or speculative.

`speculative_generate` builds a transformers DynamicCache (Qwen layer types) and its rollback helpers know
Qwen's recurrent-state layout; the Nemotron model carries its own NemotronHHybridDynamicCache. This has the
same call surface (`gen_core.run_generate` dispatches here when the model's adapter returns it from generate_fn()) and the same
return value, `(out_ids, Stats)` -- or `(out_ids, Stats, None)` with return_cache.

Drafting: a block drafter (LiLiCorrDrafter / DFlash: `capture_layers`, `observe_features`, `propose`) proposes
up to 7 tokens, the target verifies [last_token + drafts] in ONE forward and keeps the accepted prefix plus its
own next token. A KV cache is cropped to undo rejected tokens; a Mamba state cannot be, so the verify forward
runs with `cache.trace` set, which makes every Mamba layer keep its state after each of the <=8 tokens
(layers/mamba2_hybrid._mamba2_forward) and `rollback` restores the one after the last kept token. Greedy output is
token-identical to plain greedy decoding. At temperature > 0 the draft is a deterministic path (a point-mass
proposal), so acceptance is the standard rejection test with q = 1: accept d with probability p(d), else
resample from p with d removed -- exact sampling from the target.

The text model is called through `model.model` and the head applied to the needed positions only: the shipped
forward would materialise [chunk, vocab] float32 logits for every prefill chunk.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time

import torch

from qv_band.decode.spec_decode import Stats, _IncStreamer, _probs


def new_cache(model, device):
    """Hybrid cache with bf16 conv state and fp32 SSM state (config.mamba_ssm_cache_dtype). ONE cache object per model,
    zeroed in place: the CUDA graphs of the Mamba blocks hold the addresses of its state tensors."""
    import importlib
    cfg = model.config
    cache = getattr(model, "_qv_cache", None)
    if cache is None:
        mod = importlib.import_module(type(model).__module__.rsplit(".", 1)[0] + ".modeling_nemotron_h")
        cache = mod.NemotronHHybridDynamicCache(cfg, 1, dtype=torch.bfloat16, device=device)
        for i, t in enumerate(cfg.layers_block_type):
            if t == "mamba":
                cache.ssm_states[i] = cache.ssm_states[i].float()
        model._qv_cache = cache
    else:
        for i, t in enumerate(cfg.layers_block_type):
            if t == "mamba":
                cache.conv_states[i].zero_()
                cache.ssm_states[i].zero_()
            cache.key_cache[i] = torch.tensor([[]], device=device)
            cache.value_cache[i] = torch.tensor([[]], device=device)
        cache.has_previous_state = False
    cache.trace = None
    return cache


def _attn_len(cache) -> int:
    return cache.get_seq_length()


@contextlib.contextmanager
def _capture(model, layer_ids):
    """Forward hooks collecting the OUTPUT of the listed decoder layers; yields a dict filled per forward."""
    store: dict[int, torch.Tensor] = {}
    hs = [model.model.layers[i].register_forward_hook(
        lambda m, a, o, i=i: store.__setitem__(i, o[0] if isinstance(o, tuple) else o))
        for i in layer_ids]
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


class _NGram:
    """Prompt-lookup drafter: the tokens that followed the most recent earlier occurrence of the current n-token
    suffix (n = `n_long` first, then `n_short`). Incremental index, O(1) per token."""

    def __init__(self, ctx, n_long=5, n_short=0):
        self.ctx, self.ns = [], [n for n in (n_long, n_short) if n > 0]
        self.idx = {n: {} for n in self.ns}
        self.extend(ctx)

    def extend(self, toks):
        for t in toks:
            self.ctx.append(t)
            L = len(self.ctx)
            for n in self.ns:
                if L > n:                                   # key = n tokens ending BEFORE the newest one
                    self.idx[n][tuple(self.ctx[L - 1 - n:L - 1])] = L - 1

    def propose(self, k):
        L = len(self.ctx)
        for n in self.ns:
            if L < n:
                continue
            pos = self.idx[n].get(tuple(self.ctx[L - n:L]))
            if pos is not None and pos < L - n + 0:
                cont = self.ctx[pos:pos + k]
                if cont:
                    return cont
        return []


def _sample(logits, temperature, top_k, top_p):
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    p = _probs(logits.float(), temperature, top_k, top_p)
    return int(torch.multinomial(p, 1).item())


@torch.no_grad()
def hybrid_generate(model, drafter, tok, ids, *, max_new: int = 40, stream: bool = True,
                      temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
                      kv_stats: dict | None = None, skip_special_tokens: bool = True,
                      abort_check=None, prefix_cache=None, return_cache: bool = False,
                      on_heartbeat=None, prefill_chunk: int | None = None, embeds=None, on_token=None, **_ignored):
    from qv_band.layers.mamba2_hybrid import rollback

    st = Stats()
    _t_start = time.time()
    device = ids.device
    eos = set(getattr(model, "_qv_eos_ids", ())) or {tok.eos_token_id}
    cache = new_cache(model, device)
    chunk = int(prefill_chunk or 16384)
    n = ids.shape[1]
    caps = list(getattr(drafter, "capture_layers", None) or []) if drafter is not None else []
    spec = bool(caps) and hasattr(drafter, "propose")
    out_ids: list[int] = []
    ng = None
    if spec and int(os.environ.get("QV_NEMO_NGRAM", "0")) > 0:
        ng = _NGram(ids[0].tolist(), n_long=int(os.environ["QV_NEMO_NGRAM"]),
                    n_short=int(os.environ.get("QV_NEMO_NGRAM_SHORT", "0")))
    inc = _IncStreamer(tok) if stream else None
    done = lambda: (out_ids, st, None) if return_cache else (out_ids, st)

    def emit(new):
        if on_token is not None:
            for t_ in new:
                on_token(t_)
        if inc is not None:
            piece = inc.feed(new)
            if piece:
                print(piece, end="", flush=True)

    # ---- prefill (chunked; the drafter sees every position's captured features) ------------------------
    t0 = time.time()
    last = None
    # the drafter only ever attends its last `ctx_keep` context positions (sliding window): features of earlier prompt
    # chunks would be projected and then dropped, so they are neither captured nor observed
    if spec and hasattr(drafter, "_reset_state"):
        drafter._reset_state()                   # a previous request's context must never leak into this one
    need_from = max(0, n - (getattr(drafter, "ctx_keep", None) or n) - 16) if spec else n
    cm = _capture(model, caps) if spec else contextlib.nullcontext({})
    with cm as store:
        for c0 in range(0, n, chunk):
            part = ids[:, c0:c0 + chunk]
            if embeds is not None:                   # image-merged input embeddings (ids still index the context)
                last = model.model(inputs_embeds=embeds[:, c0:c0 + chunk], past_key_values=cache, use_cache=True)[0][:, -1]
            else:
                last = model.model(part, past_key_values=cache, use_cache=True)[0][:, -1]
            if spec and c0 + part.shape[1] > need_from:
                drafter.observe_features(c0, torch.cat([store[i] for i in caps], dim=-1)[0])
            if c0 + chunk < n:
                if on_heartbeat:
                    on_heartbeat()
                if time.time() - t0 > 2.0:
                    print(f"[prefill] {c0 + part.shape[1]}/{n} tok", file=sys.__stdout__, flush=True)
                    t0 = time.time()
                if abort_check is not None and abort_check():
                    return done()
    logits = model.lm_head(last).float()[0]
    t_dec = time.time()
    tok0 = _sample(logits, temperature, top_k, top_p)
    st.t_prefill = time.time() - _t_start        # the sample above synchronises the GPU
    st.n_prefill = n
    out_ids.append(tok0)
    st.steps += 1
    st.emitted += 1
    if tok0 in eos or max_new <= 1:
        st.t_verify = time.time() - t_dec
        if tok0 not in eos:
            emit([tok0])
        if inc is not None:
            tail = inc.finish()
            if tail:
                print(tail, end="", flush=True)
        return done()
    emit([tok0])
    if ng is not None:
        ng.extend([tok0])
    if spec:
        drafter.reset(ids[0].tolist() + [tok0])
        drafter.pos = n + 1                      # next token after tok0 (tok0 itself is at position n)

    # ---- decode ---------------------------------------------------------------------------------------------
    while len(out_ids) < max_new:
        last_tok = out_ids[-1]
        drafts: list[int] = []
        if spec:
            t1 = time.time()
            if ng is not None:
                drafts = ng.propose(drafter.block_size - 1)
                st.ngram_steps = getattr(st, "ngram_steps", 0) + bool(drafts)
            if not drafts:
                drafts, _ = drafter.propose(last_tok, drafter.block_size - 1)
            st.t_draft += time.time() - t1
        m = len(drafts)
        block = torch.tensor([[last_tok] + drafts], device=device)
        a_len = _attn_len(cache)
        cache.trace = {} if m > 0 else None
        t1 = time.time()
        with (_capture(model, caps) if spec else contextlib.nullcontext({})) as store:
            h = model.model(block, past_key_values=cache, use_cache=True)[0]
            feats = torch.cat([store[i] for i in caps], dim=-1)[0] if spec else None
        lg = model.lm_head(h).float()[0]                                    # (m+1, V)
        # accept
        new: list[int] = []
        n_acc = 0
        if temperature <= 0:
            tgt = lg.argmax(-1).tolist()
            while n_acc < m and drafts[n_acc] == tgt[n_acc]:
                n_acc += 1
            new = drafts[:n_acc] + [tgt[n_acc]]
        else:
            while n_acc < m:
                p = _probs(lg[n_acc], temperature, top_k, top_p)
                if torch.rand((), device=p.device).item() < p[drafts[n_acc]].item():
                    n_acc += 1
                    continue
                p[drafts[n_acc]] = 0.0                                      # residual of a point-mass proposal
                new = drafts[:n_acc] + [int(torch.multinomial(p / p.sum(), 1).item())]
                break
            else:
                new = drafts + [_sample(lg[m], temperature, top_k, top_p)]
        if m > 0 and n_acc < m:
            rollback(model, cache, n_acc + 1, a_len)
        cache.trace = None
        st.t_verify += time.time() - t1
        st.steps += 1
        st.proposed += m
        st.accepted += n_acc
        if spec:
            drafter.observe_features(drafter._ctx_end, feats[:n_acc + 1])
            drafter.pos = drafter._ctx_end + 1 + 0
        # eos / budget
        cut = next((j for j, t in enumerate(new) if t in eos), None)
        if cut is not None:
            new = new[:cut + 1]
        new = new[:max_new - len(out_ids)]
        out_ids.extend(new)
        if ng is not None:
            ng.extend(new)
        st.emitted += len(new)
        shown = [t for t in new if t not in eos]
        if shown:
            emit(shown)
        if on_heartbeat:
            on_heartbeat()
        if cut is not None or (abort_check is not None and abort_check()):
            break
    st.t_verify = max(st.t_verify, 0.0)
    if inc is not None:
        tail = inc.finish()
        if tail:
            print(tail, end="", flush=True)
    return done()
