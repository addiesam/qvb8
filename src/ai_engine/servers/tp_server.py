"""OpenAI-compatible server for ONE model service on n cards: Tensor Parallel (TP).

    torchrun --nproc-per-node 2 -m ai_engine.servers.tp_server PACK [--draft DFLASH_PACK] [--port 8080]
             [--host 0.0.0.0] [--max-len 16384] [--tp-weights 1,1]

Every card holds its share of every matrix of the model (qv_band/design_doc/tensor_parallel.md); one
message is computed by all cards together. Card 0 (rank 0) runs HTTP and the queue; for each request it broadcasts
[cmd, prompt length, max new tokens] + the prompt ids, and every card runs the same deterministic generate()
(prefill through the chunked HF forward for long prompts, decode as CUDA graphs, DFlash with exact verification).
Card 0 streams the tokens. To the router it is one replica: GET /v1/pool/state answers with the model family and
its load, like llm_server.

One request at a time per service (batch 1: the single-message speed path), or --batch N / --pp. A request's
temperature, top_k and top_p apply in every mode (temperature 0: greedy); with a drafter, sampled requests use
speculative sampling (the target's own distribution). Requests wait in the queue in arrival order. Tools: the chat template's own
contract, parsed with llm_server.parse_tool_calls into OpenAI tool_calls (stream and non-stream).

Endpoints: GET /health, /v1/models, /v1/pool/state; POST /v1/chat/completions, /v1/completions (stream or not).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time

import torch

STATE = {"served": 0, "tokens": 0, "busy": False, "queue": 0, "queued_prompt": 0, "running_prompt": 0,
         "running_decode": 0, "queued_decode": 0}
JOBS: "queue.Queue[dict]" = queue.Queue()
VIS = None            # card 0: generation.vision.VisionAdapter (image chats), None for a text-only model
TOK = None
PMS = []          # the partial matrices this service is made of (one per card), reported to the router
ARGS = None
MODEL_ID = None


# ------------------------------------------------------------------------------------------------ HTTP (card 0)
def _family(pack: str) -> str:
    try:
        with open(os.path.join(pack, "manifest.json")) as f:
            fam = os.path.basename(str(json.load(f).get("hf_model", "")).rstrip("/"))
    except (OSError, ValueError):
        fam = ""
    return fam or os.path.basename(pack.rstrip("/"))


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", None))
    return str(content or "")


DEFAULT_THINK = True    # --think / --no-think: the same default as servers/llm_server.py (single / dual / pair of the
                        # same model must answer the same way; this server had no flag and never thought by default)


def _think(body: dict) -> bool:
    return bool(body.get("think", (body.get("chat_template_kwargs") or {}).get("enable_thinking", DEFAULT_THINK)))


def _prompt_ids(body: dict, chat: bool) -> list[int]:
    """The chat template's own tool contract (as llm_server.build_prompt): "tools" render as the template's <tools>
    block, the model answers <tool_call><function=NAME><parameter=K>v</parameter></function></tool_call>."""
    if not chat:
        return TOK(str(body.get("prompt", ""))).input_ids
    from ai_engine.servers.llm_server import _denormalise_tool_calls
    msgs = _denormalise_tool_calls([{**m, "content": _text_of(m.get("content"))} if m.get("content") is not None
                                    else dict(m) for m in body.get("messages", [])])
    text = TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   tools=body.get("tools") or None, enable_thinking=_think(body))
    return TOK(text).input_ids


def _turn_mark(body: dict, chat: bool, ids: list[int]) -> int:
    """Where the conversation so far ends: the tokens the prompt shares with the same messages WITHOUT the generation
    prompt. A prefix checkpoint there lets the next turn reuse the cache even when the template writes a past answer
    differently from its own generation prompt (Qwen3.5: "assistant\n<think>..." vs "assistant\nAnswer", the next turn matched up to 4 tokens before the prompt end, where no checkpoint was, and prefilled everything)."""
    if not chat:
        return 0
    from ai_engine.servers.llm_server import _denormalise_tool_calls
    msgs = _denormalise_tool_calls([{**m, "content": _text_of(m.get("content"))} if m.get("content") is not None
                                    else dict(m) for m in body.get("messages", [])])
    ids0 = TOK(TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                       tools=body.get("tools") or None, enable_thinking=_think(body))).input_ids
    n = min(len(ids), len(ids0))
    i = 0
    while i < n and ids[i] == ids0[i]:
        i += 1
    return i if 0 < i < len(ids) else 0


def _split_answer(text: str, has_tools: bool):
    """(content, reasoning, tool_calls) of a finished answer, with llm_server's own parsers."""
    from ai_engine.servers.llm_server import parse_tool_calls
    reasoning = None
    if "</think>" in text:
        reasoning, _, text = text.partition("</think>")
        reasoning = reasoning.replace("<think>", "").strip() or None
        text = text.lstrip()
    elif text.lstrip().startswith("<think>"):
        # max_tokens ran out inside the reasoning: it is reasoning, not the answer (the raw "<think>..." went out as
        # content); the answer is llm_server's own notice, so both servers reply the same way
        return ("[thinking was cut off before an answer was produced -- try again with a larger max_tokens, or "
                "--no-think]", text.replace("<think>", "", 1).strip() or None, [])
    calls = []
    if has_tools and "<tool_call>" in text:
        text, calls = parse_tool_calls(text)
        calls = [{"index": i, **c} for i, c in enumerate(calls)]
    return (text.strip() or None) if calls else text, reasoning, calls


def _handler_cls():
    from ai_engine.web.base_handler import QVHTTPHandler

    class Handler(QVHTTPHandler):
        def do_GET(self):
            p = self.path.rstrip("/")
            if p in ("/health", ""):
                return self._send(200, {
                    "status": "ok", "model": MODEL_ID, "pack": ARGS.model, "mode": "tp",
                    "cards": int(os.environ.get("WORLD_SIZE", "1")), "tp_weights": ARGS.tp_weights,
                    "drafter": os.path.basename(ARGS.draft) if ARGS.draft else None, "ctx_size": ARGS.max_len,
                    "sampling": "per request (temperature, top_k, top_p)", "state_tau": os.environ.get("QV_STATE_TAU") or None,
                    "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                    "requests_served": STATE["served"], "tokens_generated": STATE["tokens"],
                    "busy": STATE["busy"], "queued": STATE["queue"]})
            if p == "/v1/models":
                return self._send(200, {"object": "list", "data": [
                    {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "local"}]})
            if p == "/v1/pool/state":
                return self._send(200, {
                    "model": MODEL_ID, "pack": ARGS.model, "family": _family(ARGS.model), "ctx_size": ARGS.max_len,
                    "pms": PMS,
                    "scheduler": True, "prefixes": [], "vision": VIS is not None,
                    "load": {"queued_prompt_tokens": STATE["queued_prompt"],
                             "running_prefill_tokens": STATE["running_prompt"],
                             "queued_decode_tokens": STATE["queued_decode"],
                             "running_decode_tokens": STATE["running_decode"],
                             "running": int(STATE["busy"]), "queued": STATE["queue"]}})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            p = self.path.rstrip("/")
            if p not in ("/v1/chat/completions", "/v1/completions"):
                return self._send(404, {"error": "not found"})
            chat = p.endswith("chat/completions")
            try:
                body = self._body()
                from ai_engine.generation.vision import has_images
                vision = chat and has_images(body.get("messages", []))
                if vision and VIS is None:
                    return self._send(400, {"error": "this model takes no images"})
                ids = [] if vision else _prompt_ids(body, chat)
                mark = 0 if vision else _turn_mark(body, chat, ids)
            except Exception as e:                                   # noqa: BLE001
                return self._send(400, {"error": f"bad request: {e}"})
            max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)
            room = ARGS.max_len - len(ids) - 16
            if room <= 0:
                return self._send(400, {"error": f"prompt is {len(ids)} tokens; this service holds {ARGS.max_len}"})
            max_new = max(1, min(max_new, room))
            try:
                temperature = float(body.get("temperature") or 0.0)
                top_k = int(body.get("top_k") or 0)
                top_p = float(body.get("top_p") if body.get("top_p") is not None else 1.0)
            except (TypeError, ValueError):
                return self._send(400, {"error": "temperature / top_k / top_p must be numbers"})
            job = {"ids": ids, "max_new": max_new, "out": queue.Queue(), "t0": time.time(), "mark": mark,
                   "temperature": max(0.0, temperature), "top_k": max(0, top_k), "top_p": min(max(top_p, 0.0), 1.0),
                   "tools": bool(body.get("tools")), "think": _think(body),
                   "vision": body.get("messages") if vision else None, "tool_list": body.get("tools") or None}
            job["q_prompt"], job["q_decode"] = len(ids), max_new
            STATE["queue"] += 1
            STATE["queued_prompt"] += len(ids)
            STATE["queued_decode"] += max_new
            JOBS.put(job)
            created, cid = int(time.time()), f"{'chatcmpl' if chat else 'cmpl'}-{int(time.time() * 1000)}"
            if body.get("stream"):
                return self._stream(job, chat, cid, created)
            toks = []
            while True:
                item = job["out"].get()
                if item is None:
                    break
                toks.extend(item)
            if job.get("error"):
                return self._send(400, {"error": job["error"]})
            ids = job["ids"]                                         # an image chat: set by the serving loop
            text = TOK.decode(toks, skip_special_tokens=True).rstrip("\ufffd")   # a cut multi-byte character
            fin = "length" if len(toks) >= max_new else "stop"
            content, reasoning, calls = _split_answer(("<think>" if job["think"] else "") + text, job["tools"])
            usage = {"prompt_tokens": len(ids), "completion_tokens": len(toks),
                     "total_tokens": len(ids) + len(toks)}
            if chat:
                msg = {"role": "assistant", "content": content}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                if calls:
                    msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
                    fin = "tool_calls"
                return self._send(200, {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                                        "choices": [{"index": 0, "finish_reason": fin, "message": msg}],
                                        "usage": usage})
            return self._send(200, {"id": cid, "object": "text_completion", "created": created, "model": MODEL_ID,
                                    "choices": [{"index": 0, "text": text, "finish_reason": fin}], "usage": usage})

        def _stream(self, job, chat, cid, created):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", self._cors_value())
            self._end_close()
            self.end_headers()
            alive = True

            def sse(obj):
                nonlocal alive
                if not alive:
                    return
                try:
                    self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                except OSError:
                    alive = False          # the client left; the cards still finish this request (v1: no abort)

            def chunk(text, fin=None):
                if chat:
                    delta = {"content": text} if text else {}
                    return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
                return {"id": cid, "object": "text_completion", "created": created, "model": MODEL_ID,
                        "choices": [{"index": 0, "text": text, "finish_reason": fin}]}

            if chat:
                sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                     "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            toks, shown = [], ""
            hold = "<tool_call>"
            while True:
                item = job["out"].get()
                if item is None:
                    break
                toks.extend(item)
                text = TOK.decode(toks, skip_special_tokens=True)
                if text.endswith("\ufffd"):                # an incomplete multi-byte character: wait
                    continue
                if job["think"] and "</think>" not in text:
                    continue                                # reasoning: sent at the end as reasoning_content
                vis = text.split("</think>", 1)[1].lstrip() if job["think"] else text
                if job["tools"]:
                    cut = vis.find(hold)
                    if cut < 0:                             # keep back a possible partial "<tool_call"
                        cut = max(0, len(vis) - len(hold))
                        for j in range(1, len(hold)):
                            if vis.endswith(hold[:j]):
                                cut = len(vis) - j
                                break
                        else:
                            cut = len(vis)
                    vis = vis[:cut]
                if chat and len(vis) > len(shown):
                    sse(chunk(vis[len(shown):]))
                    shown = vis
                elif not chat and len(vis) > len(shown):
                    sse(chunk(vis[len(shown):]))
                    shown = vis
            text = TOK.decode(toks, skip_special_tokens=True).rstrip("\ufffd")
            content, reasoning, calls = _split_answer(("<think>" if job["think"] else "") + text, job["tools"])
            fin = "length" if len(toks) >= job["max_new"] else "stop"
            if chat:
                last = {}
                if reasoning:
                    last["reasoning_content"] = reasoning
                rest = (content or "")[len(shown):] if (content or "").startswith(shown) else ""
                if rest:
                    last["content"] = rest
                if calls:
                    last["tool_calls"] = calls
                    fin = "tool_calls"
                sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                     "choices": [{"index": 0, "delta": last, "finish_reason": fin}],
                     "usage": {"prompt_tokens": len(job["ids"]), "completion_tokens": len(toks),
                               "total_tokens": len(job["ids"]) + len(toks)}})
            else:
                rest = text[len(shown):]
                sse(chunk(rest, fin))
            if alive:
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass

    return Handler


def _pm_table(tcfg, world, weights):
    """Each card's partial matrix in head units (the same cut for every layer), for the router: which KV heads,
    DeltaNet key heads, MLP columns and experts it holds. Computed from the config, identical on every card."""
    from qv_band.formats.qvb8.tp import TPGroup
    from qv_band.run.tp_qwen35 import Plan
    out = []
    for r in range(world):
        p = Plan(tcfg, TPGroup(r, world), weights)
        pm = {"card": r, "share": (weights[r] / sum(weights)) if weights else 1.0 / world,
              "kv_heads": [p.attn_kv[0][0] // tcfg.head_dim, p.attn_kv[0][1] // tcfg.head_dim],
              "deltanet_key_heads": [p.gdn_qkv[0][0] // tcfg.linear_key_head_dim,
                                     p.gdn_qkv[0][1] // tcfg.linear_key_head_dim]}
        if p.moe:
            from qv_band.run.tp_moe import expert_range
            pm["experts"] = list(expert_range(p.moe, TPGroup(r, world), weights))
        else:
            pm["mlp_columns"] = list(p.mlp_down)
        out.append(pm)
    return out


def _pp_main(a, device):
    """--pp: one pipeline stage per machine (run/tp_pipe.py). Stages before the last only compute and pass hidden
    states on; every card of the last stage runs the generate() loops (drafter included), its first card serves
    HTTP. --pp-slots N: N requests in flight at once (one KV cache per slot on every stage; same N on every machine)."""
    global TOK, MODEL_ID
    import queue as _q
    from transformers import AutoTokenizer
    from qv_band.run import tp_pipe as PPm
    p = PPm.Pipe()
    tp = p.tp
    st, model, caps = PPm.build_stage(p, a.model, device, a.draft, a.max_len, 8, a.pp_slots, a.kv_window,
                                      verbose=(tp.rank == 0))
    print(f"[tp-server] pp stage {p.stage}/{p.n}: layers {st.lo}..{st.hi - 1} on {tp.world} card(s), "
          f"{a.pp_slots} slot(s)", flush=True)
    torch.distributed.barrier()
    if not p.last:
        PPm.serve(st)
        os._exit(0)
    drafter = None
    if a.draft:
        from qv_band.decode.dflash_draft import build_dflash_drafter
        drafter = build_dflash_drafter(model, a.draft, device=device)
    TOK = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    MODEL_ID = os.path.basename(a.model.rstrip("/"))
    eos = {x for x in [TOK.eos_token_id, TOK.convert_tokens_to_ids("<|im_end|>"),
                       TOK.convert_tokens_to_ids("<|endoftext|>")] if isinstance(x, int) and x >= 0}
    drv = PPm.Driver(st, drafter, tp if tp.world > 1 else None, eos)
    if p.is_leader:
        from ai_engine.web.base_handler import QVHTTPServer
        srv = QVHTTPServer((a.host, a.port), _handler_cls())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[tp-server] {MODEL_ID}: {p.n} pipeline stages {p.stages}, {a.pp_slots} slot(s), "
              f"http://{a.host}:{a.port}  (drafter {os.path.basename(a.draft) if a.draft else 'none'}, "
              f"ctx {a.max_len})", flush=True)

    def get_job(block):
        try:
            job = JOBS.get(block=block)
        except _q.Empty:
            return None
        ids, max_new, t0 = job["ids"], job["max_new"], time.time()
        STATE["queue"] -= 1
        STATE["queued_prompt"] -= len(ids)
        STATE["queued_decode"] -= max_new
        STATE["running"] = STATE.get("running", 0) + 1
        STATE["busy"] = True

        def done(out, s):
            job["out"].put(None)
            STATE["running"] -= 1
            STATE["busy"] = STATE["running"] > 0
            if out is None:
                print(f"[tp-server] pp: request failed: {s}", flush=True)
                return
            dt = time.time() - t0
            STATE["served"] += 1
            STATE["tokens"] += len(out)
            print(f"[tp-server] pp: {len(ids)} prompt + {len(out)} tokens in {dt:.1f}s (prefill "
                  f"{s['t_prefill']:.1f}s, {s['steps']} steps" +
                  (f", accepted {s['accepted']}/{s['proposed']}" if drafter else "") + ")", flush=True)
        return {"ids": ids, "max_new": max_new, "on_tokens": job["out"].put, "done": done,
                "temperature": job["temperature"], "top_k": job["top_k"], "top_p": job["top_p"]}

    try:
        drv.serve(get_job)
    finally:
        drv.stop()
        sys.stdout.flush()
        os._exit(0)


def _fit_check(a, device, rank: int, world: int, weights):
    """Refuse BEFORE loading when this card cannot hold its share of the weights -- the same arithmetic as the layer
    pipeline (qv_band run/placement.py: pack bytes per layer from the manifest, each card's free memory less
    QV_PIPELINE_RESERVE_MB). Without it a 35B MoE on two 16 GB V100 loaded for 90 s and died in a CUDA OOM traceback; the pipeline refused the same model in 13 s with the numbers."""
    from qv_band.run.placement import card_capacities, weight_bytes
    from qv_band.run.tp_qwen35 import load_config
    try:
        cfg = load_config(a.model)
        n = int(getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", 0) or 0)
        per, other = weight_bytes(a.model, a.model, n) if n else ([], 0)
    except Exception as e:                                         # noqa: BLE001 -- unknown layout: no check, as before
        print(f"[tp-server] fit check skipped: {e!r}", flush=True)
        return
    share = (weights[rank] / sum(weights)) if weights else 1.0 / world
    need = (sum(per) + other) * share
    if a.draft and not a.draft_split:                              # the drafter whole on every card
        try:
            from qv_band.swap.block_stream import pack_bytes
            need += pack_bytes(a.draft)
        except Exception:                                          # noqa: BLE001 -- a dense drafter dir: not counted
            pass
    caps = card_capacities([device])
    if caps is not None and need > caps[0]:
        res = int(os.environ.get("QV_PIPELINE_RESERVE_MB", "2048") or 2048)
        print(f"[tp-server] rank {rank}: its share of the weights needs {need / 2**30:.1f} GiB, {device} has "
              f"{caps[0] / 2**30:.1f} GiB free for them (after the {res} MB reserve for KV and activations). Use more "
              f"cards, a smaller pack (higher tau), or a lower QV_PIPELINE_RESERVE_MB.", flush=True)
        raise SystemExit(2)


# ------------------------------------------------------------------------------------------------ every card
def main():
    global TOK, ARGS, MODEL_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--draft", default=None, help="DFlash drafter (dir or qvb8 pack)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--think", dest="think", action="store_true", default=True,
                    help="reason before answering unless a request says otherwise (default; as llm_server)")
    ap.add_argument("--no-think", dest="think", action="store_false", help="answer directly unless a request asks to think")
    ap.add_argument("--max-len", type=int, default=16384, help="context per request (static KV per card)")
    ap.add_argument("--tp-weights", default=None, help="share per card, e.g. 0.55,0.45 (default even)")
    ap.add_argument("--draft-split", action="store_true", help="split the drafter too (default: whole on each card)")
    ap.add_argument("--kv-window", type=int, default=None,
                    help="KV positions kept in VRAM per attention layer; older ones go to pinned host RAM (offload)")
    ap.add_argument("--batch", type=int, default=1,
                    help="B > 1: decode up to B conversations together (context --max-len each)")
    ap.add_argument("--prefix-grid", type=int, default=8192, help="DeltaNet state checkpoint every N prompt tokens")
    ap.add_argument("--prefix-max", type=int, default=8, help="checkpoints kept in pinned host RAM per card")
    ap.add_argument("--pp-slots", type=int, default=2, help="--pp: requests in flight at once (same on every machine)")
    ap.add_argument("--pp", action="store_true",
                    help="pipeline stages across machines (one stage per machine, TP inside it; for slow links). "
                         "The HTTP endpoint is on the LAST machine. Layers per stage: QV_TP_PP_LAYERS=20,44")
    ARGS = a = ap.parse_args()
    global DEFAULT_THINK
    DEFAULT_THINK = a.think

    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local}"
    torch.cuda.set_device(device)
    if a.pp:
        return _pp_main(a, device)
    import datetime
    from qv_band.formats.qvb8.tp import init_distributed
    from qv_band.run.tp_qwen35 import build
    from qv_band.run import tp_graph as G
    # one machine or several (QV_TP_XNODE=gloo: machines without matching NCCL / a fast fabric);
    # followers wait idle for requests: a year of timeout
    tp = init_distributed(device, timeout=datetime.timedelta(days=365))
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    MODEL_ID = os.path.basename(a.model.rstrip("/"))
    weights = [float(x) for x in a.tp_weights.split(",")] if a.tp_weights else None
    _fit_check(a, device, tp.rank, tp.world, weights)
    model, tcfg = build(a.model, tp, device, weights=weights, verbose=(tp.rank == 0))
    global PMS
    try:
        PMS = _pm_table(tcfg, tp.world, weights)
    except Exception as e:                                         # noqa: BLE001 -- a family without the table
        print(f"[tp-server] no PM table: {e!r}", flush=True)
    if a.batch > 1:                                               # several users per step
        if a.draft and os.environ.get("QV_BATCH_DRAFT", "0") != "1":
            print("[tp-server] --batch with --draft: drafter ignored (QV_BATCH_DRAFT=1 enables it)", flush=True)
            a.draft = None
        if a.draft:                                               # ... each with its own DFlash block, one verify
            from qv_band.decode.dflash_draft import build_dflash_drafter
            from qv_band.run.tp_batch_spec import SpecBatchDecoder, serve_follower as _sf, serve_rank0 as _sr
            d0 = build_dflash_drafter(model, a.draft, device=device)
            if a.draft_split:
                from qv_band.run.tp_dflash import split_drafter
                split_drafter(d0, tp, weights)
            drafters = [d0] + [d0.fork() for _ in range(a.batch - 1)]
            for d in drafters:
                d._graph = None                                   # each slot captures its own draft graph
            # the verify stays within ONE tensor-core pass (16 rows): T tokens per slot, T = 16 / slots (2 users x 8,
            # 4 x 4, 8 x 2); the drafter still drafts its whole block (bidirectional), the first T-1 are verified
            T = max(2, min(d0.block_size, 16 // a.batch))
            bd = SpecBatchDecoder(model, tcfg, device, a.batch, a.max_len, T, d0.capture_layers)
            serve_follower = lambda bd_, tp_, dev_: _sf(bd_, drafters, tp_, dev_)
            serve_rank0 = lambda bd_, jobs, eos_, tp_, dev_, st: _sr(bd_, drafters, jobs, eos_, tp_, dev_, st)
        else:
            from qv_band.run.tp_batch import BatchDecoder, serve_follower, serve_rank0
            bd = BatchDecoder(model, tcfg, device, a.batch, a.max_len)
        if tp.graphable:
            bd.capture()
        eos = {x for x in [TOK.eos_token_id, TOK.convert_tokens_to_ids("<|im_end|>"),
                           TOK.convert_tokens_to_ids("<|endoftext|>")] if isinstance(x, int) and x >= 0}
        if tp.world > 1:
            torch.distributed.barrier()
        if tp.rank != 0:
            serve_follower(bd, tp, device)
            torch.distributed.barrier()
            os._exit(0)
        from ai_engine.web.base_handler import QVHTTPServer
        srv = QVHTTPServer((a.host, a.port), _handler_cls())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[tp-server] {MODEL_ID} on {tp.world} card(s), http://{a.host}:{a.port}  "
              f"(batch {a.batch}{', DFlash per slot' if a.draft else ''}, ctx {a.max_len} per conversation)",
              flush=True)
        try:
            serve_rank0(bd, JOBS, eos, tp, device, STATE)
        finally:
            if tp.world > 1:
                h = torch.zeros(3, dtype=torch.long, device=device)
                torch.distributed.broadcast(h, 0, group=tp.group)
                torch.distributed.barrier()
            os._exit(0)
    drafter = None
    block = 8
    if a.draft:
        from qv_band.decode.dflash_draft import build_dflash_drafter
        drafter = build_dflash_drafter(model, a.draft, device=device)
        block = drafter.block_size
        if a.draft_split:
            from qv_band.run.tp_dflash import split_drafter
            split_drafter(drafter, tp, weights)
    dec = G.StaticDecoder(model, tcfg, device, a.max_len, t_max=block,
                          capture_layers=drafter.capture_layers if drafter else (),
                          kv_window=a.kv_window)
    if tp.graphable:
        dec.capture(sizes=(1, block))
    dec.prefix = G.PrefixCache(a.prefix_grid, a.prefix_max)       # this card's share of the saved prefixes (host RAM)
    dec.tp = tp
    eos = {x for x in [TOK.eos_token_id, TOK.convert_tokens_to_ids("<|im_end|>"),
                       TOK.convert_tokens_to_ids("<|endoftext|>")] if isinstance(x, int) and x >= 0}
    if tp.world > 1:
        torch.distributed.barrier()

    def run(ids, max_new, on_tokens, vis=None, mark=0, sp=(0.0, 0, 1.0)):
        dec.prefix.marks = {mark} if mark else set()              # one more checkpoint: the end of the last turn
        samp = dict(temperature=sp[0], top_k=int(sp[1]), top_p=sp[2])
        if os.environ.get("QV_TP_KPROF") == "1" and tp.rank == 0 and max_new >= 100 and not getattr(run, "_done", False):
            run._done = True                                      # QV_TP_KPROF=1: where the GPU time of the first long request goes
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                out = G.generate(dec, ids, max_new, drafter, tp, eos, on_tokens=on_tokens, prefill="hf", vis=vis,
                                 **samp)
                torch.cuda.synchronize()
            print("[tp-kprof] rank 0 GPU kernels:\n" + p.key_averages().table(sort_by="cuda_time_total", row_limit=14,
                  max_name_column_width=70), flush=True)
            return out
        return G.generate(dec, ids, max_new, drafter, tp, eos, on_tokens=on_tokens, prefill="hf", vis=vis, **samp)

    H = tcfg.hidden_size
    hdr = torch.zeros(4, dtype=torch.long, device=device)            # cmd, prompt tokens, max new, turn mark
    sph = torch.zeros(3, dtype=torch.float64, device=device)        # temperature, top_k, top_p
    if tp.rank != 0:                                              # follower cards: run what card 0 broadcasts
        while True:
            torch.distributed.broadcast(hdr, 0, group=tp.group)
            cmd, n, max_new, mark = hdr.tolist()
            if cmd == 0:
                break
            ids_t = torch.empty(n, dtype=torch.long, device=device)
            torch.distributed.broadcast(ids_t, 0, group=tp.group)
            torch.distributed.broadcast(sph, 0, group=tp.group)
            vis = None
            if cmd == 2:                                          # an image chat: card 0's embeddings + positions
                emb = torch.empty(n, H, dtype=torch.float16, device=device)
                pos = torch.empty(3, n, dtype=torch.long, device=device)
                dl = torch.empty(1, dtype=torch.long, device=device)
                for t_ in (emb, pos, dl):
                    torch.distributed.broadcast(t_, 0, group=tp.group)
                vis = (emb, pos, int(dl))
            try:
                run(ids_t.tolist(), max_new, None, vis, mark, tuple(sph.tolist()))
            except torch.OutOfMemoryError as e:
                print(f"[tp-server] card {tp.rank} out of memory during a request ({e}); stopping: the cards would "
                      f"run out of step", flush=True)
                os._exit(1)
            except Exception as e:                        # noqa: BLE001 -- the same request fails on card 0 too: serve on
                print(f"[tp-server] card {tp.rank}: request failed ({e!r}); serving on", flush=True)
        torch.distributed.barrier()
        os._exit(0)

    global VIS
    try:                                                          # images: the tower on card 0's host (CPU), see vision.py
        from qv_band.run.tp_qwen35 import load_config
        if getattr(load_config(a.model), "vision_config", None) is not None:
            import types
            os.environ.setdefault("LLM_VISION_TOWER", "cpu")
            from ai_engine.generation.vision import VisionAdapter
            VIS = VisionAdapter(types.SimpleNamespace(pack=a.model, gpu_dev=device, tok=TOK, model=None))
    except Exception as e:                                        # noqa: BLE001 -- text keeps working without it
        print(f"[tp-server] vision not available: {e!r}", flush=True)
    from ai_engine.web.base_handler import QVHTTPServer
    srv = QVHTTPServer((a.host, a.port), _handler_cls())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[tp-server] {MODEL_ID} on {tp.world} card(s), http://{a.host}:{a.port}  "
          f"(drafter {os.path.basename(a.draft) if a.draft else 'none'}, ctx {a.max_len}, "
          f"vision {'on' if VIS is not None else 'off'})", flush=True)
    try:
        while True:
            job = JOBS.get()
            vis = None
            if job.get("vision") is not None:                     # an image chat: prompt + embeddings on card 0
                try:
                    ids, emb, pos, dl = VIS.prepare(job["vision"], TOK, model.get_input_embeddings(), device,
                                                    tools=job.get("tool_list"), think=job["think"])
                    if len(ids) + 16 >= a.max_len or len(ids) > dec.W:
                        raise ValueError(f"image prompt of {len(ids)} tokens: this service holds {min(a.max_len, dec.W)}")
                except Exception as e:                            # noqa: BLE001
                    STATE["queue"] -= 1
                    STATE["queued_decode"] -= job.get("q_decode", 0)
                    job["error"] = f"vision: {e}"
                    job["out"].put(None)
                    continue
                job["ids"] = ids
                job["max_new"] = max(1, min(job["max_new"], a.max_len - len(ids) - 16))
                vis = (emb.to(torch.float16).contiguous(), pos.to(device).contiguous(), dl)
            ids, max_new = job["ids"], job["max_new"]
            STATE.update(busy=True, running_prompt=len(ids), running_decode=max_new)
            STATE["queue"] -= 1
            STATE["queued_prompt"] -= job.get("q_prompt", len(ids))   # what the handler counted (0 for an image chat)
            STATE["queued_decode"] -= job.get("q_decode", max_new)
            if tp.world > 1:
                hdr.copy_(torch.tensor([2 if vis else 1, len(ids), max_new, job.get("mark", 0)]))
                torch.distributed.broadcast(hdr, 0, group=tp.group)
                torch.distributed.broadcast(torch.tensor(ids, dtype=torch.long, device=device), 0, group=tp.group)
                sph.copy_(torch.tensor([job["temperature"], job["top_k"], job["top_p"]], dtype=torch.float64))
                torch.distributed.broadcast(sph, 0, group=tp.group)
                if vis:
                    for t_ in (vis[0], vis[1], torch.tensor([vis[2]], dtype=torch.long, device=device)):
                        torch.distributed.broadcast(t_, 0, group=tp.group)
            t0 = time.time()
            try:
                out, st = run(ids, max_new, job["out"].put, vis, job.get("mark", 0),
                              (job["temperature"], job["top_k"], job["top_p"]))
            except torch.OutOfMemoryError as e:                   # this request only; the next one runs normally
                if tp.world > 1:                                  # the other cards are out of step now (see above)
                    print(f"[tp-server] card 0 out of memory during a request ({len(ids)} prompt tokens, {e}); "
                          f"stopping: the cards would run out of step", flush=True)
                    os._exit(1)
                job["error"] = f"out of memory on the card for this request ({len(ids)} prompt tokens): {e}"[:300]
                out, st = [], {}
                torch.cuda.empty_cache()
                print(f"[tp-server] request failed, out of memory ({len(ids)} prompt tokens); serving on", flush=True)
            except Exception as e:                        # noqa: BLE001 -- a request the model refuses (same on every card)
                job["error"] = f"request failed: {e!r}"[:300]
                out, st = [], {}
                print(f"[tp-server] request failed ({e!r}); serving on", flush=True)
            finally:
                job["out"].put(None)
            dt = time.time() - t0
            STATE["served"] += 1
            STATE["tokens"] += len(out)
            STATE.update(busy=False, running_prompt=0, running_decode=0)
            print(f"[tp-server] {len(ids)} prompt + {len(out)} tokens in {dt:.1f}s "
                  f"(cached {st.get('cached', 0)}, prefill {len(ids) - st.get('cached', 0)} in {st.get('t_prefill', 0):.1f}s, "
                  f"host prefix {dec.prefix.nbytes() / 1e9:.2f} GB, decode {len(out) / max(1e-6, dt - st.get('t_prefill', 0)):.1f} tok/s"
                  + (f", accepted {st['accepted']}/{st['proposed']}" if drafter else "") + ")", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if tp.world > 1:
            hdr.copy_(torch.tensor([0, 0, 0, 0]))
            torch.distributed.broadcast(hdr, 0, group=tp.group)
            torch.distributed.barrier()
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
