"""Run your own text through the model with weights served from QV-band packs.

A thin CLI over ai_engine.generation.gen_core: this file owns argv parsing, the
stdout stream, and the interactive prompt loop. Everything about HOW
inference runs -- device/TP resolution, the memory ceiling, model load,
every drafter, every KV-cache mode, the speculative_generate call -- lives
in gen_core and is shared byte-for-byte with llm_server.py and any future
interface.

    python -m ai_engine.cli.chat /mnt/data/models/Qwen3.8-27B_qvb8_fp16_1e-3 "The capital of France is"
    python -m ai_engine.cli.chat /mnt/data/models/Qwen3.8-27B_qvb8_tau1e-3 "Explain entropy simply"
    python -m ai_engine.cli.chat /mnt/data/models/Qwen3.8-27B_qvb8_tau1e-3 -f prompt.txt --max-output-tokens 60
    python -m ai_engine.cli.chat /mnt/data/models/Qwen3.8-27B_qvb8_tau1e-3          # interactive chat (the default)

First argument is a self-contained pack folder (config, tokenizer and
weights all live there) -- just a model path, nothing else needed. The
folder's manifest names the format and the format registry
(qv_band.common.formats) supplies the readers.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ai_engine.runtime.sampling import resolve_sampling
from ai_engine.runtime.memlimit import rss_gb
from ai_engine.generation.gen_core import (
    DEFAULTS, add_generation_args, params_from_args,
    encode_prompt, run_generate)
from ai_engine.generation.loader import load_qvb8_llm


def main():
    # Tensor-parallel launch (torchrun, WORLD_SIZE>1): every rank runs this
    # whole script and all-gathers to the same output, so without a rank
    # gate every print() fires once per rank and their stdout interleaves
    # character-by-character. Silence every rank but 0 by swallowing its
    # stdout bytes (NOT monkeypatching builtins.print -- numba introspects
    # `print` identity and a replacement breaks its import).
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and int(os.environ.get("RANK", "0")) != 0:
        sys.stdout = open(os.devnull, "w")

    ap = argparse.ArgumentParser(
        prog="ai_engine.cli.chat",
        description="Run text through a model served from QV-band packs "
                    "(.qvband). "
                    "Zero-flag defaults: chat template on, thinking on, "
                    "interactive on, device auto, 48 GB memory ceiling, "
                    f"model's own context window, --kv-cache "
                    f"{DEFAULTS.KV_CACHE}.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("model", help="self-contained pack folder to load")
    ap.add_argument("prompt", nargs="*", help="your input text")
    ap.add_argument("-f", "--file", metavar="PATH",
                    help="read the prompt from a file instead of argv")

    mode = ap.add_argument_group("mode")
    mode.add_argument("--no-chat", dest="chat", action="store_false",
                    default=True,
                    help="raw continuation instead of the instruct "
                         "template -- the model just continues your text")
    mode.add_argument("--no-think", dest="think", action="store_false",
                    default=True,
                    help="skip the reasoning step and answer directly")
    mode.add_argument("--no-interactive", dest="interactive",
                    action="store_false", default=True,
                    help="run the given prompt once and exit, instead of "
                         "dropping into a prompt loop afterwards")

    # device / memory / ctx-size / KV mode / drafter / draft-k / draft-ctx
    # -- all shared with llm_server.py, defined once in gen_core.
    perf = ap.add_argument_group("performance")
    add_generation_args(perf)

    sampling = ap.add_argument_group("sampling")
    sampling.add_argument("--greedy", action="store_true",
                    help="deterministic decoding (equivalent to "
                         "--temperature 0); drafts verified by exact match, "
                         "so output is byte-identical to plain greedy")
    sampling.add_argument("--temperature", type=float, default=None,
                    metavar="T",
                    help="default 0.6 thinking / 0.7 non-thinking "
                         "(Qwen's recommendation); 0 == --greedy")
    sampling.add_argument("--top-k", type=int, default=None, metavar="K",
                    help="restrict sampling to the top-K logits (temperature "
                         "> 0 only; default 20)")
    sampling.add_argument("--top-p", type=float, default=None, metavar="P",
                    help="nucleus threshold (temperature > 0 only; default "
                         "0.95 thinking / 0.8 non-thinking)")

    # --kv_cache and --kv-cache are the same flag: underscores in an option NAME become dashes
    # (only the name -- a value such as a path keeps its underscores, also in --opt=value)
    argv = [(a.split("=", 1)[0].replace("_", "-") + a[len(a.split("=", 1)[0]):]) if a.startswith("--") else a
            for a in sys.argv[1:]]
    args = ap.parse_args(argv)

    args.temperature, args.top_k, args.top_p = resolve_sampling(
        args.think, args.greedy, args.temperature, args.top_k, args.top_p)

    # One GenParams -- the single generation configuration. max_output_tokens
    # defaults (via gen_core) to a modest OUTPUT budget, NOT --ctx-size (that
    # binding once decoded 200k tokens on a runaway generation, 70+ GB).
    params = params_from_args(args)

    if args.file:
        text = open(args.file, encoding="utf-8").read().strip()
    else:
        text = " ".join(args.prompt).strip()

    print(f"pack   : {args.model}")
    # apply_kv_shadow -> load_model -> build_drafter, in that order, with no
    # room for a caller to drop a step. See ai_engine.generation.loader for why
    # the shadow has to be resolved before qv_band is imported. chat.py fails
    # loudly on a bad drafter rather than degrading, so fallback stays False.
    llm = load_qvb8_llm(args.model, params, verbose=True)
    bundle = llm.bundle
    if bundle.use_gpu:
        _idx = int(bundle.gpu_dev.split(":")[1]) if ":" in bundle.gpu_dev else 0
        name = (torch.cuda.get_device_name(_idx)
                if bundle.gpu_dev.startswith("cuda")
                else torch.xpu.get_device_name(0))
        print(f"device : {bundle.gpu_dev} ({name})")
    else:
        print("device : cpu")
    print(f"kv     : {params.describe_kv()}")
    print(f"ready  : {bundle.load_seconds:.1f}s, RSS {rss_gb():.1f} GB\n")

    drafter = llm.drafter

    # Single-conversation prefix cache (see qv_band.decode.spec_decode.PrefixCache):
    # the interactive loop re-encodes the WHOLE transcript every turn (the
    # chat template is re-applied fresh each time -- there is no cheaper way
    # to get the exact prompt string), but if this turn's tokens start with
    # exactly what the previous turn's cache already consumed -- true for
    # every turn that only appends, i.e. every turn 'reset' was not used --
    # run_generate() skips re-prefilling that shared prefix. 'reset' drops
    # it explicitly since the next prompt no longer shares it.
    _prefix_cache = [None]

    def run(prompt: str, history: list | None = None):
        ids = encode_prompt(bundle, prompt, chat=args.chat, think=args.think,
                            history=history)
        n_prompt = ids.shape[1]
        ctx = params.require_ctx_size()
        if n_prompt > ctx:
            print(f"ERROR: prompt has {n_prompt} tokens, exceeds the "
                  f"configured context size of {ctx} (--ctx-size). Send a "
                  f"shorter prompt, restart with a larger --ctx-size, or "
                  f"'reset' to drop history.")
            return None
        print(f"> {prompt}")
        print("-" * 60)
        t = time.time()

        kv_stats: dict = {}
        # Invalidate up front: if run_generate raises, the cache object it
        # was mutating in place is left past what the OLD handle's
        # token_ids can account for, so that handle must never be reused.
        # A clean return overwrites this with the fresh handle right after.
        _cache_in, _prefix_cache[0] = _prefix_cache[0], None
        out, st, _prefix_cache[0] = run_generate(
            bundle, drafter, ids, params, stream=True, kv_stats=kv_stats,
            prefix_cache=_cache_in, return_cache=True)
        n = len(out)
        dt = time.time() - t
        print(f"\n{'-'*60}")
        line = f"{n} tokens in {dt:.1f}s = {n/max(dt,1e-9):.3f} tok/s   RSS {rss_gb():.1f} GB"
        # ratio_vs_dense is emitted only by BandedKVCache.memory_report().
        if kv_stats.get("ratio_vs_dense") is not None:
            line += (f"\n  QVB8-KV RESIDENT: {kv_stats['total_bytes']/2**20:.1f} MB "
                     f"({kv_stats['banded_bytes']/2**20:.1f} banded + "
                     f"{kv_stats['residual_bytes']/2**20:.1f} dense residual) vs "
                     f"{kv_stats['dense_bytes']/2**20:.1f} MB dense "
                     f"({100*kv_stats['ratio_vs_dense']:.1f}%, "
                     f"{kv_stats['bytes_per_elem']:.4f} B/elem, "
                     f"{kv_stats['layers']} full-attn layers, REAL saving)")
        elif kv_stats.get("dense_fp16_bytes"):
            line += (f"\n  INT8 KV: {kv_stats['total_bytes']/2**20:.1f} MB vs "
                     f"{kv_stats['dense_fp16_bytes']/2**20:.1f} MB dense fp16 "
                     f"({100*kv_stats['ratio_vs_fp16']:.1f}%, "
                     f"{kv_stats['bytes_per_elem']:.4f} B/elem, "
                     f"{kv_stats['layers']} full-attn layers)")
        if params.draft_k > 0 and st.proposed:
            line += (f"\n  drafted {st.proposed}, accepted {st.accepted} "
                     f"({st.acceptance:.0%}), {st.emitted/max(1,st.steps):.2f} tok/step")
        print(line + "\n")
        return bundle.tok.decode(out, skip_special_tokens=not args.think)

    if text:
        run(text)
    if args.interactive:
        # Tensor-parallel launch: run()'s generate ends in a dist.all_gather
        # inside every sharded linear, which every rank must reach in
        # lockstep. Only rank 0 has a real terminal (input() on rank 1+
        # blocks forever on detached stdin), so broadcast rank 0's line (or
        # the sentinels) to every rank to keep them walking the SAME loop
        # iterations together.
        _tp = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if _tp:
            import torch.distributed as dist
        _is_rank0 = not _tp or int(os.environ.get("RANK", "0")) == 0
        if _is_rank0:
            print("interactive mode -- blank line or Ctrl-C to quit, "
                  "'reset' to clear history")
        history: list = []
        while True:
            if _is_rank0:
                try:
                    line = input("\nprompt> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    line = "\0QUIT"
            else:
                line = None
            if _tp:
                box = [line]
                dist.broadcast_object_list(box, src=0)
                line = box[0]
            if line == "\0QUIT" or not line:
                break
            if line == "reset":
                history = []
                _prefix_cache[0] = None   # next turn no longer shares a prefix
                if _is_rank0:
                    print("(history cleared)")
                continue
            reply = run(line, history)
            if reply is None:
                continue
            # Qwen's template convention: only the live turn shows its
            # <think> block -- prior turns feed back in without one.
            spoken = reply.rsplit("</think>", 1)[-1].strip() if args.think else reply
            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": spoken})
    return 0


if __name__ == "__main__":
    sys.exit(main())
