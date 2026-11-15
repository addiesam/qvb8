#!/usr/bin/env bash
# Qwen OpenAI-compatible server (Qwen3.5 / 3.6 / 3.8, dense and MoE: 9B, 27B, 35B-A3B): qvb8 or qvb8r target pack +
# a DFlash / DFlash2 / DSpine drafter (a pack of it, or the original checkpoint).
#   MODEL=/path/Qwen3.8-27B_qvb8r_2e-3_fp16 DRAFT=/path/Qwen3.8-27B-DFlash2_qvb8r_5e-3_fp16 DRAFT_K=7 \
#     scripts/qwen_server.sh [start|stop|status|restart|log|startup] [extra serve flags]
#   port 8080. Log: scripts/logs/Qwen.log, startup log: scripts/logs/Qwen.startup.log.
#   Extra flags come last, so they win (e.g. --no-think, --dtype bf16, --vision off, --prefill-chunk 12288).
#   More than one card: add --pipeline cuda:0,cuda:1 (the cards, in layer order; the last is the main card).
# DRAFT_K = tokens drafted per step = the drafter's block size - 1: DFlash2 (27B, 35B-A3B) 7, DFlash (9B) 15,
#   DSpine (27B) 15. It is not guessed from the drafter: set it.
# ATTN = attention of the full-attention layers: flash_attention_2 (needs flash-attn; Ampere and newer) or sdpa (any
#   card; a V100 has no flash-attn).
# --prefill-chunk 4096: a repeated prompt or the next chat turn prefills at most 4096 tokens again (the saved state at
#   a prompt's end sits on a chunk boundary); 12288 prefills a new long document 13-18 % faster.
# Admission: at most MAX_LIVE requests in the live batch and MAX_LIVE_TOKENS live tokens (prompt + max output of the
#   live requests: the bound on their KV cache), MAX_QUEUE waiting (one more: HTTP 429 + Retry-After), QUEUE_WAIT
#   seconds of waiting (then 503). First in, first out. 0 = no limit. The defaults suit a 128 GB machine and a 27B;
#   a 16 GB card running a 27B needs MAX_LIVE=1.
# Env: MODEL, DRAFT, DRAFT_K (required for start), ATTN, MAX_MEM (GB, default 80), MAX_LIVE (8), MAX_LIVE_TOKENS
#      (524288), MAX_QUEUE (64), QUEUE_WAIT (600), PORT, PYTHON (default: python)

NAME=Qwen; DEF_PORT=8080
launch() {
  : "${MODEL:?set MODEL to the target pack dir (qvb8 or qvb8r)}"
  : "${DRAFT:?set DRAFT to the drafter dir (a DFlash / DFlash2 / DSpine pack or checkpoint)}"
  : "${DRAFT_K:?set DRAFT_K to the tokens drafted per step (DFlash2: 7; DFlash 9B, DSpine: 15)}"
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export QV_ATTN_IMPL="${ATTN:-flash_attention_2}"
  exec "${PYTHON:-python}" -m ai_engine.cli.serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" --device auto --max-memory-gb "${MAX_MEM:-80}" \
    --kv-cache qvb8 --kv-tau-threshold 0.2 --kv-residual 256 --kv-shadow 512MB --prefix-state qvb8 \
    --prefill-chunk 4096 --prefill-linear qvb8 \
    --drafter dflash --draft-model "$DRAFT" --draft-pack dense --draft "$DRAFT_K" \
    --max-live "${MAX_LIVE:-8}" --max-live-tokens "${MAX_LIVE_TOKENS:-524288}" --max-queue "${MAX_QUEUE:-64}" \
    --queue-wait "${QUEUE_WAIT:-600}" \
    --vision on --dtype fp16 "${EXTRA[@]}"
}
. "$(dirname "${BASH_SOURCE[0]}")/lib/serve.sh"
serve_main "$@"
