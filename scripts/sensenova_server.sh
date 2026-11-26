#!/usr/bin/env bash
# SenseNova-Loop-9B OpenAI-compatible server: qvb8 target pack + its DFlash2 drafter (a pack of it, or the checkpoint).
#   MODEL=/path/SenseNova-Loop-9B_qvb8_5e-4 DRAFT=/path/SenseNova-Loop-9B-DFlash2_qvb8_5e-4 \
#     scripts/sensenova_server.sh [start|stop|status|restart|log|startup] [extra serve flags]
#   port 8080. Log: scripts/logs/SenseNova.log, startup log: scripts/logs/SenseNova.startup.log.
#   Extra flags come last, so they win (e.g. --dtype bf16 on Ampere and newer, --vision off, --no-think).
# ATTN: flash_attention_2 (needs flash-attn; Ampere and newer) or sdpa (any card).
# Env: MODEL, DRAFT (required for start), ATTN, MAX_MEM (GB, default 24), MAX_LIVE (8), MAX_LIVE_TOKENS (262144),
#      MAX_QUEUE (64), QUEUE_WAIT (600), PORT, PYTHON (default: python)

NAME=SenseNova; DEF_PORT=8080
launch() {
  : "${MODEL:?set MODEL to the SenseNova-Loop-9B qvb8 pack dir}"
  : "${DRAFT:?set DRAFT to the SenseNova-Loop-9B DFlash2 drafter dir}"
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export QV_ATTN_IMPL="${ATTN:-flash_attention_2}"
  exec "${PYTHON:-python}" -m ai_engine.cli.serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" --device auto --max-memory-gb "${MAX_MEM:-24}" \
    --kv-cache qvb8 --kv-tau-threshold 0.2 --kv-residual 256 --kv-shadow 512MB --prefix-state qvb8 \
    --prefill-chunk 8192 --prefill-linear qvb8 \
    --drafter dflash --draft 7 --draft-model "$DRAFT" \
    --max-live "${MAX_LIVE:-8}" --max-live-tokens "${MAX_LIVE_TOKENS:-262144}" --max-queue "${MAX_QUEUE:-64}" \
    --queue-wait "${QUEUE_WAIT:-600}" \
    --vision on --dtype fp16 "${EXTRA[@]}"
}
. "$(dirname "${BASH_SOURCE[0]}")/lib/serve.sh"
serve_main "$@"
