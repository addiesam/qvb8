#!/usr/bin/env bash
# Inkling-2-Mini OpenAI-compatible server: qvb8 pack (linears) + qvbq experts (NVFP4 / MXFP8, bit-exact), with the
# checkpoint's own DSpark drafter (bundled in the pack: no separate drafter).
#   MODEL=/path/Inkling-2-Mini_qvb8_1e-3 scripts/inkling_server.sh [start|stop|status|restart|log|startup] [extra flags]
#   port 8080. Log: scripts/logs/Inkling.log, startup log: scripts/logs/Inkling.startup.log.
#   Extra flags come last, so they win (e.g. --dtype bf16 on Ampere and newer, --drafter n-gram --draft 0 for no drafter).
# Cards: one card by default. Several cards in one process: QV_TP_DEVICES=cuda:0,cuda:1 (heads split, experts split).
#   A card too small for the experts: QV_EXPERTS=host (the experts in host RAM, QV_EXPERT_CACHE_GB of them on the card).
# EFFORT: reasoning effort (none / minimal / low / medium / high / xhigh, or 0..0.99), default low.
# Env: MODEL (required for start), EFFORT, MAX_MEM (GB, default 64), CTX (default: the model's context), MAX_LIVE (4), MAX_QUEUE (32),
#      QUEUE_WAIT (600), PORT, PYTHON (default: python)

NAME=Inkling; DEF_PORT=8080
launch() {
  : "${MODEL:?set MODEL to the Inkling-2-Mini pack dir}"
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
  export QV_INKLING_EFFORT="${EFFORT-low}"
  exec "${PYTHON:-python}" -m ai_engine.cli.serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" --device cuda --max-memory-gb "${MAX_MEM:-64}" ${CTX:+--ctx-size "$CTX"} \
    --kv-cache qvb8 --kv-shadow 0 --prefill-chunk 2048 --drafter dspark --draft 7 --vision on --think \
    --max-live "${MAX_LIVE:-4}" --max-queue "${MAX_QUEUE:-32}" --queue-wait "${QUEUE_WAIT:-600}" \
    --dtype fp16 "${EXTRA[@]}"
}
. "$(dirname "${BASH_SOURCE[0]}")/lib/serve.sh"
serve_main "$@"
