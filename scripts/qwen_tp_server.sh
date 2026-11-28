#!/usr/bin/env bash
# Qwen-family OpenAI-compatible server by TENSOR PARALLEL: every layer split over the cards of this machine, one process per card
# (torchrun, NCCL). One message is computed by all cards together: the single-message speed path.
#   MODEL=/path/Qwen3.8-27B_qvb8_1e-3_fp16 DRAFT=/path/Qwen3.8-27B-DFlash2_qvb8_2e-3_fp16 scripts/qwen_tp_server.sh [start|stop|status|restart|log|startup] [extra flags]
#   port 8080. Log: scripts/logs/QwenTP.log, startup log: scripts/logs/QwenTP.startup.log.
# Default: the single-message speed path (one conversation, DFlash drafter). Several users at once: USERS=N decodes up to N
#   conversations together (batch N, no drafter, CTX default: the model's context; measured 2 x V100, 27B: 1 request 11.5 tok/s,
#   4 at once 38.6 tok/s in total). Or start more services and put the router in front (data parallel).
# A qvb8 pack is cut by plain slices; a qvb8r pack is cut in place from its coded stream.
# Env: MODEL, DRAFT (optional), CARDS (default 2: the number of processes), CTX (default: the model's context), KV_WINDOW (default 8192), USERS,
#      TP_WEIGHTS (e.g. 1,1: the share of each card), PORT, PYTHON (default: python), TORCHRUN (default: torchrun)
#      QV_TP_SHARD_WRITE=DIR saves the cut shares once, QV_TP_SHARD_DIR=DIR loads them (the load is much faster).
NAME=QwenTP; DEF_PORT=8080
launch() {
  : "${MODEL:?set MODEL to the target pack dir (qvb8 or qvb8r)}"
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
  local a=()
  if [ -n "${USERS:-}" ]; then a=(--batch "$USERS")
  else export QV_STATE_TAU="${QV_STATE_TAU:-1e-5}" QV_DRAFT_GRAPH=1
       a=(--kv-window "${KV_WINDOW:-8192}")
       [ -z "${DRAFT:-}" ] || a+=(--draft "$DRAFT" --draft-split); fi
  [ -z "${TP_WEIGHTS:-}" ] || a+=(--tp-weights "$TP_WEIGHTS")
  [ -z "${CTX:-}" ] || a+=(--max-len "$CTX")
  exec "${TORCHRUN:-torchrun}" --nproc-per-node "${CARDS:-2}" -m ai_engine.servers.tp_server "$MODEL" \
    --host 0.0.0.0 --port "$PORT" "${a[@]}" "${EXTRA[@]}"
}
. "$(dirname "${BASH_SOURCE[0]}")/lib/serve.sh"
serve_main "$@"
