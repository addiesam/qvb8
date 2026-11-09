#!/usr/bin/env bash
# Ripple-65B OpenAI-compatible server: qvb8 or qvb8r target pack + DFlash2 drafter pack.
#   MODEL=/path/Ripple-65B-Preview_qvb8_1e-3 DRAFT=/path/Ripple-65B-Preview-DFlash2_qvb8_5e-3 \
#     scripts/ripple_server.sh [start|stop|status|restart|log] [extra serve flags]
#   port 8080. Log: scripts/logs/Ripple-65B.log.  Extra flags come last, so they win (e.g. --no-think).
# Env: MODEL, DRAFT (required for start), PORT, PYTHON (default: python)

NAME=Ripple-65B; DEF_PORT=8080
launch() {
  : "${MODEL:?set MODEL to the target pack dir (qvb8 or qvb8r)}"
  : "${DRAFT:?set DRAFT to the DFlash2 drafter pack dir}"
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec "${PYTHON:-python}" -m ai_engine.cli.serve_ripple "$MODEL" \
    --host 0.0.0.0 --port "$PORT" --device auto --max-memory-gb 100 --max-output-tokens 32767 \
    --kv-cache qvb8 --kv-tau-threshold 0.1 --kv-residual 256 --kv-shadow 512MB \
    --prefill-chunk 8192 --prefill-linear qvb8 \
    --drafter dflash --draft-model "$DRAFT" --draft-pack dense --draft 15 --draft-ctx 4 \
    --vision off --think "${EXTRA[@]}"
}
. "$(dirname "${BASH_SOURCE[0]}")/lib/serve.sh"
serve_main "$@"
