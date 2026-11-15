#!/usr/bin/env bash
# Nemotron-H OpenAI-compatible server (Mamba-2 / attention / latent-MoE hybrid, e.g. Nemotron-3.5-Bolt-74B-A9B): qvb8
# target pack + NVIDIA's LiLiCorr drafter checkpoint.
#   MODEL=/path/NVIDIA-Nemotron-3.5-Bolt-74B-A9B_qvb8_1e-3_fp16 DRAFT=/path/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr \
#     scripts/nemotron_server.sh [start|stop|status|restart|log|startup] [extra serve flags]
#   port 8080. Log: scripts/logs/Nemotron.log, startup log: scripts/logs/Nemotron.startup.log.
#   Extra flags come last, so they win.
# NOT RE-TESTED in this release build: no Nemotron pack was on the build machine (2026-11-14). The code is the engine's
#   Nemotron path as it ran in October 2026 (fp16-restore pack: ~31-34 tok/s with the drafter on a DGX Spark).
# The model's own decode loop is used. Only 8 layers attend, so the KV cache stays dense bf16 (--kv-cache bf16). Use the
#   fp16-restore pack: a bf16-restore pack has escape entries in every expert, which the fused expert kernel refuses.
# Image input (--vision on) needs the pack's C-RADIO tower; it is off here.
# Env: MODEL, DRAFT (required for start), MAX_MEM (GB, default 100), PORT, PYTHON (default: python)

NAME=Nemotron; DEF_PORT=8080
launch() {
  : "${MODEL:?set MODEL to the target pack dir}"
  : "${DRAFT:?set DRAFT to the LiLiCorr drafter checkpoint dir}"
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec "${PYTHON:-python}" -m ai_engine.cli.serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" --device auto --max-memory-gb "${MAX_MEM:-100}" \
    --kv-cache bf16 --prefill-chunk 8192 --prefill-linear qvb8 \
    --drafter dflash --draft-model "$DRAFT" --draft-pack dense --draft 7 \
    --vision off --dtype fp16 "${EXTRA[@]}"
}
. "$(dirname "${BASH_SOURCE[0]}")/lib/serve.sh"
serve_main "$@"
