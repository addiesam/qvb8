#!/usr/bin/env bash
# Terminal chat with Ripple-65B (same engine as the server, runs in the foreground).
#   MODEL=... DRAFT=... scripts/ripple_chat.sh                                # interactive
#   MODEL=... DRAFT=... scripts/ripple_chat.sh "Explain entropy" --no-interactive
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
: "${MODEL:?set MODEL to the target pack dir (qvb8 or qvb8r)}"
: "${DRAFT:?set DRAFT to the DFlash2 drafter pack dir}"
export RIPPLE_PACK="$MODEL" RIPPLE_DRAFT="$DRAFT"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the compute dtype: --dtype bf16 unless the arguments choose one (--dtype fp16 on a GPU without bf16, e.g. V100);
# added after them, so the prompt stays the first positional argument
DT=(--dtype bf16)
for a in "$@"; do case "$a" in --dtype|--dtype=*) DT=() ;; esac; done
exec "${PYTHON:-python}" -m ai_engine.cli.chat_ripple "$MODEL" "$@" ${DT[@]+"${DT[@]}"}
