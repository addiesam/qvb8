#!/usr/bin/env bash
# One-step install for QVB8 on Linux with an NVIDIA GPU.      ./install.sh
# It creates ./.venv, installs PyTorch (the build that matches your NVIDIA driver) and the other packages, and checks
# that the GPU works. Nothing outside this folder is changed. Afterwards:  ./qvb8 get qwen-9b   then   ./qvb8 start
# Options (environment): TORCH_INDEX=<url> to choose the PyTorch download index yourself, PYTHON=python3.12, DRY=1 to
# only print what would be done.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
run() { if [ -n "${DRY:-}" ]; then echo "[dry run] $*"; else "$@"; fi; }

[ "$(uname -s)" = Linux ] || die "QVB8 runs on Linux. (Windows: use WSL2 with an NVIDIA driver.)"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || die "python3 not found. Install it first (Ubuntu: sudo apt install python3 python3-venv)."
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10 or newer is needed (found $("$PY" --version 2>&1))."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver first (https://www.nvidia.com/drivers)."

say "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
CUDA_VER=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)
[ -n "$CUDA_VER" ] || die "could not read the CUDA version from nvidia-smi. Set TORCH_INDEX=https://download.pytorch.org/whl/cu130 (or cu126) and run again."
CUDA_MAJOR=${CUDA_VER%%.*}; CUDA_MINOR=${CUDA_VER##*.}

if [ -z "${TORCH_INDEX:-}" ]; then
  if   [ "$CUDA_MAJOR" -ge 13 ]; then TORCH_INDEX=https://download.pytorch.org/whl/cu130
  elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then TORCH_INDEX=https://download.pytorch.org/whl/cu126
  else die "your NVIDIA driver supports CUDA $CUDA_VER; PyTorch 2.14 needs CUDA 12.6 or newer. Update the driver."
  fi
fi
echo "PyTorch index: $TORCH_INDEX   (driver CUDA $CUDA_VER)"

say "Python environment (.venv)"
[ -d .venv ] || run "$PY" -m venv .venv || die "could not create .venv (Ubuntu: sudo apt install python3-venv)."
if [ -n "${DRY:-}" ]; then PIP="pip"; else PIP=".venv/bin/python -m pip"; fi
run $PIP install --upgrade pip

say "PyTorch (large download, a few GB)"
run $PIP install "torch>=2.14" --index-url "$TORCH_INDEX"

say "Other packages"
# torch brings the triton build it needs: the pinned triton line of requirements.txt is left out here
grep -vE '^(torch|triton)([=<>~ ]|$)' requirements.txt > .requirements.install.txt
run $PIP install -r .requirements.install.txt huggingface_hub
rm -f .requirements.install.txt

if [ -z "${DRY:-}" ]; then
  say "Check"
  .venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}  GPU: {p.name}  {p.total_memory/2**30:.0f} GB  compute capability {p.major}.{p.minor}")
PY
fi
cat <<'MSG'

Done. Next:
  ./qvb8 list                 the models you can run
  ./qvb8 get qwen-9b          download a model (7 GB)
  ./qvb8 start qwen-9b        start the server (first start takes a few minutes while the GPU code is prepared)
  ./qvb8 chat                 talk to it
MSG
