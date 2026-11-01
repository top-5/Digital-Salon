#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}
export MAX_JOBS=${MAX_JOBS:-4}
python3.12 -m venv "$root/.venv"
"$root/.venv/bin/pip" install --upgrade pip setuptools==70.2.0 wheel ninja
"$root/.venv/bin/pip" install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
"$root/.venv/bin/pip" install --no-build-isolation -r "$root/requirements.txt"
