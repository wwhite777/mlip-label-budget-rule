#!/usr/bin/env bash
# Sequential, resumable G004 FT grid. Physical GPU 0 is always forbidden.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 1 || ! $1 =~ ^[1-9][0-9]*$ ]]; then
    echo 'Usage: confirm_ft_launcher.sh <physical-gpu>=1 [--dry-run] [--out-root DIR]; GPU 0 forbidden' >&2
    exit 2
fi
GPU=$1
shift
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd -- "$HERE"
# The launcher parent never imports torch or creates a CUDA context.
exec "${PYTHON:-python}" -B -u -m confirm_runner.ft_launcher --gpu "$GPU" "$@"
