#!/bin/bash

# Quick dense d12 run (README recipe style).
# Usage:
#   bash dev-hetero/run_dense_d12.sh
#   WANDB_RUN=d12_dense MODEL_TAG=d12_dense bash dev-hetero/run_dense_d12.sh

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Python venv setup
source .venv/bin/activate

# wandb setup
WANDB_RUN="${WANDB_RUN:-d12}"
MODEL_TAG="${MODEL_TAG:-d12}"

# Enough shards for quick d12 experimentation
python -m nanochat.dataset -n 64

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m scripts.base_train -- \
    --depth=12 \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
