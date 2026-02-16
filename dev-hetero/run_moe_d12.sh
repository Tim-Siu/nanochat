#!/bin/bash

# Quick MoE d12 run aligned with the dense d12 recipe.
# Usage:
#   bash dev-hetero/run_moe_d12.sh
#   WANDB_RUN=moe_d12 MODEL_TAG=moe_d12 bash dev-hetero/run_moe_d12.sh

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Python venv setup
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# wandb setup
WANDB_RUN="${WANDB_RUN:-moe_d12}"
MODEL_TAG="${MODEL_TAG:-moe_d12}"

# Enough shards for quick d12 experimentation
python -m nanochat.dataset -n 64

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m dev-hetero.train_moe -- \
    --depth=12 \
    --moe-layers="2:" \
    --n-routed-experts=7 \
    --moe-top-k=3 \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
