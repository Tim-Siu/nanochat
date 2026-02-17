#!/bin/bash

# Parameterised MoE d12 run for layer-placement sweeps.
# Usage:
#   MOE_LAYERS="2:4" WANDB_RUN=d12_moe_b3_4 MODEL_TAG=d12_moe_b3_4 bash dev-hetero/run_moe_d12_sweep.sh

set -euo pipefail

: "${MOE_LAYERS:?MOE_LAYERS is not set (e.g. \"2:4\")}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Python venv setup
source .venv/bin/activate

# wandb / model tag setup
WANDB_RUN="${WANDB_RUN:-moe_d12_sweep}"
MODEL_TAG="${MODEL_TAG:-moe_d12_sweep}"

# Enough shards for quick d12 experimentation
python -m nanochat.dataset -n 64

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m dev-hetero.train_moe -- \
    --depth=12 \
    --moe-layers="$MOE_LAYERS" \
    --n-routed-experts=7 \
    --moe-top-k=3 \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
