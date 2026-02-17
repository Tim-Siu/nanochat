#!/bin/bash

# Combined MoE + Parameter Sharing d12 run.
# Usage:
#   MOE_LAYERS="9:11" SHARED_MLP_GROUPS="1:4" \
#   WANDB_RUN=d12_combined_moe9-10_shared1-3 \
#   MODEL_TAG=d12_combined_moe9-10_shared1-3 \
#     bash dev-hetero/run_combined_d12.sh

set -euo pipefail

: "${MOE_LAYERS:?MOE_LAYERS is not set (e.g. \"9:11\")}"
: "${SHARED_MLP_GROUPS:?SHARED_MLP_GROUPS is not set (e.g. \"1:4\")}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Python venv setup
source .venv/bin/activate

# wandb / model tag setup
WANDB_RUN="${WANDB_RUN:-combined_d12}"
MODEL_TAG="${MODEL_TAG:-combined_d12}"

# Enough shards for quick d12 experimentation
python -m nanochat.dataset -n 64

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m dev-hetero.train_hetero -- \
    --depth=12 \
    --moe-layers="$MOE_LAYERS" \
    --shared-mlp-groups="$SHARED_MLP_GROUPS" \
    --n-routed-experts=7 \
    --moe-top-k=3 \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
