#!/bin/bash

# Parameterised hetero (MLP-sharing) d12 run for sharing-placement sweeps.
# Usage:
#   SHARED_MLP_GROUPS="1:4" WANDB_RUN=d12_hetero_b2_3_4 MODEL_TAG=d12_hetero_b2_3_4 bash dev-hetero/run_hetero_d12_sweep.sh

set -euo pipefail

: "${SHARED_MLP_GROUPS:?SHARED_MLP_GROUPS is not set (e.g. \"1:4\")}"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Python venv setup
source .venv/bin/activate

# wandb / model tag setup
WANDB_RUN="${WANDB_RUN:-hetero_d12_sweep}"
MODEL_TAG="${MODEL_TAG:-hetero_d12_sweep}"

# Enough shards for quick d12 experimentation
python -m nanochat.dataset -n 64

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m dev-hetero.train_hetero -- \
    --depth=12 \
    --moe-layers="" \
    --shared-mlp-groups="$SHARED_MLP_GROUPS" \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
