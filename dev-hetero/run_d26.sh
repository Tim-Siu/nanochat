#!/bin/bash

# Parameterised d26 run for hetero experiments (MoE, sharing, or combined).
# Matches speedrun.sh hyperparameters: depth=26, data:param ratio=8.25, device-batch-size=16, fp8.
#
# Usage:
#   MOE_LAYERS="20:24" SHARED_MLP_GROUPS="3:9:3" \
#   WANDB_RUN=d26_0218_combined DEVICE_BATCH_SIZE=16 \
#   MODEL_TAG=d26_0218_combined \
#     bash dev-hetero/run_d26.sh

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Python venv setup
source .venv/bin/activate

# Defaults
WANDB_RUN="${WANDB_RUN:-d26_hetero}"
MODEL_TAG="${MODEL_TAG:-d26_hetero}"
MOE_LAYERS="${MOE_LAYERS:-}"
SHARED_MLP_GROUPS="${SHARED_MLP_GROUPS:-}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Download enough shards for d26 training (~10B tokens)
python -m nanochat.dataset -n 370

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m dev-hetero.train_hetero -- \
    --depth=26 \
    --target-param-data-ratio=8.25 \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --moe-layers="$MOE_LAYERS" \
    --shared-mlp-groups="$SHARED_MLP_GROUPS" \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG" \
    --core-metric-every=999999 \
    --sample-every=-1
