#!/bin/bash

# Train a MoE model based on the d26 speedrun config.
# Layers 0-1 are dense, layers 2+ are MoE (1 shared + 7 routed experts, top-3).
#
# Usage:
#   bash dev-hetero/run_moe.sh
#   WANDB_RUN=moe_d26 bash dev-hetero/run_moe.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p $NANOCHAT_BASE_DIR

# Python venv setup
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Make sure dataset is downloaded (at least 370 shards for ~10B tokens)
python -m nanochat.dataset -n 370

# Number of GPUs
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# -----------------------------------------------------------------------------
# MoE model (d26, layers 0-1 dense, layers 2+ MoE)
# slightly overtrained: data:param ratio 12 (vs compute-optimal 10.5)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
    -m dev-hetero.train_moe -- \
    --depth=26 \
    --moe-layers="2:" \
    --n-routed-experts=7 \
    --moe-top-k=3 \
    --target-param-data-ratio=12 \
    --run=$WANDB_RUN

# Evaluate
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
