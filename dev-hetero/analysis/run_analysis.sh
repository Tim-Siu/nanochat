#!/bin/bash
# Run all analysis scripts for d12 and d26.
# Usage:
#   bash dev-hetero/analysis/run_analysis.sh              # run both d12 and d26
#   bash dev-hetero/analysis/run_analysis.sh --depth 12   # run d12 only
#   bash dev-hetero/analysis/run_analysis.sh --depth 26   # run d26 only

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${SCRIPT_DIR}/../../.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Python not found at $PYTHON"
    echo "Expected nc/.venv to exist."
    exit 1
fi

# =============================================
# Whitelists: edit these to control which runs
# are included in the analysis.
# =============================================

D12_RUNS="\
d12,\
d12_moe,\
d12_moe_b2_3,\
d12_moe_b3_4,\
d12_moe_b4_5,\
d12_moe_b5_6,\
d12_moe_b6_7,\
d12_moe_b7_8,\
d12_moe_b8_9,\
d12_moe_b9_10,\
d12_moe_b10_11,\
d12_moe_b11_12,\
d12_moe_b1-2,\
d12_moe_b5-11,\
d12_hetero_b1-3,\
d12_hetero_b2_3_4,\
d12_hetero_b3_4_5,\
d12_hetero_b4_5_6,\
d12_hetero_b5_6_7,\
d12_hetero_b6_7_8,\
d12_hetero_b7_8_9,\
d12_hetero_b8_9_10,\
d12_hetero_b9_10_11,\
d12_hetero_b10_11_12,\
d12_combined_moe10-11_shared1-3\
"

D26_RUNS="\
d26_0218_dense_bf16,\
d26_0218_moe_full,\
d26_0218_moe_b4-7,\
d26_0218_moe_b12-15,\
d26_0218_moe_b21-24,\
d26_0218_shared_b2-7,\
d26_0218_shared_b4-9,\
d26_0218_shared_b19-24,\
d26_0218_combined_b4-9_moe_b21-24,\
d26_0218_combined_b2-7_moe_b21-24\
"

# =============================================

# Parse depth arguments
DEPTHS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --depth)
            DEPTHS+=("$2")
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--depth 12] [--depth 26]"
            exit 1
            ;;
    esac
done

# Default: run both
if [ ${#DEPTHS[@]} -eq 0 ]; then
    DEPTHS=(12 26)
fi

SCRIPTS=(
    summary_table.py
    moe_placement.py
    sharing_placement.py
    benchmark_breakdown.py
    method_comparison.py
)

for depth in "${DEPTHS[@]}"; do
    if [ "$depth" -eq 12 ]; then
        RUNS="$D12_RUNS"
    elif [ "$depth" -eq 26 ]; then
        RUNS="$D26_RUNS"
    else
        echo "Unknown depth: $depth, running without whitelist"
        RUNS=""
    fi

    echo "===== Running analysis for d${depth} ====="
    for script in "${SCRIPTS[@]}"; do
        echo "--- ${script} --depth ${depth} ---"
        if [ -n "$RUNS" ]; then
            "$PYTHON" "${SCRIPT_DIR}/${script}" --depth "$depth" --runs "$RUNS"
        else
            "$PYTHON" "${SCRIPT_DIR}/${script}" --depth "$depth"
        fi
        echo ""
    done
done

echo "===== All analysis complete ====="
