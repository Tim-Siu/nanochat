"""Analyze MoE layer placement: how does the position of 2 MoE layers affect quality?"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from load_runs import load_d12_runs, OUTPUT_DIR


def main():
    df = load_d12_runs()

    # Get partial MoE runs (2 MoE layers at different positions)
    moe_partial = df[df["exp_type"] == "MoE (partial)"].sort_values("layer_start").copy()

    # Baselines
    dense = df[df["exp_type"] == "Dense"].iloc[0]
    moe_all_runs = df[df["exp_type"] == "MoE (all)"]
    moe_all = moe_all_runs.sort_values("val_bpb").iloc[0] if len(moe_all_runs) > 0 else None

    # Parse positions: "MoE 6-7" -> midpoint = 6.5 (1-indexed layers)
    positions = []
    for _, row in moe_partial.iterrows():
        parts = row["moe_layers"].split(":")
        start, end = int(parts[0]), int(parts[1])
        # These are 0-indexed layer indices; display as 1-indexed blocks
        positions.append(f"L{start+1}-{end}")
    moe_partial = moe_partial.copy()
    moe_partial["position_label"] = positions

    x = np.arange(len(moe_partial))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- BPB plot ---
    bars1 = ax1.bar(x, moe_partial["val_bpb"].values, width=0.6, color="#4C72B0", alpha=0.85)
    ax1.axhline(y=dense["val_bpb"], color="green", linestyle="--", linewidth=1.5,
                label=f"Dense baseline ({dense['val_bpb']:.4f})")
    if moe_all is not None:
        ax1.axhline(y=moe_all["val_bpb"], color="red", linestyle=":", linewidth=1.5,
                    label=f"MoE all layers ({moe_all['val_bpb']:.4f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(moe_partial["position_label"].values, rotation=45, ha="right")
    ax1.set_ylabel("Validation BPB (lower is better)")
    ax1.set_title("MoE Placement vs Validation BPB")
    ax1.legend(fontsize=9)

    # Highlight best
    best_idx = moe_partial["val_bpb"].values.argmin()
    bars1[best_idx].set_color("#2ca02c")
    bars1[best_idx].set_alpha(1.0)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, moe_partial["val_bpb"].values)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=7.5)

    # --- CORE plot ---
    bars2 = ax2.bar(x, moe_partial["core_metric"].values, width=0.6, color="#DD8452", alpha=0.85)
    ax2.axhline(y=dense["core_metric"], color="green", linestyle="--", linewidth=1.5,
                label=f"Dense baseline ({dense['core_metric']:.4f})")
    if moe_all is not None:
        ax2.axhline(y=moe_all["core_metric"], color="red", linestyle=":", linewidth=1.5,
                    label=f"MoE all layers ({moe_all['core_metric']:.4f})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(moe_partial["position_label"].values, rotation=45, ha="right")
    ax2.set_ylabel("CORE Metric (higher is better)")
    ax2.set_title("MoE Placement vs CORE Metric")
    ax2.legend(fontsize=9)

    # Highlight best
    best_idx = moe_partial["core_metric"].values.argmax()
    bars2[best_idx].set_color("#2ca02c")
    bars2[best_idx].set_alpha(1.0)

    for i, (bar, val) in enumerate(zip(bars2, moe_partial["core_metric"].values)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("Effect of MoE Layer Placement (2 MoE layers in 12-layer transformer)", fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "moe_placement.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")

    # Print numerical summary
    print("\nMoE Placement Summary (sorted by val/bpb):")
    summary = moe_partial[["position_label", "val_bpb", "core_metric"]].copy()
    summary["bpb_vs_dense"] = moe_partial["val_bpb"].values - dense["val_bpb"]
    summary["core_vs_dense"] = moe_partial["core_metric"].values - dense["core_metric"]
    summary = summary.sort_values("val_bpb")
    print(summary.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
