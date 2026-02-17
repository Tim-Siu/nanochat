"""Compare MoE vs Parameter Sharing vs Dense across quality and per-benchmark metrics."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from load_runs import load_d12_runs, get_benchmark_columns, OUTPUT_DIR


def main():
    df = load_d12_runs()

    # Get representative runs for each method
    dense = df[df["exp_type"] == "Dense"].sort_values("mfu", ascending=False).iloc[:1]
    moe_all = df[df["exp_type"] == "MoE (all)"].sort_values("val_bpb").iloc[:1]
    moe_partial = df[df["exp_type"] == "MoE (partial)"]
    shared = df[df["exp_type"] == "Shared"]

    # Best partial MoE (by bpb)
    moe_best = moe_partial.sort_values("val_bpb").iloc[:1]
    # Worst partial MoE
    moe_worst = moe_partial.sort_values("val_bpb", ascending=False).iloc[:1]

    # Best shared (by bpb)
    shared_best = shared.sort_values("val_bpb").iloc[:1]
    # Worst shared
    shared_worst = shared.sort_values("val_bpb", ascending=False).iloc[:1]

    # --- Figure 1: Aggregate metrics comparison ---
    methods = pd.concat([
        dense.assign(method="Dense"),
        moe_all.assign(method="MoE (all layers)"),
        moe_best.assign(method=f"Best MoE partial\n({moe_best['label'].iloc[0]})"),
        moe_worst.assign(method=f"Worst MoE partial\n({moe_worst['label'].iloc[0]})"),
        shared_best.assign(method=f"Best Shared\n({shared_best['label'].iloc[0]})"),
        shared_worst.assign(method=f"Worst Shared\n({shared_worst['label'].iloc[0]})"),
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]

    # BPB comparison
    ax = axes[0]
    bars = ax.barh(range(len(methods)), methods["val_bpb"].values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods["method"].values)
    ax.set_xlabel("Validation BPB (lower is better)")
    ax.set_title("Validation BPB Comparison")
    # Narrow x-range to see differences
    bpb_vals = methods["val_bpb"].values
    ax.set_xlim(min(bpb_vals) - 0.002, max(bpb_vals) + 0.002)
    for bar, val in zip(bars, bpb_vals):
        ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    # CORE comparison
    ax = axes[1]
    bars = ax.barh(range(len(methods)), methods["core_metric"].values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods["method"].values)
    ax.set_xlabel("CORE Metric (higher is better)")
    ax.set_title("CORE Metric Comparison")
    core_vals = methods["core_metric"].values
    ax.set_xlim(min(core_vals) - 0.01, max(core_vals) + 0.02)
    for bar, val in zip(bars, core_vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    fig.suptitle("Method Comparison: Dense vs MoE vs Parameter Sharing (d=12)", fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "method_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")

    # --- Figure 2: Distribution comparison (box/violin) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    group_data_bpb = [
        dense["val_bpb"].values,
        moe_all["val_bpb"].values,
        moe_partial["val_bpb"].values,
        shared["val_bpb"].values,
    ]
    group_data_core = [
        dense["core_metric"].values,
        moe_all["core_metric"].values,
        moe_partial["core_metric"].values,
        shared["core_metric"].values,
    ]
    group_labels = ["Dense", "MoE (all)", "MoE (partial)", "Shared"]
    group_colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]

    # BPB box plot
    positions = range(len(group_labels))
    for i, (data, color) in enumerate(zip(group_data_bpb, group_colors)):
        if len(data) > 1:
            bp = ax1.boxplot([data], positions=[i], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.5),
                           medianprops=dict(color="black"))
        else:
            ax1.scatter([i], data, color=color, s=100, zorder=5, marker="D")
        ax1.scatter([i] * len(data), data, color=color, s=30, zorder=6, alpha=0.7)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(group_labels)
    ax1.set_ylabel("Validation BPB")
    ax1.set_title("BPB Distribution by Method")

    # CORE box plot
    for i, (data, color) in enumerate(zip(group_data_core, group_colors)):
        if len(data) > 1:
            bp = ax2.boxplot([data], positions=[i], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.5),
                           medianprops=dict(color="black"))
        else:
            ax2.scatter([i], data, color=color, s=100, zorder=5, marker="D")
        ax2.scatter([i] * len(data), data, color=color, s=30, zorder=6, alpha=0.7)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(group_labels)
    ax2.set_ylabel("CORE Metric")
    ax2.set_title("CORE Distribution by Method")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "method_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")

    # --- Print summary statistics ---
    print("\nMethod Statistics:")
    for name, group in [("Dense", dense), ("MoE (all)", moe_all),
                        ("MoE (partial)", moe_partial), ("Shared", shared)]:
        bpb_mean = group["val_bpb"].mean()
        bpb_std = group["val_bpb"].std() if len(group) > 1 else 0
        core_mean = group["core_metric"].mean()
        core_std = group["core_metric"].std() if len(group) > 1 else 0
        print(f"  {name:15s}: bpb={bpb_mean:.4f}+/-{bpb_std:.4f}, "
              f"core={core_mean:.4f}+/-{core_std:.4f} (n={len(group)})")


if __name__ == "__main__":
    main()
