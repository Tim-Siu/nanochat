"""Compare MoE vs Parameter Sharing vs Dense vs Combined across quality metrics."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_runs import load_runs, get_output_dir


def main():
    parser = argparse.ArgumentParser(description="Compare methods head-to-head.")
    parser.add_argument("--depth", type=int, required=True, help="Model depth (e.g. 12, 26)")
    parser.add_argument("--runs", type=str, default=None, help="Comma-separated whitelist of run directory names")
    args = parser.parse_args()
    depth = args.depth
    whitelist = args.runs.split(",") if args.runs else None

    df = load_runs(depth, whitelist=whitelist)
    output_dir = get_output_dir(depth)

    # Get representative runs for each method
    dense = df[df["exp_type"] == "Dense"].sort_values("mfu", ascending=False).iloc[:1]
    moe_all_runs = df[df["exp_type"] == "MoE (all)"]
    moe_all = moe_all_runs.sort_values("val_bpb").iloc[:1] if len(moe_all_runs) > 0 else pd.DataFrame()
    moe_partial = df[df["exp_type"] == "MoE (partial)"]
    shared = df[df["exp_type"] == "Shared"]
    combined = df[df["exp_type"] == "Combined"]

    # --- Figure 1: Aggregate metrics comparison (representative runs) ---
    method_rows = [dense.assign(method="Dense")]
    colors = ["#2ca02c"]

    if len(moe_all) > 0:
        method_rows.append(moe_all.assign(method="MoE (all layers)"))
        colors.append("#d62728")

    if len(moe_partial) > 0:
        moe_best = moe_partial.sort_values("val_bpb").iloc[:1]
        method_rows.append(moe_best.assign(method=f"Best MoE partial\n({moe_best['label'].iloc[0]})"))
        colors.append("#1f77b4")
        if len(moe_partial) > 1:
            moe_worst = moe_partial.sort_values("val_bpb", ascending=False).iloc[:1]
            method_rows.append(moe_worst.assign(method=f"Worst MoE partial\n({moe_worst['label'].iloc[0]})"))
            colors.append("#aec7e8")

    if len(shared) > 0:
        shared_best = shared.sort_values("val_bpb").iloc[:1]
        method_rows.append(shared_best.assign(method=f"Best Shared\n({shared_best['label'].iloc[0]})"))
        colors.append("#ff7f0e")
        if len(shared) > 1:
            shared_worst = shared.sort_values("val_bpb", ascending=False).iloc[:1]
            method_rows.append(shared_worst.assign(method=f"Worst Shared\n({shared_worst['label'].iloc[0]})"))
            colors.append("#ffbb78")

    if len(combined) > 0:
        comb_best = combined.sort_values("val_bpb").iloc[:1]
        method_rows.append(comb_best.assign(method=f"Combined\n({comb_best['label'].iloc[0]})"))
        colors.append("#9467bd")

    methods = pd.concat(method_rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 1 + len(methods) * 0.7)))

    # BPB comparison
    ax = axes[0]
    bars = ax.barh(range(len(methods)), methods["val_bpb"].values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods["method"].values)
    ax.set_xlabel("Validation BPB (lower is better)")
    ax.set_title("Validation BPB Comparison")
    bpb_vals = methods["val_bpb"].values
    ax.set_xlim(min(bpb_vals) - 0.002, max(bpb_vals) + 0.004)
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

    fig.suptitle(f"Method Comparison: Dense vs MoE vs Sharing vs Combined (d={depth})", fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "method_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")

    # --- Figure 2: Distribution comparison (box/violin) ---
    group_data_bpb = []
    group_data_core = []
    group_labels = []
    group_colors = []

    for label, group, color in [
        ("Dense", dense, "#2ca02c"),
        ("MoE (all)", moe_all, "#d62728"),
        ("MoE (partial)", moe_partial, "#1f77b4"),
        ("Shared", shared, "#ff7f0e"),
        ("Combined", combined, "#9467bd"),
    ]:
        if len(group) > 0:
            group_data_bpb.append(group["val_bpb"].values)
            group_data_core.append(group["core_metric"].values)
            group_labels.append(label)
            group_colors.append(color)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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
    ax1.set_xticklabels(group_labels, rotation=30, ha="right")
    ax1.set_ylabel("Validation BPB")
    ax1.set_title("BPB Distribution by Method")

    for i, (data, color) in enumerate(zip(group_data_core, group_colors)):
        if len(data) > 1:
            bp = ax2.boxplot([data], positions=[i], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.5),
                           medianprops=dict(color="black"))
        else:
            ax2.scatter([i], data, color=color, s=100, zorder=5, marker="D")
        ax2.scatter([i] * len(data), data, color=color, s=30, zorder=6, alpha=0.7)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(group_labels, rotation=30, ha="right")
    ax2.set_ylabel("CORE Metric")
    ax2.set_title("CORE Distribution by Method")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "method_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")

    # --- Print summary statistics ---
    print(f"\nMethod Statistics (d{depth}):")
    for name, group in [("Dense", dense), ("MoE (all)", moe_all),
                        ("MoE (partial)", moe_partial), ("Shared", shared),
                        ("Combined", combined)]:
        if len(group) == 0:
            continue
        bpb_mean = group["val_bpb"].mean()
        bpb_std = group["val_bpb"].std() if len(group) > 1 else 0
        core_mean = group["core_metric"].mean()
        core_std = group["core_metric"].std() if len(group) > 1 else 0
        print(f"  {name:15s}: bpb={bpb_mean:.4f}+/-{bpb_std:.4f}, "
              f"core={core_mean:.4f}+/-{core_std:.4f} (n={len(group)})")


if __name__ == "__main__":
    main()
