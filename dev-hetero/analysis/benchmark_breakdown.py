"""Generate a heatmap of per-benchmark CORE scores across all experiments."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from load_runs import load_runs, get_benchmark_columns, get_output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate per-benchmark heatmaps.")
    parser.add_argument("--depth", type=int, required=True, help="Model depth (e.g. 12, 26)")
    parser.add_argument("--runs", type=str, default=None, help="Comma-separated whitelist of run directory names")
    args = parser.parse_args()
    depth = args.depth
    whitelist = args.runs.split(",") if args.runs else None

    df = load_runs(depth, whitelist=whitelist)
    output_dir = get_output_dir(depth)
    bench_cols = get_benchmark_columns(df)

    if not bench_cols:
        print(f"No benchmark columns found for d{depth}!")
        return

    # Deduplicate: keep best dense, best moe_all, all partials, shareds, and combined
    dense = df[df["exp_type"] == "Dense"].sort_values("mfu", ascending=False).iloc[:1]
    moe_all_runs = df[df["exp_type"] == "MoE (all)"]
    moe_all = moe_all_runs.sort_values("val_bpb").iloc[:1] if len(moe_all_runs) > 0 else pd.DataFrame()
    moe_partial = df[df["exp_type"] == "MoE (partial)"].sort_values("layer_start")
    shared = df[df["exp_type"] == "Shared"].sort_values("layer_start")
    combined = df[df["exp_type"] == "Combined"].sort_values("layer_start")

    all_runs = pd.concat([dense, moe_all, moe_partial, shared, combined])
    all_runs = all_runs.reset_index(drop=True)

    if len(all_runs) == 0:
        print(f"No runs to plot for d{depth}.")
        return

    # Build the heatmap matrix
    bench_data = all_runs[bench_cols].copy()
    bench_data.columns = [c.replace("bench_", "") for c in bench_cols]
    bench_data.index = all_runs["label"]

    # Also compute delta vs dense baseline
    dense_bench = dense[bench_cols].iloc[0].values
    delta_data = bench_data.copy()
    for i in range(len(delta_data)):
        delta_data.iloc[i] = bench_data.iloc[i].values - dense_bench

    # --- Figure 1: Raw benchmark scores heatmap ---
    fig_h = max(6, 2 + len(all_runs) * 0.6)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    sns.heatmap(bench_data, annot=True, fmt=".3f", cmap="RdYlGn",
                ax=ax, linewidths=0.5, annot_kws={"size": 7},
                cbar_kws={"label": "Centered Accuracy"})
    ax.set_title(f"Per-Benchmark CORE Scores (all d{depth} experiments)", fontsize=13)
    ax.set_ylabel("Experiment")
    ax.set_xlabel("Benchmark")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "benchmark_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {out_path}")

    # --- Figure 2: Delta vs dense baseline ---
    delta_no_dense = delta_data.iloc[1:]
    if len(delta_no_dense) > 0:
        fig, ax = plt.subplots(figsize=(18, fig_h))
        sns.heatmap(delta_no_dense, annot=True, fmt="+.3f", cmap="RdBu_r", center=0,
                    ax=ax, linewidths=0.5, annot_kws={"size": 7},
                    cbar_kws={"label": "Delta vs Dense Baseline"})
        ax.set_title(f"Per-Benchmark Score Delta vs Dense Baseline (d{depth})", fontsize=13)
        ax.set_ylabel("Experiment")
        ax.set_xlabel("Benchmark")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(output_dir, "benchmark_delta_heatmap.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved delta heatmap to {out_path}")

    # --- Save CSV with all benchmark scores ---
    csv_path = os.path.join(output_dir, "benchmark_scores.csv")
    bench_out = bench_data.copy()
    bench_out.insert(0, "experiment", all_runs["label"].values)
    bench_out.insert(1, "exp_type", all_runs["exp_type"].values)
    bench_out.insert(2, "val_bpb", all_runs["val_bpb"].values)
    bench_out.insert(3, "core_metric", all_runs["core_metric"].values)
    bench_out.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved benchmark CSV to {csv_path}")

    # --- Print benchmarks where methods significantly differ from Dense ---
    print(f"\nBenchmarks with largest average deltas (d{depth}):")

    if len(moe_partial) > 0:
        moe_partial_delta = delta_data.loc[moe_partial["label"].values]
        print("\n  MoE (partial) vs Dense (avg delta):")
        moe_avg = moe_partial_delta.mean()
        for name, val in moe_avg.sort_values(ascending=False).items():
            if abs(val) > 0.01:
                print(f"    {name:35s}: {val:+.4f}")

    if len(shared) > 0:
        shared_delta = delta_data.loc[shared["label"].values]
        print("\n  Shared vs Dense (avg delta):")
        shared_avg = shared_delta.mean()
        for name, val in shared_avg.sort_values(ascending=False).items():
            if abs(val) > 0.01:
                print(f"    {name:35s}: {val:+.4f}")

    if len(combined) > 0:
        combined_delta = delta_data.loc[combined["label"].values]
        print("\n  Combined vs Dense (avg delta):")
        combined_avg = combined_delta.mean()
        for name, val in combined_avg.sort_values(ascending=False).items():
            if abs(val) > 0.01:
                print(f"    {name:35s}: {val:+.4f}")


if __name__ == "__main__":
    main()
