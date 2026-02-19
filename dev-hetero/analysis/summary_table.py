"""Generate a summary table of all experiments for a given depth as CSV and markdown."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import pandas as pd
from load_runs import load_runs, get_output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate experiment summary table.")
    parser.add_argument("--depth", type=int, required=True, help="Model depth (e.g. 12, 26)")
    parser.add_argument("--runs", type=str, default=None, help="Comma-separated whitelist of run directory names")
    args = parser.parse_args()
    depth = args.depth
    whitelist = args.runs.split(",") if args.runs else None

    df = load_runs(depth, whitelist=whitelist)
    output_dir = get_output_dir(depth)

    if len(df) == 0:
        print(f"No runs found for d{depth}.")
        return

    # Sort: Dense first, then MoE (all), then MoE (partial), then Shared, then Combined
    type_order = {"Dense": 0, "MoE (all)": 1, "MoE (partial)": 2, "Shared": 3, "Combined": 4}
    df["_sort"] = df["exp_type"].map(type_order).fillna(5) * 100 + df["layer_start"]
    df = df.sort_values("_sort").drop(columns=["_sort"])

    # For dense, pick the best (highest MFU) run if duplicates
    dense_mask = df["exp_type"] == "Dense"
    if dense_mask.sum() > 1:
        best_dense = df[dense_mask].sort_values("mfu", ascending=False).iloc[:1]
        df = pd.concat([best_dense, df[~dense_mask]])
        df["_sort"] = df["exp_type"].map(type_order).fillna(5) * 100 + df["layer_start"]
        df = df.sort_values("_sort").drop(columns=["_sort"])

    # For MoE (all), pick the best run if duplicates
    moe_all_mask = df["exp_type"] == "MoE (all)"
    if moe_all_mask.sum() > 1:
        best_moe_all = df[moe_all_mask].sort_values("val_bpb").iloc[:1]
        df = pd.concat([df[~moe_all_mask].iloc[:1], best_moe_all, df[~moe_all_mask].iloc[1:]])
        df["_sort"] = df["exp_type"].map(type_order).fillna(5) * 100 + df["layer_start"]
        df = df.sort_values("_sort").drop(columns=["_sort"])

    # Select columns for the table
    table_cols = ["label", "exp_type", "moe_layers", "shared_mlp_groups",
                  "val_bpb", "core_metric", "mfu", "training_time_min", "n_iters"]
    table = df[table_cols].copy()

    # Format numbers
    table["val_bpb"] = table["val_bpb"].map(lambda x: f"{x:.4f}")
    table["core_metric"] = table["core_metric"].map(lambda x: f"{x:.4f}" if x else "N/A")
    table["mfu"] = table["mfu"].map(lambda x: f"{x:.1f}" if x else "N/A")
    table["training_time_min"] = table["training_time_min"].map(lambda x: f"{x:.1f}" if x else "N/A")

    # Compute delta columns vs dense baseline
    dense_rows = df[df["exp_type"] == "Dense"]
    if len(dense_rows) == 0:
        print("No dense baseline found, cannot compute deltas.")
        return
    dense_bpb = dense_rows["val_bpb"].iloc[0]
    dense_core = dense_rows["core_metric"].iloc[0]

    df_out = df[table_cols].copy()
    df_out["bpb_delta"] = df["val_bpb"] - dense_bpb
    df_out["core_delta"] = df["core_metric"] - dense_core

    # Save CSV
    csv_path = os.path.join(output_dir, "summary_table.csv")
    df_out.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved CSV to {csv_path}")

    # Generate markdown table
    md_lines = []
    md_lines.append(f"# D{depth} Experiment Summary\n")
    md_lines.append(f"Dense baseline: val/bpb = {dense_bpb:.4f}, CORE = {dense_core:.4f}\n")
    md_lines.append("| Experiment | Type | val/bpb | bpb delta | CORE | CORE delta | MFU |")
    md_lines.append("|------------|------|---------|-----------|------|------------|-----|")

    for _, row in df_out.iterrows():
        bpb_d = f"{row['bpb_delta']:+.4f}" if row["bpb_delta"] != 0 else "baseline"
        core_d = f"{row['core_delta']:+.4f}" if row["core_delta"] != 0 else "baseline"
        md_lines.append(
            f"| {row['label']} | {row['exp_type']} | {row['val_bpb']:.4f} | {bpb_d} "
            f"| {row['core_metric']:.4f} | {core_d} | {row['mfu']:.1f} |"
        )

    md_path = os.path.join(output_dir, "summary_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved markdown to {md_path}")

    # Print to stdout
    print("\n" + "\n".join(md_lines))


if __name__ == "__main__":
    main()
