"""Generate a summary table of all d12 experiments as CSV and markdown."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from load_runs import load_d12_runs, OUTPUT_DIR


def main():
    df = load_d12_runs()

    # Sort: Dense first, then MoE (all), then MoE (partial) by layer_start, then Shared by layer_start
    type_order = {"Dense": 0, "MoE (all)": 1, "MoE (partial)": 2, "Shared": 3}
    df["_sort"] = df["exp_type"].map(type_order) * 100 + df["layer_start"]
    df = df.sort_values("_sort").drop(columns=["_sort"])

    # For dense, pick the best (highest MFU) run if duplicates
    dense_mask = df["exp_type"] == "Dense"
    if dense_mask.sum() > 1:
        best_dense = df[dense_mask].sort_values("mfu", ascending=False).iloc[:1]
        df = pd.concat([best_dense, df[~dense_mask]])
        type_order_map = {"Dense": 0, "MoE (all)": 1, "MoE (partial)": 2, "Shared": 3}
        df["_sort"] = df["exp_type"].map(type_order_map) * 100 + df["layer_start"]
        df = df.sort_values("_sort").drop(columns=["_sort"])

    # For MoE (all), pick the best run
    moe_all_mask = df["exp_type"] == "MoE (all)"
    if moe_all_mask.sum() > 1:
        best_moe_all = df[moe_all_mask].sort_values("val_bpb").iloc[:1]
        df = pd.concat([df[~moe_all_mask].iloc[:1], best_moe_all, df[~moe_all_mask].iloc[1:]])
        type_order_map = {"Dense": 0, "MoE (all)": 1, "MoE (partial)": 2, "Shared": 3}
        df["_sort"] = df["exp_type"].map(type_order_map) * 100 + df["layer_start"]
        df = df.sort_values("_sort").drop(columns=["_sort"])

    # Select columns for the table
    table_cols = ["label", "exp_type", "moe_layers", "shared_mlp_groups",
                  "val_bpb", "core_metric", "mfu", "tok_per_sec", "step"]
    table = df[table_cols].copy()

    # Format numbers
    table["val_bpb"] = table["val_bpb"].map(lambda x: f"{x:.4f}")
    table["core_metric"] = table["core_metric"].map(lambda x: f"{x:.4f}" if x else "N/A")
    table["mfu"] = table["mfu"].map(lambda x: f"{x:.1f}" if x else "N/A")
    table["tok_per_sec"] = table["tok_per_sec"].map(lambda x: f"{x:.0f}" if x else "N/A")

    # Compute delta columns vs dense baseline
    dense_bpb = df[df["exp_type"] == "Dense"]["val_bpb"].iloc[0]
    dense_core = df[df["exp_type"] == "Dense"]["core_metric"].iloc[0]

    df_out = df[table_cols].copy()
    df_out["bpb_delta"] = df["val_bpb"] - dense_bpb
    df_out["core_delta"] = df["core_metric"] - dense_core

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "summary_table.csv")
    df_out.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved CSV to {csv_path}")

    # Generate markdown table
    md_lines = []
    md_lines.append("# D12 Experiment Summary\n")
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

    md_path = os.path.join(OUTPUT_DIR, "summary_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved markdown to {md_path}")

    # Print to stdout
    print("\n" + "\n".join(md_lines))


if __name__ == "__main__":
    import pandas as pd
    main()
