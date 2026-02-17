"""Shared utility to load all wandb runs for the d12 hetero/moe/dense experiments."""

import json
import os
import glob
import yaml
import pandas as pd


WANDB_DIR = "/data/android/xsy/debug/ncws/nc/wandb"
OUTPUT_DIR = "/data/android/xsy/debug/ncws/analysis_results/dev-hetero"


def _get_val(config: dict, key: str, default=""):
    v = config.get(key, default)
    if isinstance(v, dict):
        return v.get("value", default)
    return v


def load_all_runs() -> pd.DataFrame:
    """Load all wandb runs and return a DataFrame with parsed configs and metrics."""
    rows = []
    for run_dir in sorted(glob.glob(f"{WANDB_DIR}/run-*/")):
        config_file = os.path.join(run_dir, "files", "config.yaml")
        summary_file = os.path.join(run_dir, "files", "wandb-summary.json")
        if not os.path.exists(config_file) or not os.path.exists(summary_file):
            continue

        with open(config_file) as f:
            config = yaml.safe_load(f)
        with open(summary_file) as f:
            summary = json.load(f)

        depth = _get_val(config, "depth")
        moe_layers = str(_get_val(config, "moe_layers", "")).strip()
        shared_mlp = str(_get_val(config, "shared_mlp_groups", "")).strip()
        n_experts = _get_val(config, "n_routed_experts", 0)
        moe_topk = _get_val(config, "moe_top_k", 0)

        bpb = summary.get("val/bpb")
        core = summary.get("core_metric")
        step = summary.get("step", 0)
        mfu = summary.get("train/mfu")
        tok_sec = summary.get("train/tok_per_sec")
        centered = summary.get("centered_results", {})

        # Classify experiment type and label
        if moe_layers and ":" in moe_layers:
            parts = moe_layers.split(":")
            if len(parts) == 2 and parts[0] and parts[1]:
                start, end = int(parts[0]), int(parts[1])
                n_moe = end - start
                if n_moe == depth:
                    exp_type = "MoE (all)"
                else:
                    exp_type = "MoE (partial)"
                label = f"MoE {start}-{end-1}"
                layer_start = start
            else:
                # Malformed like "2:" — treat as all-MoE
                exp_type = "MoE (all)"
                label = "MoE (all)"
                layer_start = 0
        elif shared_mlp:
            exp_type = "Shared"
            parts = shared_mlp.split(":")
            start, end = int(parts[0]), int(parts[1])
            label = f"Shared {start}-{end-1}"
            layer_start = start
        else:
            exp_type = "Dense"
            label = "Dense"
            layer_start = 0

        # Skip failed runs (bpb > 2.0 or step == 0)
        if bpb is None or (isinstance(bpb, (float, int)) and bpb > 2.0) or step == 0:
            continue

        basename = os.path.basename(run_dir.rstrip("/"))
        run_id = basename.split("-")[-1]
        date = basename.replace("run-", "").split("-")[0]

        rows.append({
            "run_id": run_id,
            "date": date,
            "depth": depth,
            "exp_type": exp_type,
            "label": label,
            "layer_start": layer_start,
            "moe_layers": moe_layers,
            "shared_mlp_groups": shared_mlp,
            "n_routed_experts": n_experts,
            "moe_topk": moe_topk,
            "val_bpb": bpb,
            "core_metric": core,
            "step": step,
            "mfu": mfu,
            "tok_per_sec": tok_sec,
            **{f"bench_{k}": v for k, v in centered.items()},
        })

    df = pd.DataFrame(rows)
    return df


def load_d12_runs() -> pd.DataFrame:
    """Load only the d12 experiment runs (the main sweep)."""
    df = load_all_runs()
    return df[df["depth"] == 12].copy()


def get_benchmark_columns(df: pd.DataFrame) -> list:
    """Return sorted list of benchmark column names."""
    return sorted([c for c in df.columns if c.startswith("bench_")])


if __name__ == "__main__":
    df = load_d12_runs()
    print(f"Loaded {len(df)} d12 runs")
    print(df[["label", "exp_type", "val_bpb", "core_metric", "mfu"]].to_string(index=False))
