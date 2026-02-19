"""Shared utility to load experiment runs from ncws/runs/ report files.

Primary data source: ncws/runs/*/report/*.md (config + aggregate metrics)
Optional enrichment: wandb centered_results for per-benchmark breakdown.
"""

import json
import os
import re
import glob
import yaml
import pandas as pd


RUNS_DIR = "/data/android/xsy/debug/ncws/runs"
WANDB_DIR = "/data/android/xsy/debug/ncws/nc/wandb"  # for benchmark enrichment only
OUTPUT_DIR = "/data/android/xsy/debug/ncws/analysis_results/dev-hetero"


def _parse_report(report_path: str) -> dict:
    """Parse a report markdown file into a dict of key-value pairs."""
    data = {}
    with open(report_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("- ") and ": " in line:
                # "- key: value"
                key, _, value = line[2:].partition(": ")
                key = key.strip()
                value = value.strip()
                data[key] = value
    return data


def _parse_float(s: str, default=None):
    """Parse a float from a string, stripping % and commas."""
    if not s:
        return default
    s = s.replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return default


def _parse_int(s: str, default=None):
    """Parse an int from a string, stripping commas."""
    if not s:
        return default
    s = s.replace(",", "").strip()
    try:
        return int(s)
    except ValueError:
        return default


def _parse_moe_layers(moe_layers: str, depth: int):
    """Parse moe_layers string, return (start, end, n_moe, is_all)."""
    parts = moe_layers.split(":")
    if len(parts) == 2 and parts[0] and parts[1]:
        start, end = int(parts[0]), int(parts[1])
        n_moe = end - start
        is_all = (n_moe >= depth - 1)
        return start, end, n_moe, is_all
    else:
        # Malformed like "2:" — treat as all-MoE
        return 0, depth, depth, True


def _parse_shared_mlp(shared_mlp: str):
    """Parse shared_mlp_groups string (supports 'start:end' and 'start:end:step')."""
    parts = shared_mlp.split(":")
    start = int(parts[0])
    end = int(parts[1])
    step = int(parts[2]) if len(parts) >= 3 else (end - start)
    return start, end, step


def _load_wandb_benchmarks() -> dict:
    """Load per-benchmark centered_results from wandb, keyed by model_tag.

    Returns dict: model_tag -> {bench_name: score, ...}
    For duplicate model_tags, keeps the latest wandb run.
    """
    benchmarks = {}
    for run_dir in sorted(glob.glob(f"{WANDB_DIR}/run-*/")):
        config_file = os.path.join(run_dir, "files", "config.yaml")
        summary_file = os.path.join(run_dir, "files", "wandb-summary.json")
        if not os.path.exists(config_file) or not os.path.exists(summary_file):
            continue
        with open(config_file) as f:
            config = yaml.safe_load(f)
        model_tag = config.get("model_tag", "")
        if isinstance(model_tag, dict):
            model_tag = model_tag.get("value", "")
        if not model_tag or model_tag == "None":
            continue
        with open(summary_file) as f:
            summary = json.load(f)
        centered = summary.get("centered_results", {})
        if centered:
            # sorted glob means latest run wins for duplicate model_tags
            benchmarks[model_tag] = centered
    return benchmarks


def load_all_runs(whitelist: list = None) -> pd.DataFrame:
    """Load runs from ncws/runs/ report files, enriched with wandb benchmarks.

    Args:
        whitelist: If provided, only load runs whose directory name is in this list.
    """
    # Pre-load wandb benchmark data
    wandb_benchmarks = _load_wandb_benchmarks()

    run_names = sorted(os.listdir(RUNS_DIR))
    if whitelist is not None:
        run_names = [r for r in run_names if r in whitelist]

    rows = []
    for run_name in run_names:
        run_path = os.path.join(RUNS_DIR, run_name)
        if not os.path.isdir(run_path):
            continue
        report_dir = os.path.join(run_path, "report")
        if not os.path.isdir(report_dir):
            continue

        # Find the report file (could be base-model-training.md, moe-*, hetero-*)
        report_files = glob.glob(os.path.join(report_dir, "*-model-training.md"))
        if not report_files:
            continue
        report = _parse_report(report_files[0])

        depth = _parse_int(report.get("depth", ""))
        if depth is None:
            continue

        moe_layers = report.get("moe_layers", report.get("MoE layers", "")).strip()
        shared_mlp = report.get("shared_mlp_groups", report.get("Shared MLP groups", "")).strip()
        n_experts = _parse_int(report.get("n_routed_experts", report.get("Routed experts", "")), 0)
        moe_topk = _parse_int(report.get("moe_top_k", report.get("Top-k", "")), 0)

        val_bpb = _parse_float(report.get("Minimum validation bpb", ""))
        core = _parse_float(report.get("CORE metric estimate", ""))
        mfu = _parse_float(report.get("MFU %", ""))
        training_time = _parse_float(report.get("Total training time", "").replace("m", ""))
        n_iters = _parse_int(report.get("Calculated number of iterations", ""))
        n_tokens = _parse_int(report.get("Number of training tokens", ""))
        peak_mem = _parse_float(report.get("Peak memory usage", "").replace("MiB", ""))
        model_tag = report.get("model_tag", run_name)

        # Parse param counts (format varies across report types)
        n_params_total = _parse_int(report.get(
            "Number of parameters (total, for scaling)",
            report.get("Number of parameters (total)",
                       report.get("Number of parameters", ""))))
        n_params_unique = _parse_int(report.get("Number of parameters (unique)", ""))
        n_params_active = _parse_int(report.get("Number of parameters (active)", ""))

        if val_bpb is None:
            continue

        # Classify experiment type
        has_moe = bool(moe_layers and ":" in moe_layers)
        has_shared = bool(shared_mlp)

        if has_moe and has_shared:
            exp_type = "Combined"
            moe_start, moe_end, _, _ = _parse_moe_layers(moe_layers, depth)
            sh_start, sh_end, _ = _parse_shared_mlp(shared_mlp)
            label = f"MoE {moe_start+1}-{moe_end} + Sh {sh_start+1}-{sh_end}"
            layer_start = min(moe_start, sh_start)
        elif has_moe:
            moe_start, moe_end, n_moe, is_all = _parse_moe_layers(moe_layers, depth)
            if is_all:
                exp_type = "MoE (all)"
                label = "MoE (all)"
                layer_start = 0
            else:
                exp_type = "MoE (partial)"
                label = f"MoE {moe_start+1}-{moe_end}"
                layer_start = moe_start
        elif has_shared:
            exp_type = "Shared"
            sh_start, sh_end, _ = _parse_shared_mlp(shared_mlp)
            label = f"Shared {sh_start+1}-{sh_end}"
            layer_start = sh_start
        else:
            exp_type = "Dense"
            label = "Dense"
            layer_start = 0

        # Wandb benchmark enrichment
        centered = wandb_benchmarks.get(model_tag, {})

        rows.append({
            "run_name": run_name,
            "model_tag": model_tag,
            "depth": depth,
            "exp_type": exp_type,
            "label": label,
            "layer_start": layer_start,
            "moe_layers": moe_layers,
            "shared_mlp_groups": shared_mlp,
            "n_routed_experts": n_experts,
            "moe_topk": moe_topk,
            "val_bpb": val_bpb,
            "core_metric": core,
            "mfu": mfu,
            "training_time_min": training_time,
            "n_iters": n_iters,
            "n_tokens": n_tokens,
            "n_params_total": n_params_total,
            "n_params_unique": n_params_unique,
            "n_params_active": n_params_active,
            "peak_mem_mib": peak_mem,
            **{f"bench_{k}": v for k, v in centered.items()},
        })

    df = pd.DataFrame(rows)
    return df


def load_runs(depth: int, whitelist: list = None) -> pd.DataFrame:
    """Load runs filtered by depth.

    Args:
        depth: Model depth to filter by.
        whitelist: If provided, only load runs whose directory name is in this list.
    """
    df = load_all_runs(whitelist=whitelist)
    return df[df["depth"] == depth].copy()


def load_d12_runs() -> pd.DataFrame:
    """Load only the d12 experiment runs."""
    return load_runs(12)


def load_d26_runs() -> pd.DataFrame:
    """Load only the d26 experiment runs."""
    return load_runs(26)


def get_output_dir(depth: int) -> str:
    """Get depth-specific output directory, creating it if needed."""
    d = os.path.join(OUTPUT_DIR, f"d{depth}")
    os.makedirs(d, exist_ok=True)
    return d


def get_benchmark_columns(df: pd.DataFrame) -> list:
    """Return sorted list of benchmark column names."""
    return sorted([c for c in df.columns if c.startswith("bench_")])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=None,
                        help="Filter by depth (e.g. 12, 26). If not set, show all.")
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated whitelist of run directory names.")
    args = parser.parse_args()

    whitelist = args.runs.split(",") if args.runs else None
    if args.depth:
        df = load_runs(args.depth, whitelist=whitelist)
        print(f"Loaded {len(df)} d{args.depth} runs")
    else:
        df = load_all_runs(whitelist=whitelist)
        print(f"Loaded {len(df)} total runs")
    print(df[["run_name", "label", "exp_type", "depth", "val_bpb", "core_metric", "mfu"]].to_string(index=False))
