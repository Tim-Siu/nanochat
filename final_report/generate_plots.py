"""
Generate all figures for the Heterogeneity in LLM Layer Design report.

Usage:
    python generate_plots.py

Outputs:
    fig_sharing_placement.png  - Parameter sharing placement sweep (D12 + D26)
    fig_moe_placement.png      - MoE placement sweep (D12 + D26)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

BAR_COLOR = "#5B9BD5"
BAR_COLOR_ALT = "#ED7D31"  # for multi-layer MoE (different # layers)
BASELINE_COLOR = "#333333"
BEST_COLOR = "#70AD47"


# ─── Data ────────────────────────────────────────────────────────────────────────

# Dense baselines
D12_DENSE_BPB = 0.9032
D26_DENSE_BPB = 0.7448

# D12 MoE placement sweep (2 MoE layers, consecutive pairs)
d12_moe_labels = [
    "L1-2", "L2-3", "L3-4", "L4-5", "L5-6",
    "L6-7", "L7-8", "L8-9", "L9-10", "L10-11", "L11-12",
    "L5-11\n(7 layers)",
]
d12_moe_bpb = [
    0.9029, 0.9028, 0.9060, 0.9054, 0.9025,
    0.9021, 0.9016, 0.9014, 0.9014, 0.9009, 0.9034,
    0.9001,
]
d12_moe_n_layers = [2]*11 + [7]  # number of MoE layers per run

# D26 MoE placement sweep (4 MoE layers)
d26_moe_labels = ["L4-7", "L12-15", "L22-25"]
d26_moe_bpb = [0.7454, 0.7451, 0.7419]

# D12 Sharing placement sweep (3 shared layers, 1 group)
d12_sharing_labels = [
    "L1-3", "L2-4", "L3-5", "L4-6", "L5-7",
    "L6-8", "L7-9", "L8-10", "L9-11", "L10-12",
]
d12_sharing_bpb = [
    0.9063, 0.9067, 0.9070, 0.9077, 0.9077,
    0.9078, 0.9087, 0.9088, 0.9092, 0.9092,
]

# D26 Sharing placement sweep (6 shared layers unless noted)
d26_sharing_labels = [
    "L1-6\n(2x3)", "L12-15\n(1x4)", "L19-24\n(2x3)",
]
d26_sharing_bpb = [0.7465, 0.7472, 0.7483]


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def _annotate_bars(ax, bars, values, fmt=".4f"):
    """Place value labels above bars."""
    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:{fmt}}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7.5,
            rotation=45,
        )


def _highlight_best(bars, values, lower_is_better=True):
    """Color the best bar green."""
    best_idx = int(np.argmin(values) if lower_is_better else np.argmax(values))
    bars[best_idx].set_color(BEST_COLOR)


# ─── Figure 1: Parameter Sharing Placement ───────────────────────────────────

def plot_sharing_placement():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- D12 ---
    x = np.arange(len(d12_sharing_labels))
    bars = ax1.bar(x, d12_sharing_bpb, color=BAR_COLOR, width=0.6, zorder=3)
    ax1.axhline(D12_DENSE_BPB, color=BASELINE_COLOR, ls="--", lw=1.2,
                label=f"Dense baseline ({D12_DENSE_BPB:.4f})", zorder=2)
    _highlight_best(bars, d12_sharing_bpb)
    _annotate_bars(ax1, bars, d12_sharing_bpb)
    ax1.set_xticks(x)
    ax1.set_xticklabels(d12_sharing_labels, rotation=45, ha="right")
    ax1.set_ylabel("Validation BPB")
    ax1.set_title("D12  (3 shared layers)")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    # Set y-axis range to show variation clearly
    ymin = min(d12_sharing_bpb) - 0.002
    ymax = max(max(d12_sharing_bpb), D12_DENSE_BPB) + 0.003
    ax1.set_ylim(ymin, ymax)

    # --- D26 ---
    x2 = np.arange(len(d26_sharing_labels))
    bars2 = ax2.bar(x2, d26_sharing_bpb, color=BAR_COLOR, width=0.5, zorder=3)
    ax2.axhline(D26_DENSE_BPB, color=BASELINE_COLOR, ls="--", lw=1.2,
                label=f"Dense baseline ({D26_DENSE_BPB:.4f})", zorder=2)
    _highlight_best(bars2, d26_sharing_bpb)
    _annotate_bars(ax2, bars2, d26_sharing_bpb)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(d26_sharing_labels)
    ax2.set_ylabel("Validation BPB")
    ax2.set_title("D26  (6 shared layers)")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ymin2 = min(d26_sharing_bpb) - 0.001
    ymax2 = max(max(d26_sharing_bpb), D26_DENSE_BPB) + 0.002
    ax2.set_ylim(ymin2, ymax2)

    fig.suptitle("Cross-Layer Parameter Sharing: Validation BPB vs. Layer Position",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig_sharing_placement.png")
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# ─── Figure 2: MoE Placement ─────────────────────────────────────────────────

def plot_moe_placement():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5),
                                    gridspec_kw={"width_ratios": [3, 1]})

    # --- D12 ---
    x = np.arange(len(d12_moe_labels))
    colors = [BAR_COLOR if n == 2 else BAR_COLOR_ALT for n in d12_moe_n_layers]
    bars = ax1.bar(x, d12_moe_bpb, color=colors, width=0.65, zorder=3)
    ax1.axhline(D12_DENSE_BPB, color=BASELINE_COLOR, ls="--", lw=1.2,
                label=f"Dense baseline ({D12_DENSE_BPB:.4f})", zorder=2)
    _highlight_best(bars, d12_moe_bpb)
    _annotate_bars(ax1, bars, d12_moe_bpb)
    ax1.set_xticks(x)
    ax1.set_xticklabels(d12_moe_labels, rotation=45, ha="right")
    ax1.set_ylabel("Validation BPB")
    ax1.set_title("D12  (2 MoE layers; orange = 7 layers)")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ymin = min(d12_moe_bpb) - 0.002
    ymax = max(max(d12_moe_bpb), D12_DENSE_BPB) + 0.003
    ax1.set_ylim(ymin, ymax)

    # --- D26 ---
    x2 = np.arange(len(d26_moe_labels))
    bars2 = ax2.bar(x2, d26_moe_bpb, color=BAR_COLOR, width=0.5, zorder=3)
    ax2.axhline(D26_DENSE_BPB, color=BASELINE_COLOR, ls="--", lw=1.2,
                label=f"Dense baseline ({D26_DENSE_BPB:.4f})", zorder=2)
    _highlight_best(bars2, d26_moe_bpb)
    _annotate_bars(ax2, bars2, d26_moe_bpb)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(d26_moe_labels)
    ax2.set_ylabel("Validation BPB")
    ax2.set_title("D26  (4 MoE layers)")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ymin2 = D26_DENSE_BPB - 0.004
    ymax2 = max(d26_moe_bpb) + 0.002
    ax2.set_ylim(ymin2, ymax2)

    fig.suptitle("MoE Layer Placement: Validation BPB vs. Layer Position",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig_moe_placement.png")
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_sharing_placement()
    plot_moe_placement()
    print("All figures generated.")
