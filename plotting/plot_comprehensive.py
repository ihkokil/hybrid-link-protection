"""Generate publication-ready figures for Headroom Sensitivity, Baselines, and Traffic Uncertainty."""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(ROOT, "latex", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Set publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.3,
})


def plot_headroom_sensitivity():
    csv_path = os.path.join(ROOT, "results", "headroom_sensitivity.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    df_abilene = df[df["topology"] == "abilene"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # Subplot 1: Traffic Availability vs Headroom Factor
    for method, marker, color, ls in [
        ("HLP (2020)", "s", "#d95f02", "--"),
        ("Max-Traffic", "^", "#7570b3", "-."),
        ("C-TA-HLP", "o", "#1b9e77", "-"),
    ]:
        sub = df_abilene[df_abilene["method"] == method].sort_values("headroom_factor")
        ax1.plot(sub["headroom_factor"], sub["traffic_availability"] * 100, label=method, marker=marker, color=color, linestyle=ls)

    ax1.set_xlabel(r"Capacity Headroom Factor $\alpha$ ($C_e = \alpha \cdot L_0(e)$)")
    ax1.set_ylabel("Traffic Availability (%)")
    ax1.set_title("(a) Traffic Availability vs. Headroom")
    ax1.grid(True)
    ax1.legend(loc="lower right")

    # Subplot 2: Capacity Drops vs Headroom Factor
    for method, marker, color, ls in [
        ("HLP (2020)", "s", "#d95f02", "--"),
        ("Max-Traffic", "^", "#7570b3", "-."),
        ("C-TA-HLP", "o", "#1b9e77", "-"),
    ]:
        sub = df_abilene[df_abilene["method"] == method].sort_values("headroom_factor")
        ax2.plot(sub["headroom_factor"], sub["capacity_drop_count"], label=method, marker=marker, color=color, linestyle=ls)

    ax2.set_xlabel(r"Capacity Headroom Factor $\alpha$")
    ax2.set_ylabel("Congestion Drop Count (Flows)")
    ax2.set_title("(b) Downstream Drops vs. Headroom")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    pdf_out = os.path.join(FIG_DIR, "headroom_sensitivity.pdf")
    plt.savefig(pdf_out, dpi=300)
    plt.close()
    print(f"Saved {pdf_out}")


def plot_baseline_comparisons():
    csv_path = os.path.join(ROOT, "results", "baseline_comparisons.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    for idx, (topo, title) in enumerate([("abilene", "Abilene (|V|=12, |E|=15)"), ("geant", "GÉANT (|V|=22, |E|=36)")]):
        ax = axes[idx]
        sub_df = df[df["topology"] == topo]

        budgets = sorted(sub_df["budget"].unique())
        x = np.arange(len(budgets))
        width = 0.15

        methods = ["LFC Only", "Random", "HLP (2020)", "Max-Traffic", "C-TA-HLP (Proposed)"]
        colors = ["#999999", "#e7298a", "#d95f02", "#7570b3", "#1b9e77"]

        for m_idx, (m, c) in enumerate(zip(methods, colors)):
            vals = []
            for b in budgets:
                cell = sub_df[(sub_df["budget"] == b) & (sub_df["method"] == m)]
                vals.append(cell["traffic_availability"].values[0] * 100 if len(cell) > 0 else 0)
            ax.bar(x + (m_idx - 2) * width, vals, width, label=m if idx == 0 else "", color=c, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"B={b}" for b in budgets])
        ax.set_xlabel("Tunnel Budget B")
        ax.set_ylabel("Traffic Availability (%)")
        ax.set_title(f"({chr(97+idx)}) {title}")
        y_min = sub_df["traffic_availability"].min() * 100 - 0.5
        y_max = sub_df["traffic_availability"].max() * 100 + 0.2
        ax.set_ylim(max(90, y_min), min(100, y_max))
        ax.grid(True, axis="y")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=True)
    plt.tight_layout()
    pdf_out = os.path.join(FIG_DIR, "baseline_comparison.pdf")
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {pdf_out}")


def plot_traffic_uncertainty():
    csv_path = os.path.join(ROOT, "results", "traffic_uncertainty_raw.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    for idx, (topo, title) in enumerate([("abilene", "Abilene (20 Perturbed Matrices)"), ("nobel-eu", "Nobel-EU (20 Perturbed Matrices)")]):
        ax = axes[idx]
        sub = df[(df["topology"] == topo) & (df["budget"] == 2)]

        data = [sub["hlp_traffic_avail"].values * 100, sub["ta_traffic_avail"].values * 100]
        bp = ax.boxplot(data, tick_labels=["HLP (2020)", "C-TA-HLP (Ours)"], patch_artist=True, widths=0.45)
        bp['boxes'][0].set_facecolor('#fee0d2')
        bp['boxes'][1].set_facecolor('#e5f5e0')

        # Scatter individual replicate points
        for i, col in enumerate([sub["hlp_traffic_avail"].values * 100, sub["ta_traffic_avail"].values * 100]):
            jitter = np.random.normal(0, 0.04, size=len(col))
            ax.plot(np.ones_like(col) * (i + 1) + jitter, col, 'r.' if i == 0 else 'g.', alpha=0.6, markersize=7)

        ax.set_ylabel("Traffic Availability (%)")
        ax.set_title(f"({chr(97+idx)}) {title}")
        ax.grid(True, axis="y")

    plt.tight_layout()
    pdf_out = os.path.join(FIG_DIR, "traffic_uncertainty_boxplots.pdf")
    plt.savefig(pdf_out, dpi=300)
    plt.close()
    print(f"Saved {pdf_out}")


if __name__ == "__main__":
    plot_headroom_sensitivity()
    plot_baseline_comparisons()
    plot_traffic_uncertainty()
    print("All comprehensive plots generated successfully!")

