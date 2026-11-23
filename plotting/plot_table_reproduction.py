"""Diagnostic charts for the implemented method timing and failure delivery.

These plots are not reproductions of the base paper's Tables 3 and 4.
``MNP-e(ref)`` is output-equivalent to MNP-e but does not implement i-SPF, and
the delivery table is conditional on exactly one link failure.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.utils import ensure_dir

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

METHOD_ORDER = ["OSPF", "LFC", "TBFH", "MNP", "DMPA-e", "MNP-e(ref)"]
METHOD_COLORS = {
    "OSPF": "tab:gray",
    "LFC": "tab:purple",
    "TBFH": "tab:green",
    "MNP": "tab:orange",
    "DMPA-e": "tab:brown",
    "MNP-e(ref)": "tab:blue",
}


def plot_table3(csv_path: str, out_dir: str) -> None:
    if not os.path.exists(csv_path):
        print(f"  (skipping timing plot -- {csv_path} missing)")
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    x = range(len(df))
    width = 0.13
    for i, method in enumerate(METHOD_ORDER):
        if method not in df.columns:
            continue
        ax.bar([xi + (i - 2.5) * width for xi in x],
               df[method], width, label=method,
               color=METHOD_COLORS.get(method))
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["topology"], rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("Per-node compute time (us, log scale)")
    ax.set_title("Implementation diagnostic: per-node compute time (not paper Table 3)")
    ax.legend(ncol=6, loc="upper left", framealpha=0.7, fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(out_dir, "table3_compute_time_bars.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  saved -> {out}")


def plot_table4(csv_path: str, out_dir: str) -> None:
    if not os.path.exists(csv_path):
        print(f"  (skipping delivery plot -- {csv_path} missing)")
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    x = range(len(df))
    width = 0.13
    for i, method in enumerate(METHOD_ORDER):
        if method not in df.columns:
            continue
        ax.bar([xi + (i - 2.5) * width for xi in x],
               df[method], width, label=method,
               color=METHOD_COLORS.get(method))
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["topology"], rotation=15)
    ax.set_ylabel("Delivered pairs given one failure (%)")
    ax.set_ylim(75, 100.5)
    ax.set_title("Conditional single-failure delivery (not analytic availability)")
    ax.legend(ncol=6, loc="lower right", framealpha=0.7, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(out_dir, "table4_single_failure_delivery_bars.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  saved -> {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--plots-dir", default=os.path.join(ROOT, "plots"))
    args = ap.parse_args(argv)
    ensure_dir(args.plots_dir)
    plot_table3(os.path.join(args.results_dir, "table3_compute_time.csv"), args.plots_dir)
    plot_table4(
        os.path.join(args.results_dir, "table4_single_failure_delivery.csv"),
        args.plots_dir,
    )


if __name__ == "__main__":
    main()
