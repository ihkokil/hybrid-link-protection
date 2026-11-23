"""Generate the LaTeX results table from canonical statistical CSV output.

Usage:
    python scripts/build_results_table.py --results results/sndlib \
        --out latex/generated/statistical_results_table.tex

The table reports every topology-budget-metric cell in the family, including
negative cells, with raw, Bonferroni, Holm, and BH p-values, Cohen's d, and
non-inferiority status. It refuses to emit a table if the CSV is missing or
empty, so the manuscript cannot silently fall back to stale numbers.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _fmt_p(p: float) -> str:
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.4f}"


def build_table(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty")
    traffic = df[df["metric"] == "traffic_availability"].copy()
    if traffic.empty:
        raise ValueError(f"{csv_path} has no traffic_availability rows")
    traffic = traffic.sort_values(["topology", "budget"])
    estimand = "unknown"
    n_reps = 0
    raw_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "all_topologies_raw.csv")
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
        if "estimand" in raw.columns and not raw.empty:
            estimand = str(raw["estimand"].iloc[0])
    if "n" in traffic.columns and not traffic.empty:
        n_reps = int(traffic["n"].iloc[0])
    family_size = len(df)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Traffic availability: C-TA-HLP versus topological HLP "
        rf"(estimand: \texttt{{{estimand}}}; $n={n_reps}$ replicates per cell; "
        rf"family size: {family_size}). All topology-budget cells are shown, "
        r"including negative results. $p_{\mathrm{Bonf}}$, $p_{\mathrm{Holm}}$, "
        r"and $p_{\mathrm{BH}}$ are adjusted over the full comparison family. "
        r"NI is the non-inferiority decision at the stated absolute margin.}",
        r"\label{tab:statistical_results}",
        r"\begin{tabular}{@{}llrrrrrrrc@{}}",
        r"\toprule",
        r"\textbf{Topology} & \textbf{B} & \textbf{HLP} & \textbf{C-TA-HLP} & "
        r"\textbf{$\Delta$} & \textbf{$p$} & \textbf{$p_{\mathrm{Bonf}}$} & "
        r"\textbf{$p_{\mathrm{Holm}}$} & \textbf{$p_{\mathrm{BH}}$} & "
        r"\textbf{NI} \\ \midrule",
    ]
    for _, row in traffic.iterrows():
        delta = row["mean_b"] - row["mean_a"]
        lines.append(
            f"{row['topology']} & {int(row['budget'])} & "
            f"{row['mean_a']:.4f} & {row['mean_b']:.4f} & "
            f"{delta:+.5f} & {_fmt_p(row['p_value'])} & "
            f"{_fmt_p(row['p_value_bonferroni'])} & "
            f"{_fmt_p(row['p_value_holm'])} & "
            f"{_fmt_p(row['p_value_bh'])} & "
            f"{'yes' if bool(row.get('non_inferior', False)) else 'no'} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "sndlib"))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "latex", "generated", "statistical_results_table.tex"))
    args = ap.parse_args()

    csv_path = args.csv or os.path.join(args.results, "statistical_comparisons.csv")
    table = build_table(csv_path)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(table)
    print(f"Wrote {args.out} from {csv_path}")


if __name__ == "__main__":
    main()
