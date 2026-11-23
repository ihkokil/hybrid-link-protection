"""Generate LaTeX tables for Baseline Comparisons, Headroom Sensitivity, and Multi-Matrix Uncertainty."""
import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEN_DIR = os.path.join(ROOT, "latex", "generated")
os.makedirs(GEN_DIR, exist_ok=True)


def build_baseline_table():
    csv_path = os.path.join(ROOT, "results", "baseline_comparisons.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive Benchmark Comparison: Traffic Availability ($T_A$), Congestion Drops ($D$), and Peak Link Utilization ($\text{MLU}$) across LFC, Random, HLP (2020), Max-Traffic, C-TA-HLP, and Optimal-ILP.}",
        r"\label{tab:baseline_comparison}",
        r"\begin{tabular}{@{}llrrrrrrc@{}}",
        r"\toprule",
        r"\textbf{Topology} & \textbf{B} & \textbf{LFC Only} & \textbf{Random} & \textbf{HLP (2020)} & \textbf{Max-Traffic} & \textbf{C-TA-HLP (Ours)} & \textbf{Optimal-ILP} & \textbf{Opt. Ratio} \\ \midrule",
    ]

    for (topo, b), grp in df.groupby(["topology", "budget"]):
        m_map = {row["method"]: row for _, row in grp.iterrows()}
        lfc_ta = m_map.get("LFC Only", {}).get("traffic_availability", 0) * 100
        rnd_ta = m_map.get("Random", {}).get("traffic_availability", 0) * 100
        hlp_ta = m_map.get("HLP (2020)", {}).get("traffic_availability", 0) * 100
        mt_ta = m_map.get("Max-Traffic", {}).get("traffic_availability", 0) * 100
        ta_ta = m_map.get("C-TA-HLP (Proposed)", {}).get("traffic_availability", 0) * 100
        ilp_ta = m_map.get("Optimal-ILP", {}).get("traffic_availability", 0) * 100

        # Optimality ratio: ratio of C-TA-HLP traffic availability to Optimal-ILP
        opt_ratio = (ta_ta / ilp_ta * 100) if ilp_ta > 0 else 100.0
        opt_str = f"{opt_ratio:.2f}\\%"

        lines.append(
            f"{topo} & {b} & {lfc_ta:.2f}\\% & {rnd_ta:.2f}\\% & {hlp_ta:.2f}\\% & "
            f"{mt_ta:.2f}\\% & \\textbf{{{ta_ta:.2f}\\%}} & {ilp_ta:.2f}\\% & {opt_str} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    out_path = os.path.join(GEN_DIR, "baseline_results_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")


def build_headroom_table():
    csv_path = os.path.join(ROOT, "results", "headroom_sensitivity.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Capacity Headroom Sensitivity on Abilene ($B=4$): Impact of Headroom Factor $\alpha$ on Traffic Availability, Congestion Drops, and Maximum Link Utilization.}",
        r"\label{tab:headroom_sensitivity}",
        r"\begin{tabular}{@{}clrrrr@{}}",
        r"\toprule",
        r"\textbf{Headroom $\alpha$} & \textbf{Scheme} & \textbf{$T_A$ (\%)} & \textbf{Drops} & \textbf{MLU} \\ \midrule",
    ]

    df_ab = df[df["topology"] == "abilene"]
    for hf, grp in df_ab.groupby("headroom_factor"):
        for _, row in grp.iterrows():
            m = row["method"]
            ta = row["traffic_availability"] * 100
            drops = int(row["capacity_drop_count"])
            mlu = row["max_link_utilization"]
            is_best = (m == "C-TA-HLP")
            m_str = f"\\textbf{{{m}}}" if is_best else m
            ta_str = f"\\textbf{{{ta:.2f}\\%}}" if is_best else f"{ta:.2f}\\%"
            dr_str = f"\\textbf{{{drops}}}" if is_best else f"{drops}"
            lines.append(f"{hf:.2f}$\\times$ & {m_str} & {ta_str} & {dr_str} & {mlu:.1f} \\\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    out_path = os.path.join(GEN_DIR, "headroom_results_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")


def build_traffic_uncertainty_table():
    csv_path = os.path.join(ROOT, "results", "traffic_uncertainty_stats.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Traffic Uncertainty Evaluation: Paired Statistical Comparison over $n=20$ Independent Perturbed Traffic Matrices (Noise $\sigma=20\%$).}",
        r"\label{tab:traffic_uncertainty}",
        r"\begin{tabular}{@{}llrrrrrc@{}}",
        r"\toprule",
        r"\textbf{Topology} & \textbf{B} & \textbf{HLP} & \textbf{C-TA-HLP} & \textbf{$\Delta$} & \textbf{$t$-stat} & \textbf{$p$-value} & \textbf{Cohen's $d$} \\ \midrule",
    ]

    for _, row in df.iterrows():
        topo = row["topology"]
        b = int(row["budget"])
        hlp = row["mean_hlp"] * 100
        ta = row["mean_ta"] * 100
        delta = row["mean_delta"] * 100
        t_stat = row["t_statistic"]
        p_val = row["p_value"]
        p_str = "$<0.001$" if p_val < 0.001 else f"{p_val:.4f}"
        d = row["cohens_d"]

        lines.append(
            f"{topo} & {b} & {hlp:.2f}\\% & \\textbf{{{ta:.2f}\\%}} & "
            f"+{delta:.3f}\\% & {t_stat:.2f} & {p_str} & {d:.2f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    out_path = os.path.join(GEN_DIR, "traffic_uncertainty_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    build_baseline_table()
    build_headroom_table()
    build_traffic_uncertainty_table()
    print("All LaTeX tables built successfully!")

