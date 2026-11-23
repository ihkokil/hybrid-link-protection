"""Comprehensive Campaign for C-TA-HLP Q1 Manuscript.

Runs:
1. Baseline Comparison & Optimality Gap:
   LFC, Random, HLP (2020), Max-Traffic, C-TA-HLP (Proposed), and Optimal-ILP.
2. Capacity Headroom Sensitivity:
   Headroom factors [1.1, 1.25, 1.5, 2.0] measuring traffic availability,
   capacity drops, and MLU.
3. Multi-Matrix Traffic Uncertainty Campaign:
   Independent demand matrix perturbations (20 replicates with Gaussian noise)
   providing genuine independent traffic realizations and paired statistical tests.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.protection import (
    compute_lfc_scheme,
    compute_hlp,
    compute_ta_hlp,
    compute_max_traffic_hlp,
    compute_random_hlp,
    compute_optimal_milp,
)
from src.simulator import simulate_failures
from src.sndlib import load_sndlib
from src.utils import ensure_dir, get_logger

log = get_logger("comprehensive_campaign")


def run_baseline_and_optimality_campaign(
    topologies: List[str] = ["abilene", "nobel-eu", "geant"],
    budgets: List[int] = [2, 4, 6, 8],
    num_trials: int = 300,
    seed: int = 42,
    out_dir: str = "results",
) -> pd.DataFrame:
    """Evaluate LFC, Random, HLP, Max-Traffic, C-TA-HLP, and Optimal-ILP."""
    ensure_dir(out_dir)
    rows = []

    for topo_name in topologies:
        log.info("Running baseline campaign on %s...", topo_name)
        G, traffic = load_sndlib(topo_name, seed=seed, headroom_factor=1.5)

        for B in budgets:
            log.info("  Topology %s, Budget B=%d", topo_name, B)

            methods = {
                "LFC Only": lambda: compute_lfc_scheme(G),
                "Random": lambda: compute_random_hlp(G, max_links=B, seed=seed + B),
                "HLP (2020)": lambda: compute_hlp(G, R=1.0, max_links=B),
                "Max-Traffic": lambda: compute_max_traffic_hlp(G, traffic, max_links=B),
                "C-TA-HLP (Proposed)": lambda: compute_ta_hlp(G, traffic, max_links=B),
                "Optimal-ILP": lambda: compute_optimal_milp(G, traffic, max_links=B),
            }

            for method_name, method_fn in methods.items():
                prot = method_fn()
                sim = simulate_failures(
                    G, prot.next_hops, prot.protected_links, traffic,
                    num_trials=num_trials, seed=seed + B * 17, estimand="unconditional",
                )
                rows.append({
                    "topology": topo_name,
                    "budget": B,
                    "method": method_name,
                    "backup_links": len(prot.protected_links),
                    "compute_time_s": prot.compute_time_s,
                    "traffic_availability": sim.traffic_availability,
                    "topological_availability": sim.topological_availability,
                    "capacity_drop_count": sim.capacity_drop_count,
                    "max_link_utilization": sim.max_link_utilization,
                    "ilp_obj": prot.extra.get("ilp_obj", np.nan),
                })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "baseline_comparisons.csv")
    df.to_csv(out_csv, index=False)
    log.info("Saved baseline comparisons to %s", out_csv)
    return df


def run_headroom_sensitivity(
    topologies: List[str] = ["abilene", "nobel-eu"],
    headroom_factors: List[float] = [1.1, 1.25, 1.5, 2.0],
    budget: int = 4,
    num_trials: int = 300,
    seed: int = 42,
    out_dir: str = "results",
) -> pd.DataFrame:
    """Sweep capacity headroom factors to analyze congestion penalty behavior."""
    ensure_dir(out_dir)
    rows = []

    for topo_name in topologies:
        for hf in headroom_factors:
            log.info("Headroom sensitivity: %s at headroom %.2fx (B=%d)", topo_name, hf, budget)
            G, traffic = load_sndlib(topo_name, seed=seed, headroom_factor=hf)

            methods = {
                "HLP (2020)": compute_hlp(G, R=1.0, max_links=budget),
                "Max-Traffic": compute_max_traffic_hlp(G, traffic, max_links=budget),
                "C-TA-HLP": compute_ta_hlp(G, traffic, max_links=budget),
            }

            for method_name, prot in methods.items():
                sim = simulate_failures(
                    G, prot.next_hops, prot.protected_links, traffic,
                    num_trials=num_trials, seed=seed, estimand="unconditional",
                )
                rows.append({
                    "topology": topo_name,
                    "headroom_factor": hf,
                    "budget": budget,
                    "method": method_name,
                    "traffic_availability": sim.traffic_availability,
                    "capacity_drop_count": sim.capacity_drop_count,
                    "max_link_utilization": sim.max_link_utilization,
                    "backup_links": len(prot.protected_links),
                })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "headroom_sensitivity.csv")
    df.to_csv(out_csv, index=False)
    log.info("Saved headroom sensitivity to %s", out_csv)
    return df


def run_traffic_uncertainty_campaign(
    topologies: List[str] = ["abilene", "nobel-eu"],
    budgets: List[int] = [2, 4],
    n_replicates: int = 20,
    noise_std: float = 0.20,
    num_trials: int = 250,
    base_seed: int = 100,
    out_dir: str = "results",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run evaluation over perturbed traffic matrices to provide independent demand realizations."""
    ensure_dir(out_dir)
    raw_rows = []

    for topo_name in topologies:
        log.info("Running multi-matrix uncertainty campaign on %s (reps=%d)...", topo_name, n_replicates)
        for rep in range(n_replicates):
            rep_seed = base_seed + rep * 100
            G, base_traffic = load_sndlib(topo_name, seed=rep_seed, headroom_factor=1.5)

            # Perturb traffic matrix with log-normal or Gaussian noise
            rng = random.Random(rep_seed)
            perturbed_traffic = {
                (s, d): val * max(0.05, 1.0 + rng.gauss(0.0, noise_std))
                for (s, d), val in base_traffic.items()
            }

            for B in budgets:
                # Compare HLP vs C-TA-HLP
                hlp = compute_hlp(G, R=1.0, max_links=B)
                ta = compute_ta_hlp(G, perturbed_traffic, max_links=B)

                sim_hlp = simulate_failures(
                    G, hlp.next_hops, hlp.protected_links, perturbed_traffic,
                    num_trials=num_trials, seed=rep_seed + B, estimand="unconditional",
                )
                sim_ta = simulate_failures(
                    G, ta.next_hops, ta.protected_links, perturbed_traffic,
                    num_trials=num_trials, seed=rep_seed + B, estimand="unconditional",
                )

                raw_rows.append({
                    "replicate": rep,
                    "topology": topo_name,
                    "budget": B,
                    "hlp_traffic_avail": sim_hlp.traffic_availability,
                    "ta_traffic_avail": sim_ta.traffic_availability,
                    "delta": sim_ta.traffic_availability - sim_hlp.traffic_availability,
                    "hlp_cap_drops": sim_hlp.capacity_drop_count,
                    "ta_cap_drops": sim_ta.capacity_drop_count,
                    "hlp_mlu": sim_hlp.max_link_utilization,
                    "ta_mlu": sim_ta.max_link_utilization,
                })

    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(os.path.join(out_dir, "traffic_uncertainty_raw.csv"), index=False)

    # Statistical summary
    summary_rows = []
    for (topo, B), group in df_raw.groupby(["topology", "budget"]):
        a = group["hlp_traffic_avail"].values
        b = group["ta_traffic_avail"].values
        diff = b - a
        t_stat, p_val = stats.ttest_rel(b, a, alternative="greater")
        w_stat, w_p = stats.wilcoxon(b, a, alternative="greater") if not np.all(diff == 0) else (0, 1.0)
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)

        summary_rows.append({
            "topology": topo,
            "budget": B,
            "n_replicates": len(group),
            "mean_hlp": np.mean(a),
            "mean_ta": np.mean(b),
            "mean_delta": np.mean(diff),
            "std_delta": np.std(diff, ddof=1),
            "t_statistic": t_stat,
            "p_value": p_val,
            "wilcoxon_p": w_p,
            "cohens_d": cohens_d,
            "significant_at_05": bool(p_val < 0.05),
        })

    df_summary = pd.DataFrame(summary_rows)
    out_sum = os.path.join(out_dir, "traffic_uncertainty_stats.csv")
    df_summary.to_csv(out_sum, index=False)
    log.info("Saved traffic uncertainty stats to %s", out_sum)
    return df_raw, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive experiments for Q1 readiness.")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=== Step 1: Baseline & Optimality Gap Campaign ===")
    run_baseline_and_optimality_campaign(num_trials=args.trials, seed=args.seed)

    print("=== Step 2: Capacity Headroom Sensitivity ===")
    run_headroom_sensitivity(num_trials=args.trials, seed=args.seed)

    print("=== Step 3: Multi-Matrix Traffic Uncertainty Campaign ===")
    run_traffic_uncertainty_campaign(n_replicates=args.replicates, num_trials=args.trials, base_seed=args.seed)

    print("=== All comprehensive experiments completed successfully! ===")

