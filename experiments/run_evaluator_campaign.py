#!/usr/bin/env python3
"""Unified Experiment Runner for C-TA-HLP with SRLG/IBF Support.

This script runs the complete experimental campaign using the exact SRLG/IBF model
implemented in src/evaluator_srlg.py. It supports:
- SLF (Single-Link Failure) model
- SRLG (Shared Risk Link Groups) model  
- IBF (Independent Bernoulli Failures) model

Usage:
    python experiments/run_evaluator_campaign.py --help

Outputs:
    - results/evaluator/raw_results.csv
    - results/evaluator/statistical_table.tex (LaTeX table)
    - results/evaluator/baseline_comparison.csv
    - results/evaluator/headroom_sensitivity.csv
    - results/evaluator/traffic_uncertainty_stats.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import pandas as pd
import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.evaluator import (
    run_full_campaign,
    ExperimentConfig,
    BASELINE_SELECTORS,
    build_topology_data,
    load_topology_with_traffic,
    select_cta_hlp,
    select_optimal_ilp,
)
from src.evaluator_srlg import (
    build_srlg_topology_data,
    SRLG_BASELINE_SELECTORS,
    run_single_experiment_srlg,
    SRLGTopologyData,
    generate_synthetic_srlgs,
)
from src.utils import ensure_dir, get_logger

log = get_logger("evaluator_campaign")


def generate_statistical_table(raw_results: pd.DataFrame, output_path: str):
    """Generate LaTeX statistical comparison table."""
    # Filter for canonical (non-perturbed) runs
    canonical = raw_results[
        (raw_results['replicate'] == 0) & 
        (~raw_results['topology'].str.contains('headroom'))
    ].copy()
    
    comparisons = []
    
    for (topo, budget), group in canonical.groupby(['topology', 'budget']):
        if 'C_TA_HLP' not in group['baseline'].values or 'HLP_2020' not in group['baseline'].values:
            continue
            
        cta_row = group[group['baseline'] == 'C_TA_HLP'].iloc[0]
        hlp_row = group[group['baseline'] == 'HLP_2020'].iloc[0]
        
        delta = cta_row['unconditional_traffic_availability'] - hlp_row['unconditional_traffic_availability']
        
        comparisons.append({
            'topology': topo,
            'budget': budget,
            'hlp_traffic_avail': hlp_row['unconditional_traffic_availability'],
            'cta_traffic_avail': cta_row['unconditional_traffic_availability'],
            'delta': delta,
            'hlp_topo_avail': hlp_row['unconditional_topological_availability'],
            'cta_topo_avail': cta_row['unconditional_topological_availability'],
            'failure_model': cta_row.get('failure_model', 'SLF'),
        })
    
    df_comp = pd.DataFrame(comparisons)
    
    # Generate LaTeX
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Statistical Comparison: C-TA-HLP vs. HLP (2020) on SNDlib NetFlow Demands. "
        r"Values are unconditional traffic availability. Single-replicate deterministic evaluation; "
        r"statistical significance requires multi-replicate campaigns (see traffic uncertainty results).}",
        r"\label{tab:statistical_results}",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"\textbf{Topology} & \textbf{Budget} & \textbf{Model} & \textbf{HLP} & \textbf{C-TA-HLP} & \textbf{$\Delta$} & \textbf{HLP Topo} & \textbf{C-TA-HLP Topo} \\",
        r"\midrule",
    ]
    
    for _, row in df_comp.iterrows():
        latex_lines.append(
            f"{row['topology'].capitalize()} & {int(row['budget'])} & {row['failure_model']} & "
            f"{row['hlp_traffic_avail']:.6f} & {row['cta_traffic_avail']:.6f} & "
            f"{row['delta']:+.6f} & {row['hlp_topo_avail']:.4f} & {row['cta_topo_avail']:.4f} \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    log.info(f"Generated statistical table: {output_path}")
    return df_comp


def generate_baseline_comparison_table(raw_results: pd.DataFrame, output_path: str):
    """Generate comprehensive baseline comparison table."""
    canonical = raw_results[
        (raw_results['replicate'] == 0) & 
        (~raw_results['topology'].str.contains('headroom'))
    ].copy()
    
    metrics = ['unconditional_traffic_availability', 'unconditional_topological_availability',
               'unconditional_mean_mlu', 'unconditional_peak_mlu', 
               'unconditional_capacity_drops']
    
    pivot = canonical.pivot_table(
        index=['topology', 'budget', 'failure_model'],
        columns='baseline',
        values=metrics
    )
    
    pivot.to_csv(output_path)
    log.info(f"Generated baseline comparison CSV: {output_path}")
    return pivot


def generate_traffic_uncertainty_stats(raw_results: pd.DataFrame, output_path: str):
    """Generate statistical summary for traffic uncertainty campaign."""
    canonical_base = raw_results[
        (raw_results['replicate'] == 0) & 
        (raw_results['headroom_factor'].isna() | (raw_results['headroom_factor'] == ''))
    ].copy()
    
    base_totals = canonical_base.groupby('topology')['traffic_total'].first().to_dict()
    
    perturbed = raw_results[
        (raw_results['headroom_factor'].isna() | (raw_results['headroom_factor'] == '')) &
        (raw_results.apply(lambda row: row['traffic_total'] != base_totals.get(row['topology'], 0), axis=1))
    ].copy()
    
    if len(perturbed) == 0:
        log.warning("No perturbed traffic results found")
        return None
    
    summary_rows = []
    
    for (topo, budget, model), group in perturbed.groupby(['topology', 'budget', 'failure_model']):
        hlp_data = group[group['baseline'] == 'HLP_2020']
        cta_data = group[group['baseline'] == 'C_TA_HLP']
        
        if len(hlp_data) == 0 or len(cta_data) == 0:
            continue
        
        n_reps = min(len(hlp_data), len(cta_data))
        if n_reps < 2:
            continue
        
        hlp_sorted = hlp_data.sort_values('replicate').head(n_reps)
        cta_sorted = cta_data.sort_values('replicate').head(n_reps)
        
        a = hlp_sorted['unconditional_traffic_availability'].values
        b = cta_sorted['unconditional_traffic_availability'].values
        diff = b - a
        
        t_stat, p_val = stats.ttest_rel(b, a)
        
        try:
            w_stat, w_p = stats.wilcoxon(b, a)
        except ValueError:
            w_p = 1.0
        
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
        
        summary_rows.append({
            'topology': topo,
            'budget': budget,
            'failure_model': model,
            'n_replicates': n_reps,
            'mean_hlp': np.mean(a),
            'mean_cta': np.mean(b),
            'mean_delta': np.mean(diff),
            'std_delta': np.std(diff, ddof=1),
            't_statistic': t_stat,
            'p_value_two_sided': p_val,
            'wilcoxon_p_two_sided': w_p,
            'cohens_d': cohens_d,
            'significant_at_05': p_val < 0.05,
        })
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_path, index=False)
    log.info(f"Generated traffic uncertainty stats: {output_path}")
    return df_summary


def generate_headroom_sensitivity_table(raw_results: pd.DataFrame, output_path: str):
    """Generate headroom sensitivity table."""
    if 'headroom_factor' not in raw_results.columns:
        log.warning("No headroom_factor column in results")
        return None
    
    hr = raw_results[raw_results['headroom_factor'].notna() & (raw_results['headroom_factor'] != '')].copy()
    
    if len(hr) == 0:
        return None
    
    hr.to_csv(output_path, index=False)
    log.info(f"Generated headroom sensitivity CSV: {output_path}")
    return hr


def run_srlg_campaign(config: ExperimentConfig, srlgs: List, srlg_probs: List, 
                       failure_model: str = "SRLG") -> List[Dict[str, Any]]:
    """Run SRLG/IBF experimental campaign."""
    all_results = []
    
    # Main campaign: canonical traffic matrices
    for topology_name in config.topology_names:
        for budget in config.budgets:
            for baseline_name in config.baselines:
                if baseline_name not in SRLG_BASELINE_SELECTORS:
                    continue
                log.info(f"Running {topology_name} B={budget} {baseline_name} ({failure_model})")
                result = run_single_experiment_srlg(
                    topology_name, budget, baseline_name, config, replicate_idx=0,
                    srlgs=srlgs, srlg_probs=srlg_probs, failure_model=failure_model
                )
                all_results.append(result)
    
    # Traffic uncertainty campaign
    if config.traffic_uncertainty:
        for topology_name in config.topology_names:
            for budget in config.budgets:
                for baseline_name in ["HLP_2020", "C_TA_HLP"]:
                    if baseline_name not in SRLG_BASELINE_SELECTORS:
                        continue
                    for rep in range(config.n_perturbations):
                        pert_seed = config.seed + 10000 + rep * 100 + hash(topology_name) % 100
                        result = run_single_experiment_srlg(
                            topology_name, budget, baseline_name, config,
                            replicate_idx=rep,
                            traffic_perturbation_seed=pert_seed,
                            srlgs=srlgs, srlg_probs=srlg_probs, failure_model=failure_model
                        )
                        all_results.append(result)
    
    # Headroom sensitivity
    for topology_name in config.topology_names:
        for hf in config.headroom_factors:
            for budget in config.budgets:
                for baseline_name in ["HLP_2020", "Max_Traffic", "C_TA_HLP"]:
                    if baseline_name not in SRLG_BASELINE_SELECTORS:
                        continue
                    base_seed = config.seed + hash(topology_name) % 10000
                    graph, base_traffic = load_topology_with_traffic(
                        topology_name, seed=base_seed, headroom_factor=hf
                    )
                    if failure_model == "SRLG":
                        topology = build_srlg_topology_data(graph, base_traffic, topology_name, srlgs, srlg_probs, "SRLG")
                    else:
                        topology = build_srlg_topology_data(graph, base_traffic, topology_name, [], [], "IBF")
                    
                    selector = SRLG_BASELINE_SELECTORS[baseline_name]
                    selected_tunnels = selector(topology, budget)
                    
                    from src.evaluator_srlg import compute_estimands_srlg
                    estimand_results = compute_estimands_srlg(topology, selected_tunnels, ["unconditional"])
                    
                    result = {
                        "topology": topology_name,
                        "headroom_factor": hf,
                        "budget": budget,
                        "baseline": baseline_name,
                        "num_protected_links": len(selected_tunnels),
                        "traffic_total": sum(base_traffic.values()),
                        "failure_model": failure_model,
                    }
                    for estimand, sim_res in estimand_results.items():
                        prefix = f"{estimand}_"
                        result[f"{prefix}traffic_availability"] = sim_res.traffic_availability
                        result[f"{prefix}topological_availability"] = sim_res.topological_availability
                        result[f"{prefix}mean_mlu"] = sim_res.mean_max_link_utilization
                        result[f"{prefix}peak_mlu"] = sim_res.peak_max_link_utilization
                        result[f"{prefix}capacity_drops"] = sim_res.expected_capacity_drops
                        result[f"{prefix}dropped_volume"] = sim_res.expected_dropped_volume
                        result[f"{prefix}dropped_flows"] = sim_res.expected_dropped_flows
                    
                    all_results.append(result)
    
    return all_results


def run_srlg_campaign_per_topo(
    config: ExperimentConfig, 
    all_srlgs: Dict[str, List], 
    all_srlg_probs: Dict[str, List],
    failure_model: str = "SRLG"
) -> List[Dict[str, Any]]:
    """Run SRLG/IBF experimental campaign with per-topology SRLGs."""
    all_results = []
    
    # Main campaign: canonical traffic matrices
    for topology_name in config.topology_names:
        srlgs = all_srlgs.get(topology_name, [])
        srlg_probs = all_srlg_probs.get(topology_name, [])
        
        for budget in config.budgets:
            for baseline_name in config.baselines:
                if baseline_name not in SRLG_BASELINE_SELECTORS:
                    continue
                log.info(f"Running {topology_name} B={budget} {baseline_name} ({failure_model})")
                result = run_single_experiment_srlg(
                    topology_name, budget, baseline_name, config, replicate_idx=0,
                    srlgs=srlgs, srlg_probs=srlg_probs, failure_model=failure_model
                )
                all_results.append(result)
    
    # Traffic uncertainty campaign
    if config.traffic_uncertainty:
        for topology_name in config.topology_names:
            srlgs = all_srlgs.get(topology_name, [])
            srlg_probs = all_srlg_probs.get(topology_name, [])
            
            for budget in config.budgets:
                for baseline_name in ["HLP_2020", "C_TA_HLP"]:
                    if baseline_name not in SRLG_BASELINE_SELECTORS:
                        continue
                    for rep in range(config.n_perturbations):
                        pert_seed = config.seed + 10000 + rep * 100 + hash(topology_name) % 100
                        result = run_single_experiment_srlg(
                            topology_name, budget, baseline_name, config,
                            replicate_idx=rep,
                            traffic_perturbation_seed=pert_seed,
                            srlgs=srlgs, srlg_probs=srlg_probs, failure_model=failure_model
                        )
                        all_results.append(result)
    
    # Headroom sensitivity
    for topology_name in config.topology_names:
        srlgs = all_srlgs.get(topology_name, [])
        srlg_probs = all_srlg_probs.get(topology_name, [])
        
        for hf in config.headroom_factors:
            for budget in config.budgets:
                for baseline_name in ["HLP_2020", "Max_Traffic", "C_TA_HLP"]:
                    if baseline_name not in SRLG_BASELINE_SELECTORS:
                        continue
                    base_seed = config.seed + hash(topology_name) % 10000
                    graph, base_traffic = load_topology_with_traffic(
                        topology_name, seed=base_seed, headroom_factor=hf
                    )
                    if failure_model == "SRLG":
                        topology = build_srlg_topology_data(graph, base_traffic, topology_name, srlgs, srlg_probs, "SRLG")
                    else:
                        topology = build_srlg_topology_data(graph, base_traffic, topology_name, [], [], "IBF")
                    
                    selector = SRLG_BASELINE_SELECTORS[baseline_name]
                    selected_tunnels = selector(topology, budget)
                    
                    from src.evaluator_srlg import compute_estimands_srlg
                    estimand_results = compute_estimands_srlg(topology, selected_tunnels, ["unconditional"])
                    
                    result = {
                        "topology": topology_name,
                        "headroom_factor": hf,
                        "budget": budget,
                        "baseline": baseline_name,
                        "num_protected_links": len(selected_tunnels),
                        "traffic_total": sum(base_traffic.values()),
                        "failure_model": failure_model,
                    }
                    for estimand, sim_res in estimand_results.items():
                        prefix = f"{estimand}_"
                        result[f"{prefix}traffic_availability"] = sim_res.traffic_availability
                        result[f"{prefix}topological_availability"] = sim_res.topological_availability
                        result[f"{prefix}mean_mlu"] = sim_res.mean_max_link_utilization
                        result[f"{prefix}peak_mlu"] = sim_res.peak_max_link_utilization
                        result[f"{prefix}capacity_drops"] = sim_res.expected_capacity_drops
                        result[f"{prefix}dropped_volume"] = sim_res.expected_dropped_volume
                        result[f"{prefix}dropped_flows"] = sim_res.expected_dropped_flows
                    
                    all_results.append(result)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run unified C-TA-HLP evaluation campaign")
    parser.add_argument("--topologies", nargs="+", 
                        default=["abilene", "geant", "germany50", "nobel-eu"],
                        help="SNDlib topology names (without .json)")
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crn-seed", type=int, default=12345)
    parser.add_argument("--estimands", nargs="+", default=["unconditional", "conditional"])
    parser.add_argument("--baselines", nargs="+", 
                        default=["LFC_Only", "Random", "HLP_2020", "Max_Traffic", "C_TA_HLP", "Optimal_ILP"])
    parser.add_argument("--traffic-uncertainty", action="store_true", default=True)
    parser.add_argument("--n-perturbations", type=int, default=20)
    parser.add_argument("--perturbation-std", type=float, default=0.20)
    parser.add_argument("--headroom-factors", type=float, nargs="+", default=[1.1, 1.25, 1.5, 2.0])
    parser.add_argument("--out-dir", type=str, default="results/evaluator")
    parser.add_argument("--run-validation", action="store_true", default=True)
    parser.add_argument("--skip-main", action="store_true", help="Skip main campaign, only generate tables")
    parser.add_argument("--failure-model", choices=["SLF", "SRLG", "IBF"], default="SLF",
                        help="Failure model to use")
    parser.add_argument("--srlg-file", type=str, default=None,
                        help="JSON file with SRLG definitions (list of edge sets)")
    args = parser.parse_args()
    
    config = ExperimentConfig(
        topology_names=args.topologies,
        budgets=args.budgets,
        num_trials=args.trials,
        n_replicates=args.replicates,
        estimands=args.estimands,
        baselines=args.baselines,
        traffic_uncertainty=args.traffic_uncertainty,
        n_perturbations=args.n_perturbations,
        perturbation_std=args.perturbation_std,
        headroom_factors=args.headroom_factors,
        seed=args.seed,
        crn_seed=args.crn_seed,
    )
    
    ensure_dir(args.out_dir)
    
    raw_path = os.path.join(args.out_dir, "raw_results.csv")
    
    if args.skip_main and os.path.exists(raw_path):
        log.info(f"Loading existing results from {raw_path}")
        raw_results = pd.read_csv(raw_path)
    else:
        if args.run_validation:
            log.info("Running validation tests...")
            import subprocess
            if args.failure_model == "SLF":
                result = subprocess.run([sys.executable, "-m", "tests.test_evaluator"], 
                                        capture_output=True, text=True, cwd=ROOT)
            else:
                result = subprocess.run([sys.executable, "-m", "tests.test_evaluator_srlg"], 
                                        capture_output=True, text=True, cwd=ROOT)
            if result.returncode != 0:
                log.error("Validation tests failed!")
                print(result.stdout)
                print(result.stderr)
                return 1
            log.info("Validation tests passed")
        
        log.info(f"Starting full campaign with {args.failure_model} model...")
        start_time = time.time()
        
        if args.failure_model == "SLF":
            results = run_full_campaign(config)
        else:
            # Load or generate SRLG definitions
            srlgs = []
            srlg_probs = []
            if args.srlg_file and os.path.exists(args.srlg_file):
                with open(args.srlg_file) as f:
                    srlg_data = json.load(f)
                    srlgs = [set(tuple(e) for e in s) for s in srlg_data["srlgs"]]
                    srlg_probs = srlg_data.get("probs", [0.01] * len(srlgs))
            else:
                # Generate synthetic SRLGs for each topology
                log.info("Generating synthetic SRLGs for each topology...")
                all_srlgs = {}
                all_srlg_probs = {}
                for topo_name in config.topology_names:
                    base_seed = config.seed + hash(topo_name) % 10000
                    graph, _ = load_topology_with_traffic(topo_name, seed=base_seed)
                    srlgs_topo, probs_topo = generate_synthetic_srlgs(
                        graph, num_srlgs=5, max_srlg_size=3, seed=config.seed
                    )
                    all_srlgs[topo_name] = srlgs_topo
                    all_srlg_probs[topo_name] = probs_topo
                
                # For campaign, we'll pass per-topology SRLGs to the runner
                results = run_srlg_campaign_per_topo(config, all_srlgs, all_srlg_probs, args.failure_model)
        
        df = pd.DataFrame(results)
        df.to_csv(raw_path, index=False)
        log.info(f"Saved raw results to {raw_path} ({len(df)} rows)")
        log.info(f"Campaign completed in {time.time() - start_time:.1f}s")
        
        # Save config
        config_path = os.path.join(args.out_dir, "experiment_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "topologies": config.topology_names,
                "budgets": config.budgets,
                "estimands": config.estimands,
                "baselines": config.baselines,
                "traffic_uncertainty": config.traffic_uncertainty,
                "n_perturbations": config.n_perturbations,
                "perturbation_std": config.perturbation_std,
                "headroom_factors": config.headroom_factors,
                "seed": config.seed,
                "crn_seed": config.crn_seed,
                "failure_model": args.failure_model,
            }, f, indent=2)
        
        raw_results = df
    
    # Generate all tables
    log.info("Generating tables...")
    
    generate_statistical_table(raw_results, os.path.join(args.out_dir, "statistical_table.tex"))
    generate_baseline_comparison_table(raw_results, os.path.join(args.out_dir, "baseline_comparison.csv"))
    
    if config.traffic_uncertainty:
        generate_traffic_uncertainty_stats(raw_results, os.path.join(args.out_dir, "traffic_uncertainty_stats.csv"))
    
    if config.headroom_factors:
        generate_headroom_sensitivity_table(raw_results, os.path.join(args.out_dir, "headroom_sensitivity.csv"))
    
    # Print summary
    print("\n" + "="*80)
    print("CAMPAIGN SUMMARY")
    print("="*80)
    
    canonical = raw_results[
        (raw_results['replicate'] == 0) & 
        (raw_results['headroom_factor'].isna() | (raw_results['headroom_factor'] == ''))
    ]
    
    for topo in args.topologies:
        for budget in args.budgets:
            subset = canonical[(canonical['topology'] == topo) & (canonical['budget'] == budget)]
            if len(subset) == 0:
                continue
            
            for model in subset.get('failure_model', pd.Series(['SLF'])).unique():
                model_subset = subset[subset.get('failure_model', 'SLF') == model]
                row_hlp = model_subset[model_subset['baseline'] == 'HLP_2020']
                row_cta = model_subset[model_subset['baseline'] == 'C_TA_HLP']
                
                if len(row_hlp) > 0 and len(row_cta) > 0:
                    hlp_ta = row_hlp['unconditional_traffic_availability'].values[0]
                    cta_ta = row_cta['unconditional_traffic_availability'].values[0]
                    delta = cta_ta - hlp_ta
                    print(f"  {topo:12s} B={budget} {model}: HLP={hlp_ta:.6f}  C-TA-HLP={cta_ta:.6f}  delta={delta:+.6f}")
    
    print("="*80)
    log.info("All done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())