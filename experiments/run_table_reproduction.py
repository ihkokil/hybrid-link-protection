"""Base-paper comparison diagnostics -- NOT a timing reproduction.

Two honest diagnostics are produced:

``table3_compute_time.csv``
    Per-node wall-clock time for the implemented methods.  The ``MNP-e(ref)``
    column is an output-equivalent reference implemented via the LFC
    condition; it does **not** use i-SPF and therefore cannot reproduce the
    base paper's MNP-e timing advantage.  Treat the OSPF/LFC/TBFH/MNP/DMPA-e
    columns as implementation diagnostics, not as the paper's Table 3.

``table4_single_failure_delivery.csv``
    Fraction of source-destination pairs delivered under exactly one sampled
    link failure.  This is a conditional delivery proxy, not the analytic
    availability ``A(G)`` reported in the base paper's Table 4.  The two
    quantities are not comparable.

The previous filenames (``table4_availability.csv``) and the "reproduction"
wording were removed because they overstated what the code computes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, Dict, List

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.topology import NetworkTopology
from src.shortest_path import dijkstra
from src.lfc import compute_lfc
from src.mnp_e import compute_mnp_e_reference
from src.tbfh import compute_tbfh
from src.dmpa_e import compute_dmpa_e
from src.simulator import simulate_failures
from src.utils import ensure_dir, get_logger

log = get_logger("table3")


# ---------------------------------------------------------------- algorithm map
def compute_ospf(graph, c):
    """OSPF: one SPT, primary next-hops only (no backup)."""
    dist_c, pred_c = dijkstra(graph, c)
    nh = {v: set() for v in graph.nodes() if v != c}
    for v in nh:
        cur = v
        while cur != c and pred_c[cur] is not None:
            if pred_c[cur] == c:
                nh[v].add(cur)
                break
            cur = pred_c[cur]
    return nh


def compute_mnp(graph, c):
    """MNP diagnostic (paper Ref [36]) -- direct tree-neighbour restriction."""
    dist_c, pred_c = dijkstra(graph, c)
    nh = {v: set() for v in graph.nodes() if v != c}
    for v in nh:
        cur = v
        while cur != c and pred_c[cur] is not None:
            if pred_c[cur] == c:
                nh[v].add(cur)
                break
            cur = pred_c[cur]
    for v in nh:
        for u in graph.neighbors(v):
            if u == c or u == v:
                continue
            if pred_c[u] is None or pred_c[v] is None:
                continue
            cost_vu = graph[v][u].get("cost", 1.0)
            d = dist_c[u] - dist_c.get(pred_c[u], 0) + cost_vu
            if d < dist_c[v] + cost_vu:
                primary_u = next(iter(nh[u])) if nh[u] else u
                if primary_u != c:
                    nh[v].add(primary_u)
    return nh


METHODS: Dict[str, Callable] = {
    "OSPF": compute_ospf,
    "LFC": compute_lfc,
    "TBFH": compute_tbfh,
    "MNP": compute_mnp,
    "DMPA-e": compute_dmpa_e,
    "MNP-e(ref)": compute_mnp_e_reference,
}


def _per_node_us(graph, fn, repeat: int = 5) -> float:
    t0 = time.perf_counter()
    for _ in range(repeat):
        for c in graph.nodes():
            fn(graph, c)
    elapsed = (time.perf_counter() - t0) / (repeat * graph.number_of_nodes())
    return elapsed * 1e6


def _load_topology(name: str) -> NetworkTopology:
    if name == "Abilene":
        path = os.path.join(ROOT, "data", "gml", "Abilene.gml")
        if os.path.exists(path):
            return NetworkTopology.from_gml(path)
    from src.rocketfuel import load_rocketfuel
    G = load_rocketfuel(name)
    return NetworkTopology(graph=G, name=name)


def run_table3(topologies: List[str], repeat: int, out_dir: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for topo_name in topologies:
        topo = _load_topology(topo_name)
        G = topo.graph
        log.info("Topology %s: V=%d E=%d", topo_name, G.number_of_nodes(),
                 G.number_of_edges())
        row = {"topology": topo_name,
               "|V|": G.number_of_nodes(),
               "|E|": G.number_of_edges()}
        for method_name, fn in METHODS.items():
            t_us = _per_node_us(G, fn, repeat=repeat)
            row[method_name] = round(t_us, 2)
            log.info("    %-10s = %7.2f us", method_name, t_us)
        rows.append(row)
    df = pd.DataFrame(rows)
    ensure_dir(out_dir)
    out = os.path.join(out_dir, "table3_compute_time.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)
    print("\nPer-node compute time (us) -- implementation diagnostics, not the paper's Table 3")
    print("MNP-e(ref) is output-equivalent to MNP-e but uses the LFC condition, not i-SPF.")
    print(df.to_string(index=False))
    return df


def run_table4(topologies: List[str], num_trials: int, seed: int,
               out_dir: str) -> pd.DataFrame:
    """Conditional single-failure delivery proxy (not analytic availability)."""
    rows: List[Dict] = []
    for topo_name in topologies:
        topo = _load_topology(topo_name)
        G = topo.graph
        row = {"topology": topo_name,
               "|V|": G.number_of_nodes(),
               "|E|": G.number_of_edges()}
        for method_name, fn in METHODS.items():
            nh = {c: fn(G, c) for c in G.nodes()}
            out = simulate_failures(
                G, nh, [], traffic=None,
                num_trials=num_trials, seed=seed, estimand="conditional",
            )
            row[method_name] = round(out.topological_availability * 100, 2)
            log.info("    %-10s  delivery|failure = %6.2f%%", method_name, row[method_name])
        rows.append(row)
    df = pd.DataFrame(rows)
    ensure_dir(out_dir)
    out = os.path.join(out_dir, "table4_single_failure_delivery.csv")
    df.to_csv(out, index=False)
    log.info("Saved -> %s", out)
    print("\nSingle-link-failure delivery proxy (%) -- NOT the paper's analytic A(G)")
    print(df.to_string(index=False))
    return df


def cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topologies", nargs="+",
                    default=["Abilene", "Exodus", "Telstra", "Tiscali", "Sprint315"])
    ap.add_argument("--repeat", type=int, default=5,
                    help="Repetitions for the timing measurement")
    ap.add_argument("--trials", type=int, default=300,
                    help="Monte Carlo trials for the conditional delivery proxy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "results"))
    ap.add_argument("--only", choices=["table3", "table4", "both"], default="both")
    args = ap.parse_args()

    if args.only in ("table3", "both"):
        run_table3(args.topologies, args.repeat, args.out)
    if args.only in ("table4", "both"):
        run_table4(args.topologies, args.trials, args.seed, args.out)


if __name__ == "__main__":
    cli()
