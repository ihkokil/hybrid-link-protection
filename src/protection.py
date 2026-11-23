"""Compose the MNP-e reference with BPP / C-TA-BPP.

Stage 1 is output-equivalent to the base paper's MNP-e (Theorem 8) but is
implemented through the LFC condition, not i-SPF.  Stage 2 is the budgeted
key-link selection.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from .bpp import bpp, build_edge_to_sd_index, find_key_links
from .lfc import compute_lfc
from .mnp_e import compute_mnp_e_reference
from .path_construction import build_primary_paths
from .ta_bpp import compute_baseline_loads, compute_delta_T, ta_bpp
from .utils import normalize_edge


@dataclass
class ProtectionResult:
    """Holds everything a scheme produces about a topology."""

    method: str
    next_hops: Dict[int, Dict[int, Set]] = field(default_factory=dict)
    protected_links: List[Tuple[int, int]] = field(default_factory=list)
    compute_time_s: float = 0.0
    extra: Dict = field(default_factory=dict)


def compute_lfc_scheme(graph: nx.Graph) -> ProtectionResult:
    """Just LFC -- no BPP tunnel protection. Baseline for our experiments."""
    t0 = time.perf_counter()
    next_hops = {c: compute_lfc(graph, c) for c in graph.nodes()}
    t1 = time.perf_counter()
    return ProtectionResult(
        method="LFC",
        next_hops=next_hops,
        protected_links=[],
        compute_time_s=t1 - t0,
    )


def compute_hlp(
    graph: nx.Graph,
    R: float = 0.999,
    max_links: Optional[int] = None,
) -> ProtectionResult:
    """HLP-style scheme = MNP-e reference + BPP."""
    t0 = time.perf_counter()
    next_hops = {c: compute_mnp_e_reference(graph, c) for c in graph.nodes()}
    protected = bpp(graph, next_hops, R=R, max_links=max_links, verbose=False)
    t1 = time.perf_counter()
    return ProtectionResult(
        method="HLP",
        next_hops=next_hops,
        protected_links=protected,
        compute_time_s=t1 - t0,
    )


def compute_ta_hlp(
    graph: nx.Graph,
    traffic: Dict[Tuple[int, int], float],
    R: Optional[float] = None,
    max_links: Optional[int] = None,
    capacity_constrained: bool = True,
    iterative: bool = True,
) -> ProtectionResult:
    """C-TA-HLP = MNP-e reference + C-TA-BPP."""
    t0 = time.perf_counter()
    next_hops = {c: compute_mnp_e_reference(graph, c) for c in graph.nodes()}
    protected = ta_bpp(
        graph,
        next_hops,
        traffic,
        max_links=max_links,
        R=R,
        capacity_constrained=capacity_constrained,
        iterative=iterative,
        verbose=False,
    )
    t1 = time.perf_counter()
    return ProtectionResult(
        method="TA-HLP",
        next_hops=next_hops,
        protected_links=protected,
        compute_time_s=t1 - t0,
        extra={"traffic_total": sum(traffic.values())},
    )


def compute_max_traffic_hlp(
    graph: nx.Graph,
    traffic: Dict[Tuple[int, int], float],
    max_links: Optional[int] = None,
) -> ProtectionResult:
    """Max-Traffic Baseline: Protect key links carrying highest volume of traffic.
    Does not enforce downstream capacity derating (capacity_constrained=False, iterative=False)."""
    t0 = time.perf_counter()
    next_hops = {c: compute_mnp_e_reference(graph, c) for c in graph.nodes()}
    protected = ta_bpp(
        graph,
        next_hops,
        traffic,
        max_links=max_links,
        R=None,
        capacity_constrained=False,
        iterative=False,
        verbose=False,
    )
    t1 = time.perf_counter()
    return ProtectionResult(
        method="Max-Traffic",
        next_hops=next_hops,
        protected_links=protected,
        compute_time_s=t1 - t0,
    )


def compute_random_hlp(
    graph: nx.Graph,
    max_links: Optional[int] = None,
    seed: int = 42,
) -> ProtectionResult:
    """Random Baseline: Randomly select key links up to the budget."""
    t0 = time.perf_counter()
    next_hops = {c: compute_mnp_e_reference(graph, c) for c in graph.nodes()}
    paths = build_primary_paths(graph, next_hops)
    edge_index = build_edge_to_sd_index(paths)
    key_links_dict = find_key_links(graph, next_hops, edge_index=edge_index)
    key_links = sorted(key_links_dict.keys())
    rng = random.Random(seed)
    k = min(len(key_links), max_links if max_links is not None else len(key_links))
    protected = sorted(rng.sample(key_links, k)) if k > 0 else []
    t1 = time.perf_counter()
    return ProtectionResult(
        method="Random",
        next_hops=next_hops,
        protected_links=protected,
        compute_time_s=t1 - t0,
    )


def compute_optimal_milp(
    graph: nx.Graph,
    traffic: Dict[Tuple[int, int], float],
    max_links: Optional[int] = None,
    capacity_constrained: bool = True,
) -> ProtectionResult:
    """Exact 0-1 MIP solver using SciPy HiGHS.
    Maximizes linear traffic availability gain subject to cardinality and expected capacity bounds."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    import numpy as np

    t0 = time.perf_counter()
    next_hops = {c: compute_mnp_e_reference(graph, c) for c in graph.nodes()}
    paths = build_primary_paths(graph, next_hops)
    edge_index = build_edge_to_sd_index(paths)
    key_links_dict = find_key_links(graph, next_hops, edge_index=edge_index)
    key_links = sorted(key_links_dict.keys())

    if not key_links or max_links == 0:
        t1 = time.perf_counter()
        return ProtectionResult(method="Optimal-ILP", next_hops=next_hops, protected_links=[], compute_time_s=t1 - t0)

    n_k = len(key_links)
    target_budget = min(n_k, max_links if max_links is not None else n_k)
    baseline_loads = compute_baseline_loads(graph, paths, traffic)

    gains = np.zeros(n_k)
    edges_list = sorted(list(graph.edges()))
    n_e = len(edges_list)

    if capacity_constrained:
        A_cap = np.zeros((n_e, n_k))
        rhs_cap = np.zeros(n_e)
        for j, (u, v) in enumerate(edges_list):
            e_norm = normalize_edge(u, v)
            cap = graph[u][v].get("capacity", 1e9)
            base = baseline_loads.get(e_norm, 0.0)
            rhs_cap[j] = max(0.0, cap - base)

    for i, e in enumerate(key_links):
        if capacity_constrained:
            gain, shifts = compute_delta_T(
                graph,
                e,
                key_links_dict[e],
                paths,
                traffic,
                baseline_loads=baseline_loads,
                capacity_constrained=False,
                return_shifts=True,
            )
            p_e = graph[e[0]][e[1]].get("prob_fail", 0.01)
            for j, (u, v) in enumerate(edges_list):
                e_norm = normalize_edge(u, v)
                if e_norm in shifts:
                    A_cap[j, i] = p_e * shifts[e_norm]
        else:
            gain = compute_delta_T(
                graph,
                e,
                key_links_dict[e],
                paths,
                traffic,
                baseline_loads=baseline_loads,
                capacity_constrained=False,
            )
        gains[i] = gain

    # Budget constraint
    A_budget = np.ones((1, n_k))
    rhs_budget = np.array([target_budget])

    if capacity_constrained:
        A = np.vstack([A_budget, A_cap])
        lb = np.zeros(1 + n_e)
        ub = np.concatenate([rhs_budget, rhs_cap])
    else:
        A = A_budget
        lb = np.zeros(1)
        ub = rhs_budget

    res = milp(
        c=-gains,
        integrality=np.ones(n_k),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(A, lb, ub),
    )

    if res.success:
        protected = [key_links[i] for i in range(n_k) if res.x[i] > 0.5]
    else:
        # Fallback to sorting by gains if problem is infeasible under tight bounds
        idx_sorted = np.argsort(-gains)[:target_budget]
        protected = [key_links[i] for i in idx_sorted]

    t1 = time.perf_counter()
    return ProtectionResult(
        method="Optimal-ILP",
        next_hops=next_hops,
        protected_links=protected,
        compute_time_s=t1 - t0,
        extra={"ilp_status": int(res.status), "ilp_obj": -float(res.fun) if res.success else 0.0},
    )
