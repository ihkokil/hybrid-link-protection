"""C-TA-BPP -- Capacity-Constrained Traffic-Adaptive Backup Path Protection.

The selection skeleton mirrors ``bpp.bpp``; the marginal-gain criterion
incorporates both traffic demand weights and a downstream congestion penalty:

    Δ_{C-T}(e) = Φ(e) * Σ_{(s,d) ∈ e.P} T(s,d) * p_e * A(P_repair(e))

where ``P_repair(e)`` starts at the failure-detecting node (FDN) and
``Φ(e) = Π_{k ∈ P_repair(e)} min(1.0, C_k / L_post(k))``.

Capacity accounting is deliberately explicit: ``L_post(k)`` is the baseline
load plus the *expected* transient load from previously selected tunnels and
the candidate tunnel (``p_e * shifted volume``).  This is a reservation
heuristic, not a simultaneous multi-failure model.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from .availability import path_availability
from .bpp import (
    build_edge_to_sd_index,
    find_key_links,
    _walk_pred,
)
from .path_construction import build_primary_paths, detect_failure_point
from .shortest_path import dijkstra
from .utils import normalize_edge


def compute_baseline_loads(
    graph: nx.Graph,
    paths: Dict[Tuple[int, int], List],
    traffic: Dict[Tuple[int, int], float],
) -> Dict[Tuple[int, int], float]:
    """Compute baseline shortest-path traffic load L_0(k) for all edges."""
    loads: Dict[Tuple[int, int], float] = {normalize_edge(*e): 0.0 for e in graph.edges()}
    for (s, d), path in paths.items():
        vol = traffic.get((s, d), 0.0)
        if vol <= 0.0:
            continue
        for i in range(len(path) - 1):
            e_norm = normalize_edge(path[i], path[i + 1])
            if e_norm in loads:
                loads[e_norm] += vol
    return loads


def compute_delta_T(
    graph: nx.Graph,
    e: Tuple[int, int],
    sd_pairs: Set[Tuple[int, int]],
    paths: Dict[Tuple[int, int], List],
    traffic: Dict[Tuple[int, int], float],
    baseline_loads: Optional[Dict[Tuple[int, int], float]] = None,
    capacity_constrained: bool = False,
    accumulated_detour_loads: Optional[Dict[Tuple[int, int], float]] = None,
    return_shifts: bool = False,
) -> float | Tuple[float, Dict[Tuple[int, int], float]]:
    """Traffic-weighted marginal contribution of protecting link ``e``.

    Repair paths are computed from the failure-detecting node to each
    destination.  Only the repair-path edges carry *additional* transient
    load; the prefix before the failed link is already present in the
    baseline load.
    """
    by_detector: Dict = {}
    for (s, d) in sd_pairs:
        path = paths.get((s, d), [])
        idx = detect_failure_point(path, e)
        if idx < 0:
            continue
        by_detector.setdefault(path[idx], []).append((s, d))

    H = graph.copy()
    a, b = normalize_edge(*e)
    if H.has_edge(a, b):
        H.remove_edge(a, b)

    delta = 0.0
    detour_shifts: Dict[Tuple[int, int], float] = {}
    p_fail_e = float(graph[a][b].get("prob_fail", 0.01))

    for detector, flows in by_detector.items():
        dist, pred = dijkstra(H, detector)
        for s, d in flows:
            if dist.get(d, float("inf")) == float("inf"):
                continue
            new_path = _walk_pred(detector, d, pred)
            if not new_path:
                continue
            flow_vol = float(traffic.get((s, d), 0.0))
            if flow_vol <= 0.0:
                continue
            a_repair = path_availability(H, new_path)
            delta += flow_vol * p_fail_e * a_repair

            if capacity_constrained and baseline_loads is not None:
                for i in range(len(new_path) - 1):
                    k_edge = normalize_edge(new_path[i], new_path[i + 1])
                    detour_shifts[k_edge] = detour_shifts.get(k_edge, 0.0) + flow_vol

    # Phi(e) = product over repair-path links of min(1, C_k / L_post(k)).
    if capacity_constrained and baseline_loads is not None and detour_shifts:
        phi = 1.0
        for k_edge, shift in detour_shifts.items():
            u_k, v_k = k_edge
            if not graph.has_edge(u_k, v_k):
                continue
            cap = graph[u_k][v_k].get("capacity", None)
            if cap is None or cap <= 0:
                continue
            prior_acc = (
                accumulated_detour_loads.get(k_edge, 0.0)
                if accumulated_detour_loads is not None
                else 0.0
            )
            # The candidate's own detour is active whenever e fails, so use
            # its full shifted volume.  Prior tunnels are assumed to be
            # active concurrently only in expectation, so their accumulated
            # contribution is already probability-weighted.
            post_load = baseline_loads.get(k_edge, 0.0) + prior_acc + shift
            if post_load > cap:
                phi *= cap / post_load
        delta *= phi

    if return_shifts:
        return delta, detour_shifts
    return delta


def ta_bpp(
    graph: nx.Graph,
    next_hops_per_c: Dict[int, Dict[int, Set]],
    traffic: Dict[Tuple[int, int], float],
    max_links: Optional[int] = None,
    R: Optional[float] = None,
    capacity_constrained: bool = False,
    iterative: bool = True,
    verbose: bool = False,
) -> List[Tuple[int, int]]:
    """C-TA-BPP: capacity-constrained traffic-adaptive key-link selection.

    Parameters
    ----------
    graph : nx.Graph
    next_hops_per_c : dict
        Output of the MNP-e reference (or LFC).
    traffic : dict
        ``T[(s,d)]`` from empirical demand data or a gravity model.
    max_links : int or None
        Backup-link budget ``B``. Hard cap on protected links.
    R : float or None
        Optional topological-availability requirement.  The threshold is
        checked after adding a candidate, so the link that reaches ``R`` is
        included.
    capacity_constrained : bool
        Apply the Congestion Penalty Factor Phi(e).
    iterative : bool
        Re-evaluate marginal gains and cumulative detour reservations after
        each selection (default).  ``False`` is a static single-pass
        ablation.
    """
    paths = build_primary_paths(graph, next_hops_per_c)
    edge_index = build_edge_to_sd_index(paths)
    key_links = find_key_links(graph, next_hops_per_c, edge_index=edge_index)
    baseline_loads = compute_baseline_loads(graph, paths, traffic) if capacity_constrained else None

    if verbose:
        print(f"[C-TA-BPP] {len(key_links)} key links detected (iterative={iterative})")

    if not key_links:
        return []

    protected: List[Tuple[int, int]] = []
    deltas: Dict[Tuple[int, int], float] = {}

    if not iterative:
        # Static single-pass ranking (ablation mode).
        for e in sorted(key_links):
            deltas[e] = compute_delta_T(
                graph, e, key_links[e], paths, traffic,
                baseline_loads=baseline_loads,
                capacity_constrained=capacity_constrained,
            )
        sorted_keys = sorted(deltas, key=lambda e: (-deltas[e], e))
        for e in sorted_keys:
            if max_links is not None and len(protected) >= max_links:
                break
            protected.append(e)
            if R is not None and _approx_A_meets(graph, protected, paths, R, deltas):
                break
    else:
        # Iterative greedy: re-evaluate every remaining candidate each round.
        remaining_keys = sorted(key_links)
        accumulated_detour_loads: Dict[Tuple[int, int], float] = {
            normalize_edge(*e): 0.0 for e in graph.edges()
        }
        target_budget = max_links if max_links is not None else len(key_links)

        while remaining_keys and len(protected) < target_budget:
            best_e = None
            best_delta = float("-inf")
            best_shifts: Dict[Tuple[int, int], float] = {}

            for e in remaining_keys:
                delta_e, shifts_e = compute_delta_T(
                    graph, e, key_links[e], paths, traffic,
                    baseline_loads=baseline_loads,
                    capacity_constrained=capacity_constrained,
                    accumulated_detour_loads=accumulated_detour_loads,
                    return_shifts=True,
                )
                if (
                    delta_e > best_delta + 1e-15
                    or (abs(delta_e - best_delta) <= 1e-15 and (best_e is None or e < best_e))
                ):
                    best_delta = delta_e
                    best_e = e
                    best_shifts = shifts_e

            if best_e is None or best_delta <= 0.0:
                break

            deltas[best_e] = best_delta
            protected.append(best_e)
            remaining_keys.remove(best_e)

            # Reserve expected transient load for the next marginal-gain pass.
            a_e, b_e = best_e
            p_fail = float(graph[a_e][b_e].get("prob_fail", 0.01))
            for k_edge, shift_vol in best_shifts.items():
                accumulated_detour_loads[k_edge] = (
                    accumulated_detour_loads.get(k_edge, 0.0) + p_fail * shift_vol
                )

            if R is not None and _approx_A_meets(graph, protected, paths, R, deltas):
                break

    if verbose:
        print(f"[TA-BPP] Protected {len(protected)} links")
    return protected


def _approx_A_meets(
    graph: nx.Graph,
    protected: List[Tuple[int, int]],
    paths: Dict[Tuple[int, int], List],
    R: float,
    deltas: Dict[Tuple[int, int], float],
) -> bool:
    """Check the approximate availability threshold after a selection."""
    n = graph.number_of_nodes()
    norm = n * (n - 1) if n > 1 else 1
    base = sum(path_availability(graph, p) for p in paths.values()) / norm
    accumulated = sum(deltas.get(e, 0.0) for e in protected) / norm
    return (base + accumulated) >= R
