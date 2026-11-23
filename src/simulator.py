"""Monte-Carlo failure simulator.

Two estimands are supported:

``estimand="unconditional"`` (default, availability semantics)
    Every link is independently operational with probability ``1 - p_e``.
    Trials with zero failures are included.  The reported values estimate
    the unconditional pair/traffic availability used in the paper's
    definitions.

``estimand="conditional"`` (legacy one-failure analysis)
    Exactly one link is drawn per trial, either uniformly or proportionally
    to its failure probability.  The reported values are the fraction of
    pairs/traffic delivered *given* that one link failed, not availability.

For each trial the resolver walks the primary FIB path from the source.  At
the first failed edge it tries the MNP/LFC alternate next-hops; if none can
reach the destination it falls back to a precomputed BPP tunnel when that
edge is protected.  The resolver is recursive, so a recovery path that hits
a second independent failure is handled the same way.

Capacity enforcement accumulates post-failure loads from all delivered
flows and applies a bottleneck (or multiplicative) survival ratio.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from .path_construction import build_primary_paths
from .shortest_path import dijkstra, reconstruct_path
from .utils import normalize_edge


@dataclass
class SimulationOutcome:
    topological_availability: float = 0.0
    traffic_availability: float = 0.0
    packet_loss_proxy: float = 0.0
    recovery_count: int = 0
    bpp_recovery_count: int = 0
    mnp_recovery_count: int = 0
    capacity_drop_count: int = 0
    max_link_utilization: float = 0.0
    num_trials: int = 0
    estimand: str = "unconditional"
    mean_failed_links: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "topological_availability": self.topological_availability,
            "traffic_availability": self.traffic_availability,
            "packet_loss_proxy": self.packet_loss_proxy,
            "recovery_count": self.recovery_count,
            "bpp_recovery_count": self.bpp_recovery_count,
            "mnp_recovery_count": self.mnp_recovery_count,
            "capacity_drop_count": self.capacity_drop_count,
            "max_link_utilization": self.max_link_utilization,
            "num_trials": self.num_trials,
            "estimand": self.estimand,
            "mean_failed_links": self.mean_failed_links,
        }


def _protected_set(protected_links: Sequence[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    return {normalize_edge(*e) for e in protected_links}


def _resolve_flow(
    source,
    destination,
    paths: Dict[Tuple[int, int], List],
    next_hops_per_c: Dict[int, Dict[int, Set]],
    protected: Set[Tuple[int, int]],
    failed_edges: Set[Tuple[int, int]],
    repair_graph: nx.Graph,
    visited: Optional[Set] = None,
    depth: int = 0,
) -> Tuple[Optional[List[Tuple[int, int]]], Set[str]]:
    """Resolve one flow, returning ``(edge_list, recovery_modes)``.

    ``edge_list`` is ``None`` if the flow cannot be delivered.  The edge list
    includes the traversed prefix, any alternate hop-by-hop segment, and any
    BPP repair segment.
    """
    if depth > 64:
        return None, set()
    if visited is None:
        visited = {source}

    path = paths.get((source, destination), [])
    if len(path) < 2:
        return ([ ] if source == destination else None), set()

    traversed: List[Tuple[int, int]] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = normalize_edge(u, v)
        if edge in failed_edges:
            # Stage 1: try every MNP/LFC alternate at the detecting node.
            candidates = sorted(
                h for h in next_hops_per_c.get(u, {}).get(destination, set())
                if normalize_edge(u, h) not in failed_edges
            )
            for alt in candidates:
                if alt in visited:
                    continue
                sub_edges, sub_modes = _resolve_flow(
                    alt, destination, paths, next_hops_per_c,
                    protected, failed_edges, repair_graph,
                    visited | {alt}, depth + 1,
                )
                if sub_edges is not None:
                    return (
                        traversed + [normalize_edge(u, alt)] + sub_edges,
                        sub_modes | {"mnp"},
                    )

            # Stage 2: fall back to a BPP tunnel if this edge is protected.
            if edge in protected:
                try:
                    dist, pred = dijkstra(repair_graph, u)
                    if dist.get(destination, float("inf")) != float("inf"):
                        repair_path = reconstruct_path(pred, u, destination)
                        if len(repair_path) >= 2:
                            repair_edges = [
                                normalize_edge(repair_path[j], repair_path[j + 1])
                                for j in range(len(repair_path) - 1)
                            ]
                            return traversed + repair_edges, {"bpp"}
                except nx.NetworkXNoPath:
                    pass
            return None, set()

        traversed.append(edge)

    if source == destination:
        return [], set()
    return traversed, set()


def simulate_failures(
    graph: nx.Graph,
    next_hops_per_c: Dict[int, Dict[int, Set]],
    protected_links: Sequence[Tuple[int, int]],
    traffic: Optional[Dict[Tuple[int, int], float]] = None,
    num_trials: int = 300,
    seed: Optional[int] = None,
    failure_sampling: str = "probability_weighted",
    estimand: str = "unconditional",
    enforce_capacity: bool = True,
    drop_model: str = "bottleneck",
) -> SimulationOutcome:
    """Run the Monte-Carlo failure experiment.

    Parameters
    ----------
    failure_sampling
        ``'probability_weighted'`` or ``'uniform'``; used only when
        ``estimand='conditional'``.
    estimand
        ``'unconditional'`` samples independent link states; ``'conditional'``
        samples exactly one failed link and reports delivery given failure.
    enforce_capacity
        Apply capacity drops when edge capacities are present.
    drop_model
        ``'bottleneck'`` (minimum survival ratio) or ``'multiplicative'``.
    """
    if failure_sampling not in {"uniform", "probability_weighted"}:
        raise ValueError("failure_sampling must be 'uniform' or 'probability_weighted'")
    if estimand not in {"unconditional", "conditional"}:
        raise ValueError("estimand must be 'unconditional' or 'conditional'")
    if drop_model not in {"bottleneck", "multiplicative"}:
        raise ValueError("drop_model must be 'bottleneck' or 'multiplicative'")
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1")

    rng = random.Random(seed)
    edges = list(graph.edges())
    if not edges:
        return SimulationOutcome(estimand=estimand)

    paths = build_primary_paths(graph, next_hops_per_c)
    protected = _protected_set(protected_links)

    has_capacities = any(
        "capacity" in graph[u][v] and graph[u][v]["capacity"] is not None
        for u, v in edges
    )
    should_enforce_capacity = enforce_capacity and has_capacities

    total_traffic = sum(traffic.values()) if traffic else 0.0
    edge_weights = [max(0.0, float(graph[u][v].get("prob_fail", 0.0))) for u, v in edges]
    nodes = list(graph.nodes())

    total_attempts = 0
    successful_deliveries = 0
    traffic_delivered = 0.0
    recovery_count = 0
    bpp_recovery = 0
    mnp_recovery = 0
    capacity_drop_count = 0
    peak_mlu = 0.0
    total_failed_links = 0

    def draw_conditional_edge():
        if failure_sampling == "probability_weighted" and sum(edge_weights) > 0:
            draw = rng.random() * sum(edge_weights)
            running = 0.0
            for edge, weight in zip(edges, edge_weights):
                running += weight
                if draw <= running:
                    return edge
        return edges[rng.randrange(len(edges))]

    for _ in range(num_trials):
        if estimand == "conditional":
            failed_edges = {normalize_edge(*draw_conditional_edge())}
        else:
            failed_edges = {
                normalize_edge(u, v)
                for (u, v), p in zip(edges, edge_weights)
                if p > 0.0 and rng.random() < p
            }
        total_failed_links += len(failed_edges)

        repair_graph = graph.copy()
        for edge in failed_edges:
            if repair_graph.has_edge(*edge):
                repair_graph.remove_edge(*edge)

        resolved_flows: List[Tuple[Tuple[int, int], str, float, List[Tuple[int, int]]]] = []
        post_link_loads: Dict[Tuple[int, int], float] = {}

        for s in nodes:
            for d in nodes:
                if s == d:
                    continue
                total_attempts += 1
                tm_val = float(traffic.get((s, d), 0.0)) if traffic else 1.0
                route, modes = _resolve_flow(
                    s, d, paths, next_hops_per_c, protected, failed_edges, repair_graph
                )
                if route is None:
                    resolved_flows.append(((s, d), "lost", tm_val, []))
                    continue

                for edge in route:
                    post_link_loads[edge] = post_link_loads.get(edge, 0.0) + tm_val
                if "mnp" in modes:
                    mnp_recovery += 1
                if "bpp" in modes:
                    bpp_recovery += 1
                if modes:
                    recovery_count += 1
                resolved_flows.append(((s, d), "delivered", tm_val, route))

        trial_max_mlu = 0.0
        edge_surv_ratio: Dict[Tuple[int, int], float] = {}
        if should_enforce_capacity:
            for edge, load in post_link_loads.items():
                u, v = edge
                cap = graph[u][v].get("capacity") if graph.has_edge(u, v) else None
                if cap is not None and cap > 0:
                    util = load / cap
                    trial_max_mlu = max(trial_max_mlu, util)
                    edge_surv_ratio[edge] = max(0.0, cap / load) if load > cap else 1.0
                else:
                    edge_surv_ratio[edge] = 1.0

        peak_mlu = max(peak_mlu, trial_max_mlu)

        for (s, d), status, tm_val, route_edges in resolved_flows:
            if status == "lost":
                continue
            delivery_factor = 1.0
            if should_enforce_capacity and route_edges:
                for edge in route_edges:
                    ratio = edge_surv_ratio.get(edge, 1.0)
                    if drop_model == "multiplicative":
                        delivery_factor *= ratio
                    else:
                        delivery_factor = min(delivery_factor, ratio)
            if delivery_factor < 1.0:
                capacity_drop_count += 1
            if delivery_factor > 0.0:
                successful_deliveries += 1
            traffic_delivered += tm_val * delivery_factor

    avail = successful_deliveries / total_attempts if total_attempts > 0 else 1.0
    if total_traffic > 0:
        t_avail = traffic_delivered / (total_traffic * num_trials)
    else:
        t_avail = avail

    return SimulationOutcome(
        topological_availability=avail,
        traffic_availability=t_avail,
        packet_loss_proxy=1.0 - t_avail,
        recovery_count=recovery_count,
        bpp_recovery_count=bpp_recovery,
        mnp_recovery_count=mnp_recovery,
        capacity_drop_count=capacity_drop_count,
        max_link_utilization=peak_mlu,
        num_trials=num_trials,
        estimand=estimand,
        mean_failed_links=total_failed_links / num_trials,
    )
