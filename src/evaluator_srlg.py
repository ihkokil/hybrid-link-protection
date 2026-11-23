"""SRLG/IBF Extension for C-TA-HLP Evaluator.

This module extends the SLF evaluator with:
1. Shared Risk Link Groups (SRLG) failure model
2. Independent Bernoulli Failures (IBF) model
3. Congestion penalty Φ(e) derived from fluid traffic model
4. Iterative greedy algorithm for capacitated submodular problem
5. MIP formulation for capacitated problem

Key differences from SLF:
- Multiple links can fail simultaneously (SRLG or IBF)
- Repair paths must avoid entire failed SRLG
- Cross-tunnel capacity coupling exists (multiple tunnels active simultaneously)
- Objective is submodular, not modular
- Greedy achieves (1-1/e) approx for uncapacitated submodular case
- With capacities: capacitated submodular maximization (NP-hard, heuristic)

Failure Models:
- SRLG: Conduits fail with probability q_j; all links in R_j fail together
- IBF: Each link fails independently with probability p_e (Bernoulli)
"""

from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, FrozenSet, Any, Callable

import networkx as nx
import numpy as np

from .shortest_path import dijkstra, shortest_path, shortest_path_in_graph_minus_edge
from .lfc import compute_lfc_next_hops
from .path_construction import build_primary_paths, detect_failure_point
from .availability import path_availability
from .utils import normalize_edge, get_logger
from .evaluator import (
    TopologyData,
    ScenarioResult,
    SimulationResult,
    ExperimentConfig,
    BASELINE_SELECTORS,
    load_topology_with_traffic,
    compute_all_next_hops,
    compute_primary_paths_all_pairs,
    compute_key_links_and_candidates,
    compute_baseline_loads,
    build_topology_data,
    get_forwarding_path,
    compute_scenario_loads,
    apply_capacity_constraints,
    compute_delivered_per_flow,
    evaluate_scenario,
    compute_estimands,
    select_lfc_only,
    select_random,
    select_hlp_2020,
    select_max_traffic,
    select_cta_hlp,
    select_optimal_ilp,
    check_tunnel_feasibility,
    normalize_edge,
)

log = get_logger("evaluator_srlg")


# =============================================================================
# SYNTHETIC SRLG GENERATION
# =============================================================================

def generate_synthetic_srlgs(
    graph: nx.Graph,
    num_srlgs: int = 5,
    max_srlg_size: int = 3,
    seed: int = 42,
) -> Tuple[List[Set[Tuple[int, int]]], List[float]]:
    """Generate synthetic SRLGs based on geographical clustering.
    
    For testing purposes, we group edges that are geographically close
    (share nodes) into SRLGs.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    edges = list(graph.edges())
    n_edges = len(edges)
    
    if n_edges == 0:
        return [], []
    
    # Build adjacency of edges (share a node)
    edge_adj = {i: set() for i in range(n_edges)}
    for i, (u1, v1) in enumerate(edges):
        for j, (u2, v2) in enumerate(edges):
            if i != j and (u1 in (u2, v2) or v1 in (u2, v2)):
                edge_adj[i].add(j)
    
    srlgs = []
    used_edges = set()
    
    for _ in range(min(num_srlgs, n_edges)):
        # Pick a random unused edge as seed
        available = [i for i in range(n_edges) if i not in used_edges]
        if not available:
            break
        seed_idx = rng.choice(available)
        
        # Grow SRLG via BFS from seed
        srlg = {seed_idx}
        used_edges.add(seed_idx)
        frontier = [seed_idx]
        
        while frontier and len(srlg) < max_srlg_size:
            idx = frontier.pop(0)
            neighbors = [n for n in edge_adj[idx] if n not in used_edges]
            if neighbors:
                n_new = rng.choice(neighbors)
                srlg.add(n_new)
                used_edges.add(n_new)
                frontier.append(n_new)
        
        # Convert edge indices to edge tuples
        srlg_edges = {normalize_edge(*edges[i]) for i in srlg}
        srlgs.append(srlg_edges)
    
    # Assign probabilities (uniform for simplicity)
    probs = [0.01] * len(srlgs)
    
    return srlgs, probs


def load_srlg_file(filepath: str) -> Tuple[List[Set[Tuple[int, int]]], List[float]]:
    """Load SRLG definitions from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    srlgs = [set(tuple(e) for e in s) for s in data["srlgs"]]
    probs = data.get("probs", [0.01] * len(srlgs))
    
    return srlgs, probs


# =============================================================================
# SRLG/IBF DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class SRLGScenario:
    """A failure scenario in SRLG or IBF model."""
    scenario_id: int
    failed_edges: FrozenSet[Tuple[int, int]]
    probability: float
    is_no_failure: bool
    failed_srlg: Optional[int] = None  # For SRLG model


@dataclass
class SRLGCandidateTunnel:
    """A candidate backup tunnel for SRLG/IBF model.
    
    In SRLG model, a tunnel protects a key link (v,d) against a specific SRLG R_j.
    The repair path must avoid ALL edges in R_j.
    """
    key_link: Tuple[int, int]          # (v, d) - the key link being protected
    fdn: int                           # = v (failure-detecting node)
    dst: int                           # = d (destination)
    failed_srlg: int                   # SRLG index that causes this failure
    failed_edges: FrozenSet[Tuple[int, int]]  # all edges in the SRLG
    repair_path: List[int]             # node list from fdn to dst avoiding failed_srlg
    flows: Set[Tuple[int, int]]        # (src,dst) using this key link under this SRLG
    srlg_prob: float                   # q_j (SRLG failure probability)
    marginal_gain: float = 0.0         # w_τ = q_j * Σ T(s,d) * A(repair_path)
    feasible: bool = True              # capacity check passed
    detour_shifts: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    # For IBF model: each tunnel is for a single link failure
    # (same structure, but failed_srlg = None, failed_edges = {single_edge}, srlg_prob = p_e)


@dataclass
class SRLGTopologyData(TopologyData):
    """Extended topology data for SRLG/IBF model."""
    srlgs: List[Set[Tuple[int, int]]] = field(default_factory=list)  # List of SRLG edge sets
    srlg_probs: List[float] = field(default_factory=list)  # q_j for each SRLG
    srlg_scenarios: List[SRLGScenario] = field(default_factory=list)
    srlg_candidate_tunnels: List[SRLGCandidateTunnel] = field(default_factory=list)
    failure_model: str = "SLF"  # "SLF", "SRLG", or "IBF"


# =============================================================================
# SRLG/IBF SCENARIO ENUMERATION
# =============================================================================

def enumerate_srlg_scenarios(
    graph: nx.Graph,
    srlgs: List[Set[Tuple[int, int]]],
    srlg_probs: List[float],
    candidate_tunnels: List[SRLGCandidateTunnel],
) -> List[SRLGScenario]:
    """Enumerate SRLG failure scenarios.
    
    Ω = {∅} ∪ {R_j : j ∈ {1..g}}
    P(∅) = 1 - Σ q_j
    P(R_j) = q_j
    """
    scenarios = []
    
    # No-failure scenario
    total_q = sum(srlg_probs)
    p_no_fail = max(0.0, 1.0 - total_q)
    
    scenarios.append(SRLGScenario(
        scenario_id=0,
        failed_edges=frozenset(),
        probability=p_no_fail,
        is_no_failure=True,
        failed_srlg=None,
    ))
    
    # SRLG failure scenarios
    for idx, (srlg_edges, q_j) in enumerate(zip(srlgs, srlg_probs)):
        if q_j > 0:
            scenarios.append(SRLGScenario(
                scenario_id=idx + 1,
                failed_edges=frozenset(srlg_edges),
                probability=q_j,
                is_no_failure=False,
                failed_srlg=idx,
            ))
    
    return scenarios


def enumerate_ibf_scenarios(
    graph: nx.Graph,
    candidate_tunnels: List[SRLGCandidateTunnel],
    max_scenarios: int = 1000,
) -> List[SRLGScenario]:
    """Enumerate IBF (Independent Bernoulli Failures) scenarios.
    
    Since 2^|E| is exponential, we use importance sampling or limit to
    single-link and no-failure scenarios plus high-probability multi-link failures.
    For exact computation with small graphs, enumerate all.
    """
    edges = list(graph.edges())
    failure_probs = [graph[u][v].get("prob_fail", 0.0) for u, v in edges]
    
    # For now, use exact enumeration for small graphs, Monte Carlo for large
    n_edges = len(edges)
    if n_edges <= 10:
        # Exact enumeration
        scenarios = []
        scenario_id = 0
        
        for mask in range(1 << n_edges):
            failed_edges = frozenset(
                normalize_edge(*edges[i]) for i in range(n_edges) if (mask >> i) & 1
            )
            prob = 1.0
            for i in range(n_edges):
                if (mask >> i) & 1:
                    prob *= failure_probs[i]
                else:
                    prob *= (1 - failure_probs[i])
            
            if prob > 0:
                scenarios.append(SRLGScenario(
                    scenario_id=scenario_id,
                    failed_edges=failed_edges,
                    probability=prob,
                    is_no_failure=(mask == 0),
                    failed_srlg=None,
                ))
                scenario_id += 1
        
        return scenarios
    else:
        # Monte Carlo sampling for large graphs
        # Include no-failure + single-link failures exactly, sample multi-link
        scenarios = []
        scenario_id = 0
        
        # No-failure
        p_no_fail = np.prod([1 - p for p in failure_probs])
        scenarios.append(SRLGScenario(
            scenario_id=scenario_id,
            failed_edges=frozenset(),
            probability=p_no_fail,
            is_no_failure=True,
        ))
        scenario_id += 1
        
        # Single-link failures
        for i, (u, v) in enumerate(edges):
            p_e = failure_probs[i]
            if p_e > 0:
                scenarios.append(SRLGScenario(
                    scenario_id=scenario_id,
                    failed_edges=frozenset([normalize_edge(u, v)]),
                    probability=p_e * np.prod([1 - p for j, p in enumerate(failure_probs) if j != i]),
                    is_no_failure=False,
                ))
                scenario_id += 1
        
        # Sample remaining multi-link scenarios
        remaining_budget = max_scenarios - len(scenarios)
        if remaining_budget > 0:
            rng = np.random.default_rng(42)
            for _ in range(remaining_budget):
                failed = frozenset(
                    normalize_edge(*edges[i]) for i in range(n_edges) if rng.random() < failure_probs[i]
                )
                if len(failed) >= 2:  # Only multi-link
                    prob = 1.0
                    for i in range(n_edges):
                        if normalize_edge(*edges[i]) in failed:
                            prob *= failure_probs[i]
                        else:
                            prob *= (1 - failure_probs[i])
                    if prob > 0:
                        scenarios.append(SRLGScenario(
                            scenario_id=scenario_id,
                            failed_edges=failed,
                            probability=prob,
                            is_no_failure=False,
                        ))
                        scenario_id += 1
        
        return scenarios


# =============================================================================
# SRLG/IBF CANDIDATE TUNNELS
# =============================================================================

def compute_srlg_candidate_tunnels(
    graph: nx.Graph,
    next_hops: Dict[int, Dict[int, Set[int]]],
    primary_paths: Dict[Tuple[int, int], List[int]],
    traffic: Dict[Tuple[int, int], float],
    srlgs: List[Set[Tuple[int, int]]],
    srlg_probs: List[float],
) -> Tuple[Dict[Tuple[int, int], Set[Tuple[int, int]]], List[SRLGCandidateTunnel]]:
    """Find key links and build candidate tunnels for SRLG model.
    
    In SRLG model, a key link (v,d) is vulnerable to SRLG R_j if:
    - The primary path of some flow (s,d) uses an edge in R_j
    - AND |N_v(d)| == 1 (no LFA)
    
    The repair path must avoid ALL edges in R_j.
    """
    from .bpp import find_key_links, build_edge_to_sd_index
    
    edge_index = build_edge_to_sd_index(primary_paths)
    key_links_dict = find_key_links(graph, next_hops, edge_index=edge_index)
    
    candidate_tunnels = []
    
    for (v, d), sd_pairs in key_links_dict.items():
        if len(next_hops[v][d]) != 1:
            continue
        hop = next(iter(next_hops[v][d]))
        failed_edge = normalize_edge(v, hop)
        
        # For each SRLG that contains the failed edge, create a tunnel
        for srlg_idx, (srlg_edges, q_j) in enumerate(zip(srlgs, srlg_probs)):
            if failed_edge not in srlg_edges:
                continue
            
            # Repair path avoiding entire SRLG
            H = graph.copy()
            for e in srlg_edges:
                if H.has_edge(*e):
                    H.remove_edge(*e)
            
            repair_path = shortest_path(H, v, d)
            if not repair_path:
                continue
            
            # Flows that use this key link AND whose primary path hits this SRLG
            affected_flows = set()
            for (s, dst) in sd_pairs:
                path = primary_paths.get((s, dst), [])
                if path:
                    path_edges = {normalize_edge(path[i], path[i+1]) for i in range(len(path)-1)}
                    if path_edges & srlg_edges:  # primary path hits this SRLG
                        affected_flows.add((s, dst))
            
            if not affected_flows:
                continue
            
            # Detour shifts
            detour_shifts = {}
            for i in range(len(repair_path) - 1):
                e_norm = normalize_edge(repair_path[i], repair_path[i + 1])
                detour_shifts[e_norm] = detour_shifts.get(e_norm, 0.0)
            
            # Marginal gain
            flow_volume = sum(traffic.get((s, d), 0.0) for (s, d) in affected_flows)
            a_repair = path_availability(H, repair_path)
            marginal_gain = q_j * flow_volume * a_repair
            
            tunnel = SRLGCandidateTunnel(
                key_link=(v, d),
                fdn=v,
                dst=d,
                failed_srlg=srlg_idx,
                failed_edges=frozenset(srlg_edges),
                repair_path=repair_path,
                flows=affected_flows,
                srlg_prob=q_j,
                marginal_gain=marginal_gain,
                feasible=True,
                detour_shifts=detour_shifts,
            )
            candidate_tunnels.append(tunnel)
    
    return key_links_dict, candidate_tunnels


def compute_ibf_candidate_tunnels(
    graph: nx.Graph,
    next_hops: Dict[int, Dict[int, Set[int]]],
    primary_paths: Dict[Tuple[int, int], List[int]],
    traffic: Dict[Tuple[int, int], float],
) -> Tuple[Dict[Tuple[int, int], Set[Tuple[int, int]]], List[SRLGCandidateTunnel]]:
    """Candidate tunnels for IBF model (same as SLF but with per-link probs)."""
    # Reuse SLF logic but store in SRLGCandidateTunnel format
    key_links_dict, slf_tunnels = compute_key_links_and_candidates(
        graph, next_hops, primary_paths, traffic
    )
    
    ibf_tunnels = []
    for t in slf_tunnels:
        ibf_tunnels.append(SRLGCandidateTunnel(
            key_link=t.key_link,
            fdn=t.fdn,
            dst=t.dst,
            failed_srlg=None,  # IBF uses single-link failures
            failed_edges=frozenset([t.failed_edge]),
            repair_path=t.repair_path,
            flows=t.flows,
            srlg_prob=graph[t.failed_edge[0]][t.failed_edge[1]].get("prob_fail", 0.01),
            marginal_gain=t.marginal_gain,
            feasible=t.feasible,
            detour_shifts=t.detour_shifts,
        ))
    
    return key_links_dict, ibf_tunnels


# =============================================================================
# CONGESTION PENALTY Φ(e) FROM FLUID MODEL
# =============================================================================

def compute_congestion_penalty(
    tunnel: SRLGCandidateTunnel,
    topology: SRLGTopologyData,
    accumulated_shifts: Dict[Tuple[int, int], float],
    accumulated_probs: Dict[Tuple[int, int], float],
) -> float:
    """Compute congestion penalty Φ(e) from fluid traffic model.
    
    In SRLG/IBF, multiple tunnels can be active simultaneously.
    The expected load on link k is:
        E[L_k] = L_0(k) + Σ_τ q_τ * shift_τ(k) * x_τ
    where q_τ is the probability that tunnel τ is activated.
    
    For SRLG: q_τ = srlg_prob of the tunnel's SRLG
    For IBF: q_τ = failure prob of the link
    
    The fluid model assumes traffic is splittable and capacity drops are proportional.
    The penalty for tunnel τ is:
        Φ(τ) = Π_{k ∈ P_repair(τ)} min(1, C_k / E[L_k | τ selected])
    
    This is a heuristic approximation of the expected delivered traffic fraction.
    """
    phi = 1.0
    
    for i in range(len(tunnel.repair_path) - 1):
        k_edge = normalize_edge(tunnel.repair_path[i], tunnel.repair_path[i + 1])
        cap = topology.capacities.get(k_edge, float('inf'))
        
        if cap <= 0:
            continue
        
        base_load = topology.baseline_loads.get(k_edge, 0.0)
        
        # Expected load from previously selected tunnels
        expected_prior_load = accumulated_probs.get(k_edge, 0.0)
        
        # This tunnel's contribution (activated with its srlg_prob)
        this_shift = tunnel.detour_shifts.get(k_edge, 0.0)
        this_prob = tunnel.srlg_prob
        expected_this_load = this_prob * this_shift
        
        total_expected_load = base_load + expected_prior_load + expected_this_load
        
        if total_expected_load > cap:
            phi *= cap / total_expected_load
    
    return phi


def compute_marginal_gain_with_penalty(
    tunnel: SRLGCandidateTunnel,
    topology: SRLGTopologyData,
    accumulated_shifts: Dict[Tuple[int, int], float],
    accumulated_probs: Dict[Tuple[int, int], float],
) -> Tuple[float, Dict[Tuple[int, int], float]]:
    """Compute marginal gain with congestion penalty.
    
    Returns (penalized_gain, shifts_dict)
    """
    phi = compute_congestion_penalty(tunnel, topology, accumulated_shifts, accumulated_probs)
    penalized_gain = tunnel.marginal_gain * phi
    return penalized_gain, tunnel.detour_shifts


# =============================================================================
# SRLG/IBF GREEDY ALGORITHM (ITERATIVE)
# =============================================================================

def select_cta_hlp_srlg(
    topology: SRLGTopologyData,
    budget: int,
    iterative: bool = True,
) -> Set[Tuple[int, int]]:
    """C-TA-HLP for SRLG/IBF model with congestion penalty.
    
    Iterative greedy: re-evaluate marginal gains after each selection
    because capacity coupling makes gains dependent on previous selections.
    """
    feasible = [t for t in topology.srlg_candidate_tunnels if t.feasible]
    
    if not iterative:
        # Static single-pass
        tunnels_with_gain = []
        for t in feasible:
            # Need topology with accumulated shifts for static eval
            # Create temporary topology with no accumulated shifts
            temp_topology = topology
            temp_topology.accumulated_shifts = {}
            temp_topology.accumulated_probs = {}
            gain, _ = compute_marginal_gain_with_penalty(t, temp_topology, {}, {})
            tunnels_with_gain.append((gain, t.key_link))
        
        tunnels_with_gain.sort(key=lambda x: (-x[0], x[1]))
        return set(kl for _, kl in tunnels_with_gain[:budget])
    
    # Iterative greedy
    protected = set()
    remaining = list(feasible)
    
    # Track accumulated expected shifts and probabilities per edge
    accumulated_shifts: Dict[Tuple[int, int], float] = {e: 0.0 for e in topology.capacities}
    accumulated_probs: Dict[Tuple[int, int], float] = {e: 0.0 for e in topology.capacities}
    
    while remaining and len(protected) < budget:
        best_tunnel = None
        best_gain = float('-inf')
        best_shifts = {}
        
        for tunnel in remaining:
            gain, shifts = compute_marginal_gain_with_penalty(
                tunnel, topology, accumulated_shifts, accumulated_probs
            )
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_tunnel = tunnel
                best_shifts = shifts
        
        if best_tunnel is None or best_gain <= 0:
            break
        
        # Select this tunnel
        protected.add(best_tunnel.key_link)
        remaining.remove(best_tunnel)
        
        # Update accumulated shifts
        for k_edge, shift in best_shifts.items():
            accumulated_shifts[k_edge] = accumulated_shifts.get(k_edge, 0.0) + shift
            accumulated_probs[k_edge] = accumulated_probs.get(k_edge, 0.0) + best_tunnel.srlg_prob
    
    return protected


# =============================================================================
# SRLG/IBF MIP FORMULATION
# =============================================================================

def select_optimal_ilp_srlg(
    topology: SRLGTopologyData,
    budget: int,
) -> Set[Tuple[int, int]]:
    """MIP for SRLG/IBF model with expected capacity constraints.
    
    Maximize Σ w_τ x_τ
    s.t. Σ x_τ ≤ B
         L_0(k) + Σ q_τ shift_τ(k) x_τ ≤ C_k  ∀k
    
    This is a 0-1 Multidimensional Knapsack Problem (NP-hard).
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
    except ImportError:
        log.warning("SciPy not available, falling back to greedy")
        return select_cta_hlp_srlg(topology, budget)
    
    feasible_tunnels = [t for t in topology.srlg_candidate_tunnels if t.feasible]
    n_k = len(feasible_tunnels)
    if n_k == 0 or budget == 0:
        return set()
    
    target_budget = min(n_k, budget)
    
    # Objective: expected traffic gain
    gains = np.array([t.marginal_gain for t in feasible_tunnels])
    
    # Budget constraint
    A_budget = np.ones((1, n_k))
    rhs_budget = np.array([target_budget])
    
    # Capacity constraints: L_0(k) + Σ q_τ shift_τ(k) x_τ ≤ C_k
    edges_list = list(topology.capacities.keys())
    n_e = len(edges_list)
    
    A_cap = np.zeros((n_e, n_k))
    rhs_cap = np.zeros(n_e)
    
    for j, k_edge in enumerate(edges_list):
        cap = topology.capacities[k_edge]
        base = topology.baseline_loads.get(k_edge, 0.0)
        rhs_cap[j] = max(0.0, cap - base)
        
        for i, tunnel in enumerate(feasible_tunnels):
            if k_edge in tunnel.detour_shifts:
                A_cap[j, i] = tunnel.srlg_prob * tunnel.detour_shifts[k_edge]
    
    if n_e > 0:
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
    
    if res.success and res.x is not None:
        selected = [feasible_tunnels[i].key_link for i in range(n_k) if res.x[i] > 0.5]
        return set(selected)
    else:
        log.warning(f"SRLG MIP failed (status={res.status}), falling back to greedy")
        return select_cta_hlp_srlg(topology, budget)


# =============================================================================
# SRLG/IBF FORWARDING LOGIC
# =============================================================================

def get_forwarding_path_srlg(
    src: int,
    dst: int,
    scenario: SRLGScenario,
    primary_paths: Dict[Tuple[int, int], List[int]],
    next_hops: Dict[int, Dict[int, Set[int]]],
    selected_tunnels: Set[Tuple[int, int]],
    topology: SRLGTopologyData,
) -> Optional[List[int]]:
    """Get forwarding path for SRLG/IBF scenario.
    
    When an SRLG fails, the FDN tries LFA first, then BPP tunnel.
    The tunnel must be for the specific SRLG that failed.
    """
    primary = primary_paths.get((src, dst))
    if primary is None or len(primary) < 2:
        return None
    
    # Find first failed edge on primary path
    failed_on_primary = None
    for i in range(len(primary) - 1):
        u, v = primary[i], primary[i + 1]
        edge = normalize_edge(u, v)
        if edge in scenario.failed_edges:
            failed_on_primary = (u, v, edge)
            break
    
    if failed_on_primary is None:
        return primary
    
    fdn, _, failed_edge = failed_on_primary
    
    # Stage 1: LFA alternates
    lfa_alternates = next_hops.get(fdn, {}).get(dst, set())
    valid_lfas = [
        nh for nh in sorted(lfa_alternates)
        if normalize_edge(fdn, nh) not in scenario.failed_edges
    ]
    
    for lfa_nh in valid_lfas:
        path_to_fdn = primary[:primary.index(fdn) + 1]
        if lfa_nh == dst:
            return path_to_fdn + [dst]
        sub_path = shortest_path(topology.graph, lfa_nh, dst)
        if sub_path:
            sub_edges = {normalize_edge(sub_path[i], sub_path[i+1]) for i in range(len(sub_path)-1)}
            if not sub_edges & scenario.failed_edges:
                return path_to_fdn + sub_path[1:]
    
    # Stage 2: BPP tunnel for this SRLG
    # Need tunnel with key_link=(fdn, dst) AND failed_srlg matching scenario
    if scenario.failed_srlg is not None:
        for tunnel in topology.srlg_candidate_tunnels:
            if (tunnel.key_link == (fdn, dst) and 
                tunnel.failed_srlg == scenario.failed_srlg and
                tunnel.key_link in selected_tunnels):
                path_to_fdn = primary[:primary.index(fdn) + 1]
                repair_edges = {normalize_edge(tunnel.repair_path[i], tunnel.repair_path[i+1]) 
                               for i in range(len(tunnel.repair_path)-1)}
                if not repair_edges & scenario.failed_edges:
                    return path_to_fdn + tunnel.repair_path[1:]
    else:
        # IBF model: match by failed_edge
        for tunnel in topology.srlg_candidate_tunnels:
            if (tunnel.key_link == (fdn, dst) and 
                failed_edge in tunnel.failed_edges and
                tunnel.key_link in selected_tunnels):
                path_to_fdn = primary[:primary.index(fdn) + 1]
                repair_edges = {normalize_edge(tunnel.repair_path[i], tunnel.repair_path[i+1]) 
                               for i in range(len(tunnel.repair_path)-1)}
                if not repair_edges & scenario.failed_edges:
                    return path_to_fdn + tunnel.repair_path[1:]
    
    return None


# =============================================================================
# SRLG/IBF ESTIMAND COMPUTATION
# =============================================================================

def evaluate_scenario_srlg(
    scenario: SRLGScenario,
    topology: SRLGTopologyData,
    selected_tunnels: Set[Tuple[int, int]],
) -> ScenarioResult:
    """Evaluate one SRLG/IBF scenario."""
    loads = compute_scenario_loads_srlg(
        scenario, topology.traffic, topology.primary_paths, 
        topology.next_hops, selected_tunnels, topology
    )
    
    delivered_loads, _ = apply_capacity_constraints(loads, topology.capacities)
    
    delivered_flows = compute_delivered_per_flow_srlg(
        scenario, topology.traffic, topology.primary_paths,
        topology.next_hops, selected_tunnels, topology, loads
    )
    
    total_volume = sum(topology.traffic.values())
    delivered_volume = sum(delivered_flows.values())
    connected = sum(1 for v in delivered_flows.values() if v > 0)
    total_pairs = len([k for k, v in topology.traffic.items() if v > 0])
    
    max_mlu = 0.0
    for edge, load in loads.items():
        cap = topology.capacities.get(edge, float('inf'))
        if cap > 0:
            max_mlu = max(max_mlu, load / cap)
    
    dropped_flows = sum(1 for k, v in delivered_flows.items() if v < topology.traffic.get(k, 0))
    total_dropped = sum(topology.traffic.get(k, 0) - v for k, v in delivered_flows.items())
    
    return ScenarioResult(
        delivered_traffic=delivered_volume,
        connected_pairs=connected,
        total_pairs=total_pairs,
        max_link_utilization=max_mlu,
        capacity_drops=dropped_flows,
        total_dropped_volume=total_dropped,
    )


def compute_scenario_loads_srlg(
    scenario: SRLGScenario,
    traffic: Dict[Tuple[int, int], float],
    primary_paths: Dict[Tuple[int, int], List[int]],
    next_hops: Dict[int, Dict[int, Set[int]]],
    selected_tunnels: Set[Tuple[int, int]],
    topology: SRLGTopologyData,
) -> Dict[Tuple[int, int], float]:
    """Compute loads for SRLG/IBF scenario."""
    loads = {e: 0.0 for e in topology.capacities}
    
    for (src, dst), volume in traffic.items():
        if volume <= 0:
            continue
        path = get_forwarding_path_srlg(src, dst, scenario, primary_paths, next_hops, 
                                        selected_tunnels, topology)
        if path is None:
            continue
        for i in range(len(path) - 1):
            e_norm = normalize_edge(path[i], path[i + 1])
            if e_norm in loads:
                loads[e_norm] += volume
    
    return loads


def compute_delivered_per_flow_srlg(
    scenario: SRLGScenario,
    traffic: Dict[Tuple[int, int], float],
    primary_paths: Dict[Tuple[int, int], List[int]],
    next_hops: Dict[int, Dict[int], Set[int]],
    selected_tunnels: Set[Tuple[int, int]],
    topology: SRLGTopologyData,
    scenario_loads: Dict[Tuple[int, int], float],
) -> Dict[Tuple[int, int], float]:
    """Compute delivered volume per flow for SRLG/IBF."""
    delivered = {}
    edge_survival = {}
    for edge, load in scenario_loads.items():
        cap = topology.capacities.get(edge, float('inf'))
        if load > cap and cap > 0:
            edge_survival[edge] = cap / load
        else:
            edge_survival[edge] = 1.0
    
    for (src, dst), volume in traffic.items():
        if volume <= 0:
            delivered[(src, dst)] = 0.0
            continue
        path = get_forwarding_path_srlg(src, dst, scenario, primary_paths, next_hops,
                                        selected_tunnels, topology)
        if path is None:
            delivered[(src, dst)] = 0.0
            continue
        
        min_survival = 1.0
        for i in range(len(path) - 1):
            e_norm = normalize_edge(path[i], path[i + 1])
            min_survival = min(min_survival, edge_survival.get(e_norm, 1.0))
        
        delivered[(src, dst)] = volume * min_survival
    
    return delivered


def compute_estimands_srlg(
    topology: SRLGTopologyData,
    selected_tunnels: Set[Tuple[int, int]],
    estimands: List[str],
) -> Dict[str, SimulationResult]:
    """Compute estimands for SRLG/IBF."""
    results = {}
    
    for estimand in estimands:
        if estimand == "unconditional":
            total_delivered = 0.0
            total_connected = 0
            total_pairs = 0
            total_mlu = 0.0
            peak_mlu = 0.0
            total_drops = 0.0
            total_dropped_volume = 0.0
            total_dropped_flows = 0.0
            scenario_details = []
            
            for scenario in topology.srlg_scenarios:
                if scenario.probability == 0:
                    continue
                res = evaluate_scenario_srlg(scenario, topology, selected_tunnels)
                scenario_details.append(res)
                
                total_delivered += scenario.probability * res.delivered_traffic
                total_connected += scenario.probability * res.connected_pairs
                total_pairs += scenario.probability * res.total_pairs
                total_mlu += scenario.probability * res.max_link_utilization
                peak_mlu = max(peak_mlu, res.max_link_utilization)
                total_drops += scenario.probability * res.capacity_drops
                total_dropped_volume += scenario.probability * res.total_dropped_volume
                total_dropped_flows += scenario.probability * res.capacity_drops
            
            total_volume = sum(topology.traffic.values())
            
            results[estimand] = SimulationResult(
                traffic_availability=total_delivered / total_volume if total_volume > 0 else 1.0,
                topological_availability=total_connected / total_pairs if total_pairs > 0 else 1.0,
                mean_max_link_utilization=total_mlu,
                peak_max_link_utilization=peak_mlu,
                expected_capacity_drops=total_drops,
                expected_dropped_volume=total_dropped_volume,
                expected_dropped_flows=total_dropped_flows,
                num_scenarios=len(topology.srlg_scenarios),
                estimand=estimand,
                scenario_details=scenario_details,
            )
            
        elif estimand == "conditional":
            failure_scenarios = [s for s in topology.srlg_scenarios if not s.is_no_failure]
            total_prob = sum(s.probability for s in failure_scenarios)
            
            if total_prob == 0:
                results[estimand] = SimulationResult(estimand=estimand)
                continue
            
            total_delivered = 0.0
            total_connected = 0
            total_pairs = 0
            total_mlu = 0.0
            peak_mlu = 0.0
            total_drops = 0.0
            total_dropped_volume = 0.0
            total_dropped_flows = 0.0
            scenario_details = []
            
            for scenario in failure_scenarios:
                res = evaluate_scenario_srlg(scenario, topology, selected_tunnels)
                scenario_details.append(res)
                weight = scenario.probability / total_prob
                
                total_delivered += weight * res.delivered_traffic
                total_connected += weight * res.connected_pairs
                total_pairs += weight * res.total_pairs
                total_mlu += weight * res.max_link_utilization
                peak_mlu = max(peak_mlu, res.max_link_utilization)
                total_drops += weight * res.capacity_drops
                total_dropped_volume += weight * res.total_dropped_volume
                total_dropped_flows += weight * res.capacity_drops
            
            total_volume = sum(topology.traffic.values())
            
            results[estimand] = SimulationResult(
                traffic_availability=total_delivered / total_volume if total_volume > 0 else 1.0,
                topological_availability=total_connected / total_pairs if total_pairs > 0 else 1.0,
                mean_max_link_utilization=total_mlu,
                peak_max_link_utilization=peak_mlu,
                expected_capacity_drops=total_drops,
                expected_dropped_volume=total_dropped_volume,
                expected_dropped_flows=total_dropped_flows,
                num_scenarios=len(failure_scenarios),
                estimand=estimand,
                scenario_details=scenario_details,
            )
    
    return results


# =============================================================================
# SRLG/IBF TOPOLOGY BUILDING
# =============================================================================

def build_srlg_topology_data(
    graph: nx.Graph,
    traffic: Dict[Tuple[int, int], float],
    name: str,
    srlgs: List[Set[Tuple[int, int]]],
    srlg_probs: List[float],
    failure_model: str = "SRLG",
) -> SRLGTopologyData:
    """Build topology data for SRLG/IBF model."""
    log.info(f"Building {failure_model} topology data for {name}...")
    
    next_hops = compute_all_next_hops(graph)
    primary_paths = compute_primary_paths_all_pairs(graph)
    baseline_loads = compute_baseline_loads(graph, primary_paths, traffic)
    
    if failure_model == "SRLG":
        key_links_dict, candidate_tunnels = compute_srlg_candidate_tunnels(
            graph, next_hops, primary_paths, traffic, srlgs, srlg_probs
        )
    elif failure_model == "IBF":
        key_links_dict, candidate_tunnels = compute_ibf_candidate_tunnels(
            graph, next_hops, primary_paths, traffic
        )
    else:
        raise ValueError(f"Unknown failure model: {failure_model}")
    
    # Check feasibility (same as SLF for now)
    for tunnel in candidate_tunnels:
        tunnel.feasible = check_tunnel_feasibility(tunnel, graph, baseline_loads)
    
    feasible_count = sum(1 for t in candidate_tunnels if t.feasible)
    log.info(f"  {len(candidate_tunnels)} candidate tunnels, {feasible_count} feasible")
    
    if failure_model == "SRLG":
        scenarios = enumerate_srlg_scenarios(graph, srlgs, srlg_probs, candidate_tunnels)
    else:
        scenarios = enumerate_ibf_scenarios(graph, candidate_tunnels)
    
    log.info(f"  {len(scenarios)} {failure_model} scenarios (including no-failure)")
    
    capacities = {normalize_edge(u, v): graph[u][v].get("capacity", float('inf')) for u, v in graph.edges()}
    failure_probs = {normalize_edge(u, v): graph[u][v].get("prob_fail", 0.0) for u, v in graph.edges()}
    
    return SRLGTopologyData(
        graph=graph,
        name=name,
        traffic=traffic,
        primary_paths=primary_paths,
        next_hops=next_hops,
        key_links=key_links_dict,
        candidate_tunnels=candidate_tunnels,  # SLF format for compatibility
        baseline_loads=baseline_loads,
        scenarios=scenarios,  # SLF format for compatibility
        capacities=capacities,
        failure_probs=failure_probs,
        srlgs=srlgs,
        srlg_probs=srlg_probs,
        srlg_scenarios=scenarios,
        srlg_candidate_tunnels=candidate_tunnels,
        failure_model=failure_model,
    )


# =============================================================================
# BASELINE SELECTORS FOR SRLG/IBF
# =============================================================================

# =============================================================================
# SRLG/IBF BASELINE SELECTORS
# =============================================================================

def select_hlp_2020_srlg(
    topology: SRLGTopologyData, 
    budget: int,
) -> Set[Tuple[int, int]]:
    """Original HLP: topological marginal gain (unweighted) for SRLG.
    
    For SRLG, Δ(e) = q_j * Σ 1 * A(P_repair) = q_j * |flows| * A(P_repair)
    """
    tunnels_with_gain = []
    for tunnel in topology.srlg_candidate_tunnels:
        if not tunnel.feasible:
            continue
        # A(repair) - compute on graph without failed SRLG
        H = topology.graph.copy()
        for e in tunnel.failed_edges:
            if H.has_edge(*e):
                H.remove_edge(*e)
        a_repair = path_availability(H, tunnel.repair_path)
        
        num_flows = len(tunnel.flows)
        topo_gain = tunnel.srlg_prob * num_flows * a_repair
        tunnels_with_gain.append((topo_gain, tunnel.key_link))
    
    tunnels_with_gain.sort(key=lambda x: (-x[0], x[1]))
    return set(kl for _, kl in tunnels_with_gain[:budget])


def select_max_traffic_srlg(
    topology: SRLGTopologyData, 
    budget: int,
) -> Set[Tuple[int, int]]:
    """Max-Traffic: greedy by raw traffic volume, IGNORING capacity."""
    tunnels_with_gain = []
    for tunnel in topology.srlg_candidate_tunnels:
        H = topology.graph.copy()
        for e in tunnel.failed_edges:
            if H.has_edge(*e):
                H.remove_edge(*e)
        a_repair = path_availability(H, tunnel.repair_path)
        flow_volume = sum(topology.traffic.get((s, d), 0.0) for (s, d) in tunnel.flows)
        gain = tunnel.srlg_prob * flow_volume * a_repair
        tunnels_with_gain.append((gain, tunnel.key_link))
    
    tunnels_with_gain.sort(key=lambda x: (-x[0], x[1]))
    return set(kl for _, kl in tunnels_with_gain[:budget])


def select_cta_hlp_srlg(
    topology: SRLGTopologyData, 
    budget: int,
    iterative: bool = True,
) -> Set[Tuple[int, int]]:
    """C-TA-HLP for SRLG/IBF model with congestion penalty."""
    feasible = [t for t in topology.srlg_candidate_tunnels if t.feasible]
    
    if not iterative:
        tunnels_with_gain = []
        for t in feasible:
            gain, _ = compute_marginal_gain_with_penalty(t, topology, {}, {})
            tunnels_with_gain.append((gain, t.key_link))
        
        tunnels_with_gain.sort(key=lambda x: (-x[0], x[1]))
        return set(kl for _, kl in tunnels_with_gain[:budget])
    
    # Iterative greedy
    protected = set()
    remaining = list(feasible)
    
    accumulated_shifts: Dict[Tuple[int, int], float] = {e: 0.0 for e in topology.capacities}
    accumulated_probs: Dict[Tuple[int, int], float] = {e: 0.0 for e in topology.capacities}
    
    while remaining and len(protected) < budget:
        best_tunnel = None
        best_gain = float('-inf')
        best_shifts = {}
        
        for tunnel in remaining:
            gain, shifts = compute_marginal_gain_with_penalty(
                tunnel, topology, accumulated_shifts, accumulated_probs
            )
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_tunnel = tunnel
                best_shifts = shifts
        
        if best_tunnel is None or best_gain <= 0:
            break
        
        protected.add(best_tunnel.key_link)
        remaining.remove(best_tunnel)
        
        for k_edge, shift in best_shifts.items():
            accumulated_shifts[k_edge] = accumulated_shifts.get(k_edge, 0.0) + shift
            accumulated_probs[k_edge] = accumulated_probs.get(k_edge, 0.0) + best_tunnel.srlg_prob
    
    return protected


def select_optimal_ilp_srlg(
    topology: SRLGTopologyData, 
    budget: int,
) -> Set[Tuple[int, int]]:
    """MIP for SRLG/IBF model with expected capacity constraints."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
    except ImportError:
        log.warning("SciPy not available, falling back to greedy")
        return select_cta_hlp_srlg(topology, budget)
    
    feasible_tunnels = [t for t in topology.srlg_candidate_tunnels if t.feasible]
    n_k = len(feasible_tunnels)
    if n_k == 0 or budget == 0:
        return set()
    
    target_budget = min(n_k, budget)
    
    # Objective: expected traffic gain
    gains = np.array([t.marginal_gain for t in feasible_tunnels])
    
    # Budget constraint
    A_budget = np.ones((1, n_k))
    rhs_budget = np.array([target_budget])
    
    # Capacity constraints: L_0(k) + Σ q_τ shift_τ(k) x_τ ≤ C_k
    edges_list = list(topology.capacities.keys())
    n_e = len(edges_list)
    
    A_cap = np.zeros((n_e, n_k))
    rhs_cap = np.zeros(n_e)
    
    for j, k_edge in enumerate(edges_list):
        cap = topology.capacities[k_edge]
        base = topology.baseline_loads.get(k_edge, 0.0)
        rhs_cap[j] = max(0.0, cap - base)
        
        for i, tunnel in enumerate(feasible_tunnels):
            if k_edge in tunnel.detour_shifts:
                A_cap[j, i] = tunnel.srlg_prob * tunnel.detour_shifts[k_edge]
    
    if n_e > 0:
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
    
    if res.success and res.x is not None:
        selected = [feasible_tunnels[i].key_link for i in range(n_k) if res.x[i] > 0.5]
        return set(selected)
    else:
        log.warning(f"SRLG MIP failed (status={res.status}), falling back to greedy")
        return select_cta_hlp_srlg(topology, budget)


SRLG_BASELINE_SELECTORS = {
    "LFC_Only": lambda t, b: set(),
    "Random": lambda t, b, s=42: select_random(t, b, s),
    "HLP_2020": select_hlp_2020_srlg,
    "Max_Traffic": select_max_traffic_srlg,
    "C_TA_HLP": lambda t, b: select_cta_hlp_srlg(t, b, iterative=True),
    "C_TA_HLP_Static": lambda t, b: select_cta_hlp_srlg(t, b, iterative=False),
    "Optimal_ILP": select_optimal_ilp_srlg,
}


# =============================================================================
# EXPERIMENT RUNNER FOR SRLG/IBF
# =============================================================================

def run_single_experiment_srlg(
    topology_name: str,
    budget: int,
    baseline_name: str,
    config: ExperimentConfig,
    replicate_idx: int = 0,
    traffic_perturbation_seed: Optional[int] = None,
    srlgs: Optional[List[Set[Tuple[int, int]]]] = None,
    srlg_probs: Optional[List[float]] = None,
    failure_model: str = "SRLG",
) -> Dict[str, Any]:
    """Run one SRLG/IBF experiment."""
    base_seed = config.seed + hash(topology_name) % 10000
    graph, base_traffic = load_topology_with_traffic(topology_name, seed=base_seed)
    
    if traffic_perturbation_seed is not None:
        rng = random.Random(traffic_perturbation_seed)
        traffic = {
            (s, d): val * max(0.05, 1.0 + rng.gauss(0.0, config.perturbation_std))
            for (s, d), val in base_traffic.items()
        }
    else:
        traffic = base_traffic
    
    # Build SRLG topology data
    if failure_model == "SRLG" and srlgs is not None:
        topology = build_srlg_topology_data(graph, traffic, topology_name, srlgs, srlg_probs, "SRLG")
        selectors = SRLG_BASELINE_SELECTORS
    elif failure_model == "IBF":
        topology = build_srlg_topology_data(graph, traffic, topology_name, [], [], "IBF")
        selectors = SRLG_BASELINE_SELECTORS
    else:
        # Fall back to SLF evaluator
        from .evaluator import run_single_experiment
        return run_single_experiment(topology_name, budget, baseline_name, config, replicate_idx, traffic_perturbation_seed)
    
    # Select tunnels
    selector = selectors[baseline_name]
    if baseline_name == "Random":
        select_seed = config.crn_seed + hash((baseline_name, budget, replicate_idx)) % 10000
        selected_tunnels = selector(topology, budget, select_seed)
    else:
        selected_tunnels = selector(topology, budget)
    
    # Compute estimands
    estimand_results = compute_estimands_srlg(topology, selected_tunnels, config.estimands)
    
    result = {
        "topology": topology_name,
        "budget": budget,
        "baseline": baseline_name,
        "replicate": replicate_idx,
        "num_protected_links": len(selected_tunnels),
        "protected_links": sorted(selected_tunnels),
        "traffic_total": sum(traffic.values()),
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
    
    return result