"""Validation tests for the unified evaluator.

Run with: python -m tests.test_evaluator
"""
from __future__ import annotations

import sys
import os
import tempfile

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import networkx as nx
import numpy as np

from src.evaluator import (
    build_topology_data,
    select_cta_hlp,
    select_optimal_ilp,
    select_lfc_only,
    select_random,
    select_hlp_2020,
    select_max_traffic,
    compute_estimands,
    run_single_experiment,
    ExperimentConfig,
    run_all_validation_tests,
    FailureScenario,
    CandidateTunnel,
)


def test_triangle_topology():
    """Test on a simple 3-node triangle where we can compute exact answers."""
    print("Testing triangle topology...")
    
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 2, cost=3, capacity=1000, prob_fail=0.01)
    
    traffic = {(0, 2): 100.0, (2, 0): 100.0, (0, 1): 50.0, (1, 0): 50.0, (1, 2): 50.0, (2, 1): 50.0}
    
    topology = build_topology_data(G, traffic, "triangle")
    
    print(f"  Nodes: {topology.graph.number_of_nodes()}, Edges: {topology.graph.number_of_edges()}")
    print(f"  Candidate tunnels: {len(topology.candidate_tunnels)}")
    print(f"  Scenarios: {len(topology.scenarios)}")
    
    for tunnel in topology.candidate_tunnels:
        print(f"    Tunnel {tunnel.key_link}: failed_edge={tunnel.failed_edge}, "
              f"repair_path={tunnel.repair_path}, feasible={tunnel.feasible}, "
              f"gain={tunnel.marginal_gain:.4f}")
    
    # Test greedy vs MIP
    greedy = select_cta_hlp(topology, 1)
    milp = select_optimal_ilp(topology, 1)
    print(f"  Greedy selection (B=1): {greedy}")
    print(f"  MIP selection (B=1): {milp}")
    assert greedy == milp, f"Greedy {greedy} != MIP {milp}"
    
    # Test LFC only
    lfc = select_lfc_only(topology, 1)
    print(f"  LFC selection: {lfc}")
    assert lfc == set()
    
    # Test Random (deterministic with seed)
    rand1 = select_random(topology, 1, seed=42)
    rand2 = select_random(topology, 1, seed=42)
    assert rand1 == rand2, "Random not reproducible"
    print(f"  Random selection (seed=42): {rand1}")
    
    print("  Triangle topology test PASSED")


def test_slf_modularity():
    """Verify that greedy selection matches MIP optimum for SLF (modular objective)."""
    print("Testing SLF modularity...")
    
    # Create a graph where multiple key links have non-zero traffic
    # 0 -- 1 -- 2 -- 3
    #  \         /
    #   \       /
    #    \     /
    #     4 - 5
    # But simpler: a line 0-1-2-3-4 with traffic 0<->4, 1<->3
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(2, 3, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(3, 4, cost=1, capacity=1000, prob_fail=0.01)
    # Add a shortcut 0-4 with high cost so primary path is the line
    G.add_edge(0, 4, cost=10, capacity=1000, prob_fail=0.01)
    
    # Traffic: 0<->4 (uses all edges), 1<->3 (uses 1-2-3)
    traffic = {
        (0, 4): 200.0, (4, 0): 200.0,
        (1, 3): 150.0, (3, 1): 150.0,
    }
    
    topology = build_topology_data(G, traffic, "line")
    
    print(f"  Candidate tunnels: {len(topology.candidate_tunnels)}")
    feasible = [t for t in topology.candidate_tunnels if t.feasible]
    print(f"  Feasible tunnels: {len(feasible)}")
    for t in feasible:
        print(f"    Tunnel {t.key_link}: failed_edge={t.failed_edge}, "
              f"repair_path={t.repair_path}, flows={t.flows}, gain={t.marginal_gain:.4f}")
    
    for budget in [1, 2, 3]:
        # Check gains before each call
        gains_before = [t.marginal_gain for t in topology.candidate_tunnels if t.feasible]
        print(f"  Before greedy: gains={gains_before}")
        greedy = select_cta_hlp(topology, budget)
        gains_after_greedy = [t.marginal_gain for t in topology.candidate_tunnels if t.feasible]
        print(f"  After greedy: gains={gains_after_greedy}")
        milp = select_optimal_ilp(topology, budget)
        print(f"  Budget {budget}: Greedy={greedy}, MIP={milp}")
        assert greedy == milp, f"Budget {budget}: Greedy {greedy} != MIP {milp}"
    
    print("  SLF modularity test PASSED")


def test_capacity_feasibility():
    """Test that feasible tunnels actually respect capacity in their failure scenario."""
    print("Testing capacity feasibility...")
    
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=100, prob_fail=0.01)  # Low capacity
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 2, cost=3, capacity=1000, prob_fail=0.01)
    
    # Traffic that exceeds capacity on 0-1 when it becomes detour
    traffic = {(0, 2): 150.0, (2, 0): 150.0}
    
    topology = build_topology_data(G, traffic, "capacity_test")
    
    # Check each tunnel's feasibility flag
    for tunnel in topology.candidate_tunnels:
        # Simulate the failure scenario for this tunnel
        scenario = None
        for s in topology.scenarios:
            if s.failed_key_link == tunnel.key_link:
                scenario = s
                break
        
        if scenario is None:
            continue
            
        # Compute loads with this tunnel protected
        from src.evaluator import compute_scenario_loads
        loads = compute_scenario_loads(
            scenario, traffic, topology.primary_paths, topology.next_hops,
            {tunnel.key_link}, topology
        )
        
        # Check if any repair path edge exceeds capacity
        exceeds = False
        for i in range(len(tunnel.repair_path) - 1):
            e_norm = (min(tunnel.repair_path[i], tunnel.repair_path[i+1]), 
                      max(tunnel.repair_path[i], tunnel.repair_path[i+1]))
            cap = topology.capacities.get(e_norm, float('inf'))
            if loads.get(e_norm, 0) > cap:
                exceeds = True
                break
        
        if exceeds:
            assert not tunnel.feasible, f"Tunnel {tunnel.key_link} marked feasible but exceeds capacity"
            print(f"  Tunnel {tunnel.key_link}: correctly marked INFEASIBLE (exceeds capacity)")
        else:
            assert tunnel.feasible, f"Tunnel {tunnel.key_link} marked infeasible but fits"
            print(f"  Tunnel {tunnel.key_link}: correctly marked feasible")
    
    print("  Capacity feasibility test PASSED")


def test_estimands():
    """Test that estimands compute correctly."""
    print("Testing estimands...")
    
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 2, cost=3, capacity=1000, prob_fail=0.01)
    
    traffic = {(0, 2): 100.0, (2, 0): 100.0}
    
    topology = build_topology_data(G, traffic, "estimand_test")
    
    # Test with no protection
    results = compute_estimands(topology, set(), ["unconditional", "conditional"])
    
    print(f"  Unconditional traffic availability: {results['unconditional'].traffic_availability:.6f}")
    print(f"  Conditional traffic availability: {results['conditional'].traffic_availability:.6f}")
    print(f"  Unconditional topological availability: {results['unconditional'].topological_availability:.6f}")
    print(f"  Conditional topological availability: {results['conditional'].topological_availability:.6f}")
    
    # With protection
    protected = select_cta_hlp(topology, 1)
    results_prot = compute_estimands(topology, protected, ["unconditional", "conditional"])
    
    print(f"  Protected - Unconditional traffic availability: {results_prot['unconditional'].traffic_availability:.6f}")
    print(f"  Protected - Conditional traffic availability: {results_prot['conditional'].traffic_availability:.6f}")
    
    # Traffic availability should be higher with protection
    assert results_prot['unconditional'].traffic_availability >= results['unconditional'].traffic_availability
    assert results_prot['conditional'].traffic_availability >= results['conditional'].traffic_availability
    
    print("  Estimands test PASSED")


def test_crn_reproducibility():
    """Verify same config + seed = identical results."""
    print("Testing CRN reproducibility...")
    
    config = ExperimentConfig(
        topology_names=["abilene"],
        budgets=[4],
        baselines=["Random", "C_TA_HLP"],
        seed=42,
        crn_seed=12345,
    )
    
    # This requires actual SNDlib data - skip if not available
    try:
        result1 = run_single_experiment("abilene", 4, "Random", config, replicate_idx=0)
        result2 = run_single_experiment("abilene", 4, "Random", config, replicate_idx=0)
        
        for k in ["unconditional_traffic_availability", "unconditional_topological_availability"]:
            assert abs(result1[k] - result2[k]) < 1e-10, f"CRN failed for {k}: {result1[k]} vs {result2[k]}"
        
        print("  CRN reproducibility test PASSED")
    except FileNotFoundError:
        print("  CRN test skipped (SNDlib data not found)")


def test_forwarding_logic():
    """Test forwarding path resolution in various scenarios."""
    print("Testing forwarding logic...")
    
    from src.evaluator import get_forwarding_path
    
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 2, cost=3, capacity=1000, prob_fail=0.01)
    
    traffic = {(0, 2): 100.0, (2, 0): 100.0}
    
    topology = build_topology_data(G, traffic, "forwarding_test")
    
    # Test no-failure scenario
    no_fail_scenario = topology.scenarios[0]
    path = get_forwarding_path(0, 2, no_fail_scenario, topology.primary_paths, 
                               topology.next_hops, set(), topology)
    print(f"  No-failure path 0->2: {path}")
    assert path == [0, 1, 2], f"Expected [0,1,2], got {path}"
    
    # Test failure scenario (0,1) fails
    fail_scenario = topology.scenarios[1]  # (0,1) fails
    path = get_forwarding_path(0, 2, fail_scenario, topology.primary_paths,
                               topology.next_hops, set(), topology)
    print(f"  Failure (0,1) path 0->2 (no protection): {path}")
    # In this triangle, (0,2) is an LFA for destination 2 at node 0,
    # so the path should be [0, 2] even without protection!
    assert path == [0, 2], f"Expected [0,2] via LFA, got {path}"
    
    # Test failure scenario (1,2) fails - no LFA exists
    fail_scenario_12 = None
    for s in topology.scenarios:
        if s.failed_edges == frozenset({(1, 2)}):
            fail_scenario_12 = s
            break
    
    if fail_scenario_12:
        path = get_forwarding_path(0, 2, fail_scenario_12, topology.primary_paths,
                                   topology.next_hops, set(), topology)
        print(f"  Failure (1,2) path 0->2 (no protection): {path}")
        # No LFA exists at node 1 for destination 2 (only neighbor is 0 which is towards source)
        assert path is None, f"Expected None (disconnected), got {path}"
    
    # With protection for (1,2)
    protected = select_cta_hlp(topology, 1)
    if fail_scenario_12:
        path = get_forwarding_path(0, 2, fail_scenario_12, topology.primary_paths,
                                   topology.next_hops, protected, topology)
        print(f"  Failure (1,2) path 0->2 (with protection): {path}")
        # Repair path should be 1-0-2
        assert path == [0, 1, 0, 2], f"Expected [0,1,0,2], got {path}"
    
    print("  Forwarding logic test PASSED")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("RUNNING EVALUATOR VALIDATION TESTS")
    print("=" * 60)
    
    test_triangle_topology()
    print()
    test_slf_modularity()
    print()
    test_capacity_feasibility()
    print()
    test_estimands()
    print()
    test_forwarding_logic()
    print()
    test_crn_reproducibility()
    print()
    
    print("=" * 60)
    print("ALL VALIDATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()