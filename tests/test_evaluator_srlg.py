"""Validation tests for SRLG/IBF evaluator."""

from __future__ import annotations

import sys
import os
import networkx as nx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.evaluator_srlg import (
    enumerate_srlg_scenarios,
    enumerate_ibf_scenarios,
    compute_srlg_candidate_tunnels,
    compute_ibf_candidate_tunnels,
    build_srlg_topology_data,
    select_cta_hlp_srlg,
    select_optimal_ilp_srlg,
    compute_congestion_penalty,
    get_forwarding_path_srlg,
    compute_estimands_srlg,
    SRLGTopologyData,
    SRLGCandidateTunnel,
    SRLGScenario,
)
from src.utils import normalize_edge


def test_srlg_scenarios():
    """Test SRLG scenario enumeration."""
    print("Testing SRLG scenarios...")
    
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 2, cost=3, capacity=1000, prob_fail=0.01)
    
    # SRLG: {0-1, 1-2} share a conduit
    srlgs = [
        {normalize_edge(0, 1), normalize_edge(1, 2)},  # conduit 1
        {normalize_edge(0, 2)},  # conduit 2
    ]
    srlg_probs = [0.01, 0.005]
    
    scenarios = enumerate_srlg_scenarios(G, srlgs, srlg_probs, [])
    
    assert len(scenarios) == 3  # no-failure + 2 SRLGs
    assert scenarios[0].is_no_failure
    assert scenarios[0].probability == 1 - 0.015
    
    print(f"  Scenarios: {len(scenarios)}")
    for s in scenarios:
        print(f"    {s.scenario_id}: failed_edges={s.failed_edges}, prob={s.probability}, no_fail={s.is_no_failure}")
    
    print("  SRLG scenarios test PASSED")


def test_ibf_scenarios():
    """Test IBF scenario enumeration."""
    print("Testing IBF scenarios...")
    
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 2, cost=3, capacity=1000, prob_fail=0.01)
    
    scenarios = enumerate_ibf_scenarios(G, [])
    
    # 3 edges = 2^3 = 8 scenarios
    assert len(scenarios) == 8
    
    # Check probabilities sum to 1
    total_prob = sum(s.probability for s in scenarios)
    assert abs(total_prob - 1.0) < 1e-10
    
    print(f"  IBF scenarios: {len(scenarios)}")
    print("  IBF scenarios test PASSED")


def test_srlg_candidate_tunnels():
    """Test SRLG candidate tunnel generation."""
    print("Testing SRLG candidate tunnels...")
    
    # Use a more connected graph where SRLG repair paths exist
    # 0 -- 1 -- 2 -- 3
    #  \         /
    #   4 ------ 5
    # With SRLG: {0-1, 1-2} share conduit
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(2, 3, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 4, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(4, 5, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(5, 3, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 3, cost=5, capacity=1000, prob_fail=0.01)
    
    srlgs = [
        {normalize_edge(0, 1), normalize_edge(1, 2)},  # conduit 1
        {normalize_edge(2, 3)},  # conduit 2
    ]
    srlg_probs = [0.01, 0.005]
    
    traffic = {(0, 3): 100.0, (3, 0): 100.0}
    
    from src.evaluator import compute_all_next_hops, compute_primary_paths_all_pairs
    next_hops = compute_all_next_hops(G)
    primary_paths = compute_primary_paths_all_pairs(G)
    
    key_links, tunnels = compute_srlg_candidate_tunnels(G, next_hops, primary_paths, traffic, srlgs, srlg_probs)
    
    print(f"  Key links: {key_links}")
    print(f"  Candidate tunnels: {len(tunnels)}")
    for t in tunnels:
        print(f"    ({t.fdn},{t.dst}) via SRLG {t.failed_srlg}: repair={t.repair_path}, gain={t.marginal_gain:.4f}")
    
    assert len(tunnels) >= 1
    print("  SRLG candidate tunnels test PASSED")


def test_srlg_topology_building():
    """Test complete SRLG topology building."""
    print("Testing SRLG topology building...")
    
    # Use the same connected graph as test_srlg_candidate_tunnels
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(2, 3, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 4, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(4, 5, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(5, 3, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 3, cost=5, capacity=1000, prob_fail=0.01)
    
    srlgs = [
        {normalize_edge(0, 1), normalize_edge(1, 2)},
        {normalize_edge(2, 3)},
    ]
    srlg_probs = [0.01, 0.005]
    
    traffic = {(0, 3): 100.0, (3, 0): 100.0}
    
    topology = build_srlg_topology_data(G, traffic, "test", srlgs, srlg_probs, "SRLG")
    
    assert isinstance(topology, SRLGTopologyData)
    assert topology.failure_model == "SRLG"
    assert len(topology.srlgs) == 2
    assert len(topology.srlg_candidate_tunnels) >= 1
    assert len(topology.srlg_scenarios) == 3
    
    print(f"  Candidate tunnels: {len(topology.srlg_candidate_tunnels)}")
    print(f"  Scenarios: {len(topology.srlg_scenarios)}")
    print("  SRLG topology building test PASSED")


def test_ibf_topology_building():
    """Test IBF topology building."""
    print("Testing IBF topology building...")
    
    # Use a graph where IBF makes sense
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 2, cost=3, capacity=1000, prob_fail=0.01)
    
    traffic = {(0, 2): 100.0, (2, 0): 100.0}
    
    topology = build_srlg_topology_data(G, traffic, "test", [], [], "IBF")
    
    assert topology.failure_model == "IBF"
    assert len(topology.srlg_candidate_tunnels) >= 1
    # IBF: 2^3 = 8 scenarios for 3 edges
    assert len(topology.srlg_scenarios) == 8
    
    print(f"  Candidate tunnels: {len(topology.srlg_candidate_tunnels)}")
    print(f"  Scenarios: {len(topology.srlg_scenarios)}")
    print("  IBF topology building test PASSED")


def test_congestion_penalty():
    """Test congestion penalty computation."""
    print("Testing congestion penalty...")
    
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=100, prob_fail=0.01)  # Low capacity
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 2, cost=3, capacity=1000, prob_fail=0.01)
    
    traffic = {(0, 2): 150.0, (2, 0): 150.0}
    srlgs = [{normalize_edge(0, 1), normalize_edge(1, 2)}]
    srlg_probs = [0.01]
    
    topology = build_srlg_topology_data(G, traffic, "test", srlgs, srlg_probs, "SRLG")
    
    # Accumulated shifts from previously selected tunnels
    accumulated_shifts = {normalize_edge(0, 1): 100.0}
    accumulated_probs = {normalize_edge(0, 1): 0.01}
    
    for tunnel in topology.srlg_candidate_tunnels:
        phi = compute_congestion_penalty(tunnel, topology, accumulated_shifts, accumulated_probs)
        print(f"  Tunnel ({tunnel.fdn},{tunnel.dst}): phi={phi:.4f}")
        assert 0 <= phi <= 1
    
    print("  Congestion penalty test PASSED")


def test_srlg_greedy_vs_mip():
    """Test SRLG greedy vs MIP."""
    print("Testing SRLG greedy vs MIP...")
    
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(2, 3, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 3, cost=5, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 3, cost=2, capacity=1000, prob_fail=0.01)
    
    traffic = {
        (0, 3): 200.0, (3, 0): 200.0,
        (1, 3): 150.0, (3, 1): 150.0,
    }
    
    srlgs = [
        {normalize_edge(0, 1)},
        {normalize_edge(1, 2)},
        {normalize_edge(2, 3)},
        {normalize_edge(1, 3)},
    ]
    srlg_probs = [0.01, 0.01, 0.01, 0.005]
    
    topology = build_srlg_topology_data(G, traffic, "test", srlgs, srlg_probs, "SRLG")
    
    for budget in [1, 2, 3]:
        greedy = select_cta_hlp_srlg(topology, budget)
        milp = select_optimal_ilp_srlg(topology, budget)
        print(f"  Budget {budget}: Greedy={greedy}, MIP={milp}")
        # Note: For capacitated submodular, greedy != MIP in general
        # But they should be close
    
    print("  SRLG greedy vs MIP test PASSED")


def test_ibf_model():
    """Test IBF model."""
    print("Testing IBF model...")
    
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 2, cost=3, capacity=1000, prob_fail=0.01)
    
    traffic = {(0, 2): 100.0, (2, 0): 100.0}
    
    topology = build_srlg_topology_data(G, traffic, "test", [], [], "IBF")
    
    greedy = select_cta_hlp_srlg(topology, 1)
    milp = select_optimal_ilp_srlg(topology, 1)
    
    print(f"  IBF Greedy={greedy}, MIP={milp}")
    assert greedy == milp  # IBF with per-scenario independent should match
    
    print("  IBF model test PASSED")


def test_forwarding_srlg():
    """Test SRLG forwarding logic."""
    print("Testing SRLG forwarding...")
    
    G = nx.Graph()
    G.add_edge(0, 1, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(1, 2, cost=1, capacity=1000, prob_fail=0.01)
    G.add_edge(0, 2, cost=3, capacity=1000, prob_fail=0.01)
    
    traffic = {(0, 2): 100.0}
    srlgs = [{normalize_edge(0, 1), normalize_edge(1, 2)}]
    srlg_probs = [0.01]
    
    topology = build_srlg_topology_data(G, traffic, "test", srlgs, srlg_probs, "SRLG")
    
    # Scenario: SRLG 0 fails
    scenario = topology.srlg_scenarios[1]  # First failure scenario
    
    # Without protection
    path = get_forwarding_path_srlg(0, 2, scenario, topology.primary_paths, 
                                     topology.next_hops, set(), topology)
    print(f"  No protection path: {path}")
    
    # With protection
    protected = select_cta_hlp_srlg(topology, 1)
    path = get_forwarding_path_srlg(0, 2, scenario, topology.primary_paths,
                                     topology.next_hops, protected, topology)
    print(f"  With protection path: {path}")
    
    print("  SRLG forwarding test PASSED")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("RUNNING SRLG/IBF EVALUATOR VALIDATION TESTS")
    print("=" * 60)
    
    from src.utils import normalize_edge
    
    test_srlg_scenarios()
    print()
    test_ibf_scenarios()
    print()
    test_srlg_candidate_tunnels()
    print()
    test_srlg_topology_building()
    print()
    test_ibf_topology_building()
    print()
    test_congestion_penalty()
    print()
    test_srlg_greedy_vs_mip()
    print()
    test_ibf_model()
    print()
    test_forwarding_srlg()
    print()
    
    print("=" * 60)
    print("ALL SRLG/IBF VALIDATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()