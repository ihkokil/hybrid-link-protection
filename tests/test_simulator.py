import os
import sys
import networkx as nx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.lfc import compute_lfc
from src.simulator import simulate_failures


def _ring_with_attributes(n=5, p_fail=0.01):
    G = nx.cycle_graph(n)
    for u, v in G.edges():
        G[u][v]["cost"] = 1.0
        G[u][v]["prob_fail"] = p_fail
    return G


def test_simulator_basic_run():
    G = _ring_with_attributes()
    next_hops = {c: compute_lfc(G, c) for c in G.nodes()}
    outcome = simulate_failures(
        G, next_hops, [], num_trials=50, seed=42, estimand="conditional"
    )

    assert 0.0 <= outcome.topological_availability <= 1.0
    assert outcome.num_trials == 50
    assert outcome.estimand == "conditional"
    assert outcome.mean_failed_links == pytest.approx(1.0)
    assert outcome.recovery_count >= 0


def test_simulator_unconditional_includes_no_failure_trials():
    # With very small failure probabilities, most trials have no failure and
    # must still count as successful deliveries.
    G = _ring_with_attributes(p_fail=1e-6)
    next_hops = {c: compute_lfc(G, c) for c in G.nodes()}
    outcome = simulate_failures(G, next_hops, [], num_trials=100, seed=7)
    assert outcome.estimand == "unconditional"
    assert outcome.mean_failed_links < 0.1
    assert outcome.topological_availability == pytest.approx(1.0)
    assert outcome.traffic_availability == pytest.approx(1.0)


def test_simulator_loop_protection():
    # A line graph 0 - 1 - 2 cannot survive a cut of either edge without
    # tunnels.  Use the conditional estimand so a failure actually occurs.
    G = nx.Graph()
    G.add_edge(0, 1, cost=1.0, prob_fail=0.01)
    G.add_edge(1, 2, cost=1.0, prob_fail=0.01)

    next_hops = {c: compute_lfc(G, c) for c in G.nodes()}
    outcome = simulate_failures(
        G, next_hops, [], num_trials=20, seed=42, estimand="conditional"
    )
    assert outcome.topological_availability < 1.0


def test_simulator_capacity_enforcement():
    # Triangle graph: 0-1 (cost 1), 1-2 (cost 1), 0-2 (cost 10, capacity 50)
    G = nx.Graph()
    G.add_edge(0, 1, cost=1.0, prob_fail=0.01, capacity=1000.0)
    G.add_edge(1, 2, cost=1.0, prob_fail=0.01, capacity=1000.0)
    G.add_edge(0, 2, cost=10.0, prob_fail=0.01, capacity=50.0)

    next_hops = {c: compute_lfc(G, c) for c in G.nodes()}
    traffic = {(0, 1): 200.0}

    out_cap = simulate_failures(
        G, next_hops, protected_links=[(0, 1)], traffic=traffic,
        num_trials=30, seed=42, enforce_capacity=True, estimand="conditional",
    )
    out_nocap = simulate_failures(
        G, next_hops, protected_links=[(0, 1)], traffic=traffic,
        num_trials=30, seed=42, enforce_capacity=False, estimand="conditional",
    )

    assert out_cap.max_link_utilization > 1.0
    assert out_cap.capacity_drop_count > 0
    assert out_cap.traffic_availability < out_nocap.traffic_availability


def test_simulator_bpp_recovery_from_detector_not_source():
    # 0 - 1 - 2 with a backup tunnel on (0,1).  The flow 0->2 loses (0,1);
    # the repair path must start at the failure-detecting node (0 here).
    G = nx.Graph()
    G.add_edge(0, 1, cost=1.0, prob_fail=1.0)
    G.add_edge(1, 2, cost=1.0, prob_fail=0.0)
    G.add_edge(0, 2, cost=10.0, prob_fail=0.0)
    next_hops = {c: compute_lfc(G, c) for c in G.nodes()}

    out = simulate_failures(
        G, next_hops, protected_links=[(0, 1)],
        num_trials=1, seed=1, estimand="conditional",
        enforce_capacity=False,
    )
    assert out.bpp_recovery_count >= 1
