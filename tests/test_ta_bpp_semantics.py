import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bpp import build_edge_to_sd_index, find_key_links
from src.lfc import compute_lfc
from src.path_construction import build_primary_paths
from src.ta_bpp import compute_baseline_loads, compute_delta_T


def _diamond():
    G = nx.Graph()
    G.add_edge(0, 1, cost=1.0, prob_fail=0.01, capacity=1000.0)
    G.add_edge(0, 2, cost=1.0, prob_fail=0.01, capacity=1000.0)
    G.add_edge(1, 3, cost=1.0, prob_fail=0.01, capacity=1000.0)
    G.add_edge(2, 3, cost=1.0, prob_fail=0.01, capacity=1000.0)
    return G


def test_compute_delta_uses_failure_detecting_node():
    # 0 - 1 - 2 - 3, with a detour 1 - 4 - 3 available after (1,2) fails.
    # The primary path from 0 to 3 is 0-1-2-3 (cost 3) vs 0-1-4-3 (cost 4).
    G = nx.Graph()
    G.add_edge(0, 1, cost=1.0, prob_fail=0.01, capacity=1000.0)
    G.add_edge(1, 2, cost=1.0, prob_fail=0.01, capacity=1000.0)
    G.add_edge(2, 3, cost=1.0, prob_fail=0.01, capacity=1000.0)
    G.add_edge(1, 4, cost=1.0, prob_fail=0.01, capacity=1000.0)
    G.add_edge(4, 3, cost=1.0, prob_fail=0.01, capacity=1000.0)

    next_hops = {c: compute_lfc(G, c) for c in G.nodes()}
    paths = build_primary_paths(G, next_hops)
    traffic = {(0, 3): 100.0}
    loads = compute_baseline_loads(G, paths, traffic)

    delta = compute_delta_T(
        G, (1, 2), {(0, 3)}, paths, traffic,
        baseline_loads=loads, capacity_constrained=True,
    )
    assert delta > 0.0


def test_find_key_links_includes_both_orientations():
    # A path 0-1-2 where the reversed-direction flows also use the edge.
    G = nx.Graph()
    G.add_edge(0, 1, cost=1.0, prob_fail=0.01)
    G.add_edge(1, 2, cost=1.0, prob_fail=0.01)
    next_hops = {c: compute_lfc(G, c) for c in G.nodes()}
    paths = build_primary_paths(G, next_hops)
    edge_index = build_edge_to_sd_index(paths)
    key_links = find_key_links(G, next_hops, edge_index=edge_index)
    assert (0, 1) in key_links
    assert (1, 0) not in key_links  # normalized physical key
    assert key_links[(0, 1)] != set()


def test_missing_traffic_is_zero_not_one():
    G = _diamond()
    next_hops = {c: compute_lfc(G, c) for c in G.nodes()}
    paths = build_primary_paths(G, next_hops)
    loads = compute_baseline_loads(G, paths, {})
    assert all(v == 0.0 for v in loads.values())

    delta = compute_delta_T(
        G, (0, 1), {(0, 3)}, paths, {},
        baseline_loads=loads, capacity_constrained=True,
    )
    assert delta == 0.0
