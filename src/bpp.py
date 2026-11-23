"""Original BPP -- Algorithm 3 in the paper.

For every node ``v`` and destination ``d``, link ``(v, B_v(d))`` is a
*key link* if MNP-e/LFC could not find any backup next-hop, i.e.
``|N_v(d)| == 1``. Key links are aggregated, their marginal
contribution ``Δ(e)`` (Eq. 6) is computed, and they are greedily
protected in descending order until the network-availability
requirement ``A(G) >= R`` is met.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from .availability import network_availability, path_availability
from .path_construction import build_primary_paths, detect_failure_point
from .shortest_path import dijkstra
from .utils import normalize_edge


# --------------------------------------------------------------------------- #
# Forwarding-path index: edge -> set of (s,d) using it directionally.
# --------------------------------------------------------------------------- #


def build_edge_to_sd_index(
    paths: Dict[Tuple[int, int], List]
) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    """Return ``{directed edge (u,v): set of (s,d) pairs whose primary path uses (u,v)}``.

    The same (s,d) can appear under multiple directed edges (one per hop).
    Building this once is far cheaper (O(|V|^2 * avg_path_len)) than
    iterating over all paths inside key-link detection.
    """
    idx: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    for (s, d), path in paths.items():
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            idx.setdefault((u, v), set()).add((s, d))
    return idx


def find_key_links(
    graph: nx.Graph,
    next_hops_per_c: Dict[int, Dict[int, Set]],
    edge_index: Optional[Dict[Tuple[int, int], Set[Tuple[int, int]]]] = None,
) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    """Return ``{undirected e: set of (s,d) pairs using e as a *directed* key link}``.

    Implements Algorithm 3 lines 1-7. A directed key link occurs at
    node ``v`` toward ``B_v(d)``, when ``|N_v(d)| == 1``. The
    undirected link is what we register as "to protect".

    ``edge_index`` is the output of ``build_edge_to_sd_index``. We
    accept it as an optional input to avoid rebuilding inside BPP and
    TA-BPP for the same graph.
    """
    if edge_index is None:
        paths = build_primary_paths(graph, next_hops_per_c)
        edge_index = build_edge_to_sd_index(paths)

    key_links: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    for v in graph.nodes():
        for d, hops in next_hops_per_c.get(v, {}).items():
            if d == v:
                continue
            if len(hops) == 1:
                hop = next(iter(hops))
                e = normalize_edge(v, hop)
                # A physical link failure affects both directions; include
                # flows whose primary path uses either orientation.
                key_links.setdefault(e, set()).update(edge_index.get((v, hop), set()))
                key_links[e].update(edge_index.get((hop, v), set()))
    return key_links


# --------------------------------------------------------------------------- #
# Δ(e) -- topological per Algorithm 3 (Eq. 6)
# --------------------------------------------------------------------------- #


def compute_delta_e(
    graph: nx.Graph,
    e: Tuple[int, int],
    sd_pairs: Set[Tuple[int, int]],
    paths: Dict[Tuple[int, int], List],
) -> float:
    """Eq. (6): ``Δ(e) = Σ_{(s,d)∈e.P} [A*(s,d) - A(s,d)]``.

    Under the single-link failure model, protecting key link ``e`` alters
    end-to-end availability solely during the failure event ``F_e``
    (occurring with probability ``p_e = prob_fail(e)``). Without
    protection, flow ``(s,d)`` is dropped under ``F_e``; with protection,
    the flow is rerouted from the *failure-detecting node* across
    ``P_repair(e)`` with operational availability ``A(P_repair(e))``. Thus
        A*(s,d) - A(s,d) = P(F_e) * A*(s,d | F_e)
                         = p_e * A(P_repair(e)).
    """
    # Group flows by the node that detects the failure.  The repair path
    # starts at that node, not at the flow source.
    by_detector: Dict = {}
    for (s, d) in sd_pairs:
        path = paths.get((s, d), [])
        idx = detect_failure_point(path, e)
        if idx < 0:
            continue
        by_detector.setdefault(path[idx], []).append(d)

    H = graph.copy()
    a, b = normalize_edge(*e)
    if H.has_edge(a, b):
        H.remove_edge(a, b)

    delta = 0.0
    p_fail_e = float(graph[a][b].get("prob_fail", 0.01))
    for detector, targets in by_detector.items():
        dist, pred = dijkstra(H, detector)
        for d in targets:
            if dist.get(d, float("inf")) == float("inf"):
                continue
            new_path = _walk_pred(detector, d, pred)
            if not new_path:
                continue
            a_repair = path_availability(H, new_path)
            delta += p_fail_e * a_repair
    return delta


def _walk_pred(source, target, predecessors: Dict) -> List:
    cur, path = target, []
    while cur != source:
        path.append(cur)
        cur = predecessors[cur]
        if cur is None:
            return []
    path.append(source)
    path.reverse()
    return path


def bpp(
    graph: nx.Graph,
    next_hops_per_c: Dict[int, Dict[int, Set]],
    R: float = 0.999,
    max_links: Optional[int] = None,
    verbose: bool = False,
) -> List[Tuple[int, int]]:
    """Algorithm 3 in the paper.

    Parameters
    ----------
    graph : nx.Graph
    next_hops_per_c : dict
        Output of MNP-e (or LFC) for every computing node.
    R : float
        Network-availability requirement. We stop when ``A(G) >= R``.
    max_links : int or None
        Optional hard cap on the number of backup links (used by the
        TA-HLP experiments to enforce a budget). If given, the loop
        stops once ``len(protected) >= max_links`` even if R is not met.

    Returns
    -------
    protected_links : list of undirected edges (sorted by Δ descending)
    """
    paths = build_primary_paths(graph, next_hops_per_c)
    edge_index = build_edge_to_sd_index(paths)
    key_links = find_key_links(graph, next_hops_per_c, edge_index=edge_index)

    if verbose:
        print(f"[BPP] {len(key_links)} key links detected")

    deltas: Dict[Tuple[int, int], float] = {}
    for e, sd_pairs in key_links.items():
        deltas[e] = compute_delta_e(graph, e, sd_pairs, paths)

    sorted_keys = sorted(deltas, key=lambda x: deltas[x], reverse=True)
    V = graph.number_of_nodes()
    norm = V * (V - 1) if V > 1 else 1
    current_A = network_availability(graph, paths)

    protected: List[Tuple[int, int]] = []
    for e in sorted_keys:
        if max_links is not None and len(protected) >= max_links:
            break
        if max_links is None and current_A >= R:
            break
        protected.append(e)
        current_A += deltas[e] / norm  # Eq. update in Algorithm 3 line 19

    if verbose:
        print(f"[BPP] Protected {len(protected)} links; A(G) ~ {current_A:.4f}")
    return protected
