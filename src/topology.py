"""Network topology: load GML files or generate synthetic graphs.

Each edge carries:
    cost        -- integer OSPF-style metric (default 1..10)
    prob_fail   -- per-link failure probability, uniform in [0.001, 0.02]
                   (matches the paper's evaluation setup)
    capacity    -- arbitrary integer, available for capacity-aware
                   extensions (e.g., TA-HLP with a real traffic matrix)
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np

from .utils import get_logger

log = get_logger(__name__)


@dataclass
class NetworkTopology:
    """A small wrapper around an undirected graph + helpers."""

    graph: nx.Graph
    name: str = "unnamed"
    traffic_matrix: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # ------------------------------------------------------------------ load

    @classmethod
    def from_gml(
        cls,
        filepath: str,
        name: Optional[str] = None,
        seed: int = 42,
    ) -> "NetworkTopology":
        """Load an Internet Topology Zoo GML file.

        Falls back to ``random.randint`` if a numeric ``LinkMetric`` is missing.
        Re-labels nodes to consecutive integers starting from 0.
        Handles duplicate edges by keeping the one with lowest cost.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        # Parse GML manually to handle duplicate edges and custom attributes
        with open(filepath, 'r') as f:
            content = f.read()

        # Extract nodes
        node_pattern = r'node\s*\[\s*(.*?)\s*\]'
        node_blocks = re.findall(node_pattern, content, re.DOTALL)
        
        node_map = {}  # original_id -> new_id
        nodes_data = {}
        for i, block in enumerate(node_blocks):
            orig_id_match = re.search(r'id\s+(\d+)', block)
            label_match = re.search(r'label\s+"([^"]+)"', block)
            if orig_id_match:
                orig_id = int(orig_id_match.group(1))
                node_map[orig_id] = i
                label = label_match.group(1) if label_match else str(orig_id)
                nodes_data[i] = {'original_id': orig_id, 'label': label}

        # Extract edges
        edge_pattern = r'edge\s*\[\s*(.*?)\s*\]'
        edge_blocks = re.findall(edge_pattern, content, re.DOTALL)
        
        MG = nx.MultiGraph()
        MG.add_nodes_from(nodes_data.keys())
        
        for block in edge_blocks:
            src_match = re.search(r'source\s+(\d+)', block)
            tgt_match = re.search(r'target\s+(\d+)', block)
            if src_match and tgt_match:
                src = node_map.get(int(src_match.group(1)))
                tgt = node_map.get(int(tgt_match.group(1)))
                if src is not None and tgt is not None:
                    # Extract edge data
                    data = {}
                    link_metric_match = re.search(r'LinkMetric\s+"?(\d+)"?', block)
                    if link_metric_match:
                        data['LinkMetric'] = int(link_metric_match.group(1))
                    link_speed_match = re.search(r'LinkSpeed\s+"?([\d.]+)"?', block)
                    if link_speed_match:
                        data['LinkSpeed'] = link_speed_match.group(1)
                    link_speed_units_match = re.search(r'LinkSpeedUnits\s+"?(\w+)"?', block)
                    if link_speed_units_match:
                        data['LinkSpeedUnits'] = link_speed_units_match.group(1)
                    link_speed_raw_match = re.search(r'LinkSpeedRaw\s+([\d.]+)', block)
                    if link_speed_raw_match:
                        data['LinkSpeedRaw'] = float(link_speed_raw_match.group(1))
                    id_match = re.search(r'id\s+"([^"]+)"', block)
                    if id_match:
                        data['id'] = id_match.group(1)
                    
                    MG.add_edge(src, tgt, **data)

        # Convert to simple Graph by merging parallel edges (keep lowest cost)
        G = nx.Graph()
        G.add_nodes_from(MG.nodes(data=True))
        for u, v, key, data in MG.edges(keys=True, data=True):
            if G.has_edge(u, v):
                existing_cost = G[u][v].get("cost", float('inf'))
                new_cost = data.get("LinkMetric", data.get("cost", float('inf')))
                try:
                    new_cost = float(new_cost)
                except (ValueError, TypeError):
                    new_cost = float('inf')
                if new_cost < existing_cost:
                    G[u][v].update(data)
            else:
                G.add_edge(u, v, **data)

        if not nx.is_connected(G):
            # Keep the largest connected component, like the paper does.
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
            log.warning("Topology not connected; keeping largest CC (%d nodes)", G.number_of_nodes())

        # Re-label to integers 0..n-1 to simplify downstream indexing.
        G = nx.convert_node_labels_to_integers(G)

        # Derive all generated edge attributes from a local, deterministic
        # RNG.  The previous implementation mixed module-global ``random``
        # and NumPy state, which made a loaded topology change between runs.
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        for u, v in G.edges():
            cost = G[u][v].get("LinkMetric", rng.randint(1, 10))
            cost = float(cost) if isinstance(cost, (int, float)) else rng.randint(1, 10)
            G[u][v]["cost"] = float(cost)
            G[u][v]["prob_fail"] = float(np_rng.uniform(0.001, 0.02))
            G[u][v]["capacity"] = float(rng.randint(100, 1000))

        name = name or os.path.splitext(os.path.basename(filepath))[0]
        G.graph["topology_source"] = "internet_topology_zoo_gml"
        G.graph["failure_probability_model"] = "uniform_random_[0.001,0.02]"
        G.graph["capacity_model"] = "uniform_random_[100,1000]"
        G.graph["dataset_name"] = name
        log.info("Loaded topology '%s': %d nodes, %d edges", name, G.number_of_nodes(), G.number_of_edges())
        return cls(graph=G, name=name)

    # -------------------------------------------------------------- generate

    @classmethod
    def generate_random(
        cls,
        n: int = 50,
        m: int = 2,
        seed: int = 42,
        name: str = "synthetic_ba",
    ) -> "NetworkTopology":
        """Barabási–Albert graph with random weights.

        BA is used as a generic Internet-like generator when no GML is
        available. The paper uses BRITE Waxman; BA is a comparable model.
        """
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        for u, v in G.edges():
            G[u][v]["cost"] = float(rng.randint(1, 10))
            G[u][v]["prob_fail"] = float(np_rng.uniform(0.001, 0.02))
            G[u][v]["capacity"] = float(rng.randint(100, 1000))
        G.graph["topology_source"] = "synthetic_barabasi_albert"
        G.graph["failure_probability_model"] = "uniform_random_[0.001,0.02]"
        G.graph["capacity_model"] = "uniform_random_[100,1000]"
        G.graph["dataset_name"] = name
        log.info("Generated synthetic topology '%s': %d nodes, %d edges", name, n, G.number_of_edges())
        return cls(graph=G, name=name)

    # ------------------------------------------------------------ properties

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    def summary(self) -> str:
        G = self.graph
        return (
            f"Topology '{self.name}': |V|={G.number_of_nodes()}, "
            f"|E|={G.number_of_edges()}, avg_degree={2*G.number_of_edges()/G.number_of_nodes():.2f}"
        )


# --------------------------------------------------------------------------- #


def _safe_number(value, default):
    """Best-effort numeric coercion."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return float(default)