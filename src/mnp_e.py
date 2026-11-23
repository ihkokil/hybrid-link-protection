"""MNP-e reference -- output-equivalent to Algorithm 2 in the base paper.

The base paper's MNP-e changes the weight of the computing router's link to
a neighbour to its negative value and runs incremental SPF (i-SPF).  The
paper proves in Theorem 8 that the resulting next-hop sets are exactly the
LFC sets.  NetworkX exposes neither i-SPF nor negative-weight Dijkstra, so
this module computes the *output* of MNP-e by the LFC condition.

**Important scope:** this is an output-equivalent reference, not a
reproduction of MNP-e's incremental computation or of the paper's
sub-SPF timing.  ``experiments/run_table_reproduction.py`` must label the
column accordingly; compute-time comparisons between this function and
``compute_lfc`` measure implementation noise, not an algorithmic gap.

If a true i-SPF implementation is added later, it must be validated against
``compute_mnp_e_reference`` on random graphs before replacing it.
"""
from __future__ import annotations

from typing import Dict, Set

import networkx as nx

from .lfc import compute_lfc_next_hops


def compute_mnp_e_reference(graph: nx.Graph, c) -> Dict[int, Set]:
    """Return the MNP-e next-hop sets for computing router ``c``.

    The result is identical to LFC's by Theorem 8 of the base paper.  The
    function exists as an explicit reference so that experiments cannot
    silently substitute LFC for MNP-e without a visible name change.
    """
    return compute_lfc_next_hops(graph, c)


def compute_mnp_e(graph: nx.Graph, c) -> Dict[int, Set]:
    """Backward-compatible name for the reference implementation."""
    return compute_mnp_e_reference(graph, c)
