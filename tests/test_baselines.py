import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.protection import (
    compute_lfc_scheme,
    compute_hlp,
    compute_ta_hlp,
    compute_max_traffic_hlp,
    compute_random_hlp,
    compute_optimal_milp,
)
from src.sndlib import load_sndlib


def test_baseline_methods_abilene():
    G, T = load_sndlib("abilene")
    
    lfc = compute_lfc_scheme(G)
    assert lfc.method == "LFC"
    assert len(lfc.protected_links) == 0

    hlp = compute_hlp(G, max_links=2)
    assert hlp.method == "HLP"
    assert len(hlp.protected_links) <= 2

    ta = compute_ta_hlp(G, T, max_links=2)
    assert ta.method == "TA-HLP"
    assert len(ta.protected_links) <= 2

    mt = compute_max_traffic_hlp(G, T, max_links=2)
    assert mt.method == "Max-Traffic"
    assert len(mt.protected_links) <= 2

    rnd = compute_random_hlp(G, max_links=2, seed=123)
    assert rnd.method == "Random"
    assert len(rnd.protected_links) <= 2

    ilp = compute_optimal_milp(G, T, max_links=2)
    assert ilp.method == "Optimal-ILP"
    assert len(ilp.protected_links) <= 2
    assert "ilp_obj" in ilp.extra

