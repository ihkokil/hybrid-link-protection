import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_statistical_experiments import (
    _bh_adjust,
    _holm_adjust,
    compute_statistics,
)


def test_holm_adjust_known_values():
    # Holm: sorted p = 0.01, 0.02, 0.04 with m=3.
    # adjusted = 0.03, 0.04, 0.04 (monotone non-decreasing).
    assert _holm_adjust([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])


def test_bh_adjust_known_values():
    # BH with m=3 and p = 0.01, 0.02, 0.04: adjusted = 0.03, 0.03, 0.04.
    assert _bh_adjust([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.03])


def _make_df(n_reps=5, topology="tiny", budget=2, effect=0.01):
    rows = []
    for rep in range(n_reps):
        base = 0.9 + 0.001 * rep
        for method, value in [
            ("LFC Only", base - 0.01),
            ("HLP (Original)", base),
            ("TA-HLP (Proposed)", base + effect + 0.0001 * rep),
        ]:
            rows.append(
                {
                    "replicate": rep,
                    "replicate_seed": 42 + rep,
                    "topology": topology,
                    "topology_type": "test",
                    "node_count": 6,
                    "edge_count": 8,
                    "budget": budget,
                    "method": method,
                    "num_trials": 10,
                    "topological_availability": value,
                    "traffic_availability": value,
                    "packet_loss_proxy": 1.0 - value,
                    "backup_links": 0 if method == "LFC Only" else budget,
                    "compute_time_s": 0.01,
                    "mnp_recovery_count": 0,
                    "bpp_recovery_count": 0,
                    "capacity_drop_count": 0,
                    "max_link_utilization": 0.5,
                    "estimand": "unconditional",
                }
            )
    return pd.DataFrame(rows)


def test_compute_statistics_full_family_adjustments():
    df = _make_df()
    summary, comparisons, checks = compute_statistics(df, noninferiority_margin=0.001)
    assert not summary.empty
    assert len(comparisons) == 2  # two metrics for one topology/budget
    for comparison in comparisons:
        assert 0.0 <= comparison.p_value_holm <= 1.0
        assert 0.0 <= comparison.p_value_bh <= 1.0
        assert comparison.p_value_bonferroni >= comparison.p_value
        assert comparison.non_inferior is True
    assert len(checks) == 2


def test_compute_statistics_flags_negative_cell():
    df = _make_df(effect=-0.01)
    _, comparisons, checks = compute_statistics(df)
    traffic = [c for c in comparisons if c.metric == "traffic_availability"][0]
    assert traffic.mean_b < traffic.mean_a
    assert traffic.non_inferior is False
    assert checks[0].all_ta_ge_hlp is False
