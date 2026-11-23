import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.build_results_table import build_table


def _row(topology, budget, metric, mean_a, mean_b, p, p_bonf, p_holm, p_bh, ni):
    return {
        "topology": topology,
        "budget": budget,
        "metric": metric,
        "method_a": "HLP (Original)",
        "method_b": "TA-HLP (Proposed)",
        "mean_a": mean_a,
        "mean_b": mean_b,
        "std_a": 0.001,
        "std_b": 0.001,
        "n": 5,
        "t_statistic": 1.0,
        "p_value": p,
        "p_value_bonferroni": p_bonf,
        "wilcoxon_w": 1.0,
        "wilcoxon_p": p,
        "wilcoxon_p_bonferroni": p_bonf,
        "cohens_d": 1.0,
        "ci95_diff": (0.0, 0.0),
        "p_value_holm": p_holm,
        "p_value_bh": p_bh,
        "noninferiority_margin": 0.001,
        "noninferiority_p": p,
        "non_inferior": ni,
    }


def test_build_table_includes_negative_cells(tmp_path):
    csv = tmp_path / "comparisons.csv"
    df = pd.DataFrame([
        _row("abilene", 2, "traffic_availability", 0.81, 0.82, 0.01, 0.04, 0.03, 0.03, True),
        _row("abilene", 4, "traffic_availability", 0.82, 0.81, 0.80, 1.0, 1.0, 0.80, False),
    ])
    df.to_csv(csv, index=False)
    table = build_table(str(csv))
    assert "abilene" in table
    assert "0.8200" in table and "0.8100" in table
    assert "no" in table
    assert "statistical_results" in table


def test_build_table_rejects_empty_csv(tmp_path):
    csv = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(csv, index=False)
    with pytest.raises(ValueError):
        build_table(str(csv))
