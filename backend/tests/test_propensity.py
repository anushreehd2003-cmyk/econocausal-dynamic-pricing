import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import generate
from propensity import balance_table, estimate_propensity, weighted_smd


def test_scores_are_probabilities():
    customers, _ = generate(n=2000, seed=3)
    ps, t = estimate_propensity(customers)
    assert len(ps) == len(customers)
    assert ps.min() > 0 and ps.max() < 1


def test_smd_zero_for_identical_groups():
    x = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    t = np.array([1, 1, 1, 0, 0, 0])
    assert abs(weighted_smd(x, t)) < 1e-9


def test_ipw_improves_balance():
    customers, _ = generate(n=3000, seed=3)
    ps, t = estimate_propensity(customers)
    table = balance_table(customers, t, ps)
    assert table["smd_after_ipw"].abs().max() < table["smd_before"].abs().max()