import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from efficiency import efficiency_curve, parity_fixture


def _held_out(n=400, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.04, 0.08, n)
    basket = rng.gamma(4.0, 15.0, n)
    pred = np.column_stack([0.6 * latent + rng.normal(0, 0.02, n),
                            latent + rng.normal(0, 0.02, n)])
    true = np.column_stack([0.6 * latent, latent])
    return pred, true, basket


def test_curve_grows_with_budget_and_ordering_holds():
    pred, true, basket = _held_out()
    table = efficiency_curve(pred, true, basket, [200, 1000, 4000], n_total=1000)
    assert len(table) == 3
    for col in ("optimizer_revenue", "greedy_revenue", "oracle_revenue"):
        assert table[col].is_monotonic_increasing
    assert (table["oracle_revenue"] >= table["optimizer_revenue"] - 1e-6).all()
    assert (table["optimizer_revenue"] > table["blanket_revenue"]).all()


def test_parity_fixture_structure():
    pred, _, basket = _held_out(n=120)
    frame = pd.DataFrame({"ite_10": pred[:, 0], "ite_20": pred[:, 1],
                          "avg_basket_value": basket})
    fx = parity_fixture(frame, budgets=[50, 500], n=100)
    assert len(fx["customers"]) == 100
    assert len(fx["cases"]) == 6
    for case in fx["cases"]:
        assert len(case["allocation"]) == 100
        assert sum(case["allocation"]) <= case["budget"]