import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from optimizer import (
    LEVELS,
    blanket_allocation,
    evaluate,
    greedy_allocation,
    optimize_allocation,
    values_from_ite,
)


def _random_values(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ite = rng.normal(0.03, 0.08, size=(n, 2))
    basket = rng.gamma(4.0, 15.0, n)
    return ite, basket, values_from_ite(ite, basket)


def _total(values, alloc):
    return sum(
        values[alloc == level, j].sum() for j, level in enumerate(LEVELS)
    )


def test_known_small_case():
    # Customer 0: $20 is worth 5.0, $10 is worth 1.0
    # Customer 1: $10 is worth 4.5, $20 is worth 4.6
    # Budget 20: best is $10 + $10 = 5.5, better than a single $20 (5.0)
    values = np.array([[1.0, 5.0], [4.5, 4.6]])
    alloc = optimize_allocation(values, budget=20)
    assert alloc.tolist() == [10, 10]


def test_all_strategies_respect_budget():
    _, basket, values = _random_values()
    for budget in (0, 5, 10, 95, 500, 5000):
        assert optimize_allocation(values, budget).sum() <= budget
        assert greedy_allocation(values, budget).sum() <= budget
        assert blanket_allocation(len(values), budget, 10).sum() <= budget


def test_never_targets_negative_value_customers():
    _, _, values = _random_values()
    alloc = optimize_allocation(values, 100_000)  # huge budget
    never = (values <= 0).all(axis=1)
    assert (alloc[never] == 0).all()
    for j, level in enumerate(LEVELS):
        assert (values[alloc == level, j] > 0).all()


def test_optimizer_at_least_as_good_as_greedy():
    _, _, values = _random_values()
    opt = optimize_allocation(values, 500)
    greedy = greedy_allocation(values, 500)
    assert _total(values, opt) >= _total(values, greedy) - 1e-6


def test_more_budget_never_reduces_value():
    _, _, values = _random_values()
    totals = [_total(values, optimize_allocation(values, b)) for b in (100, 400, 1600)]
    assert totals[0] <= totals[1] <= totals[2]


def test_evaluate_counts():
    ite = np.array([[0.05, 0.10], [0.02, -0.03], [0.04, 0.08]])
    basket = np.array([100.0, 50.0, 10.0])
    out = evaluate(np.array([20, 10, 0]), ite, basket)
    assert out["customers_targeted"] == 2
    assert out["spend"] == 30.0
    assert np.isclose(out["extra_purchases"], 0.10 + 0.02)
    assert np.isclose(out["extra_revenue"], 0.10 * 100 + 0.02 * 50)
    assert out["share_with_positive_effect"] == 1.0