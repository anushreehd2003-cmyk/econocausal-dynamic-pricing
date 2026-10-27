"""Budget-constrained discount allocation for EconoCausal (SciPy).

Chooses a discount from {$0, $10, $20} for every customer to maximize total
predicted incremental revenue, with the total face value of all discounts
capped at the budget. This is a multiple-choice knapsack problem, solved
exactly as an integer program with scipy.optimize.milp.

Value of discount d for customer i = predicted change in purchase
probability x average basket value. This is GROSS incremental revenue: it
ignores margin given away to customers who would have bought anyway, so
compare strategies at the same budget instead of reading totals as profit.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOCS = ROOT / "docs"
LEVELS = (10, 20)


def values_from_ite(ite, basket):
    """Predicted incremental revenue per customer and discount level.

    ite: (n, L) predicted change in purchase probability. basket: (n,)."""
    return np.asarray(ite, dtype=float) * np.asarray(basket, dtype=float)[:, None]


def _candidates(values, levels):
    """(customer, level index, value, cost) for options with positive value."""
    cust, k = np.nonzero(values > 0)
    cost = np.asarray(levels, dtype=float)[k]
    return cust, k, values[cust, k], cost


def optimize_allocation(values, budget, levels=LEVELS, time_limit=60):
    """Exact optimum: at most one discount per customer, total <= budget."""
    n = values.shape[0]
    alloc = np.zeros(n, dtype=int)
    cust, k, val, cost = _candidates(values, levels)
    if len(val) == 0 or budget < min(levels):
        return alloc

    m = len(val)
    uniq, row = np.unique(cust, return_inverse=True)
    one_per_customer = coo_matrix(
        (np.ones(m), (row, np.arange(m))), shape=(len(uniq), m)
    ).tocsr()
    constraints = [
        LinearConstraint(one_per_customer, -np.inf, 1),
        LinearConstraint(cost.reshape(1, -1), -np.inf, float(budget)),
    ]
    res = milp(
        c=-val,
        constraints=constraints,
        integrality=np.ones(m),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit},
    )
    if res.x is None:
        raise RuntimeError(f"Optimizer failed: {res.message}")
    chosen = res.x > 0.5
    alloc[cust[chosen]] = np.asarray(levels)[k[chosen]]
    return alloc


def greedy_allocation(values, budget, levels=LEVELS):
    """Fast heuristic: best value per budget dollar first."""
    n = values.shape[0]
    alloc = np.zeros(n, dtype=int)
    cust, _, val, cost = _candidates(values, levels)
    left = float(budget)
    for j in np.argsort(-(val / cost), kind="stable"):
        i = cust[j]
        if alloc[i] == 0 and cost[j] <= left:
            alloc[i] = int(cost[j])
            left -= cost[j]
    return alloc


def blanket_allocation(n, budget, level, seed=0):
    """Baseline: the same discount to randomly chosen customers."""
    rng = np.random.default_rng(seed)
    alloc = np.zeros(n, dtype=int)
    k = min(n, int(budget // level))
    alloc[rng.choice(n, size=k, replace=False)] = level
    return alloc


def evaluate(alloc, ite, basket, levels=LEVELS):
    """Totals for an allocation under a given set of effects."""
    alloc = np.asarray(alloc)
    purchases = revenue = 0.0
    positive = 0
    for j, level in enumerate(levels):
        mask = alloc == level
        purchases += ite[mask, j].sum()
        revenue += (ite[mask, j] * basket[mask]).sum()
        positive += int((ite[mask, j] > 0).sum())
    targeted = int((alloc > 0).sum())
    return {
        "customers_targeted": targeted,
        "spend": float(alloc.sum()),
        "extra_purchases": float(purchases),
        "extra_revenue": float(revenue),
        "share_with_positive_effect": positive / targeted if targeted else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=5000.0)
    parser.add_argument("--blanket-level", type=int, default=10, choices=LEVELS)
    args = parser.parse_args()

    needed = ["retail_campaign.csv", "ite_predictions.csv", "ground_truth_effects.csv"]
    if any(not (DATA_DIR / f).exists() for f in needed):
        raise SystemExit(
            "Run generate_data.py and dml_engine.py first (see README Setup)."
        )
    campaign = pd.read_csv(DATA_DIR / "retail_campaign.csv")
    preds = pd.read_csv(DATA_DIR / "ite_predictions.csv")
    truth = pd.read_csv(DATA_DIR / "ground_truth_effects.csv")
    data = (
        campaign[["customer_id", "avg_basket_value"]]
        .merge(preds[["customer_id", "is_test", "ite_10", "ite_20"]], on="customer_id")
        .merge(truth[["customer_id", "true_uplift_10", "true_uplift_20"]], on="customer_id")
    )

    # Honest evaluation: customers the model never trained on, with the
    # budget scaled to the smaller group so spend per customer stays equal.
    held = data[data["is_test"] == 1]
    budget_held = args.budget * len(held) / len(data)
    pred = held[["ite_10", "ite_20"]].to_numpy()
    true = held[["true_uplift_10", "true_uplift_20"]].to_numpy()
    basket = held["avg_basket_value"].to_numpy()

    strategies = {
        "Optimizer (exact, predicted effects)":
            optimize_allocation(values_from_ite(pred, basket), budget_held),
        "Greedy (value per dollar)":
            greedy_allocation(values_from_ite(pred, basket), budget_held),
        f"Blanket ${args.blanket_level} (random customers)":
            blanket_allocation(len(held), budget_held, args.blanket_level),
        "Oracle (exact, TRUE effects: ceiling)":
            optimize_allocation(values_from_ite(true, basket), budget_held),
    }
    rows = []
    for name, alloc in strategies.items():
        p = evaluate(alloc, pred, basket)
        t = evaluate(alloc, true, basket)
        rows.append({
            "strategy": name,
            "customers_targeted": t["customers_targeted"],
            "spend": t["spend"],
            "predicted_extra_revenue": p["extra_revenue"],
            "true_extra_revenue": t["extra_revenue"],
            "true_extra_purchases": t["extra_purchases"],
            "share_with_positive_true_effect": t["share_with_positive_effect"],
        })
    table = pd.DataFrame(rows)

    print(f"Held-out customers: {len(held)}, budget ${budget_held:,.0f} "
          f"(= ${args.budget:,.0f} scaled to this group)\n")
    shown = table.copy()
    shown["share_with_positive_true_effect"] = (
        shown["share_with_positive_true_effect"] * 100).round(1)
    print(shown.round(2).to_string(index=False))

    # Allocation for all customers at the full budget (dashboard demo).
    all_pred = data[["ite_10", "ite_20"]].to_numpy()
    all_basket = data["avg_basket_value"].to_numpy()
    alloc = optimize_allocation(values_from_ite(all_pred, all_basket), args.budget)
    summary = evaluate(alloc, all_pred, all_basket)
    print(f"\nAll {len(data)} customers, budget ${args.budget:,.0f}:")
    print(f"  $10 discounts: {int((alloc == 10).sum())}, "
          f"$20 discounts: {int((alloc == 20).sum())}, "
          f"spend ${summary['spend']:,.0f}")
    print(f"  Predicted extra revenue: ${summary['extra_revenue']:,.0f}, "
          f"predicted extra purchases: {summary['extra_purchases']:.1f}")

    DOCS.mkdir(exist_ok=True)
    table.round(4).to_csv(DOCS / "optimizer_results.csv", index=False)
    print("\nSaved docs/optimizer_results.csv")


if __name__ == "__main__":
    main()