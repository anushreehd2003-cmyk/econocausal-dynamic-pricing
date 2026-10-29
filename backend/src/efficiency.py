"""Budget-efficiency curve and JavaScript parity fixture (Week 3, Day 16).

1. Efficiency curve: how much true extra revenue each strategy earns as the
   budget grows, measured on held-out customers and scaled to a budget per
   10,000 customers. Exported for the dashboard.
2. Parity fixture: inputs and expected greedy allocations from Python, used
   by a JavaScript test to prove the browser version gives the same answer.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimizer import (
    LEVELS,
    blanket_allocation,
    evaluate,
    greedy_allocation,
    optimize_allocation,
    values_from_ite,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOCS = ROOT / "docs"
DEMO_DIR = ROOT / "frontend" / "public" / "demo"
FIXTURE_DIR = ROOT / "frontend" / "tests" / "fixtures"
BUDGETS = [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000]
STRATEGIES = ["optimizer", "greedy", "blanket", "oracle"]
BLANKET_SEEDS = 10


def efficiency_curve(pred, true, basket, budgets, n_total, blanket_level=10):
    """True extra revenue and purchases per 10,000 customers, by budget.

    pred, true: (n, 2) effects for the held-out group. Budgets are given for
    n_total customers and scaled down to the held-out group for the solve;
    results are scaled back up by the same factor."""
    scale = len(basket) / n_total
    rows = []
    for budget in budgets:
        b = budget * scale
        allocs = {
            "optimizer": optimize_allocation(values_from_ite(pred, basket), b),
            "greedy": greedy_allocation(values_from_ite(pred, basket), b),
            "oracle": optimize_allocation(values_from_ite(true, basket), b),
        }
        row = {"budget": budget}
        for name, alloc in allocs.items():
            t = evaluate(alloc, true, basket)
            row[f"{name}_revenue"] = t["extra_revenue"] / scale
            row[f"{name}_purchases"] = t["extra_purchases"] / scale
        # Blanket targeting is random, so average several draws.
        draws = [
            evaluate(blanket_allocation(len(basket), b, blanket_level, seed=s), true, basket)
            for s in range(BLANKET_SEEDS)
        ]
        row["blanket_revenue"] = np.mean([d["extra_revenue"] for d in draws]) / scale
        row["blanket_purchases"] = np.mean([d["extra_purchases"] for d in draws]) / scale
        rows.append(row)
    return pd.DataFrame(rows)


def parity_fixture(frame, budgets, n=400):
    """Inputs plus the expected Python greedy allocation for several cases."""
    sub = frame.head(n)
    ite = sub[["ite_10", "ite_20"]].to_numpy()
    basket = sub["avg_basket_value"].to_numpy()
    values = values_from_ite(ite, basket)
    cases = []
    for levels in [(10, 20), (10,), (20,)]:
        cols = [LEVELS.index(lv) for lv in levels]
        for budget in budgets:
            alloc = greedy_allocation(values[:, cols], budget, levels)
            cases.append({
                "budget": budget,
                "levels": list(levels),
                "allocation": [int(a) for a in alloc],
            })
    customers = [
        {"ite_10": float(a), "ite_20": float(b), "avg_basket_value": float(c)}
        for a, b, c in zip(ite[:, 0], ite[:, 1], basket)
    ]
    return {"customers": customers, "cases": cases}


def plot_curve(table, path):
    plt.figure(figsize=(8, 5))
    styles = {
        "oracle": ("Oracle (true effects)", "k:"),
        "optimizer": ("Optimizer (exact)", "-"),
        "greedy": ("Greedy", "--"),
        "blanket": ("Blanket $10 (random)", "-"),
    }
    for name, (label, style) in styles.items():
        plt.plot(table["budget"], table[f"{name}_revenue"], style, label=label, linewidth=2)
    plt.xlabel("Budget ($, per 10,000 customers)")
    plt.ylabel("True extra revenue ($, per 10,000 customers)")
    plt.title("Budget efficiency (held-out customers, true effects)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    needed = ["retail_campaign.csv", "ite_predictions.csv", "ground_truth_effects.csv"]
    demo_file = DEMO_DIR / "customers_ite.csv"
    if any(not (DATA_DIR / f).exists() for f in needed) or not demo_file.exists():
        raise SystemExit("Run generate_data.py, dml_engine.py and export_demo.py first.")

    campaign = pd.read_csv(DATA_DIR / "retail_campaign.csv")
    preds = pd.read_csv(DATA_DIR / "ite_predictions.csv")
    truth = pd.read_csv(DATA_DIR / "ground_truth_effects.csv")
    data = (
        campaign[["customer_id", "avg_basket_value"]]
        .merge(preds[["customer_id", "is_test", "ite_10", "ite_20"]], on="customer_id")
        .merge(truth[["customer_id", "true_uplift_10", "true_uplift_20"]], on="customer_id")
    )
    held = data[data["is_test"] == 1]
    print(f"Computing efficiency curve on {len(held)} held-out customers "
          f"({len(BUDGETS)} budgets; this can take a minute)...")
    table = efficiency_curve(
        held[["ite_10", "ite_20"]].to_numpy(),
        held[["true_uplift_10", "true_uplift_20"]].to_numpy(),
        held["avg_basket_value"].to_numpy(),
        BUDGETS,
        n_total=len(data),
    )

    shown = table[["budget", "optimizer_revenue", "greedy_revenue",
                   "blanket_revenue", "oracle_revenue"]].copy()
    shown["optimizer_pct_of_oracle"] = 100 * shown["optimizer_revenue"] / shown["oracle_revenue"]
    shown["vs_blanket_x"] = shown["optimizer_revenue"] / shown["blanket_revenue"]
    print("\nTrue extra revenue per 10,000 customers:")
    print(shown.round(1).to_string(index=False))

    DOCS.mkdir(exist_ok=True)
    table.round(3).to_csv(DOCS / "budget_efficiency.csv", index=False)
    plot_curve(table, DOCS / "budget_efficiency.png")

    payload = {
        "note": "Held-out synthetic customers, true effects, scaled to 10,000 customers.",
        "unit": "Extra revenue ($) per 10,000 customers",
        "budget": [int(b) for b in table["budget"]],
        "series": {
            name: {
                "revenue": [round(float(v), 2) for v in table[f"{name}_revenue"]],
                "purchases": [round(float(v), 3) for v in table[f"{name}_purchases"]],
            }
            for name in STRATEGIES
        },
    }
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEMO_DIR / "budget_efficiency.json", "w") as f:
        json.dump(payload, f)

    frame = pd.read_csv(demo_file)
    fixture = parity_fixture(frame, budgets=[100, 500, 1500, 5000])
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_DIR / "greedy_parity.json", "w") as f:
        json.dump(fixture, f)

    print("\nSaved docs/budget_efficiency.csv, docs/budget_efficiency.png,")
    print("frontend/public/demo/budget_efficiency.json and")
    print("frontend/tests/fixtures/greedy_parity.json")


if __name__ == "__main__":
    main()