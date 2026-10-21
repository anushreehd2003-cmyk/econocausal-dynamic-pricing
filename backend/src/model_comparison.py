"""Model comparison and Qini curves for EconoCausal (Day 12).

Compares targeting strategies for the $20 discount on held-out customers.
Qini curves use the TRUE individual effects from the synthetic ground
truth, which is only possible because the data is synthetic.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from causal_graph import CONFOUNDERS
from dml_engine import SEED, fit_dml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOCS = ROOT / "docs"
EXPORT_DIR = ROOT / "frontend" / "public" / "demo"
LEVEL = 20
FOREST = "Causal forest DML"
LINEAR = "Linear DML"
BASELINE = "Standard model (targets least likely to buy)"

_trapz = getattr(np, "trapezoid", None) or np.trapz


def qini_curve(score, uplift):
    """Cumulative incremental purchases per 1,000 customers when targeting
    customers in order of decreasing score. x is the percent targeted."""
    order = np.argsort(-score, kind="stable")
    n = len(score)
    gains = np.concatenate([[0.0], np.cumsum(uplift[order])]) / n * 1000
    x = np.arange(n + 1) / n * 100
    return x, gains


def qini_summary(score, uplift):
    """Returns x, model curve, best possible curve, random curve and the
    Qini coefficient (0 = random targeting, 1 = best possible ranking)."""
    x, gains = qini_curve(score, uplift)
    _, best = qini_curve(uplift, uplift)
    random = x / 100 * gains[-1]
    denom = _trapz(best, x) - _trapz(random, x)
    coef = (_trapz(gains, x) - _trapz(random, x)) / denom if denom > 0 else 0.0
    return x, gains, best, random, float(coef)


def avg_uplift_top(score, uplift, frac):
    k = max(1, int(round(frac * len(score))))
    idx = np.argsort(-score, kind="stable")[:k]
    return float(uplift[idx].mean())


def plot_qini(x, curves, best, random, path):
    plt.figure(figsize=(8, 5))
    for name, gains in curves.items():
        plt.plot(x, gains, label=name, linewidth=2)
    plt.plot(x, random, "k--", label="Random targeting")
    plt.plot(x, best, "k:", label="Best possible ranking")
    plt.axhline(0, color="gray", linewidth=0.6)
    plt.xlabel("% of customers targeted (highest score first)")
    plt.ylabel("Extra purchases per 1,000 customers")
    plt.title(f"Qini curves, ${LEVEL} discount (held-out, true effects)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    data_file = DATA_DIR / "retail_campaign.csv"
    truth_file = DATA_DIR / "ground_truth_effects.csv"
    if not data_file.exists() or not truth_file.exists():
        raise SystemExit("Run python backend\\src\\generate_data.py first.")
    df = pd.read_csv(data_file)
    truth = pd.read_csv(truth_file)
    data = df.merge(truth, on="customer_id", validate="one_to_one")

    X = data[CONFOUNDERS].to_numpy(dtype=float)
    Y = data["purchased"].to_numpy(dtype=float)
    T = data["discount"].to_numpy(dtype=int)

    # Same split as dml_engine.py, so results are comparable with Day 10
    train, test = train_test_split(
        np.arange(len(data)), test_size=0.3, random_state=SEED, stratify=T
    )
    true = data[f"true_uplift_{LEVEL}"].to_numpy()[test]

    scores = {}
    print("Fitting causal forest DML (a minute or two)...")
    forest = fit_dml(X[train], Y[train], T[train], kind="forest")
    scores[FOREST] = np.ravel(forest.effect(X[test], T0=0, T1=LEVEL))

    print("Fitting linear DML...")
    linear = fit_dml(X[train], Y[train], T[train], kind="linear")
    scores[LINEAR] = np.ravel(linear.effect(X[test], T0=0, T1=LEVEL))

    print("Fitting standard purchase-prediction model...")
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                         min_child_samples=30, random_state=SEED, verbose=-1)
    clf.fit(X[train], Y[train])
    scores[BASELINE] = -clf.predict_proba(X[test])[:, 1]

    rows, curves = [], {}
    x = best = random = None
    for name, score in scores.items():
        x, gains, best, random, coef = qini_summary(score, true)
        curves[name] = gains
        rows.append({
            "model": name,
            "qini_coefficient": coef,
            "avg_uplift_top_10pct": avg_uplift_top(score, true, 0.1),
            "avg_uplift_top_30pct": avg_uplift_top(score, true, 0.3),
            "avg_uplift_top_50pct": avg_uplift_top(score, true, 0.5),
        })
    rows.append({
        "model": "Best possible ranking (ceiling)", "qini_coefficient": 1.0,
        "avg_uplift_top_10pct": avg_uplift_top(true, true, 0.1),
        "avg_uplift_top_30pct": avg_uplift_top(true, true, 0.3),
        "avg_uplift_top_50pct": avg_uplift_top(true, true, 0.5),
    })
    table = pd.DataFrame(rows)

    print(f"\nHeld-out customers: {len(test)}. Average true uplift if everyone "
          f"is targeted: {true.mean():+.3f}\n")
    print(table.round(3).to_string(index=False))

    for name in (FOREST, LINEAR):
        mask = scores[name] > 0
        if mask.any():
            print(f"\n{name}: predicts a positive effect for {mask.mean():.1%} of customers; "
                  f"their average true uplift is {true[mask].mean():+.3f}")

    best_dml = max((FOREST, LINEAR), key=lambda m: float(
        table.loc[table.model == m, "qini_coefficient"].iloc[0]))
    print(f"\nSelected final model (higher Qini coefficient): {best_dml}")

    DOCS.mkdir(exist_ok=True)
    table.round(4).to_csv(DOCS / "model_comparison.csv", index=False)
    plot_qini(x, curves, best, random, DOCS / "qini_curves.png")

    grid = np.linspace(0, 100, 101)
    payload = {
        "note": "Computed on held-out synthetic customers using true individual effects.",
        "unit": "Extra purchases per 1,000 customers targeted",
        "discount": LEVEL,
        "held_out_customers": int(len(test)),
        "selected_model": best_dml,
        "x_percent": grid.tolist(),
        "curves": {n: np.round(np.interp(grid, x, g), 3).tolist() for n, g in curves.items()},
        "random": np.round(np.interp(grid, x, random), 3).tolist(),
        "best_possible": np.round(np.interp(grid, x, best), 3).tolist(),
    }
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_DIR / "qini_curves.json", "w") as f:
        json.dump(payload, f)
    print("\nSaved docs/model_comparison.csv, docs/qini_curves.png and "
          "frontend/public/demo/qini_curves.json")


if __name__ == "__main__":
    main()