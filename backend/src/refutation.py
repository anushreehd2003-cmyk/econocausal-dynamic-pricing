"""Causal audit for EconoCausal: refutation tests.

Part A: DoWhy refuters on the $20-vs-none comparison (linear estimator).
Part B: the same ideas applied to the Double ML model.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dowhy import CausalModel

from causal_graph import CONFOUNDERS, OUTCOME
from dml_engine import fit_dml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOCS = ROOT / "docs"
LEVEL = 20
SEED = 42


def build_gml(treatment, confounders=CONFOUNDERS, outcome=OUTCOME):
    nodes = list(confounders) + [treatment, outcome]
    parts = ["graph [directed 1"]
    parts += [f'node [id "{n}" label "{n}"]' for n in nodes]
    for c in confounders:
        parts.append(f'edge [source "{c}" target "{treatment}"]')
        parts.append(f'edge [source "{c}" target "{outcome}"]')
    parts.append(f'edge [source "{treatment}" target "{outcome}"]')
    parts.append("]")
    return "\n".join(parts)


def _summarize(ref):
    p = None
    res = getattr(ref, "refutation_result", None)
    if isinstance(res, dict):
        p = res.get("p_value")
    return float(ref.estimated_effect), float(np.mean(ref.new_effect)), p


def dowhy_refutations(df, n_sims=20):
    sub = df[df["discount"].isin([0, LEVEL])].copy()
    sub["treated"] = (sub["discount"] == LEVEL).astype(int)
    sub = sub[CONFOUNDERS + ["treated", OUTCOME]]

    model = CausalModel(data=sub, treatment="treated", outcome=OUTCOME,
                        graph=build_gml("treated"))
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    est = model.estimate_effect(estimand, method_name="backdoor.linear_regression")

    tests = {
        "placebo_treatment": dict(method_name="placebo_treatment_refuter",
                                  placebo_type="permute", num_simulations=n_sims),
        "random_common_cause": dict(method_name="random_common_cause",
                                    num_simulations=n_sims),
        "data_subset": dict(method_name="data_subset_refuter",
                            subset_fraction=0.8, num_simulations=n_sims),
    }
    rows = []
    for name, kwargs in tests.items():
        ref = model.refute_estimate(estimand, est, **kwargs)
        original, new, p = _summarize(ref)
        rows.append({"part": "A: DoWhy (linear)", "test": name,
                     "original_effect": original, "refuted_effect": new,
                     "p_value": p})
    return rows, sub.index


def dml_ate(X, Y, T, seed=SEED):
    model = fit_dml(X, Y, T, kind="linear", seed=seed)
    return float(np.ravel(model.ate(X, T0=0, T1=LEVEL))[0])


def dml_refutations(df, reps=3):
    rng = np.random.default_rng(SEED)
    X = df[CONFOUNDERS].to_numpy(dtype=float)
    Y = df[OUTCOME].to_numpy(dtype=float)
    T = df["discount"].to_numpy(dtype=int)
    n = len(df)

    original = dml_ate(X, Y, T)
    placebo = [dml_ate(X, Y, rng.permutation(T), seed=SEED + i) for i in range(reps)]

    noise = []
    for i in range(reps):
        Xa = np.column_stack([X, rng.normal(size=n)])
        noise.append(dml_ate(Xa, Y, T, seed=SEED + i))

    subset = []
    for i in range(reps):
        idx = rng.choice(n, int(0.8 * n), replace=False)
        subset.append(dml_ate(X[idx], Y[idx], T[idx], seed=SEED + i))

    drop = CONFOUNDERS.index("days_since_last_purchase")
    omitted = dml_ate(np.delete(X, drop, axis=1), Y, T)

    def row(test, value, verdict):
        return {"part": "B: Double ML", "test": test, "original_effect": original,
                "refuted_effect": value, "verdict": verdict}

    p_mean, n_mean, s_mean = np.mean(placebo), np.mean(noise), np.mean(subset)
    return [
        row("placebo_treatment", p_mean,
            "pass" if abs(p_mean) < 0.5 * abs(original) else "check"),
        row("random_common_cause", n_mean,
            "pass" if abs(n_mean - original) < 0.01 else "check"),
        row("data_subset_80pct", s_mean,
            "pass" if abs(s_mean - original) < 0.01 else "check"),
        row("omit_days_since_last_purchase", omitted,
            "expected to differ"),
    ]


def plot_summary(dml_rows, true_ate, path):
    labels = ["Original DML"] + [r["test"] for r in dml_rows] + ["True average effect"]
    values = [dml_rows[0]["original_effect"]] + [r["refuted_effect"] for r in dml_rows] + [true_ate]
    colors = ["tab:blue", "tab:gray", "tab:green", "tab:green", "tab:red", "black"]
    plt.figure(figsize=(8, 4.5))
    y = np.arange(len(labels))
    plt.barh(y, values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.yticks(y, labels)
    plt.xlabel(f"Average effect of a ${LEVEL} discount on purchase probability")
    plt.title("Double ML refutation tests")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=20, help="DoWhy simulations per test")
    parser.add_argument("--reps", type=int, default=3, help="repeats per Double ML test")
    args = parser.parse_args()

    data_file = DATA_DIR / "retail_campaign.csv"
    truth_file = DATA_DIR / "ground_truth_effects.csv"
    if not data_file.exists() or not truth_file.exists():
        raise SystemExit("Run python backend\\src\\generate_data.py first.")
    df = pd.read_csv(data_file)
    truth = pd.read_csv(truth_file)

    print("Part A: DoWhy refuters ($20 vs no discount, linear estimator)...")
    a_rows, sub_index = dowhy_refutations(df, n_sims=args.sims)
    sub_true = truth.loc[sub_index, "true_uplift_20"].mean()
    for r in a_rows:
        p = "n/a" if r["p_value"] is None else f"{r['p_value']:.3f}"
        print(f"  {r['test']:<22} original {r['original_effect']:+.3f} "
              f"-> refuted {r['refuted_effect']:+.3f}   p-value {p}")
    print(f"  True average effect for these customers: {sub_true:+.3f}")

    print("\nPart B: Double ML refutations (this takes a few minutes)...")
    b_rows = dml_refutations(df, reps=args.reps)
    true_ate = truth["true_uplift_20"].mean()
    print(f"  Original DML average effect: {b_rows[0]['original_effect']:+.3f} "
          f"(true: {true_ate:+.3f})")
    for r in b_rows:
        print(f"  {r['test']:<30} -> {r['refuted_effect']:+.3f}   [{r['verdict']}]")

    DOCS.mkdir(exist_ok=True)
    pd.DataFrame(a_rows + b_rows).round(4).to_csv(DOCS / "refutation_results.csv", index=False)
    plot_summary(b_rows, true_ate, DOCS / "refutation_summary.png")
    print("\nSaved docs/refutation_results.csv and docs/refutation_summary.png")


if __name__ == "__main__":
    main()