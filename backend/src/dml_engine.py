"""Double Machine Learning for EconoCausal (EconML).

Estimates the individual treatment effect (ITE) of a $10 and a $20
discount on purchase probability, relative to no discount. The binary
outcome is modelled on the probability scale (linear probability),
which is the same scale as the true uplift in the synthetic ground truth.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from causal_graph import CONFOUNDERS

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOCS = ROOT / "docs"
LEVELS = [10, 20]
SEED = 42


def nuisance_models(seed=SEED):
    params = dict(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=seed, verbose=-1,
    )
    return LGBMRegressor(**params), LGBMClassifier(**params)


def build_model(kind="forest", seed=SEED):
    model_y, model_t = nuisance_models(seed)
    common = dict(
        model_y=model_y, model_t=model_t, discrete_treatment=True,
        cv=5, random_state=seed,
    )
    if kind == "forest":
        return CausalForestDML(n_estimators=400, min_samples_leaf=30, **common)
    if kind == "linear":
        return LinearDML(
            featurizer=PolynomialFeatures(degree=2, include_bias=False), **common
        )
    raise ValueError("kind must be 'forest' or 'linear'")


def fit_dml(X, Y, T, kind="forest", seed=SEED):
    model = build_model(kind, seed)
    model.fit(Y, T, X=X)
    return model


def predict_effects(model, X, alpha=0.1):
    """Effect of each discount level vs no discount, with 90% intervals."""
    out = {}
    for level in LEVELS:
        out[f"ite_{level}"] = np.asarray(model.effect(X, T0=0, T1=level)).ravel()
        lo, hi = model.effect_interval(X, T0=0, T1=level, alpha=alpha)
        out[f"ite_{level}_lo"] = np.asarray(lo).ravel()
        out[f"ite_{level}_hi"] = np.asarray(hi).ravel()
    return pd.DataFrame(out)


def evaluate(pred, true):
    err = pred - true
    return {
        "mean_pred": float(pred.mean()),
        "mean_true": float(true.mean()),
        "corr": float(np.corrcoef(pred, true)[0, 1]),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "rmse_constant": float(true.std()),  # error if we always predicted the mean
        "sign_agree": float(((pred > 0) == (true > 0)).mean()),
    }


def decile_table(pred, true, n_bins=10):
    order = np.argsort(-pred)  # decile 1 = highest predicted uplift
    groups = np.array_split(order, n_bins)
    return pd.DataFrame({
        "decile": np.arange(1, n_bins + 1),
        "avg_predicted": [pred[g].mean() for g in groups],
        "avg_true": [true[g].mean() for g in groups],
    })


def plot_pred_vs_true(pred, true, level, path):
    plt.figure(figsize=(5.5, 5))
    plt.scatter(true, pred, s=6, alpha=0.3)
    lo = min(true.min(), pred.min())
    hi = max(true.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "k--", label="perfect prediction")
    plt.xlabel("True uplift (synthetic ground truth)")
    plt.ylabel("Predicted uplift (DML)")
    plt.title(f"${level} discount: predicted vs true (held-out)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_deciles(table, level, path):
    x = np.arange(len(table))
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.2, table["avg_true"], width=0.4, label="Average true uplift")
    plt.bar(x + 0.2, table["avg_predicted"], width=0.4, label="Average predicted uplift")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, table["decile"])
    plt.xlabel("Decile by predicted uplift (1 = highest)")
    plt.ylabel("Uplift in purchase probability")
    plt.title(f"${level} discount: uplift by predicted decile (held-out)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["forest", "linear"], default="forest")
    args = parser.parse_args()

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

    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.3, random_state=SEED, stratify=T
    )
    print(f"Fitting {args.model} DML on {len(train_idx)} customers "
          f"(testing on {len(test_idx)}). This can take a minute or two...")
    model = fit_dml(X[train_idx], Y[train_idx], T[train_idx], kind=args.model)
    pred = predict_effects(model, X)

    for level in LEVELS:
        p = pred[f"ite_{level}"].to_numpy()[test_idx]
        tv = data[f"true_uplift_{level}"].to_numpy()[test_idx]
        naive = Y[T == level].mean() - Y[T == 0].mean()
        m = evaluate(p, tv)
        print(f"\n=== ${level} discount (held-out customers) ===")
        print(f"Naive effect (raw data)    : {naive:+.3f}")
        print(f"Mean predicted effect      : {m['mean_pred']:+.3f}")
        print(f"Mean true effect           : {m['mean_true']:+.3f}")
        print(f"Correlation (pred vs true) : {m['corr']:.3f}")
        print(f"RMSE                       : {m['rmse']:.3f} "
              f"(always predicting the mean: {m['rmse_constant']:.3f})")
        print(f"Sign agreement             : {m['sign_agree']:.3f}")

        if level == 20:
            table = decile_table(p, tv)
            print("\nUplift by predicted decile ($20, held-out):")
            print(table.round(3).to_string(index=False))
            DOCS.mkdir(exist_ok=True)
            plot_pred_vs_true(p, tv, level, DOCS / "dml_pred_vs_true.png")
            plot_deciles(table, level, DOCS / "dml_deciles.png")
            table.round(4).to_csv(DOCS / "dml_decile_table.csv", index=False)

    # Predictions for the dashboard. No ground truth columns in this file.
    out = pred.copy()
    out.insert(0, "customer_id", data["customer_id"].to_numpy())
    out.insert(1, "is_test", np.isin(np.arange(len(data)), test_idx).astype(int))
    out.round(5).to_csv(DATA_DIR / "ite_predictions.csv", index=False)
    out.head(500).round(5).to_csv(DATA_DIR / "ite_predictions_sample.csv", index=False)
    print("\nSaved data/ite_predictions.csv, data/ite_predictions_sample.csv "
          "and the plots in docs/")


if __name__ == "__main__":
    main()