"""Propensity scores and covariate balance diagnostics for EconoCausal.

Treatment here is binary (any discount vs none). The multi-level
$0/$10/$20 treatment is handled by Double ML later.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_graph import CONFOUNDERS

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "retail_campaign.csv"
DOCS = ROOT / "docs"


def estimate_propensity(df, covariates=CONFOUNDERS, seed=42, clip=0.01):
    """Cross-fitted propensity scores P(any discount | covariates)."""
    X = df[covariates]
    t = (df["discount"] > 0).astype(int).to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    ps = cross_val_predict(model, X, t, cv=cv, method="predict_proba")[:, 1]
    return np.clip(ps, clip, 1 - clip), t


def ipw_weights(ps, t):
    """Inverse-probability weights for the average treatment effect."""
    return t / ps + (1 - t) / (1 - ps)


def weighted_smd(x, t, w=None):
    """Standardized mean difference (treated minus untreated)."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t)
    w = np.ones(len(x)) if w is None else np.asarray(w, dtype=float)

    def wmean(v, ww):
        return np.sum(ww * v) / np.sum(ww)

    def wvar(v, ww):
        m = wmean(v, ww)
        return np.sum(ww * (v - m) ** 2) / np.sum(ww)

    t1, t0 = t == 1, t == 0
    m1, m0 = wmean(x[t1], w[t1]), wmean(x[t0], w[t0])
    v1, v0 = wvar(x[t1], w[t1]), wvar(x[t0], w[t0])
    pooled = np.sqrt((v1 + v0) / 2)
    return 0.0 if pooled == 0 else float((m1 - m0) / pooled)


def balance_table(df, t, ps, covariates=CONFOUNDERS):
    w = ipw_weights(ps, t)
    rows = {
        c: {
            "smd_before": weighted_smd(df[c], t),
            "smd_after_ipw": weighted_smd(df[c], t, w),
        }
        for c in covariates
    }
    return pd.DataFrame(rows).T


def plot_overlap(ps, t, path):
    plt.figure(figsize=(8, 4))
    bins = np.linspace(0, 1, 41)
    plt.hist(ps[t == 0], bins=bins, alpha=0.6, density=True, label="No discount")
    plt.hist(ps[t == 1], bins=bins, alpha=0.6, density=True, label="Got discount")
    plt.xlabel("Propensity score")
    plt.ylabel("Density")
    plt.title("Propensity score overlap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_love(table, path):
    order = table["smd_before"].abs().sort_values().index
    y = np.arange(len(order))
    plt.figure(figsize=(8, 4))
    plt.scatter(table.loc[order, "smd_before"].abs(), y, color="tab:red", label="Before")
    plt.scatter(table.loc[order, "smd_after_ipw"].abs(), y, color="tab:blue", label="After IPW")
    plt.axvline(0.1, linestyle="--", color="gray", label="0.1 threshold")
    plt.yticks(y, order)
    plt.xlabel("Absolute standardized mean difference")
    plt.title("Covariate balance (Love plot)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    if not DATA_FILE.exists():
        raise SystemExit("Run python backend\\src\\generate_data.py first.")
    df = pd.read_csv(DATA_FILE)

    ps, t = estimate_propensity(df)
    print(f"Treated share: {t.mean():.3f}")
    print(f"Propensity model AUC: {roc_auc_score(t, ps):.3f}")
    print(f"Score range (treated)  : {ps[t == 1].min():.3f} to {ps[t == 1].max():.3f}")
    print(f"Score range (untreated): {ps[t == 0].min():.3f} to {ps[t == 0].max():.3f}")

    w = ipw_weights(ps, t)
    ess = w.sum() ** 2 / (w ** 2).sum()
    print(f"Effective sample size after weighting: {ess:.0f} of {len(df)}")

    table = balance_table(df, t, ps)
    print("\nStandardized mean differences:")
    print(table.round(3))

    DOCS.mkdir(exist_ok=True)
    plot_overlap(ps, t, DOCS / "propensity_overlap.png")
    plot_love(table, DOCS / "love_plot_ipw.png")
    table.round(4).to_csv(DOCS / "balance_table.csv")
    print("\nSaved docs/propensity_overlap.png, docs/love_plot_ipw.png, docs/balance_table.csv")


if __name__ == "__main__":
    main()