"""Propensity score matching for EconoCausal.

1:1 nearest-neighbour matching (with replacement) on the logit of the
propensity score, with a caliper. Treatment is binary here (any discount
vs none). Estimates the ATT: the effect on customers who actually got a
discount. Multi-level and individual effects come from Double ML.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from causal_graph import CONFOUNDERS
from propensity import estimate_propensity, weighted_smd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "retail_campaign.csv"
TRUTH_FILE = ROOT / "data" / "ground_truth_effects.csv"
DOCS = ROOT / "docs"


def logit(p):
    return np.log(p / (1 - p))


def match(ps, t, caliper_sd=0.2):
    """Match each treated unit to its nearest untreated unit on logit(ps).

    Returns (treated_idx, control_idx, n_dropped). Treated units with no
    control within the caliper are dropped. Controls can be reused.
    """
    lp = logit(ps)
    caliper = caliper_sd * lp.std()
    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]

    nn = NearestNeighbors(n_neighbors=1).fit(lp[control_idx].reshape(-1, 1))
    dist, pos = nn.kneighbors(lp[treated_idx].reshape(-1, 1))
    dist, pos = dist.ravel(), pos.ravel()

    keep = dist <= caliper
    return treated_idx[keep], control_idx[pos[keep]], int((~keep).sum())


def matched_frame(df, treated_idx, control_idx):
    return pd.concat(
        [df.iloc[treated_idx].assign(treated=1),
         df.iloc[control_idx].assign(treated=0)],
        ignore_index=True,
    )


def att_from_pairs(df, treated_idx, control_idx, outcome="purchased"):
    y = df[outcome].to_numpy()
    diff = y[treated_idx] - y[control_idx]
    att = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))  # rough: ignores control reuse
    return att, se


def true_att(df, truth, treated_idx):
    """True average effect for the treated, at the discount each one got."""
    tr = truth.set_index("customer_id")
    rows = df.iloc[treated_idx]
    u10 = tr.loc[rows["customer_id"], "true_uplift_10"].to_numpy()
    u20 = tr.loc[rows["customer_id"], "true_uplift_20"].to_numpy()
    return float(np.where(rows["discount"].to_numpy() == 10, u10, u20).mean())


def balance_after_matching(df, t, matched, covariates=CONFOUNDERS):
    rows = {
        c: {
            "smd_before": weighted_smd(df[c], t),
            "smd_after_matching": weighted_smd(matched[c], matched["treated"]),
        }
        for c in covariates
    }
    return pd.DataFrame(rows).T


def plot_love(table, path):
    order = table["smd_before"].abs().sort_values().index
    y = np.arange(len(order))
    plt.figure(figsize=(8, 4))
    plt.scatter(table.loc[order, "smd_before"].abs(), y, color="tab:red", label="Before")
    plt.scatter(table.loc[order, "smd_after_matching"].abs(), y, color="tab:blue",
                label="After matching")
    plt.axvline(0.1, linestyle="--", color="gray", label="0.1 threshold")
    plt.yticks(y, order)
    plt.xlabel("Absolute standardized mean difference")
    plt.title("Covariate balance before and after matching")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    if not DATA_FILE.exists() or not TRUTH_FILE.exists():
        raise SystemExit("Run python backend\\src\\generate_data.py first.")
    df = pd.read_csv(DATA_FILE)
    truth = pd.read_csv(TRUTH_FILE)

    ps, t = estimate_propensity(df)
    ti, ci, dropped = match(ps, t)

    print(f"Treated customers: {int(t.sum())}")
    print(f"Matched pairs: {len(ti)} (dropped by caliper: {dropped})")
    reuse = pd.Series(ci).value_counts()
    print(f"Unique controls used: {len(reuse)} of {int((t == 0).sum())}")
    print(f"Most times one control was reused: {int(reuse.max())}")

    matched = matched_frame(df, ti, ci)
    table = balance_after_matching(df, t, matched)
    print("\nStandardized mean differences:")
    print(table.round(3))

    y = df["purchased"].to_numpy()
    naive = y[t == 1].mean() - y[t == 0].mean()
    att, se = att_from_pairs(df, ti, ci)
    truth_att = true_att(df, truth, ti)
    print("\nEffect of getting a discount (effect on the treated):")
    print(f"  Naive difference    : {naive:+.3f}")
    print(f"  Matched ATT         : {att:+.3f}  (95% CI {att - 1.96 * se:+.3f} to {att + 1.96 * se:+.3f})")
    print(f"  True ATT (synthetic): {truth_att:+.3f}")

    DOCS.mkdir(exist_ok=True)
    plot_love(table, DOCS / "love_plot_matching.png")
    table.round(4).to_csv(DOCS / "balance_table_matching.csv")
    print("\nSaved docs/love_plot_matching.png and docs/balance_table_matching.csv")


if __name__ == "__main__":
    main()