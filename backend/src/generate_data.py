"""Synthetic retail campaign data for EconoCausal.

Historical discounts are NOT random: the business targeted customers who
had been inactive for a while and were not loyalty members. Those same
traits also affect buying, so naive comparisons are biased (confounding).
True individual effects are saved separately for validating the DML model.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 10_000
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def discount_lift(d, response):
    """Effect on buy-logit of discount d; diminishing returns as d grows."""
    return response * (1.0 - np.exp(-d / 10.0))


def generate(n=N_CUSTOMERS, seed=SEED):
    rng = np.random.default_rng(seed)

    # ---- Customer features (confounders) ----
    age = rng.integers(18, 71, n)
    income = np.clip(rng.lognormal(10.8, 0.5, n), 15_000, 300_000)
    loyalty_member = rng.binomial(1, 0.35, n)
    past_purchases = rng.poisson(3 + 4 * loyalty_member)
    days_since_last_purchase = np.clip(rng.exponential(45, n), 1, 365)
    avg_basket_value = rng.gamma(4.0, 15.0, n)

    age_z = (age - 44) / 15
    income_z = (np.log(income) - 10.8) / 0.5
    recency_z = (days_since_last_purchase - 45) / 45
    freq_z = (past_purchases - 5) / 3

    # Latent price sensitivity (not in the output file)
    price_sens = sigmoid(
        -0.8 * income_z + 0.3 * recency_z - 0.2 * loyalty_member
        + rng.normal(0, 0.3, n)
    )

    # ---- Treatment assignment (confounded) ----
    p_any = sigmoid(-0.2 + 0.9 * recency_z - 0.6 * loyalty_member + 0.3 * price_sens)
    got_discount = rng.binomial(1, p_any)
    p_20 = sigmoid(-0.3 + 0.5 * recency_z)
    big = rng.binomial(1, p_20)
    discount = np.where(got_discount == 1, np.where(big == 1, 20, 10), 0)

    # ---- Outcome ----
    base_logit = (
        -0.8 + 0.5 * freq_z + 0.9 * loyalty_member
        - 0.6 * recency_z + 0.2 * age_z
    )
    response = 3.0 * price_sens - 1.2  # negative => "sleeping dog"
    p0 = sigmoid(base_logit)
    p10 = sigmoid(base_logit + discount_lift(10, response))
    p20_buy = sigmoid(base_logit + discount_lift(20, response))
    p_obs = np.select([discount == 0, discount == 10], [p0, p10], default=p20_buy)
    purchased = rng.binomial(1, p_obs)
    revenue = purchased * avg_basket_value * rng.lognormal(0, 0.1, n)

    customers = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": age,
        "income": income.round(0),
        "loyalty_member": loyalty_member,
        "past_purchases": past_purchases,
        "days_since_last_purchase": days_since_last_purchase.round(1),
        "avg_basket_value": avg_basket_value.round(2),
        "discount": discount,
        "purchased": purchased,
        "revenue": revenue.round(2),
    })

    up10, up20 = p10 - p0, p20_buy - p0
    segment = np.select(
        [up20 >= 0.08, up20 < 0, p0 >= 0.6],
        ["persuadable", "sleeping_dog", "sure_thing"],
        default="lost_cause",
    )
    truth = pd.DataFrame({
        "customer_id": customers["customer_id"],
        "true_baseline_prob": p0.round(4),
        "true_uplift_10": up10.round(4),
        "true_uplift_20": up20.round(4),
        "true_segment": segment,
    })
    return customers, truth


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    customers, truth = generate()
    customers.to_csv(DATA_DIR / "retail_campaign.csv", index=False)
    customers.head(500).to_csv(DATA_DIR / "retail_campaign_sample.csv", index=False)
    truth.to_csv(DATA_DIR / "ground_truth_effects.csv", index=False)

    rates = customers.groupby("discount")["purchased"].mean()
    naive = rates[20] - rates[0]
    true_avg = truth["true_uplift_20"].mean()
    print(f"Rows: {len(customers)}")
    print("Purchase rate by discount:\n", rates.round(3).to_string())
    print("Segments:\n", truth["true_segment"].value_counts().to_string())
    print(f"Naive effect of $20 discount : {naive:+.3f}")
    print(f"TRUE avg effect of $20 discount: {true_avg:+.3f}")