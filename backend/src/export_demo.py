"""Export dashboard demo data: customer features plus predicted effects.

Ground-truth columns are deliberately excluded.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "frontend" / "public" / "demo"

FEATURES = [
    "customer_id", "age", "income", "loyalty_member", "past_purchases",
    "days_since_last_purchase", "avg_basket_value",
]
PREDICTIONS = ["customer_id", "is_test", "ite_10", "ite_20"]


def build_demo_frame(campaign, predictions):
    frame = campaign[FEATURES].merge(
        predictions[PREDICTIONS], on="customer_id", validate="one_to_one"
    )
    frame["income"] = frame["income"].round(0).astype(int)
    frame[["ite_10", "ite_20"]] = frame[["ite_10", "ite_20"]].round(4)
    return frame


def main():
    campaign_file = DATA_DIR / "retail_campaign.csv"
    pred_file = DATA_DIR / "ite_predictions.csv"
    if not campaign_file.exists():
        raise SystemExit("Run python backend\\src\\generate_data.py first.")
    if not pred_file.exists():
        raise SystemExit("Run python backend\\src\\dml_engine.py first.")

    frame = build_demo_frame(pd.read_csv(campaign_file), pd.read_csv(pred_file))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / "customers_ite.csv", index=False)
    print(f"Saved {len(frame)} customers to frontend/public/demo/customers_ite.csv")
    print("Columns:", ", ".join(frame.columns))


if __name__ == "__main__":
    main()