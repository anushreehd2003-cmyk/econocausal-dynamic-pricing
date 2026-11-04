"""Train the production Double ML model on ALL customers and save it for the API.

Evaluation numbers in the docs come from the held-out experiment in
dml_engine.py. This model is trained on every customer, as it would be in
production, and is what the API serves.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dml_engine import fit_dml
from model_store import FEATURES, make_bundle, model_path, predict_effects, save_bundle

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "retail_campaign.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["forest", "linear"], default="forest")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        raise SystemExit("Run python backend\\src\\generate_data.py first.")
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES].to_numpy(dtype=float)
    Y = df["purchased"].to_numpy(dtype=float)
    T = df["discount"].to_numpy(dtype=int)

    print(f"Training {args.model} DML on all {len(df)} customers "
          "(this can take a minute or two)...")
    start = time.time()
    model = fit_dml(X, Y, T, kind=args.model)
    print(f"Trained in {time.time() - start:.0f} seconds.")

    bundle = make_bundle(model, X, kind=args.model)
    path = save_bundle(bundle)
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")

    # Sanity check on a sample: the average effects should be small and positive
    effects = predict_effects(bundle, X[:2000])
    for level in bundle["levels"]:
        print(f"Mean predicted effect of ${level} (first 2000 customers): "
              f"{np.mean(effects[f'ite_{level}']):+.3f}")
    print("Model path for the API:", model_path())


if __name__ == "__main__":
    main()