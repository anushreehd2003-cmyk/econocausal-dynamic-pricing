"""Drift detector demo (Week 4, Day 20).

Runs the detector on three batches of new customers: one from the same
process as the training data, and two with shifts built in. The detector
should stay quiet on the first and raise alerts on the others.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drift import PSI_ALERT, PSI_WARN, drift_report, format_report, untreated_counts
from generate_data import generate
from model_store import load_bundle, model_path, predict_effects

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def scenario_frames(n=3000):
    rng = np.random.default_rng(2026)
    same, _ = generate(n=n, seed=999)  # new customers, same process as training

    shifted = same.copy()  # customers become less active and less loyal
    shifted["days_since_last_purchase"] = np.clip(
        shifted["days_since_last_purchase"] * 1.8, 1, 365
    )
    shifted["loyalty_member"] = (rng.random(n) < 0.10).astype(int)

    behaviour = shifted.copy()  # ...and about 30% of purchases disappear
    keep = rng.random(n) < 0.7
    behaviour["purchased"] = (behaviour["purchased"] * keep).astype(int)

    return {
        "A. No change": same,
        "B. Less active, less loyal": shifted,
        "C. B plus lower purchasing": behaviour,
    }


def main():
    if not model_path().exists():
        raise SystemExit("Model not found. Run python backend\\src\\train_final_model.py first.")
    bundle = load_bundle()
    reference = bundle.get("reference")
    if reference is None:
        raise SystemExit("This model file has no reference data. Rerun train_final_model.py.")

    features = bundle["features"]
    ref_effects = predict_effects(bundle, reference["X"])
    ref_untreated = (reference["untreated_buys"], reference["untreated_n"])

    rows = []
    for name, frame in scenario_frames().items():
        X = frame[features].to_numpy(dtype=float)
        report = drift_report(
            features, reference["X"], X, ref_effects, predict_effects(bundle, X),
            ref_untreated, untreated_counts(frame["discount"], frame["purchased"]),
        )
        print(f"\n=== Scenario {name} ===")
        print(format_report(report))
        for item in report["items"]:
            rows.append({"scenario": name, "variable": item["name"],
                         "psi": item["psi"], "severity": item["severity"]})
        if report["behaviour"]:
            rows.append({"scenario": name, "variable": "no-discount purchase rate",
                         "psi": np.nan, "severity": report["behaviour"]["severity"]})

    table = pd.DataFrame(rows)
    DOCS.mkdir(exist_ok=True)
    table.to_csv(DOCS / "drift_demo.csv", index=False)

    psi_rows = table.dropna(subset=["psi"])
    variables = list(dict.fromkeys(psi_rows["variable"]))
    scenarios = list(dict.fromkeys(psi_rows["scenario"]))
    height = 0.8 / len(scenarios)
    plt.figure(figsize=(9, 5.5))
    for k, scenario in enumerate(scenarios):
        sub = psi_rows[psi_rows["scenario"] == scenario].set_index("variable")
        values = [sub.loc[v, "psi"] for v in variables]
        plt.barh(np.arange(len(variables)) + k * height, values, height=height, label=scenario)
    plt.axvline(PSI_WARN, color="orange", linestyle="--", label=f"warning ({PSI_WARN})")
    plt.axvline(PSI_ALERT, color="red", linestyle="--", label=f"alert ({PSI_ALERT})")
    plt.yticks(np.arange(len(variables)) + height, variables)
    plt.gca().invert_yaxis()
    plt.xlabel("Population Stability Index (PSI)")
    plt.title("Drift detector demo")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(DOCS / "drift_demo.png", dpi=150)
    plt.close()
    print("\nSaved docs/drift_demo.csv and docs/drift_demo.png")


if __name__ == "__main__":
    main()