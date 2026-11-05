"""Data drift detection for EconoCausal.

Compares new customer data with the data the model was trained on:
  - input drift: Population Stability Index (PSI) and a KS test per feature
  - prediction drift: PSI on the predicted effect of each discount
  - behaviour drift: purchase rate of customers who received no discount

Thresholds are common rules of thumb, not statistical guarantees.

Check a CSV of new customers from the command line:
    python backend\\src\\drift.py path\\to\\new_customers.csv
Exit code: 0 = ok, 1 = warning, 2 = alert.
"""
import argparse
import sys

import numpy as np
from scipy import stats

PSI_WARN = 0.10     # moderate shift
PSI_ALERT = 0.25    # large shift
RATE_WARN = 0.02    # purchase-rate change (absolute) worth a warning
RATE_ALERT = 0.05   # purchase-rate change (absolute) worth an alert
RATE_P = 0.01       # ...and it must also be statistically clear
MIN_BATCH = 100     # fewer customers than this gives unreliable drift tests
MIN_UNTREATED = 50  # fewer untreated customers than this skips the rate check
EPS = 1e-4
ORDER = {"ok": 0, "warning": 1, "alert": 2}


def psi(reference, current, n_bins=10):
    """Population Stability Index between two samples.

    Rule of thumb: below 0.10 stable, 0.10 to 0.25 moderate shift, above
    0.25 large shift. Features with few distinct values (such as loyalty
    status) use each value as a category; others use reference deciles."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    unique = np.unique(ref)
    if len(unique) <= n_bins:
        cuts = (unique[:-1] + unique[1:]) / 2
    else:
        cuts = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))

    def shares(values):
        idx = np.searchsorted(cuts, values, side="right")
        counts = np.bincount(idx, minlength=len(cuts) + 1).astype(float)
        return np.clip(counts / max(counts.sum(), 1.0), EPS, None)

    p, q = shares(ref), shares(cur)
    return float(np.sum((q - p) * np.log(q / p)))


def _num(x):
    """Readable number: 0.35, 44.9, 52,000."""
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def severity_from_psi(value):
    if value >= PSI_ALERT:
        return "alert"
    if value >= PSI_WARN:
        return "warning"
    return "ok"


def rate_drift(ref_buyers, ref_n, cur_buyers, cur_n):
    """Two-proportion z-test on purchase rates, with a practical-size rule."""
    p1, p2 = ref_buyers / ref_n, cur_buyers / cur_n
    pooled = (ref_buyers + cur_buyers) / (ref_n + cur_n)
    se = np.sqrt(pooled * (1 - pooled) * (1 / ref_n + 1 / cur_n))
    z = (p2 - p1) / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    diff = p2 - p1
    severity = "ok"
    if p_value < RATE_P and abs(diff) >= RATE_ALERT:
        severity = "alert"
    elif p_value < RATE_P and abs(diff) >= RATE_WARN:
        severity = "warning"
    return {
        "reference_rate": round(float(p1), 4),
        "current_rate": round(float(p2), 4),
        "difference": round(float(diff), 4),
        "p_value": round(p_value, 5),
        "severity": severity,
    }


def _distribution_item(kind, name, ref, cur):
    value = psi(ref, cur)
    ks = stats.ks_2samp(ref, cur)
    return {
        "kind": kind,
        "name": name,
        "psi": round(value, 4),
        "ks_statistic": round(float(ks.statistic), 4),
        "ks_p_value": round(float(ks.pvalue), 5),
        "reference_mean": round(float(np.mean(ref)), 4),
        "current_mean": round(float(np.mean(cur)), 4),
        "severity": severity_from_psi(value),
    }


def drift_report(features, ref_X, cur_X, ref_effects=None, cur_effects=None,
                 ref_untreated=None, cur_untreated=None):
    """Compare a new batch with the training data.

    ref_effects / cur_effects: dicts like {"ite_10": array, "ite_20": array}.
    ref_untreated / cur_untreated: (buyers, customers) among customers who
    received no discount. Optional checks are skipped when inputs are missing."""
    ref_X = np.asarray(ref_X, dtype=float)
    cur_X = np.asarray(cur_X, dtype=float)
    if len(cur_X) < MIN_BATCH:
        raise ValueError(
            f"Need at least {MIN_BATCH} customers for a drift check, got {len(cur_X)}."
        )

    items = [
        _distribution_item("input", name, ref_X[:, i], cur_X[:, i])
        for i, name in enumerate(features)
    ]
    skipped = []

    if ref_effects is not None and cur_effects is not None:
        for key in sorted(ref_effects):
            items.append(_distribution_item("prediction", key, ref_effects[key], cur_effects[key]))
    else:
        skipped.append("prediction drift")

    behaviour = None
    if (ref_untreated is not None and cur_untreated is not None
            and cur_untreated[1] >= MIN_UNTREATED and ref_untreated[1] >= MIN_UNTREATED):
        behaviour = rate_drift(ref_untreated[0], ref_untreated[1],
                               cur_untreated[0], cur_untreated[1])
        behaviour["customers_without_discount"] = int(cur_untreated[1])
    else:
        skipped.append("behaviour drift (needs discount and purchased columns, "
                       f"with at least {MIN_UNTREATED} customers who got no discount)")

    levels = [i["severity"] for i in items] + ([behaviour["severity"]] if behaviour else [])
    status = max(levels, key=lambda s: ORDER[s])

    messages = []
    for item in items:
        if item["severity"] == "ok":
            continue
        if item["kind"] == "input":
            messages.append(
                f"{item['name']}: new customers look different from the training data "
                f"(PSI {item['psi']:.2f}, average {_num(item['reference_mean'])} to "
                f"{_num(item['current_mean'])})."
            )
        else:
            level = item["name"].replace("ite_", "$")
            messages.append(
                f"Predicted effect of a {level} discount has shifted "
                f"(PSI {item['psi']:.2f}, average {item['reference_mean'] * 100:+.1f} to "
                f"{item['current_mean'] * 100:+.1f} points)."
            )
    if behaviour and behaviour["severity"] != "ok":
        messages.append(
            "Purchase rate of customers who received no discount moved from "
            f"{behaviour['reference_rate']:.1%} to {behaviour['current_rate']:.1%} "
            f"(p-value {behaviour['p_value']:.4f})."
        )

    recommendation = {
        "ok": "No meaningful drift detected. The model can be used as it is.",
        "warning": "Some drift detected. Keep monitoring, and consider retraining if it persists.",
        "alert": "Significant drift detected. Retrain the model on recent data before relying on new prescriptions.",
    }[status]

    return {
        "status": status,
        "n_reference": int(len(ref_X)),
        "n_current": int(len(cur_X)),
        "items": items,
        "behaviour": behaviour,
        "skipped": skipped,
        "messages": messages,
        "recommendation": recommendation,
        "thresholds": {
            "psi_warning": PSI_WARN,
            "psi_alert": PSI_ALERT,
            "rate_warning": RATE_WARN,
            "rate_alert": RATE_ALERT,
        },
    }


def untreated_counts(discount, purchased):
    """(buyers, customers) among customers who received no discount, or None."""
    if discount is None or purchased is None:
        return None
    discount = np.asarray(discount, dtype=float)
    purchased = np.asarray(purchased, dtype=float)
    mask = (discount == 0) & np.isfinite(purchased)
    return int(purchased[mask].sum()), int(mask.sum())


def format_report(report):
    lines = [f"Drift status: {report['status'].upper()}  "
             f"({report['n_current']} new customers vs {report['n_reference']} reference)"]
    lines.append(f"{'check':<12}{'variable':<28}{'PSI':>7}{'status':>10}")
    for item in report["items"]:
        lines.append(f"{item['kind']:<12}{item['name']:<28}{item['psi']:>7.3f}{item['severity']:>10}")
    b = report["behaviour"]
    if b:
        lines.append(f"{'behaviour':<12}{'no-discount purchase rate':<28}"
                     f"{b['reference_rate']:>7.1%}->{b['current_rate']:.1%}{b['severity']:>8}")
    for note in report["skipped"]:
        lines.append(f"(skipped: {note})")
    for message in report["messages"]:
        lines.append("- " + message)
    lines.append(report["recommendation"])
    return "\n".join(lines)


def main():
    import pandas as pd

    from model_store import load_bundle, model_path, predict_effects

    parser = argparse.ArgumentParser(description="Check a CSV of new customers for drift.")
    parser.add_argument("csv", help="CSV with the feature columns (and optionally discount, purchased)")
    args = parser.parse_args()

    if not model_path().exists():
        raise SystemExit("Model not found. Run python backend\\src\\train_final_model.py first.")
    bundle = load_bundle()
    reference = bundle.get("reference")
    if reference is None:
        raise SystemExit("This model file has no reference data. Rerun train_final_model.py.")

    df = pd.read_csv(args.csv)
    missing = [f for f in bundle["features"] if f not in df.columns]
    if missing:
        raise SystemExit("CSV is missing columns: " + ", ".join(missing))
    cur_X = df[bundle["features"]].to_numpy(dtype=float)

    ref_effects = predict_effects(bundle, reference["X"])
    cur_effects = predict_effects(bundle, cur_X)
    cur_untreated = None
    if "discount" in df.columns and "purchased" in df.columns:
        cur_untreated = untreated_counts(df["discount"], df["purchased"])

    report = drift_report(
        bundle["features"], reference["X"], cur_X, ref_effects, cur_effects,
        (reference["untreated_buys"], reference["untreated_n"]), cur_untreated,
    )
    print(format_report(report))
    sys.exit(ORDER[report["status"]])


if __name__ == "__main__":
    main()