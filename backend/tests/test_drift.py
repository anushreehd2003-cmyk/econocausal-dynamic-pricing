import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from api import create_app
from dml_engine import fit_dml
from drift import (
    MIN_BATCH,
    PSI_ALERT,
    PSI_WARN,
    drift_report,
    format_report,
    psi,
    rate_drift,
    severity_from_psi,
    untreated_counts,
)
from generate_data import generate
from model_store import FEATURES, make_bundle


def shift(df, seed=0):
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["days_since_last_purchase"] = np.clip(out["days_since_last_purchase"] * 1.8, 1, 365)
    out["loyalty_member"] = (rng.random(len(out)) < 0.10).astype(int)
    return out


def test_psi_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < PSI_WARN / 5


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    small = psi(a, rng.normal(0.3, 1, 5000))
    large = psi(a, rng.normal(1.5, 1, 5000))
    assert small < large and large > PSI_ALERT
    assert np.isfinite(psi(a, a + 100))  # values outside the reference range


def test_psi_handles_binary_and_constant_features():
    ref = np.array([0] * 65 + [1] * 35)
    assert psi(ref, ref) < 1e-9
    assert psi(ref, np.array([0] * 90 + [1] * 10)) > PSI_ALERT
    assert psi(np.ones(100), np.ones(100)) == 0.0


def test_severity_thresholds():
    assert severity_from_psi(PSI_WARN - 0.01) == "ok"
    assert severity_from_psi(PSI_WARN) == "warning"
    assert severity_from_psi(PSI_ALERT) == "alert"


def test_rate_drift_cases():
    assert rate_drift(3000, 10000, 300, 1000)["severity"] == "ok"
    assert rate_drift(3000, 10000, 2700, 10000)["severity"] == "warning"
    assert rate_drift(3000, 10000, 200, 1000)["severity"] == "alert"


def test_untreated_counts_ignores_treated_and_missing():
    discount = np.array([0, 0, 10, np.nan, 0])
    purchased = np.array([1, 0, 1, 1, np.nan])
    assert untreated_counts(discount, purchased) == (1, 2)
    assert untreated_counts(None, purchased) is None


def test_report_needs_a_minimum_batch():
    X = np.zeros((MIN_BATCH - 1, len(FEATURES)))
    with pytest.raises(ValueError):
        drift_report(FEATURES, np.zeros((500, len(FEATURES))), X)


def test_report_is_quiet_for_the_same_process_and_alerts_on_a_shift():
    ref, _ = generate(n=3000, seed=5)
    new, _ = generate(n=1500, seed=77)
    ref_X = ref[FEATURES].to_numpy(dtype=float)
    quiet = drift_report(FEATURES, ref_X, new[FEATURES].to_numpy(dtype=float))
    assert quiet["status"] == "ok" and quiet["messages"] == []
    loud = drift_report(FEATURES, ref_X, shift(new)[FEATURES].to_numpy(dtype=float))
    assert loud["status"] == "alert"
    flagged = {i["name"] for i in loud["items"] if i["severity"] == "alert"}
    assert {"loyalty_member", "days_since_last_purchase"} <= flagged
    assert "ALERT" in format_report(loud)


@pytest.fixture(scope="module")
def reference_frame():
    df, _ = generate(n=1500, seed=5)
    return df


@pytest.fixture(scope="module")
def ref_bundle(reference_frame):
    X = reference_frame[FEATURES].to_numpy(dtype=float)
    Y = reference_frame["purchased"].to_numpy(dtype=float)
    T = reference_frame["discount"].to_numpy(dtype=int)
    return make_bundle(fit_dml(X, Y, T, kind="linear"), X, kind="linear", Y=Y, T=T)


def records(df, with_outcome=True):
    out = []
    for row in df.itertuples():
        rec = {f: float(getattr(row, f)) for f in FEATURES}
        if with_outcome:
            rec["discount"] = int(row.discount)
            rec["purchased"] = int(row.purchased)
        out.append(rec)
    return out


def test_api_drift_quiet_and_alert(ref_bundle):
    new, _ = generate(n=600, seed=77)
    new = new.head(400)
    with TestClient(create_app(ref_bundle)) as client:
        assert client.get("/health").json()["drift_reference_available"] is True
        quiet = client.post("/drift", json={"customers": records(new)})
        assert quiet.status_code == 200
        assert quiet.json()["status"] == "ok"
        assert quiet.json()["behaviour"] is not None
        loud = client.post("/drift", json={"customers": records(shift(new))})
        assert loud.json()["status"] == "alert"
        assert loud.json()["messages"]


def test_api_drift_skips_behaviour_without_outcomes(ref_bundle):
    new, _ = generate(n=300, seed=78)
    with TestClient(create_app(ref_bundle)) as client:
        body = client.post("/drift", json={"customers": records(new, with_outcome=False)}).json()
    assert body["behaviour"] is None
    assert any("behaviour" in s for s in body["skipped"])


def test_api_drift_rejects_small_batches_and_missing_reference(ref_bundle, reference_frame):
    with TestClient(create_app(ref_bundle)) as client:
        small = client.post("/drift", json={"customers": records(reference_frame.head(MIN_BATCH - 1))})
        assert small.status_code == 422
    X = reference_frame[FEATURES].to_numpy(dtype=float)
    no_ref = make_bundle(ref_bundle["model"], X, kind="linear")  # no Y and T
    with TestClient(create_app(no_ref)) as client:
        r = client.post("/drift", json={"customers": records(reference_frame.head(MIN_BATCH))})
        assert r.status_code == 409
        assert "train_final_model" in r.json()["detail"]