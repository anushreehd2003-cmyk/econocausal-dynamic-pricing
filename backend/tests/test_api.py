import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from api import create_app
from causal_graph import CONFOUNDERS
from dml_engine import fit_dml
from generate_data import generate
from model_store import (
    FEATURES,
    load_bundle,
    make_bundle,
    predict_effects,
    save_bundle,
)
from optimizer import greedy_allocation, values_from_ite


@pytest.fixture(scope="module")
def customers():
    df, _ = generate(n=1500, seed=5)
    return df


@pytest.fixture(scope="module")
def bundle(customers):
    X = customers[FEATURES].to_numpy(dtype=float)
    Y = customers["purchased"].to_numpy(dtype=float)
    T = customers["discount"].to_numpy(dtype=int)
    return make_bundle(fit_dml(X, Y, T, kind="linear"), X, kind="linear")


@pytest.fixture()
def client(bundle):
    with TestClient(create_app(bundle)) as c:
        yield c


def records(df, n, basket=False):
    out = []
    for row in df.head(n).itertuples():
        rec = {"customer_id": int(row.customer_id)}
        for f in FEATURES:
            rec[f] = float(getattr(row, f))
        if basket:
            rec["avg_basket_value"] = float(row.avg_basket_value)
        out.append(rec)
    return out


def test_features_match_causal_graph():
    assert FEATURES == CONFOUNDERS


def test_save_and_load_roundtrip(tmp_path, bundle, customers):
    path = save_bundle(bundle, tmp_path / "model.joblib")
    loaded = load_bundle(path)
    X = customers[FEATURES].head(50).to_numpy(dtype=float)
    a = predict_effects(bundle, X)["ite_20"]
    b = predict_effects(loaded, X)["ite_20"]
    assert np.allclose(a, b)
    assert loaded["feature_ranges"] == bundle["feature_ranges"]


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["features"] == FEATURES


def test_health_and_predict_without_model(monkeypatch, tmp_path, customers):
    monkeypatch.setenv("ECONOCAUSAL_MODEL", str(tmp_path / "missing.joblib"))
    with TestClient(create_app()) as c:
        assert c.get("/health").status_code == 503
        r = c.post("/predict", json={"customers": records(customers, 1)})
        assert r.status_code == 503
        assert "train_final_model" in r.json()["detail"]


def test_predict_shape_and_intervals(client, customers):
    r = client.post("/predict", json={"customers": records(customers, 5)})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 5
    assert [p["index"] for p in preds] == list(range(5))
    for p in preds:
        for level in (10, 20):
            assert np.isfinite(p[f"ite_{level}"])
            assert p[f"ite_{level}_lo"] <= p[f"ite_{level}_hi"]


def test_predict_rejects_bad_input(client, customers):
    good = records(customers, 1)[0]
    assert client.post("/predict", json={"customers": []}).status_code == 422
    assert client.post("/predict", json={"customers": [dict(good, loyalty_member=2)]}).status_code == 422
    assert client.post("/predict", json={"customers": [dict(good, age=200)]}).status_code == 422
    missing = {k: v for k, v in good.items() if k != "income"}
    assert client.post("/predict", json={"customers": [missing]}).status_code == 422


def test_out_of_range_customer_gets_a_warning(client, customers):
    odd = dict(records(customers, 1)[0], days_since_last_purchase=900.0)
    body = client.post("/predict", json={"customers": [odd]}).json()
    assert any("days_since_last_purchase" in w for w in body["warnings"])


def test_allocate_respects_budget_and_counts(client, customers):
    req = {"customers": records(customers, 300, basket=True), "budget": 300}
    body = client.post("/allocate", json=req).json()
    assert len(body["allocation"]) == 300
    assert body["spend"] <= 300
    discounts = [row["discount"] for row in body["allocation"]]
    assert set(discounts) <= {0, 10, 20}
    assert body["spend"] == sum(discounts)
    assert body["customers_targeted"] == sum(d > 0 for d in discounts)
    assert body["n_10"] + body["n_20"] == body["customers_targeted"]


def test_allocate_only_uses_allowed_levels(client, customers):
    req = {"customers": records(customers, 300, basket=True), "budget": 1000, "levels": [20]}
    discounts = {row["discount"] for row in client.post("/allocate", json=req).json()["allocation"]}
    assert discounts <= {0, 20}


def test_allocate_matches_the_python_optimizer(client, bundle, customers):
    recs = records(customers, 300, basket=True)
    body = client.post("/allocate", json={"customers": recs, "budget": 400}).json()
    X = customers[FEATURES].head(300).to_numpy(dtype=float)
    effects = predict_effects(bundle, X)
    ite = np.column_stack([effects["ite_10"], effects["ite_20"]])
    basket = customers["avg_basket_value"].head(300).to_numpy()
    expected = greedy_allocation(values_from_ite(ite, basket), 400, (10, 20))
    assert [row["discount"] for row in body["allocation"]] == expected.tolist()


def test_exact_is_at_least_as_good_as_greedy(client, customers):
    recs = records(customers, 300, basket=True)
    greedy = client.post("/allocate", json={"customers": recs, "budget": 400, "method": "greedy"}).json()
    exact = client.post("/allocate", json={"customers": recs, "budget": 400, "method": "exact"}).json()
    assert exact["spend"] <= 400
    assert exact["predicted_extra_revenue"] >= greedy["predicted_extra_revenue"] - 0.01


def test_allocate_rejects_bad_requests(client, customers):
    recs = records(customers, 20, basket=True)
    assert client.post("/allocate", json={"customers": recs, "budget": 0}).status_code == 422
    assert client.post("/allocate", json={"customers": recs, "budget": 100, "levels": [15]}).status_code == 422
    assert client.post("/allocate", json={"customers": recs, "budget": 100, "levels": [10, 10]}).status_code == 422
    assert client.post("/allocate", json={"customers": recs, "budget": 100, "method": "magic"}).status_code == 422
    no_basket = records(customers, 20)
    assert client.post("/allocate", json={"customers": no_basket, "budget": 100}).status_code == 422