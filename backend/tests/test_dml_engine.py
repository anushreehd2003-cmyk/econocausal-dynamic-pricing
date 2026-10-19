import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_graph import CONFOUNDERS
from dml_engine import LEVELS, evaluate, fit_dml, predict_effects
from generate_data import generate


def test_evaluate_perfect_prediction():
    x = np.linspace(-0.1, 0.2, 50)
    m = evaluate(x, x)
    assert m["rmse"] < 1e-12
    assert abs(m["corr"] - 1) < 1e-9


def test_dml_predictions_shape_and_finite():
    customers, _ = generate(n=1500, seed=5)
    X = customers[CONFOUNDERS].to_numpy(dtype=float)
    Y = customers["purchased"].to_numpy(dtype=float)
    T = customers["discount"].to_numpy(dtype=int)
    model = fit_dml(X, Y, T, kind="linear")
    pred = predict_effects(model, X[:200])
    assert len(pred) == 200
    assert np.isfinite(pred.to_numpy()).all()
    for level in LEVELS:
        assert (pred[f"ite_{level}_lo"] <= pred[f"ite_{level}_hi"]).all()