import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_graph import CONFOUNDERS
from generate_data import generate
from refutation import build_gml, dml_ate


def test_gml_has_expected_nodes_and_edges():
    gml = build_gml("treated")
    assert gml.count("edge [") == 2 * len(CONFOUNDERS) + 1
    assert 'node [id "treated"' in gml


def test_dml_ate_is_finite():
    customers, _ = generate(n=1500, seed=5)
    X = customers[CONFOUNDERS].to_numpy(dtype=float)
    Y = customers["purchased"].to_numpy(dtype=float)
    T = customers["discount"].to_numpy(dtype=int)
    assert np.isfinite(dml_ate(X, Y, T, seed=1))