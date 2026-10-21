import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_comparison import qini_curve, qini_summary

U = np.array([0.30, 0.10, -0.05, -0.20, 0.00, 0.20])


def test_curve_starts_at_zero_and_ends_at_total():
    x, gains = qini_curve(U, U)
    assert gains[0] == 0
    assert x[0] == 0 and x[-1] == 100
    assert np.isclose(gains[-1], U.sum() / len(U) * 1000)


def test_perfect_ranking_scores_one():
    assert np.isclose(qini_summary(U, U)[-1], 1.0)


def test_reversed_ranking_scores_negative():
    assert qini_summary(-U, U)[-1] < 0