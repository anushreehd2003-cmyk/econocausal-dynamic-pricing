import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import generate
from matching import balance_after_matching, logit, match, matched_frame
from propensity import estimate_propensity


def _setup():
    customers, _ = generate(n=3000, seed=3)
    ps, t = estimate_propensity(customers)
    return customers, ps, t


def test_pairs_are_valid():
    _, ps, t = _setup()
    ti, ci, _ = match(ps, t)
    assert len(ti) == len(ci) > 0
    assert np.all(t[ti] == 1)
    assert np.all(t[ci] == 0)


def test_pairs_respect_caliper():
    _, ps, t = _setup()
    ti, ci, _ = match(ps, t, caliper_sd=0.2)
    lp = logit(ps)
    assert np.all(np.abs(lp[ti] - lp[ci]) <= 0.2 * lp.std() + 1e-9)


def test_matching_improves_balance():
    customers, ps, t = _setup()
    ti, ci, _ = match(ps, t)
    matched = matched_frame(customers, ti, ci)
    table = balance_after_matching(customers, t, matched)
    assert table["smd_after_matching"].abs().max() < table["smd_before"].abs().max()