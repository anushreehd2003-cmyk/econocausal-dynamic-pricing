import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import generate


def test_shapes_and_values():
    customers, truth = generate(n=500, seed=1)
    assert len(customers) == len(truth) == 500
    assert set(customers["discount"].unique()) <= {0, 10, 20}
    assert set(customers["purchased"].unique()) <= {0, 1}
    assert customers["customer_id"].is_unique


def test_reproducible():
    a, _ = generate(n=200, seed=7)
    b, _ = generate(n=200, seed=7)
    assert a.equals(b)