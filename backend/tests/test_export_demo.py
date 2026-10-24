import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from export_demo import build_demo_frame
from generate_data import generate


def test_demo_frame_has_no_ground_truth():
    customers, _ = generate(n=300, seed=2)
    preds = pd.DataFrame({
        "customer_id": customers["customer_id"], "is_test": 0,
        "ite_10": 0.01, "ite_10_lo": 0.0, "ite_10_hi": 0.02,
        "ite_20": 0.02, "ite_20_lo": 0.0, "ite_20_hi": 0.04,
    })
    frame = build_demo_frame(customers, preds)
    assert len(frame) == 300
    assert not any(c.startswith("true_") for c in frame.columns)
    assert frame.notna().all().all()