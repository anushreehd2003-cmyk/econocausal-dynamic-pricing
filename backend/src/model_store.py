"""Saving, loading and using the trained Double ML model (used by the API)."""
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "dml_forest.joblib"
FEATURES = [
    "age",
    "income",
    "loyalty_member",
    "past_purchases",
    "days_since_last_purchase",
]
LEVELS = (10, 20)


def model_path():
    """Where the API looks for the model (override with ECONOCAUSAL_MODEL)."""
    return Path(os.environ.get("ECONOCAUSAL_MODEL", DEFAULT_MODEL_PATH))


def _versions():
    import econml
    import sklearn

    return {
        "econml": econml.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def make_bundle(model, X, kind, Y=None, T=None):
    """Model plus the metadata the API needs to use it safely.

    Pass Y (purchased) and T (discount) to also store the reference data used
    by the drift detector."""
    X = np.asarray(X, dtype=float)
    bundle = {
        "model": model,
        "kind": kind,
        "features": list(FEATURES),
        "levels": list(LEVELS),
        "n_training_rows": int(len(X)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_ranges": {
            f: [float(X[:, i].min()), float(X[:, i].max())]
            for i, f in enumerate(FEATURES)
        },
        "library_versions": _versions(),
    }
    if Y is not None and T is not None:
        untreated = np.asarray(T) == 0
        bundle["reference"] = {
            "X": X,
            "untreated_n": int(untreated.sum()),
            "untreated_buys": int(np.asarray(Y, dtype=float)[untreated].sum()),
        }
    return bundle


def save_bundle(bundle, path=None):
    path = Path(path) if path else model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path=None):
    """Only load model files you created yourself: they are pickles."""
    return joblib.load(Path(path) if path else model_path())


def version_warnings(bundle):
    """Library versions differ from the ones the model was trained with."""
    now = _versions()
    return [
        f"{name}: trained with {old}, running {now[name]}"
        for name, old in bundle["library_versions"].items()
        if name in now and now[name] != old
    ]


def predict_effects(bundle, X, intervals=False, alpha=0.1):
    """Effect of each discount level vs no discount, per customer."""
    model = bundle["model"]
    X = np.asarray(X, dtype=float)
    out = {}
    for level in bundle["levels"]:
        out[f"ite_{level}"] = np.asarray(model.effect(X, T0=0, T1=level)).ravel()
        if intervals:
            lo, hi = model.effect_interval(X, T0=0, T1=level, alpha=alpha)
            out[f"ite_{level}_lo"] = np.asarray(lo).ravel()
            out[f"ite_{level}_hi"] = np.asarray(hi).ravel()
    return out


def range_warnings(bundle, X):
    """Flag customers outside the feature ranges the model was trained on."""
    X = np.asarray(X, dtype=float)
    messages = []
    for i, name in enumerate(bundle["features"]):
        lo, hi = bundle["feature_ranges"][name]
        outside = int(((X[:, i] < lo) | (X[:, i] > hi)).sum())
        if outside:
            messages.append(
                f"{name}: {outside} of {len(X)} customers outside the training "
                f"range {lo:g} to {hi:g}; predictions may be unreliable"
            )
    return messages