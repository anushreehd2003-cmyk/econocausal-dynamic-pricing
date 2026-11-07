"""Run the whole EconoCausal pipeline in the right order.

    python backend\\src\\run_pipeline.py                  core steps
    python backend\\src\\run_pipeline.py --all            also the optional analyses
    python backend\\src\\run_pipeline.py --check          only check packages and outputs
    python backend\\src\\run_pipeline.py --start-at train_final_model.py

Every script uses fixed random seeds, so a rerun reproduces the same results.
"""
import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[1]

# Needed to run the dashboard and the API. Order matters: later steps read
# files written by earlier ones.
CORE_STEPS = [
    ("generate_data.py", "Generate the synthetic retail data"),
    ("dml_engine.py", "Train Double ML and evaluate it on held-out customers"),
    ("model_comparison.py", "Compare models and export the Qini curves"),
    ("export_demo.py", "Export the dashboard demo data"),
    ("efficiency.py", "Compute the budget-efficiency curve"),
    ("train_final_model.py", "Train and save the model served by the API"),
]

# The same steps plus the analyses that produce the documents in docs/.
ALL_STEPS = [
    ("generate_data.py", "Generate the synthetic retail data"),
    ("causal_graph.py", "Build the causal graph and a first adjusted estimate"),
    ("propensity.py", "Propensity scores and covariate balance"),
    ("matching.py", "Propensity score matching"),
    ("dml_engine.py", "Train Double ML and evaluate it on held-out customers"),
    ("refutation.py", "Refutation tests (causal audit)"),
    ("model_comparison.py", "Compare models and export the Qini curves"),
    ("export_demo.py", "Export the dashboard demo data"),
    ("optimizer.py", "Budget optimizer on held-out customers"),
    ("efficiency.py", "Compute the budget-efficiency curve"),
    ("train_final_model.py", "Train and save the model served by the API"),
    ("drift_demo.py", "Drift detector demo"),
]

# pip name -> import name
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "lightgbm": "lightgbm",
    "econml": "econml",
    "dowhy": "dowhy",
    "networkx": "networkx",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "httpx": "httpx",
    "pytest": "pytest",
}

# Files the dashboard and the API need after the core steps have run.
EXPECTED_OUTPUTS = [
    "data/retail_campaign.csv",
    "data/ground_truth_effects.csv",
    "data/ite_predictions.csv",
    "frontend/public/demo/customers_ite.csv",
    "frontend/public/demo/qini_curves.json",
    "frontend/public/demo/budget_efficiency.json",
    "models/dml_forest.joblib",
]


def build_steps(include_optional=False):
    return ALL_STEPS if include_optional else CORE_STEPS


def missing_packages(required=None):
    """pip names of required packages that cannot be imported."""
    required = REQUIRED_PACKAGES if required is None else required
    return [pip for pip, module in required.items() if importlib.util.find_spec(module) is None]


def missing_files(root, expected=None):
    expected = EXPECTED_OUTPUTS if expected is None else expected
    return [name for name in expected if not (Path(root) / name).exists()]


def run_steps(steps, src_dir=SRC, python=None):
    """Run each script in turn and stop at the first failure."""
    python = python or sys.executable
    results = []
    for number, (script, description) in enumerate(steps, 1):
        print(f"\n[{number}/{len(steps)}] {description} ({script})", flush=True)
        start = time.time()
        code = subprocess.run([python, str(Path(src_dir) / script)]).returncode
        elapsed = time.time() - start
        results.append((script, code, elapsed))
        if code != 0:
            print(
                f"\nStep failed: {script} (exit code {code}). Fix the error above, "
                f"then resume with:\n    python backend\\src\\run_pipeline.py --start-at {script}",
                flush=True,
            )
            return False, results
        print(f"Done in {elapsed:.0f} s.", flush=True)
    return True, results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the EconoCausal pipeline.")
    parser.add_argument("--all", action="store_true", help="include the optional analyses")
    parser.add_argument("--check", action="store_true", help="only check packages and outputs")
    parser.add_argument("--start-at", metavar="SCRIPT", help="resume from this step")
    args = parser.parse_args(argv)

    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("Warning: Python 3.10 or newer is required.")
    missing = missing_packages()
    if missing:
        print("Missing packages: " + ", ".join(missing))
        print("Install them with: pip install -r backend/requirements.txt")
        return 1
    print("All required packages are installed.")

    if args.check:
        absent = missing_files(ROOT)
        for name in EXPECTED_OUTPUTS:
            print(f"  {'missing' if name in absent else 'found':<8}{name}")
        if absent:
            print("Run: python backend\\src\\run_pipeline.py")
        return 1 if absent else 0

    steps = build_steps(args.all)
    if args.start_at:
        names = [script for script, _ in steps]
        if args.start_at not in names:
            print(f"Unknown step {args.start_at}. Steps: " + ", ".join(names))
            return 2
        steps = steps[names.index(args.start_at):]

    ok, results = run_steps(steps)
    if not ok:
        return 1

    print("\nSummary")
    for script, _, elapsed in results:
        print(f"  {script:<26}{elapsed:>6.0f} s")
    absent = missing_files(ROOT)
    if absent:
        print("\nThese expected files are still missing: " + ", ".join(absent))
        return 1
    print("\nPipeline complete. Next:")
    print("  python -m uvicorn api:app --app-dir backend\\src --port 8000   (API)")
    print("  cd frontend; npm install; npm run dev                          (dashboard)")
    return 0


if __name__ == "__main__":
    sys.exit(main())