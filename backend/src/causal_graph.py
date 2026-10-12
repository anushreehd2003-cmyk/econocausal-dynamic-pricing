"""Causal DAG for EconoCausal using DoWhy."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from dowhy import CausalModel

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "retail_campaign.csv"
DAG_IMAGE = ROOT / "docs" / "causal_dag.png"

TREATMENT = "discount"
OUTCOME = "purchased"
CONFOUNDERS = [
    "age",
    "income",
    "loyalty_member",
    "past_purchases",
    "days_since_last_purchase",
]


def get_edges():
    edges = []
    for c in CONFOUNDERS:
        edges.append((c, TREATMENT))
        edges.append((c, OUTCOME))
    edges.append((TREATMENT, OUTCOME))
    return edges


def build_gml():
    nodes = CONFOUNDERS + [TREATMENT, OUTCOME]
    parts = ["graph [directed 1"]
    parts += [f'node [id "{n}" label "{n}"]' for n in nodes]
    parts += [f'edge [source "{s}" target "{t}"]' for s, t in get_edges()]
    parts.append("]")
    return "\n".join(parts)


def draw_dag(path=DAG_IMAGE):
    g = nx.DiGraph(get_edges())
    pos = {c: (i * 2, 2) for i, c in enumerate(CONFOUNDERS)}
    pos[TREATMENT] = (2, 0)
    pos[OUTCOME] = (6, 0)
    colors = [
        "tab:green" if n == TREATMENT else "tab:red" if n == OUTCOME else "lightgray"
        for n in g.nodes
    ]
    plt.figure(figsize=(12, 6))
    nx.draw(g, pos, with_labels=True, node_color=colors, node_size=3500,
            font_size=8, arrows=True, arrowsize=18)
    plt.title("EconoCausal DAG: confounders -> discount -> purchased")
    path.parent.mkdir(exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    if not DATA_FILE.exists():
        raise SystemExit("Run python backend\\src\\generate_data.py first.")
    df = pd.read_csv(DATA_FILE)

    model = CausalModel(data=df, treatment=TREATMENT, outcome=OUTCOME,
                        graph=build_gml())
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(estimand)

    draw_dag()
    print(f"DAG image saved to {DAG_IMAGE}")

    # Rough sanity check only; Week 2 replaces this with Double ML.
    rates = df.groupby(TREATMENT)[OUTCOME].mean()
    naive = rates[20] - rates[0]
    est = model.estimate_effect(
        estimand, method_name="backdoor.linear_regression",
        control_value=0, treatment_value=20,
    )
    print(f"Naive effect of $20 discount   : {naive:+.3f}")
    print(f"Adjusted (linear) effect of $20: {est.value:+.3f}")


if __name__ == "__main__":
    main()