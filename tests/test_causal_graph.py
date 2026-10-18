import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_graph import CONFOUNDERS, OUTCOME, TREATMENT, get_edges


def test_graph_is_acyclic():
    assert nx.is_directed_acyclic_graph(nx.DiGraph(get_edges()))


def test_confounders_point_to_treatment_and_outcome():
    edges = set(get_edges())
    for c in CONFOUNDERS:
        assert (c, TREATMENT) in edges
        assert (c, OUTCOME) in edges
    assert (TREATMENT, OUTCOME) in edges