import networkx as nx
import pytest

from causal_fairness_sdg.data.causal_graphs import ADULT_DAG, COMPAS_DAG, as_digraph
from causal_fairness_sdg.data.datasets import ADULT_COLUMNS

ADULT_COLUMNS_NO_FNLWGT = [c for c in ADULT_COLUMNS if c != "fnlwgt"]
COMPAS_COLUMNS = [
    "sex", "race", "age_cat", "juv_fel_count", "juv_misd_count",
    "priors_count", "c_charge_degree", "two_year_recid",
]


@pytest.mark.parametrize(
    "edges,columns",
    [(ADULT_DAG, ADULT_COLUMNS_NO_FNLWGT), (COMPAS_DAG, COMPAS_COLUMNS)],
)
def test_causal_graph_is_valid_dag_over_dataset_columns(edges, columns):
    dag = as_digraph(edges, columns)
    assert set(dag.nodes) == set(columns)
    assert nx.is_directed_acyclic_graph(dag)


def test_as_digraph_rejects_unknown_columns():
    with pytest.raises(ValueError, match="unknown columns"):
        as_digraph([("not_a_real_attr", "income")], ["income"])


def test_as_digraph_rejects_cycles():
    with pytest.raises(ValueError, match="acyclic"):
        as_digraph([("a", "b"), ("b", "a")], ["a", "b"])


@pytest.mark.parametrize(
    "edges,columns,outcome",
    [(ADULT_DAG, ADULT_COLUMNS_NO_FNLWGT, "income"), (COMPAS_DAG, COMPAS_COLUMNS, "two_year_recid")],
)
def test_protected_attributes_reach_outcome(edges, columns, outcome):
    # Sanity check that these DAGs are actually interesting test cases for
    # the fairness mechanisms: every protected attribute should have at
    # least one directed path to the outcome.
    dag = as_digraph(edges, columns)
    protected = {"sex", "race"} & set(columns)
    for p in protected:
        assert nx.has_path(dag, p, outcome)


def test_discovered_graph_is_looked_up_by_training_table(tmp_path, monkeypatch):
    import json

    import pandas as pd
    import pytest

    from causal_fairness_sdg.data import causal_graphs

    train = pd.DataFrame({"a": [0, 1, 1], "b": [1, 0, 1], "y": [0, 0, 1]})
    record = {"graphs": {causal_graphs.table_key(train): {"edges": [["a", "y"]]}}}
    (tmp_path / "fges_toy.json").write_text(json.dumps(record))
    monkeypatch.setattr(causal_graphs, "DISCOVERED_GRAPH_DIR", tmp_path)
    causal_graphs._discovered.cache_clear()

    assert causal_graphs.resolve_graph("toy", train, "fges") == [("a", "y")]
    # A graph belongs to the rows it was discovered on, and to nothing else.
    with pytest.raises(ValueError, match="No 'fges' graph for this training table"):
        causal_graphs.resolve_graph("toy", train.iloc[::-1], "fges")
    with pytest.raises(ValueError, match="unknown graph source"):
        causal_graphs.resolve_graph("toy", train, "pc")
    causal_graphs._discovered.cache_clear()


def test_every_dataset_has_a_discovered_graph_for_its_sweep_splits():
    from sklearn.model_selection import train_test_split

    from causal_fairness_sdg.data import causal_graphs
    from causal_fairness_sdg.data.datasets import DATASETS

    for name in ("compas", "adult"):
        data, _domain, roles = DATASETS[name]()
        outcome = next(iter(roles.outcome))
        for seed in range(5):
            train, _ = train_test_split(data, test_size=0.3, random_state=seed)
            dag = as_digraph(
                causal_graphs.resolve_graph(name, train, "fges"), list(data.columns)
            )
            assert dag.out_degree(outcome) == 0
            assert all(dag.in_degree(r) == 0 for r in causal_graphs.DISCOVERY_ROOTS[name])


def test_consistent_extension_orients_without_new_colliders():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from discover_graphs import consistent_extension, v_structures

    nodes = ["a", "b", "c", "d"]
    directed = [("a", "c"), ("b", "c")]  # the one collider
    undirected = [("c", "d")]
    edges = consistent_extension(nodes, directed, undirected)
    assert ("c", "d") in edges  # d -> c would add colliders at c
    assert v_structures(nodes, edges) == v_structures(nodes, directed)

    chain = consistent_extension(["x", "y", "z"], [], [("x", "y"), ("y", "z")])
    assert v_structures(["x", "y", "z"], chain) == set()
