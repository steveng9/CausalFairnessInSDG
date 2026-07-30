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
