import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causal_fairness_sdg.fairness import AttributeRoles, get_fairness_mechanism
from causal_fairness_sdg.sdg.mst import MST


@pytest.fixture
def toy_dataset():
    rng = np.random.default_rng(0)
    n = 2000
    A = rng.binomial(1, 0.5, n)
    R = rng.binomial(1, 0.5, n)
    B = rng.binomial(1, 0.5, n)
    Y = ((A + R + rng.binomial(1, 0.3, n)) >= 2).astype(int)
    df = pd.DataFrame({"A": A, "R": R, "B": B, "Y": Y})
    domain = {"A": 2, "R": 2, "B": 2, "Y": 2}
    roles = AttributeRoles.create(protected={"A"}, admissible={"R"}, outcome={"Y"})
    return df, domain, roles


def _edge_set(edges):
    return {frozenset(e) for e in edges}


def test_mst_baseline_produces_a_spanning_tree(toy_dataset):
    df, domain, roles = toy_dataset
    result = MST().fit_generate(
        df, domain, roles, get_fairness_mechanism("none"),
        epsilon=3.0, delta=1e-9, n_synth=200, seed=1,
    )
    g = nx.Graph()
    g.add_nodes_from(domain.keys())
    g.add_edges_from(result.graph_edges)
    assert nx.is_connected(g)
    assert len(result.graph_edges) == len(domain) - 1
    assert result.synthetic_data.shape == (200, len(domain))


def test_ftu_never_selects_the_direct_protected_outcome_edge(toy_dataset):
    df, domain, roles = toy_dataset
    result = MST().fit_generate(
        df, domain, roles, get_fairness_mechanism("ftu"),
        epsilon=3.0, delta=1e-9, n_synth=200, seed=1,
    )
    assert frozenset({"A", "Y"}) not in _edge_set(result.graph_edges)


def test_cf_never_leaves_an_unblocked_protected_outcome_path(toy_dataset):
    df, domain, roles = toy_dataset
    result = MST().fit_generate(
        df, domain, roles, get_fairness_mechanism("cf"),
        epsilon=3.0, delta=1e-9, n_synth=200, seed=1,
    )
    g = nx.Graph()
    g.add_nodes_from(domain.keys())
    g.add_edges_from(result.graph_edges)
    for p in roles.protected:
        for o in roles.outcome:
            if nx.has_path(g, p, o):
                path = nx.shortest_path(g, p, o)
                assert set(path[1:-1]) & roles.admissible, (
                    f"unblocked path {path} found under CF"
                )


def test_dp_produces_a_forest_with_protected_and_outcome_disconnected(toy_dataset):
    df, domain, roles = toy_dataset
    result = MST().fit_generate(
        df, domain, roles, get_fairness_mechanism("dp"),
        epsilon=3.0, delta=1e-9, n_synth=200, seed=1,
    )
    g = nx.Graph()
    g.add_nodes_from(domain.keys())
    g.add_edges_from(result.graph_edges)
    # DP must produce a forest, not necessarily a single spanning tree
    assert nx.is_forest(g)
    for p in roles.protected:
        for o in roles.outcome:
            assert not nx.has_path(g, p, o), (
                f"{p} and {o} are connected -- DP should keep them in separate components"
            )


@pytest.mark.parametrize("mechanism_name", ["none", "ftu", "dp", "cf"])
def test_all_mechanisms_run_end_to_end_without_error(toy_dataset, mechanism_name):
    df, domain, roles = toy_dataset
    result = MST().fit_generate(
        df, domain, roles, get_fairness_mechanism(mechanism_name),
        epsilon=2.0, delta=1e-9, n_synth=100, seed=2,
    )
    assert result.synthetic_data.shape == (100, len(domain))
    assert set(result.synthetic_data.columns) == set(domain.keys())
