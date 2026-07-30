import numpy as np
import pandas as pd
import pytest

pytest.importorskip("decaf", reason="requires the optional 'decaf' extra")

from causal_fairness_sdg.data import causal_graphs
from causal_fairness_sdg.fairness import AttributeRoles, get_fairness_mechanism
from causal_fairness_sdg.sdg.decaf import DECAFMethod

TOY_DATASET_NAME = "_toy_decaf_test"


@pytest.fixture(autouse=True)
def _register_toy_causal_graph():
    # DECAF needs a ground-truth DAG registered under `dataset_name`; give
    # the toy 4-column dataset used below its own entry for the duration of
    # this module's tests, matching the A/R/B/Y scenario used elsewhere
    # (test_mst_integration.py, test_fairness_biased_edges.py).
    causal_graphs.CAUSAL_GRAPHS[TOY_DATASET_NAME] = [
        ("A", "R"), ("R", "Y"), ("A", "B"), ("B", "Y"), ("A", "Y"),
    ]
    yield
    del causal_graphs.CAUSAL_GRAPHS[TOY_DATASET_NAME]


@pytest.fixture
def toy_dataset():
    rng = np.random.default_rng(0)
    n = 200
    A = rng.binomial(1, 0.5, n)
    R = rng.binomial(1, 0.5, n)
    B = rng.binomial(1, 0.5, n)
    Y = ((A + R + rng.binomial(1, 0.3, n)) >= 2).astype(int)
    df = pd.DataFrame({"A": A, "R": R, "B": B, "Y": Y}).astype(float)
    domain = {"A": 2, "R": 2, "B": 2, "Y": 2}
    roles = AttributeRoles.create(protected={"A"}, admissible={"R"}, outcome={"Y"})
    return df, domain, roles


@pytest.mark.slow
def test_decaf_requires_a_registered_causal_graph(toy_dataset):
    df, domain, roles = toy_dataset
    with pytest.raises(ValueError, match="No causal graph registered"):
        DECAFMethod(max_epochs=1).fit_generate(
            df, domain, roles, get_fairness_mechanism("none"),
            epsilon=1.0, delta=1e-9, n_synth=20, seed=0, dataset_name="not_registered",
        )


@pytest.mark.slow
@pytest.mark.parametrize("mechanism_name", ["none", "ftu", "dp", "cf"])
def test_decaf_runs_end_to_end_with_each_mechanism(toy_dataset, mechanism_name):
    df, domain, roles = toy_dataset
    result = DECAFMethod(max_epochs=2, h_dim=16, batch_size=32).fit_generate(
        df, domain, roles, get_fairness_mechanism(mechanism_name),
        epsilon=1.0, delta=1e-9, n_synth=50, seed=0, dataset_name=TOY_DATASET_NAME,
    )
    assert result.synthetic_data.shape == (50, len(domain))
    assert set(result.synthetic_data.columns) == set(domain.keys())
    for col, size in domain.items():
        assert result.synthetic_data[col].between(0, size - 1).all()
    assert set(result.graph_edges) == {("A", "R"), ("R", "Y"), ("A", "B"), ("B", "Y"), ("A", "Y")}


@pytest.mark.slow
def test_decaf_biased_edges_differ_across_mechanisms(toy_dataset):
    # Structural check that the fairness mechanism actually changes what's
    # passed to gen_synthetic, independent of GAN training stochasticity.
    df, domain, roles = toy_dataset
    dag = causal_graphs.as_digraph(
        causal_graphs.CAUSAL_GRAPHS[TOY_DATASET_NAME], list(df.columns)
    )
    none_biased = get_fairness_mechanism("none").select_biased_edges(dag, roles)
    ftu_biased = get_fairness_mechanism("ftu").select_biased_edges(dag, roles)
    dp_biased = get_fairness_mechanism("dp").select_biased_edges(dag, roles)
    cf_biased = get_fairness_mechanism("cf").select_biased_edges(dag, roles)

    assert none_biased == {}
    assert ftu_biased == {"Y": ["A"]}
    assert cf_biased == {"Y": ["A", "B"]}
    assert dp_biased == {"Y": ["A", "B", "R"]}
