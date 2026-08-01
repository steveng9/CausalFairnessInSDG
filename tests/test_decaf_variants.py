import numpy as np
import pandas as pd
import pytest

pytest.importorskip("decaf", reason="requires the optional 'decaf' extra")
pytest.importorskip("opacus", reason="requires opacus for DP accounting")

from causal_fairness_sdg.data import causal_graphs
from causal_fairness_sdg.fairness import AttributeRoles, get_fairness_mechanism
from causal_fairness_sdg.sdg._decaf_ctgan import ColumnBlocks, ConditionSampler
from causal_fairness_sdg.sdg._decaf_dp import find_noise_multiplier, spent_epsilon
from causal_fairness_sdg.sdg.decaf_variants import (
    DECAFCTGANMethod,
    DECAFDPCTGANMethod,
    DECAFDPGANMethod,
)

TOY_DATASET_NAME = "_toy_decaf_variants_test"


@pytest.fixture(autouse=True)
def _register_toy_causal_graph():
    # Same A/R/B/Y scenario the other integration tests use: A protected,
    # R admissible, B inadmissible mediator, Y outcome.
    causal_graphs.CAUSAL_GRAPHS[TOY_DATASET_NAME] = [
        ("A", "R"), ("R", "Y"), ("A", "B"), ("B", "Y"), ("A", "Y"),
    ]
    yield
    del causal_graphs.CAUSAL_GRAPHS[TOY_DATASET_NAME]


@pytest.fixture
def toy_dataset():
    rng = np.random.default_rng(0)
    n = 300
    A = rng.binomial(1, 0.5, n)
    R = rng.binomial(1, 0.5, n)
    B = rng.binomial(1, 0.5, n)
    Y = ((A + R + rng.binomial(1, 0.3, n)) >= 2).astype(int)
    df = pd.DataFrame({"A": A, "R": R, "B": B, "Y": Y})
    domain = {"A": 2, "R": 2, "B": 2, "Y": 2}
    roles = AttributeRoles.create(protected={"A"}, admissible={"R"}, outcome={"Y"})
    return df, domain, roles


# --------------------------------------------------------------------------
# Column blocks / conditional sampling
# --------------------------------------------------------------------------


def test_column_blocks_round_trip():
    blocks = ColumnBlocks(["a", "b"], {"a": 3, "b": 2})
    assert blocks.total == 5
    assert blocks.span(0) == (0, 3)
    assert blocks.span(1) == (3, 5)
    codes = np.array([[0, 1], [2, 0], [1, 1]])
    assert np.array_equal(blocks.decode(blocks.encode(codes)), codes)


def test_column_blocks_clip_folds_out_of_domain_codes():
    """`load_adult` leaves `education-num` at its raw 1-16 range while
    declaring a domain of 16, so a code can equal `domain`. Encoding must fold
    it into the last bin the way mbi already does, not raise or silently write
    into the next column's block."""
    blocks = ColumnBlocks(["a", "b"], {"a": 3, "b": 2})
    encoded = blocks.encode(np.array([[3, 0]]))  # 3 is out of range for domain 3
    assert encoded.sum() == 2.0  # exactly one hot per column, nothing spilled
    assert encoded[0, 2] == 1.0  # folded into the last bin of column `a`
    assert encoded[0, 3] == 1.0  # column `b` untouched


def test_condition_sampler_log_frequency_lifts_rare_categories():
    codes = np.array([[0]] * 99 + [[1]])  # 99:1 imbalance
    blocks = ColumnBlocks(["a"], {"a": 2})
    log_s = ConditionSampler(codes, blocks, log_frequency=True)
    raw_s = ConditionSampler(codes, blocks, log_frequency=False)
    # The whole point of log-frequency weighting: the rare category gets a far
    # bigger share of training conditions than its raw frequency.
    assert log_s.category_probs[0][1] > 10 * raw_s.category_probs[0][1]
    # ... while the generation-time marginal stays faithful to the data.
    assert log_s.marginal_probs[0][1] == pytest.approx(0.01)


def test_condition_sampler_matching_rows_respect_the_condition():
    codes = np.array([[0], [1], [0], [1]])
    blocks = ColumnBlocks(["a"], {"a": 2})
    sampler = ConditionSampler(codes, blocks)
    rng = np.random.default_rng(0)
    cols = np.zeros(20, dtype=int)
    cats = np.array([1] * 20)
    idx = sampler.sample_matching_rows(cols, cats, rng)
    assert (codes[idx, 0] == 1).all()


# --------------------------------------------------------------------------
# Privacy accounting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("target_eps", [1.0, 10.0])
def test_noise_multiplier_spends_the_requested_budget(target_eps):
    sigma = find_noise_multiplier(
        target_epsilon=target_eps, delta=1e-9, sample_rate=0.05, steps=500
    )
    spent = spent_epsilon(sigma, sample_rate=0.05, steps=500, delta=1e-9)
    assert spent <= target_eps + 1e-6
    # And it doesn't waste budget by over-noising.
    assert spent > 0.9 * target_eps


def test_tighter_epsilon_requires_more_noise():
    kwargs = dict(delta=1e-9, sample_rate=0.05, steps=500)
    assert (
        find_noise_multiplier(target_epsilon=1.0, **kwargs)
        > find_noise_multiplier(target_epsilon=10.0, **kwargs)
    )


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------


def _run(method, toy_dataset, mechanism="none", epsilon=10.0):
    df, domain, roles = toy_dataset
    return method.fit_generate(
        df, domain, roles, get_fairness_mechanism(mechanism),
        epsilon=epsilon, delta=1e-9, n_synth=len(df), seed=0,
        dataset_name=TOY_DATASET_NAME,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "factory",
    [
        lambda: DECAFDPGANMethod(max_epochs=2, batch_size=64),
        lambda: DECAFCTGANMethod(max_epochs=2, batch_size=64, pac=2),
        lambda: DECAFDPCTGANMethod(
            max_epochs=2, batch_size=64, pac=1, use_conditional=False
        ),
    ],
)
def test_variants_produce_in_domain_synthetic_data(factory, toy_dataset):
    df, domain, _ = toy_dataset
    result = _run(factory(), toy_dataset)
    synth = result.synthetic_data
    assert list(synth.columns) == list(df.columns)
    assert len(synth) == len(df)
    for col, size in domain.items():
        assert synth[col].between(0, size - 1).all()
    assert result.graph_edges


@pytest.mark.slow
def test_dp_variants_report_their_privacy_calibration(toy_dataset):
    result = _run(DECAFDPGANMethod(max_epochs=2, batch_size=64), toy_dataset)
    assert result.extra["noise_multiplier"] > 0
    assert result.extra["dp_steps"] > 0
    # The realized spend must not exceed what was asked for.
    assert result.extra["spent_epsilon"] <= 10.0 + 1e-6


@pytest.mark.slow
def test_tighter_epsilon_gets_more_noise_end_to_end(toy_dataset):
    loose = _run(DECAFDPGANMethod(max_epochs=2, batch_size=64), toy_dataset, epsilon=100.0)
    tight = _run(DECAFDPGANMethod(max_epochs=2, batch_size=64), toy_dataset, epsilon=1.0)
    assert tight.extra["noise_multiplier"] > loose.extra["noise_multiplier"]


@pytest.mark.slow
def test_fairness_mechanisms_reach_the_generator(toy_dataset):
    """Each mechanism should hand the generator a different set of edges to
    shuffle -- otherwise FTU/DP/CF would be silently identical, which is
    exactly the failure mode that made MST's FTU a no-op on Adult."""
    df, domain, roles = toy_dataset
    import networkx as nx

    dag = nx.DiGraph(causal_graphs.CAUSAL_GRAPHS[TOY_DATASET_NAME])
    selections = {
        name: get_fairness_mechanism(name).select_biased_edges(dag, roles)
        for name in ("none", "ftu", "dp", "cf")
    }
    assert selections["none"] == {}
    assert selections["ftu"], "FTU should cut the direct A->Y edge"
    assert selections["dp"] != selections["ftu"]
    assert selections["cf"] != selections["dp"]


@pytest.mark.slow
def test_dp_ctgan_rejects_conditional_sampling(toy_dataset):
    """Conditional training reads unprivatized private counts, so the DP path
    must refuse it rather than silently emit an unsound guarantee."""
    method = DECAFDPCTGANMethod(max_epochs=2, batch_size=64, pac=1)
    method.settings = {TOY_DATASET_NAME: {"use_conditional": True, "pac": 1}}
    with pytest.raises(ValueError, match="use_conditional=False"):
        _run(method, toy_dataset)


@pytest.mark.slow
def test_train_cache_is_reused_across_fairness_mechanisms(toy_dataset):
    """Fairness acts only at generation time, so all four mechanisms must
    share one trained model -- this is the saving that makes the GAN backbones
    affordable in the grid."""
    method = DECAFDPGANMethod(max_epochs=2, batch_size=64)
    _run(method, toy_dataset, mechanism="none")
    assert len(method._cache) == 1
    trained = next(iter(method._cache.values()))
    for mechanism in ("ftu", "dp", "cf"):
        _run(method, toy_dataset, mechanism=mechanism)
        assert next(iter(method._cache.values())) is trained
