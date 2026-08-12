import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("decaf", reason="requires the optional 'decaf' extra")
pytest.importorskip("opacus", reason="requires opacus for DP accounting")

from causal_fairness_sdg.data import causal_graphs
from causal_fairness_sdg.fairness import AttributeRoles, get_fairness_mechanism
from causal_fairness_sdg.sdg._decaf_ctgan import ColumnBlocks
from causal_fairness_sdg.sdg._decaf_diffusion import (
    DDPMCausalTrainer,
    DiffusionSchedule,
    SinusoidalTimeEmbedding,
)
from causal_fairness_sdg.sdg.decaf_variants import (
    DECAFDiffusionMethod,
    DECAFDPDiffusionMethod,
)

TOY_DATASET_NAME = "_toy_decaf_diffusion_test"


@pytest.fixture(autouse=True)
def _register_toy_causal_graph():
    # Same A/R/B/Y scenario as test_decaf_variants.py: A protected, R
    # admissible, B inadmissible mediator, Y outcome.
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
# Schedule / embedding unit tests
# --------------------------------------------------------------------------


def test_diffusion_schedule_alpha_bars_decrease_monotonically():
    sched = DiffusionSchedule(timesteps=20, beta_start=1e-4, beta_end=0.2)
    ab = sched.alpha_bars
    assert (ab[1:] < ab[:-1]).all()
    assert ab[0] <= 1.0 and ab[-1] >= 0.0


def test_diffusion_schedule_q_sample_matches_closed_form():
    sched = DiffusionSchedule(timesteps=10)
    x0 = torch.ones(4, 3)
    noise = torch.zeros(4, 3)
    t = torch.tensor([0, 3, 6, 9])
    x_t = sched.q_sample(x0, t, noise)
    # With zero noise, x_t should equal sqrt(alpha_bar_t) * x0 exactly.
    expected = sched.sqrt_alpha_bars[t].unsqueeze(-1) * x0
    assert torch.allclose(x_t, expected)


def test_timestep_embedding_distinguishes_timesteps():
    embed = SinusoidalTimeEmbedding(dim=16)
    t = torch.tensor([0, 25, 49])
    out = embed(t)
    assert out.shape == (3, 16)
    # Different timesteps must produce different embeddings, or the
    # per-column denoiser has no way to tell diffusion steps apart.
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])


def test_timestep_embedding_handles_odd_dim():
    embed = SinusoidalTimeEmbedding(dim=15)
    out = embed(torch.tensor([0, 1]))
    assert out.shape == (2, 15)


# --------------------------------------------------------------------------
# Trainer-level tests (small, direct instantiation)
# --------------------------------------------------------------------------


def test_masked_parents_zeroes_non_parent_blocks():
    blocks = ColumnBlocks(["A", "R"], {"A": 2, "R": 2})
    dag_seed = [[0, 1]]  # A -> R
    trainer = DDPMCausalTrainer(blocks, dag_seed, h_dim=8, time_dim=4, timesteps=5)
    generated = torch.tensor([[1.0, 0.0, 0.5, 0.5]])  # A one-hot, R mid-generation
    masked = trainer.model.masked_parents(generated, j=1, biased_edges=None)
    # R's only parent is A, so R's own block must be zeroed in the masked view.
    assert torch.equal(masked, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_masked_parents_shuffles_biased_edge():
    blocks = ColumnBlocks(["A", "R"], {"A": 2, "R": 2})
    dag_seed = [[0, 1]]
    trainer = DDPMCausalTrainer(blocks, dag_seed, h_dim=8, time_dim=4, timesteps=5)
    generated = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    generated = torch.cat([generated, torch.zeros(2, 2)], dim=1)  # pad to R's block
    torch.manual_seed(0)
    masked = trainer.model.masked_parents(generated, j=1, biased_edges={1: [0]})
    # A's block (the biased parent) must still be a valid one-hot row after
    # shuffling -- exactly one of the two columns set for each sample.
    assert torch.equal(masked[:, :2].sum(dim=1), torch.ones(2))


# --------------------------------------------------------------------------
# End-to-end (matches the pattern in test_decaf_variants.py)
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
        lambda: DECAFDiffusionMethod(max_epochs=3, batch_size=64, timesteps=10),
        lambda: DECAFDPDiffusionMethod(max_epochs=3, batch_size=64, timesteps=10),
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
def test_dp_diffusion_reports_its_privacy_calibration(toy_dataset):
    result = _run(
        DECAFDPDiffusionMethod(max_epochs=3, batch_size=64, timesteps=10), toy_dataset
    )
    assert result.extra["noise_multiplier"] > 0
    assert result.extra["dp_steps"] > 0
    assert result.extra["spent_epsilon"] <= 10.0 + 1e-6


@pytest.mark.slow
def test_tighter_epsilon_gets_more_noise_end_to_end(toy_dataset):
    loose = _run(
        DECAFDPDiffusionMethod(max_epochs=3, batch_size=64, timesteps=10),
        toy_dataset, epsilon=100.0,
    )
    tight = _run(
        DECAFDPDiffusionMethod(max_epochs=3, batch_size=64, timesteps=10),
        toy_dataset, epsilon=1.0,
    )
    assert tight.extra["noise_multiplier"] > loose.extra["noise_multiplier"]


@pytest.mark.slow
def test_fairness_mechanisms_reach_the_generator(toy_dataset):
    """Same regression the GAN backbones guard against: each mechanism must
    hand the generator a different edge set, or FTU/DP/CF would be silently
    identical."""
    import networkx as nx

    dag = nx.DiGraph(causal_graphs.CAUSAL_GRAPHS[TOY_DATASET_NAME])
    _, _, roles = toy_dataset
    selections = {
        name: get_fairness_mechanism(name).select_biased_edges(dag, roles)
        for name in ("none", "ftu", "dp", "cf")
    }
    assert selections["none"] == {}
    assert selections["ftu"], "FTU should cut the direct A->Y edge"
    assert selections["dp"] != selections["ftu"]
    assert selections["cf"] != selections["dp"]


@pytest.mark.slow
def test_train_cache_is_reused_across_fairness_mechanisms(toy_dataset):
    """Fairness acts only at generation time here too, so training must be
    shared across FTU/DP/CF -- this is what makes the backbone affordable in
    the grid, same as the GAN variants."""
    method = DECAFDiffusionMethod(max_epochs=3, batch_size=64, timesteps=10)
    _run(method, toy_dataset, mechanism="none")
    assert len(method._cache) == 1
    trained = next(iter(method._cache.values()))
    for mechanism in ("ftu", "dp", "cf"):
        _run(method, toy_dataset, mechanism=mechanism)
        assert next(iter(method._cache.values())) is trained
