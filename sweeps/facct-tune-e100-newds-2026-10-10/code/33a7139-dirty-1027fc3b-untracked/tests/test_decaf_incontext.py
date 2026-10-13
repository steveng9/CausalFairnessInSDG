"""The estimator-agnostic half of `decaf_tabpfn`: generation order, root
marginals, tabulated vs. on-demand columns, and DECAF's edge cut. Driven by an
exact conditional-frequency estimator, so no foundation model is imported."""

import numpy as np
import pandas as pd
import pytest

from causal_fairness_sdg.experiments.runner import SDG_METHODS
from causal_fairness_sdg.fairness import AttributeRoles, get_fairness_mechanism
from causal_fairness_sdg.sdg._decaf_incontext import InContextCausalSampler


class FrequencyEstimator:
    """Empirical `p(child | parents)`, uniform where the parents are unseen.
    Counts how often it is fitted and how many rows it is asked about."""

    def __init__(self):
        self.fits = 0
        self.rows_asked = 0
        self.released = 0

    def fit(self, parents, child, n_classes):
        self.fits += 1
        outer = self
        frame = pd.DataFrame(parents)
        frame["child"] = child
        counts = {
            key: np.bincount(group["child"], minlength=n_classes)
            for key, group in frame.groupby(list(range(parents.shape[1])))
        }

        class Fitted:
            def predict_proba(self, query):
                outer.rows_asked += len(query)
                rows = [counts.get(tuple(row), np.ones(n_classes)) for row in query]
                return np.array(rows, dtype=float)

            def release(self):
                outer.released += 1

        return Fitted()


# sex -> job -> pay, and sex -> pay directly.
CARDS = [2, 3, 2]
DAG = [[0, 1], [1, 2], [0, 2]]


def _real(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, n)
    job = np.where(rng.random(n) < 0.8, sex * 2, 1)
    pay = (rng.random(n) < 0.15 + 0.3 * (job == 2) + 0.4 * sex).astype(int)
    return np.stack([sex, job, pay], axis=1)


def _fit(table_cap=100_000):
    estimator = FrequencyEstimator()
    sampler = InContextCausalSampler(CARDS, DAG, estimator, table_cap=table_cap)
    return sampler.fit(_real()), estimator


def _gap(rows):
    return abs(rows[rows[:, 0] == 1, 2].mean() - rows[rows[:, 0] == 0, 2].mean())


def test_uncut_generation_reproduces_the_joint():
    sampler, _ = _fit()
    real = _real()
    synth = sampler.generate(40_000, rng=np.random.default_rng(1))
    for column, card in enumerate(CARDS):
        np.testing.assert_allclose(
            np.bincount(synth[:, column], minlength=card) / len(synth),
            np.bincount(real[:, column], minlength=card) / len(real),
            atol=0.02,
        )
    assert _gap(synth) == pytest.approx(_gap(real), abs=0.03)


def test_cutting_every_edge_into_the_outcome_removes_the_gap():
    sampler, _ = _fit()
    cut = sampler.generate(40_000, {2: [0, 1]}, np.random.default_rng(1))
    assert _gap(cut) < 0.02
    # The cut permutes the conditioning only: both parents keep their joint.
    assert np.corrcoef(cut[:, 0], cut[:, 1])[0, 1] > 0.5


def test_cutting_only_the_direct_edge_leaves_the_mediated_path():
    sampler, _ = _fit()
    uncut = sampler.generate(40_000, rng=np.random.default_rng(1))
    direct = sampler.generate(40_000, {2: [0]}, np.random.default_rng(1))
    assert 0.05 < _gap(direct) < _gap(uncut) - 0.1


def test_small_columns_are_tabulated_once_and_never_asked_again():
    sampler, estimator = _fit()
    assert estimator.fits == 2 and estimator.released == 2
    asked = estimator.rows_asked
    assert asked == 2 + 2 * 3  # every configuration of each column's parents
    for edges in ({}, {2: [0]}, {2: [0, 1]}):
        sampler.generate(5_000, edges, np.random.default_rng(3))
    assert estimator.rows_asked == asked


def test_large_columns_are_asked_on_demand_and_remembered():
    sampler, estimator = _fit(table_cap=0)
    assert estimator.rows_asked == 0 and estimator.released == 0
    sampler.generate(5_000, rng=np.random.default_rng(3))
    first = estimator.rows_asked
    assert 0 < first <= 2 + 6  # distinct configurations, not rows
    sampler.generate(5_000, {2: [0, 1]}, np.random.default_rng(4))
    assert estimator.rows_asked <= 2 + 6
    assert sampler.stats()["incontext_lazy_columns"] == 2


def test_both_paths_give_the_same_rows():
    tabulated = _fit()[0].generate(5_000, {2: [0]}, np.random.default_rng(5))
    on_demand = _fit(table_cap=0)[0].generate(5_000, {2: [0]}, np.random.default_rng(5))
    np.testing.assert_array_equal(tabulated, on_demand)


def test_constant_column_needs_no_estimator():
    real = _real()
    real[:, 1] = 2
    estimator = FrequencyEstimator()
    sampler = InContextCausalSampler(CARDS, DAG, estimator).fit(real)
    assert estimator.fits == 1  # pay only
    assert (sampler.generate(1_000, rng=np.random.default_rng(0))[:, 1] == 2).all()


def test_method_is_registered_and_cuts_what_the_mechanism_selects(monkeypatch):
    """`decaf_tabpfn` end to end on COMPAS with the frequency estimator in
    TabPFN's place: the registered DAG, the mechanism's edges, the provenance."""
    from causal_fairness_sdg.data.datasets import DATASETS
    from causal_fairness_sdg.sdg import decaf_tabpfn

    method = SDG_METHODS["decaf_tabpfn"]
    assert not method.is_private
    monkeypatch.setattr(method, "_require_backend", lambda: None)
    monkeypatch.setattr(
        decaf_tabpfn, "TabPFNConditional", lambda **kwargs: FrequencyEstimator()
    )
    monkeypatch.setattr(method, "_provenance", lambda checkpoint, rows: {"rows": rows})
    method._cache.clear()

    data, domain, _ = DATASETS["compas"]()
    roles = AttributeRoles.create({"race"}, {"priors_count"}, {"two_year_recid"})
    gaps = {}
    for name in ("none", "dp"):
        result = method.fit_generate(
            data, domain, roles, get_fairness_mechanism(name),
            epsilon=float("nan"), delta=1e-9, n_synth=20_000, seed=0,
            dataset_name="compas",
        )
        synth = result.synthetic_data
        assert list(synth.columns) == list(data.columns)
        assert all(synth[c].between(0, domain[c] - 1).all() for c in synth)
        assert result.extra["rows"] == len(data)
        # The two largest groups: the small ones are too noisy at this size.
        largest = synth["race"].value_counts().index[:2]
        rates = synth.groupby("race")["two_year_recid"].mean()[largest]
        gaps[name] = rates.max() - rates.min()
    assert gaps["dp"] < 0.03 < gaps["none"]
    assert len(method._cache) == 1  # one fit served both mechanisms
    method._cache.clear()
