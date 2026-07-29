import numpy as np
import pandas as pd
import pytest

from causal_fairness_sdg.eval import fairness_metrics, utility


@pytest.fixture
def real_and_identical_synth():
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame(
        {
            "A": rng.binomial(1, 0.5, n),
            "B": rng.binomial(1, 0.3, n),
            "Y": rng.binomial(1, 0.4, n),
        }
    )
    return df, df.copy()


def test_tvd_is_near_zero_for_identical_distributions(real_and_identical_synth):
    real, synth = real_and_identical_synth
    assert utility.one_way_tvd(real, synth) == pytest.approx(0.0, abs=1e-9)
    assert utility.two_way_tvd(real, synth) == pytest.approx(0.0, abs=1e-9)


def test_tvd_is_positive_for_shifted_distribution():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"A": rng.binomial(1, 0.1, 1000)})
    synth = pd.DataFrame({"A": rng.binomial(1, 0.9, 1000)})
    assert utility.one_way_tvd(real, synth) > 0.5


def test_correlation_diff_is_near_zero_for_identical_distributions(real_and_identical_synth):
    real, synth = real_and_identical_synth
    assert utility.average_correlation_difference(real, synth) == pytest.approx(0.0, abs=1e-9)


def test_downstream_accuracy_is_perfect_on_a_deterministic_rule():
    df = pd.DataFrame({"A": [0, 0, 1, 1] * 50, "Y": [0, 0, 1, 1] * 50})
    acc = utility.downstream_accuracy(df, df, outcome="Y", classifier="lr")
    assert acc == pytest.approx(1.0)


def test_downstream_accuracy_rejects_unknown_classifier():
    df = pd.DataFrame({"A": [0, 1], "Y": [0, 1]})
    with pytest.raises(ValueError):
        utility.downstream_accuracy(df, df, outcome="Y", classifier="not_a_real_model")


def test_demographic_parity_gap_zero_when_independent():
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame(
        {"S": rng.binomial(1, 0.5, n), "O": rng.binomial(1, 0.3, n)}
    )
    gap = fairness_metrics.demographic_parity_gap(df, "S", "O")
    assert abs(gap) < 0.05


def test_demographic_parity_gap_large_when_outcome_determined_by_protected():
    n = 1000
    df = pd.DataFrame({"S": [0] * n + [1] * n, "O": [0] * n + [1] * n})
    gap = fairness_metrics.demographic_parity_gap(df, "S", "O")
    assert gap == pytest.approx(1.0)


def test_conditional_metrics_present_only_when_admissible_attrs_given():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "S": rng.binomial(1, 0.5, n),
            "O": rng.binomial(1, 0.5, n),
            "Y": rng.binomial(1, 0.5, n),
            "R": rng.binomial(1, 0.5, n),
        }
    )
    with_admissible = fairness_metrics.compute_fairness_metrics(df, ["S"], "O", "Y", ["R"])
    without_admissible = fairness_metrics.compute_fairness_metrics(df, ["S"], "O", "Y", [])

    assert "cdp_gap__S" in with_admissible
    assert "cdp_gap__S" not in without_admissible


def test_binarize_protected():
    s = pd.Series(["black", "white", "asian", "white"])
    result = fairness_metrics.binarize_protected(s, "black")
    assert list(result) == [1, 0, 0, 0]
