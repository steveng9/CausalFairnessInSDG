"""Utility metrics comparing synthetic data to the original. Mirrors PreFair's
evaluation protocol (Sec 5.1: 1-way/2-way marginal TVD, Cramer's V-based
correlation difference, downstream classifier accuracy) for direct
comparability with published numbers.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency
from scipy.special import rel_entr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


def _shared_columns(real: pd.DataFrame, synth: pd.DataFrame) -> List[str]:
    return [c for c in real.columns if c in synth.columns]


def _marginal_tvd(real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]) -> float:
    real_counts = real.groupby(cols).size() / len(real)
    synth_counts = synth.groupby(cols).size() / len(synth)
    idx = real_counts.index.union(synth_counts.index)
    real_counts = real_counts.reindex(idx, fill_value=0.0)
    synth_counts = synth_counts.reindex(idx, fill_value=0.0)
    return 0.5 * float(np.abs(real_counts - synth_counts).sum())


def one_way_tvd(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Average total variation distance between 1-way marginals."""
    cols = _shared_columns(real, synth)
    if not cols:
        return 0.0
    return float(np.mean([_marginal_tvd(real, synth, [c]) for c in cols]))


def two_way_tvd(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Average total variation distance across all 2-way marginals."""
    pairs = list(combinations(_shared_columns(real, synth), 2))
    if not pairs:
        return 0.0
    return float(np.mean([_marginal_tvd(real, synth, list(p)) for p in pairs]))


def _aligned_marginals(real: pd.DataFrame, synth: pd.DataFrame, col: str):
    """The two 1-way distributions over `col`, on a shared support."""
    p = real[col].value_counts(normalize=True)
    q = synth[col].value_counts(normalize=True)
    idx = p.index.union(q.index)
    return p.reindex(idx, fill_value=0.0).to_numpy(), q.reindex(idx, fill_value=0.0).to_numpy()


def mean_jsd(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Mean Jensen-Shannon divergence (base 2, so in [0, 1]) between the 1-way
    marginals. Unlike TVD it is finite and smooth when a category the real
    table has is missing from the release, which is the failure mode a
    collapsed generator shows."""
    cols = _shared_columns(real, synth)
    if not cols:
        return 0.0
    out = []
    for c in cols:
        p, q = _aligned_marginals(real, synth, c)
        m = 0.5 * (p + q)
        jsd = 0.5 * (rel_entr(p, m).sum() + rel_entr(q, m).sum()) / np.log(2.0)
        out.append(float(np.clip(jsd, 0.0, 1.0)))
    return float(np.mean(out))


def mean_wasserstein(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Mean 1-D Wasserstein distance between the 1-way marginals over the
    integer codes, each normalized by that column's code range so columns with
    41 categories (Adult's `native-country`) do not swamp binary ones. Reads
    the codes as ordered, which is exact for the binned columns (`age`,
    `hours-per-week`, `priors_count`) and a proxy for the nominal ones."""
    cols = _shared_columns(real, synth)
    if not cols:
        return 0.0
    out = []
    for c in cols:
        p, q = _aligned_marginals(real, synth, c)
        support = real[c].value_counts(normalize=True).index.union(
            synth[c].value_counts(normalize=True).index
        ).to_numpy(dtype=float)
        order = np.argsort(support)
        width = support.max() - support.min()
        if width <= 0:
            out.append(0.0)
            continue
        # W1 between discrete distributions on a line = integral of the
        # absolute CDF difference, evaluated on the sorted support.
        gaps = np.diff(support[order])
        cdf_diff = np.abs(np.cumsum(p[order] - q[order])[:-1])
        out.append(float((cdf_diff * gaps).sum() / width))
    return float(np.mean(out))


def nearest_neighbour_privacy(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synth: pd.DataFrame,
    n_sample: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """A distance-to-closest-record privacy check. Every column here is a
    discrete code, so 'distance' is the Hamming distance (fraction of columns
    that differ).

    `dcr_ratio` is the median nearest-neighbour distance from a synthetic row
    to the *training* table divided by the same quantity to a *held-out* real
    table. A generator that memorized its training rows sits closer to them
    than to fresh real people and scores below 1; a value at or above 1 means
    the release is no more revealing of the people who were in the training
    data than of people who were not. `exact_match_rate` is the share of
    synthetic rows that reproduce a training row verbatim."""
    cols = _shared_columns(real_train, synth)
    rng = np.random.default_rng(seed)

    def _sub(df, n):
        if len(df) <= n:
            return df[cols].to_numpy()
        return df[cols].to_numpy()[rng.choice(len(df), n, replace=False)]

    s = _sub(synth, n_sample)
    if len(s) == 0 or not cols:
        return {"dcr_ratio": float("nan"), "exact_match_rate": float("nan")}
    # Mean rather than median: on a table of 14 discrete columns a Hamming
    # distance can only take the values k/14, so a median lands on one of them
    # and the ratio of two medians moves in visible jumps.
    d_train = cdist(s, _sub(real_train, 5 * n_sample), metric="hamming").min(axis=1)
    d_holdout = cdist(s, _sub(real_holdout, 5 * n_sample), metric="hamming").min(axis=1)
    mean_holdout = float(d_holdout.mean())
    ratio = float(d_train.mean() / mean_holdout) if mean_holdout > 0 else float("nan")
    real_rows = set(map(tuple, real_train[cols].to_numpy()))
    matches = np.fromiter(
        (tuple(row) in real_rows for row in synth[cols].to_numpy()), dtype=bool,
        count=len(synth),
    )
    return {"dcr_ratio": ratio, "exact_match_rate": float(matches.mean())}


def _cramers_v(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013), matching PreFair's metric."""
    confusion = pd.crosstab(df[col_a], df[col_b])
    n = confusion.to_numpy().sum()
    if n == 0 or confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(confusion, correction=False)[0]
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def average_correlation_difference(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Mean |Cramer's V(real) - Cramer's V(synth)| across all attribute pairs."""
    pairs = list(combinations(_shared_columns(real, synth), 2))
    if not pairs:
        return 0.0
    diffs = [
        abs(_cramers_v(real, a, b) - _cramers_v(synth, a, b)) for a, b in pairs
    ]
    return float(np.mean(diffs))


_CLASSIFIERS = {
    # early_stopping avoids burning through all 300 iterations once the
    # validation score plateaus -- ~15x faster on Adult-sized data with
    # comparable accuracy (68.6s -> 4.2s in local testing).
    "mlp": lambda: MLPClassifier(
        max_iter=300, random_state=0, early_stopping=True, n_iter_no_change=10
    ),
    "lr": lambda: LogisticRegression(max_iter=1000),
    "rf": lambda: RandomForestClassifier(random_state=0),
}


def fit_classifier(train_data: pd.DataFrame, outcome: str, classifier: str = "mlp"):
    """Fit `classifier` on `train_data`, return (model, feature_cols). Shared
    by `downstream_accuracy` and the experiment runner (which also needs the
    fitted model's predictions for fairness metrics, not just accuracy)."""
    if classifier not in _CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier {classifier!r}; available: {sorted(_CLASSIFIERS)}"
        )
    feature_cols = [c for c in train_data.columns if c != outcome]
    model = _CLASSIFIERS[classifier]()
    model.fit(train_data[feature_cols], train_data[outcome])
    return model, feature_cols


def downstream_scores(
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    outcome: str,
    classifier: str = "mlp",
) -> Dict[str, float]:
    """Train `classifier` on `train_data` (typically synthetic) and score it on
    `eval_data` (typically real held-out data).

    Accuracy alone is a poor read on a table as imbalanced as Adult, where the
    majority class is already 75% and the whole learnable range is 7 points
    wide. AUC (chance = 0.5) and macro-F1 (a classifier that always answers
    with the majority class = 0.41 on a binary outcome) both keep a floor that
    a signal-free release actually falls to, so they separate generators that
    accuracy compresses together."""
    model, feature_cols = fit_classifier(train_data, outcome, classifier)
    y_true = eval_data[outcome].to_numpy()
    preds = model.predict(eval_data[feature_cols])
    scores = {
        "accuracy": float((preds == y_true).mean()),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
    }
    # AUC needs a score, and needs both classes present in the evaluation set;
    # a generator whose release has only one outcome class cannot be ranked.
    try:
        proba = model.predict_proba(eval_data[feature_cols])
        classes = list(model.classes_)
        if len(classes) == 2 and len(np.unique(y_true)) == 2:
            scores["auc"] = float(roc_auc_score(y_true, proba[:, classes.index(1)]))
        else:
            scores["auc"] = float("nan")
    except (AttributeError, ValueError, IndexError):
        scores["auc"] = float("nan")
    return scores


def downstream_accuracy(
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    outcome: str,
    classifier: str = "mlp",
) -> float:
    """Train `classifier` on `train_data` (typically synthetic), evaluate
    accuracy on `eval_data` (typically real held-out data)."""
    return downstream_scores(train_data, eval_data, outcome, classifier)["accuracy"]


def compute_utility_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    outcome: str,
    include_downstream: bool = True,
    real_train: pd.DataFrame = None,
) -> Dict[str, float]:
    """Marginal-fidelity metrics plus (optionally) train-on-synthetic /
    test-on-real scores for each classifier. Set `include_downstream=False`
    when the synthetic outcome column is degenerate (a single class), where no
    classifier can be fitted -- the fidelity metrics are still meaningful and
    worth recording. Pass `real_train` (the table the generator was fit on,
    where `real` is the held-out half) to also get the nearest-neighbour
    privacy pair."""
    metrics = {
        "tvd_1way": one_way_tvd(real, synth),
        "tvd_2way": two_way_tvd(real, synth),
        "jsd_1way": mean_jsd(real, synth),
        "wasserstein_1way": mean_wasserstein(real, synth),
        "avg_correlation_diff": average_correlation_difference(real, synth),
    }
    if real_train is not None:
        metrics.update(nearest_neighbour_privacy(real_train, real, synth))
    if include_downstream:
        for clf in _CLASSIFIERS:
            for name, value in downstream_scores(
                synth, real, outcome, classifier=clf
            ).items():
                key = "accuracy" if name == "accuracy" else name
                metrics[f"downstream_{key}_{clf}"] = value
    return metrics
