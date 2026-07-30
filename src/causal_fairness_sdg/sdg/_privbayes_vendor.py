"""PrivBayes (Zhang et al. 2014) structure search + conditional-distribution
estimation, adapted from `DataResponsibly/DataSynthesizer`'s
`DataSynthesizer/lib/PrivBayes.py` (MIT License, dataresponsibly.com), with a
`FairnessMechanism` hook injected into the greedy structure search exactly
where MST's `select_fair` (`sdg/mst.py`) injects its own -- one static
`filter_candidates` pass plus an incremental `allow_edge` check per round,
here via the `edges_allowed` helper since PrivBayes candidates are
multi-parent, not single edges.

Deviations from the original, deliberate rather than oversights:
  - No JSON description-file round trip -- structure and conditional
    distributions stay as plain Python/pandas objects, and conditional
    distributions are keyed by parent-value tuples directly rather than
    `str(list(...))` + `eval()` (the original's approach; functionally
    equivalent, avoids `eval`).
  - No separate "independent attribute mode" pre-pass. That step exists in
    DataSynthesizer to privately bin continuous attributes and estimate
    per-attribute marginals for its JSON description format; our datasets
    (`data/datasets.py`) are already discretized into a fixed small-integer
    domain upstream, so this step is redundant -- and skipping it means the
    whole run's privacy budget is a clean `epsilon/2` (structure) +
    `epsilon/2` (conditional distributions), rather than +epsilon more spent
    on outputs this project doesn't consume.
  - PrivBayes is pure epsilon-DP (Laplace + exponential mechanisms); `delta`
    is accepted by the caller (`sdg/privbayes.py`) for `SDGMethod` interface
    parity but unused here, same as DECAF ignores epsilon/delta entirely.
"""

from __future__ import annotations

import itertools
from math import ceil, log
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from sklearn.metrics import mutual_info_score

from ..fairness.base import AttributeRoles, Edge, FairnessMechanism, edges_allowed

BayesianNetwork = List[Tuple[str, List[str]]]


def normalize_given_distribution(frequencies) -> np.ndarray:
    distribution = np.array(frequencies, dtype=float).clip(0)
    total = distribution.sum()
    if total > 0:
        return distribution / total
    return np.full_like(distribution, 1 / distribution.size)


def mutual_information(dataset: pd.DataFrame, child: str, parents: Sequence[str]) -> float:
    y = dataset[child].to_numpy()
    if len(parents) == 1:
        x = dataset[parents[0]].to_numpy()
    else:
        # Combine parent columns into one integer label per row without a
        # row-wise Python-level string join (`" ".join(...)`, what the
        # original DataSynthesizer code does) -- that dominates runtime on
        # wide categorical datasets: for Adult (14 attrs, cardinalities up
        # to 41) it made greedy_bayes_fair take minutes per attribute at
        # k=3. mutual_info_score only needs label *identity*, so any
        # injective combination works; np.ravel_multi_index is vectorized.
        parent_vals = dataset[list(parents)].to_numpy()
        dims = parent_vals.max(axis=0) + 1
        x = np.ravel_multi_index(parent_vals.T, dims=dims)
    return mutual_info_score(x, y)


def calculate_sensitivity(num_tuples: int, is_binary: bool) -> float:
    """PrivBayes Lemma 1."""
    if is_binary:
        a = log(num_tuples) / num_tuples
        b = (num_tuples - 1) / num_tuples
        return a + b * log(num_tuples / (num_tuples - 1))
    a = (2 / num_tuples) * log((num_tuples + 1) / 2)
    b = (1 - 1 / num_tuples) * log(1 + 2 / (num_tuples - 1))
    return a + b


def calculate_delta(num_attributes: int, sensitivity: float, epsilon: float) -> float:
    """PrivBayes Sec. 4.2, "A First-Cut Solution" -- unrelated to (epsilon,
    delta)-DP's delta; this is PrivBayes' own per-round noise-scale term."""
    return (num_attributes - 1) * sensitivity / epsilon


def _usefulness_minus_target(k, num_attributes, num_tuples, target_usefulness, epsilon) -> float:
    if k == num_attributes:
        return target_usefulness
    return num_tuples * epsilon / ((num_attributes - k) * (2 ** (k + 3))) - target_usefulness


def calculate_k(num_attributes: int, num_tuples: int, target_usefulness: float = 4, epsilon: float = 0.1) -> int:
    """PrivBayes Lemma 3: maximum parent-set size."""
    default_k = 3
    if _usefulness_minus_target(default_k, num_attributes, num_tuples, 0, epsilon) > target_usefulness:
        return default_k
    try:
        k = ceil(fsolve(
            _usefulness_minus_target, np.array([int(num_attributes / 2)]),
            args=(num_attributes, num_tuples, target_usefulness, epsilon),
        )[0])
    except Exception:
        return default_k
    return k if 1 <= k <= num_attributes else default_k


def _pb_exponential_mechanism(
    epsilon: float,
    mutual_info_list: List[float],
    parents_pair_list: List[Tuple[str, List[str]]],
    attr_to_is_binary: Dict[str, bool],
    num_tuples: int,
    num_attributes: int,
) -> np.ndarray:
    deltas = []
    for child, parents in parents_pair_list:
        is_binary = attr_to_is_binary[child] or (len(parents) == 1 and attr_to_is_binary[parents[0]])
        sensitivity = calculate_sensitivity(num_tuples, is_binary)
        deltas.append(calculate_delta(num_attributes, sensitivity, epsilon))
    scores = np.array(mutual_info_list) / (2 * np.array(deltas))
    return normalize_given_distribution(np.exp(scores - scores.max()))


def greedy_bayes_fair(
    dataset: pd.DataFrame,
    k: int,
    epsilon: float,
    roles: AttributeRoles,
    mechanism: FairnessMechanism,
    rng: np.random.Generator,
) -> Tuple[BayesianNetwork, List[str]]:
    """Greedy structure search (PrivBayes Algorithm 1's BN-construction loop),
    with the fairness hook applied to each round's candidate (child,
    parent_set) pairs before scoring. Mirrors `sdg/mst.py::select_fair`'s
    two-hook pattern via `edges_allowed`.

    Returns `(bn, roots)`. Normally `roots == [root]` (a single randomly
    chosen root, as in the original algorithm). But once a mechanism has
    forbidden every remaining candidate for some attribute (e.g. under FTU,
    once every already-placed attribute must be used as a parent set and
    that set contains the outcome, a protected attribute can never legally
    be added again), that attribute can never be placed and is left in
    `roots` too, to be sampled independently from its own noisy marginal --
    the same "give up connecting rather than error" behavior MST's DP
    mechanism produces as a spanning forest, adapted to a BN where "won't
    connect" means "root with no parents" rather than "separate tree
    component".
    """
    columns = list(dataset.columns)
    num_tuples, num_attributes = dataset.shape
    if not k:
        k = calculate_k(num_attributes, num_tuples, epsilon=epsilon)
    attr_to_is_binary = {c: dataset[c].nunique() <= 2 for c in columns}

    root = str(rng.choice(columns))
    placed = [root]
    remaining = [c for c in columns if c != root]

    graph = nx.Graph()
    graph.add_nodes_from(columns)

    bn: BayesianNetwork = []
    while remaining:
        num_parents = min(len(placed), k)
        candidates: List[Tuple[str, List[str]]] = []
        mi_values: List[float] = []
        for child in remaining:
            for parents in itertools.combinations(placed, num_parents):
                parents = list(parents)
                if not edges_allowed([(p, child) for p in parents], graph, roles, mechanism):
                    continue
                candidates.append((child, parents))
                mi_values.append(mutual_information(dataset, child, parents))

        if not candidates:
            break

        if epsilon:
            probs = _pb_exponential_mechanism(
                epsilon, mi_values, candidates, attr_to_is_binary, num_tuples, num_attributes
            )
            idx = rng.choice(len(candidates), p=probs)
        else:
            idx = int(np.argmax(mi_values))

        child, parents = candidates[idx]
        bn.append((child, parents))
        for p in parents:
            graph.add_edge(p, child)
        placed.append(child)
        remaining.remove(child)

    return bn, [root, *remaining]


def _noisy_full_domain_counts(
    dataset: pd.DataFrame, attributes: List[str], domain: Dict[str, int], epsilon: float, rng: np.random.Generator
) -> pd.Series:
    """Joint counts of `attributes` over their *full* domain (every
    combination present, zero-filled where unobserved), Laplace-noised.
    Mirrors DataSynthesizer's `get_noisy_distribution_of_attributes`, which
    enumerates the full cartesian product so every conditional distribution
    the sampler needs is always defined (no fallback-to-marginal case)."""
    grouped = dataset.groupby(attributes).size()
    if len(attributes) == 1:
        full_index = pd.Index(range(domain[attributes[0]]), name=attributes[0])
    else:
        full_index = pd.MultiIndex.from_product(
            [range(domain[a]) for a in attributes], names=attributes
        )
    counts = grouped.reindex(full_index, fill_value=0).astype(float)
    if epsilon:
        k = len(attributes) - 1
        num_tuples, num_attributes = dataset.shape
        noise_para = (num_attributes - k) / epsilon  # PrivBayes' laplace_noise_parameter
        counts = counts + rng.laplace(0, noise_para, size=len(counts))
        counts = counts.clip(lower=0)
    return counts


def construct_noisy_conditional_distributions(
    bn: BayesianNetwork,
    roots: List[str],
    dataset: pd.DataFrame,
    domain: Dict[str, int],
    epsilon: float,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Returns `{**{r: prob_array for r in roots}, **{child: {parent_value_tuple: prob_array}}}`."""
    distributions: Dict[str, object] = {
        r: normalize_given_distribution(
            _noisy_full_domain_counts(dataset, [r], domain, epsilon, rng).to_numpy()
        )
        for r in roots
    }
    for child, parents in bn:
        counts = _noisy_full_domain_counts(dataset, parents + [child], domain, epsilon, rng)
        child_dists: Dict[Tuple[int, ...], np.ndarray] = {}
        group_level = 0 if len(parents) == 1 else list(range(len(parents)))
        for parent_values, group in counts.groupby(level=group_level):
            key = parent_values if isinstance(parent_values, tuple) else (parent_values,)
            child_dists[key] = normalize_given_distribution(group.to_numpy())
        distributions[child] = child_dists
    return distributions


def sample_from_bn(
    bn: BayesianNetwork,
    roots: List[str],
    distributions: Dict[str, object],
    domain: Dict[str, int],
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Ancestral sampling from the (noisy) conditional distributions, in BN
    order. Mirrors `DataGenerator.generate_encoded_dataset`, minus the JSON
    description round trip. Every attribute in `roots` is sampled
    independently from its own marginal (see `greedy_bayes_fair`'s
    docstring for when this is more than one attribute)."""
    columns = list(roots) + [child for child, _ in bn]
    df = pd.DataFrame(index=range(n), columns=columns, dtype="int64")
    for r in roots:
        r_dist = distributions[r]
        df[r] = rng.choice(len(r_dist), size=n, p=r_dist)

    for child, parents in bn:
        child_dists = distributions[child]
        values = np.empty(n, dtype="int64")
        for key, idx in df.groupby(parents).groups.items():
            parent_key = key if isinstance(key, tuple) else (key,)
            dist = child_dists.get(parent_key)
            if dist is None:  # full-domain enumeration should make this unreachable
                dist = np.full(domain[child], 1 / domain[child])
            positions = df.index.get_indexer(idx)
            values[positions] = rng.choice(len(dist), size=len(idx), p=dist)
        df[child] = values

    return df
