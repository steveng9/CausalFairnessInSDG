"""MST (McKenna et al. 2021), vendor-adapted with fairness hooks injected into
the structure-selection step.

Core marginal-measurement and inference machinery (`measure`,
`exponential_mechanism`, and the use of `mbi.Dataset`/`Domain`/
`FactoredInference`) is unchanged from private-pgm's reference implementation
(https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py, also
vendored by reprosyn: ../SyntheticData_MIA/reprosyn-main). Only `select()` is
modified -- renamed `select_fair()` -- to call into a `FairnessMechanism` at
the two points described in `fairness/base.py`.

Simplification vs. the original: domain compression (`compress_domain`/
`reverse_data` in private-pgm's version, an optimization for high-cardinality
columns) is omitted in this first pass. Reintroduce if measurement efficiency
becomes a bottleneck on wider datasets.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from disjoint_set import DisjointSet
from mbi import Dataset, Domain, FactoredInference
from scipy import sparse
from scipy.special import logsumexp

from ..fairness.base import AttributeRoles, Edge, FairnessMechanism
from ._cdp2adp import cdp_rho
from .base import SDGMethod, SDGResult


def exponential_mechanism(
    q: np.ndarray, eps: float, sensitivity: float, prng=np.random, monotonic: bool = False
) -> int:
    coef = 1.0 if monotonic else 0.5
    scores = coef * eps / sensitivity * q
    probas = np.exp(scores - logsumexp(scores))
    return prng.choice(q.size, p=probas)


def measure(data: Dataset, cliques: Sequence[tuple], sigma: float, weights=None) -> list:
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    for proj, wgt in zip(cliques, weights):
        x = data.project(proj).datavector()
        y = x + np.random.normal(loc=0, scale=sigma / wgt, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append((Q, y, sigma / wgt, proj))
    return measurements


def select_fair(
    data: Dataset,
    rho: float,
    measurement_log: list,
    roles: AttributeRoles,
    mechanism: FairnessMechanism,
    cliques: Sequence[Edge] = (),
) -> List[Edge]:
    """MST's structure-selection step with fairness hooks.

    Mirrors the original: candidate generation -> noisy L1 scoring against a
    coarse graphical-model estimate -> per-round exponential-mechanism
    Kruskal's construction. Two additions:

      1. `mechanism.filter_candidates` statically prunes the candidate list
         once, before any scoring happens (used by FTU).
      2. `mechanism.allow_edge` is checked each round, after the usual
         same-component filter and before the exponential mechanism runs
         (used by CF and DP). If no candidates remain allowed in a round,
         selection stops early rather than raising -- the result is a
         spanning *forest* rather than a full spanning tree. This is how
         DP's disconnection requirement actually manifests structurally.
    """
    engine = FactoredInference(data.domain, iters=50)
    est = engine.estimate(measurement_log)

    candidates = list(itertools.combinations(data.domain.attrs, 2))
    candidates = mechanism.filter_candidates(candidates, roles)

    weights: Dict[Edge, float] = {}
    for a, b in candidates:
        xhat = est.project([a, b]).datavector()
        x = data.project([a, b]).datavector()
        weights[a, b] = np.linalg.norm(x - xhat, 1)

    T = nx.Graph()
    T.add_nodes_from(data.domain.attrs)
    ds: DisjointSet = DisjointSet()
    for e in cliques:
        T.add_edge(*e)
        ds.union(*e)

    r = len(list(nx.connected_components(T)))
    if r > 1:
        epsilon = np.sqrt(8 * rho / (r - 1))
        for _ in range(r - 1):
            round_candidates = [e for e in candidates if not ds.connected(*e)]
            round_candidates = [
                e for e in round_candidates if mechanism.allow_edge(e, T, roles)
            ]
            if not round_candidates:
                break
            wgts = np.array([weights[e] for e in round_candidates])
            idx = exponential_mechanism(wgts, epsilon, sensitivity=1.0)
            e = round_candidates[idx]
            T.add_edge(*e)
            ds.union(*e)

    return list(T.edges)


class MST(SDGMethod):
    name = "mst"

    def fit_generate(
        self,
        data,
        domain: Dict[str, int],
        roles: AttributeRoles,
        fairness_mechanism: FairnessMechanism,
        epsilon: float,
        delta: float,
        n_synth: int,
        seed: Optional[int] = None,
    ) -> SDGResult:
        if seed is not None:
            np.random.seed(seed)

        dataset = Dataset(data, Domain.fromdict(domain))
        rho = cdp_rho(epsilon, delta)
        sigma = np.sqrt(3 / (2 * rho))

        cliques_1way = [(col,) for col in dataset.domain]
        log1 = measure(dataset, cliques_1way, sigma)

        edges = select_fair(dataset, rho / 3.0, log1, roles, fairness_mechanism)

        log2 = measure(dataset, edges, sigma) if edges else []
        engine = FactoredInference(dataset.domain, iters=1000)
        est = engine.estimate(log1 + log2)
        synth = est.synthetic_data(n_synth)

        return SDGResult(synthetic_data=synth.df, graph_edges=edges)
