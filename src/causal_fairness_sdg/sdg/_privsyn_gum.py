"""PrivSyn (Zhang et al., USENIX Security '21), built from the paper's
description rather than vendored from an existing package -- no clean
official implementation is importable (see README.md for what was checked).
Two pieces, mirroring the paper's two-stage design:

  - Marginal *selection*: privately choose a set of 2-way marginals under
    budget (the paper's "InDif"-scored, "DenseMarg"-selected approach).
    Structurally this reuses the exact same private-selection idiom as
    `sdg/mst.py`/`sdg/aim.py` -- score candidates, pick via the exponential
    mechanism, repeat -- just without a spanning-tree/connectivity
    constraint (`select_marginals_fair` below can select any subset of
    pairwise marginals, not a tree). The `FairnessMechanism` hooks apply
    exactly as they do for MST/AIM: a static `filter_candidates` pass plus
    an incremental `allow_edge` check against the graph of attributes
    connected so far by an already-selected marginal.
  - *Generation*: PrivSyn does not use graphical-model/junction-tree
    sampling like MST/AIM/PrivBayes -- it uses GUM (Gradually Update
    Method), which repeatedly nudges a randomly-initialized record pool's
    *joint* distribution toward each selected marginal by duplicating
    under-represented attribute-value combinations and overwriting
    over-represented ones, rather than fitting a probabilistic model.
    `gum_generate` below is a direct implementation of that idea: for each
    marginal (in shuffled order, repeated for several passes), compare the
    pool's current joint counts on that marginal's attributes to the noisy
    target counts, and duplicate-and-replace records to close the gap.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from mbi import Dataset

from ..fairness.base import AttributeRoles, Edge, FairnessMechanism
from .mst import exponential_mechanism

Clique = Tuple[str, str]


def select_marginals_fair(
    data: Dataset,
    rho: float,
    roles: AttributeRoles,
    mechanism: FairnessMechanism,
    target_count: Optional[int] = None,
) -> Tuple[List[Clique], List[Tuple[Clique, np.ndarray]]]:
    """Privately select up to `target_count` 2-way marginals (default: a
    heuristic capping how many of the (already fairness-filtered) candidate
    pairs get selected, matching PrivSyn's own "many but budget-limited"
    framing rather than always selecting every candidate pair) via repeated
    exponential-mechanism draws -- no spanning-tree constraint, unlike MST.
    Splits `rho` in half between the selection rounds and a single
    Gaussian-noise measurement pass over the selected marginals, the same
    two-phase split `sdg/mst.py::MST.fit_generate` uses.

    Returns `(selected_cliques, [(clique, noisy_datavector), ...])`.
    """
    import itertools

    candidates = list(itertools.combinations(data.domain.attrs, 2))
    candidates = mechanism.filter_candidates(candidates, roles)
    if not candidates:
        return [], []

    target_count = target_count or min(len(candidates), 3 * len(data.domain))
    weights = {cl: data.project(list(cl)).datavector().std() for cl in candidates}

    graph = nx.Graph()
    graph.add_nodes_from(data.domain.attrs)
    selected: List[Clique] = []

    rounds = min(target_count, len(candidates))
    select_rho = rho / 2
    epsilon = np.sqrt(8 * select_rho / rounds) if rounds else 0.0
    for _ in range(rounds):
        round_candidates = [
            cl for cl in candidates if cl not in selected and mechanism.allow_edge(cl, graph, roles)
        ]
        if not round_candidates:
            break
        scores = np.array([weights[cl] for cl in round_candidates])
        idx = exponential_mechanism(scores, epsilon, sensitivity=float(scores.max() or 1.0))
        cl = round_candidates[idx]
        selected.append(cl)
        graph.add_edge(*cl)

    if not selected:
        return [], []

    measure_rho = rho / 2
    sigma = np.sqrt(len(selected) / (2 * measure_rho))
    measurements = []
    for cl in selected:
        x = data.project(list(cl)).datavector()
        y = x + np.random.normal(0, sigma, size=x.size)
        measurements.append((cl, np.clip(y, 0, None)))

    return selected, measurements


def _joint_counts(pool: pd.DataFrame, clique: Sequence[str], dims: Sequence[int]) -> np.ndarray:
    idx = np.ravel_multi_index([pool[a].to_numpy() for a in clique], dims)
    return np.bincount(idx, minlength=int(np.prod(dims))).astype(float)


def _gum_update_one_marginal(
    pool: pd.DataFrame, clique: Sequence[str], target: np.ndarray, domain: Dict[str, int], rng: np.random.Generator
) -> None:
    dims = [domain[a] for a in clique]
    current = _joint_counts(pool, clique, dims)
    # round target counts to an integer record budget matching the pool size
    target_int = np.round(target * (len(pool) / max(target.sum(), 1e-9))).astype(int)
    diff = target_int - current.astype(int)

    cell_index = np.ravel_multi_index([pool[a].to_numpy() for a in clique], dims)
    surplus_positions: List[int] = []
    for cell in np.flatnonzero(diff < 0):
        rows = np.flatnonzero(cell_index == cell)
        take = min(len(rows), int(-diff[cell]))
        if take:
            surplus_positions.extend(rng.choice(rows, size=take, replace=False).tolist())
    rng.shuffle(surplus_positions)

    col_positions = {a: pool.columns.get_loc(a) for a in clique}
    pos_iter = iter(surplus_positions)
    for cell in np.flatnonzero(diff > 0):
        combo = np.unravel_index(cell, dims)
        for _ in range(int(diff[cell])):
            row = next(pos_iter, None)
            if row is None:
                break
            for a, v in zip(clique, combo):
                pool.iat[row, col_positions[a]] = v


def gum_generate(
    measurements: List[Tuple[Clique, np.ndarray]],
    domain: Dict[str, int],
    n: int,
    rng: np.random.Generator,
    passes: int = 10,
) -> pd.DataFrame:
    """GUM: initialize a uniform-random record pool, then repeatedly
    duplicate-and-replace records to pull the pool's joint counts on each
    selected marginal toward its noisy target, in shuffled order, for
    `passes` sweeps over all selected marginals."""
    attrs = sorted({a for cl, _ in measurements for a in cl}) or list(domain)
    pool = pd.DataFrame({a: rng.integers(0, domain[a], size=n) for a in attrs})

    order = list(range(len(measurements)))
    for _ in range(passes):
        rng.shuffle(order)
        for i in order:
            clique, target = measurements[i]
            _gum_update_one_marginal(pool, clique, target, domain, rng)

    for a in domain:
        if a not in pool.columns:
            pool[a] = rng.integers(0, domain[a], size=n)
    return pool
