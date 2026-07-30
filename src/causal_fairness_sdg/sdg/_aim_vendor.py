"""AIM (McKenna, Miklau, Sheldon 2022, "AIM: An Adaptive and Iterative
Mechanism for Differentially Private Synthetic Data"), adapted from
`private-pgm/mechanisms/aim.py` (Apache 2.0, same upstream as `sdg/mst.py`'s
vendor) -- but re-targeted against the "classic" `mbi` API
(`FactoredInference` + `(Q, y, sigma, proj)` measurement tuples) that
`sdg/mst.py::select_fair` already uses, since the installed `mbi` version
predates the `estimation`/`LinearMeasurement` module split the reference
`aim.py` script imports (it won't run against what's actually installed).

Configured with a *pairwise* workload only (all 2-way marginals, same
candidate universe as MST) -- so the `FairnessMechanism` hooks apply exactly
as they do for MST (`filter_candidates`/`allow_edge` on plain 2-tuples, zero
adaptation), and AIM vs. MST stays an apples-to-apples comparison of the
*selection algorithm*, not the candidate space. AIM's higher-order-clique /
model-size-budgeted mode is a possible future extension, out of scope here.

What's kept from the original algorithm: the core adaptive loop -- repeatedly
pick the worst-approximated candidate under the current model estimate via
the exponential mechanism, measure it (even if already measured before),
refit, and anneal (halve) the per-measurement noise scale once progress on
the just-measured clique stalls. This -- not just a different selection rule
over the same candidates -- is AIM's actual differentiator from MST's
single-pass, fixed-sigma tree construction.

What's dropped: AIM's own model-size-budgeted candidate filtering (unrelated
to our `FairnessMechanism.filter_candidates`) and `initial_cliques`/
higher-order clique support -- both exist to keep a *high-order* graphical
model tractable, which is moot for a pairwise-only workload (model size is
already bounded the same way MST's is).
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from mbi import Dataset, FactoredInference
from scipy import sparse

from ..fairness.base import AttributeRoles, Edge, FairnessMechanism
from .mst import exponential_mechanism


def select_aim(
    data: Dataset,
    rho: float,
    roles: AttributeRoles,
    mechanism: FairnessMechanism,
    rounds: Optional[int] = None,
) -> Tuple[List[Edge], list]:
    """AIM's adaptive measurement loop over a pairwise workload: `rho` is the
    *entire* privacy budget for both the initial 1-way marginals and the
    adaptive 2-way selection loop, following the original algorithm's own
    accounting (a single sigma/epsilon schedule covers both, rather than
    MST's separate fixed splits for its two stages). Returns the distinct
    2-way edges measured at least once, plus the full measurement log for
    the final model fit."""
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    candidates = mechanism.filter_candidates(candidates, roles)
    answers = {cl: data.project(cl).datavector() for cl in candidates}

    rounds = rounds or 16 * len(data.domain)
    sigma = np.sqrt(rounds / (2 * 0.9 * rho))
    epsilon = np.sqrt(8 * 0.1 * rho / rounds)

    oneway = [(col,) for col in data.domain]
    measurements = []
    for cl in oneway:
        x = data.project(cl).datavector()
        y = x + np.random.normal(0, sigma, size=x.size)
        measurements.append((sparse.eye(x.size), y, sigma, cl))
    rho_used = len(oneway) * 0.5 / sigma**2

    # Cheap per-round refits (mirrors MST's select_fair, which uses iters=50
    # for its intermediate estimate and reserves iters=1000 for the caller's
    # final fit after the structure is settled) -- AIM's loop refits once
    # per round, so a full-precision fit here would be `rounds`x more
    # expensive for no benefit until the structure is final.
    engine = FactoredInference(data.domain, iters=50)
    model = engine.estimate(measurements)

    graph = nx.Graph()
    graph.add_nodes_from(data.domain.attrs)
    selected: List[Edge] = []
    selected_set = set()

    terminate = False
    while not terminate:
        remaining = rho - rho_used
        if remaining < 2 * (0.5 / sigma**2 + 0.125 * epsilon**2):
            if remaining <= 0:
                break
            sigma = np.sqrt(1 / (2 * 0.9 * remaining))
            epsilon = np.sqrt(8 * 0.1 * remaining)
            terminate = True
        rho_used += 0.125 * epsilon**2 + 0.5 / sigma**2

        round_candidates = [e for e in candidates if mechanism.allow_edge(e, graph, roles)]
        if not round_candidates:
            break

        bias = np.sqrt(2 / np.pi) * sigma
        errors = np.array([
            np.linalg.norm(answers[cl] - model.project(cl).datavector(), 1)
            - bias * data.domain.size(cl)
            for cl in round_candidates
        ])
        idx = exponential_mechanism(errors, epsilon, sensitivity=1.0)
        cl = round_candidates[idx]

        z = model.project(cl).datavector()
        x = answers[cl]
        y = x + np.random.normal(0, sigma, size=x.size)
        measurements.append((sparse.eye(x.size), y, sigma, cl))
        if cl not in selected_set:
            selected.append(cl)
            selected_set.add(cl)
            graph.add_edge(*cl)

        model = engine.estimate(measurements)
        w = model.project(cl).datavector()
        if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * data.domain.size(cl):
            sigma /= 2
            epsilon *= 2

    return selected, measurements
