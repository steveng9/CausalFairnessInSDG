"""PrivBayes (Zhang et al. 2014), structure search + sampling vendored from
`DataResponsibly/DataSynthesizer` into `sdg/_privbayes_vendor.py` (see that
module's docstring for what was adapted and why), with the same
`FairnessMechanism` hooks `sdg/mst.py::select_fair` uses.

Targets DataSynthesizer's pure-Python implementation rather than reprosyn's
compiled C++ version, which has no Python-level hook point before its greedy
structure search (see README.md, "Prior art: PreFair" section).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..fairness.base import AttributeRoles, FairnessMechanism
from ._privbayes_vendor import (
    construct_noisy_conditional_distributions,
    greedy_bayes_fair,
    sample_from_bn,
)
from .base import SDGMethod, SDGResult


class PrivBayes(SDGMethod):
    name = "privbayes"

    def __init__(self, k: int = 0):
        # k=0 -> auto-computed max parent-set size (PrivBayes Lemma 3).
        self.k = k

    def fit_generate(
        self,
        data: pd.DataFrame,
        domain: Dict[str, int],
        roles: AttributeRoles,
        fairness_mechanism: FairnessMechanism,
        epsilon: float,
        delta: float,
        n_synth: int,
        seed: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ) -> SDGResult:
        self._validate_domain(data, domain)
        rng = np.random.default_rng(seed)

        # PrivBayes is pure epsilon-DP (Laplace + exponential mechanisms);
        # `delta` is accepted for SDGMethod interface parity but unused, same
        # as DECAF ignores epsilon/delta entirely.
        bn, roots = greedy_bayes_fair(data, self.k, epsilon / 2, roles, fairness_mechanism, rng)
        distributions = construct_noisy_conditional_distributions(
            bn, roots, data, domain, epsilon / 2, rng
        )
        synth = sample_from_bn(bn, roots, distributions, domain, n_synth, rng)

        edges = [(p, child) for child, parents in bn for p in parents]
        return SDGResult(synthetic_data=synth, graph_edges=edges)

    @staticmethod
    def _validate_domain(data: pd.DataFrame, domain: Dict[str, int]) -> None:
        missing = set(domain) - set(data.columns)
        if missing:
            raise ValueError(
                f"domain references columns not present in data: {sorted(missing)}"
            )
