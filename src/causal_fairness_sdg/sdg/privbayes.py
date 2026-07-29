"""PrivBayes (Zhang et al. 2014) interface skeleton.

Full integration is scoped as a follow-up. This project targets
`DataResponsibly/DataSynthesizer`'s pure-Python `PrivBayes.py` rather than
reprosyn's compiled C++ version, which has no Python-level hook point before
its greedy structure search (see README.md, "Prior art: PreFair" section) --
so the same `FairnessMechanism` hooks used by `sdg/mst.py`'s `select_fair()`
(a static `filter_candidates` over the initial candidate-parent list per
node, plus an incremental `allow_edge` check on each greedy parent pick) can
be wired in the same way.

This module provides the `SDGMethod`-conformant shell now -- domain/role
validation -- so the experiment runner and database schema can already treat
`"privbayes"` as a first-class method name and log attempted runs; the actual
structure-learning loop raises `NotImplementedError` until it's vendored.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ..fairness.base import AttributeRoles, FairnessMechanism
from .base import SDGMethod, SDGResult


class PrivBayes(SDGMethod):
    name = "privbayes"

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
    ) -> SDGResult:
        self._validate_domain(data, domain)
        raise NotImplementedError(
            "PrivBayes structure learning is not yet vendored into this repo. "
            "Plan: port DataResponsibly/DataSynthesizer's PrivBayes.py greedy "
            "parent-selection loop into this method, injecting "
            "FairnessMechanism.filter_candidates on each node's initial "
            "candidate-parent set and allow_edge on each greedy parent pick -- "
            "mirroring sdg/mst.py's select_fair(). See README.md for the full "
            "gap analysis and the reasoning behind targeting DataSynthesizer "
            "over reprosyn's compiled C++ implementation."
        )

    @staticmethod
    def _validate_domain(data: pd.DataFrame, domain: Dict[str, int]) -> None:
        missing = set(domain) - set(data.columns)
        if missing:
            raise ValueError(
                f"domain references columns not present in data: {sorted(missing)}"
            )
