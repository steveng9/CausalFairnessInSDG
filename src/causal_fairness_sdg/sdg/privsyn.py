"""PrivSyn (Zhang et al., USENIX Security '21) -- see `sdg/_privsyn_gum.py`
for the marginal-selection + GUM implementation and what was built from the
paper's description vs. reused from existing code.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from mbi import Dataset, Domain

from ..fairness.base import AttributeRoles, FairnessMechanism
from ._cdp2adp import cdp_rho
from ._privsyn_gum import gum_generate, select_marginals_fair
from .base import SDGMethod, SDGResult


class PrivSyn(SDGMethod):
    name = "privsyn"

    def __init__(self, gum_passes: int = 10):
        self.gum_passes = gum_passes

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
        dataset_name: Optional[str] = None,
    ) -> SDGResult:
        rng = np.random.default_rng(seed)

        dataset = Dataset(data, Domain.fromdict(domain))
        rho = cdp_rho(epsilon, delta)

        edges, measurements = select_marginals_fair(dataset, rho, roles, fairness_mechanism)
        synth = gum_generate(measurements, domain, n_synth, rng, passes=self.gum_passes)

        return SDGResult(synthetic_data=synth, graph_edges=edges)
