"""AIM (McKenna et al. 2022), the second private-pgm-family method alongside
MST -- see `sdg/_aim_vendor.py` for what was adapted and why. Configured
with a pairwise workload so it's directly comparable to `sdg/mst.py`: same
candidate universe, same fairness hooks, different (adaptive) selection
algorithm and no spanning-tree constraint.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from mbi import Dataset, Domain, FactoredInference

from ..fairness.base import AttributeRoles, FairnessMechanism
from ._aim_vendor import select_aim
from ._cdp2adp import cdp_rho
from .base import SDGMethod, SDGResult


class AIM(SDGMethod):
    name = "aim"

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
        if seed is not None:
            np.random.seed(seed)

        dataset = Dataset(data, Domain.fromdict(domain))
        rho = cdp_rho(epsilon, delta)

        edges, measurements = select_aim(dataset, rho, roles, fairness_mechanism)

        engine = FactoredInference(dataset.domain, iters=1000)
        est = engine.estimate(measurements)
        synth = est.synthetic_data(n_synth)

        return SDGResult(synthetic_data=synth.df, graph_edges=edges)
