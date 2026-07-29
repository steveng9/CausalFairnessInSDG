from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..fairness.base import AttributeRoles, FairnessMechanism


@dataclass
class SDGResult:
    synthetic_data: pd.DataFrame
    graph_edges: List[Tuple[str, str]]


class SDGMethod(ABC):
    """Common interface every graphical/marginal-based SDG method implements,
    so the experiment runner can swap MST/PrivBayes/(future methods) without
    caring which one it's driving."""

    name: str

    @abstractmethod
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
        """Fit the private graphical model under the given fairness mechanism
        and return the synthetic data plus the structure (edges) selected."""
        raise NotImplementedError
