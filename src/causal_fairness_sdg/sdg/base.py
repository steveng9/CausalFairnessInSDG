from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..fairness.base import AttributeRoles, FairnessMechanism


@dataclass
class SDGResult:
    synthetic_data: pd.DataFrame
    graph_edges: List[Tuple[str, str]]

    #: Method-specific diagnostics to record alongside the run, e.g. the DP-SGD
    #: noise multiplier and step count for the DP-GAN backbones. Merged into
    #: the run's `extra_params`, so the privacy calibration that produced a row
    #: is recoverable from `experiments.db` rather than only from the code.
    extra: Dict[str, Any] = field(default_factory=dict)


class SDGMethod(ABC):
    """Common interface every graphical/marginal-based SDG method implements,
    so the experiment runner can swap MST/PrivBayes/(future methods) without
    caring which one it's driving."""

    name: str

    #: Whether `epsilon`/`delta` actually mean anything for this method. False
    #: for non-private baselines (DECAF), which accept them only for interface
    #: parity -- the runner logs NULL epsilon for those so `experiments.db`
    #: never implies a privacy guarantee that wasn't provided.
    is_private: bool = True

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
        dataset_name: Optional[str] = None,
    ) -> SDGResult:
        """Fit the private graphical model under the given fairness mechanism
        and return the synthetic data plus the structure (edges) selected.

        `dataset_name` (the key into `data.datasets.DATASETS`) is unused by
        methods that discover their own structure privately (MST, PrivBayes,
        AIM, PrivSyn). DECAF needs it to look up its ground-truth causal DAG
        from `data.causal_graphs.CAUSAL_GRAPHS`, since -- unlike the others --
        it doesn't discover a structure at all, it requires one as input.
        """
        raise NotImplementedError
