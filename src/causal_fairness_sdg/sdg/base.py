from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..fairness.base import AttributeRoles, FairnessMechanism

#: Datasets whose torch-backbone training settings have not been tuned. Each
#: trainer runs them on its SNAKE settings (SNAKE being the same kind of wide,
#: mixed-cardinality table) until an epoch sweep says otherwise. To retire a
#: name from this list, give it its own entry in every `settings` table in
#: `decaf_variants.py` and in `run_experiments.DECAF_SETTINGS`.
UNTUNED_DATASETS = (
    "acsincome_ca", "acsincome_tx", "acspubcov_ca", "acspubcov_tx", "cdc_diabetes",
    "hmda_ca", "hmda_tx",
)
UNTUNED_SETTINGS_SOURCE = "snake"


def borrow_untuned_settings(settings: Dict[str, Dict[str, Any]]) -> None:
    """Fill `settings` for every untuned dataset that has no entry of its own."""
    if UNTUNED_SETTINGS_SOURCE in settings:
        for name in UNTUNED_DATASETS:
            settings.setdefault(name, dict(settings[UNTUNED_SETTINGS_SOURCE]))


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

    #: The fairness mechanisms this method implements, or None for all of
    #: them. The grid gives a method only these (`run_experiments.build_configs`).
    mechanisms: Optional[Tuple[str, ...]] = None

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
        AIM, PrivSyn). The DECAF-style generators need it to look up their
        causal DAG (`data.causal_graphs.resolve_graph`), since -- unlike the
        others -- they take a structure as input.
        """
        raise NotImplementedError
