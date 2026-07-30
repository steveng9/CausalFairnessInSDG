"""DECAF (Kyono, van Breugel, Berrevoets, van der Schaar; NeurIPS 2021)
integration -- imports the actual maintained `decaf-synthetic-data` package
(github.com/trentkyono/DECAF, BSD-3-Clause) rather than reimplementing its
GAN, so results are directly comparable to the paper's real implementation.
Requires the `decaf` extra: `pip install -e '.[decaf]'`.

DECAF is architecturally different from MST/PrivBayes/AIM/PrivSyn: it needs a
full ground-truth causal DAG as input (`data.causal_graphs`) rather than
discovering one privately, and applies fairness by shuffling ("surrogate
value substitution") specific parent contributions at generation time
instead of restricting which structure gets measured in the first place. It
also has no DP mechanism at all -- `epsilon`/`delta` are accepted for
`SDGMethod` interface parity but unused, and callers should log them as NULL
in `experiments.db`, marking DECAF rows as a non-private baseline rather than
a competing member of the graphical-DP-synthesizer family this project
otherwise targets.

Preprocessing choice: DECAF's own reference usage (`tests/utils.py::load_adult`
in the upstream repo) label-encodes categories to raw integers and feeds them
directly into the network -- no one-hot expansion. We do the same, reusing
the ordinally-encoded dataframe `data/datasets.py` already produces for
MST/PrivBayes/AIM. This keeps `x_dim == len(columns)`, so `dag_seed` and
`biased_edges` index directly by column position with no block-expansion
bookkeeping.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.causal_graphs import CAUSAL_GRAPHS, as_digraph
from ..fairness.base import AttributeRoles, FairnessMechanism
from .base import SDGMethod, SDGResult

try:
    import pytorch_lightning as pl
    import torch
    from decaf import DECAF as _DECAFModel
    from decaf.data import DataModule
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    _IMPORT_ERROR: Optional[ImportError] = exc
else:
    _IMPORT_ERROR = None


class DECAFMethod(SDGMethod):
    name = "decaf"
    is_private = False

    def __init__(
        self,
        max_epochs: int = 10,
        h_dim: int = 200,
        batch_size: int = 64,
        accelerator: str = "cpu",
        devices="auto",
        output_activation: Optional[str] = "sigmoid",
        cache_trained_models: bool = True,
    ):
        self.max_epochs = max_epochs
        self.h_dim = h_dim
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.devices = devices
        # Output-head configuration, which also selects the input scaling --
        # the two only make sense together:
        #
        #   None      DECAF exactly as shipped: unbounded linear heads fed raw
        #             ordinal codes. Works well when every column has a
        #             similar, narrow code range (COMPAS: all 0-5).
        #   "sigmoid" DECAF's own `nonlin_out` hook bounding every column to
        #             [0, 1], with the data min-max scaled to match. Needed
        #             when code ranges differ wildly (Adult spans 0-1 for
        #             `income` up to 0-40 for `native-country`): with linear
        #             heads the wide columns dominate the loss and the narrow
        #             ones collapse -- Adult's binary `income` degenerated to
        #             a single class, along with 3 other constant columns.
        #
        # Measured on this project's data (1-way TVD, lower better):
        # COMPAS 0.048 unbounded vs 0.321 sigmoid; Adult collapses when
        # unbounded vs 0.438 sigmoid. So it is genuinely per-dataset, and the
        # experiment grid sets it per dataset rather than picking one globally.
        self.output_activation = output_activation
        # DECAF's fairness mechanisms act *only* at generation time
        # (`biased_edges` shuffling inside `gen_synthetic`) -- the trained
        # weights are identical across FTU/DP/CF/none and across different
        # protected/admissible role splits of the same data. Caching the
        # fitted model therefore trains once per (data, seed, hyperparams)
        # instead of once per experiment cell, which is a 4x-plus saving on
        # the single most expensive method in the grid. The cache key hashes
        # the actual training array, so any change to the data invalidates it.
        self.cache_trained_models = cache_trained_models
        self._train_cache: Dict[Tuple, object] = {}

    def _fit_model(self, values: np.ndarray, dag_seed, n_cols: int, seed):
        key = (
            hashlib.sha1(values.tobytes()).hexdigest(),
            tuple(tuple(e) for e in dag_seed),
            n_cols,
            seed,
            self.max_epochs,
            self.h_dim,
            self.batch_size,
            self.output_activation,
        )
        if self.cache_trained_models and key in self._train_cache:
            return self._train_cache[key]

        dm = DataModule(values, batch_size=self.batch_size)
        model = _DECAFModel(
            input_dim=n_cols,
            dag_seed=dag_seed,
            h_dim=self.h_dim,
            nonlin_out=(
                [(self.output_activation, n_cols)] if self.output_activation else None
            ),
        )
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(model, dm)
        model = model.cpu().eval()
        if self.cache_trained_models:
            self._train_cache.clear()  # only ever need the most recent dataset
            self._train_cache[key] = model
        return model

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
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "DECAF requires the optional 'decaf' extra: "
                "pip install -e '.[decaf]' (decaf-synthetic-data, torch, "
                "pytorch-lightning)."
            ) from _IMPORT_ERROR
        if dataset_name not in CAUSAL_GRAPHS:
            raise ValueError(
                f"No causal graph registered for dataset_name={dataset_name!r}; "
                f"available: {sorted(CAUSAL_GRAPHS)}"
            )

        if seed is not None:
            pl.seed_everything(seed)

        columns = list(data.columns)
        col_index = {c: i for i, c in enumerate(columns)}
        dag = as_digraph(CAUSAL_GRAPHS[dataset_name], columns)
        dag_seed = [[col_index[p], col_index[c]] for p, c in dag.edges]

        # Min-max scale onto [0, 1] iff the generator's heads are bounded to
        # [0, 1] too (see `output_activation`); DECAF's `DataModule` does no
        # normalization of its own. Either way this is preprocessing on our
        # side only -- the GAN, its training loop, and the `biased_edges`
        # mechanism are untouched.
        if self.output_activation == "sigmoid":
            scale = np.array([max(domain[c] - 1, 1) for c in columns], dtype="float32")
        else:
            scale = np.ones(len(columns), dtype="float32")
        values = data.to_numpy(dtype="float32") / scale
        model = self._fit_model(values, dag_seed, len(columns), seed)

        # Re-seed before generation so a cache hit samples the same way a
        # fresh train-then-generate would have.
        if seed is not None:
            pl.seed_everything(seed)

        biased_edges_by_name = fairness_mechanism.select_biased_edges(dag, roles)
        biased_edges = {
            col_index[child]: [col_index[p] for p in parents]
            for child, parents in biased_edges_by_name.items()
        }

        if n_synth == len(values):
            seed_rows = values
        else:
            rng = np.random.default_rng(seed)
            seed_rows = values[rng.choice(len(values), size=n_synth, replace=True)]

        x = torch.as_tensor(seed_rows)
        synth = model.gen_synthetic(x, biased_edges=biased_edges).detach().cpu().numpy()

        synth_df = pd.DataFrame(synth * scale, columns=columns)
        for col in columns:
            synth_df[col] = synth_df[col].round().clip(0, domain[col] - 1).astype(int)

        return SDGResult(synthetic_data=synth_df, graph_edges=list(dag.edges))
