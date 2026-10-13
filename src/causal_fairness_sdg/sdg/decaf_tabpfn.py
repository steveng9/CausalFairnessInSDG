"""DECAF's causal generator over a tabular foundation model (`decaf_tabpfn`).

The third backbone family after the GANs and the diffusion models in
`decaf_variants.py`: the same hand-specified DAG, the same topological
generation, the same generation-time edge cut for FTU/DP/CF. What changes is
the per-attribute conditional `p(x_j | parents(x_j))`, which is TabPFN's
in-context prediction with the real training rows as the context (Hollmann et
al., Nature 2025; the v3.5 weights). Nothing is trained and there are no
epochs to tune, so the settings below are the whole configuration.

**Not private.** The real rows are the model's prompt; `epsilon`/`delta` are
accepted for interface parity and logged as NULL, as for `decaf`,
`decaf_ctgan` and `decaf_diffusion`.

The DECAF semantics live in `_decaf_incontext.py`, which works with any
estimator of class probabilities; this module is only the TabPFN estimator
and the `SDGMethod` around it.

Settings, and why they are what they are (measured 2026-10-10 on held-out
rows of ACSIncome-CA, 35,000 context rows; `scripts/validate_tabpfn_conditionals.py`
repeats the measurement on any dataset):

  `model_version`        "v3.5". Its checkpoint handles up to 160 classes and
                         10^6 context rows natively, so no column here needs a
                         many-class wrapper and no table needs subsampling.
  `softmax_temperature`  1.0, i.e. the model's predictive distribution as it
                         is. We sample from these probabilities, so sharpening
                         them (earlier checkpoints defaulted to 0.9) would
                         bias every conditional towards its mode. Set
                         explicitly so a library default cannot change it.
  `n_estimators`         8, the checkpoint's own ensemble size. Held-out
                         log-loss is flat from 4 to 32 (0.3885 / 0.3884 /
                         0.3883 / 0.3883 on the outcome column) while cost is
                         linear.
  `table_cap`            100,000 parent configurations: a column below it is
                         tabulated once at fit time, one above it (the
                         outcome) is queried on demand. Changes cost only.
  `max_context_rows`     None: every training row is context.
  `cut_strength`         1.0, DECAF's cut. Below 1 the mechanism's cut is
                         applied to that share of the generated rows only
                         (`InContextCausalSampler.generate`), which traces the
                         fairness-utility curve between `none` and the full
                         mechanism. Not a tuned value: an experimental axis.

Parent columns are passed as their integer codes and TabPFN decides which to
treat as categorical; declaring all of them categorical changed held-out
log-loss in the fourth decimal (2.5555 vs 2.5553 on the 25-level occupation
column).

Needs `tabpfn` (>= 9.1, torch >= 2.5), which the main `sdg` environment
cannot hold alongside its pinned torch. Run this method from the `sdg_tabpfn`
environment -- a clone of `sdg` with the same numpy / pandas / scikit-learn,
so the train/holdout split and the evaluation are identical to every other
method's -- which `scripts/sweep.py` does by itself (see README). The weights
are licence-gated; they are read from TabPFN's cache directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..fairness.base import AttributeRoles, FairnessMechanism
from ._decaf_incontext import FittedConditional, InContextCausalSampler
from .base import SDGResult
from .decaf_variants import _BaseDECAFVariant

#: Query rows per `predict_proba` call. Bounds GPU memory; results do not
#: depend on it beyond float noise (each query row attends to the context only).
PREDICT_CHUNK_ROWS = 20_000


class _FittedTabPFN:
    def __init__(self, classifier, n_classes: int):
        self.classifier = classifier
        self.n_classes = n_classes

    def predict_proba(self, parents: np.ndarray) -> np.ndarray:
        # The classifier reports probabilities for the classes it saw, which
        # after subsampling need not be every code the column may take.
        seen = np.asarray(self.classifier.classes_, dtype=np.int64)
        out = np.zeros((len(parents), self.n_classes), dtype=np.float64)
        for start in range(0, len(parents), PREDICT_CHUNK_ROWS):
            chunk = parents[start:start + PREDICT_CHUNK_ROWS]
            out[start:start + len(chunk), seen] = self.classifier.predict_proba(chunk)
        return out

    def release(self) -> None:
        self.classifier = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class TabPFNConditional:
    """`p(child | parents)` by in-context prediction, one fit per column."""

    model_version: str = "v3.5"
    n_estimators: int = 8
    softmax_temperature: float = 1.0
    device: str = "cpu"
    seed: int = 0
    max_context_rows: Optional[int] = None

    def fit(
        self, parents: np.ndarray, child: np.ndarray, n_classes: int
    ) -> FittedConditional:
        # Read at import by tabpfn; this method never reports usage.
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        if self.max_context_rows is not None and len(child) > self.max_context_rows:
            keep = np.random.default_rng(self.seed).choice(
                len(child), size=self.max_context_rows, replace=False
            )
            parents, child = parents[keep], child[keep]
        classifier = TabPFNClassifier.create_default_for_version(
            ModelVersion(self.model_version),
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            device=self.device,
            random_state=self.seed,
            # Encode the context once and keep it: a column is queried many
            # times (per fairness mechanism, per role split), always against
            # the same context.
            fit_mode="fit_with_cache",
        )
        classifier.fit(np.asarray(parents, dtype=np.float32), np.asarray(child))
        self.checkpoint = Path(str(classifier.model_path))
        return _FittedTabPFN(classifier, n_classes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DECAFTabPFNMethod(_BaseDECAFVariant):
    """DECAF's causal structure with TabPFN as every attribute's conditional."""

    name = "decaf_tabpfn"
    is_private = False

    #: The whole configuration; see the module docstring. Per-dataset entries
    #: in `settings` and constructor / `--set` overrides are layered on top.
    defaults: Dict[str, Any] = {
        "model_version": "v3.5",
        "n_estimators": 8,
        "softmax_temperature": 1.0,
        "table_cap": 100_000,
        "max_context_rows": None,
        "cut_strength": 1.0,
        "device": "cpu",
    }
    settings: Dict[str, Dict[str, Any]] = {}

    def _settings_for(self, dataset_name: Optional[str]) -> Dict[str, Any]:
        return {**self.defaults, **super()._settings_for(dataset_name)}

    @staticmethod
    def _require_backend() -> None:
        if importlib.util.find_spec("tabpfn") is None:
            raise ImportError(
                "decaf_tabpfn needs the `tabpfn` package (>= 9.1, torch >= 2.5), "
                "which is installed in the `sdg_tabpfn` conda environment and "
                "not in `sdg`. Run this method with that environment's python."
            )

    def _cached(self, key, build):
        if key not in self._cache:
            for stale in self._cache.values():
                stale.release()
        return super()._cached(key, build)

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
        self._require_backend()
        columns, col_index, dag, dag_seed = self._dag(data, dataset_name)
        cfg = self._trainer_settings(dataset_name)
        cards = np.array([domain[c] for c in columns], dtype=np.int64)
        # Same convention as `_decaf_ctgan.ColumnBlocks.encode`: Adult's
        # `education-num` runs 1..16 against a declared domain of 16.
        codes = np.clip(data.to_numpy(dtype=np.int64), 0, cards - 1)

        # The cut acts at generation time, so the fit does not depend on it.
        fit_cfg = {k: v for k, v in cfg.items() if k != "cut_strength"}
        key = (
            "tabpfn",
            hashlib.sha1(np.ascontiguousarray(codes).tobytes()).hexdigest(),
            tuple(tuple(e) for e in dag_seed),
            seed,
            tuple(sorted(fit_cfg.items(), key=lambda item: item[0])),
        )

        def build():
            estimator = TabPFNConditional(
                model_version=cfg["model_version"],
                n_estimators=cfg["n_estimators"],
                softmax_temperature=cfg["softmax_temperature"],
                device=cfg["device"],
                seed=seed or 0,
                max_context_rows=cfg["max_context_rows"],
            )
            sampler = InContextCausalSampler(
                cards.tolist(), dag_seed, estimator,
                table_cap=cfg["table_cap"],
            )
            sampler.fit(codes)
            checkpoint = getattr(estimator, "checkpoint", None)
            context_rows = min(len(codes), cfg["max_context_rows"] or len(codes))
            sampler.provenance = self._provenance(checkpoint, context_rows)
            return sampler

        sampler = self._cached(key, build)

        biased_edges = self._biased_edges(fairness_mechanism, dag, roles, col_index)
        rng = np.random.default_rng(seed)
        out = sampler.generate(
            n_synth, biased_edges=biased_edges, rng=rng,
            cut_strength=float(cfg["cut_strength"]),
        )

        synth = pd.DataFrame(out, columns=columns)
        for col in columns:
            synth[col] = synth[col].clip(0, domain[col] - 1).astype(int)
        return SDGResult(
            synthetic_data=synth,
            graph_edges=list(dag.edges),
            # Counters are cumulative over the generations sharing this fit.
            extra={**sampler.provenance, **sampler.stats()},
        )

    @staticmethod
    def _provenance(checkpoint: Optional[Path], context_rows: int) -> Dict[str, Any]:
        """Exactly which model produced the rows -- there is no training run
        to point at, so the checkpoint file is the whole answer."""
        import tabpfn
        import torch

        out: Dict[str, Any] = {
            "tabpfn_version": tabpfn.__version__,
            "torch_version": torch.__version__,
            "tabpfn_context_rows": context_rows,
        }
        if checkpoint is not None and checkpoint.exists():
            out["tabpfn_checkpoint"] = checkpoint.name
            out["tabpfn_checkpoint_sha256"] = _sha256(checkpoint)
        return out
