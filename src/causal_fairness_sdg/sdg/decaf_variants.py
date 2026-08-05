"""DECAF with its GAN backbone swapped out.

Three `SDGMethod`s, all keeping DECAF's causal generator structure and its
generation-time fairness mechanism (`biased_edges` surrogate-value
substitution) so FTU/DP/CF mean exactly what they mean for the baseline, and
all varying only *how the generator is trained or parameterized*:

  `decaf_dpgan`    stock DECAF's causal generator, trained with DP-SGD on the
                   discriminator. **Private** -- this is the only DECAF variant
                   with a real epsilon, so it is the one that can be compared
                   against MST/PrivBayes/PrivSyn at matched privacy budgets.
  `decaf_ctgan`    CTGAN's representation and training procedure (one-hot
                   blocks, gumbel-softmax heads, conditional vectors,
                   training-by-sampling, PacGAN critic). **Not private** --
                   this variant exists to fix DECAF's fidelity, not its privacy.
  `decaf_dpctgan`  both: the CTGAN representation trained under DP-SGD.

Why the baseline in `decaf.py` is untouched: its rows in `experiments.db` are
the published-DECAF reference point, and they need to stay reproducible. These
are additive.

See `_decaf_dp.py` for the privacy argument and the three deviations DP-SGD
forces on the WGAN training loop, and `_decaf_ctgan.py` for the block-level
lift of the causal generator.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.causal_graphs import CAUSAL_GRAPHS, as_digraph
from ..fairness.base import AttributeRoles, FairnessMechanism
from .base import SDGMethod, SDGResult

try:
    from ._decaf_ctgan import ColumnBlocks, CTGANCausalTrainer
    from ._decaf_dp import DPCausalGANTrainer
except ImportError as exc:  # pragma: no cover - only without the decaf extra
    _IMPORT_ERROR: Optional[ImportError] = exc
else:
    _IMPORT_ERROR = None


class _BaseDECAFVariant(SDGMethod):
    """Shared plumbing: resolve the ground-truth DAG, translate the fairness
    mechanism's chosen edges into column indices, and cache the trained model.

    Deliberately duplicates a little of `decaf.py` rather than refactoring it,
    so the published-DECAF baseline keeps producing byte-identical results.
    """

    #: Per-dataset training settings, filled in by subclasses.
    settings: Dict[str, Dict[str, Any]] = {}

    def __init__(self, cache_trained_models: bool = True, **overrides: Any):
        self.cache_trained_models = cache_trained_models
        self.overrides = overrides
        self._cache: Dict[Tuple, Any] = {}

    def _settings_for(self, dataset_name: Optional[str]) -> Dict[str, Any]:
        base = dict(self.settings.get(dataset_name or "", {}))
        base.update(self.overrides)
        return base

    @staticmethod
    def _dag(data: pd.DataFrame, dataset_name: Optional[str]):
        if dataset_name not in CAUSAL_GRAPHS:
            raise ValueError(
                f"No causal graph registered for dataset_name={dataset_name!r}; "
                f"available: {sorted(CAUSAL_GRAPHS)}"
            )
        columns = list(data.columns)
        col_index = {c: i for i, c in enumerate(columns)}
        dag = as_digraph(CAUSAL_GRAPHS[dataset_name], columns)
        dag_seed = [[col_index[p], col_index[c]] for p, c in dag.edges]
        return columns, col_index, dag, dag_seed

    @staticmethod
    def _biased_edges(
        fairness_mechanism: FairnessMechanism, dag, roles: AttributeRoles, col_index
    ) -> Dict[int, List[int]]:
        by_name = fairness_mechanism.select_biased_edges(dag, roles)
        return {
            col_index[child]: [col_index[p] for p in parents]
            for child, parents in by_name.items()
        }

    def _cached(self, key: Tuple, build):
        """One trained model per (data, seed, hyperparameters).

        Fairness acts only at generation time, so every mechanism and every
        role config over the same data reuses one training run -- the single
        largest compute saving available in the grid.
        """
        if self.cache_trained_models and key in self._cache:
            return self._cache[key]
        model = build()
        if self.cache_trained_models:
            self._cache.clear()  # only the most recent dataset is ever reused
            self._cache[key] = model
        return model

    @staticmethod
    def _require_backend() -> None:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "DECAF variants require the optional 'decaf' extra: "
                "pip install -e '.[decaf]' (decaf-synthetic-data, torch, "
                "pytorch-lightning, opacus)."
            ) from _IMPORT_ERROR


class DECAFDPGANMethod(_BaseDECAFVariant):
    """DECAF's causal generator trained with DP-SGD (DP-GAN).

    Epoch counts are per dataset for the same reason the baseline's are: they
    were measured, not guessed. Both were tuned at epsilon=1000, where the
    injected noise is negligible, so the count reflects what the *optimizer*
    needs; the privacy cost of those steps is then paid honestly at every
    epsilon via the accountant.
    """

    name = "decaf_dpgan"
    is_private = True

    # Measured at epsilon=1000 (1-way/2-way TVD, downstream accuracy):
    #   COMPAS  100 epochs -> TVD1 0.130 / TVD2 0.206 / acc 0.625
    #           200 epochs -> TVD1 0.160 / TVD2 0.239 / acc 0.642
    #           More steps buy a little accuracy and cost more fidelity, and
    #           under DP they also cost more noise, so 100 wins on both counts.
    #
    # Epoch counts are held *fixed across epsilon* on purpose. Spending fewer
    # steps at a tight budget would produce better numbers at epsilon=1, but it
    # would confound "how much does privacy cost this architecture" with "how
    # well did we tune each cell". The honest consequence, seen in tuning: at
    # epsilon=1 on COMPAS the noise multiplier reaches ~12 and the generator
    # collapses to a single class. That is a result about DP-GANs on small
    # tables, not a configuration to paper over.
    #
    #   ADULT    30 epochs -> eps=1000 collapses to 1 class; eps=1 TVD1 0.436
    #            60 epochs -> eps=1000 TVD1 0.662 acc 0.598; eps=1 TVD1 0.467
    #           120 epochs -> collapses at both budgets
    #            Every Adult cell is bad (TVD1 0.43-0.68 against CTGAN's
    #            0.074), and note the inversion: eps=1 scores *better* than
    #            eps=1000. That is not privacy helping -- it is the weight-
    #            clipped critic being unstable, with noise acting as
    #            accidental regularization. 60 is chosen as the only setting
    #            that avoids single-class collapse at both budgets. Read the
    #            Adult DP-GAN rows as "this architecture does not work here",
    #            not as a tuned number.
    # SNAKE and SBO are registered so the grid can reach them, but their
    # epoch counts are NOT tuned -- they are Adult's, carried over on the
    # grounds that both are wide mixed-cardinality tables like Adult. Run
    # the epoch sweep before quoting any number from these two.
    settings = {
        # Re-tuned 2026-08-04. Epochs stay fixed across epsilon (see above);
        # the count is chosen at eps=1000 where noise is negligible.
        #   ADULT   60 and 300 collapse in 2 of 3 seeds at eps=1000; 120 is the
        #           only count that never collapses. Every Adult DP-GAN cell is
        #           still bad (TVD2 0.84) -- this is damage control, not tuning.
        #   COMPAS  100 keeps the best lift (+0.066, all seeds positive); 500
        #           buys TVD2 0.193 vs 0.228 but drops a seed below baseline.
        #           At eps=1 COMPAS collapses at 100/200/500 alike -- a result
        #           about DP-GANs on small tables, not a setting to fix.
        "compas": {"max_epochs": 100, "batch_size": 256},
        "adult": {"max_epochs": 120, "batch_size": 256},
        "snake": {"max_epochs": 60, "batch_size": 256},
        "sbo": {"max_epochs": 60, "batch_size": 256},
    }

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
        cfg = self._settings_for(dataset_name)
        values = data.to_numpy(dtype="float32")

        key = (
            "dpgan",
            hashlib.sha1(values.tobytes()).hexdigest(),
            tuple(tuple(e) for e in dag_seed),
            seed,
            round(float(epsilon), 6),
            round(float(delta), 12),
            tuple(sorted(cfg.items())),
        )

        def build():
            trainer = DPCausalGANTrainer(
                input_dim=len(columns), dag_seed=dag_seed, **cfg
            )
            trainer.fit(values, epsilon=epsilon, delta=delta, seed=seed or 0)
            return trainer

        trainer = self._cached(key, build)

        biased_edges = self._biased_edges(fairness_mechanism, dag, roles, col_index)
        raw = trainer.generate(n_synth, biased_edges=biased_edges)

        synth = pd.DataFrame(raw, columns=columns)
        for col in columns:
            synth[col] = synth[col].round().clip(0, domain[col] - 1).astype(int)
        return SDGResult(
            synthetic_data=synth,
            graph_edges=list(dag.edges),
            extra={
                "noise_multiplier": trainer.noise_multiplier,
                "dp_steps": trainer.steps_taken,
                "spent_epsilon": getattr(trainer, "spent_epsilon", None),
            },
        )


class DECAFCTGANMethod(_BaseDECAFVariant):
    """DECAF's causal structure over CTGAN's representation and training.

    Not private -- like the published DECAF baseline, `epsilon`/`delta` are
    accepted only for interface parity and logged as NULL.
    """

    name = "decaf_ctgan"
    is_private = False

    # Measured (1-way/2-way TVD, downstream accuracy):
    #   COMPAS  20 epochs -> TVD1 0.042 / TVD2 0.078 / acc 0.551
    #           30 epochs -> TVD1 0.050 / TVD2 0.083 / acc 0.571
    #           60 epochs -> TVD1 0.091 / TVD2 0.154 / acc 0.543
    #           Fidelity degrades past ~20 epochs, so this is early-stopped
    #           rather than trained to convergence. At 20 epochs it beats the
    #           published DECAF baseline's fidelity on COMPAS (TVD2 0.084)
    #           while giving up downstream accuracy (0.551 vs 0.659) -- a real
    #           tradeoff, not a strict improvement.
    #   ADULT   30 epochs -> TVD1 0.074 / TVD2 0.172 / acc 0.756
    #           60 epochs -> TVD1 0.113 / TVD2 0.214 / acc 0.758
    #          120 epochs -> TVD1 0.126 / TVD2 0.229 / acc 0.768
    #           Same shape as COMPAS: fidelity degrades monotonically with
    #           training while accuracy barely moves, so 30 is the pick. This
    #           is the headline -- published DECAF on Adult is TVD2 0.658, so
    #           the CTGAN representation is a ~3.8x fidelity improvement at
    #           *higher* downstream accuracy (0.756 vs 0.695).
    # SNAKE and SBO are registered so the grid can reach them, but their
    # epoch counts are NOT tuned -- they are Adult's, carried over on the
    # grounds that both are wide mixed-cardinality tables like Adult. Run
    # the epoch sweep before quoting any number from these two.
    settings = {
        # Re-tuned 2026-08-04 by `epoch_sweep_all`, 3 seeds, ranked on 2-way TVD
        # among settings where *every* seed beats the trivial predictor. The
        # previous 20/30 were selected on single-seed 1-way TVD and both
        # produced models that lose to always-guessing-the-majority.
        #   ADULT   30 -> lift -0.125 (one seed inverts, acc 0.45)
        #           120 -> lift +0.023 | 300 -> lift -0.134 (seed 0 inverts,
        #           pos_rate 0.83 vs true 0.25) | 600 -> lift +0.046, TVD2
        #           0.171, pos_rate 0.18-0.23 across seeds. 300 is NOT stable:
        #           it survived a CPU sweep and inverted on GPU with the same
        #           seeds, i.e. it was a lucky RNG draw. 600 is where the
        #           outcome marginal actually converges.
        #   COMPAS  20 -> TVD2 0.070 but lift -0.006 (below baseline);
        #           500 -> TVD2 0.160, lift +0.085. Fidelity is genuinely
        #           worse at 500 -- we take that trade because a generator
        #           that cannot beat guessing has no usable fairness signal.
        "compas": {"max_epochs": 500, "batch_size": 500},
        "adult": {"max_epochs": 600, "batch_size": 500},
        "snake": {"max_epochs": 30, "batch_size": 500},
        "sbo": {"max_epochs": 30, "batch_size": 500},
    }

    #: Subclass hook -- the DP variant flips this.
    _dp = False

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
        cfg = self._settings_for(dataset_name)
        codes = data.to_numpy()
        blocks = ColumnBlocks(columns, domain)

        key = (
            "dpctgan" if self._dp else "ctgan",
            hashlib.sha1(np.ascontiguousarray(codes).tobytes()).hexdigest(),
            tuple(tuple(e) for e in dag_seed),
            seed,
            round(float(epsilon), 6) if self._dp else None,
            round(float(delta), 12) if self._dp else None,
            tuple(sorted(cfg.items())),
        )

        def build():
            trainer = CTGANCausalTrainer(blocks, dag_seed, **cfg)
            if self._dp:
                trainer.fit_dp(codes, epsilon=epsilon, delta=delta, seed=seed or 0)
            else:
                trainer.fit(codes, seed=seed or 0)
            return trainer

        trainer = self._cached(key, build)

        biased_edges = self._biased_edges(fairness_mechanism, dag, roles, col_index)
        rng = np.random.default_rng(seed)
        out = trainer.generate(n_synth, biased_edges=biased_edges, rng=rng)

        synth = pd.DataFrame(out, columns=columns)
        for col in columns:
            synth[col] = synth[col].clip(0, domain[col] - 1).astype(int)
        extra = {}
        if self._dp:
            extra = {
                "noise_multiplier": trainer.noise_multiplier,
                "dp_steps": trainer.steps_taken,
            }
        return SDGResult(
            synthetic_data=synth, graph_edges=list(dag.edges), extra=extra
        )


class DECAFDPCTGANMethod(DECAFCTGANMethod):
    """CTGAN representation + causal generator, trained under DP-SGD.

    PacGAN is disabled here (`pac=1`): the critic's per-sample gradient is the
    unit of privacy accounting, and packing `pac` records into one critic
    decision makes a "sample" a group of records, so the sensitivity analysis
    no longer matches a per-record neighbouring-datasets definition. Losing
    PacGAN costs some mode coverage -- an honest, documented consequence of
    the privacy requirement rather than a tuning choice.
    """

    name = "decaf_dpctgan"
    is_private = True
    _dp = True

    # SNAKE and SBO are registered so the grid can reach them, but their
    # epoch counts are NOT tuned -- they are Adult's, carried over on the
    # grounds that both are wide mixed-cardinality tables like Adult. Run
    # the epoch sweep before quoting any number from these two.
    settings = {
        # Re-tuned 2026-08-04: COMPAS 200 -> 400 (TVD2 0.164 -> 0.161 and lift
        # +0.023 -> +0.084 at eps=1000). Adult stays at 60 -- it never clears
        # the trivial baseline at any count tried (lift ~= 0.000 at 60/120/300),
        # so the cheapest is the honest choice; see Section 7.2's null result.
        "compas": {
            "max_epochs": 400, "batch_size": 256, "pac": 1, "use_conditional": False,
        },
        "adult": {
            "max_epochs": 60, "batch_size": 256, "pac": 1, "use_conditional": False,
        },
        "snake": {
            "max_epochs": 60, "batch_size": 256, "pac": 1, "use_conditional": False,
        },
        "sbo": {
            "max_epochs": 60, "batch_size": 256, "pac": 1, "use_conditional": False,
        },
    }
