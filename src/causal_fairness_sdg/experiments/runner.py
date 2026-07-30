"""Orchestrates one experiment configuration end-to-end: load a dataset, fit
a fairness-constrained SDG method, compute utility + fairness metrics against
the DF-selected reference distribution, and log everything to the experiment
database. `run_grid` sweeps the cartesian product of the axes that matter for
"many-variable tracking": dataset x sdg_method x fairness_mechanism x epsilon
x seed.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from ..data.datasets import DATASETS
from ..eval import fairness_metrics, utility
from ..fairness import AttributeRoles, EvalReference, get_fairness_mechanism
from ..sdg.aim import AIM
from ..sdg.base import SDGMethod
from ..sdg.mst import MST
from ..sdg.privbayes import PrivBayes
from ..sdg.privsyn import PrivSyn
from . import db

try:
    from ..sdg.decaf import DECAFMethod
except ImportError:
    DECAFMethod = None  # decaf extra not installed

SDG_METHODS: Dict[str, SDGMethod] = {
    "mst": MST(), "privbayes": PrivBayes(), "aim": AIM(), "privsyn": PrivSyn(),
}
if DECAFMethod is not None:
    SDG_METHODS["decaf"] = DECAFMethod()


#: `eval_reference` value meaning "compute fairness metrics against *both*
#: reference distributions in a single run". DECAF's Distributional Fairness
#: axis only changes which dataframe the trained predictor is scored on, not
#: how the synthetic data is generated -- so both sides come essentially free
#: from one generation, instead of doubling the grid. Synthetic-reference
#: metrics are stored under the same names with a `synthref__` prefix.
EVAL_REFERENCE_BOTH = "both"
SYNTH_REFERENCE_PREFIX = "synthref__"


@dataclass
class RunConfig:
    dataset: str
    sdg_method: str
    fairness_mechanism: str
    epsilon: float
    delta: float = 1e-9
    seed: int = 0
    synth_size: Optional[int] = None
    eval_reference: str = EvalReference.ORIGINAL.value
    test_size: float = 0.3
    classifier: str = "mlp"
    #: Label for the protected/admissible role split below, so runs sharing a
    #: split are groupable in `experiments.db` without re-parsing JSON columns.
    role_config: str = "default"
    #: Optional overrides of the dataset's built-in roles. Any left as None
    #: falls back to what `data/datasets.py` defines for that dataset.
    protected: Optional[Sequence[str]] = None
    admissible: Optional[Sequence[str]] = None
    outcome: Optional[Sequence[str]] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


def resolve_roles(config: RunConfig, default_roles: AttributeRoles, columns: Sequence[str]) -> AttributeRoles:
    """Apply `config`'s role overrides on top of the dataset's default split,
    validating that every named attribute actually exists in the data."""
    protected = default_roles.protected if config.protected is None else config.protected
    admissible = default_roles.admissible if config.admissible is None else config.admissible
    outcome = default_roles.outcome if config.outcome is None else config.outcome
    unknown = (set(protected) | set(admissible) | set(outcome)) - set(columns)
    if unknown:
        raise ValueError(
            f"role_config {config.role_config!r} references attributes absent from "
            f"{config.dataset!r}: {sorted(unknown)}"
        )
    return AttributeRoles.create(protected=protected, admissible=admissible, outcome=outcome)


def _binarized_protected(
    df: pd.DataFrame, protected: Sequence[str], majority_values: Dict[str, Any]
) -> List[str]:
    """Add a `_bin_<attr>` disadvantaged-vs-rest indicator per protected
    attribute, using the *real* data's modal value as "advantaged" for every
    reference distribution, so the two DF references stay comparable."""
    cols = []
    for p in protected:
        col_name = f"_bin_{p}"
        df[col_name] = (df[p] != majority_values[p]).astype(int)
        cols.append(col_name)
    return cols


def run_single(conn: sqlite3.Connection, config: RunConfig) -> int:
    """Run one configuration, logging results (or the failure) to `conn`.
    Returns the run_id either way, so failed runs stay inspectable."""
    if config.dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset {config.dataset!r}; available: {sorted(DATASETS)}"
        )
    if config.sdg_method not in SDG_METHODS:
        raise ValueError(
            f"Unknown sdg_method {config.sdg_method!r}; available: {sorted(SDG_METHODS)}"
        )

    data, domain, default_roles = DATASETS[config.dataset]()
    roles = resolve_roles(config, default_roles, list(data.columns))
    train_df, holdout_df = train_test_split(
        data, test_size=config.test_size, random_state=config.seed
    )
    outcome_attr = next(iter(roles.outcome))
    synth_size = config.synth_size or len(train_df)
    method = SDG_METHODS[config.sdg_method]

    run_id = db.insert_run(
        conn,
        dataset=config.dataset,
        sdg_method=config.sdg_method,
        fairness_mechanism=config.fairness_mechanism,
        eval_reference=config.eval_reference,
        protected_attrs=sorted(roles.protected),
        admissible_attrs=sorted(roles.admissible),
        outcome_attr=outcome_attr,
        # Non-private baselines (DECAF) get NULL epsilon so no row in the DB
        # ever claims a privacy guarantee the method doesn't provide.
        epsilon=config.epsilon if method.is_private else None,
        delta=config.delta if method.is_private else None,
        seed=config.seed,
        synth_size=synth_size,
        extra_params={"role_config": config.role_config, **config.extra_params},
    )

    start = time.time()
    try:
        mechanism = get_fairness_mechanism(config.fairness_mechanism)
        result = method.fit_generate(
            train_df,
            domain,
            roles,
            mechanism,
            epsilon=config.epsilon,
            delta=config.delta,
            n_synth=synth_size,
            seed=config.seed,
            dataset_name=config.dataset,
        )
        synth = result.synthetic_data

        # A collapsed (single-class) synthetic outcome makes every
        # classifier-based metric undefined -- but the marginal-fidelity
        # metrics are still meaningful, and knowing *which* configurations
        # collapse is itself a result. Record what we can and flag the run
        # 'partial' rather than throwing the whole cell away.
        n_outcome_classes = int(synth[outcome_attr].nunique())
        degenerate = n_outcome_classes < 2

        metrics: Dict[str, Optional[float]] = {
            "synth_outcome_n_classes": float(n_outcome_classes),
            **utility.compute_utility_metrics(
                holdout_df, synth, outcome=outcome_attr, include_downstream=not degenerate
            ),
        }

        if not degenerate:
            model, feature_cols = utility.fit_classifier(
                synth, outcome_attr, classifier=config.classifier
            )
            # Pragmatic default binarization for (possibly multi-category)
            # protected attributes: modal value = advantaged (0), everything
            # else disadvantaged (1). Taken from the real holdout and reused
            # for every reference so the DF comparison is like-for-like.
            # Override by pre-binarizing with
            # `fairness_metrics.binarize_protected` for a specific group.
            majority_values = {p: holdout_df[p].mode().iloc[0] for p in roles.protected}

            # DF axis: which distribution is the trained predictor's fairness
            # measured against? `both` scores each reference from this one
            # generation (see EVAL_REFERENCE_BOTH).
            references = {}
            if config.eval_reference in (EvalReference.ORIGINAL.value, EVAL_REFERENCE_BOTH):
                references[""] = holdout_df
            if config.eval_reference in (EvalReference.SYNTHETIC.value, EVAL_REFERENCE_BOTH):
                references[SYNTH_REFERENCE_PREFIX] = synth

            for prefix, eval_target in references.items():
                fairness_eval_df = eval_target.copy()
                fairness_eval_df["_predicted"] = model.predict(eval_target[feature_cols])
                binarized_cols = _binarized_protected(
                    fairness_eval_df, sorted(roles.protected), majority_values
                )
                raw = fairness_metrics.compute_fairness_metrics(
                    fairness_eval_df,
                    protected_attrs=binarized_cols,
                    predicted_col="_predicted",
                    label_col=outcome_attr,
                    admissible_attrs=sorted(roles.admissible),
                )
                metrics.update(
                    {f"{prefix}{k.replace('_bin_', '')}": v for k, v in raw.items()}
                )

        db.log_metrics(conn, run_id, metrics)
        db.log_edges(conn, run_id, result.graph_edges)
        db.update_run_status(
            conn,
            run_id,
            status="partial" if degenerate else "done",
            error_message=(
                f"synthetic {outcome_attr!r} collapsed to a single class -- "
                "classifier-based utility/fairness metrics skipped"
                if degenerate else None
            ),
            duration_seconds=time.time() - start,
        )
    except Exception as exc:
        db.update_run_status(
            conn, run_id, status="failed", error_message=str(exc),
            duration_seconds=time.time() - start,
        )
        raise
    return run_id


def run_grid(
    conn: sqlite3.Connection,
    datasets: Sequence[str],
    sdg_methods: Sequence[str],
    fairness_mechanisms: Sequence[str],
    epsilons: Sequence[float],
    seeds: Sequence[int] = (0,),
    **kwargs,
) -> List[int]:
    """Sweep the cartesian product of the given axes. Failures are logged
    (status='failed' in the DB) and skipped rather than aborting the sweep."""
    run_ids = []
    for dataset in datasets:
        for sdg_method in sdg_methods:
            for mechanism in fairness_mechanisms:
                for epsilon in epsilons:
                    for seed in seeds:
                        config = RunConfig(
                            dataset=dataset,
                            sdg_method=sdg_method,
                            fairness_mechanism=mechanism,
                            epsilon=epsilon,
                            seed=seed,
                            **kwargs,
                        )
                        try:
                            run_ids.append(run_single(conn, config))
                        except Exception as exc:  # noqa: BLE001 - logged, then continue the sweep
                            print(f"[run_grid] FAILED {config}: {exc}")
    return run_ids
