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
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from sklearn.model_selection import train_test_split

from ..data.datasets import DATASETS
from ..eval import fairness_metrics, utility
from ..fairness import EvalReference, get_fairness_mechanism
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

    data, domain, roles = DATASETS[config.dataset]()
    train_df, holdout_df = train_test_split(
        data, test_size=config.test_size, random_state=config.seed
    )
    outcome_attr = next(iter(roles.outcome))
    synth_size = config.synth_size or len(train_df)

    run_id = db.insert_run(
        conn,
        dataset=config.dataset,
        sdg_method=config.sdg_method,
        fairness_mechanism=config.fairness_mechanism,
        eval_reference=config.eval_reference,
        protected_attrs=sorted(roles.protected),
        admissible_attrs=sorted(roles.admissible),
        outcome_attr=outcome_attr,
        epsilon=config.epsilon,
        delta=config.delta,
        seed=config.seed,
        synth_size=synth_size,
    )

    start = time.time()
    try:
        mechanism = get_fairness_mechanism(config.fairness_mechanism)
        method = SDG_METHODS[config.sdg_method]
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

        # DF axis: which distribution do we evaluate fairness/utility against?
        eval_target = (
            holdout_df
            if config.eval_reference == EvalReference.ORIGINAL.value
            else synth
        )

        u_metrics = utility.compute_utility_metrics(holdout_df, synth, outcome=outcome_attr)

        model, feature_cols = utility.fit_classifier(
            synth, outcome_attr, classifier=config.classifier
        )
        fairness_eval_df = eval_target.copy()
        fairness_eval_df["_predicted"] = model.predict(eval_target[feature_cols])

        # Pragmatic default binarization for (possibly multi-category)
        # protected attributes: modal value = advantaged (0), everything else
        # disadvantaged (1). Override by pre-binarizing with
        # `fairness_metrics.binarize_protected` for a specific chosen group.
        binarized_cols = []
        for p in roles.protected:
            majority_value = fairness_eval_df[p].mode().iloc[0]
            col_name = f"_bin_{p}"
            fairness_eval_df[col_name] = (fairness_eval_df[p] != majority_value).astype(int)
            binarized_cols.append(col_name)

        f_metrics_raw = fairness_metrics.compute_fairness_metrics(
            fairness_eval_df,
            protected_attrs=binarized_cols,
            predicted_col="_predicted",
            label_col=outcome_attr,
            admissible_attrs=sorted(roles.admissible),
        )
        f_metrics = {k.replace("_bin_", ""): v for k, v in f_metrics_raw.items()}

        db.log_metrics(conn, run_id, {**u_metrics, **f_metrics})
        db.log_edges(conn, run_id, result.graph_edges)
        db.update_run_status(conn, run_id, status="done", duration_seconds=time.time() - start)
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
