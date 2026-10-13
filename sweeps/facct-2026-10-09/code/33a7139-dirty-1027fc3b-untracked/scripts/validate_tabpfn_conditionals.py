#!/usr/bin/env python
"""Held-out check of the conditionals `decaf_tabpfn` samples from.

`decaf_tabpfn` has nothing to train, so the usual question -- did the
generator fit -- becomes: is TabPFN's `p(column | DAG parents)` a good
estimate of that conditional? This scores it, column by column, by mean
log-loss on the holdout rows of the same split the experiments use, against
two references that need no model:

    marginal   the column's training frequencies, ignoring its parents
    cpt        the training conditional frequency table, shrunk towards the
               marginal (one pseudo-count per class), which is what any
               count-based generator has to work with

Lower is better; `tabpfn` should be at or below `cpt` on every column, and
well below it where the parents span many configurations. Uses the settings
`decaf_tabpfn` runs with, read from the method itself. Needs the `sdg_tabpfn`
environment.

    python scripts/validate_tabpfn_conditionals.py --datasets acsincome_ca,sbo \\
        --seed 0 --device cuda:0 --out sweeps/<batch>/tabpfn_conditionals.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from causal_fairness_sdg.data.causal_graphs import CAUSAL_GRAPHS, as_digraph
from causal_fairness_sdg.data.datasets import DATASETS
from causal_fairness_sdg.experiments.runner import SDG_METHODS
from causal_fairness_sdg.sdg.decaf_tabpfn import TabPFNConditional

#: `RunConfig.test_size`, so the holdout here is the experiments' holdout.
TEST_SIZE = 0.3


def _log_loss(probs: np.ndarray, target: np.ndarray) -> float:
    picked = probs[np.arange(len(target)), target]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def _cpt(train_x, train_y, test_x, n_classes, marginal):
    """Conditional frequencies with `n_classes` pseudo-counts spread as the marginal."""
    def keys(x):
        return pd.DataFrame(x).astype(str).agg("|".join, axis=1).to_numpy()
    table = pd.crosstab(keys(train_x), train_y).reindex(columns=range(n_classes), fill_value=0)
    counts = table.reindex(keys(test_x)).fillna(0).to_numpy()
    return (counts + n_classes * marginal) / (counts.sum(axis=1, keepdims=True) + n_classes)


def validate(dataset: str, seed: int, device: str) -> pd.DataFrame:
    data, domain, _roles = DATASETS[dataset]()
    train, holdout = train_test_split(data, test_size=TEST_SIZE, random_state=seed)
    columns = list(data.columns)
    dag = as_digraph(CAUSAL_GRAPHS[dataset], columns)
    cfg = SDG_METHODS["decaf_tabpfn"]._settings_for(dataset)
    estimator = TabPFNConditional(
        model_version=cfg["model_version"], n_estimators=cfg["n_estimators"],
        softmax_temperature=cfg["softmax_temperature"], device=device, seed=seed,
        max_context_rows=cfg["max_context_rows"],
    )
    rows = []
    for column in columns:
        parents = sorted(dag.predecessors(column), key=columns.index)
        if not parents or train[column].nunique() < 2:
            continue
        k = max(int(domain[column]), int(data[column].max()) + 1)
        train_x, train_y = train[parents].to_numpy(), train[column].to_numpy()
        test_x, test_y = holdout[parents].to_numpy(), holdout[column].to_numpy()
        marginal = (np.bincount(train_y, minlength=k) + 1) / (len(train_y) + k)
        distinct, inverse = np.unique(test_x, axis=0, return_inverse=True)
        started = time.time()
        fitted = estimator.fit(train_x, train_y, k)
        probs = fitted.predict_proba(distinct)[inverse.ravel()]
        fitted.release()
        rows.append({
            "dataset": dataset, "seed": seed, "column": column, "n_classes": k,
            "n_parents": len(parents), "holdout_parent_configs": len(distinct),
            "logloss_marginal": _log_loss(np.tile(marginal, (len(test_y), 1)), test_y),
            "logloss_cpt": _log_loss(_cpt(train_x, train_y, test_x, k, marginal), test_y),
            "logloss_tabpfn": _log_loss(probs / probs.sum(axis=1, keepdims=True), test_y),
            "seconds": round(time.time() - started, 1),
            **{key: cfg[key] for key in ("model_version", "n_estimators", "softmax_temperature")},
        })
        print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in rows[-1].items()},
              flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", required=True, help="comma-separated")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True, help="CSV, appended to if it exists")
    args = parser.parse_args()
    out = Path(args.out)
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        frame = validate(dataset, args.seed, args.device)
        frame.to_csv(out, mode="a", header=not out.exists(), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
