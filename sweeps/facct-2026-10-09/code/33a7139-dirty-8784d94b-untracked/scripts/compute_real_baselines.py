#!/usr/bin/env python
"""Train-on-real / test-on-real reference for every downstream score the
sweep records, plus the real-vs-real nearest-neighbour distance the privacy
ratio is calibrated against.

Every synthetic-data number in the paper is only interpretable against these:
an AUC of 0.75 on Adult means one thing if a real-data classifier reaches
0.90 and another if it reaches 0.76. Writes `report/data/real_baselines.csv`.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[1]

import pandas as pd
from sklearn.model_selection import train_test_split

from causal_fairness_sdg.data.datasets import DATASETS
from causal_fairness_sdg.eval import utility

SEEDS = [0, 1, 2, 3, 4]


def main() -> int:
    rows = []
    for name in ["adult", "compas"]:
        data, _domain, roles = DATASETS[name]()
        outcome = next(iter(roles.outcome))
        for seed in SEEDS:
            train_df, holdout_df = train_test_split(
                data, test_size=0.3, random_state=seed
            )
            row = {
                "dataset": name, "seed": seed, "outcome": outcome,
                "n_train": len(train_df), "n_holdout": len(holdout_df),
                "base_rate_pos": float((holdout_df[outcome] == 1).mean()),
                "majority_baseline": float(holdout_df[outcome].value_counts(normalize=True).max()),
            }
            for clf in ["mlp", "rf", "lr"]:
                for metric, value in utility.downstream_scores(
                    train_df, holdout_df, outcome, classifier=clf
                ).items():
                    row[f"trtr_{metric}_{clf}"] = value
            # The privacy floor: a *real* sample drawn from the training half
            # scores this, so a release at or above it is no more identifying
            # than a real record would be.
            ref = train_df.sample(n=min(len(train_df), 4000), random_state=seed)
            row.update({
                f"real_{k}": v for k, v in
                utility.nearest_neighbour_privacy(train_df, holdout_df, ref, seed=seed).items()
            })
            rows.append(row)
            print(f"{name} seed={seed} done", flush=True)

    out = REPO_ROOT / "report" / "data" / "real_baselines.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
