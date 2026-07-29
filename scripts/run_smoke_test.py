#!/usr/bin/env python
"""Smoke test: MST x {none, ftu, dp, cf} x {adult, compas} at one
epsilon/seed. Writes 8 rows to experiments.db and prints a summary table.
Confirms the whole pipeline works end-to-end before scaling up to a full
experiment grid.

Usage: python scripts/run_smoke_test.py
"""

import warnings

warnings.filterwarnings("ignore")

from causal_fairness_sdg.experiments import db
from causal_fairness_sdg.experiments.runner import run_grid


def main():
    conn = db.get_connection()
    run_ids = run_grid(
        conn,
        datasets=["adult", "compas"],
        sdg_methods=["mst"],
        fairness_mechanisms=["none", "ftu", "dp", "cf"],
        epsilons=[1.0],
        seeds=[0],
        synth_size=2000,
    )
    print(f"\nCompleted {len(run_ids)} runs -> {db.DEFAULT_DB_PATH}\n")

    summary = db.query_runs(conn)
    cols = [
        "run_id", "dataset", "fairness_mechanism", "status",
        "tvd_1way", "downstream_accuracy_mlp",
    ]
    dp_cols = [c for c in summary.columns if c.startswith("dp_gap__")]
    print(summary[cols + dp_cols].to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
