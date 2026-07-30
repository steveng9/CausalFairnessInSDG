#!/usr/bin/env python
"""Smoke test: {MST, PrivBayes, AIM, PrivSyn} x {none, ftu, dp, cf} x
{adult, compas} at one epsilon/seed, plus DECAF x {none, ftu, dp, cf} x
{adult, compas} (no epsilon -- DECAF has no DP mechanism) if the optional
`decaf` extra is installed. Writes rows to experiments.db and prints a
summary table. Confirms the whole pipeline works end-to-end before scaling
up to a full experiment grid.

Usage: python scripts/run_smoke_test.py
"""

import warnings

warnings.filterwarnings("ignore")

from causal_fairness_sdg.experiments import db
from causal_fairness_sdg.experiments.runner import SDG_METHODS, run_grid


def main():
    conn = db.get_connection()
    run_ids = []

    private_methods = [m for m in ["mst", "privbayes", "aim", "privsyn"] if m in SDG_METHODS]
    run_ids += run_grid(
        conn,
        datasets=["adult", "compas"],
        sdg_methods=private_methods,
        fairness_mechanisms=["none", "ftu", "dp", "cf"],
        epsilons=[1.0],
        seeds=[0],
        synth_size=2000,
    )

    if "decaf" in SDG_METHODS:
        # DECAF ignores epsilon/delta entirely (no DP mechanism); epsilon=1.0
        # here is a placeholder so it still gets a row in `runs`, not a real
        # privacy parameter. max_epochs=1 keeps this smoke test's GAN
        # training fast -- it's meant to confirm the pipeline runs
        # end-to-end, not to produce a well-trained DECAF model (use
        # sdg/decaf.py's default max_epochs=10 for real experiment runs).
        SDG_METHODS["decaf"].max_epochs = 1
        run_ids += run_grid(
            conn,
            datasets=["adult", "compas"],
            sdg_methods=["decaf"],
            fairness_mechanisms=["none", "ftu", "dp", "cf"],
            epsilons=[1.0],
            seeds=[0],
            synth_size=2000,
        )
    else:
        print("[run_smoke_test] 'decaf' extra not installed -- skipping DECAF runs.")

    print(f"\nCompleted {len(run_ids)} runs -> {db.DEFAULT_DB_PATH}\n")

    summary = db.query_runs(conn)
    cols = [
        "run_id", "dataset", "sdg_method", "fairness_mechanism", "status",
        "tvd_1way", "downstream_accuracy_mlp",
    ]
    dp_cols = [c for c in summary.columns if c.startswith("dp_gap__")]
    print(summary[cols + dp_cols].to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
