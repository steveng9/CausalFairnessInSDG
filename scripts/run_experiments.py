#!/usr/bin/env python
"""Full experiment grid: {MST, PrivBayes, PrivSyn} x {none, FTU, DP, CF} x
{eps 1, 10, 1000} x {3 protected/admissible role configs per dataset} x
{adult, compas}, plus DECAF (non-private, epsilon-free) over the same
datasets/role-configs/mechanisms.

Design notes:

  - `none` is included alongside the three fairness mechanisms as the control.
    Without it there's nothing to measure FTU/DP/CF's fairness gain or utility
    cost *against*.
  - DECAF is run once per (dataset, role_config, mechanism) rather than once
    per epsilon: it has no DP mechanism at all, so sweeping epsilon would just
    burn compute re-running an identical configuration. Its rows carry NULL
    epsilon in the database.
  - Every run scores fairness against *both* DECAF Distributional-Fairness
    reference distributions (real holdout and the synthetic data's own), from
    a single generation -- see `runner.EVAL_REFERENCE_BOTH`.
  - Resumable: a config that already has a done/partial row for this batch tag
    is skipped, so the script can be re-launched after an interruption.
  - The results report is rewritten after every run, so partial results are
    readable at any point while the grid is still going.

Usage: python scripts/run_experiments.py [--batch TAG] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from causal_fairness_sdg.experiments import db  # noqa: E402
from causal_fairness_sdg.experiments.runner import (  # noqa: E402
    EVAL_REFERENCE_BOTH,
    SDG_METHODS,
    RunConfig,
    run_single,
)
from report_results import write_report  # noqa: E402

DEFAULT_BATCH = "overnight-2026-07-30"

PRIVATE_METHODS = ["mst", "privbayes", "privsyn"]
MECHANISMS = ["none", "ftu", "dp", "cf"]
EPSILONS = [1.0, 10.0, 1000.0]
SEEDS = [0]

# DECAF trains one GAN per (dataset, seed) thanks to `DECAFMethod`'s train
# cache -- fairness is applied at generation time only -- so epochs are cheap
# here relative to the rest of the grid. Sized per dataset by gradient steps,
# not epochs: COMPAS has ~5x fewer batches per epoch than Adult, and the
# earlier smoke test showed 1-epoch COMPAS training collapsing the outcome
# column entirely.
DECAF_EPOCHS = {"adult": 50, "compas": 200}

# Protected/admissible role splits. The first entry for each dataset is the
# PreFair Table 1 split (comparable to published numbers); the others vary
# *which* attributes are protected and *how much* is admissible, since the
# admissible set is exactly what CF is allowed to use to block a path -- a
# narrow admissible set should make CF behave much more like DP.
RoleConfig = Tuple[str, Sequence[str], Sequence[str]]

ROLE_CONFIGS: Dict[str, List[RoleConfig]] = {
    "adult": [
        (
            "prefair",
            ["sex", "race", "native-country"],
            ["workclass", "education", "occupation",
             "capital-gain", "capital-loss", "hours-per-week"],
        ),
        # Single binary protected attribute + a broad "merit" admissible set:
        # the classic Adult fairness setup, and closest to DECAF's own.
        (
            "sex_only_broad_adm",
            ["sex"],
            ["education", "education-num", "occupation", "workclass", "hours-per-week"],
        ),
        # Same protected attributes as `prefair` minus native-country, but a
        # deliberately narrow admissible set: only hours worked and years of
        # education are considered legitimate mediators.
        (
            "sex_race_narrow_adm",
            ["sex", "race"],
            ["education-num", "hours-per-week"],
        ),
    ],
    "compas": [
        ("prefair", ["sex", "race"], ["priors_count", "c_charge_degree"]),
        # Race alone, with the full criminal-history record admissible -- the
        # most permissive reading of "what may legitimately drive recidivism".
        (
            "race_only_broad_adm",
            ["race"],
            ["priors_count", "c_charge_degree", "juv_fel_count", "juv_misd_count"],
        ),
        # Age added as a protected attribute and priors_count *removed* from
        # the admissible set -- priors_count is itself the most race-correlated
        # attribute in COMPAS, so treating it as non-admissible is the strict
        # reading, and should cost the most utility.
        ("sex_race_age_narrow_adm", ["sex", "race", "age_cat"], ["c_charge_degree"]),
    ],
}


def build_configs(batch: str) -> List[RunConfig]:
    """Ordered so that (a) COMPAS -- the fast dataset -- finishes completely
    first, giving a full readable result set early, and (b) all of a dataset's
    DECAF cells are contiguous, so DECAF's train cache is hit instead of
    retraining the same GAN 12 times."""
    configs: List[RunConfig] = []
    for dataset in ("compas", "adult"):
        for role_name, protected, admissible in ROLE_CONFIGS[dataset]:
            common = dict(
                dataset=dataset,
                role_config=role_name,
                protected=protected,
                admissible=admissible,
                eval_reference=EVAL_REFERENCE_BOTH,
                extra_params={"batch": batch},
            )
            for epsilon in EPSILONS:
                for method in PRIVATE_METHODS:
                    for mechanism in MECHANISMS:
                        for seed in SEEDS:
                            configs.append(
                                RunConfig(
                                    sdg_method=method,
                                    fairness_mechanism=mechanism,
                                    epsilon=epsilon,
                                    seed=seed,
                                    **common,
                                )
                            )
        if "decaf" in SDG_METHODS:
            for role_name, protected, admissible in ROLE_CONFIGS[dataset]:
                for mechanism in MECHANISMS:
                    for seed in SEEDS:
                        configs.append(
                            RunConfig(
                                dataset=dataset,
                                role_config=role_name,
                                protected=protected,
                                admissible=admissible,
                                eval_reference=EVAL_REFERENCE_BOTH,
                                extra_params={"batch": batch},
                                sdg_method="decaf",
                                fairness_mechanism=mechanism,
                                # Placeholder only: `run_single` writes NULL
                                # epsilon/delta for non-private methods.
                                epsilon=float("nan"),
                                seed=seed,
                            )
                        )
    return configs


def _completed_keys(conn, batch: str) -> set:
    """Identity of every already-finished run in this batch, so a relaunch
    resumes instead of duplicating work."""
    rows = conn.execute(
        "SELECT dataset, sdg_method, fairness_mechanism, epsilon, seed, extra_params "
        "FROM runs WHERE status IN ('done', 'partial')"
    ).fetchall()
    keys = set()
    for dataset, method, mechanism, epsilon, seed, extra in rows:
        try:
            params = json.loads(extra) if extra else {}
        except (TypeError, ValueError):
            params = {}
        if params.get("batch") != batch:
            continue
        keys.add(
            (dataset, method, mechanism, params.get("role_config"),
             None if epsilon is None else round(epsilon, 6), seed)
        )
    return keys


def _key(config: RunConfig) -> tuple:
    epsilon = None if config.sdg_method == "decaf" else round(config.epsilon, 6)
    return (
        config.dataset, config.sdg_method, config.fairness_mechanism,
        config.role_config, epsilon, config.seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--db-path", default=str(db.DEFAULT_DB_PATH))
    parser.add_argument(
        "--only", default=None,
        help="substring filter on '<dataset>/<role_config>/<method>/<mechanism>' "
             "-- for smoke-testing a slice of the grid",
    )
    args = parser.parse_args()

    configs = build_configs(args.batch)
    if args.only:
        configs = [
            c for c in configs
            if args.only in f"{c.dataset}/{c.role_config}/{c.sdg_method}/{c.fairness_mechanism}"
        ]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"{len(configs)} configs in batch {args.batch!r}")
        for c in configs:
            eps = "n/a" if c.sdg_method == "decaf" else c.epsilon
            print(f"  {c.dataset:7s} {c.role_config:24s} {c.sdg_method:10s} "
                  f"{c.fairness_mechanism:5s} eps={eps}")
        return 0

    conn = db.get_connection(args.db_path)
    done = _completed_keys(conn, args.batch)
    todo = [c for c in configs if _key(c) not in done]

    started = time.time()
    print(f"[{datetime.now(timezone.utc).isoformat()}] batch={args.batch} "
          f"pid={os.getpid()} total={len(configs)} already_done={len(done)} "
          f"todo={len(todo)}", flush=True)

    n_ok = n_failed = 0
    for i, config in enumerate(todo, start=1):
        if config.sdg_method == "decaf":
            SDG_METHODS["decaf"].max_epochs = DECAF_EPOCHS[config.dataset]
        eps = "n/a" if config.sdg_method == "decaf" else f"{config.epsilon:g}"
        label = (f"{config.dataset}/{config.role_config}/{config.sdg_method}/"
                 f"{config.fairness_mechanism}/eps={eps}")
        t0 = time.time()
        try:
            run_single(conn, config)
            n_ok += 1
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - logged to the DB, keep sweeping
            n_failed += 1
            status = f"FAILED: {exc}"
        elapsed = time.time() - started
        rate = elapsed / i
        print(f"[{i}/{len(todo)}] {label} -> {status} "
              f"({time.time() - t0:.0f}s; elapsed {elapsed / 60:.0f}m; "
              f"eta {rate * (len(todo) - i) / 60:.0f}m)", flush=True)
        try:
            write_report(conn, args.batch, results_dir)
        except Exception as exc:  # noqa: BLE001 - never let reporting kill the sweep
            print(f"  [report] skipped: {exc}", flush=True)

    print(f"\nFinished batch {args.batch}: {n_ok} ok, {n_failed} failed, "
          f"{(time.time() - started) / 60:.0f} minutes total.", flush=True)
    write_report(conn, args.batch, results_dir)
    print(f"Report written to {results_dir}", flush=True)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
