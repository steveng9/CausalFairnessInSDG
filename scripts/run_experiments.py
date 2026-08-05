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

#: Default grid: the three DP graphical synthesizers plus the published DECAF
#: baseline. The DECAF GAN-backbone variants (`decaf_dpgan`, `decaf_ctgan`,
#: `decaf_dpctgan`) are deliberately *not* here -- select them explicitly with
#: `--methods`, so re-running or resuming an existing batch never silently
#: grows the grid underneath it.
DEFAULT_METHODS = ["mst", "privbayes", "privsyn", "decaf"]
GAN_BACKBONE_METHODS = ["decaf_dpgan", "decaf_ctgan", "decaf_dpctgan"]

MECHANISMS = ["none", "ftu", "dp", "cf"]
EPSILONS = [1.0, 10.0, 1000.0]
SEEDS = [0]

# DECAF trains one GAN per (dataset, seed) thanks to `DECAFMethod`'s train
# cache -- fairness is applied at generation time only -- so epochs are cheap
# here relative to the rest of the grid.
#
# Both knobs are per dataset because measurement said they had to be (1-way
# TVD / accuracy, from scripts run against the real data):
#
#   COMPAS  every column has a narrow 0-5 code range, so DECAF as shipped
#           (unbounded linear heads, raw codes) works well: TVD 0.048,
#           accuracy 0.659. Bounding the heads made it much worse (TVD 0.321).
#           200 epochs because COMPAS has ~5x fewer batches per epoch than
#           Adult -- at 1 epoch the outcome column collapsed entirely.
#   ADULT   code ranges span 0-1 (`income`) to 0-40 (`native-country`), and
#           with unbounded heads the wide columns dominate the loss: `income`
#           collapsed to a single class at both 10 and 50 epochs, plus 3 other
#           constant columns. Bounding the heads to [0, 1] (DECAF's own
#           `nonlin_out`) with matching min-max scaling fixes it, and unlike
#           the unbounded version it keeps improving with training:
#           10 epochs -> TVD 0.438/accuracy 0.634, 30 -> 0.395/0.695.
#
# DECAF is a baseline here, not the contribution, so this is deliberately
# "good enough and measured" rather than a full hyperparameter search --
# Adult's fidelity (TVD 0.395) remains clearly worse than the DP synthesizers'
# and is worth a dedicated tuning pass before it goes in a paper.
#   SNAKE/SBO both span wide code ranges like Adult (SNAKE's `statefips` is
#           0-50 against a binary outcome; SBO's `SECTOR` is 0-19), so they
#           take Adult's bounded-head configuration for the same reason. Epoch
#           counts are placeholders pending their own sweep -- see the
#           `epoch_sweep_all` protocol; do not read them as tuned.
# Re-tuned 2026-08-04 by the same sweep: COMPAS 200 -> 1000 (TVD2 0.330 ->
# 0.091, lift +0.064 -> +0.083, no collapse; 500 was better still on lift but
# collapsed in 1 of 3 seeds). Adult 30 -> 300 improves TVD2 0.707 -> 0.548 at
# unchanged lift -- but note Adult DECAF never beats the trivial predictor at
# any epoch count, so read those rows as "this baseline does not work here".
DECAF_SETTINGS = {
    "adult": {"max_epochs": 300, "output_activation": "sigmoid"},
    "compas": {"max_epochs": 1000, "output_activation": None},
    "snake": {"max_epochs": 30, "output_activation": "sigmoid"},
    "sbo": {"max_epochs": 30, "output_activation": "sigmoid"},
}

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
    "snake": [
        # Closest analogue to Adult's `prefair` split: all three ascribed
        # characteristics protected, the whole labour-market chain admissible.
        (
            "prefair_like",
            ["female", "wbhaom", "citistat"],
            ["gradeatn", "mocc10", "mind16", "cow1", "hoursut", "ftptstat"],
        ),
        # The classic wage-gap setup: sex alone, with education and hours as
        # the legitimate explanations.
        ("sex_only_broad_adm", ["female"], ["gradeatn", "mocc10", "hoursut", "ftptstat"]),
        # Occupation and industry removed from the admissible set. Occupational
        # segregation is itself one of the main channels of the wage gap, so
        # treating it as non-admissible is the strict reading and should make
        # CF behave much more like DP.
        ("sex_race_narrow_adm", ["female", "wbhaom"], ["gradeatn", "hoursut"]),
    ],
    "sbo": [
        # Everything the survey asks about the owner is protected; everything
        # about how the firm is set up and run is admissible.
        (
            "owner_demographics",
            ["SEX1", "RACE1", "ETH1", "VET1", "BORNUS1"],
            ["EDUC1", "AGE1", "HOURS1", "PRMINC1", "SELFEMP1",
             "SECTOR", "ESTABLISHED", "NUMOWNERS"],
        ),
        # Sex and race only, with sector admissible -- the standard "is this
        # just industry mix?" question in the business-lending literature.
        (
            "sex_race_broad_adm",
            ["SEX1", "RACE1"],
            ["SECTOR", "EDUC1", "ESTABLISHED", "HOURS1", "NUMOWNERS", "EMPLOYMENT_NOISY"],
        ),
        # Sector removed from the admissible set. On this graph SECTOR is the
        # highest-degree mediator, so denying it to CF is the sharpest
        # available test of whether CF's advantage over DP is really coming
        # from routing through admissible attributes.
        (
            "sex_race_no_sector",
            ["SEX1", "RACE1", "ETH1"],
            ["EDUC1", "HOURS1", "ESTABLISHED"],
        ),
    ],
}


def _is_gan(method: str) -> bool:
    """GAN-backed methods cache one trained model per (data, seed, epsilon)."""
    return method == "decaf" or method in GAN_BACKBONE_METHODS


def _apply_device(device: str) -> None:
    """Point every GAN-backed method at `device`.

    The two families take it differently: the DECAF variants pass `device`
    straight through to their torch trainers, while the published DECAF
    baseline goes through PyTorch Lightning and wants an accelerator plus an
    index. Both are set through the same knobs the grid already records, so
    the choice shows up in `extra_params` rather than being invisible.
    """
    for name in GAN_BACKBONE_METHODS:
        method = SDG_METHODS.get(name)
        if method is not None:
            method.overrides["device"] = device

    decaf = SDG_METHODS.get("decaf")
    if decaf is not None and device.startswith("cuda"):
        index = int(device.split(":")[1]) if ":" in device else 0
        decaf.accelerator, decaf.devices = "gpu", [index]
        DECAF_SETTINGS_DEVICE.update(accelerator="gpu", devices=[index])


#: Device knobs merged into every `decaf` row's provenance, empty on CPU.
DECAF_SETTINGS_DEVICE: Dict[str, object] = {}


def _training_settings(method: str, dataset: str) -> Dict[str, object]:
    """The training knobs that will actually be in force for this cell.

    `decaf` keeps its own per-dataset table here in the script; the backbone
    variants carry theirs on the method object (`settings` + any constructor
    `overrides`), which `_settings_for` already resolves in the same order the
    trainer will see them.
    """
    if method == "decaf":
        return {**DECAF_SETTINGS[dataset], **DECAF_SETTINGS_DEVICE}
    settings_for = getattr(SDG_METHODS.get(method), "_settings_for", None)
    return dict(settings_for(dataset)) if settings_for else {}


#: Datasets in the default grid. `snake` and `sbo` are opt-in via --datasets
#: for the same reason the GAN backbones are opt-in via --methods: resuming an
#: existing batch must never silently grow it.
DEFAULT_DATASETS = ["compas", "adult"]


def build_configs(
    batch: str,
    seeds: Sequence[int] = SEEDS,
    methods: Sequence[str] = tuple(DEFAULT_METHODS),
    datasets: Sequence[str] = tuple(DEFAULT_DATASETS),
) -> List[RunConfig]:
    """Ordered so that (a) COMPAS -- the fast dataset -- finishes completely
    first, giving a full readable result set early, and (b) every GAN-backed
    method's cells are grouped so its train cache is hit instead of retraining
    the same network for each role config and mechanism.

    Loop nesting differs by method family, and only for cache reasons:

      - **Marginal methods** (MST/PrivBayes/PrivSyn/AIM) hold no cache, so
        seeds vary innermost. Adding trials to an existing batch then keeps a
        cell's replicates adjacent, and an interrupted sweep leaves whole
        cells rather than a ragged seed frontier.
      - **GAN methods** cache one trained model keyed on (data, seed, epsilon)
        -- not on role config or mechanism, since DECAF-style fairness is
        applied at generation time only. So seed and epsilon go *outermost*
        and all 12 role x mechanism cells run underneath one training. With
        seeds innermost instead, every run would evict the cache: 60 GAN
        trainings per dataset instead of 5.
    """
    configs: List[RunConfig] = []
    marginal = [m for m in methods if not _is_gan(m)]
    gan = [m for m in methods if _is_gan(m)]

    for dataset in datasets:
        roles_for_dataset = ROLE_CONFIGS[dataset]

        for role_name, protected, admissible in roles_for_dataset:
            common = dict(
                dataset=dataset,
                role_config=role_name,
                protected=protected,
                admissible=admissible,
                eval_reference=EVAL_REFERENCE_BOTH,
                extra_params={"batch": batch},
            )
            for epsilon in EPSILONS:
                for method in marginal:
                    for mechanism in MECHANISMS:
                        for seed in seeds:
                            configs.append(
                                RunConfig(
                                    sdg_method=method,
                                    fairness_mechanism=mechanism,
                                    epsilon=epsilon,
                                    seed=seed,
                                    **common,
                                )
                            )

        for method in gan:
            if method not in SDG_METHODS:
                continue
            private = SDG_METHODS[method].is_private
            # Non-private backbones ignore epsilon entirely, so sweeping it
            # would just re-run an identical configuration three times.
            epsilons = EPSILONS if private else [float("nan")]
            # Record the *effective* training settings on every GAN row. Without
            # this a row cannot be attributed to an epoch count, and epoch count
            # is exactly what the 4.3b sweep showed to be outcome-determining --
            # the original `gan-backbones-2026-07-30` rows carry no trace of the
            # 20/30/60/100/200 epochs that produced them.
            extra = {"batch": batch, **_training_settings(method, dataset)}
            for seed in seeds:
                for epsilon in epsilons:
                    for role_name, protected, admissible in roles_for_dataset:
                        for mechanism in MECHANISMS:
                            configs.append(
                                RunConfig(
                                    dataset=dataset,
                                    role_config=role_name,
                                    protected=protected,
                                    admissible=admissible,
                                    eval_reference=EVAL_REFERENCE_BOTH,
                                    extra_params=dict(extra),
                                    sdg_method=method,
                                    fairness_mechanism=mechanism,
                                    # For non-private methods this is a
                                    # placeholder: `run_single` writes NULL
                                    # epsilon/delta for those.
                                    epsilon=epsilon,
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
    # Non-private methods are logged with NULL epsilon, so their resume key
    # must be NULL too or nothing would ever match and every relaunch would
    # redo the whole GAN grid.
    private = SDG_METHODS[config.sdg_method].is_private
    epsilon = round(config.epsilon, 6) if private else None
    return (
        config.dataset, config.sdg_method, config.fairness_mechanism,
        config.role_config, epsilon, config.seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument(
        "--seeds", default=",".join(str(s) for s in SEEDS),
        help="comma-separated trial seeds. Adding seeds to a batch that "
             "already has results is safe -- completed (cell, seed) pairs are "
             "skipped, so only the new trials run.",
    )
    parser.add_argument(
        "--methods", default=",".join(DEFAULT_METHODS),
        help="comma-separated SDG methods. The DECAF GAN-backbone variants "
             f"({', '.join(GAN_BACKBONE_METHODS)}) are opt-in so that "
             "resuming an existing batch never grows its grid.",
    )
    parser.add_argument(
        "--datasets", default=",".join(DEFAULT_DATASETS),
        help="comma-separated datasets. `snake` and `sbo` are opt-in so that "
             "resuming an existing batch never grows its grid.",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device for the GAN-backed methods, e.g. 'cuda:0'. Measured "
             "3x faster than CPU for the CTGAN backbone (2.33 vs 6.99 s/epoch "
             "on Adult) and, on a box whose CPUs are saturated by other users, "
             "also the politer choice. Marginal methods ignore it.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--db-path", default=str(db.DEFAULT_DB_PATH))
    parser.add_argument(
        "--only", default=None,
        help="substring filter on '<dataset>/<role_config>/<method>/<mechanism>' "
             "-- for smoke-testing a slice of the grid",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in SDG_METHODS]
    if unknown:
        parser.error(
            f"unknown method(s) {unknown}; available: {sorted(SDG_METHODS)}"
        )
    if args.device:
        # Set before build_configs so `_training_settings` records the device
        # on every row -- a run on GPU is not bit-identical to one on CPU (the
        # sweep found a config that survived on CPU and inverted on GPU under
        # the same seeds), so it belongs in the provenance.
        _apply_device(args.device)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown_ds = [d for d in datasets if d not in ROLE_CONFIGS]
    if unknown_ds:
        parser.error(
            f"unknown dataset(s) {unknown_ds}; available: {sorted(ROLE_CONFIGS)}"
        )
    configs = build_configs(args.batch, seeds, methods, datasets)
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
          f"pid={os.getpid()} seeds={seeds} total={len(configs)} "
          f"already_done={len(done)} todo={len(todo)}", flush=True)

    n_ok = n_failed = 0
    for i, config in enumerate(todo, start=1):
        if config.sdg_method == "decaf":
            for attr, value in DECAF_SETTINGS[config.dataset].items():
                setattr(SDG_METHODS["decaf"], attr, value)
        eps = (
            f"{config.epsilon:g}"
            if SDG_METHODS[config.sdg_method].is_private
            else "n/a"
        )
        label = (f"{config.dataset}/{config.role_config}/{config.sdg_method}/"
                 f"{config.fairness_mechanism}/eps={eps}/seed={config.seed}")
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
