#!/usr/bin/env python
"""Score saved synthetic releases under arms the sweep did not run.

Every arm reuses the sweep's own pieces: the seed's 70/30 split of the real
data, the run's protected/admissible/outcome roles from its database row, the
real holdout's modal value as each protected attribute's "advantaged" group,
and `fairness_metrics.compute_fairness_metrics`. So a row here has the same
metric names as a row of `report/data/facct/<batch>_runs.csv` (`dp_gap__X`,
`tprb__X`, ..., `synthref__dp_gap__X`) and joins it on (batch, shard, run_id).

Arms
  unaware     recipient side: the sweep's random forest, trained on an
              unmitigated release with the role split's protected attributes
              dropped (fairness through unawareness by the recipient).
  resample    publisher side, after generation (Krchova et al. 2023): score
              every row of an unmitigated release with an out-of-fold
              classifier, map each disadvantaged group's scores onto the
              quantiles of the group with the highest outcome rate (among
              groups holding at least MIN_REF_SHARE of the rows), mix with
              weight `lam`, and redraw those rows' outcomes. Groups are the
              cells of the role split's binarised protected attributes;
              cells under MIN_GROUP rows are left alone. Scores come from
              5-fold out-of-fold gradient boosting on the release itself. lam = 0 redraws from
              the unmapped scores: the redraw-only control. The edited table
              is then scored with the sweep's random forest.
  recipients  other recipient models on every release: one-hot logistic
              regression (`lr_ohe`) and gradient boosting (`hgb`).
  placebo     the label-scramble placebo on one unmitigated release per shard
              (epsilon none or 1000): replace a share `frac` of outcomes with
              draws at the table's base rate, retrain, score. All recipient
              models see the same scrambled table at each `frac`.
  placebo_split
              the same scrambled tables and models as `placebo`, scored once
              per role split of the shard (its protected and admissible
              attributes), so the placebo has conditional gaps and matches
              each edit's split exactly.
  unaware_recipients, resample_recipients
              `unaware` and `resample` with `lr_ohe` and `hgb` in place of the
              forest (the re-sampled tables are regenerated, identical to the
              saved ones, so lam 0 is there too).

Output (resumable; finished keys are skipped):
  sweeps/arms-2026-10-22/<arm>/rows/<batch>/<shard>.jsonl   one row per key
  sweeps/arms-2026-10-22/<arm>/preds/<batch>/<shard>/<key>.npy
      holdout scores (float16), for unaware and resample
  sweeps/arms-2026-10-22/resample/synth/<batch>/<shard>/run_<id>_lam<l>.parquet
  `collect` folds an arm's rows into report/data/facct/arm_<arm>.csv.

    release_arms.py run --arm unaware --batches B1,B2 --workers 10
    release_arms.py collect --arm unaware
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from causal_fairness_sdg.data.datasets import DATASETS  # noqa: E402
from causal_fairness_sdg.eval import fairness_metrics  # noqa: E402
from causal_fairness_sdg.experiments.runner import (  # noqa: E402
    SYNTH_REFERENCE_PREFIX,
    _binarized_protected,
)

SWEEPS = REPO / "sweeps"
OUT = SWEEPS / "arms-2026-10-22"
TEST_SIZE = 0.3
MIN_GROUP = 50  # rows a group needs to be re-sampled
MIN_REF_SHARE = 0.05  # share of rows the reference group needs (so noise can't pick it)
LAMS = (0.0, 0.5, 1.0)
FRACS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0)
PLACEBO_MODELS = ("rf", "lr_ohe", "hgb")
OTHER_RECIPIENTS = ("lr_ohe", "hgb")


def make_model(name: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    if name == "rf":  # the sweep's recipient (utility._CLASSIFIERS["rf"])
        return RandomForestClassifier(random_state=0)
    if name == "lr_ohe":  # every column is an integer-coded categorical
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                             LogisticRegression(max_iter=1000))
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=0)
    raise ValueError(name)


# --------------------------------------------------------------------------
# Runs and data


def read_runs(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        runs = pd.read_sql_query(
            "SELECT run_id, dataset, sdg_method, fairness_mechanism, epsilon, seed, "
            "protected_attrs, admissible_attrs, outcome_attr, extra_params "
            "FROM runs WHERE status = 'done'", conn)
    runs["role_config"] = [json.loads(e or "{}").get("role_config") for e in runs["extra_params"]]
    return runs


_SPLIT: Dict = {}


def split(dataset: str, seed: int):
    from sklearn.model_selection import train_test_split

    key = (dataset, seed)
    if key not in _SPLIT:
        _SPLIT.clear()
        data, _domain, _roles = DATASETS[dataset]()
        _SPLIT[key] = train_test_split(data, test_size=TEST_SIZE, random_state=seed)
    return _SPLIT[key]


def synth_path(batch: str, shard: str, run_id: int) -> Path:
    return SWEEPS / batch / "synth" / shard / f"run_{run_id}.parquet"


# --------------------------------------------------------------------------
# Scoring (the sweep's metrics, plus group levels for "who pays")


def fast_fairness_metrics(df: pd.DataFrame, protected_attrs: Sequence[str], predicted_col: str,
                          label_col: str, admissible_attrs: Sequence[str]) -> Dict[str, float]:
    """`fairness_metrics.compute_fairness_metrics`, vectorised with bincount
    (same definitions, same group-size weights; checked equal on real runs).
    The original loops over every admissible cell in pandas, about 75 s a
    release on ACSIncome against 3 s for the forest itself."""
    yes = df[predicted_col].to_numpy() == 1
    label = df[label_col].to_numpy()
    admissible_attrs = list(admissible_attrs)
    if admissible_attrs:
        cell = df.groupby(admissible_attrs, sort=False).ngroup().to_numpy()
        keep = cell >= 0
        n_cells = int(cell.max()) + 1
        cell_size = np.bincount(cell[keep], minlength=n_cells)

    def mean(values, mask):
        return float(values[mask].mean()) if mask.any() else np.nan

    out: Dict[str, float] = {}
    for s in protected_attrs:
        s1 = df[s].to_numpy() == 1
        for name, cname, mask, hit in (("dp_gap", "cdp_gap", np.ones(len(df), bool), yes),
                                       ("tprb", "ctprb", label == 1, yes),
                                       ("tnrb", "ctnrb", label == 0, ~yes)):
            a, b = mean(hit, mask & s1), mean(hit, mask & ~s1)
            out[f"{name}__{s}"] = a - b if not (np.isnan(a) or np.isnan(b)) else np.nan
            if not admissible_attrs:
                continue
            m1, m0 = mask & s1 & keep, mask & ~s1 & keep
            n1 = np.bincount(cell[m1], minlength=n_cells)
            n0 = np.bincount(cell[m0], minlength=n_cells)
            h1 = np.bincount(cell[m1], weights=hit[m1], minlength=n_cells)
            h0 = np.bincount(cell[m0], weights=hit[m0], minlength=n_cells)
            valid = (n1 > 0) & (n0 > 0)
            out[f"{cname}__{s}"] = (float(np.average(h1[valid] / n1[valid] - h0[valid] / n0[valid],
                                                     weights=cell_size[valid]))
                                    if valid.any() else np.nan)
    return out


def score(model, features: Sequence[str], synth: pd.DataFrame, holdout: pd.DataFrame,
          protected: Sequence[str], admissible: Sequence[str], outcome: str,
          majority: Dict) -> Dict[str, float]:
    from sklearn.metrics import f1_score, roc_auc_score

    out: Dict[str, float] = {}
    for prefix, frame in (("", holdout), (SYNTH_REFERENCE_PREFIX, synth)):
        ev = frame.copy()
        ev["_predicted"] = model.predict(frame[list(features)])
        bins = _binarized_protected(ev, sorted(protected), majority)
        raw = fast_fairness_metrics(ev, bins, "_predicted", outcome, sorted(admissible))
        out.update({f"{prefix}{k.replace('_bin_', '')}": v for k, v in raw.items()})
        yes = ev["_predicted"].to_numpy() == 1
        pos = ev[outcome].to_numpy() == 1
        out[f"{prefix}pos_rate"] = float(yes.mean())
        for p in sorted(protected):
            dis = ev[f"_bin_{p}"].to_numpy() == 1
            for tag, g in (("dis", dis), ("adv", ~dis)):
                out[f"{prefix}sel_{tag}__{p}"] = float(yes[g].mean()) if g.any() else np.nan
                out[f"{prefix}tpr_{tag}__{p}"] = float(yes[g & pos].mean()) if (g & pos).any() else np.nan
                out[f"{prefix}acc_{tag}__{p}"] = float((yes[g] == pos[g]).mean()) if g.any() else np.nan
    y_true = holdout[outcome].to_numpy()
    proba = model.predict_proba(holdout[list(features)])[:, list(model.classes_).index(1)]
    hard = model.predict(holdout[list(features)])
    out["downstream_auc"] = float(roc_auc_score(y_true, proba))
    out["downstream_accuracy"] = float((hard == y_true).mean())
    out["downstream_f1_macro"] = float(f1_score(y_true, hard, average="macro"))
    out["_proba"] = proba
    return out


def base_row(batch: str, shard: str, run) -> Dict:
    return {"batch": batch, "shard": shard, "run_id": int(run.run_id), "dataset": run.dataset,
            "sdg_method": run.sdg_method, "fairness_mechanism": run.fairness_mechanism,
            "epsilon": None if pd.isna(run.epsilon) else float(run.epsilon),
            "seed": int(run.seed), "role_config": run.role_config,
            "protected_attrs": run.protected_attrs, "admissible_attrs": run.admissible_attrs}


# --------------------------------------------------------------------------
# Arms. Each yields (key, row, proba-or-None) for one run.


def arm_unaware(batch, shard, run, synth, holdout, protected, admissible, outcome, majority,
                models=("rf",)):
    features = [c for c in synth.columns if c != outcome and c not in set(protected)]
    for name in models:
        model = make_model(name)
        model.fit(synth[features], synth[outcome])
        m = score(model, features, synth, holdout, protected, admissible, outcome, majority)
        proba = m.pop("_proba")
        key = f"{run.run_id}" if name == "rf" else f"{run.run_id}_{name}"
        yield key, {**base_row(batch, shard, run), "arm": "unaware", "recipient": name,
                    "dropped": json.dumps(sorted(protected)), **m}, proba


def oof_scores(synth: pd.DataFrame, outcome: str, seed: int) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X, y = synth.drop(columns=[outcome]), synth[outcome]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X, y,
                              cv=cv, method="predict_proba")
    return proba[:, 1]


def resample_outcome(synth, outcome, protected, majority, scores, lam, rng):
    """Krchova-style levelling up. Returns (new table, diagnostics)."""
    from scipy.stats import rankdata

    cells = np.zeros(len(synth), dtype=np.int64)
    for p in sorted(protected):
        cells = cells * 2 + (synth[p].to_numpy() != majority[p]).astype(np.int64)
    y = synth[outcome].to_numpy()
    ids, sizes = np.unique(cells, return_counts=True)
    eligible = ids[sizes >= MIN_GROUP]
    big = ids[sizes >= max(MIN_GROUP, MIN_REF_SHARE * len(cells))]
    rates = {g: float((y[cells == g] == 1).mean()) for g in big}
    ref = max(rates, key=rates.get)
    ref_scores = np.sort(scores[cells == ref])
    new = synth.copy()
    new_y = y.copy()
    redrawn = np.zeros(len(synth), dtype=bool)
    for g in eligible:
        if g == ref:
            continue
        idx = np.flatnonzero(cells == g)
        u = (rankdata(scores[idx], method="average") - 0.5) / len(idx)
        target = np.quantile(ref_scores, u)
        p_new = np.clip((1 - lam) * scores[idx] + lam * target, 0.0, 1.0)
        new_y[idx] = (rng.random(len(idx)) < p_new).astype(y.dtype)
        redrawn[idx] = True
    new[outcome] = new_y
    diag = {"ref_cell": int(ref), "n_cells": int(len(ids)), "n_eligible": int(len(eligible)),
            "share_redrawn": float(redrawn.mean()),
            "share_label_changed": float((new_y != y).mean()),
            "base_rate_before": float((y == 1).mean()), "base_rate_after": float((new_y == 1).mean())}
    return new, diag


def table_label_gaps(table, outcome, protected, majority, prefix):
    y = table[outcome].to_numpy() == 1
    out = {}
    for p in sorted(protected):
        dis = table[p].to_numpy() != majority[p]
        out[f"{prefix}label_gap__{p}"] = (float(y[dis].mean() - y[~dis].mean())
                                          if dis.any() and (~dis).any() else np.nan)
    return out


def arm_resample(batch, shard, run, synth, holdout, protected, admissible, outcome, majority,
                 models=("rf",), save=True):
    scores = oof_scores(synth, outcome, int(run.seed))
    before = table_label_gaps(synth, outcome, protected, majority, "orig_")
    for lam in LAMS:
        rng = np.random.default_rng([int(run.seed), int(run.run_id), int(lam * 100)])
        edited, diag = resample_outcome(synth, outcome, protected, majority, scores, lam, rng)
        if edited[outcome].nunique() < 2:
            continue
        if lam > 0 and save:
            d = OUT / "resample" / "synth" / batch / shard
            d.mkdir(parents=True, exist_ok=True)
            edited.to_parquet(d / f"run_{run.run_id}_lam{lam:g}.parquet", index=False)
        features = [c for c in edited.columns if c != outcome]
        for name in models:
            model = make_model(name)
            model.fit(edited[features], edited[outcome])
            m = score(model, features, edited, holdout, protected, admissible, outcome, majority)
            proba = m.pop("_proba")
            key = f"{run.run_id}_lam{lam:g}" if name == "rf" else f"{run.run_id}_lam{lam:g}_{name}"
            yield key, {
                **base_row(batch, shard, run), "arm": "resample", "recipient": name, "lam": lam,
                **diag, **before, **table_label_gaps(edited, outcome, protected, majority, "edited_"),
                **m}, proba


def arm_recipients(batch, shard, run, synth, holdout, protected, admissible, outcome, majority):
    features = [c for c in synth.columns if c != outcome]
    for name in OTHER_RECIPIENTS:
        model = make_model(name)
        model.fit(synth[features], synth[outcome])
        m = score(model, features, synth, holdout, protected, admissible, outcome, majority)
        m.pop("_proba")
        yield f"{run.run_id}_{name}", {**base_row(batch, shard, run), "arm": "recipients",
                                       "recipient": name, **m}, None


def scrambled_tables(synth, outcome, seed):
    """The placebo's tables: a share `frac` of outcomes redrawn at the base rate."""
    base_rate = float(synth[outcome].mean())
    for frac in FRACS:
        rng = np.random.default_rng([int(seed), int(round(frac * 1000))])
        scrambled = synth.copy()
        mask = rng.random(len(scrambled)) < frac
        scrambled.loc[mask, outcome] = (rng.random(int(mask.sum())) < base_rate).astype(
            scrambled[outcome].dtype)
        if scrambled[outcome].nunique() >= 2:
            yield frac, scrambled


def arm_placebo(batch, shard, run, synth, holdout, protected, admissible, outcome, majority):
    features = [c for c in synth.columns if c != outcome]
    for frac, scrambled in scrambled_tables(synth, outcome, run.seed):
        for name in PLACEBO_MODELS:
            model = make_model(name)
            model.fit(scrambled[features], scrambled[outcome])
            m = score(model, features, scrambled, holdout, protected, admissible, outcome, majority)
            m.pop("_proba")
            yield f"{run.run_id}_{frac:g}_{name}", {
                **base_row(batch, shard, run), "arm": "placebo", "frac": frac,
                "recipient": name, **m}, None


def arm_placebo_split(batch, shard, run, synth, holdout, protected, admissible, outcome,
                      majority):
    features = [c for c in synth.columns if c != outcome]
    for frac, scrambled in scrambled_tables(synth, outcome, run.seed):
        for name in PLACEBO_MODELS:
            model = make_model(name)
            model.fit(scrambled[features], scrambled[outcome])
            for role_config, split_protected, split_admissible in json.loads(run.splits):
                split_majority = {p: holdout[p].mode().iloc[0] for p in split_protected}
                m = score(model, features, scrambled, holdout, split_protected, split_admissible,
                          outcome, split_majority)
                m.pop("_proba")
                yield f"{run.run_id}_{frac:g}_{name}_{role_config}", {
                    **base_row(batch, shard, run), "role_config": role_config,
                    "protected_attrs": json.dumps(split_protected),
                    "admissible_attrs": json.dumps(split_admissible),
                    "arm": "placebo_split", "frac": frac, "recipient": name, **m}, None


ARMS = {"unaware": arm_unaware, "resample": arm_resample,
        "recipients": arm_recipients, "placebo": arm_placebo,
        "placebo_split": arm_placebo_split,
        "unaware_recipients": partial(arm_unaware, models=OTHER_RECIPIENTS),
        "resample_recipients": partial(arm_resample, models=OTHER_RECIPIENTS, save=False)}


def select_runs(arm: str, runs: pd.DataFrame) -> pd.DataFrame:
    if arm in ("unaware", "resample", "unaware_recipients", "resample_recipients"):
        return runs[runs.fairness_mechanism == "none"]
    if arm in ("placebo", "placebo_split"):
        none = runs[(runs.fairness_mechanism == "none")
                    & (runs.epsilon.isna() | (runs.epsilon == 1000))]
        return none.sort_values("run_id").head(1)
    return runs


# --------------------------------------------------------------------------
# Driver


def do_shard(arm: str, batch: str, shard: str) -> str:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    runs = read_runs(SWEEPS / batch / "db" / f"{shard}.db")
    if runs.empty:
        return f"{shard}: no finished runs"
    todo = select_runs(arm, runs)
    if arm == "placebo":
        # Score the placebo on every attribute any of the shard's splits protects.
        union = sorted({a for p in runs.protected_attrs for a in json.loads(p)})
        todo = todo.assign(protected_attrs=json.dumps(union), admissible_attrs="[]",
                           role_config="placebo")
    if arm == "placebo_split":
        splits = sorted({(r, p, a) for r, p, a in
                         zip(runs.role_config, runs.protected_attrs, runs.admissible_attrs)})
        todo = todo.assign(splits=json.dumps([(r, json.loads(p), json.loads(a))
                                              for r, p, a in splits]))
    rows_path = OUT / arm / "rows" / batch / f"{shard}.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if rows_path.exists():
        with open(rows_path) as handle:
            done = {json.loads(line)["key"] for line in handle if line.strip()}
    started, n_rows = time.time(), 0
    done_runs = {k.split("_")[0] for k in done}
    for run in todo.itertuples():
        if arm != "placebo" and str(run.run_id) in done_runs:
            continue
        if arm == "placebo" and len(done) >= len(FRACS) * len(PLACEBO_MODELS):
            continue
        path = synth_path(batch, shard, int(run.run_id))
        if not path.exists():
            continue
        synth = pd.read_parquet(path)
        outcome = run.outcome_attr
        if synth[outcome].nunique() < 2:
            continue
        _train, holdout = split(run.dataset, int(run.seed))
        protected = json.loads(run.protected_attrs)
        admissible = json.loads(run.admissible_attrs)
        majority = {p: holdout[p].mode().iloc[0] for p in protected}
        try:
            results = list(ARMS[arm](batch, shard, run, synth, holdout, protected,
                                     admissible, outcome, majority))
        except Exception as error:  # keep the shard going; record the failure
            with open(OUT / arm / "errors.log", "a") as handle:
                handle.write(f"{batch}\t{shard}\t{run.run_id}\t{type(error).__name__}: {error}\n")
            continue
        with open(rows_path, "a") as handle:
            for key, row, proba in results:
                if proba is not None:
                    d = OUT / arm / "preds" / batch / shard
                    d.mkdir(parents=True, exist_ok=True)
                    np.save(d / f"{key}.npy", proba.astype(np.float16))
                handle.write(json.dumps({"key": key, **row}, default=float) + "\n")
                n_rows += 1
    return f"{shard}: {n_rows} rows, {time.time() - started:.0f}s"


def cmd_run(args) -> None:
    jobs = []
    for batch in [b for b in args.batches.split(",") if b]:
        names = sorted(p.stem for p in (SWEEPS / batch / "db").glob("*.db"))
        if args.methods:
            keep = set(args.methods.split(","))
            names = [n for n in names if n.split("-")[1] in keep]
        jobs += [(batch, n) for n in names]
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {args.arm}: {len(jobs)} shards", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(do_shard, args.arm, b, n): (b, n) for b, n in jobs}
        for future in as_completed(futures):
            batch, name = futures[future]
            try:
                print(f"[{time.strftime('%m-%d %H:%M:%S')}] [{batch}] {future.result()}", flush=True)
            except Exception as error:
                print(f"[{batch}] {name}: FAILED {type(error).__name__}: {error}", flush=True)
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] done", flush=True)


def cmd_collect(args) -> None:
    files = sorted((OUT / args.arm / "rows").glob("*/*.jsonl"))
    rows = [json.loads(line) for f in files for line in open(f) if line.strip()]
    frame = pd.DataFrame(rows).drop_duplicates(["batch", "shard", "key"], keep="last")
    dest = REPO / "report" / "data" / "facct" / f"arm_{args.arm}.csv"
    frame.to_csv(dest, index=False)
    print(f"{len(frame)} rows from {len(files)} shards -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arm", required=True, choices=sorted(ARMS))
    run.add_argument("--batches", required=True)
    run.add_argument("--methods", default="", help="restrict to these sdg methods")
    run.add_argument("--workers", type=int, default=4)
    run.set_defaults(func=cmd_run)
    collect = sub.add_parser("collect")
    collect.add_argument("--arm", required=True, choices=sorted(ARMS))
    collect.set_defaults(func=cmd_collect)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
