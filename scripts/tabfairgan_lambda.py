#!/usr/bin/env python
"""Pick TabFairGAN's lambda per dataset x role split, as its publisher would.

TabFairGAN's parity loss is signed and unbounded, so at the paper's lambda
(0.5; COMPAS 2.2) it reverses the gap in its own table on most of our datasets
(notes/2026-10-09-facct-sweep.md section 21). The paper set lambda per
dataset; here the publisher does the same from a grid, looking only at its
own release: the outcome-rate gap on TabFairGAN's S (majority - rest, the
majority taken on the training rows as the wrapper does) in the released
table. No real holdout is used for the choice.

Reads the dp runs of facct-2026-10-09 (lambda 0.5, COMPAS 2.2) and of the
facct-tfg-lam* batches, and writes
(COMPAS comes only from the facct-tfg-compas* batches, re-run at 1000 epochs
because 70 left its release without signal; sweep notes section 21.2)

  report/data/facct/tabfairgan_lambda.csv          one row per run: lambda,
      S, label gap on S and on every attribute the split protects, and the
      run's real-holdout metrics (dp_gap__S, downstream_auc_rf)
  report/data/facct/tabfairgan_lambda_choice.csv   per dataset x role split:
      the chosen lambda (smallest mean |label gap on S| over seeds; the
      paper's unless the grid beats it by 1 pp or more) and the
      paper-lambda row beside it, and the batch it comes from (read by
      `baselines/facct_bench.py export --choose`)

Usage: tabfairgan_lambda.py
"""
from __future__ import annotations

import glob
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from causal_fairness_sdg.data.datasets import DATASETS  # noqa: E402

SWEEPS = REPO / "sweeps"
OUT = REPO / "report" / "data" / "facct"
MAIN = "facct-2026-10-09"
COMPAS_REDO = "facct-tfg-compas"  # batch prefix; replaces COMPAS in every other batch
TEST_SIZE = 0.3  # the runner's holdout share; the training rows give S's majority
TIE = 0.01  # 1 pp of label gap


def runs_of(batch: str) -> pd.DataFrame:
    rows = []
    for db in sorted((SWEEPS / batch / "db").glob("*tabfairgan*.db")):
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            runs = pd.read_sql_query(
                "SELECT run_id, dataset, seed, protected_attrs, outcome_attr, extra_params "
                "FROM runs WHERE status IN ('done', 'partial') AND sdg_method = 'tabfairgan' "
                "AND fairness_mechanism = 'dp'", conn)
            metrics = pd.read_sql_query(
                "SELECT run_id, metric_name, metric_value FROM metrics WHERE "
                "metric_name = 'downstream_auc_rf' OR metric_name LIKE 'dp\\_gap\\_\\_%' ESCAPE '\\'",
                conn)
        if runs.empty:
            continue
        wide = metrics.pivot_table(index="run_id", columns="metric_name", values="metric_value")
        runs = runs.join(wide, on="run_id")
        runs["batch"], runs["shard"] = batch, db.stem
        rows.append(runs)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    batches = [MAIN] + sorted(Path(p).name for p in glob.glob(str(SWEEPS / "facct-tfg-lam*")))
    batches += sorted(Path(p).name for p in glob.glob(str(SWEEPS / f"{COMPAS_REDO}*")))
    runs = pd.concat([runs_of(b) for b in batches], ignore_index=True)
    runs = runs[(runs.dataset != "compas") | runs.batch.str.startswith(COMPAS_REDO)]
    extra = runs.extra_params.map(lambda e: json.loads(e or "{}"))
    runs["role_config"] = extra.map(lambda e: e.get("role_config"))
    runs["S"] = extra.map(lambda e: e.get("tabfairgan_s"))
    runs["lamda"] = extra.map(lambda e: e.get("lamda"))
    out, data_key, train = [], None, None
    for run in runs.sort_values(["dataset", "seed"]).to_dict("records"):
        if (run["dataset"], run["seed"]) != data_key:
            data, _domain, _roles = DATASETS[run["dataset"]]()
            train, _holdout = train_test_split(data, test_size=TEST_SIZE, random_state=int(run["seed"]))
            data_key = (run["dataset"], run["seed"])
        path = SWEEPS / run["batch"] / "synth" / run["shard"] / f"run_{run['run_id']}.parquet"
        if not path.exists():
            continue
        table = pd.read_parquet(path)
        y = table[run["outcome_attr"]].to_numpy() == 1
        row = {k: run[k] for k in
               ("batch", "shard", "run_id", "dataset", "seed", "role_config", "S", "lamda")}
        row["synth_pos_rate"] = float(y.mean())
        for p in json.loads(run["protected_attrs"]):
            major = table[p].to_numpy() == train[p].mode().iloc[0]
            row[f"label_gap__{p}"] = (float(y[major].mean() - y[~major].mean())
                                      if major.any() and (~major).any() else np.nan)
        row["label_gap_S"] = row.get(f"label_gap__{run['S']}", np.nan)
        row["real_dp_gap_S"] = run.get(f"dp_gap__{run['S']}", np.nan)
        row["downstream_auc_rf"] = run.get("downstream_auc_rf", np.nan)
        out.append(row)
    frame = pd.DataFrame(out)
    frame.to_csv(OUT / "tabfairgan_lambda.csv", index=False)

    frame["abs_label_gap_S"] = frame.label_gap_S.abs()
    cell = frame.groupby(["dataset", "role_config", "S", "lamda"]).agg(
        n_seeds=("seed", "nunique"), label_gap_S=("label_gap_S", "mean"),
        abs_label_gap_S=("abs_label_gap_S", "mean"), real_dp_gap_S=("real_dp_gap_S", "mean"),
        auc=("downstream_auc_rf", "mean")).reset_index()
    best = cell.loc[cell.groupby(["dataset", "role_config"]).abs_label_gap_S.idxmin()]
    paper = cell[cell.lamda == np.where(cell.dataset == "compas", 2.2, 0.5)]
    choice = best.merge(paper, on=["dataset", "role_config", "S"], suffixes=("", "_paper"))
    # Keep the paper's lambda unless the grid beats it by at least TIE (a
    # publisher would not move off it for a seed-noise difference).
    keep = choice.abs_label_gap_S_paper - choice.abs_label_gap_S < TIE
    for c in [c for c in choice.columns if c.endswith("_paper")]:
        choice.loc[keep, c[:-len("_paper")]] = choice.loc[keep, c]
    # the batch each choice comes from, so `facct_bench.py export --choose` can read it
    batch_of = frame.groupby(["dataset", "lamda"]).batch.first()
    choice["batch"] = [batch_of[d, l] for d, l in zip(choice.dataset, choice.lamda)]
    choice["sdg_method"], choice["fairness_mechanism"] = "tabfairgan", "dp"
    choice.to_csv(OUT / "tabfairgan_lambda_choice.csv", index=False)
    print(f"{len(frame)} runs -> {OUT / 'tabfairgan_lambda.csv'}")
    print(choice[["dataset", "role_config", "S", "lamda", "label_gap_S", "real_dp_gap_S", "auc",
                  "label_gap_S_paper", "real_dp_gap_S_paper", "auc_paper"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
