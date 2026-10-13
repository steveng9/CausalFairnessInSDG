#!/usr/bin/env python
"""Seed-averaged tables for the FairPFN downstream baseline.

Reads `sweeps/<batch>/fairpfn_baseline/results.csv` (written by
`fairpfn_baseline.py run`) and the batch's shard databases, and writes beside
the results:

  summary_by_cell.csv        mean, std and seed count of every score, per
                             dataset x protected attribute x context source x
                             context size x model
  generator_comparison.csv   the sweep's own rows for the same attributes:
                             downstream RF AUC and |DP gap| per dataset x
                             generator x mechanism x role split x attribute

Gaps are averaged as absolute values, so opposite signs across seeds do not
cancel.
"""
from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CELL = ["dataset", "protected", "source", "ctx_size", "model"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="facct-2026-10-09")
    parser.add_argument("--generators", default="mst,decaf_tabpfn")
    args = parser.parse_args()
    batch = REPO_ROOT / "sweeps" / args.batch
    out = batch / "fairpfn_baseline"

    r = pd.read_csv(out / "results.csv")
    print(f"{len(r)} rows, status {r.status.value_counts().to_dict()}")
    r = r[r.status == "done"].copy()
    gap_cols = [c for c in r.columns
                if c.split("@")[0].replace("_matched", "") in
                ("dp_gap", "tprb", "tnrb", "cdp_gap", "mean_score_gap")]
    for c in gap_cols:
        r[f"abs_{c}"] = r[c].abs()
    scores = (["auc", "accuracy", "accuracy_matched", "pos_rate", "pos_rate_matched"]
              + [f"abs_{c}" for c in gap_cols])
    g = r.groupby(CELL)[scores].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["n_seeds"] = r.groupby(CELL).seed.nunique()
    g["n_ctx"] = r.groupby(CELL).n_ctx.mean()
    g.reset_index().to_csv(out / "summary_by_cell.csv", index=False)

    rows = []
    methods = args.generators.split(",")
    marks = ",".join("?" * len(methods))
    for db_path in sorted(glob.glob(str(batch / "db" / "*.db"))):
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows += conn.execute(
            "select r.dataset, r.sdg_method, r.fairness_mechanism, r.epsilon, r.seed,"
            " r.extra_params, m.metric_name, m.metric_value"
            " from runs r join metrics m using (run_id)"
            f" where r.status = 'done' and r.sdg_method in ({marks})"
            " and (m.metric_name = 'downstream_auc_rf' or m.metric_name like 'dp_gap\\_\\_%' escape '\\')",
            methods,
        ).fetchall()
        conn.close()
    s = pd.DataFrame(rows, columns="dataset method mechanism epsilon seed extra metric value".split())
    s["role"] = s.extra.map(lambda e: json.loads(e).get("role_config"))
    s["generator"] = np.where(
        s.epsilon.isna(), s.method,
        s.method + "_eps" + s.epsilon.map(lambda e: f"{e:g}" if pd.notna(e) else ""),
    )
    by = ["dataset", "generator", "mechanism", "role"]
    auc = s[s.metric == "downstream_auc_rf"].groupby(by).value.agg(["mean", "std"])
    auc.columns = ["auc_mean", "auc_std"]
    dp = s[s.metric.str.startswith("dp_gap__")].copy()
    dp["protected"] = dp.metric.str[len("dp_gap__"):]
    dp["value"] = dp.value.abs()
    dp = dp.groupby(by + ["protected"]).value.agg(["mean", "std", "count"])
    dp.columns = ["abs_dp_gap_mean", "abs_dp_gap_std", "n_seeds"]
    dp.reset_index().join(auc, on=by).to_csv(out / "generator_comparison.csv", index=False)
    print(f"wrote {out / 'summary_by_cell.csv'} and {out / 'generator_comparison.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
