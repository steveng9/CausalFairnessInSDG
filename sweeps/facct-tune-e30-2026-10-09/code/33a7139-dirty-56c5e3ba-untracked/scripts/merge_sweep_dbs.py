#!/usr/bin/env python
"""Fold the per-shard SQLite files of a sharded sweep into one flat CSV.

`run_experiments.py` writes one database per shard so that seven or more
concurrent writers are not contending for a single SQLite file. Run ids are
therefore only unique within a shard, so this rewrites them as
`<shard>:<run_id>` rather than renumbering.

Usage: python scripts/merge_sweep_dbs.py sweeps/<batch>/db -o report/data/<batch>.csv
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

RUN_COLS = (
    "run_id, created_at, dataset, sdg_method, fairness_mechanism, eval_reference, "
    "protected_attrs, admissible_attrs, outcome_attr, epsilon, delta, seed, "
    "synth_size, extra_params, status, duration_seconds"
)


def load_shard(path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    runs = pd.read_sql(f"SELECT {RUN_COLS} FROM runs", conn)
    metrics = pd.read_sql(
        "SELECT run_id, metric_name, metric_value FROM metrics", conn
    )
    conn.close()
    if runs.empty:
        return runs
    wide = metrics.pivot_table(
        index="run_id", columns="metric_name", values="metric_value"
    ).reset_index()
    out = runs.merge(wide, on="run_id", how="left")
    params = out.extra_params.apply(lambda s: json.loads(s) if s else {})
    for field in ("batch", "role_config"):
        out[field] = params.apply(lambda d, f=field: d.get(f))
    out["shard"] = path.stem
    out["run_id"] = path.stem + ":" + out.run_id.astype(str)
    return out.drop(columns=["extra_params"])


# A run is identified by its experimental cell, not by its row. Shards were
# restarted several times during the sweep (to move off the wrong GPU, and to
# re-split the remaining work), and a restart writes into a fresh database
# without seeing the rows an earlier shard already produced for the same cell.
# Concatenating would therefore double-count those cells and quietly reweight
# every mean and confidence interval that pools over runs.
CELL = [
    "dataset", "sdg_method", "fairness_mechanism", "eval_reference",
    "protected_attrs", "admissible_attrs", "epsilon", "seed", "role_config",
]


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per experimental cell, preferring a completed run and then
    the most recent one."""
    before = len(df)
    # Rank best-first (a completed run beats a partial one, which beats the
    # `pending` stub a killed shard leaves behind), then most-recent-first
    # within a rank, and keep the head of each cell.
    rank = df.status.map({"done": 0, "partial": 1}).fillna(2)
    order = df.assign(_rank=rank).sort_values(
        ["_rank", "created_at"], ascending=[True, False]
    )
    out = order.drop_duplicates(subset=CELL, keep="first").drop(columns="_rank")
    if before != len(out):
        print(f"  deduped {before - len(out)} duplicate cell(s) from restarts")
    return out.sort_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    frames = [load_shard(p) for p in sorted(args.db_dir.glob("*.db"))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit(f"no runs found under {args.db_dir}")
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = dedupe(merged)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    done = merged[merged.status.isin(["done", "partial"])]
    print(f"{len(merged)} rows -> {args.out}")
    print(f"  {len(done)} done/partial across {merged.shard.nunique()} shards")
    print(done.groupby(["dataset", "sdg_method"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
