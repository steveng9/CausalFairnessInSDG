#!/usr/bin/env python
"""Gather every FAccT result table into `report/data/facct/`, the one place
the paper's figures and tables read from.

`sweeps/<batch>/` stays the source of record (per-shard SQLite files, every
synthetic table, logs); those are large, change while a sweep runs, and are
not in git. This script folds them into flat CSVs that are:

    <batch>_runs.csv          one row per run of a `facct-*` sweep
                              (`merge_sweep_dbs.py`: restarts de-duplicated)
    real_baselines.csv        train-on-real reference, every dataset
    tabpfn_conditionals.csv   held-out log-loss of TabPFN's conditionals
    fairpfn/*.csv             the FairPFN downstream baseline
    MANIFEST.json             when this ran, from which commit, and how
                              complete each sweep was at that moment

Safe to run while sweeps are running (databases are opened read-only); run it
again when they finish. `report/data/README.md` says what each file is for.

Usage: python scripts/collect_results.py
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from merge_sweep_dbs import dedupe, load_shard  # noqa: E402

SWEEPS = REPO_ROOT / "sweeps"
OUT = REPO_ROOT / "report" / "data" / "facct"
MAIN_BATCH = "facct-2026-10-09"
FAIRPFN_FILES = ("results.csv", "summary_by_cell.csv", "generator_comparison.csv")


def _merge(batch_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((batch_dir / "db").glob("*.db")):
        try:
            frames.append(load_shard(path))
        except (sqlite3.Error, pd.errors.DatabaseError) as exc:
            print(f"  skipped {path.name}: {exc}")
    frames = [f for f in frames if not f.empty]
    return dedupe(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True,
        ).stdout.strip(),
        "sweeps": {},
    }
    for batch_dir in sorted(SWEEPS.glob("facct-*")):
        if not (batch_dir / "db").is_dir():
            continue
        print(batch_dir.name)
        runs = _merge(batch_dir)
        spec = json.loads((batch_dir / "sweep.json").read_text())
        entry = {
            "datasets": spec["datasets"], "methods": spec["methods"],
            "seeds": spec["seeds"], "epsilons": spec["epsilons"],
            "settings": spec.get("settings", {}), "only": spec.get("only"),
            "rows": int(len(runs)),
            "status": runs["status"].value_counts().to_dict() if len(runs) else {},
        }
        if len(runs):
            runs.to_csv(OUT / f"{batch_dir.name}_runs.csv", index=False)
            entry["runs_per_dataset_method"] = {
                f"{d}/{m}": int(n)
                for (d, m), n in runs.groupby(["dataset", "sdg_method"]).size().items()
            }
        manifest["sweeps"][batch_dir.name] = entry

    main_dir = SWEEPS / MAIN_BATCH
    baselines = [pd.read_csv(p) for p in sorted(main_dir.glob("real_baselines*.csv"))]
    if baselines:
        pd.concat(baselines, ignore_index=True).drop_duplicates(
            ["dataset", "seed"], keep="last"
        ).to_csv(OUT / "real_baselines.csv", index=False)
    if (main_dir / "tabpfn_conditionals.csv").exists():
        shutil.copyfile(main_dir / "tabpfn_conditionals.csv", OUT / "tabpfn_conditionals.csv")
    (OUT / "fairpfn").mkdir(exist_ok=True)
    for name in FAIRPFN_FILES:
        source = main_dir / "fairpfn_baseline" / name
        if source.exists():
            shutil.copyfile(source, OUT / "fairpfn" / name)

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
