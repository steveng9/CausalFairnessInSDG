"""SQLite-backed experiment tracking.

One `runs` row per (dataset, sdg_method, fairness_mechanism, eval_reference,
epsilon, seed, ...) configuration. Metrics live in a separate EAV-style table
keyed by (run_id, metric_name) so adding a new metric later never requires a
schema migration -- the "robust, many-variable tracking" goal as the number
of fairness mechanisms, SDG methods, and metrics all grow independently.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "experiments.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  git_commit TEXT,
  dataset TEXT NOT NULL,
  sdg_method TEXT NOT NULL,
  fairness_mechanism TEXT NOT NULL,
  eval_reference TEXT NOT NULL DEFAULT 'original',
  protected_attrs TEXT NOT NULL,
  admissible_attrs TEXT,
  outcome_attr TEXT NOT NULL,
  epsilon REAL,
  delta REAL,
  seed INTEGER NOT NULL,
  synth_size INTEGER,
  extra_params TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  error_message TEXT,
  duration_seconds REAL
);

CREATE TABLE IF NOT EXISTS metrics (
  metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES runs(run_id),
  metric_name TEXT NOT NULL,
  metric_value REAL,
  extra TEXT,
  UNIQUE(run_id, metric_name)
);

CREATE TABLE IF NOT EXISTS graph_edges (
  edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES runs(run_id),
  node_a TEXT NOT NULL,
  node_b TEXT NOT NULL,
  weight REAL
);
"""


def get_connection(db_path: "str | Path" = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
            timeout=2,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def insert_run(
    conn: sqlite3.Connection,
    dataset: str,
    sdg_method: str,
    fairness_mechanism: str,
    protected_attrs: Sequence[str],
    outcome_attr: str,
    seed: int,
    eval_reference: str = "original",
    admissible_attrs: Optional[Sequence[str]] = None,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    synth_size: Optional[int] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO runs (
          created_at, git_commit, dataset, sdg_method, fairness_mechanism,
          eval_reference, protected_attrs, admissible_attrs, outcome_attr,
          epsilon, delta, seed, synth_size, extra_params, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            _git_commit(),
            dataset,
            sdg_method,
            fairness_mechanism,
            eval_reference,
            json.dumps(list(protected_attrs)),
            json.dumps(list(admissible_attrs)) if admissible_attrs is not None else None,
            outcome_attr,
            epsilon,
            delta,
            seed,
            synth_size,
            json.dumps(extra_params) if extra_params is not None else None,
        ),
    )
    conn.commit()
    return cur.lastrowid


def update_run_status(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    error_message: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> None:
    conn.execute(
        "UPDATE runs SET status = ?, error_message = ?, duration_seconds = ? WHERE run_id = ?",
        (status, error_message, duration_seconds, run_id),
    )
    conn.commit()


def log_metric(
    conn: sqlite3.Connection,
    run_id: int,
    metric_name: str,
    metric_value: Optional[float],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO metrics (run_id, metric_name, metric_value, extra)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(run_id, metric_name) DO UPDATE SET
          metric_value = excluded.metric_value, extra = excluded.extra
        """,
        (
            run_id,
            metric_name,
            metric_value,
            json.dumps(extra) if extra is not None else None,
        ),
    )
    conn.commit()


def log_metrics(conn: sqlite3.Connection, run_id: int, metrics: Dict[str, Optional[float]]) -> None:
    for name, value in metrics.items():
        log_metric(conn, run_id, name, value)


def log_edges(
    conn: sqlite3.Connection,
    run_id: int,
    edges: Iterable[Tuple[str, str]],
    weights: Optional[Dict[Tuple[str, str], float]] = None,
) -> None:
    weights = weights or {}
    rows = [
        (run_id, a, b, weights.get((a, b), weights.get((b, a))))
        for a, b in edges
    ]
    if rows:
        conn.executemany(
            "INSERT INTO graph_edges (run_id, node_a, node_b, weight) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def query_runs(conn: sqlite3.Connection, **filters: Any) -> pd.DataFrame:
    """Runs joined with their metrics as a wide dataframe (one row per run,
    one column per metric_name). Filter by exact-match column values, e.g.
    `query_runs(conn, dataset="adult", sdg_method="mst")`."""
    runs = pd.read_sql_query("SELECT * FROM runs", conn)
    for col, val in filters.items():
        runs = runs[runs[col] == val]
    if runs.empty:
        return runs
    metrics = pd.read_sql_query("SELECT * FROM metrics", conn)
    if metrics.empty:
        return runs
    wide = metrics.pivot_table(
        index="run_id", columns="metric_name", values="metric_value", aggfunc="first"
    )
    return runs.merge(wide, how="left", on="run_id")
