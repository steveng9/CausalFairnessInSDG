#!/usr/bin/env python
"""Turn `experiments.db` rows into a readable results report.

Written after every run by `scripts/run_experiments.py`, so the report is
always current even while a sweep is still going. Can also be run standalone:

    python scripts/report_results.py [--batch TAG]

Produces, in `results/`:
  <batch>_runs.csv    one row per run, all metrics, ready for pandas/plots
  <batch>_report.md   the human-readable summary

Per-run scalar summaries collapse the per-protected-attribute metrics
(`dp_gap__sex`, `dp_gap__race`, ...) into a single worst-case number, because
role configs differ in *how many* protected attributes they declare and a mean
would otherwise reward configs that simply protect more attributes. `max_abs_*`
is the largest absolute gap over that run's protected attributes -- i.e. how
unfair the worst-served group is.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from causal_fairness_sdg.experiments import db
from causal_fairness_sdg.experiments.runner import SYNTH_REFERENCE_PREFIX

# Fairness families collapsed into worst-case scalars, per DF reference.
_GAP_FAMILIES = ["dp_gap", "cdp_gap", "tprb", "ctprb", "tnrb", "ctnrb"]
_UTILITY_COLS = [
    "tvd_1way", "tvd_2way", "avg_correlation_diff",
    "downstream_accuracy_mlp", "downstream_accuracy_lr", "downstream_accuracy_rf",
]


def _parse_extra(value) -> dict:
    try:
        return json.loads(value) if value else {}
    except (TypeError, ValueError):
        return {}


def load_batch(conn, batch: Optional[str]) -> pd.DataFrame:
    df = db.query_runs(conn)
    if df.empty:
        return df
    extras = df["extra_params"].map(_parse_extra)
    df["batch"] = extras.map(lambda d: d.get("batch"))
    df["role_config"] = extras.map(lambda d: d.get("role_config", "default"))
    if batch is not None:
        df = df[df["batch"] == batch]
    return df.reset_index(drop=True)


def add_scalar_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-attribute fairness columns into worst-case scalars, for
    both the real-holdout reference and the synthetic-data reference (DECAF's
    Distributional Fairness axis)."""
    df = df.copy()
    for prefix, tag in [("", "real"), (SYNTH_REFERENCE_PREFIX, "synth")]:
        for family in _GAP_FAMILIES:
            # `dp_gap__` never matches `cdp_gap__`/`synthref__dp_gap__`, so a
            # plain prefix match separates the families cleanly.
            cols = [c for c in df.columns if c.startswith(f"{prefix}{family}__")]
            df[f"max_abs_{family}__{tag}"] = (
                df[cols].abs().max(axis=1) if cols else np.nan
            )
    # Headline single number: worst-case demographic-parity gap of a model
    # trained on the synthetic data and deployed against the real world.
    df["fairness_gap"] = df["max_abs_dp_gap__real"]
    df["cond_fairness_gap"] = df["max_abs_cdp_gap__real"]
    return df


def _fmt(df: pd.DataFrame, floatfmt: str = "%.4f") -> str:
    if df.empty:
        return "_(no rows yet)_\n"
    return df.to_string(float_format=lambda v: floatfmt % v) + "\n"


def _agg(df: pd.DataFrame, by: List[str], cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if df.empty or not cols:
        return pd.DataFrame()
    return df.groupby(by, dropna=False)[cols].mean().round(4)


def build_markdown(df: pd.DataFrame, batch: Optional[str]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: List[str] = [
        f"# Experiment results — batch `{batch}`",
        "",
        f"_Report generated {now}. Rewritten after every run, so this is "
        "current even if the sweep is still running._",
        "",
    ]

    if df.empty:
        lines.append("No runs recorded yet.")
        return "\n".join(lines)

    counts = df["status"].value_counts().to_dict()
    total_minutes = df["duration_seconds"].fillna(0).sum() / 60
    lines += [
        "## Progress",
        "",
        f"- runs recorded: **{len(df)}** ({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))})",
        f"- compute so far: {total_minutes:.0f} minutes",
        "",
        "`partial` = the synthetic outcome column collapsed to a single class, so "
        "marginal-fidelity metrics were recorded but no classifier could be fitted.",
        "",
        "## How to read this",
        "",
        "- `fairness_gap` — worst-case |demographic parity gap| over the run's protected "
        "attributes, for a classifier trained on the synthetic data and evaluated on the "
        "**real holdout** (DECAF's DF-original reference: the case that matters in practice).",
        "- `cond_fairness_gap` — the same, conditioned on the admissible attributes (CDP). "
        "This is the gap CF is specifically designed to remove; DP-style mechanisms target "
        "`fairness_gap`.",
        "- `*__synth` — the same metric scored against the synthetic data's own distribution "
        "(DECAF's DF-synthetic reference, which the paper calls the uninteresting case). A big "
        "`real` vs `synth` divergence means fairness that only holds inside the synthetic data.",
        "- `tvd_1way`/`tvd_2way` — marginal fidelity, **lower is better**. "
        "`downstream_accuracy_*` — train-on-synthetic/test-on-real, **higher is better**.",
        "- DECAF rows have NULL epsilon: it has no DP mechanism and is the non-private "
        "reference point, not a competitor at a given epsilon.",
        "",
    ]

    ok = df[df["status"].isin(["done", "partial"])]

    lines += [
        "## Headline: does each fairness mechanism actually reduce the gap?",
        "",
        "Averaged over role configs and epsilons, within each dataset+method.",
        "",
        "```",
        _fmt(_agg(ok, ["dataset", "sdg_method", "fairness_mechanism"],
                  ["fairness_gap", "cond_fairness_gap", "tvd_1way",
                   "downstream_accuracy_mlp"])),
        "```",
        "",
        "## Privacy/fairness/utility interaction (epsilon sweep)",
        "",
        "```",
        _fmt(_agg(ok[ok["sdg_method"] != "decaf"],
                  ["dataset", "sdg_method", "epsilon", "fairness_mechanism"],
                  ["fairness_gap", "cond_fairness_gap", "tvd_1way",
                   "downstream_accuracy_mlp"])),
        "```",
        "",
        "## Sensitivity to the protected/admissible split",
        "",
        "The admissible set is what CF is allowed to use to block a path, so a narrower "
        "admissible set should push CF toward DP's behaviour.",
        "",
        "```",
        _fmt(_agg(ok, ["dataset", "role_config", "fairness_mechanism"],
                  ["fairness_gap", "cond_fairness_gap", "tvd_1way",
                   "downstream_accuracy_mlp"])),
        "```",
        "",
        "## Distributional Fairness axis: real vs synthetic reference",
        "",
        "```",
        _fmt(_agg(ok, ["dataset", "sdg_method", "fairness_mechanism"],
                  ["max_abs_dp_gap__real", "max_abs_dp_gap__synth",
                   "max_abs_cdp_gap__real", "max_abs_cdp_gap__synth"])),
        "```",
        "",
    ]

    # Best configurations: low gap AND high accuracy. Ranked on the real
    # reference only, since the synthetic reference is trivially gameable.
    scored = ok.dropna(subset=["fairness_gap", "downstream_accuracy_mlp"])
    if not scored.empty:
        show = [
            "dataset", "role_config", "sdg_method", "fairness_mechanism", "epsilon",
            "fairness_gap", "cond_fairness_gap", "tvd_1way", "downstream_accuracy_mlp",
        ]
        best = scored.sort_values(
            ["fairness_gap", "downstream_accuracy_mlp"], ascending=[True, False]
        ).head(20)[show]
        acc_floor = scored["downstream_accuracy_mlp"].quantile(0.5)
        useful = scored[scored["downstream_accuracy_mlp"] >= acc_floor].sort_values(
            "fairness_gap"
        ).head(20)[show]
        lines += [
            "## Lowest fairness gap overall",
            "",
            "```",
            _fmt(best.reset_index(drop=True)),
            "```",
            "",
            f"## Lowest fairness gap among configs with usable accuracy (>= median, {acc_floor:.3f})",
            "",
            "Guards against the degenerate 'perfectly fair because the model predicts one "
            "class' solution.",
            "",
            "```",
            _fmt(useful.reset_index(drop=True)),
            "```",
            "",
        ]

    failed = df[df["status"] == "failed"]
    if not failed.empty:
        lines += [
            "## Failures",
            "",
            "```",
            _fmt(failed[["dataset", "role_config", "sdg_method", "fairness_mechanism",
                         "epsilon", "error_message"]].reset_index(drop=True)),
            "```",
            "",
        ]

    partial = df[df["status"] == "partial"]
    if not partial.empty:
        lines += [
            "## Partial runs (collapsed synthetic outcome)",
            "",
            "```",
            _fmt(partial[["dataset", "role_config", "sdg_method", "fairness_mechanism",
                          "epsilon", "tvd_1way"]].reset_index(drop=True)),
            "```",
            "",
        ]

    return "\n".join(lines)


def write_report(conn, batch: Optional[str], results_dir: Path) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df = add_scalar_summaries(load_batch(conn, batch))
    tag = batch or "all"
    csv_path = results_dir / f"{tag}_runs.csv"
    md_path = results_dir / f"{tag}_report.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown(df, batch))
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="overnight-2026-07-30")
    parser.add_argument("--all", action="store_true", help="ignore the batch filter")
    parser.add_argument(
        "--results-dir", default=str(Path(__file__).resolve().parents[1] / "results")
    )
    args = parser.parse_args()
    conn = db.get_connection()
    path = write_report(conn, None if args.all else args.batch, Path(args.results_dir))
    print(path.read_text())
    print(f"\n-> {path}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
