#!/usr/bin/env python
"""FairPFN as a downstream baseline for the sweep's generation-time cuts.

FairPFN (Robertson et al., ICML 2025) is a tabular foundation model that
predicts an outcome in context while removing the causal effect of ONE BINARY
protected attribute, with no causal graph supplied. It is a classifier, not a
generator, so it sits beside the downstream mitigators: the recipient of a
release fixes fairness at prediction time instead of the publisher cutting
edges at generation time.

For every (dataset, seed) of a sweep batch this scores FairPFN on the same
real holdout the sweep scores against, with its context drawn from

  real              the real training split (fairness fix on real data)
  mst_eps<e>        MST's release with no fairness mechanism, per epsilon
  decaf_tabpfn      the TabPFN generator's release with no fairness mechanism
  (any generator in --synth-methods, labelled <method>[_eps<e>])
  <src>_<mech>@<role>  with --stack: an *edited* release (dp or ftu for the
                    split <role>) as the context, i.e. the publisher and the
                    recipient both act; scored only on that split's
                    protected attributes

once per protected attribute named in any of the dataset's role splits. Each
attribute is binarized exactly as the sweep's fairness metrics binarize it
(real-holdout modal value = 0, everything else = 1) and replaces its raw
column; the dataset's other protected attributes stay in as ordinary
features. Two controls share each context and feature layout, so the cost of
the small context can be told apart from the cost of the fairness fix:

  rf                random forest, protected attribute included
  rf_unaware        random forest, protected attribute dropped

Two stages, because FairPFN's released code needs its own old environment:

  export   (sdg env)      writes splits and a manifest of synthetic sources
  run      (fairpfn env)  predicts, scores, appends to results.csv (resumable)

FairPFN's code and checkpoint live outside this repository (no licence file
upstream): --fairpfn-home, default ~/external/FairPFN.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "results/models_diff/model_submitit_0c_id_e3e764c1_epoch_-1.cpkt"
MODELS = ["fairpfn", "rf", "rf_unaware"]
KEY = ["dataset", "seed", "source", "ctx_size", "protected", "model"]


def _out_dir(args) -> Path:
    return REPO_ROOT / "sweeps" / args.batch / "fairpfn_baseline"


# --------------------------------------------------------------------- export
def _none_runs(batch_dir: Path) -> pd.DataFrame:
    """Every finished no-mechanism run of the batch whose table was kept."""
    rows = []
    for db_path in sorted(glob.glob(str(batch_dir / "db" / "*.db"))):
        shard = Path(db_path).stem
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        for r in conn.execute(
            "select run_id, dataset, sdg_method, epsilon, seed, protected_attrs,"
            " admissible_attrs, extra_params, fairness_mechanism from runs"
            " where status = 'done'"
        ):
            rows.append((shard,) + r)
        conn.close()
    df = pd.DataFrame(
        rows,
        columns="shard run_id dataset method epsilon seed protected admissible"
                " extra mechanism".split(),
    )
    df["role"] = df.extra.map(lambda s: json.loads(s).get("role_config"))
    return df


def export(args) -> int:
    from sklearn.model_selection import train_test_split

    from causal_fairness_sdg.data.datasets import DATASETS

    batch_dir = REPO_ROOT / "sweeps" / args.batch
    out = _out_dir(args) / "inputs"
    runs = _none_runs(batch_dir)
    for name in args.datasets.split(","):
        data, _domain, default_roles = DATASETS[name]()
        outcome = next(iter(default_roles.outcome))
        mine = runs[runs.dataset == name]
        roles = {
            role: {
                "protected": json.loads(g.protected.iloc[0]),
                "admissible": json.loads(g.admissible.iloc[0]),
            }
            for role, g in mine.groupby("role")
        } or {
            "default": {
                "protected": sorted(default_roles.protected),
                "admissible": sorted(default_roles.admissible),
            }
        }
        protected = sorted({p for r in roles.values() for p in r["protected"]})
        # Anything not already numeric gets one fixed code per level, taken
        # from the full table so every split and source agrees.
        levels = {
            c: sorted(data[c].astype(str).unique())
            for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])
        }

        def coded(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            for c, lv in levels.items():
                df[c] = pd.Categorical(df[c].astype(str), categories=lv).codes
            return df

        for seed in args.seeds:
            train_df, holdout_df = train_test_split(
                data, test_size=0.3, random_state=seed
            )
            cell = out / f"{name}-s{seed}"
            cell.mkdir(parents=True, exist_ok=True)
            coded(train_df).to_parquet(cell / "train.parquet", index=False)
            coded(holdout_df).to_parquet(cell / "holdout.parquet", index=False)
            sources = {"real": str(cell / "train.parquet")}
            none = mine[(mine.seed == seed) & (mine.mechanism == "none")]
            for (method, eps), g in none.groupby(["method", "epsilon"], dropna=False):
                if method not in args.synth_methods.split(","):
                    continue
                # The role split does not touch a no-mechanism release; take
                # the first so the choice is reproducible.
                r = g.sort_values("role").iloc[0]
                path = batch_dir / "synth" / r.shard / f"run_{r.run_id}.parquet"
                if not path.exists():
                    continue
                label = method if pd.isna(eps) else f"{method}_eps{eps:g}"
                if levels:
                    coded(pd.read_parquet(path)).to_parquet(
                        cell / f"synth_{label}.parquet", index=False
                    )
                    path = cell / f"synth_{label}.parquet"
                sources[label] = str(path)
            # Edited releases (--stack "method[:epsilon],..."): one per
            # mechanism and role split, scored on that split's attributes.
            stack = dict((s.split(":") + [None])[:2] for s in args.stack.split(",") if s)
            source_protected = {}
            edited = mine[(mine.seed == seed)
                          & mine.mechanism.isin(args.stack_mechanisms.split(","))]
            for (method, eps, mech, role), g in edited.groupby(
                    ["method", "epsilon", "mechanism", "role"], dropna=False):
                if method not in stack:
                    continue
                if stack[method] is not None and (pd.isna(eps) or float(eps) != float(stack[method])):
                    continue
                r = g.sort_values("run_id").iloc[0]
                path = batch_dir / "synth" / r.shard / f"run_{r.run_id}.parquet"
                if not path.exists():
                    continue
                label = (method if pd.isna(eps) else f"{method}_eps{eps:g}") + f"_{mech}@{role}"
                if levels:
                    coded(pd.read_parquet(path)).to_parquet(
                        cell / f"synth_{label}.parquet", index=False
                    )
                    path = cell / f"synth_{label}.parquet"
                sources[label] = str(path)
                source_protected[label] = json.loads(r.protected)
            meta = {
                "dataset": name, "seed": seed, "outcome": outcome,
                "protected": protected, "roles": roles,
                "source_protected": source_protected,
                "majority_values": {
                    p: int(coded(holdout_df)[p].mode().iloc[0]) for p in protected
                },
                "sources": sources,
            }
            (cell / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"{name} s{seed}: {len(sources)} sources, protected {protected}",
                  flush=True)
    return 0


# ------------------------------------------------------------------------ run
def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fairpfn_proba(home: Path, X_ctx, y_ctx, X_test, seed: int, device: str):
    import torch

    from scripts.transformer_prediction_interface.base import FairPFNClassifier

    torch.manual_seed(seed)
    np.random.seed(seed)
    clf = FairPFNClassifier()
    clf.model_string = CHECKPOINT  # the shipped config points at the authors' cluster
    clf.device = device
    clf.fp16_inference = False  # upstream's half-precision path fails an internal dtype assert
    clf.seed = seed
    clf.fit(X_ctx, y_ctx)
    proba = clf.predict_proba(torch.tensor(X_test))
    return proba[:, list(clf.classes_).index(1)]


def _cdp_fast(df: pd.DataFrame, admissible: list) -> float:
    """`fairness_metrics._conditional_average(..., "dp")` in one groupby: the
    size-weighted mean of the within-stratum parity gap over the strata of
    the admissible attributes that contain both groups. `run` checks it
    against the repository's own function once per process."""
    g = df.groupby(admissible + ["_s"])["_predicted"].agg(["sum", "count"]).unstack("_s")
    if 0 not in g["count"] or 1 not in g["count"]:
        return float("nan")
    ok = g["count"][0].notna() & g["count"][1].notna()
    g = g[ok]
    if not len(g):
        return float("nan")
    gap = g["sum"][1] / g["count"][1] - g["sum"][0] / g["count"][0]
    return float(np.average(gap, weights=g["count"][0] + g["count"][1]))


def _score(fm, holdout, proba, bin_prot, outcome, roles, attr, ctx_pos_rate) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    y = holdout[outcome].to_numpy()
    out = {
        "auc": float(roc_auc_score(y, proba)),
        # Threshold-free companion to dp_gap: difference in mean score.
        "mean_score_gap": float(proba[bin_prot == 1].mean() - proba[bin_prot == 0].mean()),
    }
    # Two decision rules. "" is the sweep's own (score >= 0.5). FairPFN's
    # scores are pulled towards the base rate, so at 0.5 it can predict
    # almost nobody positive and look fair for free; "_matched" instead
    # accepts the top scores until the positive rate equals the context's,
    # which puts every model on the same acceptance budget.
    cut = np.quantile(proba, 1.0 - ctx_pos_rate)
    for suffix, pred in [("", (proba >= 0.5).astype(int)),
                         ("_matched", (proba > cut).astype(int))]:
        df = holdout.copy()
        df["_s"], df["_predicted"] = bin_prot, pred
        out.update({
            f"accuracy{suffix}": float((pred == y).mean()),
            f"f1_macro{suffix}": float(f1_score(y, pred, average="macro")),
            f"pos_rate{suffix}": float(pred.mean()),
            f"dp_gap{suffix}": fm.demographic_parity_gap(df, "_s", "_predicted"),
            f"tprb{suffix}": fm.true_positive_rate_balance(df, "_s", "_predicted", outcome),
            f"tnrb{suffix}": fm.true_negative_rate_balance(df, "_s", "_predicted", outcome),
        })
        for role, r in roles.items():
            if attr in r["protected"] and r["admissible"]:
                out[f"cdp_gap{suffix}@{role}"] = _cdp_fast(df, r["admissible"])
    return out


def _check_cdp_fast(fm) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(0, 3, size=(4000, 3)), columns=["a", "b", "c"])
    df["_s"], df["_predicted"] = rng.integers(0, 2, 4000), rng.integers(0, 2, 4000)
    df.loc[df.a == 2, "_s"] = 1  # strata with one group only must be skipped
    want = fm._conditional_average(df, "_s", "_predicted", ["a", "b", "c"], "dp")
    got = _cdp_fast(df, ["a", "b", "c"])
    assert abs(want - got) < 1e-12, (want, got)


def run(args) -> int:
    from sklearn.ensemble import RandomForestClassifier

    fm = _load_by_path(
        "fairness_metrics",
        REPO_ROOT / "src" / "causal_fairness_sdg" / "eval" / "fairness_metrics.py",
    )
    _check_cdp_fast(fm)
    out = _out_dir(args)
    # Parallel processes each append to their own --results file; every
    # results*.csv in the folder counts as done.
    results_path = out / args.results
    preds_dir = out / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for prev_path in sorted(out.glob("results*.csv")):
        prev = pd.read_csv(prev_path)
        done |= set(map(tuple, prev[KEY].astype(str).values))

    home = Path(args.fairpfn_home).expanduser().resolve()
    sys.path.insert(0, str(home))
    os.chdir(home)  # upstream opens its artifacts by relative path

    cells = sorted(glob.glob(str(out / "inputs" / "*" / "meta.json")))
    if args.datasets:
        keep = set(args.datasets.split(","))
        cells = [c for c in cells if json.loads(Path(c).read_text())["dataset"] in keep]
    # Breadth first over seeds, so partial results already cover every dataset.
    cells.sort(key=lambda c: (json.loads(Path(c).read_text())["seed"], c))
    for meta_path in cells:
        meta = json.loads(Path(meta_path).read_text())
        name, seed, outcome = meta["dataset"], meta["seed"], meta["outcome"]
        holdout = pd.read_parquet(Path(meta_path).parent / "holdout.parquet")
        y_test = holdout[outcome].to_numpy()
        for source, path in meta["sources"].items():
            table = pd.read_parquet(path)[holdout.columns]
            for ctx_size in args.ctx_sizes:
                ctx = table.sample(n=min(ctx_size, len(table)), random_state=seed)
                for attr in meta["protected"]:
                    if attr not in meta.get("source_protected", {}).get(source, [attr]):
                        continue
                    key = [name, seed, source, ctx_size, attr]
                    todo = [m for m in MODELS
                            if tuple(map(str, key + [m])) not in done]
                    if not todo:
                        continue
                    major = meta["majority_values"][attr]
                    others = [c for c in holdout.columns if c not in (outcome, attr)]

                    def layout(df):
                        s = (df[attr] != major).astype(int).to_numpy()
                        return np.column_stack([s, df[others].to_numpy()]).astype(np.float32)

                    X_ctx, X_test = layout(ctx), layout(holdout)
                    y_ctx = ctx[outcome].to_numpy().astype(int)
                    rows = []
                    for model in todo:
                        row = dict(zip(KEY, key + [model]), n_ctx=len(ctx),
                                   n_test=len(holdout), status="done")
                        start = time.time()
                        try:
                            if len(np.unique(y_ctx)) < 2 or len(np.unique(X_ctx[:, 0])) < 2:
                                raise ValueError("context has one outcome class or one group")
                            if model == "fairpfn":
                                proba = _fairpfn_proba(
                                    home, X_ctx, y_ctx, X_test, seed, args.device
                                )
                                np.save(
                                    preds_dir / f"{name}-s{seed}-{source}-n{ctx_size}-{attr}.npy",
                                    proba.astype(np.float16),
                                )
                            else:
                                cols = slice(1, None) if model == "rf_unaware" else slice(None)
                                rf = RandomForestClassifier(random_state=0, n_jobs=args.n_jobs)
                                rf.fit(X_ctx[:, cols], y_ctx)
                                proba = rf.predict_proba(X_test[:, cols])[:, list(rf.classes_).index(1)]
                            row["seconds"] = round(time.time() - start, 2)
                            row.update(_score(fm, holdout, proba, X_test[:, 0].astype(int),
                                              outcome, meta["roles"], attr,
                                              float(y_ctx.mean())))
                        except Exception as exc:  # keep the cell inspectable, move on
                            row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
                            if "out of memory" in str(exc).lower():
                                import torch
                                torch.cuda.empty_cache()
                        row.setdefault("seconds", round(time.time() - start, 2))
                        rows.append(row)
                    new = pd.DataFrame(rows)
                    if results_path.exists():
                        new = pd.concat([pd.read_csv(results_path), new], ignore_index=True)
                    tmp = results_path.with_suffix(".csv.tmp")
                    new.to_csv(tmp, index=False)
                    tmp.replace(results_path)
                    fp = next((r for r in rows if r["model"] == "fairpfn"), rows[0])
                    print(f"{name} s{seed} {source} n={ctx_size} {attr}: "
                          f"{fp['status']} auc={fp.get('auc', float('nan')):.3f} "
                          f"dp@matched={fp.get('dp_gap_matched', float('nan')):+.3f} "
                          f"{fp['seconds']}s",
                          flush=True)
    print("fairpfn baseline complete", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["export", "run"])
    parser.add_argument("--batch", default="facct-2026-10-09")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--synth-methods", default="mst,decaf_tabpfn")
    parser.add_argument("--ctx-sizes", type=lambda s: [int(x) for x in s.split(",")],
                        default=[2000, 10000])
    parser.add_argument("--stack", default="",
                        help='edited releases as contexts, "method[:epsilon],..."')
    parser.add_argument("--stack-mechanisms", default="dp,ftu")
    parser.add_argument("--results", default="results.csv",
                        help="file this process appends to (one per parallel process)")
    parser.add_argument("--fairpfn-home", default="~/external/FairPFN")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "export":
        if not args.datasets:
            parser.error("export needs --datasets")
        return export(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
