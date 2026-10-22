"""Angelozzi & Arcolezi's eight downstream mitigators on the FAccT datasets.

`gen_ours.py` / `run_bench.py` compare the two approaches on the benchmark's
own Adult and COMPAS files. This script does the same for the datasets of the
FAccT sweep, which the benchmark does not ship: it hands their unmodified
`benchmark()` *our* splits and the synthetic tables the sweep already saved
(`sweeps/<batch>/synth/`), laid out the way their default loader expects.
Nothing is generated here.

Their harness takes one binary sensitive attribute per run, privileged = 1.
So one benchmark "dataset" is a (sweep dataset, protected attribute) pair,
named `<dataset>__<attribute>`, for every attribute protected in any of the
dataset's role splits:

  * the attribute is binarised majority-versus-rest (the value most common in
    the full real table against everything else), as the sweep's own gap
    metrics and the FairPFN baseline do;
  * of those two groups, the one with the higher real rate of outcome = 1 is
    coded 1 ("privileged"). Where outcome = 1 is the unfavourable label
    (COMPAS recidivism, CDC diabetes) read the signed metrics accordingly;
  * the dataset's other protected attributes stay in as ordinary features.

Arms (their `dp_method`, a directory name) for each sweep generator G:

  G-none               G's no-mechanism release: their unmitigated row plus
                       their eight mitigators. This is the downstream baseline.
  G-<mech>@<split>     G with our mechanism inside the generator, for every
                       role split that protects the attribute. Their
                       unmitigated row is our number under their protocol.

Their loader reads the real training split for the reference pass, the
synthetic table for each epsilon, and splits the real holdout 50/50 into
calibration and test (random_state = seed). Non-private generators sit in the
epsilon slot `inf`.

    # 1. splits, manifest, grid (sdg env)
    python baselines/facct_bench.py export --batches facct-2026-10-09
    # 2. every cell; resumable (sdg env drives, dpfair env scores)
    JOBS=14 CPUS=32-47 baselines/launch_facct_grid.sh
    # 3. one tidy table
    python baselines/facct_bench.py collect
    # re-run batches that supersede some arms (e.g. facct-cffix-*): export with
    # --merge, then rerun those cells with `cell ... --force`
    # TabFairGAN at its per-split lambda: export --merge --methods tabfairgan
    # --batches facct-2026-10-09,<facct-tfg-lam*> --choose report/data/facct/tabfairgan_lambda_choice.csv
"""
from __future__ import annotations

import argparse, glob, json, os, shutil, sqlite3, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = ROOT / "facct_data"
OUT = ROOT / "facct_output"
DPFAIR_PY = "/home/golobs/miniconda3/envs/dpfair/bin/python"
TEST_SIZE = 0.3  # must match `experiments.runner.RunConfig.test_size`
INDEX_COL = "Unnamed: 0"


def eps_label(eps) -> str:
    if eps is None or (isinstance(eps, float) and np.isnan(eps)):
        return "inf"
    return str(int(eps)) if float(eps).is_integer() else str(float(eps))


def binarise(frame: pd.DataFrame, attr: str, majority, privileged_is_majority: bool) -> pd.DataFrame:
    out = frame.copy()
    is_majority = out[attr] == majority
    out[attr] = (is_majority if privileged_is_majority else ~is_majority).astype(int)
    return out


def cmd_export(args) -> None:
    sys.path.insert(0, str(REPO / "src"))
    from sklearn.model_selection import train_test_split
    from causal_fairness_sdg.data.datasets import DATASETS

    runs = []
    for batch in args.batches.split(","):
        for db in sorted((REPO / "sweeps" / batch / "db").glob("*.db")):
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            frame = pd.read_sql("SELECT * FROM runs WHERE status = 'done'", conn)
            conn.close()
            frame["batch"], frame["shard"] = batch, db.stem
            runs.append(frame)
    runs = pd.concat(runs, ignore_index=True)
    runs["role_config"] = runs["extra_params"].map(lambda s: json.loads(s or "{}").get("role_config"))
    if args.datasets:
        runs = runs[runs["dataset"].isin(args.datasets.split(","))]
    if args.methods:
        runs = runs[runs["sdg_method"].isin(args.methods.split(","))]
    if args.eps:
        keep = {float(e) for e in args.eps.split(",")}
        runs = runs[runs["epsilon"].isna() | runs["epsilon"].isin(keep)]
    if args.choose:
        # For each (dataset, method, mechanism, split) the file lists, keep only
        # the runs of the batch it names (e.g. TabFairGAN at its chosen lambda).
        key = ["dataset", "sdg_method", "fairness_mechanism", "role_config"]
        pick = pd.read_csv(args.choose)[key + ["batch"]].rename(columns={"batch": "chosen"})
        chosen = runs.reset_index(drop=True).merge(pick, on=key, how="left")["chosen"]
        runs = runs[(chosen.isna() | (chosen == runs["batch"].to_numpy())).to_numpy()]

    grid = []
    for dataset, ds_runs in runs.groupby("dataset"):
        data, _domain, _roles = DATASETS[dataset]()
        target = ds_runs["outcome_attr"].iloc[0]
        splits = {int(s): train_test_split(data, test_size=TEST_SIZE, random_state=int(s))
                  for s in sorted(ds_runs["seed"].unique())}
        role_protected = {rc: json.loads(p) for rc, p in
                          ds_runs.groupby("role_config")["protected_attrs"].first().items()}
        first_split = sorted(role_protected)[0]
        for attr in sorted({a for prot in role_protected.values() for a in prot}):
            name = f"{dataset}__{attr}"
            majority = data[attr].mode().iloc[0]
            rate_major = data.loc[data[attr] == majority, target].mean()
            rate_rest = data.loc[data[attr] != majority, target].mean()
            privileged_is_majority = bool(rate_major >= rate_rest)
            base = DATA / name
            for seed, (train, holdout) in splits.items():
                for part, frame in (("train", train), ("test", holdout)):
                    path = base / "_splits" / part / f"{name}_split_dataset_seed_{seed}_{part}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    binarise(frame, attr, majority, privileged_is_majority).to_csv(path)
            tags = {}
            for (method, mech, split), cell in ds_runs.groupby(
                    ["sdg_method", "fairness_mechanism", "role_config"]):
                if mech == "none":
                    if split != first_split:
                        continue  # the role split does not affect a no-mechanism release
                    tag = f"{method}-none"
                elif attr in role_protected[split]:
                    tag = f"{method}-{mech}@{split}"
                else:
                    continue
                files = {}
                for _, run in cell.iterrows():
                    eps = eps_label(run["epsilon"])
                    files.setdefault(eps, {})[str(int(run["seed"]))] = str(
                        REPO / "sweeps" / run["batch"] / "synth" / run["shard"]
                        / f"run_{run['run_id']}.parquet")
                tags[tag] = files
                grid.append((name, tag))
            if args.merge and (base / "manifest.json").exists():
                # Replace only the arms exported now (e.g. a CF re-run batch),
                # keep every other arm of the existing manifest.
                tags = {**json.loads((base / "manifest.json").read_text())["tags"], **tags}
            manifest = {
                "dataset": dataset, "attribute": attr, "target": target,
                "columns": list(data.columns), "majority_value": int(majority),
                "privileged_is_majority": privileged_is_majority,
                "real_positive_rate": {"majority": float(rate_major), "rest": float(rate_rest)},
                "role_splits_protecting": sorted(rc for rc, p in role_protected.items() if attr in p),
                "seeds": sorted(splits), "tags": tags,
            }
            (base / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
            print(f"{name}: {len(tags)} arms", flush=True)
    with open(DATA / "grid.txt", "a") as handle:
        for name, tag in grid:
            for clf in args.classifiers.split(","):
                handle.write(f"{name} {tag} {clf}\n")
    lines = sorted(set((DATA / "grid.txt").read_text().splitlines()))
    (DATA / "grid.txt").write_text("\n".join(lines) + "\n")
    print(f"grid: {len(lines)} cells in {DATA / 'grid.txt'}")


def cmd_cell(args) -> None:
    """Materialise one arm's synthetic tables, score it, clean up."""
    name, tag, clf = args.name, args.tag, args.classifier
    out_dir = OUT / f"{name}__{tag}__{clf}"
    if (out_dir / ".done").exists() and not args.force:
        print(f"SKIP {name} {tag} {clf}")
        return
    manifest = json.loads((DATA / name / "manifest.json").read_text())
    attr, cols = manifest["attribute"], manifest["columns"]
    arm = DATA / name / tag
    if arm.exists():
        shutil.rmtree(arm)
    arm.mkdir(parents=True)
    for part in ("train", "test"):
        os.symlink(DATA / name / "_splits" / part, arm / f"DP-dataset-{part}")
    files = manifest["tags"][tag]
    seeds = sorted({int(s) for by_seed in files.values() for s in by_seed})
    eps_done = []
    for eps, by_seed in files.items():
        if set(map(int, by_seed)) != set(seeds):
            continue  # their loop needs every seed at every epsilon
        eps_done.append(eps)
        for seed, parquet in by_seed.items():
            synth = pd.read_parquet(parquet)[cols]
            synth = binarise(synth, attr, manifest["majority_value"], manifest["privileged_is_majority"])
            path = arm / f"DP-dataset-epsilon-{eps}" / f"{name}_split_dataset_seed_{seed}_epsilon-{eps}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            synth.to_csv(path)
    started = time.time()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    code = subprocess.call([
        DPFAIR_PY, str(ROOT / "run_bench_facct.py"), "--name", name, "--tag", tag,
        "--classifier", clf, "--target", manifest["target"], "--sensitive", attr,
        "--columns", ",".join(cols), "--seeds", ",".join(map(str, seeds)),
        "--eps", ",".join(sorted(eps_done, key=float)), "--data", str(DATA), "--out", str(out_dir),
    ])
    shutil.rmtree(arm)
    if code == 0:
        (out_dir / ".done").write_text(f"{time.time() - started:.1f}s\n")
    print(f"{'DONE' if code == 0 else 'FAILED'} {name} {tag} {clf} in {time.time() - started:.0f}s", flush=True)
    sys.exit(code)


def cmd_collect(args) -> None:
    rows = []
    for done in sorted(OUT.glob("*/.done")):
        name_tag, clf = done.parent.name.rsplit("__", 1)
        dataset, attr, tag = name_tag.split("__", 2)
        for f in glob.glob(f"{done.parent}/**/results/*.csv", recursive=True):
            d = pd.read_csv(f)
            d["dataset"], d["attribute"], d["arm"], d["classifier"] = dataset, attr, tag, clf
            rows.append(d)
    if not rows:
        print("nothing collected")
        return
    out = pd.concat(rows, ignore_index=True)
    arm = out["arm"].str.extract(r"^(?P<sdg_method>[^-]+(?:_[^-]+)*)-(?P<fairness_mechanism>[a-z]+)(?:@(?P<role_config>.+))?$")
    out = pd.concat([out, arm], axis=1)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"{len(out)} rows from {out['arm'].nunique()} arms -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--batches", required=True)
    export.add_argument("--datasets", default=None)
    export.add_argument("--methods", default=None)
    export.add_argument("--eps", default=None, help="keep only these epsilons (non-private runs always kept)")
    export.add_argument("--classifiers", default="XGB")
    export.add_argument("--merge", action="store_true",
                        help="update the arms exported now in each existing manifest, keep the rest "
                             "(for re-run batches that supersede some arms)")
    export.add_argument("--choose", default=None,
                        help="CSV of dataset, sdg_method, fairness_mechanism, role_config, batch: "
                             "export those arms from that batch only")
    export.set_defaults(func=cmd_export)
    cell = sub.add_parser("cell")
    cell.add_argument("name"), cell.add_argument("tag"), cell.add_argument("classifier")
    cell.add_argument("--force", action="store_true")
    cell.set_defaults(func=cmd_cell)
    collect = sub.add_parser("collect")
    collect.add_argument("--out", default=str(REPO / "report/data/facct/angelozzi_mitigators.csv"))
    collect.set_defaults(func=cmd_collect)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
