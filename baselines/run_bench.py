"""Run the Angelozzi & Arcolezi benchmark unmodified over both arms.

One process handles one (dataset, synthesizer-tag, classifier) cell and writes
into its own output directory, so cells are independent and the whole grid can
be fanned out across cores. A `.done` marker makes the grid resumable: a cell
that already finished is skipped rather than appended to twice (their
`save_experiment` opens the CSV in append mode).

`--dp-method` is just a directory name under data/<Dataset>/, so their own
`mst`/`aim` outputs and our `ours-mst-*` outputs are driven by identical code.
"""
from __future__ import annotations

import argparse, os, sys, time, warnings

warnings.filterwarnings("ignore")

from BenchmarkDPFair.Benchmark import BenchmarkDatasetConfig, BenchmarkInfo, benchmark
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BENCH = "/home/golobs/CausalFairnessInSDG/baselines/dp-fair-intervention-benchmark"

LR_PARAMS = {"max_iter": 10000, "solver": "saga", "penalty": "elasticnet",
             "l1_ratio": 0.5, "C": 0.8}
CLASSIFIERS = {
    "LR":  (LogisticRegression, LR_PARAMS),
    "RF":  (RandomForestClassifier, {}),
    "XGB": (XGBClassifier, {"objective": "binary:logistic"}),
}

# Their own example configs, verbatim, so the schema we score is the schema
# their published numbers were produced on.
DATASETS = {
    "Adult": dict(
        target="income", sensitive_attr="sex", index_col="Unnamed: 0",
        categorical_cols=["workclass", "education", "marital-status", "occupation",
                          "relationship", "race", "sex", "native-country", "income",
                          "age", "hours-per-week"],
        sensitive_cols=["race", "sex"]),
    "Compas": dict(
        target="two_year_recid", sensitive_attr="race", index_col="Unnamed: 0",
        categorical_cols=["race", "score_text", "c_charge_degree", "age", "sex",
                          "two_year_recid"],
        ordinal_cols=["priors_count"], sensitive_cols=["race", "sex"]),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--dp-method", required=True,
                   help="directory name under data/<Dataset>/ (mst, aim, ours-mst-cf, ...)")
    p.add_argument("--classifier", required=True, choices=sorted(CLASSIFIERS))
    p.add_argument("--seeds", default="5,42,253,4112,32645")
    p.add_argument("--eps", default="1,10")
    p.add_argument("--out", default="/home/golobs/CausalFairnessInSDG/baselines/output")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    eps = [float(e) if "." in e else int(e) for e in a.eps.split(",")]

    cell = f"{a.dataset}__{a.dp_method}__{a.classifier}"
    out_dir = os.path.join(a.out, cell)
    marker = os.path.join(out_dir, ".done")
    if os.path.exists(marker) and not a.force:
        print(f"SKIP {cell} (already done)", flush=True)
        return
    if a.force and os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    clf, ckwargs = CLASSIFIERS[a.classifier]
    t0 = time.time()
    print(f"START {cell} seeds={seeds} eps={eps}", flush=True)

    benchmark(
        benchmark_info=BenchmarkInfo(
            dp_method=a.dp_method, output_dir=out_dir + "/",
            seeds=seeds, eps=eps, classifier=clf, classifier_kwargs=ckwargs),
        data_conf=BenchmarkDatasetConfig(
            name=a.dataset, root_dir=f"{BENCH}/data", **DATASETS[a.dataset]),
    )

    with open(marker, "w") as f:
        f.write(f"{time.time() - t0:.1f}s\n")
    print(f"DONE {cell} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
