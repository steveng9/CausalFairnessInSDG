"""One (dataset__attribute, arm, classifier) cell of the Angelozzi & Arcolezi
benchmark on a FAccT sweep dataset. Runs under the `dpfair` environment and is
called by `facct_bench.py cell`, which lays the data out first.

Same call as `run_bench.py`: their `benchmark()`, their loader, their
classifier settings. Only the dataset description differs -- every column of
our tables is an integer-coded categorical.
"""
from __future__ import annotations

import argparse, warnings

warnings.filterwarnings("ignore")

from BenchmarkDPFair.Benchmark import BenchmarkDatasetConfig, BenchmarkInfo, benchmark
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

LR_PARAMS = {"max_iter": 10000, "solver": "saga", "penalty": "elasticnet",
             "l1_ratio": 0.5, "C": 0.8}
CLASSIFIERS = {
    "LR":  (LogisticRegression, LR_PARAMS),
    "RF":  (RandomForestClassifier, {}),
    "XGB": (XGBClassifier, {"objective": "binary:logistic"}),
}


def parse_eps(text: str):
    value = float(text)
    return int(value) if value.is_integer() else value  # float("inf") stays "inf"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--classifier", required=True, choices=sorted(CLASSIFIERS))
    p.add_argument("--target", required=True)
    p.add_argument("--sensitive", required=True)
    p.add_argument("--columns", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--eps", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    clf, ckwargs = CLASSIFIERS[a.classifier]
    benchmark(
        benchmark_info=BenchmarkInfo(
            dp_method=a.tag, output_dir=a.out + "/",
            seeds=[int(s) for s in a.seeds.split(",")],
            eps=[parse_eps(e) for e in a.eps.split(",")],
            classifier=clf, classifier_kwargs=ckwargs),
        data_conf=BenchmarkDatasetConfig(
            name=a.name, root_dir=a.data, target=a.target, sensitive_attr=a.sensitive,
            index_col="Unnamed: 0", categorical_cols=a.columns.split(","),
            sensitive_cols=[a.sensitive]),
    )


if __name__ == "__main__":
    main()
