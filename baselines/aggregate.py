"""Collect the benchmark grid into one tidy table and print the comparison.

Every row of every cell's CSV is (Seed, Epsilon, Fair-Method, DP-Method, ...).
Their `save_experiment` writes two rows per mitigator -- the unmitigated
reference and the mitigated result -- so the unmitigated row is duplicated
across mitigators within a (seed, eps) block. We de-duplicate it into a single
`none` row per block, then report SPD (statistical parity difference, their
headline fairness metric) and ACC.

Arms:
  mst / aim                    their synthesizers, their eight mitigators
  ours-mst-{none,ftu,dp,cf}    our MST with the causal cut inside the generator
  ours-mst-none-halfeps        our MST at eps/2, the budget theirs gives the
                               model after preprocessor_eps
"""
from __future__ import annotations

import argparse, glob, os
import numpy as np
import pandas as pd

ROOT = "/home/golobs/CausalFairnessInSDG/baselines"


def load(out_dir: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(f"{out_dir}/*/*/*/results/*.csv"):
        cell = os.path.relpath(f, out_dir).split(os.sep)[0]
        ds, arm, clf = cell.split("__")
        d = pd.read_csv(f)
        d["dataset"], d["arm"], d["classifier"] = ds, arm, clf
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    d["Fair-Method"] = d["Fair-Method"].fillna("none")
    for c in ("SPD", "ACC", "DI", "EOD", "EODD"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def tidy(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, arm, classifier, seed, eps, fair-method).

    The unmitigated reference appears once per mitigator; keep one copy. It is
    identified as the row whose metrics repeat within the block -- simpler and
    safer: within each block the *first* row of each mitigator pair is the
    reference, so group and take the mitigated one, then add the reference back
    once from the row that has no mitigator recorded.
    """
    key = ["dataset", "arm", "classifier", "Seed", "Epsilon"]
    d = d.copy()
    d["Epsilon"] = d["Epsilon"].fillna(-1.0)  # -1 = the real-data reference pass
    # Within a block, each mitigator contributes exactly two rows in order:
    # reference then mitigated. Tag them.
    d["_i"] = d.groupby(key + ["Fair-Method"]).cumcount()
    mitig = d[(d["_i"] == 1) & (d["Fair-Method"] != "none")].copy()
    ref = (d[d["_i"] == 0].groupby(key, as_index=False)
             .first().assign(**{"Fair-Method": "none"}))
    out = pd.concat([ref, mitig], ignore_index=True)
    return out.drop(columns=["_i"])


STAGE = {"none": "0 none", "reweigh": "1 pre", "dir": "1 pre", "lfr": "1 pre",
         "egr": "2 in", "gsr": "2 in",
         "roc": "3 post", "eqodds": "3 post", "ceop": "3 post"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=f"{ROOT}/output")
    p.add_argument("--classifier", default="XGB")
    p.add_argument("--csv", default=f"{ROOT}/output/tidy.csv")
    a = p.parse_args()

    raw = load(a.out)
    if raw.empty:
        print("no results yet")
        return
    d = tidy(raw)
    d.to_csv(a.csv, index=False)
    print(f"wrote {a.csv}  ({len(d)} rows, "
          f"{d.arm.nunique()} arms, {d.classifier.nunique()} classifiers)\n")

    d = d[d.classifier == a.classifier]
    d = d[d.Epsilon > 0]
    d["absSPD"] = d.SPD.abs()

    for ds in sorted(d.dataset.unique()):
        for eps in sorted(d[d.dataset == ds].Epsilon.unique()):
            s = d[(d.dataset == ds) & (d.Epsilon == eps)]
            print(f"===== {ds}  eps={eps:g}  classifier={a.classifier} "
                  f"(|SPD| lower is fairer) =====")
            rows = []
            # their arms: every downstream mitigator
            for arm in sorted(x for x in s.arm.unique() if not x.startswith("ours")):
                for fm, g in s[s.arm == arm].groupby("Fair-Method"):
                    rows.append((STAGE.get(fm, "?"), f"{arm}+{fm}", len(g),
                                 g.absSPD.mean(), g.absSPD.std(ddof=1), g.ACC.mean()))
            # our arms: the in-generator cut, no downstream mitigator
            for arm in sorted(x for x in s.arm.unique() if x.startswith("ours")):
                g = s[(s.arm == arm) & (s["Fair-Method"] == "none")]
                if len(g):
                    rows.append(("4 in-generator", arm, len(g),
                                 g.absSPD.mean(), g.absSPD.std(ddof=1), g.ACC.mean()))
            t = pd.DataFrame(rows, columns=["stage", "method", "n",
                                            "|SPD|", "sd", "ACC"])
            t = t.sort_values(["stage", "|SPD|"])
            print(t.to_string(index=False,
                              float_format=lambda v: f"{v:.4f}"), "\n")


if __name__ == "__main__":
    main()
