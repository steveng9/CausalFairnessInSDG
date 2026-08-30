"""Generate our causal-fairness MST tables *inside the Angelozzi & Arcolezi
benchmark's own data layout*, so that both arms are scored by identical code.

The benchmark (PETS 2026) asks "where to intervene?" and answers with three
stages downstream of the synthesizer: pre-processing, in-processing and
post-processing fairness mitigators applied to DP synthetic data. Our
mechanism is a fourth stage it does not have -- inside the generator. To make
that a like-for-like comparison and not a re-run of our own protocol, we:

  * read *their* preprocessed real training split (already compressed,
    integer-coded and sensitive-column-binarised by their pipeline),
  * fit our MST on it under each fairness mechanism at each epsilon,
  * write the sample into a directory shaped exactly like theirs, under a
    synthesizer tag such as `ours-mst-cf`,
  * symlink their real test/train splits into that tag,

after which `run_bench.py` runs their unmodified `benchmark()` over both arms.
Every number in the comparison therefore comes from their loader, their
classifier, their MinMax scaling and their aif360 metric objects.

Two protocol notes, both recorded in manifest.json next to the output:
  * They call smartnoise-synth's MST with `preprocessor_eps = eps/2`, so half
    their nominal budget is spent discovering the domain. We treat the domain
    as public and give the whole budget to the model. We therefore also emit
    an `eps/2` arm (`--halve-eps`) so the comparison can be read either way.
  * Their MST is smartnoise's; ours is the private-pgm reference `mst.py` with
    fairness hooks in `select()`. `ours-mst-none` is the control that isolates
    that difference from the mechanism's effect.
"""
from __future__ import annotations

import argparse, json, os, sys, time, traceback
import numpy as np
import pandas as pd

from causal_fairness_sdg.sdg import MST
from causal_fairness_sdg.sdg.aim import AIM
from causal_fairness_sdg.fairness.base import AttributeRoles
from causal_fairness_sdg.fairness.registry import get_fairness_mechanism

BENCH = "/home/golobs/CausalFairnessInSDG/baselines/dp-fair-intervention-benchmark"
DATA = os.path.join(BENCH, "data")
DELTA = 1e-9

# Role splits, restricted to the columns their schema actually keeps. Adult
# loses education-num and the capital-gain/loss pair; COMPAS loses the two
# juvenile-count columns. `sex_only_broad_adm` on Adult is exactly DECAF's own
# admissible set (occupation, hours-per-week, workclass, education).
ROLES = {
    ("Adult", "sex_only_broad_adm"): (
        ["sex"], ["education", "occupation", "workclass", "hours-per-week"], ["income"]),
    ("Adult", "prefair"): (
        ["sex", "race", "native-country"],
        ["workclass", "education", "occupation", "hours-per-week"], ["income"]),
    ("Compas", "race_only_broad_adm"): (
        ["race"], ["priors_count", "c_charge_degree"], ["two_year_recid"]),
    ("Compas", "prefair"): (
        ["sex", "race"], ["priors_count", "c_charge_degree"], ["two_year_recid"]),
}


def load_split(name: str, seed: int, ref_synth: str = "mst"):
    """Their real train/test splits, as written by their own generator."""
    base = f"{DATA}/{name}/{ref_synth}"
    tr = pd.read_csv(f"{base}/DP-dataset-train/{name}_split_dataset_seed_{seed}_train.csv",
                     index_col=0)
    te = pd.read_csv(f"{base}/DP-dataset-test/{name}_split_dataset_seed_{seed}_test.csv",
                     index_col=0)
    return tr, te


def encode(train: pd.DataFrame, test: pd.DataFrame):
    """Contiguous 0..k-1 codes per column, over train union test.

    MST needs a dense domain. Their integer coding is dense for most columns
    but not guaranteed to be (COMPAS `age` and `priors_count` are raw counts),
    so we recode and keep the inverse to write the sample back out in their
    coding -- otherwise the synthetic table and the real test set would not
    share a vocabulary.
    """
    cats, enc = {}, pd.DataFrame(index=train.index)
    for c in train.columns:
        vals = np.union1d(train[c].unique(), test[c].unique())
        vals = np.sort(vals)
        cats[c] = vals
        enc[c] = pd.Categorical(train[c], categories=vals).codes.astype(int)
    domain = {c: int(len(v)) for c, v in cats.items()}
    return enc, domain, cats


def decode(df: pd.DataFrame, cats) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = np.asarray(cats[c])[df[c].to_numpy().astype(int)]
    return out


SYNTH = {"mst": MST, "aim": AIM}


def run_one(name, seed, eps_label, eps_used, mech, role_key, tag, cols_order,
            force=False, synth="mst"):
    """Fit at `eps_used`, but name the file by `eps_label`, so the half-budget
    arm lands in the same epsilon slot their loader iterates over."""
    out_dir = f"{DATA}/{name}/{tag}/DP-dataset-epsilon-{eps_str(eps_label)}"
    out = f"{out_dir}/{name}_split_dataset_seed_{seed}_epsilon-{eps_str(eps_label)}.csv"
    if os.path.exists(out) and not force:
        return "skip", 0.0

    t0 = time.time()
    train, test = load_split(name, seed)
    enc, domain, cats = encode(train, test)
    prot, adm, outc = ROLES[(name, role_key)]
    roles = AttributeRoles.create(protected=prot, admissible=adm, outcome=outc)

    res = SYNTH[synth]().fit_generate(
        data=enc, domain=domain, roles=roles,
        fairness_mechanism=get_fairness_mechanism(mech),
        epsilon=float(eps_used), delta=DELTA, n_synth=len(enc), seed=seed,
        dataset_name=name.lower(),
    )
    synth = decode(res.synthetic_data, cats)[cols_order].reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    synth.to_csv(out, index=True)

    with open(f"{DATA}/{name}/{tag}/edges_seed{seed}_eps{eps_str(eps_label)}.json",
              "w") as f:
        json.dump({"edges": [list(e) for e in res.graph_edges],
                   "mechanism": mech, "role_config": role_key,
                   "epsilon_nominal": eps_label, "epsilon_spent": eps_used,
                   "delta": DELTA, "seed": seed}, f, indent=1)
    return "done", time.time() - t0


def eps_str(e):
    """Match their filename convention: 1 and 10 are written without a
    decimal point, fractional budgets keep theirs."""
    return str(int(e)) if float(e) == int(e) else str(e)


def link_real_splits(name, tag, seeds):
    """Point the tag at their real train/test files, so their loader finds the
    identical held-out people for both arms."""
    for sub in ("DP-dataset-test", "DP-dataset-train"):
        dst = f"{DATA}/{name}/{tag}/{sub}"
        os.makedirs(dst, exist_ok=True)
        for s in seeds:
            fn = f"{name}_split_dataset_seed_{s}_{sub.split('-')[-1]}.csv"
            src = f"{DATA}/{name}/mst/{sub}/{fn}"
            lnk = f"{dst}/{fn}"
            if not os.path.exists(lnk) and os.path.exists(src):
                os.symlink(src, lnk)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="Adult,Compas")
    p.add_argument("--seeds", default="5,42,253,4112,32645")
    p.add_argument("--eps", default="1,10")
    p.add_argument("--mechs", default="none,ftu,dp,cf")
    p.add_argument("--halve-eps", action="store_true",
                   help="also emit an arm at eps/2, matching the budget their "
                        "MST actually gives the model after preprocessor_eps")
    p.add_argument("--force", action="store_true")
    p.add_argument("--synth", default="mst", choices=sorted(SYNTH))
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    epss = [float(e) for e in a.eps.split(",")]
    role_for = {"Adult": "sex_only_broad_adm", "Compas": "race_only_broad_adm"}

    for name in a.datasets.split(","):
        cols_order = list(pd.read_csv(
            f"{DATA}/{name}/mst/DP-dataset-train/"
            f"{name}_split_dataset_seed_{seeds[0]}_train.csv", index_col=0).columns)
        for mech in a.mechs.split(","):
            for half in ([False, True] if a.halve_eps else [False]):
                tag = f"ours-{a.synth}-{mech}" + ("-halfeps" if half else "")
                link_real_splits(name, tag, seeds)
                for eps in epss:
                    for seed in seeds:
                        used = eps / 2 if half else eps
                        try:
                            st, dt = run_one(name, seed, eps, used, mech,
                                             role_for[name], tag, cols_order,
                                             force=a.force, synth=a.synth)
                        except Exception as exc:
                            print(f"FAIL {name} {tag} eps={eps} seed={seed}: {exc}",
                                  flush=True)
                            traceback.print_exc()
                            continue
                        print(f"{st:4s} {name:7s} {tag:22s} eps={eps:<6g} "
                              f"seed={seed:<7d} {dt:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
