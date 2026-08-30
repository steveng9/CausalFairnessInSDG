"""Render manuscript/table2.tex from the benchmark grid.

Rerun after new arms land (e.g. the AIM arm) and the table picks them up.
"""
import numpy as np, pandas as pd

TIDY = "/home/golobs/CausalFairnessInSDG/baselines/output/tidy.csv"
OUT = "/home/golobs/CausalFairnessInSDG/manuscript/table2.tex"
# Majority-class rate measured on their own real held-out test splits, averaged
# over the five seeds we run (see README): Adult .7543-.7588, COMPAS .5293-.5398.
MAJ = {"Adult": 0.7554, "Compas": 0.5338}
CELLS = [("Adult", 1), ("Adult", 10), ("Compas", 1), ("Compas", 10)]
NICE = {"none": "no mitigation", "reweigh": "Reweighing",
        "dir": "Disp.\\ impact remover", "lfr": "Learned fair repr.",
        "egr": "Exp.\\ gradient", "gsr": "Grid search", "roc": "Reject option",
        "eqodds": "Equalised odds", "ceop": "Calib.\\ eq.\\ odds"}
STAGE = {"none": "", "reweigh": "pre", "dir": "pre", "lfr": "pre",
         "egr": "in", "gsr": "in", "roc": "post", "eqodds": "post", "ceop": "post"}
MECH = {"ftu": "FTU", "dp": "DemPar", "cf": "CF"}


def main(base="mst", clf="XGB"):
    d = pd.read_csv(TIDY)
    d = d[(d.classifier == clf) & (d.Epsilon > 0)].copy()
    d["a"] = d.SPD.abs().replace([np.inf, -np.inf], np.nan)
    d["ACC"] = d.ACC.replace([np.inf, -np.inf], np.nan)

    def cellvals(arm, fm):
        out = []
        for ds, eps in CELLS:
            g = d[(d.arm == arm) & (d["Fair-Method"] == fm)
                  & (d.dataset == ds) & (d.Epsilon == eps)]
            out.append((g.a.mean(), g.ACC.mean(), ds, g.TPR.mean()) if len(g)
                       else (np.nan, np.nan, ds, np.nan))
        return out

    rows = []
    for fm in ["none", "reweigh", "dir", "lfr", "egr", "gsr", "roc", "eqodds", "ceop"]:
        rows.append([STAGE[fm], NICE[fm], cellvals(base, fm)])
    for mech, nm in MECH.items():
        arm = f"ours-{base}-{mech}"
        if (d.arm == arm).any():
            rows.append([r"\textbf{in-gen}", nm, cellvals(arm, "none")])

    def f(t):
        a, c, ds, tpr = t
        if np.isnan(a) or np.isnan(c):
            return r"n/a"
        # A cell is uninterpretable when the classifier learned nothing: either it
        # fails to beat the majority-class rate (tolerance absorbs seed-to-seed
        # jitter in that rate, .7543-.7588 on Adult), or it is a degenerate
        # constant predictor, which scores perfectly fair at zero recall.
        s = f"{a:.3f}/{c:.3f}"
        degenerate = c <= MAJ[ds] + 0.002 or (not np.isnan(tpr) and tpr <= 0.01)
        return r"\textcolor{gray}{" + s + "}" if degenerate else s

    L = [r"\begin{table}[t]", r"\centering\footnotesize",
         r"\caption{\textbf{Where to intervene, on identical data.} " + base.upper() +
         r" synthetic training data, real held-out test set, XGBoost, 5 seeds, run"
         r" inside the unmodified benchmark harness of \citet{angelozzi2026where}."
         r" Entries are $|$parity gap$|$\,/\,accuracy; lower gap and higher accuracy"
         r" are better. The eight mitigators act on the released table or on the"
         r" model trained from it; our three act inside the generator."
         r" \textcolor{gray}{Grey} marks a cell that taught the classifier nothing"
         r" --- accuracy at or below the majority-class rate ($0.755$ Adult, $0.534$"
         r" COMPAS) or a degenerate constant predictor (zero recall) --- so its parity"
         r" gap is not interpretable. \texttt{n/a}: the"
         r" mitigator returned an undefined metric in every seed.}",
         r"\label{tab:baselines}", r"\begin{tabular}{llcccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{Adult} & \multicolumn{2}{c}{COMPAS} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"stage & intervention & $\varepsilon{=}1$ & $\varepsilon{=}10$ & "
         r"$\varepsilon{=}1$ & $\varepsilon{=}10$ \\", r"\midrule"]
    prev = None
    for st, nm, vals in rows:
        if prev is not None and st != prev:
            L.append(r"\addlinespace[2pt]")
        prev = st
        L.append(f"{st} & {nm} & " + " & ".join(f(t) for t in vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT} (base={base}, classifier={clf})")


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:] or []))
