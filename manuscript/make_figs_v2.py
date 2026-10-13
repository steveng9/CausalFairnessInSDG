"""Figures built from the `rf-2026-09-01` sweep (epsilon=10 / non-private,
random-forest downstream model, expanded metric suite).

Two outputs:
  figs/results.pdf  -- fidelity | utility | fairness-vs-epsilon, where the
                       utility panel is now TSTR AUC against the
                       train-on-real reference rather than accuracy lift.
  figs/radar.pdf    -- one hexagon per generator family, six axes covering
                       fidelity, utility, fairness gain and the utility the
                       fairness cut costs.

The epsilon panel still comes from the first sweep (`all_runs.csv`), which is
the only one that swept epsilon; everything else comes from the new one.
"""
import os

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
NEW = os.path.join(HERE, os.pardir, "report", "data", "rf_2026_09_01.csv")
OLD = os.path.join(HERE, os.pardir, "report", "data", "all_runs.csv")
REAL = os.path.join(HERE, os.pardir, "report", "data", "real_baselines.csv")

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.0, "axes.labelsize": 6.8, "axes.titlesize": 7.4,
    "legend.fontsize": 6.5, "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "grid.linewidth": 0.4,
    "grid.color": "#d8d7d2", "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.012, "pdf.fonttype": 42,
})

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUT = "#0b0b0b", "#8a8880"
FAMC = {"marginal": BLUE, "gan": ORANGE, "diff": AQUA}
FAMN = {"marginal": "Marginals-based", "gan": "Causal GAN", "diff": "Causal diffusion"}
FAM = {"mst": "marginal", "privbayes": "marginal", "privsyn": "marginal",
       "decaf": "gan", "decaf_ctgan": "gan", "decaf_dpgan": "gan",
       "decaf_dpctgan": "gan", "decaf_diffusion": "diff",
       "decaf_dpdiffusion": "diff"}
# dagger marks the generators that carry a differential-privacy guarantee
NICE = {"mst": "MST$^\\dagger$", "privbayes": "PrivBayes$^\\dagger$",
        "privsyn": "PrivSyn$^\\dagger$", "decaf": "DECAF",
        "decaf_ctgan": "+CTGAN", "decaf_dpgan": "+DP-GAN$^\\dagger$",
        "decaf_dpctgan": "+DP-CTGAN$^\\dagger$", "decaf_diffusion": "+Diff",
        "decaf_dpdiffusion": "+DP-Diff$^\\dagger$"}
ORDER = ["mst", "privbayes", "privsyn", "decaf", "decaf_ctgan", "decaf_dpgan",
         "decaf_dpctgan", "decaf_diffusion", "decaf_dpdiffusion"]
FAMS = ["marginal", "gan", "diff"]


def load():
    r = pd.read_csv(NEW, low_memory=False)
    r = r[r.status.isin(["done", "partial"])].copy()
    r["family"] = r.sdg_method.map(FAM)
    r = r[r.family.notna()]
    real = pd.read_csv(REAL).groupby("dataset").mean(numeric_only=True)
    # AUC as a fraction of what a classifier trained on the *real* table gets,
    # so 1.0 means the release is as useful as the data it stands in for.
    r["auc_ratio"] = r.downstream_auc_rf / r.dataset.map(real.trtr_auc_rf)
    r["gap"] = r[[c for c in r.columns if c.startswith("dp_gap__")]].abs().max(axis=1)
    return r, real


def _paired(r, mech, value):
    """Value under `mech` beside the value under no mechanism, for runs that
    match on everything but the mechanism."""
    key = ["dataset", "sdg_method", "epsilon", "role_config", "seed"]
    piv = r.pivot_table(index=key, columns="fairness_mechanism", values=value, dropna=False)
    if "none" not in piv or mech not in piv:
        return pd.DataFrame()
    out = piv[["none", mech]].dropna()
    out["family"] = [FAM[i[1]] for i in out.index]
    return out


# =====================================================================
def radar(r):
    base = r[r.fairness_mechanism == "none"]
    axes_spec = [
        ("1-way fidelity", lambda f: 1 - base[base.family == f].tvd_1way.mean()),
        ("2-way fidelity", lambda f: 1 - base[base.family == f].tvd_2way.mean()),
        ("association\npreserved",
         lambda f: 1 - base[base.family == f].avg_correlation_diff.mean()),
        ("utility\n(AUC vs real)", lambda f: base[base.family == f].auc_ratio.mean()),
    ]
    vals = {f: [fn(f) for _, fn in axes_spec] for f in FAMS}

    # Fairness gain (CF): matched-pair share of the baseline gap removed, on
    # pairs where there was a gap to remove at all.
    pg = _paired(r, "cf", "gap")
    for f in FAMS:
        g = pg[(pg.family == f) & (pg["none"] > 0.05)] if len(pg) else pg
        vals[f].append(float((1 - g["cf"] / g["none"]).clip(0, 1).mean()) if len(g) else np.nan)
    # Absolute post-cut fairness under CF and DemPar: complement of the mean
    # final parity gap in that family.
    for mech in ("cf", "dp"):
        for f in FAMS:
            s = r[(r.fairness_mechanism == mech) & (r.family == f)]
            vals[f].append(1 - s.gap.mean() if len(s) else np.nan)
    labels = [n for n, _ in axes_spec] + [
        "fairness gain\n(CF)", "fairness after\nCF", "fairness after\nDemPar"]

    # Fairness axes are anchored at 0 so the reader sees absolute magnitude
    # (a family with a low gain does not get pushed to the centre just because
    # the other two happen to be higher). The four baseline axes stay
    # rescaled between the weakest and strongest family so the shape
    # compares them; the true endpoints for every axis are printed beside
    # each label so nothing is hidden.
    M = np.array([vals[f] for f in FAMS], dtype=float)
    lo, hi = np.nanmin(M, axis=0), np.nanmax(M, axis=0)
    FAIRNESS = {4, 5, 6}  # gain (CF), after CF, after DemPar
    lo_adj = lo.copy()
    for i in FAIRNESS:
        lo_adj[i] = 0.0
    span = np.where(hi - lo_adj < 1e-9, 1.0, hi - lo_adj)
    S = 0.18 + 0.74 * (M - lo_adj) / span
    # Use lo_adj for the printed range so the label matches what the ring shows.
    lo = lo_adj

    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.concatenate([ang, ang[:1]])
    fig, ax = plt.subplots(figsize=(3.35, 3.05), subplot_kw={"polar": True})
    for i, f in enumerate(FAMS):
        v = np.concatenate([S[i], S[i][:1]])
        ax.plot(closed, v, color=FAMC[f], lw=1.3, label=FAMN[f], zorder=3)
        ax.fill(closed, v, color=FAMC[f], alpha=0.11, zorder=2)
    ax.set_xticks(ang)
    ax.set_xticklabels(
        [f"{l}\n{lo[i]:.2f}\u2013{hi[i]:.2f}" for i, l in enumerate(labels)],
        fontsize=6.1)
    ax.set_ylim(0, 1.06)
    ax.set_yticklabels([])
    ax.tick_params(pad=1.5)
    ax.grid(color="#d8d7d2", lw=0.4)
    ax.spines["polar"].set_linewidth(0.5)
    ax.spines["polar"].set_color("#d8d7d2")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              handlelength=1.1, columnspacing=1.0)
    fig.savefig(os.path.join(OUT, "radar.pdf"))
    plt.close(fig)

    tbl = pd.DataFrame(M, index=[FAMN[f] for f in FAMS],
                       columns=[l.replace("\n", " ") for l in labels])
    print("\n=== radar values (raw, higher is better) ===")
    print(tbl.round(3).to_string())
    return tbl


# =====================================================================
def results_panel(r, real):
    base = r[r.fairness_mechanism == "none"]
    agg = base.groupby(["dataset", "sdg_method"]).agg(
        tvd2=("tvd_2way", "mean"), auc=("downstream_auc_rf", "mean")
    ).reset_index()

    old = pd.read_csv(OLD, low_memory=False)
    old = old[~old.superseded.astype(bool)]
    old["family"] = old.sdg_method.map(FAM)

    fig = plt.figure(figsize=(5.7, 1.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.24, 1.06, 0.98], wspace=0.24)
    ax0, ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    y = np.arange(len(ORDER))

    for ax, col, lab in zip([ax0, ax1], ["tvd2", "auc"],
                            ["(a) 2-way TVD (lower = truer)",
                             "(b) AUC (higher = better)"]):
        for ds, hatch, alpha in [("adult", None, 1.0), ("compas", "////", 0.5)]:
            d = agg[agg.dataset == ds].set_index("sdg_method").reindex(ORDER)
            off = -0.19 if ds == "adult" else 0.19
            v = d[col].to_numpy()
            # AUC is only meaningful relative to chance, so those bars are
            # drawn from 0.5 rather than from 0.
            left = 0.5 if col == "auc" else 0.0
            ax.barh(y + off, v - left, left=left, height=0.36,
                    hatch=hatch, alpha=alpha, color=[FAMC[FAM[m]] for m in ORDER],
                    linewidth=0)
            # Cells where no bar is drawn (NaN value) get a visible marker so
            # the reader does not mistake absence for zero.
            for yy, vv in zip(y + off, v):
                if not np.isfinite(vv):
                    x_mark = (0.505 if col == "auc" else 0.005)
                    ax.text(x_mark, yy, "× collapsed", fontsize=4.6,
                            color="#a03030", va="center", ha="left")
        ax.set_yticks(y)
        ax.set_yticklabels([NICE[m] for m in ORDER] if ax is ax0 else [])
        ax.invert_yaxis()
        ax.set_title(lab, loc="left")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    # the two reference lines that make panel (b) readable; labels are placed
    # just above the top bar (y is inverted so a small negative sits above)
    ax1.axvline(0.5, color=INK, lw=0.7, ls=":")
    ax1.text(0.5, -0.55, "chance", fontsize=5.4, color=INK, ha="center", va="bottom")
    for ds, ls in [("adult", "--"), ("compas", "-.")]:
        v = real.trtr_auc_rf[ds]
        ax1.axvline(v, color=MUT, lw=0.7, ls=ls)
        ax1.text(v, -0.55, f"real ({ds[:1].upper()})", fontsize=5.4, color=MUT, ha="center", va="bottom")
    ax1.set_xlim(0.41, 0.93)
    # widen top margin on both bar panels so reference labels clear the title
    for a in (ax0, ax1):
        a.set_ylim(8.5, -0.9)

    # (c) FINAL parity gap remaining after CF, across epsilon. Matches the
    # §4.4 claim that the cut equalises the gap across DP budgets.
    key = ["dataset", "sdg_method", "epsilon", "role_config", "seed"]
    piv = old.pivot_table(index=key, columns="fairness_mechanism",
                          values="fairness_gap", dropna=False)
    lv = piv.index.get_level_values
    for ds, ls, mk in [("adult", "-", "o"), ("compas", "--", "s")]:
        for f in ["marginal", "diff"]:
            xs, ys = [], []
            for eps in [1.0, 10.0, 1000.0]:
                m = (lv("dataset") == ds) & (lv("epsilon") == eps) & \
                    np.array([FAM.get(s) == f for s in lv("sdg_method")])
                sel = piv[m]
                if "cf" not in sel:
                    continue
                v = sel["cf"].dropna()
                if len(v):
                    xs.append(eps); ys.append(v.mean())
            if xs:
                ax2.plot(xs, ys, ls, marker=mk, ms=2.8, lw=1.0, color=FAMC[f])
    ax2.set_xscale("log")
    ax2.set_xticks([1, 10, 1000])
    ax2.set_xticklabels(["1", "10", "1000"])
    ax2.set_xlabel(r"privacy budget $\varepsilon$", labelpad=-2)
    ax2.set_title("(c) parity gap after CF", loc="left")
    ax2.set_ylabel("parity gap", labelpad=1)
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)
    ax2.set_xlim(0.6, 9000)
    ax2.set_ylim(0, None)

    # 5-element key at the bottom: three families + two datasets.
    handles = [
        Patch(facecolor=FAMC["marginal"], edgecolor="none", label="marginals-based"),
        Patch(facecolor=FAMC["gan"], edgecolor="none", label="causal GAN"),
        Patch(facecolor=FAMC["diff"], edgecolor="none", label="causal diffusion"),
        Patch(facecolor="#808080", edgecolor="none", label="Adult"),
        Patch(facecolor="#c8c8c8", edgecolor="none", hatch="////", label="COMPAS"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.13), ncol=5,
               fontsize=6.4, handlelength=1.4, columnspacing=1.4, frameon=False)

    fig.savefig(os.path.join(OUT, "results.pdf"))
    plt.close(fig)
    print("\n=== panel values ===")
    print(agg.pivot_table(index="sdg_method", columns="dataset",
                          values=["tvd2", "auc"]).reindex(ORDER).round(3).to_string())


if __name__ == "__main__":
    r, real = load()
    print(f"{len(r)} runs; per family: "
          f"{r[r.fairness_mechanism=='none'].family.value_counts().to_dict()}")
    radar(r)
    results_panel(r, real)
    print("\nwrote figs/radar.pdf and figs/results.pdf")
