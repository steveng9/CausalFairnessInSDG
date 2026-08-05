# Causal Fairness in SDG — findings package

Two files, and the notebook is self-contained:

| File | What to do with it |
|---|---|
| `CausalFairnessInSDG_findings.ipynb` | The report. Open in Colab or Jupyter, Run All. |
| `CausalFairnessInSDG_results.zip` | The results data. **Do not unzip it** — just drop it next to the notebook. |

## Running it

**In Colab:** upload the notebook, then drag `CausalFairnessInSDG_results.zip`
into the file pane. Run All. The setup cell finds the zip, unpacks it, and reports where it
loaded from — there is nothing to configure. If it cannot find a zip it will prompt you to
upload one.

**Locally:** put both files in the same directory and run
`jupyter notebook CausalFairnessInSDG_findings.ipynb`. Needs only `pandas`, `numpy`,
`matplotlib`.

Every section heading is collapsible — click the triangle to fold away what you don't need.

## What is in the results zip

| File | What it is |
|---|---|
| `data/all_runs.csv` | One row per experiment run (3,000 rows × 187 columns). 2,040 are **active**; 960 are superseded and used only by Section 4.3c. |
| `data/real_baselines.csv` | Reference accuracies measured on the **real** data. |
| `data/adult_ctgan_epoch_sweep.csv` | First sweep, diagnosing the DECAF+CTGAN training-length bug (Section 4.3b). |
| `data/gan_epoch_sweep_all.csv` | Full sweep: all 8 GAN method x dataset combinations, 114 fits (Section 4.3c). |
| `data/newdata_runs.csv` | First SNAKE and SBO results — 216 runs, marginal methods, 1 seed (Section 7.8). |
| `data/newdata_baselines.csv` | Real-data reference accuracies for SNAKE and SBO. |

`all_runs.csv` columns split into three groups:

**Which cell of the grid this is** — `dataset`, `sdg_method`, `fairness_mechanism`, `epsilon`,
`role_config`, `seed`, `batch`, `protected_attrs`, `admissible_attrs`, `outcome_attr`,
`status`, `max_epochs`, `git_commit`, `superseded`.

**`superseded` is the column to filter on.** The GAN family was re-run at corrected epoch
settings; the old rows are retained so the correction is auditable, but every analysis except
Section 4.3c uses `superseded == False`. The notebook does this in its setup cell and exposes
the two views as `runs` (active) and `superseded`.

**What was measured** — fidelity (`tvd_1way`, `tvd_2way`, `avg_correlation_diff`), utility
(`downstream_accuracy_mlp` / `_lr` / `_rf`), fairness (`fairness_gap`, `cond_fairness_gap`,
and the per-attribute `dp_gap__*`, `tprb__*`, `tnrb__*`, `cdp_gap__*`, `ctprb__*`, `ctnrb__*`),
privacy accounting (`spent_epsilon`, `noise_multiplier`, `dp_steps`).

**The Distributional Fairness axis** — every fairness metric appears twice: bare (scored
against the real holdout) and prefixed `synthref__` (scored against the synthetic data
itself). This is DECAF's Definition 4; Sections 7.3 and 7.7 of the notebook analyse it.

Section 2 of the notebook defines all of these in plain language, with worked examples.

## Batches included

| Batch tag | Runs | What |
|---|---|---|
| `overnight-2026-07-30` | 1,200 | MST, PrivBayes, PrivSyn, DECAF (Adult, COMPAS) |
| `gan-backbones-2026-07-30` | 840 | DECAF+CTGAN, DECAF+DP-GAN, DECAF+DP-CTGAN (Adult, COMPAS) |
| `newdata-2026-08-03` | 216 | MST, PrivBayes, PrivSyn on **SNAKE and SBO**, 1 seed. Complete, 0 failures. |
| `gan-retuned-2026-08-04` | 960 | The GAN family re-run at the swept epoch settings. **Complete** — Adult 480/480, COMPAS 479/480. |

`all_runs.csv` holds the three Adult/COMPAS batches; the new datasets are in
`newdata_runs.csv` because their metric columns differ (different protected attributes).
`gan-retuned-2026-08-04` supersedes the GAN rows of the first two batches — see `superseded`.

316 runs are `partial` — the generator collapsed to a single outcome class, so downstream
metrics are undefined. In the active set all of them are DECAF+DP-GAN (168 of 360, 47%); that
collapse rate is itself the main result of Section 8.

One run failed outright: `compas/prefair/decaf_dpctgan/dp/eps=1/seed=3`, with *"the least
populated class in y has only 1 member"*. That is a near-total collapse rather than an
infrastructure fault — the same phenomenon as the `partial` rows, just past the point where a
stratified split is possible. It should arguably be reclassified `partial` so it counts as a
collapse instead of dropping out of the denominator.

## Known issues affecting these numbers

1. **Adult `education-num` encoding.** Only object-dtype columns get ordinal-coded, so
   `education-num` keeps its raw UCI 1–16 values against a declared domain of 16. Consistent
   across every method, so comparisons are internally valid, but absolute Adult numbers will
   shift when it is fixed. The equivalent bug is fixed for the two new datasets (`_recode` in
   `data/datasets.py`); Adult is deliberately left alone so the published numbers stay
   comparable to each other.
2. ~~**DECAF+CTGAN on Adult was trained 10× too little**~~ — **fixed and re-run.** Epochs had
   been tuned by minimising **1-way TVD**, a metric that is non-monotone in training here and
   blind to relationships, on a single seed. It selected 30 epochs; the answer is **600**. The
   corrected batch moves Adult/DECAF+CTGAN from accuracy **0.569 to 0.766** against a majority
   baseline of 0.747 — from 18 points below the trivial predictor to 2 above it — while also
   *improving* fidelity. Sections 4.3, 4.3b, 4.3c.
3. **Every other GAN cell was tuned the same wrong way — swept, 6 of 8 changed, all re-run.**
   The full sweep (114 fits, 0 failures, `gan_epoch_sweep_all.csv`) is in Section 4.3c. Two
   things came out of it. First, **300 epochs was itself unstable**: re-running Adult/CTGAN at
   300 on GPU rather than CPU — same seeds, only the RNG stream differs — produced a seed with
   a synthetic positive rate of 0.83 against a true 0.247 and accuracy 0.282. 600 epochs is
   where all three seeds converge. Second, there is **no global epoch count**: on COMPAS
   fidelity degrades with training, and under DP the optimum moves with ε. The re-run is batch
   `gan-retuned-2026-08-04`, now complete and analysed in Sections 4–8.

   Two consequences worth knowing before reading the fairness sections. **COMPAS fairness gaps
   went up substantially** (DECAF+CTGAN 0.075 → 0.270) — this is not a regression but the
   trivially-fair failure mode resolving: the old models had ~zero lift and so had no gap to
   show. And **COMPAS DP-GAN now collapses in 100% of runs at ε=1**, so that cell has no usable
   setting at all.
4. **AIM is missing.** Implemented and registered in the codebase, but absent from both
   batches.
5. **`spent_epsilon` is only logged for DP-GAN**, not DP-CTGAN.

## Experiment-tracking audit

The results database was audited for this revision. Clean: no duplicate configurations, no
orphaned metrics, metric counts consistent within every (batch, method, n-protected) group,
zero failed runs inside either reported batch.

Three defects were found and fixed in the code (they affect *provenance*, not the numbers):

1. **`git_commit` recorded HEAD only, ignoring uncommitted changes.** 227 GAN rows are tagged
   with a commit that predates the existence of the code that produced them. Now records
   `<sha>-dirty:<hash-of-diff>`.
2. **GAN rows recorded no training settings.** You could not tell from the database what epoch
   count produced a row — precisely the variable issue #2 above showed to be decisive. All GAN
   rows now carry their effective `max_epochs`/`batch_size`/etc.
3. **12 DECAF/COMPAS rows in `overnight-2026-07-30` predate the DECAF output-head fix**
   (commit `3da19b0`) and have a 1-way TVD of 0.048 against 0.152 for the other 108 rows in
   the same cell. Mixed generator code inside one batch; those rows should be re-run or
   excluded before publication.
