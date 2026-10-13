# Result tables

Everything a figure or table in the paper reads comes from this directory.
`facct/` is the FAccT benchmark; the files beside this README are the earlier
(AI4Good workshop) results and are kept unchanged.

`facct/` is regenerated from `sweeps/` by one command, safe to run while
sweeps are running:

```
python scripts/collect_results.py
```

`facct/MANIFEST.json` records when it last ran, the commit, and how complete
each sweep was at that moment. **Check it before quoting a number.**

## `facct/` — what each file answers

| File | Content | Produced by | Write-up |
|---|---|---|---|
| `facct-2026-10-09_runs.csv` | **The main sweep.** One row per run: 11 datasets x {MST, PrivBayes, PrivSyn at eps 1/10/1000; TabPFN causal family, not private} x {none, FTU, DP, CF} x 3 role splits x 5 seeds, random forest as the reference classifier. 6,600 runs. | `scripts/sweep.py`, `scripts/run_experiments.py` | `notes/2026-10-09-facct-sweep.md` |
| `facct-cut025/050/075-2026-10-12_runs.csv` | Partial-cut axis for the TabPFN family: each mechanism's cut applied to 25 / 50 / 75% of generated rows. With `none` (0%) and the main sweep (100%) this is the gap-vs-AUC curve. 9 datasets, 540 runs each. | same, `--set cut_strength=...` | notes §12 |
| `facct-tune-e30/e100*_runs.csv` | Epoch-tuning pilots for the GAN and diffusion backbones (no mechanism, 3 seeds). Not paper results; they justify training settings. | same | notes §9 |
| `real_baselines.csv` | Train-on-real reference per dataset and seed (MLP, RF, LR accuracy / F1 / AUC, base rate). | `scripts/compute_real_baselines.py` | notes §1, §6, §12 |
| `tabpfn_conditionals.csv` | Held-out log-loss of every TabPFN conditional against the marginal and a count table, seed 0. Soundness check for the TabPFN family. | `scripts/validate_tabpfn_conditionals.py` | notes §11 |
| `fairpfn/results.csv` | FairPFN downstream baseline: one row per dataset x seed x context source x context size x protected attribute x model (`fairpfn`, `rf`, `rf_unaware`). | `scripts/fairpfn_baseline.py` | `notes/2026-10-11-fairpfn-baseline.md` |
| `fairpfn/summary_by_cell.csv` | The same, mean / std / seed count per cell. | `scripts/summarize_fairpfn_baseline.py` | same |
| `fairpfn/generator_comparison.csv` | The main sweep's RF AUC and DP gap per dataset x generator x mechanism x role split x attribute, laid out to sit beside FairPFN's rows. | same | same |

### Reading a `*_runs.csv`

- Cell: `dataset`, `sdg_method`, `fairness_mechanism`, `role_config`, `epsilon`
  (empty for non-private methods), `seed`. `protected_attrs` / `admissible_attrs`
  spell the role split out. `status` is `done`, or `partial` when the synthetic
  outcome collapsed to one class.
- Utility: `downstream_auc_rf` (also `_accuracy_`, `_f1_macro_`; `_mlp`, `_lr`),
  trained on synthetic, scored on the real holdout.
- Fidelity: `tvd_1way`, `tvd_2way`, `jsd_1way`, `avg_correlation_diff`.
  Privacy proxies: `dcr_ratio`, `exact_match_rate`.
- Fairness, per protected attribute `<a>`, on the real holdout: `dp_gap__<a>`
  (signed; take the absolute value), `cdp_gap__<a>` (conditional on the
  admissible set), `tprb__<a>`, `tnrb__<a>` and their conditional versions.
  The same names prefixed `synthref__` are measured on the synthetic table.
  Protected attributes are binarised as modal value vs. everyone else.
- FairPFN files: quote the `_matched` columns, not the 0.5-threshold ones
  (see its notes, "Decision rule").

### Datasets

`acsincome_ca`, `acsincome_tx`, `acspubcov_ca`, `acspubcov_tx` (ACS PUMS 2018),
`hmda_ca`, `hmda_tx` (HMDA 2018 mortgage applications), `cdc_diabetes`,
`snake` (CPS), `sbo` (Survey of Business Owners), `adult`, `compas`.
Loaders and encodings: `src/causal_fairness_sdg/data/`; graphs:
`data/causal_graphs.py`; role splits: `scripts/run_experiments.py::ROLE_CONFIGS`;
graph figures: `manuscript/figs/graphs/`.

### What is not in git

Per-shard SQLite files (`sweeps/<batch>/db/`), every synthetic table
(`sweeps/<batch>/synth/<shard>/run_<id>.parquet`, about 1 GB), FairPFN's raw
scores (`sweeps/facct-2026-10-09/fairpfn_baseline/preds/`) and the raw data
(`data/`) stay on the machine that ran them. A new metric can be computed
from the synthetic tables without rerunning a generator. The databases also
hold each run's full settings (e.g. the TabPFN checkpoint hash), which the
flat CSVs drop.

## Earlier results (AI4Good workshop paper)

| File | Content |
|---|---|
| `all_runs.csv` | The workshop sweeps, MLP classifier (3,480 rows; 2,520 active). Read by `manuscript/make_figs.py`. |
| `rf_2026_09_01.csv` | The September random-forest sweep at eps = 10 on Adult and COMPAS (1,080 runs); Table 1 of the workshop paper. See `report/RESULTS-FINAL-2026-09-02.md`. |
| `real_baselines.csv` | Train-on-real reference, Adult and COMPAS. |
| `newdata_runs.csv`, `newdata_baselines.csv` | First SNAKE / SBO runs (August). |
| `adult_ctgan_epoch_sweep.csv`, `gan_epoch_sweep_all.csv` | GAN epoch sweeps. |
