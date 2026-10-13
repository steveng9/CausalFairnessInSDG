# FAccT sweep — what was launched on 2026-10-09 and what comes next

Plan this implements: `manuscript/reviews/FAccT-2027-plan.md` §1f (datasets) and §1h
("one sweep, one classifier"). Tooling: `scripts/sweep.py` (see README, "Running a sweep").

## 1. Datasets

| Dataset | Status | Notes |
|---|---|---|
| `acsincome_ca`, `acsincome_tx` | **new**, in the sweep | ACS PUMS 2018 1-Year via `folktables`; row counts after the folktables filter match the published ones (CA 195,665; TX 135,924); 50k-row seeded subsample like SNAKE/SBO. Loader, encoding and codebook: `data/acs.py`, `data/processed/*.json`. |
| `snake`, `sbo` | already wired, in the sweep | Both pass the encoding contract check. |
| `adult`, `compas` | not in this sweep | Appendix continuity only; `rf-2026-09-01` covers them at eps=10. |
| ACSPublicCoverage, HMDA, Law School | **not started** | Next datasets, in that order. PublicCoverage reuses `data/acs.py` (same PUMS files, already downloaded) and needs its own DAG. |

Encoding decisions on ACSIncome that a co-author should look at (all in the `data/acs.py`
module docstring): `HISP` added as a Hispanic-origin indicator (not in folktables'
ACSIncome); `POBP` replaced by the Census `WAOB` recode; `OCCP` grouped to 25 major
groups; `SCHL`/`RELP` collapsed; fixed public bin edges for age and hours. Changing any
of these means bumping `ENCODING_VERSION` and re-running the ACS shards.

The ACSIncome DAG (`data/causal_graphs.py::ACSINCOME_DAG`) and its three role splits
(`run_experiments.py::ACSINCOME_ROLE_CONFIGS`) are **our construction**, mirroring
SNAKE's. They only affect the torch (DECAF-family) generators and the role splits; the
marginal generators learn their own graph.

Train-on-real reference (RF, 5 seeds), `sweeps/facct-2026-10-09/real_baselines.csv`:

| dataset | positive rate | accuracy | AUC |
|---|---|---|---|
| acsincome_ca | 0.411 | 0.796 | 0.872 |
| acsincome_tx | 0.362 | 0.787 | 0.857 |
| snake | 0.585 | 0.689 | 0.744 |
| sbo | 0.518 | 0.880 | 0.949 |

## 2. What is running

**`facct-2026-10-09`** — the marginal half of the unified sweep.
{acsincome_ca, acsincome_tx, snake, sbo} × {MST, PrivBayes, PrivSyn} × {none, FTU, DP, CF}
× eps {1, 10, 1000} × 3 role splits × 5 seeds = **2,160 runs**, RF as the reference
classifier, every synthetic table saved. 60 shards, 14 at a time, pinned to cores 32-47
at nice 19. Started 23:14 UTC; roughly 7 hours.

**`facct-tune-e30-2026-10-09`, `facct-tune-e100-2026-10-09`** — epoch-tuning pilots for
the six torch generators on acsincome_ca, snake and sbo (no mechanism, eps=1000, seeds
0-2; 162 runs each). Queued by `sweeps/chain-tuning-after-main.sh` to start on the same
16 cores when the main sweep's scheduler exits, **on CPU**. Kill that script to stop them.

```
python scripts/sweep.py status --batch facct-2026-10-09
tail sweeps/facct-2026-10-09/scheduler-*.log
python scripts/merge_sweep_dbs.py sweeps/facct-2026-10-09/db -o report/data/facct-2026-10-09.csv
```

More cores: start a second scheduler on other cores (it only takes unlocked shards), e.g.
`python scripts/sweep.py run --batch facct-2026-10-09 --workers 8 --cpus 16-23`.
A free GPU: run the tuning pilots there instead of waiting for the CPU queue, e.g.
`python scripts/sweep.py run --batch facct-tune-e30-2026-10-09 --workers 4 --cpus 24-31 --gpu 1`.
Stop a sweep: `pkill -f "sweep.py run --batch <batch>"`, then `pkill -f "batch <batch>"`.

## 3. Not yet done (in order)

1. **Torch half of the unified sweep.** The per-dataset epoch counts for the new
   datasets in `sdg/decaf_variants.py` and `DECAF_SETTINGS` are placeholders copied from
   SNAKE. Read the two tuning pilots, set the counts, then add the six torch methods to
   `sweeps/facct-2026-10-09/sweep.json` and `run` again (with `--gpu` if one is free).
   CPU cost at the placeholder counts, measured on acsincome_ca with one thread:
   roughly 12 core-hours per (dataset, seed), so ~240 core-hours for the whole torch half.
2. **Plan §1a/§1b pieces that need new code**: the cut-strength knob, the flip-test
   metric, post-cut AUC floor. The saved synthetic tables cover the metrics; the knob
   needs new generation.
3. **ACSPublicCoverage, HMDA, Law School** loaders and DAGs.

## 4. Caveats

- The code was not committed before launch. Each run row carries a `git_commit`
  fingerprint of the form `33a7139-dirty:<hash>`, and `sweeps/<batch>/code/` holds the
  diff and the new files for the fingerprint each shard was started under. Scripts were
  edited after the first 14 shards started (only `--set`/`--only` plumbing and docs; no
  generator or metric code), so later rows of those shards show a newer fingerprint than
  the code they ran.
- Shards run single-threaded with `CUDA_VISIBLE_DEVICES=""` unless `--gpu` is given.
- Smoke check on acsincome_ca at eps=10: MST / PrivBayes / PrivSyn reach RF AUC
  0.83 / 0.82 / 0.83 without a mechanism (train-on-real 0.87); PrivSyn under the DemPar
  mechanism fell to AUC 0.49, the dead-classifier case from review point R1.

## 5. Update, 2026-10-10 04:55 UTC

- `facct-2026-10-09` finished: 2,160/2,160 runs, 0 failed, 0 collapsed, in about 5.5 hours.
- `facct-tune-e30-2026-10-09` started on CPU (14 shards, cores 32-47) at 04:52.
- One GPU worker (GPU 1, shared with daniilf, ~600 MB) is running the 100-epoch pilot's
  slow generators (`decaf_ctgan`, `decaf_diffusion`, `decaf_dpctgan`, `decaf_dpdiffusion`):
  `sweeps/facct-tune-e100-2026-10-09/scheduler-gpu1-*.log`. The CPU queue will pick up
  whatever of that pilot the GPU worker has not taken once the 30-epoch pilot ends.
  Stop the GPU worker with `pkill -f "sweep.py run --batch facct-tune-e100.*--gpu 1"`
  and then `pkill -f "batch facct-tune-e100-2026-10-09"`.

## 6. Update, 2026-10-10 05:30 UTC — three more datasets

Added to `facct-2026-10-09` (marginal generators, same grid; 1,620 more runs, started
05:23 with 8 workers on cores 32-47, sharing them with the tuning pilots):

| Dataset | Rows × cols | Positive rate | Real-data RF AUC | What it adds |
|---|---|---|---|---|
| `acspubcov_ca` | 50k × 20 | 0.37 | 0.72 | health access; 8 binary columns; eligibility rules make "admissible" statutory |
| `acspubcov_tx` | 50k × 20 | 0.19 | 0.80 | same graph in a state that did not expand Medicaid |
| `cdc_diabetes` | 50k × 22 | 0.14 | 0.79 | 16 binary columns, rare outcome, no race (sex + income/education protected) |

Both were added on the understanding that one may be dropped later. For review:

- PublicCoverage encoding and roles: `data/acs.py` (module docstring,
  `ACS_PUBLIC_COVERAGE`), DAG `causal_graphs.py::ACSPUBCOV_DAG` (60 edges), role splits
  `run_experiments.py::ACSPUBCOV_ROLE_CONFIGS`. One split treats disability as protected
  instead of admissible.
- CDC Diabetes: `data/cdc_diabetes.py`, DAG `CDC_DIABETES_DAG` (93 edges; the outcome is
  kept a sink although clinically some arrows run the other way), role splits under
  `ROLE_CONFIGS["cdc_diabetes"]`.
- PublicCoverage is a hard prediction task even on real data (CA AUC 0.72). In the smoke
  run MST at eps=10 reached only 0.59 on CA, so post-cut AUC floors will bite here.
- The torch backbones run these three on SNAKE's settings (`sdg/base.py::UNTUNED_DATASETS`);
  they are not in the tuning pilots yet.
- ACSIncome was rebuilt after the loader refactor and is identical to the tables the
  finished shards used.

## 7. Open decision, 2026-10-10 — which DAG the GAN/diffusion runs use

Not decided. DECAF's own experiments use *discovered* graphs (PC / FGES with
protected attributes as roots); ours are hand-drawn. Settle this **before
adding the torch methods to `sweeps/facct-2026-10-09/sweep.json`** (§3).
Full write-up: `manuscript/reviews/FAccT-2027-plan.md` §1d, "OPEN DECISION".

## 8. TODO added 2026-10-10 — foundation-model causal family

A third DECAF-style family (TabPFN or similar as each column's conditional,
same DAG and edge cuts as GAN/diffusion). Not started. See
`manuscript/reviews/FAccT-2027-plan.md` §1i.

## 9. Update, 2026-10-10 20:00 UTC — pilots read, new-dataset pilot, TabPFN family

**Pilots (`facct-tune-e30-`, `facct-tune-e100-2026-10-09`), mean AUC, no mechanism:**
100 epochs is at least as good as 30 nearly everywhere and never collapses (two fits
collapsed at 30). Diffusion is best and cheapest. DP-CTGAN is at chance on all three
datasets at both lengths and is parked (Steven, 2026-10-10). DP-GAN gets worse with more
training on SNAKE. Not yet applied to the settings tables.

**`facct-tune-e100-newds-2026-10-10`** — the same 100-epoch pilot on acspubcov_ca,
acspubcov_tx, cdc_diabetes; five generators (no DP-CTGAN), seeds 0-2, 45 shards, 14 CPU
workers on cores 32-47. Started 19:18 UTC.

**`decaf_tabpfn` in `facct-2026-10-09`** — third causal family: DECAF's generator with
TabPFN-3.5 as each attribute's conditional (`sdg/decaf_tabpfn.py`). 35 shards (7 datasets
x 5 seeds, 12 runs each), one worker on GPU 1, started 19:51 UTC, about 9 minutes a shard
on ACSIncome. Hand-drawn DAGs.

```
python scripts/sweep.py status --batch facct-2026-10-09
tail sweeps/facct-2026-10-09/scheduler-tabpfn-gpu1-*.log
# restart / resume (any environment; shards run under sdg_tabpfn by themselves):
python scripts/sweep.py run --batch facct-2026-10-09 --workers 1 --cpus 32-47 --gpu 1 --methods decaf_tabpfn
```

- Settings on every row's `extra_params`: model version, ensemble size, temperature,
  device, tabpfn and torch versions, checkpoint file name and SHA-256. Counters
  (`incontext_*`) are in `metrics`.
- First shard (acsincome_ca, seed 0), no mechanism: AUC 0.868 against 0.872 on real data
  and 0.845 for diffusion in the pilot; two-way TVD 0.054 against 0.087.
- `sweeps/facct-2026-10-09/tabpfn_conditionals.csv`: held-out log-loss of each
  conditional vs. a count table. ACSIncome-CA is in; `sweeps/chain-tabpfn-validation.sh`
  adds the other six datasets when the GPU worker finishes.
- Not bit-reproducible across GPUs: TabPFN's probabilities move in the fourth decimal
  with batching (measured), which can flip an occasional sampled value.
- The main `sdg` environment cannot run this method and says so before writing any row.
- A plain `sweep.py run --batch facct-2026-10-09` without `--gpu` skips these shards.

## 10. Update, 2026-10-11 — TabPFN family complete; FairPFN baseline

`decaf_tabpfn` finished 420/420. A FairPFN downstream baseline was run on the
same splits and on the no-mechanism releases of `mst` and `decaf_tabpfn`:
see `notes/2026-10-11-fairpfn-baseline.md`.

## 11. Update, 2026-10-12 — TabPFN conditionals validated on all seven datasets

`sweeps/chain-tabpfn-validation.sh` never ran: it waited for no process to match the
TabPFN scheduler's command line, and the shell that launched it carried that same text,
so the wait could not end. Found 2026-10-12, waiters killed, validation run directly on
GPU 1 (one worker, 25 minutes, no errors).

`sweeps/facct-2026-10-09/tabpfn_conditionals.csv` now has 94 rows: every non-root column
of every dataset, seed 0 (35,000 context rows, 15,000 held-out rows, TabPFN-3.5,
8 ensemble members, temperature 1). Each row gives held-out log-loss (nats) of the
column given its DAG parents under three predictors: the training marginal, a
count-based conditional table over the parents, and TabPFN.

- TabPFN is at or below the count table on all 94 columns. The two non-improvements are
  ties to four decimals (`VET1` in SBO, `DREM` in PublicCoverage-TX), both two-parent
  binary columns.
- Mean gain over the count table per dataset, non-outcome and outcome columns together:
  acsincome_ca 0.054, acsincome_tx 0.049, snake 0.040, sbo 0.031, cdc_diabetes 0.021,
  acspubcov_ca 0.011, acspubcov_tx 0.011.
- The gain is concentrated in the outcome column, which has the most parents and
  therefore the sparsest count table:

| Dataset | Outcome | Parents | Held-out parent configs | Marginal | Count table | TabPFN |
|---|---|---|---|---|---|---|
| acsincome_ca | `high_income` | 10 | 13,121 | 0.677 | 0.611 | 0.388 |
| acsincome_tx | `high_income` | 10 | 12,843 | 0.655 | 0.589 | 0.398 |
| snake | `high_income` | 11 | 14,500 | 0.680 | 0.662 | 0.562 |
| sbo | `high_receipts` | 15 | 14,283 | 0.692 | 0.654 | 0.265 |
| acspubcov_ca | `public_coverage` | 12 | 5,537 | 0.660 | 0.607 | 0.557 |
| acspubcov_tx | `public_coverage` | 12 | 5,017 | 0.479 | 0.398 | 0.349 |
| cdc_diabetes | `Diabetes_binary` | 15 | 12,376 | 0.406 | 0.393 | 0.311 |

For the paper: this is the soundness check for the TabPFN causal family (plan §1i). It
supports one sentence of the form "each conditional predicts held-out rows at least as
well as the empirical conditional table, and much better where parent configurations
are sparse". It is one seed, and it says nothing about joint fidelity, which the
sweep's TVD columns cover.

Reproduce (sdg_tabpfn env, about 17 s a column):
```
CUDA_VISIBLE_DEVICES=1 taskset -c 32-47 nice ~/miniconda3/envs/sdg_tabpfn/bin/python -u \
  scripts/validate_tabpfn_conditionals.py --datasets <names> --seed 0 --device cuda:0 \
  --out sweeps/facct-2026-10-09/tabpfn_conditionals.csv
```

## 12. Update, 2026-10-12/13 — HMDA added; Adult and COMPAS folded in; partial-cut batches

### HMDA (the lending dataset)

`hmda_ca`, `hmda_tx`: 2018 HMDA loan/application register from the FFIEC data browser,
same two states and year as the ACS tables. Loader, filters, encoding and the caveats:
`src/causal_fairness_sdg/data/hmda.py` (module docstring). Codebook and per-filter row
counts: `data/processed/hmda_*_2018_n50000_s20260803_v1.json`.

- Population: decided applications (originated, approved-not-accepted, denied) for a
  first-lien, closed-end, non-commercial loan on a one-unit, site-built, owner-occupied
  home, purchase or refinance.
- Outcome `approved` (favourable = positive): 0.84 in CA, 0.86 in TX.
- 16 attributes + outcome, 50k-row seeded subsample of 508,821 (CA) / 382,657 (TX).
- Rows after each filter, CA: 1,192,073 downloaded -> 683,898 in population -> 546,374
  with the first applicant's sex/race/ethnicity/age -> 508,821 with underwriting and
  tract variables. TX: 762,502 -> 501,651 -> 417,192 -> 382,657.
- Approval rate by group in the 50k tables: CA White 0.844, Asian 0.850, Black 0.749,
  AIAN 0.721, Pacific Islander 0.748; Hispanic 0.825 vs 0.842; female 0.833 vs 0.840;
  falls with age from 0.89 (25-34) to 0.72 (>74). TX Black 0.792 vs White 0.872;
  Hispanic 0.836 vs 0.873; female 0.847 vs 0.870.
- Train-on-real reference (5 seeds), `sweeps/facct-2026-10-09/real_baselines_added-1012.csv`:
  RF AUC 0.781 (CA), 0.796 (TX); accuracy 0.876 / 0.891 against a majority baseline of
  0.839 / 0.862. Same file has Adult (RF AUC 0.863) and COMPAS (0.697).
- DAG: `causal_graphs.py::HMDA_DAG`, 17 nodes, 65 edges, hand-drawn, one graph for both
  states. Role splits `run_experiments.py::HMDA_ROLE_CONFIGS`:
  `ecoa_underwriting_adm` (sex, race, ethnicity, age protected; income, dti, ltv,
  loan_amount, loan_term admissible), `sex_only_broad_adm`, `race_narrow_adm` (race and
  ethnicity protected; only dti and ltv admissible). The three mechanisms cut 4 / 10 / 16
  of the outcome's 16 parents on the first split, so FTU, CF and DP are well separated.
- Tract minority share and tract income are never admissible (the redlining channel).

Caveats a co-author should look at before this goes in the paper:
1. **No credit score in public HMDA.** A race-approval dependence that survives the
   columns here is not by itself evidence of disparate treatment. Standard caveat for
   every public-HMDA analysis; say it in the paper.
2. About 20% of in-population applications have no reported applicant demographics and
   are dropped; they are denied more often (CA 23% vs 17%). About 4% more lack DTI or
   LTV and are also denied more often. The table understates denial.
3. Race is the first race the first applicant reported; co-applicant demographics are
   not used beyond a yes/no co-applicant flag.
4. The DAG and role splits are ours, like every other graph here; the open DAG decision
   (plan §1d) applies.
5. Torch backbones would run HMDA on SNAKE's untuned settings (`UNTUNED_DATASETS`).
6. **The sweep's standard gap is small on HMDA, and hides the gap that matters.**
   Measured 2026-10-13 with a random forest trained on real rows (seed 0):
   - The classifier predicts "approved" for 92-93% of the holdout (true rate 84-86%),
     so there is little room for a gap in 0/1 predictions.
   - The sweep binarises each protected attribute as modal value against everyone
     else. For race that is White against the rest, and in California the rest is
     mostly Asian applicants, who are approved slightly more often than White
     applicants. Predicted-approval gap, majority vs rest: race -0.013 (CA), -0.048
     (TX); ethnicity -0.016 / -0.019; sex -0.014 / -0.016; age -0.032 / -0.043.
   - Black vs White is the contrast the lending literature reports: predicted approval
     0.857 vs 0.922 in CA and 0.871 vs 0.936 in TX; actual approval 0.755 vs 0.849 and
     0.786 vs 0.872.
   First `decaf_tabpfn` shard (hmda_ca, seed 0) agrees: unmitigated gaps of 0.01-0.03
   in absolute value, so the mechanisms have little to remove on this metric. Before
   HMDA carries a headline, compute a Black-vs-White (and Hispanic-vs-not) gap, and a
   gap in predicted probability or at a matched rate, from the saved synthetic tables
   (`synth/<shard>/run_<id>.parquet`); no generator needs rerunning. The same
   binarisation is used on every dataset, so this is worth checking on ACS race too.

### What was launched (2026-10-12 23:58 UTC onward)

- `facct-2026-10-09` grew by `hmda_ca`, `hmda_tx`, `adult`, `compas`: marginal generators
  (2,160 runs, 14 CPU workers on cores 32-47) and `decaf_tabpfn` (240 runs). Adult and
  COMPAS were only in `rf-2026-09-01` at eps = 10 before; they are in for the appendix
  and so FairPFN can be run on their synthetic releases. Sweep total is now 6,600 runs.
- GPU 1, one process at a time, `sweeps/chain-gpu1-2026-10-12.sh` (log beside it):
  1. `decaf_tabpfn` on the four added datasets;
  2. TabPFN conditional validation on them (appends to `tabpfn_conditionals.csv`);
  3. `facct-cut050-2026-10-12`;
  4. FairPFN baseline extended to the added datasets and to PrivBayes / PrivSyn
     releases as context (appends to `fairpfn_baseline/results.csv`, then re-summarises);
  5. `facct-cut025-2026-10-12`, `facct-cut075-2026-10-12`.
- Partial-cut batches (plan §1a, first family): `decaf_tabpfn` with a new setting
  `cut_strength` (default 1.0 = unchanged). At strength s the mechanism's cut is applied
  to a share s of the generated rows, so the outcome is drawn from
  `(1 - s) p + s p_cut`. Nine datasets (no Adult/COMPAS) x 5 seeds x 3 splits x 4
  mechanisms = 540 runs per batch. With `none` (s = 0) and the main sweep (s = 1) this
  gives five points per mechanism on the gap-AUC curve. Code:
  `sdg/_decaf_incontext.py::InContextCausalSampler.generate`, test
  `tests/test_decaf_incontext.py::test_partial_cut_interpolates_between_no_cut_and_the_full_cut`.
  The `none` rows in these batches do not depend on the setting and repeat the main
  sweep's as a check.

```
python scripts/sweep.py status --batch facct-2026-10-09
python scripts/sweep.py status --batch facct-cut050-2026-10-12
tail sweeps/chain-gpu1-2026-10-12.log
```

### Why the DP cut costs so much accuracy (read from the code, not yet tested)

`fairness/base.py::biased_edges_from_paths` cuts each protected-to-outcome path at the
edge nearest the outcome (DECAF's convention). Under DP no path is blocked, so every
outcome parent that descends from a protected attribute is shuffled: on HMDA's first
split that is all 16 parents, on ACSIncome nearly all. The synthetic outcome is then
close to independent of every feature, which is the AUC 0.60 seen for `decaf_tabpfn`
+DP. Cutting at the protected end instead would keep the mediator-outcome
relationships in the table, but a classifier trained on it and scored on real rows
would still pick up the real association between mediators and the protected
attribute, so the gap on the real holdout would not close. That trade-off, and the
partial-cut curve above, are what to compare FairPFN against. A cut-position variant
was not implemented; it changes the mechanism's definition and needs a decision.
