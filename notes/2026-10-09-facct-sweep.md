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

## 13. Update, 2026-10-13 — causal families rerun on DECAF-style discovered graphs

**Correction.** Earlier notes and code said DECAF takes its graph as given. It does
not, on real data. Re-read of arXiv 2110.12884:

- §5: "We assume the underlying DGP's graph G is known; otherwise, G needs to be
  approximated first using any causal discovery method, see Section 6."
- §6: "In both experiments, the ground-truth DAG is unknown. We use causal discovery
  to uncover the underlying DAG."
- Adult (§6.1, App. D.2): the graph "discovered and presented by Zhang et al." (2017);
  the Discussion says it was found with the PC algorithm in Tetrad.
- Credit Approval (§6.2, App. D.3): "found using the Fast Greedy Equivalence Search
  (FGES) with the pycausal library. We provide the prior knowledge that age and
  ethnicity are root nodes to the FGES algorithm." Score and parameters not given.
- §5.1 / App. B: any graph in the discovered Markov equivalence class is compatible
  with the data and can be used; FTU and DP need only some independence statements
  right, CF needs edge directions.
- In both graphs the paper draws (Figs. 6, 7) the target is a sink.
- Edge removal (Cor. 1-3, App. D.2) is as we implement it: DP removes every edge into
  the target from a parent that has the protected attribute as an ancestor, CF spares
  the admissible parents, FTU removes the direct edge. With protected attributes as
  roots and the outcome a sink, our directed-path rule equals the paper's
  d-separation rule for FTU and DP.

**What was built** (commit `b9eb89c`).

- `scripts/discover_graphs.py`: Tetrad's FGES (causal-cmd 1.4.1 from Maven Central,
  kept at `~/external/tetrad/`, runs on the system Java 8), BDeu score, Tetrad's default
  parameters (equivalent sample size 10, structure prior 1, max degree 1000), on each
  seed's **training split**. Knowledge: age and the birth-fixed protected attributes
  are roots (`causal_graphs.DISCOVERY_ROOTS`), the outcome is a sink. The CPDAG is
  oriented by a Dor-Tarsi consistent extension (needed for one edge, on Adult only).
  About one second per graph; re-running gives identical graphs.
- Graphs: `src/causal_fairness_sdg/data/discovered_graphs/fges_<dataset>.json`, one
  per training split, keyed by a hash of the training table. A generator asked for
  `graph=fges` on rows it has no graph for raises an error.
- Generators: setting `graph` (`hand` default, `fges`), for `decaf`, the GAN/diffusion
  variants and `decaf_tabpfn`. Recorded in `extra_params` only when set; hand-graph
  behaviour is unchanged. Each run's edges are in its `graph_edges` table as before.
- `scripts/summarize_graphs.py` -> `report/data/facct/graphs_summary.csv`.

**Three choices here are ours, not the paper's**: the outcome declared a sink; discovery
on the training rows only, per seed (the paper used the whole table, once); and the
BDeu score with default parameters (the paper does not say). Adult uses the same FGES
recipe as every other dataset, not Zhang et al.'s graph, so Adult is still not a
like-for-like reproduction of DECAF's Adult numbers.

**The discovered graphs are much sparser than the hand-drawn ones** (means over seeds
and role splits; cut = outcome parents the mechanism shuffles):

| dataset | edges hand / FGES | outcome parents hand / FGES | FTU cut | CF cut | DP cut |
|---|---|---|---|---|---|
| acsincome_ca | 34 / 12 | 10 / 2 (OCCP, WKHP) | 2.3 / 0 | 5.0 / 0.3 | 8.3 / 2.0 |
| acsincome_tx | 34 / 12 | 10 / 2 | 2.3 / 0 | 5.0 / 0.3 | 8.3 / 2.0 |
| acspubcov_ca | 60 / 44 | 12 / 4 | 3.0 / 1.3 | 6.3 / 2.3 | 10.3 / 3.1 |
| acspubcov_tx | 60 / 43 | 12 / 3.6 | 3.0 / 0.3 | 6.3 / 1.1 | 10.3 / 2.4 |
| adult | 26 / 17 | 9 / 2 (education, relationship) | 0.7 / 0 | 4.3 / 1.1 | 8.7 / 2.0 |
| cdc_diabetes | 93 / 51 | 15 / 3.6 | 2.3 / 0 | 9.0 / 1.5 | 14.0 / 3.6 |
| compas | 16 / 10 | 4 / 2 (age_cat, priors_count) | 1.3 / 0.3 | 1.7 / 0.7 | 3.3 / 1.3 |
| hmda_ca | 65 / 37 | 16 / 3 | 2.3 / 0 | 8.0 / 1.1 | 14.3 / 2.5 |
| hmda_tx | 65 / 30 | 16 / 3 | 2.3 / 0 | 8.0 / 0.7 | 14.3 / 2.0 |
| sbo | 68 / 46 | 15 / 2 (PAYROLL_NOISY, SECTOR) | 2.0 / 0 | 5.3 / 0.5 | 14.0 / 1.9 |
| snake | 39 / 22 | 11 / 2 (citistat, gradeatn) | 1.7 / 0.3 | 4.0 / 0.3 | 8.7 / 1.3 |

Consequences to expect in the results:

- **FTU cuts nothing in 135 of 165 (dataset, seed, split) cells**: FGES finds a direct
  protected-to-outcome edge only on PublicCoverage (HISP, DIS), SNAKE (citistat) and
  COMPAS (age_cat, when age is protected). In those cells FTU rows equal the `none`
  rows. CF cuts nothing in 75 of 165, DP in 10 of 165.
- The outcome is generated from 2-4 parents, so the `none` rows may lose accuracy
  against the hand-graph runs. BDeu scores a full conditional table, which penalises a
  wide parent set on columns with up to 25 levels; a different score or penalty would
  give denser graphs. Untested.
- The hand-drawn graphs put a direct protected-to-outcome edge in every dataset by
  convention. The data-driven graphs mostly do not support it.

**Running.**

- `facct-fges-2026-10-13`: 11 datasets x 5 seeds x {decaf, decaf_ctgan,
  decaf_diffusion, decaf_tabpfn} x 3 splits x 4 mechanisms = 2,640 runs, 220 shards.
  Non-private families only: a graph learned from the training rows without noise
  would void the DP variants' guarantee, so `decaf_dpgan` / `decaf_dpdiffusion` are
  left out pending a decision.
- Torch shards: 14 CPU workers on cores 32-47, started 03:49 UTC. Epochs are the
  settings tables as of `b9eb89c` (100 for the SNAKE-like datasets, per the pilots).
- `decaf_tabpfn` shards: one worker on GPU 1, queued behind the existing GPU 1 queue
  (`sweeps/chain-gpu1-fges-2026-10-13.sh` waits on that queue's PID).
- The torch families have no hand-graph main-sweep rows (that half was held for this
  decision). The hand-graph counterpart exists for `decaf_tabpfn` only.

```
python scripts/sweep.py status --batch facct-fges-2026-10-13
tail sweeps/facct-fges-2026-10-13/scheduler-cpu.out sweeps/chain-gpu1-fges-2026-10-13.log
python scripts/discover_graphs.py            # regenerate the graphs
python scripts/summarize_graphs.py           # regenerate the comparison table
```

## 14. Queued, 2026-10-13 — privacy axis for the DP marginal generators

Steven, 2026-10-13: next priority after the main portability results is how the privacy
budget affects the causal fairness mechanisms, across families, mechanisms, datasets and
role splits. First step: fill the epsilon grid for MST, PrivBayes and PrivSyn.

- `facct-eps-a-2026-10-13`: epsilon 0.1, 100 (3,960 runs, about 5 hours on 14 workers).
- `facct-eps-b-2026-10-13`: epsilon 0.3, 3, 30, 300 (7,920 runs, about 10 hours).
- Same 11 datasets, 3 role splits, 4 mechanisms, 5 seeds and RF classifier as
  `facct-2026-10-09`, which holds epsilon 1, 10, 1000; the three concatenate.
- Both start by themselves when the `facct-fges` torch shards finish
  (`sweeps/chain-cpu-eps-2026-10-13.sh`, waiting on that scheduler's PID).
- Not covered: the DP causal families (`decaf_dpgan`, `decaf_dpdiffusion`), which wait
  on the graph decision in §13, and AIM.
- Analysis to do afterwards (plan §1e): separate DP noise shrinking the baseline gap
  from the mechanism's own effect at each epsilon.

## 15. Queued, 2026-10-13/14 — downstream baseline, MAMA-MIA, hand-graph torch, DP causal

Steven, 2026-10-13: four additions. Nothing new was started while the CPU share
(cores 32-47, 14 workers) was full; everything below sits in one queue,
`sweeps/chain-cpu-2026-10-14.sh`, which starts when `facct-eps-a` ends.

**State of the graph sources before this (the question behind item 3).** Nothing was
replaced. The discovered-graph batch is a separate batch; the hand-graph sweep
`facct-2026-10-09` simply never contained the torch families (they were held for the
graph decision, §13).

| Family | Hand-drawn graphs (`facct-2026-10-09`) | Discovered graphs (`facct-fges-2026-10-13`) |
|---|---|---|
| TabPFN causal | done, 660 runs | queued on GPU 1 behind the partial-cut batches |
| GAN (`decaf`), CTGAN, Diffusion | **not run until now** (pilots only) | done, 1,977 of 1,980 |
| DP GAN / CTGAN / Diffusion | not run until now | not planned (non-private graph voids the guarantee) |
| MST, PrivBayes, PrivSyn | not applicable: they select their own graph | same |

**1. Angelozzi & Arcolezi's eight mitigators on the FAccT datasets** (`baselines/`).
- `facct_bench.py export` writes our splits and a manifest; `facct_bench.py cell`
  lays one arm's saved synthetic tables out in their format, calls their unmodified
  `benchmark()` through `run_bench_facct.py` (env `dpfair`), and deletes the copies;
  `launch_facct_grid.sh` fans cells out; `facct_bench.py collect` writes
  `report/data/facct/angelozzi_mitigators.csv`.
- One benchmark dataset per (dataset, protected attribute), 43 in all, attribute
  binarised majority-versus-rest, privileged = the group with the higher real rate of
  outcome 1. Arms: `<generator>-none` (their mitigators on our no-mechanism release:
  the downstream baseline) and `<generator>-<mechanism>@<role split>` (our mechanism
  scored by their protocol). XGBoost, their pairing for MST. Seeds 0-4, the sweep's
  splits; their loader halves our holdout into calibration and test.
- Grid: 1,192 cells from the four generators now in the main sweep (172 no-mechanism,
  1,020 mechanism). A cell is 20 passes of 9 models (5 seeds x real + 3 epsilons) and
  the trial cell took 19 minutes under full contention, far more than on their Adult
  file (86 s for 11 passes), so the no-mechanism arms run first, MST's mechanism arms
  next, and the remaining mechanism arms last of all.
- Trial cell (`acsincome_ca__SEX`, `mst-none`): real-data parity gap 0.132 at accuracy
  0.817; on the epsilon-1 release 0.079 unmitigated, 0.025 after calibrated equalised
  odds, 0.039 after reject-option, at accuracy 0.75-0.76 against 0.764. As on their
  Adult file, grid-search reduction changes nothing and LFR collapses to one class;
  here XGBoost then refuses to fit and their harness logs the LFR row as an error.
- The upstream clone had disappeared from `baselines/`; re-cloned (MIT licence,
  git-ignored, 2.3 GB). `facct_data/` and `facct_output/` are git-ignored.
- Differences from the Adult/COMPAS comparison: their own `mst`/`aim` releases do not
  exist for these datasets, so the baseline arm is our generator's no-mechanism table;
  no half-epsilon control.

**2. MAMA-MIA on every saved table** (`scripts/mama_mia.py`).
- Both flavours from `~/SyntheticData_MIA/conduct_attacks.py`, reimplemented against
  our tables: `mst` (weighted marginal ratios) and `privbayes` (conditional ratios).
- Focal points come from each run's recorded graph. For the causal families that is
  the DAG minus the edges the mechanism cut. For the conditional flavour MST's forest
  is oriented from its first column; PrivSyn's graph has cycles, so only the marginal
  flavour runs there.
- Shadow modelling without new runs: the other seeds of the same cell are the same
  algorithm on other 70% samples of the same population, so their recorded graphs give
  the focal-point frequencies (4 shadows; thresholds as in the paper). An `oracle` row
  (the run's own graph) is recorded beside every `shadow` row.
- Threat model: auxiliary data = the whole dataset (overlapping, the paper's main
  setting); targets = all holdout rows plus as many training rows.
- Output `sweeps/<batch>/mama_mia/<shard>.csv`: AUC, TPR at 10% / 1% / 0.1% FPR,
  membership advantage. About 0.15 s a run.
- Trial on five shards (seed 0): ACSIncome CA, MST, marginal flavour AUC 0.513 /
  0.535 / 0.538 at epsilon 1 / 10 / 1000; PrivBayes, conditional flavour 0.51 / 0.58 /
  0.63 (shadow), 0.52 / 0.59 / 0.66 (oracle); TabPFN causal 0.506 (marginal), 0.517
  (conditional). The mechanisms move these by under 0.01 on that dataset.

**3 and 4. Torch causal families on the hand-drawn graphs, and their DP variants.**
- Added to `facct-2026-10-09`: `decaf`, `decaf_ctgan`, `decaf_diffusion` (1,980 runs,
  about 6 hours) and `decaf_dpgan`, `decaf_dpctgan`, `decaf_dpdiffusion` at epsilon 1,
  10, 1000 (5,940 runs; rough estimate 15 hours from the September Adult timings).
- The DP variants run on the hand graphs only: a fixed, data-independent graph keeps
  the guarantee. Their epochs for the ACS, CDC and HMDA datasets are SNAKE's, untuned.

**Queue order** (`sweeps/chain-cpu-2026-10-14.sh`): Angelozzi no-mechanism arms;
MAMA-MIA; hand-graph torch; Angelozzi MST mechanism arms; DP causal; the three SNAKE
PrivBayes shards of `facct-eps-a` that were killed (signal 9, 63 runs); Angelozzi
no-mechanism arms for the new families; MAMA-MIA again; `facct-eps-b`; remaining
Angelozzi mechanism arms. `facct-eps-b` (epsilon 0.3, 3, 30, 300, the "if time permits"
half) was moved to the back with a `hold` key in its `sweep.json`, a new option of
`sweep.py run`; at the observed rate it needs well over a day, not the 10 hours
estimated in §14.

```
tail sweeps/chain-cpu-2026-10-14.log baselines/logs/facct_grid.log sweeps/mama_mia.log
python scripts/sweep.py status --batch facct-2026-10-09
```

Known failures to revisit: `facct-fges` `hmda_tx` / `decaf` / seed 4, DP and CF cuts
(3 runs, the synthetic outcome has a class with one member).

## 16. Queued, 2026-10-14 — TabFairGAN comparison arm

Why: it is the generator behind Eitan et al. ("fair synthetic data does not necessarily
lead to fair models"), its own paper already shows the table-versus-people gap (COMPAS:
discrimination score 0.258 in the real data, 0.009 in its synthetic table, 0.208 for a
classifier trained on that table and tested on real people; Rajabi & Garibay 2022,
Table II), and it imposes parity through a training loss with no causal graph. It
separates "cutting edges" from "any in-table parity constraint".

- `src/causal_fairness_sdg/sdg/tabfairgan.py`, method `tabfairgan`, in
  `facct-2026-10-09`. It is the published package (`tabfairgan` 0.1.0 from PyPI, MIT
  licence, installed in the `sdg` env), used unmodified; the module only adapts it.
- Mechanisms `none` (no fairness phase) and `dp` (its parity loss) only. A method can
  now declare `mechanisms`, and the grid gives it only those
  (`run_experiments._mechanisms_of`). 5 seeds x 3 role splits x 2 = 30 runs per
  dataset, 330 in all, not private (NULL epsilon).
- One binary S per role split: the first attribute of `PROTECTED_PRIORITY[dataset]`
  that the split protects. That is sex everywhere, race on COMPAS (as in the paper),
  RAC1P on ACSPubCov `race_only_broad_adm`, and race on HMDA `race_narrow_adm`. A
  many-valued S is binarised majority-vs-rest for training. Generated "rest" rows get a
  value drawn from the training non-majority distribution. Recorded as `tabfairgan_s`
  and `tabfairgan_s_under` on each row. Y_desire = 1, S_under = the group with the lower
  real rate of y = 1; its loss is signed, so this pairing closes the gap.
- Settings are the paper's Table III. COMPAS: 40 + 30 epochs, lambda 2.2. All others:
  Adult's 170 + 30, lambda 0.5. Batch 256, n_crit 4, Adam 2e-4. All columns are
  categorical; a constant numeric column is added for training only, because the
  package fails without one.
- Cost: 2.3–5 s an epoch on one thread (Adult, SNAKE, ACSIncome, CDC), so 10–17 min a
  model. About 120 models (the `none` model and each S's `dp` model are reused across
  role splits), plus scoring: about 30 core-hours. Smoke test on COMPAS and Adult
  (3 epochs, scratch DB): 12/12 ok.
- Queued in step 3 of `sweeps/chain-cpu-2026-10-14.sh`, beside the hand-graph torch
  families. The chain waiter was restarted to pick it up (now PID 2186270, still
  waiting on the eps chain).
- Downstream: step 7's re-export gives it a `tabfairgan-none` Angelozzi arm (the eight
  mitigators on an unconstrained GAN release), and step 10 scores
  `tabfairgan-dp@<split>`. MAMA-MIA has no graph to read for it, so it gets every
  1- and 2-way marginal as focal points (marginal flavour only;
  `mama_mia.GRAPH_FREE_METHODS`).
- Not done (cheap follow-ups on tables already saved, if wanted): Krchova-style
  outcome re-drawing, and dropping the protected attribute before training as the FTU
  comparison.

## 17. Fix, 2026-10-15 — PrivBayes degree k at small epsilon; epsilon-0.1 runs redone

- Symptom: in facct-eps-a every SNAKE and SBO PrivBayes shard was SIGKILLed twice and
  given up. SBO shards had run 10–14 h without finishing one epsilon-0.1 run (9 min at
  epsilon 1); SNAKE epsilon-0.1 runs took 51 min. Peak RSS between samples was about
  1.3 GB, and the kernel log is not readable here, so out-of-memory is likely but not
  proven.
- Cause: the vendored `calculate_k` (from DataSynthesizer) root-finds from k = d/2
  when k = 3 misses the usefulness target. That lands on a spurious root and
  *raises* k as epsilon falls. At epsilon 0.1 (epsilon/2 for structure) on our training
  splits the old k was: SBO 12 (of 25 columns), CDC 11, ACSPubCov 10, HMDA 8, SNAKE 7,
  Adult 7, and 3 for ACSIncome and COMPAS. At epsilon 0.3 it was SBO 12 and COMPAS 2.
  Every other epsilon gave 3.
- Fix (`sdg/_privbayes_vendor.py`): the largest k <= 3 whose usefulness exceeds the
  target, else 1. That is what PrivBayes Lemma 3 asks. New k at epsilon 0.1: 1 or 2
  everywhere; at epsilon 0.3: SBO 2, COMPAS 1; at epsilon >= 1 unchanged (3), so the
  main batch, the cut batches and all workshop results are unaffected. No collected
  result used PrivBayes below epsilon 1.
- Redone: all 569 PrivBayes epsilon-0.1 rows of facct-eps-a (550 done, 19 pending,
  all datasets) were deleted. A copy of each shard DB from before the deletion and the
  549 synthetic tables are in `sweeps/facct-eps-a-2026-10-13/invalid-privbayes-k-2026-10-15/`
  (git-ignored). The epsilon-100 runs are kept.
- Queue: the two stuck SBO shards were killed and the eps chain ended. Step 0 of
  `sweeps/chain-cpu-2026-10-14.sh` now reruns facct-eps-a PrivBayes (14 workers)
  before the Angelozzi and MAMA-MIA steps, so those never see the old tables.
  facct-eps-b (still on hold, step 9) runs with the fixed k.

## 18. Extra capacity, 2026-10-15 17:43 UTC

The box was nearly idle (4 busy cores, GPU 1 empty), so `sweeps/extra-2026-10-15.sh`
started two more schedulers beside the CPU chain (which keeps cores 32-47, 14 workers):
- GPU 1, 1 worker: the DP causal families of facct-2026-10-09 (chain step 5),
  log `sweeps/extra-2026-10-15-gpu1.log`. These rows record `device=cuda:0`.
- Cores 20-31, 10 workers: facct-eps-b (chain step 9, on hold, run with
  `--even-if-held`), log `sweeps/extra-2026-10-15-cpu.log`.
Shard locks keep them apart from the chain, whose steps 5 and 9 skip what is finished.
Cores 0-19 stay free for other users.
- 2026-10-16 05:47: GPU 1 runs DP causal shards 3-4x faster than a CPU worker
  (e.g. ACSIncome DP-CTGAN 27 vs 110 min), but one single-threaded worker left it at
  about 0% utilisation. A second GPU 1 scheduler (`extra-2026-10-15.sh gpu1b`, 2 workers,
  cores 17-19) brought it to about 60%. Each worker uses under 1.3 GB of GPU memory.
- 2026-10-17 19:00: every sweep is finished (facct-2026-10-09 at 14849/14850, all
  facct-cffix-* complete). Only chain step 10 is left (Angelozzi, 3108 cells, ~2300 to
  go, ~42 cells/h on 14 workers). `sweeps/extra-grid-2026-10-17.sh` adds 14 workers on
  cores 16-31 walking the grid in reverse (`facct_data/grid_reverse_2026-10-17.txt`);
  cells skip once `.done` exists, so the two launchers meet in the middle.
- 2026-10-17 19:04: added facct-cffix-cut075-2026-10-17 (CF arm, 7 datasets, 105 runs, GPU 0 + cores 4-15); cut075 had been missed from the cffix set.

## 19. DP diffusion, 2026-10-18 17:24 UTC — CF re-run, privacy axis, epoch checks

Steven chose to put the spare GPU on the DP-diffusion family (our novel backbone) rather
than partial cuts for diffusion (which do not bear on its novelty). GPU 0 only; GPU 1
stays free for Daniil. Launcher `sweeps/dpdiff-2026-10-18.sh`, two lanes sharing GPU 0:
lane a (6 workers, cores 4-11, log `sweeps/dpdiff-2026-10-18-a.log`) and lane b
(3 workers, cores 12-15, log `-b.log`). Cores 0-3 stay free.
- `facct-cffix-dpdiff-2026-10-18` (lane a): CF arm of decaf_dpdiffusion, epsilon 1, 10,
  1000, the 9 datasets of facct-cffix-hand, all 45 shards (about 14 had already run on
  2379bb9, re-run anyway so one rule holds). Supersedes every decaf_dpdiffusion CF row of
  facct-2026-10-09. decaf_dpgan / decaf_dpctgan CF rows are still not re-run.
- `facct-eps-dpdiff-2026-10-18` (both lanes, after the above): epsilon 0.1, 3, 100, all
  mechanisms, 11 datasets, 5 seeds (55 shards, 1,980 runs). With facct-2026-10-09 this
  gives six points 0.1-1000. Settings unchanged (100 epochs, fixed across epsilon).
- `facct-tune-dpdiff-e30-2026-10-18` (lane b): 30 epochs, epsilon 1000, no mechanism,
  seeds 0-2, the 8 datasets the 2026-10-09/10 pilots did not compare at 30 epochs.
- `facct-tune-dpdiff-e300-2026-10-18` (lane b): 300 epochs, same protocol, adult, snake,
  acsincome_ca, hmda_ca. Both checks compare with the 100-epoch epsilon-1000 'none' rows
  of facct-2026-10-09. They check the setting in use; changing it would invalidate the
  existing DP-diffusion rows.
- Cost from facct-2026-10-09 (CPU workers): DP diffusion took 160-900 CPU-min a shard
  (COMPAS 11), about 130 CPU-h per epsilon over 11 datasets; GPU runs were 3-4x faster.
- 2026-10-19: CF re-run, both epoch checks, and epsilon 3 and 100 finished. Epsilon 0.1
  failed on all 55 shards: the accountant's Renyi orders stopped at 127, which floors
  epsilon at about log(1/delta)/126 = 0.16 at delta 1e-9. `_decaf_dp._ORDERS` now runs to
  1024 (also used by decaf_dpgan / decaf_dpctgan); noise multipliers for epsilon >= 0.3
  are identical to before (checked at three sample-rate/step settings). Epsilon 0.1
  relaunched 18:12 UTC with the same launcher, logs `sweeps/dpdiff-2026-10-19-eps01-{a,b}.log`.
  The failed rows stay in the shard DBs with status 'failed'.
- 2026-10-20, epoch checks read (decaf_dpdiffusion, epsilon 1000, no mechanism, seeds 0-2,
  mean over role splits; 100 = facct-2026-10-09 rows, 30 on acsincome_ca/snake/sbo from
  facct-tune-e30-2026-10-09). RF AUC: 100 epochs >= 30 everywhere (by 0.00-0.025), and 300
  is no better than 100 (within 0.006 on the four datasets run). Two-way TVD is mixed:
  30 epochs is closer on ACSPubCov, CDC, HMDA (by 0.01-0.03), 100 is closer on Adult (0.134
  vs 0.177); 300 is equal or worse than 100. Training loss keeps falling to 300. Verdict:
  100 epochs is a defensible setting (utility saturates there; more epochs buys nothing),
  with a mild fidelity cost against 30 on some datasets. Not re-tuned.

## 20. CF re-runs carried through, 2026-10-20 16:20 UTC

- `facct-cffix-dpgan-2026-10-20`: CF arm of decaf_dpgan and decaf_dpctgan, epsilon 1, 10,
  1000, the 9 cffix-hand datasets, all 90 shards (810 runs). GPU 0, 10 workers on cores
  4-15, `sweeps/cffix-dpgan-2026-10-20.sh`. Supersedes every DP-GAN / DP-CTGAN CF row of
  facct-2026-10-09. With it every causal family's CF rows on the hand graphs come from a
  facct-cffix-* batch.
- Angelozzi mitigators on the new CF tables. `facct_bench.py export` gained `--merge`
  (update only the exported arms of each manifest; a plain export would have dropped
  every other arm). The CF arms of decaf_tabpfn, decaf, decaf_ctgan, decaf_diffusion
  (facct-cffix-hand) and decaf_dpdiffusion (facct-cffix-dpdiff) were re-pointed: 340
  cells, all `-cf@` arms, none lost (diffed against
  `facct_data/backup-pre-cffix-2026-10-20/`). Old-rule outputs moved to
  `facct_output/invalid-cf-oldrule-2026-10-20/` (not collected). Re-scored on cores 16-47,
  28 jobs, `sweeps/angelozzi-cffix-2026-10-20.sh`, which then runs `collect`.
- `sweeps/angelozzi-cffix-dpgan-2026-10-20.sh` waits on both PIDs, then does the same for
  the DP-GAN / DP-CTGAN CF arms (grid `facct_data/grid_cffix_dpgan_2026-10-20.txt`) and
  collects again. CDC and SBO keep their CF arms: the new rule does not change their cuts.
- 2026-10-21: Angelozzi grid closed at 3,087 of 3,108 cells (`angelozzi_mitigators.csv`,
  798,473 rows). Two ACSIncome DP-CTGAN / DP-GAN CF cells had died inside the LFR
  mitigator at 28 parallel jobs (likely memory) and passed at 6 jobs. The remaining 21
  cells are all Adult decaf_dpgan arms (none, ftu, dp, cf): their harness needs all 5
  seeds at one epsilon, and on Adult DP-GAN's synthetic income collapses to one class in
  some seeds at every epsilon (e.g. dp@prefair keeps seeds 1,4 at eps 1; 0,1,2,4 at 10;
  1,2,3 at 1000). Decision (Steven): report Adult DP-GAN as not scorable under the
  Angelozzi protocol and point to the collapse; no per-epsilon seed subsets. The
  launcher's `exit=` lines said 0 for these failures; trust the `.done` markers.

## 21. Final experiments before writing, 2026-10-22 06:20 UTC

Steven asked for every remaining backlog item that needs compute, so results are ready
when writing starts. GPU 1 and cores 0-3 stay free for Daniil. Closed without compute:
the 4 "missing" runs (1 main, 3 FGES) are outcome collapse (one class has a single row,
the RF cannot be fitted), same as `partial`; hand vs FGES is no longer confounded with
family (every causal family has 660 runs on each graph source).

- **MAMA-MIA on the unattacked tables.** Every facct-cffix-* batch and
  facct-eps-dpdiff-2026-10-18: 16,170 rows in `report/data/facct/mama_mia_cffix.csv`
  (`sweeps/mama-mia-cffix-2026-10-22.sh`). Use these for causal-family CF; the CF rows of
  those families in `mama_mia.csv` are old-rule tables.
- **MAMA-MIA by group.** `mama_mia.py --by-group`: the same attack scores ranked within
  each protected attribute's two groups (holdout modal value = advantaged), one row per
  run x flavour x focal-point source x attribute x group, with member and non-member
  counts. All 15 batches with saved tables, cores 8-15
  (`sweeps/mama-mia-group-2026-10-22.sh`) -> `report/data/facct/mama_mia_group.csv`.
- **TabFairGAN lambda grid.** Its parity loss (package `modules.py` `FairLossFunc`) is
  signed and unbounded: it maximises P(y=1|under) - P(y=1|priv). At the paper's lambda
  (0.5; COMPAS 2.2) with 30 fair epochs it overshoots and reverses the gap in its own
  table on 9 of 11 datasets. Label gap in the table, majority - rest, mean of seeds,
  none -> dp: Adult sex +17.8 -> -28.1 pp; SBO +12.8 -> -50.3; SNAKE -0.6 -> -52.3;
  HMDA -0.1..+4.2 -> -14.6..-18.7; CDC -2.0 -> +15.6; ACSPubCov TX race -3.9 -> +18.2;
  only ACSIncome narrows (CA +11.4 -> +3.2, TX +15.3 -> +7.8). The paper set lambda per
  dataset, so the publisher here picks lambda per dataset x S from a grid by the table's
  label gap: batches `facct-tfg-lam{002,005,01,02,10,20}-2026-10-22` (lambda 0.02, 0.05,
  0.1, 0.2, 1.0, 2.0; 0.5 is in facct-2026-10-09), dp only (`only: /dp`, `settings:
  {lamda}`), 165 runs each, cores 24-47 (`sweeps/tfg-lambda-2026-10-22.sh`). About 11
  core-hours a batch. Follow-ups once done: collect, MAMA-MIA (both modes), choose
  lambda, re-point the Angelozzi `tabfairgan-dp` arms.
- **FairPFN extension** (`scripts/fairpfn_baseline.py`, GPU 0, cores 16-23,
  `sweeps/fairpfn-ext-2026-10-22.sh`). Re-exported with every generator's unmitigated
  release as a context (adds decaf, decaf_ctgan, decaf_diffusion, decaf_dpgan / dpctgan /
  dpdiffusion at epsilon 1, 10, 1000, tabfairgan) and, new `--stack`, edited releases as
  contexts (decaf_tabpfn, decaf_diffusion, MST at epsilon 10; dp and ftu; each role
  split; scored on that split's protected attributes only). CF is left out of the stack
  because the causal families' CF tables in facct-2026-10-09 are old-rule. 10,000-row
  contexts only. New `--results`: three processes append to `results-{a,b,c}.csv`; every
  `results*.csv` counts as done. Inputs before the re-export:
  job tmp `fairpfn-inputs-backup/` (13 MB).
- **Arms on saved releases** (`scripts/release_arms.py`, new; rows in
  `sweeps/arms-2026-10-22/<arm>/`, collected to `report/data/facct/arm_<arm>.csv`; same
  metric names as the runs tables, plus per-group selection rate, TPR and accuracy for
  both groups, `sel_dis__X`, `sel_adv__X`, ...). The sweep's conditional metrics loop
  over admissible cells in pandas (75 s a release on ACSIncome); `fast_fairness_metrics`
  is the same computation with bincount, equal to 1e-17 on three checked runs, and an RF
  refit reproduces a stored sweep row exactly (ACSIncome-CA MST run 19: dp_gap__SEX
  -0.059024, cdp_gap__SEX 0.016834).
  - `unaware`: the sweep's RF on every unmitigated release with the role split's
    protected attributes dropped. Main, FGES, eps-a, eps-b, eps-dpdiff.
  - `recipients`: one-hot logistic regression (`lr_ohe`) and gradient boosting (`hgb`,
    sklearn HistGradientBoosting, random_state 0) on every release of main, FGES,
    cffix-*, eps-*, cut*. (The sweep's own `lr` uses raw integer codes; `lr_ohe` is
    what a recipient would fit.)
  - `placebo`: one unmitigated release per shard (epsilon none or 1000), outcome
    scrambled at frac 0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0 (replacement draws
    at the table's base rate; the same scrambled table for rf, lr_ohe and hgb). Scored
    on the union of the dataset's protected attributes. Durable copy of the RF placebo
    as well (the analysis session's pl/ and pl2/ live in its job tmp).
  - `resample` (Krchova et al. 2023, agreed 2026-10-14 notes 10.7): 5-fold out-of-fold
    gradient-boosting scores on the release; each group (cell of the split's binarised
    protected attributes, >= 50 rows) other than the reference is mapped onto the
    reference group's score quantiles, mixed with weight lam, and its outcomes redrawn;
    reference = highest outcome rate among groups with >= 5% of rows. lam 0 (redraw
    only, the control), 0.5, 1. Edited tables saved for lam > 0. Scored with the sweep's
    RF. Main and FGES unmitigated releases. Trial (ACSIncome-CA MST, sex-only split,
    epsilon 1000, lam 1): table label gap -2.1 -> +0.5 pp, real-people parity gap
    -6.6 -> -3.1 pp, AUC 0.823 -> 0.823.
  - Lanes (`sweeps/arms-2026-10-22.sh`): 1 cores 4-7 unaware then recipients; 2 cores
    8-15 after the by-group MIA: placebo then resample (FGES); 3 cores 24-47 after the
    lambda grid: resample (main).

### 21.1 Results as they land

- **MAMA-MIA by group, done 07:19 UTC.** 650,976 rows (15 batches, 1,885 shards)
  in `report/data/facct/mama_mia_group.csv` (150 MB, git-ignored); averaged over
  seeds in `mama_mia_group_summary.csv` (66,132 rows: `auc_dis`, `auc_adv`,
  `auc_gap` = dis - adv, TPR at 10% / 1% FPR, group sizes). Groups have 337-14,324
  members. Findings (AUC, 0.5 = chance):
  - The minority group is slightly more exposed, and the gap grows with the
    budget: main batch, unmitigated, MST +0.0024, PrivBayes +0.0039, PrivSyn
    +0.0021; the deep DP generators ~0. Across epsilon, PrivBayes +0.0067 at
    epsilon 300 vs -0.0001 at 0.1. Largest cells: Adult native-country +0.016,
    COMPAS sex +0.009.
  - The fairness edits do not widen it. Paired with the same shard's unmitigated
    release, the dis - adv gap moves -0.0005 (dp), -0.0002 (cf), 0.0000 (ftu);
    95th percentile of the change +0.007 (dp). At epsilon >= 100, PrivBayes dp
    halves the gap (+0.0078 -> +0.0029). Chang & Shokri's "fairness costs the
    protected group privacy" is not seen for these edits.
- **Unawareness arm, done 09:16 UTC.** `report/data/facct/arm_unaware.csv`, 7,737
  rows from 1,207 shards (two shards had no finished runs; no errors log). First
  look, main batch (hand graphs), 9,561 run x attribute cells, each unmitigated
  release matched to the same shard's edited releases (same split, epsilon), real
  holdout, the sweep's RF. Mean |gap| in pp of selection rate, AUC:

  | recipient model on | parity gap | conditional gap | AUC |
  |---|---|---|---|
  | unedited release, all columns | 8.2 | 5.8 | 0.695 |
  | unedited release, protected columns dropped (unaware) | 6.0 | 3.2 | 0.686 |
  | publisher's FTU edit | 6.2 | 3.7 | 0.686 |
  | publisher's CF edit | 5.2 | 2.9 | 0.654 |
  | publisher's parity (dp) edit | 4.2 | 3.2 | 0.591 |

  Dropping the columns narrows the parity gap in 64% of cells (median -0.4 pp,
  10th percentile -8.5 pp); the sign flips in 18%. It equals the publisher's FTU
  edit on every summary, and comes within 0.3 pp of the CF edit's conditional gap
  at 0.03 more AUC. The parity edit buys a further 1.8 pp of parity gap for 0.10
  AUC. Script: job tmp `unaware_look.py` (to be moved into the analysis).
- **FairPFN extension, done 09:26 UTC.** 15,726 rows (264 failed: "context has one
  outcome class or one group", 183 of them DP-GAN epsilon 1, 60 DP-GAN epsilon 10,
  15 DECAF). Merged into `sweeps/facct-2026-10-09/fairpfn_baseline/results.csv`
  (29,916 rows; lane files moved to job tmp), re-summarised with all 11 generators
  (`summarize_fairpfn_baseline.py --generators ...`), copied to
  `report/data/facct/fairpfn/`. First look, 10,000-row contexts, mean
  |dp_gap_matched| in pp and AUC, each model on the same context:
  - FairPFN narrows the gap on every non-degenerate context: real 12.6 -> 5.9 pp
    (AUC 0.803 -> 0.747), DECAF-TabPFN 12.2 -> 7.7, TabFairGAN (unedited) 12.3 ->
    6.7, PrivSyn epsilon 1000 15.3 -> 7.3. The unaware RF on the same contexts sits
    between (8.3-10.2 pp, AUC within 0.005 of the RF).
  - On DP-CTGAN (all epsilon) and DP-GAN (epsilon 10, 1000) contexts FairPFN widens
    it: e.g. DP-CTGAN epsilon 1000 7.6 -> 15.4 pp, DP-GAN epsilon 1000 8.2 -> 14.8.
    Those contexts have RF AUC 0.51-0.70.
  - Stacked (publisher edit + FairPFN): on the FTU edit, no change vs FairPFN on the
    unedited release (7.3-8.4 pp either way). On the parity edit, FairPFN's |gap|
    rises to 9.9-12.4 pp (unedited 7.3-8.4) and AUC falls to 0.57; the signed gap
    averages about -1 pp and flips direction in 45% of cells. Two corrections on
    the same table do not add up; they make the recipient's model erratic.
  Script: job tmp `fairpfn_look.py`.
- **Placebo, done 09:12 UTC.** `report/data/facct/arm_placebo.csv`, 24,630 rows from
  821 shards (main + FGES; one error, an outcome-collapsed Adult DP-GAN release; two
  shards without finished runs). Columns `frac`, `recipient` (rf, lr_ohe, hgb), gaps
  over the union of the dataset's protected attributes (no admissible set, so no
  conditional gaps). Main batch, mean |parity gap| pp / AUC by fraction scrambled:

  | frac | rf | lr_ohe | hgb |
  |---|---|---|---|
  | 0 | 9.9 / 0.718 | 11.0 / 0.742 | 10.4 / 0.740 |
  | 0.3 | 8.1 / 0.694 | 9.1 / 0.741 | 8.9 / 0.734 |
  | 0.5 | 6.4 / 0.669 | 7.2 / 0.737 | 7.0 / 0.727 |
  | 0.7 | 4.7 / 0.630 | 4.7 / 0.727 | 4.1 / 0.711 |
  | 0.9 | 2.9 / 0.562 | 2.0 / 0.671 | 1.8 / 0.650 |
  | 1.0 | 2.9 / 0.512 | 1.5 / 0.501 | 1.6 / 0.500 |

  Positive rate falls only 0.42 -> 0.36-0.38, so the gap does not shrink by
  predicting everyone negative. Rough read against the unawareness table above
  (different attribute sets, so indicative only): the RF parity edit (8.2 -> 4.2
  pp at AUC 0.695 -> 0.591) sits about on the RF placebo curve (~3.7 pp at AUC
  0.59 by interpolation). Gradient boosting and one-hot LR lose far less AUC to
  random label noise than the RF does (0.03 vs 0.09 AUC at frac 0.7), so the
  placebo frontier depends on the recipient. Script: job tmp `placebo_look.py`.
- **TabFairGAN lambda grid, done 10:45 UTC.** 6 x 165 runs, all done; collected
  (`report/data/facct/facct-tfg-lam*_runs.csv`). `scripts/tabfairgan_lambda.py` reads
  the dp releases at every lambda (grid + facct-2026-10-09 at the paper's), computes the
  release's own outcome-rate gap on S (majority - rest, majority on the training rows),
  and picks per dataset x role split the lambda with the smallest mean |label gap|
  over seeds, keeping the paper's unless the grid beats it by >= 1 pp. Per run:
  `report/data/facct/tabfairgan_lambda.csv`; choice: `tabfairgan_lambda_choice.csv`.
  Mean over seeds, |label gap on S| in the release / |real parity gap on S| (sweep RF,
  real holdout), pp, and AUC, chosen vs paper lambda:

  | dataset (S) | chosen lambda | label gap | real gap | AUC | at paper lambda |
  |---|---|---|---|---|---|
  | ACSIncome CA (SEX) | 0.5 (paper) | 5.5 | 6.7 | 0.851 | same |
  | ACSIncome TX (SEX) | 0.5 (paper) | 7.8 | 11.1 | 0.832 | same |
  | ACSPubCov CA (SEX / RAC1P) | 0.1 / 0.05 | 0.9 / 0.6 | 0.8 / 1.6 | 0.672 / 0.670 | 4.4 / 3.9, 3.9 / 5.7, 0.67 |
  | ACSPubCov TX (SEX / RAC1P) | 0.02 / 0.05 | 0.7 / 1.4 | 1.4 / 1.1 | 0.755 / 0.751 | 6.9 / 18.2, 3.7 / 11.6, 0.75 / 0.72 |
  | Adult (sex) | 0.2 | 4.7 | 5.0 | 0.815 | 28.1, 18.8, 0.737 |
  | CDC (Sex) | 0.05 | 0.4 | 0.5 | 0.782 | 15.6, 10.2, 0.735 |
  | COMPAS (race) | 2.2 (paper) | 2.4 | 8.9 | 0.497 | same |
  | HMDA CA (sex / race) | 0.02 / 0.05 | 0.9 / 0.3 | 0.4 / 0.7 | 0.75 | 18.7 / 17.7, 7.1 / 7.7, 0.71 |
  | HMDA TX (sex / race) | 0.05 | 0.6 / 0.7 | 0.9 / 1.5 | 0.77 | 14.7 / 14.6, 5.4, 0.73 |
  | SBO (SEX1) | 0.05 | 1.8 | 8.7 | 0.935 | 50.3, 43.9, 0.874 |
  | SNAKE (female) | 0.02 | 1.2 | 2.3 | 0.668 | 52.3, 62.4, 0.587 |

  (The label-gap column's mean of |.| over seeds; the choice file's `label_gap_S` is
  the signed mean. Signs of `real_dp_gap_S` follow the sweep's convention.) The grid
  is monotone past the minimum everywhere: larger lambda, larger reversal, lower AUC.
  On 7 of 11 datasets the best lambda is 0.1 or less: SNAKE, HMDA, CDC and ACSPubCov
  are near parity unedited, so any push overshoots; SBO (+12.8 pp unedited) closes at
  0.05 and reverses from 0.1.
  **COMPAS: TabFairGAN's release carries no signal at any lambda (AUC 0.49-0.50 dp,
  0.52 unedited)**, so its COMPAS row measures nothing; flag in the paper.
  Angelozzi arms re-pointed: `facct_bench.py export` gained `--choose <csv>` (keep,
  per dataset x method x mechanism x split listed, only the named batch's runs);
  `export --merge --methods tabfairgan --batches facct-2026-10-09,<lam batches>
  --choose report/data/facct/tabfairgan_lambda_choice.csv` changed exactly 63 tags,
  all `tabfairgan-dp@` (ACSIncome, COMPAS unchanged), no other manifest field
  (manifests backed up to `facct_data/backup-pre-tfglam-2026-10-22/`). Paper-lambda
  outputs moved to `facct_output/superseded-tfg-paperlambda-2026-10-22/` (kept, not
  collected); 63 cells re-scored on cores 16-23, 6 jobs
  (`sweeps/angelozzi-tfglam-2026-10-22.sh`, then `collect`). Then MAMA-MIA pooled and
  by group on the six lambda batches (`sweeps/mama-mia-tfglam-2026-10-22.sh`, waits on
  it) -> `mama_mia_tfglam.csv`, `mama_mia_group_tfglam.csv`.
- **Re-sampling (Krchova) arm, done 11:21 UTC.** `report/data/facct/arm_resample.csv`,
  13,085 rows from 822 shards (main + FGES; 3 errors, all outcome-collapsed releases:
  Adult DP-diffusion s1, Adult DP-GAN s4, CDC DP-GAN s3). Columns: `lam`, `ref_cell`,
  `n_cells`, `n_eligible`, `share_redrawn`, `share_label_changed`,
  `base_rate_before/after`, `orig_label_gap__X` / `edited_label_gap__X` (rest -
  majority, in the release), then the sweep's metrics for the RF trained on the
  edited release. First look, main batch, 9,092 run x attribute cells matched across
  every arm below (same shard, split, epsilon), mean |gap| pp on the real holdout:

  | release the recipient's RF trains on | parity gap | conditional gap | AUC |
  |---|---|---|---|
  | unedited | 8.2 | 5.7 | 0.692 |
  | re-sampled, lam 0 (redraw only, control) | 8.1 | 5.6 | 0.688 |
  | re-sampled, lam 0.5 | 6.0 | 4.0 | 0.685 |
  | re-sampled, lam 1 | 4.5 | 3.7 | 0.676 |
  | unedited, protected columns dropped (unaware) | 5.9 | 3.1 | 0.683 |
  | FTU edit | 6.2 | 3.7 | 0.687 |
  | CF edit | 5.2 | 2.9 | 0.654 |
  | parity (dp) edit | 3.8 | 2.9 | 0.584 |

  The release's own label gap goes 5.1 -> 0.5 pp at lam 1 (5.0 at lam 0), yet real
  people keep 4.5 pp. Re-sampling at lam 1 gets 85% of the parity edit's reduction
  (3.7 of 4.4 pp) for 15% of its AUC cost (0.016 of 0.108), and sits well off the
  RF placebo curve (scrambling 20% of labels costs the same 0.015 AUC and keeps 89% of
  the gap, i.e. ~7.3 pp here). It works for every
  generator with signal (e.g. DECAF-TabPFN 10.3 -> 4.3 pp, AUC 0.794 -> 0.770; PrivBayes
  8.1 -> 3.6, 0.744 -> 0.724) and does nothing on DP-GAN / DP-CTGAN (7.7 -> 7.6,
  3.2 -> 2.9; their releases have AUC 0.53-0.65). By dataset it narrows the gap
  everywhere but CDC (1.8 -> 1.8, already small); SBO least (10.7 -> 8.7). Script:
  job tmp `resample_look.py`.
- **Other recipients (one-hot LR, gradient boosting), done 11:26 UTC.**
  `report/data/facct/arm_recipients.csv`, 69,352 rows (`recipient` = lr_ohe / hgb)
  from 1,882 shards of main, FGES, cffix-*, eps-*, cut*; no errors. Then the six
  lambda batches (`sweeps/arms-2026-10-22-4.log`), which re-collects the file. Main
  batch, each edited release matched to the same shard's unedited one, mean |gap|
  pp, unedited -> edited, and AUC:

  | edit | recipient | parity gap | conditional gap | AUC |
  |---|---|---|---|---|
  | dp | rf / lr_ohe / hgb | 8.3 -> 4.2 / 9.1 -> 4.1 / 8.5 -> 3.7 | 5.8 -> 3.2 / 6.3 -> 2.8 / 5.7 -> 2.5 | 0.695 -> 0.591 / 0.721 -> 0.604 / 0.717 -> 0.600 |
  | cf | rf / lr_ohe / hgb | 8.2 -> 5.2 / 9.0 -> 5.3 / 8.4 -> 5.0 | 5.7 -> 2.9 / 6.2 -> 2.6 / 5.7 -> 2.3 | 0.691 -> 0.654 / 0.717 -> 0.679 / 0.713 -> 0.675 |
  | ftu | rf / lr_ohe / hgb | 8.2 -> 6.2 / 9.0 -> 6.4 / 8.4 -> 6.1 | 5.7 -> 3.7 / 6.2 -> 3.5 / 5.6 -> 3.2 | 0.691 -> 0.686 / 0.716 -> 0.712 / 0.713 -> 0.708 |

  The edits act the same on every recipient; LR and boosting see slightly larger
  reductions than the RF, in every generator (dp edit, reduction in pp: DECAF 10.8 /
  13.3 / 11.9, MST 3.2 / 3.9 / 3.8, DP-CTGAN 0.3 / 0.7 / 0.4). The conclusions do not
  hang on the sweep's RF. Script: job tmp `recipients_look.py`. 131 MB, so the plain
  CSV is git-ignored and `arm_recipients.csv.gz` is committed.
- **MAMA-MIA on the lambda grid, done 11:31 UTC** (run early on the freed cores 24-47,
  `sweeps/mama-mia-tfglam-early-2026-10-22.log`; the chained script finds it done).
  `mama_mia_tfglam.csv` (1,980 rows), `mama_mia_group_tfglam.csv` (10,200 rows).
  Marginal flavour, mean AUC 0.503-0.507 at every lambda (unedited 0.506, max 0.518):
  TabFairGAN exposes no one measurably at any lambda.
- Provenance of the lambda batches: their `code/*.diff` snapshots (53 MB each, nine
  fingerprints per batch, because scripts were being edited while they ran) touch
  only `.gitignore`, `scripts/mama_mia.py`, `scripts/fairpfn_baseline.py`, report
  CSVs, notes and the staged deletion of the old FAccT zip; no file under `src/`.
  So the generator code is commit 4d04cba. They stay local (git-ignored), as does
  `sweeps/arms-2026-10-22/` (2.2 GB of rows, predictions and re-sampled tables).
