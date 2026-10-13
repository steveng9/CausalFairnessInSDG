# FairPFN downstream baseline (run 2026-10-11)

Status: complete. 1,500 cells, 4,500 scored rows, 0 failures. Nothing here is
in the manuscript yet.

## What FairPFN is

- Robertson, Hollmann, Müller, Awad, Hutter. "FairPFN: A Tabular Foundation
  Model for Causal Fairness." ICML 2025. arXiv 2506.07049.
- A **classifier**, not a generator. A transformer pretrained on about 1.5
  million synthetic causal models. For each one it sees the biased table and
  is trained to predict the label produced when the protected attribute's
  outgoing edges are dropped. At use time it is given a labelled context and
  told which column is protected; it needs **no causal graph**.
- Limits stated in the paper: **one binary protected attribute**, assumed
  exogenous; removes the **total** effect only. So it has a counterpart to our
  DP mechanism, and none to FTU or to CF with admissible attributes.
- Their evaluation: Law School and Adult; baselines Unfair, Unaware, CFP,
  EGR, and others; metrics are the causal effect of the protected attribute
  on predictions (ATE), 1 − AUC, and counterfactual absolute error.

## Install and provenance

| Item | Value |
|---|---|
| Code | `~/external/FairPFN`, clone of github.com/jr2021/FairPFN at `4799e8ccc90473993e2d4b8188d4961839981cf5` (last pushed 2025-04-01) |
| Licence | **None upstream.** Kept outside this repo; do not redistribute code or weights. Emailing the authors would settle it. |
| Checkpoint | `results/models_diff/model_submitit_0c_id_e3e764c1_epoch_-1.cpkt`, sha256 `d9f9dd9aa82858b4dbaa416e428a04239c6bf6c474f553c2d5e59decbb8559f4` |
| Environment | conda env `fairpfn`: Python 3.10, torch 1.13.1+cu117, numpy 1.23.5, scikit-learn 1.2.1, pandas 2.0.3. Full list in `sweeps/facct-2026-10-09/fairpfn_baseline/environment-fairpfn.txt` |
| Our code | `scripts/fairpfn_baseline.py` (uncommitted; repo HEAD was `33a7139`) |
| Hardware | GPU 1 (RTX 6000 Ada), one worker; peak 7.4 GB at a 10,000-row context |

Two changes were needed to make upstream run. Both are set from our script;
upstream files are untouched.

1. The shipped config points at a checkpoint path on the authors' cluster.
   We set `model_string` to the copy shipped in the repo.
2. Half-precision inference fails an internal dtype assert. We set
   `fp16_inference = False`.

Trap: `artifacts/fairpfn_model.pkl` holds the architecture with **untrained**
weights (36 of 80 parameter tensors are all zero). The trained weights are
only in the `.cpkt` file. After a correct load, zero tensors: 0.

Sanity check on a toy table with a planted gap of 0.50: logistic regression
keeps a gap of 0.52 at AUC 0.93; FairPFN gives −0.03 at AUC 0.77.

## Design

Every score is on the **same real holdout** the sweep uses (30% split,
`random_state = seed`, 15,000 rows on the sweep datasets), 5 seeds.

- **Datasets:** acsincome_ca, acsincome_tx, acspubcov_ca, acspubcov_tx,
  cdc_diabetes, sbo, snake; plus adult and compas on real data only.
- **Protected attribute:** one run per attribute named in any of the
  dataset's role splits. Each is binarized as the sweep's metrics do it
  (real-holdout modal value = 0, everything else = 1) and that indicator
  replaces the raw column. The dataset's other protected attributes stay in
  as ordinary features.
- **Context source:** `real` (the training split); `mst_eps1`, `mst_eps10`,
  `mst_eps1000` and `decaf_tabpfn` (each generator's **no-mechanism** release,
  first role split alphabetically, since the role split does not affect a
  no-mechanism release).
- **Context size:** 2,000 and 10,000 rows, sampled with `random_state = seed`
  (COMPAS is capped at its 4,320 training rows). The model was pretrained at
  sequence length 2,200 and at most 85 features.
- **Controls on the same context and feature layout:** `rf` (random forest,
  protected attribute included) and `rf_unaware` (protected attribute
  dropped).
- **Features:** integer category codes passed as numbers, FairPFN's default.

### Decision rule (read this before quoting a gap)

Two rules are recorded for every model.

- Unsuffixed columns (`dp_gap`, `tprb`, ...): positive if score ≥ 0.5, the
  sweep's rule.
- `_matched` columns: positive for the top scores until the positive rate
  equals the context's positive rate.

FairPFN's scores are pulled toward the base rate. On PublicCoverage it
predicts 0–4% positive at 0.5 against a base rate near 37%, which makes it
look fair for free. On ACSIncome this does not happen (positive rate 0.39 at
0.5, and the two rules agree to within 0.006). **Use the `_matched` gaps for
FairPFN and its controls.** The sweep's generator rows use 0.5 on a random
forest trained on the full synthetic table; that forest's positive rate is
near the base rate, so the two are close but not the same rule.

All gaps below are absolute values averaged over 5 seeds.

## Results

### Like-for-like: one binary attribute (ACSIncome, sex, `sex_only_broad_adm`)

| Method | AUC (CA) | Gap (CA) | AUC (TX) | Gap (TX) |
|---|---|---|---|---|
| Random forest, real, 10k context | 0.871 ± 0.001 | 0.139 ± 0.007 | 0.853 ± 0.002 | 0.184 ± 0.012 |
| Random forest unaware, real, 10k | 0.867 ± 0.001 | 0.069 ± 0.004 | 0.845 ± 0.004 | 0.105 ± 0.010 |
| **FairPFN, real, 10k context** | 0.850 ± 0.010 | 0.015 ± 0.015 | 0.831 ± 0.010 | 0.024 ± 0.018 |
| Our TabPFN generator, no mechanism | 0.866 ± 0.002 | 0.134 ± 0.005 | 0.848 ± 0.002 | 0.185 ± 0.010 |
| Our TabPFN generator + FTU | 0.864 ± 0.002 | 0.061 ± 0.004 | 0.839 ± 0.001 | 0.071 ± 0.007 |
| Our TabPFN generator + CF | 0.849 ± 0.004 | 0.053 ± 0.003 | 0.821 ± 0.004 | 0.067 ± 0.013 |
| Our TabPFN generator + DP | 0.606 ± 0.010 | 0.011 ± 0.006 | 0.598 ± 0.004 | 0.012 ± 0.014 |

**FairPFN matches our DP cut on the parity gap and keeps about 24 more AUC
points.** Other sex-only cells: CDC sex and SNAKE sex have almost no gap to
remove (≤ 0.03 without any mechanism), and there our DP cut still costs
12–19 AUC points on the TabPFN generator (0.792 → 0.602; 0.678 → 0.558).

Untested explanation: our DP cut shuffles every outcome parent that descends
from the protected attribute, which removes most of the signal; FairPFN
removes only the part attributable to the attribute.

### All datasets, real data, 10,000-row context (AUC, then matched-rate gap)

| Dataset | Attribute | AUC FairPFN | AUC RF | Gap FairPFN | Gap RF | Gap RF unaware |
|---|---|---|---|---|---|---|
| acsincome_ca | HISP | 0.827 | 0.872 | 0.110 | 0.337 | 0.306 |
| acsincome_ca | RAC1P | 0.854 | 0.871 | 0.028 | 0.102 | 0.081 |
| acsincome_ca | SEX | 0.850 | 0.871 | 0.015 | 0.139 | 0.069 |
| acsincome_ca | WAOB | 0.854 | 0.872 | 0.048 | 0.089 | 0.060 |
| acsincome_tx | HISP | 0.827 | 0.853 | 0.099 | 0.266 | 0.227 |
| acsincome_tx | RAC1P | 0.844 | 0.853 | 0.044 | 0.103 | 0.064 |
| acsincome_tx | SEX | 0.831 | 0.853 | 0.024 | 0.184 | 0.105 |
| acsincome_tx | WAOB | 0.843 | 0.852 | 0.063 | 0.100 | 0.073 |
| acspubcov_ca | DIS | 0.535 | 0.708 | 0.294 | 0.522 | 0.425 |
| acspubcov_ca | HISP | 0.656 | 0.709 | 0.047 | 0.132 | 0.070 |
| acspubcov_ca | NATIVITY | 0.681 | 0.709 | 0.034 | 0.011 | 0.006 |
| acspubcov_ca | RAC1P | 0.673 | 0.708 | 0.044 | 0.025 | 0.010 |
| acspubcov_ca | SEX | 0.667 | 0.709 | 0.046 | 0.058 | 0.045 |
| acspubcov_tx | DIS | 0.599 | 0.792 | 0.086 | 0.488 | 0.399 |
| acspubcov_tx | HISP | 0.735 | 0.791 | 0.014 | 0.013 | 0.035 |
| acspubcov_tx | NATIVITY | 0.737 | 0.791 | 0.076 | 0.142 | 0.141 |
| acspubcov_tx | RAC1P | 0.736 | 0.791 | 0.028 | 0.029 | 0.003 |
| acspubcov_tx | SEX | 0.737 | 0.791 | 0.037 | 0.027 | 0.027 |
| adult | native-country | 0.835 | 0.862 | 0.032 | 0.068 | 0.043 |
| adult | race | 0.823 | 0.862 | 0.042 | 0.121 | 0.107 |
| adult | sex | 0.809 | 0.862 | 0.094 | 0.223 | 0.213 |
| cdc_diabetes | Education | 0.783 | 0.795 | 0.031 | 0.093 | 0.088 |
| cdc_diabetes | Income | 0.781 | 0.793 | 0.051 | 0.112 | 0.102 |
| cdc_diabetes | Sex | 0.795 | 0.796 | 0.038 | 0.021 | 0.006 |
| compas | race | 0.702 | 0.705 | 0.069 | 0.341 | 0.238 |
| compas | sex | 0.701 | 0.697 | 0.048 | 0.227 | 0.111 |
| sbo | BORNUS1 | 0.835 | 0.946 | 0.115 | 0.114 | 0.119 |
| sbo | ETH1 | 0.832 | 0.946 | 0.079 | 0.118 | 0.115 |
| sbo | RACE1 | 0.833 | 0.945 | 0.083 | 0.151 | 0.152 |
| sbo | SEX1 | 0.845 | 0.946 | 0.073 | 0.128 | 0.116 |
| sbo | VET1 | 0.834 | 0.946 | 0.022 | 0.024 | 0.026 |
| snake | citistat | 0.662 | 0.702 | 0.027 | 0.274 | 0.224 |
| snake | female | 0.684 | 0.703 | 0.062 | 0.004 | 0.014 |
| snake | wbhaom | 0.643 | 0.700 | 0.050 | 0.280 | 0.217 |

Patterns:

- **Large unmitigated gap:** FairPFN cuts it sharply for 0–5 AUC points
  (COMPAS race 0.34 → 0.07; SNAKE citizenship 0.27 → 0.03; ACSIncome sex
  0.14 → 0.015). Adult sex is only halved (0.22 → 0.09).
- **Small unmitigated gap (below about 0.05):** FairPFN often makes it worse
  (SNAKE sex 0.004 → 0.062; CDC sex 0.021 → 0.038; PublicCoverage-CA
  nativity 0.011 → 0.034).
- **Failures:** PublicCoverage disability (AUC 0.71 → 0.54 in CA with a 0.29
  gap left; 0.79 → 0.60 in TX). SBO, every attribute (AUC 0.95 → 0.83, gap
  barely moved).
- **Context size:** 2,000 and 10,000 rows give nearly the same numbers.
- **Synthetic context:** fed our no-mechanism releases it behaves much as on
  real data with the TabPFN release; on MST at ε = 1 it is less reliable and
  sometimes raises the gap (e.g. PublicCoverage sex 0.03 → 0.10 in CA). Its
  AUC on MST context sometimes exceeds the random forest's on the same
  context (ACSIncome-TX 0.80 vs 0.76).

## Caveats for the paper

1. **Only the sex-only role split is like-for-like.** FairPFN handles one
   attribute per run. Our other role splits cut several protected attributes
   at once, so their DP rows pay for more than FairPFN is asked to do.
2. **Multi-category attributes were binarized** majority-versus-rest. The
   paper only claims support for binary attributes.
3. **Different training-set sizes and classifiers.** FairPFN sees at most
   10,000 context rows; generator rows train a random forest on 35,000
   synthetic rows. The `rf` control on the same context shows the context
   size itself costs under 1 AUC point.
4. **Different decision rules** between FairPFN rows (matched rate) and
   generator rows (0.5), as described above.
5. **Demographic parity is not FairPFN's own target.** It is trained to
   remove a causal effect; we score an observational gap. The paper's own
   metric (ATE under a fitted causal model) is not computed here.
6. **Not private.** On `real` context FairPFN reads real records at
   prediction time. The synthetic-context rows are the fair comparison for a
   private release.
7. One ensemble member (upstream default), full precision.

## How this could enter the paper

- Add FairPFN as one more point among the downstream mitigators, on the
  gap-versus-AUC plot planned in §1a of `manuscript/reviews/FAccT-2027-plan.md`.
- State the DP result plainly: for single-attribute parity a downstream
  model keeps far more accuracy than our all-or-nothing DP cut.
- Position the cut where a downstream fix cannot follow: FTU and CF with
  admissible attributes; several protected attributes at once; a property
  that ships with the table instead of depending on the recipient; a private
  release.
- This strengthens the case for the partial-cut knob (§1a): our DP mechanism
  currently has only one, very expensive, setting.

## Files

All under `sweeps/facct-2026-10-09/fairpfn_baseline/`:

| File | Content |
|---|---|
| `results.csv` | One row per dataset × seed × source × context size × attribute × model. Both decision rules, plus `cdp_gap@<role>` for each role split that protects the attribute. |
| `summary_by_cell.csv` | Mean, std, seed count per cell (absolute gaps). |
| `generator_comparison.csv` | The sweep's RF AUC and absolute DP gap per dataset × generator × mechanism × role split × attribute. |
| `preds/*.npy` | FairPFN's holdout scores (float16), one file per cell, so other metrics can be computed without rerunning. |
| `inputs/<dataset>-s<seed>/` | Splits and `meta.json` (roles, majority values, source paths). |
| `run.log`, `environment-fairpfn.txt` | Log and package list. |

## Reproduce

```bash
# 1. splits and source manifest (sdg env)
~/miniconda3/envs/sdg/bin/python scripts/fairpfn_baseline.py export \
  --datasets acsincome_ca,acsincome_tx,acspubcov_ca,acspubcov_tx,cdc_diabetes,sbo,snake,adult,compas
# 2. predictions and scores (fairpfn env; resumable, about 50 minutes)
CUDA_VISIBLE_DEVICES=1 taskset -c 32-47 nice ~/miniconda3/envs/fairpfn/bin/python -u \
  scripts/fairpfn_baseline.py run
# 3. summary tables (sdg env)
~/miniconda3/envs/sdg/bin/python scripts/summarize_fairpfn_baseline.py
```

## Not done

- FairPFN's own ATE metric and counterfactual error.
- PrivBayes and PrivSyn releases as context; Adult and COMPAS synthetic
  context.
- A test of the explanation for why our DP cut costs so much accuracy.
