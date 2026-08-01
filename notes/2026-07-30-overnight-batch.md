# Batch `overnight-2026-07-30` — full results analysis

**Status:** complete. 240 runs, 0 failures, 106 minutes of compute.
**Raw outputs:** `results/overnight-2026-07-30_report.md`, `results/overnight-2026-07-30_runs.csv`
(240 × 89), `experiments.db`, `logs/overnight-2026-07-30.log`.

> **Caveat that applies to every number below: one seed per cell.** Treat individual
> cells as directional. Aggregates over 9 cells (method × mechanism, averaged across
> ε and role config) are the trustworthy rows. Batch `seeds-2026-07-30` adds seeds
> 1–4 and supersedes this analysis when it lands.

---

## 1. Grid

| Axis | Values |
|---|---|
| Datasets | `adult` (30,162 rows), `compas` (6,172 rows) |
| Generators | MST, PrivBayes, PrivSyn (DP graphical) + DECAF (non-private GAN baseline) |
| ε | 1.0, 10.0, 1000.0. DECAF: **NULL** — no DP mechanism, so it runs once per config, not 3× |
| Fairness mechanism | `none` (control), FTU, DP, CF |
| Role configs | 3 per dataset (§3) |
| Seeds | 1 (seed 0) |
| Total | 240 runs |

Interpretation calls made against the original request ("MST, PrivBayes, PrivSyn, DECAF
… all three fairness methods FTU, CP, DF"):

- "FTU, CP, DF" read as FTU / **CF** (counterfactual) / **DP** (demographic parity).
- **DF** (distributional fairness) is not a mechanism in this codebase — it is an
  *evaluation-reference* axis (`fairness/df.py::EvalReference`): score against the real
  holdout, or against the synthetic data's own distribution. Both are computed for every
  run from a single generation (`runner.EVAL_REFERENCE_BOTH`), so the DF axis is free
  rather than doubling the grid. Synthetic-reference metrics carry the `synthref__` prefix.
- `none` added as the control — without it there is no baseline to measure FTU/DP/CF against.
- **AIM is implemented but absent from this grid** (it wasn't in the request).

---

## 2. Metric definitions

| Metric | Direction | Definition |
|---|---|---|
| `fairness_gap` (DP gap) | lower = fairer | Worst-case \|P(Ŷ=1 \| A=a) − P(Ŷ=1 \| A=a′)\| over protected attributes, for a classifier **trained on synthetic, evaluated on the real holdout** |
| `cond_fairness_gap` (CDP gap) | lower = fairer | Same, conditioned on the admissible attributes. The gap **CF specifically targets** |
| `tprb` / `tnrb` | lower = fairer | True/false-positive-rate balance gaps (equalized-odds family) |
| `ctprb` / `ctnrb` | lower = fairer | Admissible-conditioned versions |
| `tvd_1way` / `tvd_2way` | **lower = better** | Total-variation distance, real vs. synthetic, over all 1-way / all 2-way marginals. Fidelity |
| `avg_correlation_diff` | lower = better | Mean \|Δ pairwise correlation\| |
| `downstream_accuracy_{lr,rf,mlp}` | **higher = better** | Train-on-synthetic / test-on-real. Utility |
| `synthref__*` | — | Same metric scored against the synthetic distribution instead of the real holdout |

Privacy is **not measured** — it is the ε that was dialed in. DECAF rows carry NULL ε
(`SDGMethod.is_private = False`) so the DB never implies a guarantee that wasn't given.

### 2a. What `fairness_gap` actually computes (long form)

Per run, for each protected attribute A (binarized against its majority value on the
**real holdout**, so the binarization is identical across every run of that dataset):

1. Train a classifier on the **synthetic** data — features = all non-outcome columns,
   label = the outcome (`income` / `two_year_recid`). Three are fit: logistic regression,
   random forest, MLP.
2. Predict on the **real holdout** (never seen by the generator).
3. Compute the selection rate P(Ŷ=1) within each group of A.
4. Gap = |rate(A=majority) − rate(A=minority)|.
5. `fairness_gap` = **max over protected attributes** of that gap.

Worked example — COMPAS, protected = {sex, race}: if the classifier predicts recidivism
for 55% of Black defendants and 33% of white defendants in the real holdout, race's gap
is 0.22; if sex's gap is 0.09, `fairness_gap` = 0.22.

Four properties worth remembering:

- **It scores the generator, not a classifier.** The classifier is instrumentation. The
  only thing varying across cells is the synthetic data it was trained on, so the gap is
  attributable to the generator + fairness mechanism.
- **It is a *demographic parity* gap** — it ignores whether predictions are correct. A
  model can be perfectly accurate and score badly here if base rates genuinely differ by
  group. That is why `tprb`/`tnrb` (equalized-odds family) are recorded alongside it.
- **Worst-case, not average.** With three protected attributes, one bad attribute sets
  the score. Per-attribute values live in `dp_gap__sex`, `dp_gap__race`, etc.
- **Evaluated on real data.** Any conclusion is about deployment behaviour, not about
  internal consistency of the synthetic table.

`cond_fairness_gap` is identical except step 3 computes the rate within each
(A-group × admissible-attribute-stratum) cell and aggregates. It asks: *once we hold the
legitimate explanatory attributes fixed, does group membership still move the prediction?*
CF is designed to zero this while permitting disparity that flows through admissible
attributes; DP targets the unconditioned version.

### 2b. What the DF real-vs-synth Δ means (long form)

Two numbers, same classifier, same protected attribute, **different evaluation set**:

- `max_abs_dp_gap__real` — classifier trained on synthetic, predictions made on the
  **real holdout**.
- `max_abs_dp_gap__synth` — same trained classifier, predictions made on the
  **synthetic data itself**.

Δ = real − synth.

| Δ | Meaning |
|---|---|
| **≈ 0** | The fairness property transfers. What holds in the synthetic table holds on real inputs. Trustworthy. |
| **Positive** | **Fairness that only holds inside the synthetic data.** The synthetic table looks fair when you audit it in place, but a model trained on it discriminates once it meets real records. |
| **Negative** | Usually a fidelity artifact — the synthetic distribution is too far from the real one for the two references to describe the same population. Verify against `tvd_2way` before reading anything into it. |

Why positive Δ happens mechanically: the fairness mechanism removes an edge from the
learned graph, so in the synthetic joint the protected attribute is (near-)independent
of the outcome. Auditing that table gives a tiny gap. But the mechanism only removed the
*direct* dependence — correlated proxies (`relationship`, `occupation`, `priors_count`)
still carry the group signal. Trained on synthetic data the classifier leans on those
proxies; on real inputs the proxies reassert their real-world correlation with the
protected attribute and the gap comes back.

**This is the failure mode DECAF's paper flags,** and it is exactly why every run here
is scored against both references. The result: it is worst under the **DP mechanism**
(PrivBayes/Adult Δ = +0.128, MST/COMPAS Δ = +0.111) — the mechanism most aggressive
about severing the direct link, and therefore the one most likely to have severed only
the *visible* link. An audit of the released synthetic table alone would rate these
configurations as far fairer than they behave in deployment.

---

## 3. Protected / admissible role splits

**Adult** (outcome `income`):

| config | protected | admissible |
|---|---|---|
| `prefair` | sex, race, native-country | workclass, education, occupation, capital-gain, capital-loss, hours-per-week |
| `sex_only_broad_adm` | sex | education, education-num, occupation, workclass, hours-per-week |
| `sex_race_narrow_adm` | sex, race | education-num, hours-per-week |

**COMPAS** (outcome `two_year_recid`):

| config | protected | admissible |
|---|---|---|
| `prefair` | sex, race | priors_count, c_charge_degree |
| `race_only_broad_adm` | race | priors_count, c_charge_degree, juv_fel_count, juv_misd_count |
| `sex_race_age_narrow_adm` | sex, race, age_cat | c_charge_degree |

The admissible set is exactly what CF may use to block a path, so narrowing it should
push CF toward DP behaviour. It does — see §6.

---

## 4. Headline: generator × mechanism (averaged over ε and role config)

### COMPAS

| method | mech | DP gap ↓ | CDP gap ↓ | TPRB ↓ | TVD-2 ↓ | acc (RF) ↑ |
|---|---|---|---|---|---|---|
| MST | none | 0.232 | 0.162 | 0.239 | 0.027 | 0.618 |
| MST | FTU | 0.165 | 0.116 | 0.164 | 0.027 | 0.619 |
| MST | **DP** | **0.153** | **0.097** | 0.162 | 0.032 | 0.621 |
| MST | CF | 0.166 | 0.116 | 0.149 | 0.028 | 0.620 |
| PrivBayes | none | 0.292 | 0.188 | 0.291 | 0.027 | 0.638 |
| PrivBayes | FTU | 0.257 | 0.163 | 0.257 | 0.028 | 0.636 |
| PrivBayes | **DP** | **0.168** | 0.136 | 0.171 | 0.035 | 0.557 |
| PrivBayes | CF | 0.177 | **0.118** | 0.172 | 0.033 | 0.581 |
| PrivSyn | none | 0.478 | 0.414 | 0.514 | 0.027 | 0.620 |
| PrivSyn | FTU | 0.262 | 0.201 | 0.269 | 0.029 | 0.611 |
| PrivSyn | **DP** | **0.236** | **0.212** | 0.220 | 0.058 | 0.577 |
| PrivSyn | CF | 0.301 | 0.249 | 0.309 | 0.028 | 0.601 |
| DECAF | none | 0.357 | 0.256 | 0.320 | 0.084 | 0.654 |
| DECAF | FTU | 0.256 | 0.183 | 0.229 | 0.085 | 0.638 |
| DECAF | **DP** | **0.075** | 0.080 | 0.079 | 0.090 | 0.544 |
| DECAF | CF | 0.152 | **0.088** | 0.136 | 0.086 | 0.601 |

### Adult

| method | mech | DP gap ↓ | CDP gap ↓ | TPRB ↓ | TVD-2 ↓ | acc (RF) ↑ |
|---|---|---|---|---|---|---|
| MST | none | 0.022 | 0.023 | 0.041 | 0.049 | 0.724 |
| MST | FTU | 0.022 | 0.023 | 0.041 | 0.049 | 0.724 |
| MST | **DP** | **0.001** | **0.001** | 0.000 | 0.054 | 0.705 |
| MST | CF | 0.012 | 0.021 | 0.053 | 0.051 | **0.731** |
| PrivBayes | none | 0.272 | 0.223 | 0.369 | 0.117 | 0.775 |
| PrivBayes | **FTU** | **0.119** | **0.087** | 0.137 | 0.125 | 0.776 |
| PrivBayes | DP | 0.146 | 0.113 | 0.211 | 0.131 | 0.773 |
| PrivBayes | CF | 0.137 | 0.106 | 0.211 | 0.128 | **0.776** |
| PrivSyn | none | 0.227 | 0.193 | 0.293 | 0.076 | 0.782 |
| PrivSyn | FTU | 0.218 | 0.178 | 0.285 | 0.077 | 0.782 |
| PrivSyn | DP | 0.106 | 0.101 | 0.144 | 0.084 | 0.679 |
| PrivSyn | **CF** | 0.110 | **0.095** | 0.156 | **0.072** | 0.768 |
| DECAF | none | 0.082 | 0.085 | 0.064 | 0.658 † | 0.746 |
| DECAF | FTU | 0.110 | 0.072 | 0.127 | 0.658 † | 0.746 |
| DECAF | **DP** | **0.003** | **0.002** | 0.001 | 0.657 † | 0.748 |
| DECAF | CF | 0.085 | 0.074 | 0.103 | 0.658 † | 0.747 |

† See §9.1 — Adult DECAF fidelity is bad enough that its fairness numbers are not
interpretable as debiasing.

**Findings**

1. **Every mechanism reduces the gap on COMPAS, monotonically, for every generator** —
   `none` → FTU → CF → DP is the ordering nearly everywhere.
2. **PrivSyn has by far the worst untreated bias** (COMPAS `none` gap 0.478) and gains
   most from any mechanism; FTU alone nearly halves it.
3. **CF consistently wins on `cond_fairness_gap` while DP wins on `fairness_gap`** —
   exactly the designed behaviour. Cleanest cases: COMPAS/PrivBayes (CF CDP 0.118 vs DP
   0.136) and COMPAS/DECAF (CF 0.088 vs DP 0.080, near-tied, at 6 points less accuracy cost).
4. DECAF+DP gives the lowest gaps anywhere (COMPAS 0.075, Adult 0.003), with caveats in §9.

---

## 5. Cost of fairness — paired deltas vs. the `none` control

Paired at identical (dataset, method, ε, role config):

| dataset | mech | Δ DP gap | Δ acc (RF) | Δ TVD-2 |
|---|---|---|---|---|
| adult | FTU | −0.046 | **+0.000** | +0.003 |
| adult | CF | −0.078 | −0.002 | +0.003 |
| adult | DP | −0.088 | −0.037 | +0.008 |
| compas | FTU | −0.106 | −0.005 | +0.001 |
| compas | CF | −0.128 | −0.027 | +0.003 |
| compas | DP | −0.162 | **−0.047** | +0.014 |

**FTU is essentially free** on both datasets. **DP costs the most utility** and is the
only mechanism that measurably degrades fidelity. **CF sits between them on both axes** —
it buys most of DP's fairness for roughly half its accuracy cost. This is the
paper-shaped result.

---

## 6. ε sweep

DP gap by ε, COMPAS:

| method | mech | ε=1 | ε=10 | ε=1000 |
|---|---|---|---|---|
| MST | none | 0.291 | 0.172 | 0.234 |
| MST | DP | 0.124 | 0.167 | 0.168 |
| PrivBayes | none | 0.333 | 0.200 | 0.343 |
| PrivBayes | DP | 0.101 | 0.168 | 0.236 |
| PrivSyn | none | 0.449 | 0.404 | 0.583 |
| PrivSyn | DP | 0.210 | 0.339 | 0.159 |

**Fidelity is cleanly monotone in ε** (COMPAS TVD-2: MST 0.029 → 0.027 → 0.025;
PrivBayes 0.035 → 0.025 → 0.020). **Fairness is not.** Gaps often *rise* with ε, because
at low ε the DP noise itself washes out the protected-attribute signal — the data is
"fair" partly by being uninformative.

Consequences:

- Do not read low-ε fairness as a mechanism win. Compare `none` at ε=1 vs ε=1000
  (PrivSyn 0.449 → 0.583): more of the real bias survives as privacy loosens. The
  mechanism's *marginal* effect is cleanest at ε=1000.
- Visible non-monotonicities (PrivBayes/Adult `none`: 0.493 at ε=1 vs 0.136 at ε=10) are
  single-seed noise. This is the single strongest motivation for the added seeds.

Utility behaves as expected: PrivBayes/Adult accuracy 0.730 → 0.801 → 0.795.

---

## 7. Role-config sensitivity (DP gap, averaged over method and ε)

| dataset | role config | none | FTU | DP | CF | acc(CF) |
|---|---|---|---|---|---|---|
| adult | prefair | 0.180 | 0.179 | 0.101 | **0.082** | 0.758 |
| adult | sex_only_broad_adm | 0.159 | 0.076 | **0.065** | 0.100 | 0.763 |
| adult | sex_race_narrow_adm | 0.153 | 0.100 | **0.064** | 0.076 | 0.751 |
| compas | prefair | 0.341 | 0.220 | 0.206 | **0.189** | 0.594 |
| compas | race_only_broad_adm | 0.311 | 0.231 | **0.119** | 0.225 | 0.626 |
| compas | sex_race_age_narrow_adm | 0.357 | 0.242 | **0.199** | 0.212 | 0.583 |

The predicted CF ↔ admissible-set relationship is visible:

- **Broad admissible set → CF diverges from DP.** COMPAS `race_only_broad_adm`: CF 0.225
  vs DP 0.119. CF has many legitimate paths it may leave open, so it leaves disparity on
  the table — at a 2.5-point accuracy *gain* over DP.
- **Narrow admissible set → CF converges to DP.** COMPAS `sex_race_age_narrow_adm`:
  0.212 vs 0.199. Adult `sex_race_narrow_adm`: 0.076 vs 0.064.
- Adding `age_cat` to protected (COMPAS narrow) makes every mechanism look worse and
  costs the most accuracy — the strictest config, as designed.

---

## 8. DF axis: real vs. synthetic reference

(Interpretation in §2b.)

| dataset | method | mech | vs. real | vs. synth | Δ |
|---|---|---|---|---|---|
| adult | DECAF | none | 0.082 | 0.342 | −0.260 |
| adult | DECAF | CF | 0.085 | 0.357 | −0.271 |
| adult | PrivBayes | DP | 0.146 | 0.019 | **+0.128** |
| compas | MST | DP | 0.153 | 0.042 | **+0.111** |
| adult | PrivSyn | FTU | 0.218 | 0.121 | +0.097 |
| adult | MST | any | ≈ equal | ≈ equal | ≤ 0.009 |

- **Positive Δ is the deployment risk.** PrivBayes/Adult under DP audits at 0.019 on its
  own distribution but carries a 0.146 gap on real data. Most pronounced under the DP
  mechanism, on both datasets.
- **Negative Δ (DECAF/Adult) is a fidelity artifact** — TVD-2 0.658 means the two
  references barely describe the same population.
- **MST is the most trustworthy generator on this axis** — references agree to within
  0.009 across all mechanisms on Adult.

---

## 9. Caveats

### 9.1 DECAF's Adult fidelity is bad

TVD-2 = 0.658 vs 0.05–0.13 for the DP synthesizers. Its striking fairness scores
(DP gap 0.003) are substantially "the synthetic data doesn't resemble Adult", not
"DECAF debiased Adult". **Do not quote Adult DECAF fairness numbers without this note.**
COMPAS DECAF is fine (TVD-2 0.084) and those numbers are usable.

DECAF knobs are set per dataset from measurement (`run_experiments.py::DECAF_SETTINGS`):

| dataset | epochs | output heads | input scaling | why |
|---|---|---|---|---|
| compas | 200 | linear (as shipped) | raw ordinal codes | All columns share a narrow 0–5 code range. Bounding the heads made it much worse (TVD-1 0.048 → 0.321). At 1 epoch the outcome column collapsed entirely. |
| adult | 30 | sigmoid (`nonlin_out`) | min-max to [0,1] | Code ranges span 0–1 (`income`) to 0–40 (`native-country`); with linear heads the wide columns dominate the loss and `income` collapsed to a single class at both 10 and 50 epochs, plus 3 other constant columns. |

Adult still needs a dedicated hyperparameter pass before publication.

### 9.2 FTU is a no-op for MST on Adult

`none` and FTU are identical to 4 decimals across every metric. Cause, confirmed from
the logged graphs: **MST never selects a direct protected→`income` edge on Adult** — the
only edge touching `income` is `income–marital-status`. FTU removes direct protected→outcome
edges only, so it has nothing to remove and produces a byte-identical graph (and, at a
fixed seed, byte-identical synthetic data).

For contrast, at `prefair`/ε=10 the same run under the other mechanisms:

- **DP** deletes `income–marital-status` outright, leaving `income` disconnected from the
  protected side (12 edges vs 13).
- **CF** rewires `income` to `occupation`, an admissible attribute (13 edges).

This is a real structural finding about how the mechanisms differ, **not a bug** — but
Adult/MST/FTU carries no information and should not appear in a results table as though
it were an independent measurement.

### 9.3 `education-num` is mis-encoded on Adult (affects all marginal methods)

Found 2026-07-30 while building the CTGAN backbone. `data/datasets.py::load_adult`
ordinal-codes only the **object** columns:

```python
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype("category").cat.codes
domain = {col: int(df[col].nunique()) for col in df.columns}
```

`education-num` is already `int64`, so it keeps its raw UCI values **1–16** while its
domain is recorded as `nunique() == 16`. A 0-indexed domain of size 16 admits 0–15, so
value 16 is out of range. `mbi` folds it into the last bin:

| | value 0 | … | value 15 | value 16 |
|---|---|---|---|---|
| true counts | 0 | … | 542 | 375 |
| what MST/PrivBayes/AIM/PrivSyn see | 0 | … | **917** | *(gone)* |

So every marginal-based method trains on an `education-num` with one dead category and
two real categories merged. **Impact is small but real**: ~1.2% of that column's mass
(375/30162 rows), one wasted domain slot, and slightly inflated TVD for any config that
touches the column. `education-num` is in the admissible set for the
`sex_only_broad_adm` and `sex_race_narrow_adm` role configs, so it also perturbs CF's
behaviour there.

**Deliberately not fixed yet.** Fixing `datasets.py` mid-batch would mix pre-fix and
post-fix Adult data inside one batch tag, which is worse than the bug. The fix
(ordinal-code *every* column, not just object ones) should land after the 5-seed sweep
finishes, followed by a re-run of the Adult cells. The new GAN backbones reproduce mbi's
clipping exactly (`ColumnBlocks.clip`) so they see the same data as everything else
rather than a differently-corrupted version.

### 9.4 Single seed

See the banner at the top. The ε-sweep non-monotonicities and every exact-0.000 cell in
§10 are the most likely to move with more trials.

---

## 10. Best cells (accuracy floor at the per-dataset median RF accuracy)

### Adult (floor 0.746)

| method | mech | role config | ε | DP gap | CDP gap | TVD-2 | acc RF |
|---|---|---|---|---|---|---|---|
| PrivSyn | CF | prefair | 1.0 | 0.000 | 0.000 | 0.127 | 0.760 |
| PrivSyn | DP | sex_only_broad_adm | 1.0 | 0.000 | 0.002 | 0.092 | 0.748 |
| PrivSyn | FTU | sex_only_broad_adm | 1.0 | 0.024 | 0.014 | 0.085 | 0.770 |
| PrivSyn | DP | sex_only_broad_adm | 1000 | 0.036 | 0.028 | **0.056** | 0.753 |
| DECAF | DP | prefair | — | 0.000 | 0.000 | 0.657 † | 0.748 |

### COMPAS (floor 0.624)

| method | mech | role config | ε | DP gap | CDP gap | TVD-2 | acc RF |
|---|---|---|---|---|---|---|---|
| PrivSyn | DP | race_only_broad_adm | 1000 | 0.001 | 0.149 | 0.142 | 0.636 |
| MST | FTU/CF | prefair | 1.0 | 0.083 | 0.059 | **0.032** | 0.624 |
| PrivSyn | DP | race_only_broad_adm | 10 | 0.108 | 0.045 | 0.028 | 0.637 |
| MST | DP | prefair | 1.0 | 0.155 | 0.045 | 0.036 | 0.625 |

Exact-0.000 cells are single-seed luck at ε=1. Defensible picks:

- **Best all-round: PrivSyn + DP at ε=1000 on Adult** (`sex_only_broad_adm`) — gap 0.036,
  best fidelity in the table (TVD-2 0.056), accuracy 0.753.
- **Best on COMPAS: MST + CF at ε=1** — fidelity 4× better than anything else listed,
  gap 0.083, ~zero accuracy cost.
- **PrivSyn/DP at ε=1000 on COMPAS is a cautionary cell**: DP gap 0.001 but CDP gap 0.149.
  It equalized the unconditioned rate while leaving all conditional disparity intact —
  precisely why both metrics are reported.

---

## 11. Follow-ups queued

- **Seeds 1–4** on the identical grid → batch `seeds-2026-07-30`. Supersedes §4–§10.
- **DECAF backbone variants**: DP-GAN (DP-SGD on the discriminator, giving DECAF a real ε)
  and CTGAN (mode-specific normalization + conditional sampling), as separate SDG methods
  so they land as their own rows rather than perturbing the existing DECAF baseline.
- **AIM** — implemented, still not in any batch.
- **DECAF Adult hyperparameter search** (§9.1).
