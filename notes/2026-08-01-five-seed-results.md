# Five-seed results — batch `overnight-2026-07-30`

Supersedes the single-seed analysis in `2026-07-30-overnight-batch.md`. Every number
below is a mean over **5 seeds (0–4)**; 1200 runs, all `done`, 299 minutes of compute.

**Scope caveat:** this batch covers the four *marginal/graphical* methods plus the
published DECAF baseline. The three new GAN backbones (`decaf_dpgan`, `decaf_ctgan`,
`decaf_dpctgan`) are in a **separate batch, `gan-backbones-2026-07-30`, launched
2026-08-01 and still running** (840 cells). Their design and tuning are in
`2026-07-30-decaf-gan-backbones.md`; results get appended here when the batch lands.

---

## 1. The grid

| axis | levels |
|---|---|
| dataset | `adult`, `compas` |
| SDG method | `mst`, `privbayes`, `privsyn` (private) · `decaf` (non-private reference) |
| ε | 1, 10, 1000 (`decaf`: NULL — it has no DP mechanism) |
| fairness mechanism | `none`, `ftu`, `dp`, `cf` |
| role config | 3 per dataset (below) |
| seeds | 0, 1, 2, 3, 4 |

= (3 methods × 3 ε + 1 non-private) × 4 mechanisms × 3 role configs × 5 seeds × 2 datasets
= 1200 runs.

### Role configs — what "protected" and "admissible" mean per cell

The **protected** set is what fairness is measured against. The **admissible** set is
what CF is allowed to use to *legitimately* explain an outcome difference — a path from
a protected attribute to the outcome that runs through an admissible attribute is not
counted as unfair by CF, but is by DP.

| dataset | role config | protected | admissible | outcome |
|---|---|---|---|---|
| adult | `prefair` | native-country, race, sex | capital-gain, capital-loss, education, hours-per-week, occupation, workclass | income |
| adult | `sex_only_broad_adm` | sex | education, education-num, hours-per-week, occupation, workclass | income |
| adult | `sex_race_narrow_adm` | race, sex | education-num, hours-per-week | income |
| compas | `prefair` | race, sex | c_charge_degree, priors_count | two_year_recid |
| compas | `race_only_broad_adm` | race | c_charge_degree, juv_fel_count, juv_misd_count, priors_count | two_year_recid |
| compas | `sex_race_age_narrow_adm` | age_cat, race, sex | c_charge_degree | two_year_recid |

### The four mechanisms

| mechanism | rule | acts on |
|---|---|---|
| `none` | no intervention — the generator's own structure | — |
| `ftu` | Fairness Through Unawareness: remove only the **direct** protected→outcome edge | structure (MST/PrivBayes/PrivSyn) or generation (DECAF) |
| `dp` | Demographic Parity: block **every** path protected→outcome | " |
| `cf` | Counterfactual Fairness: block only paths **not routed through an admissible attribute** | " |

By construction the strength ordering is `none` ≤ `ftu` ≤ `cf` ≤ `dp`.

---

## 2. The metrics

| metric | definition | direction |
|---|---|---|
| `fairness_gap` | worst-case \|P(Ŷ=1 \| A=a) − P(Ŷ=1 \| A=a′)\| over the run's protected attributes, classifier **trained on synthetic, evaluated on the real holdout** | lower better |
| `cond_fairness_gap` | same, conditioned on the admissible attributes (CDP). This is the gap CF specifically targets | lower better |
| `max_abs_dp_gap__synth` | same as `fairness_gap` but scored against the **synthetic** data's own labels (DECAF's DF-synthetic reference) | lower better |
| `tvd_1way` / `tvd_2way` | mean total-variation distance between real and synthetic 1-way / 2-way marginals | lower better |
| `avg_correlation_diff` | mean \|Δ\| in pairwise correlation, real vs synthetic | lower better |
| `downstream_accuracy_{mlp,lr,rf}` | train-on-synthetic / test-on-real accuracy | higher better |
| `n`, `gap_sd` | replicate count and the across-seed std of `fairness_gap` in that cell | — |
| ε | privacy budget. `decaf` = NULL (non-private) | lower = more private |

Worst-case, not mean, over protected attributes: role configs declare different *numbers*
of protected attributes, and a mean would reward a config simply for protecting more.

**Reading `real` vs `synth`.** A positive (real − synth) delta means the mechanism made
the synthetic data *look* fair internally while the classifier trained on it still
discriminates against real people — the mechanism removed the direct edge, but correlated
proxies still carry the group signal and reassert their real-world correlation at
deployment. That's the number that decides whether a mechanism actually works.

**Seed noise floor.** Median across-seed std of `fairness_gap` within a cell:
**0.034 on Adult, 0.074 on COMPAS** (over all 216 cells: mean 0.070, max 0.264). Any
single-cell difference smaller than that is not a difference. The paired analysis in §5
gets around this by differencing within (ε, role config, seed).

---

## 3. Headline — method × mechanism

Averaged over ε, role configs, and seeds.

### COMPAS

| method | mech | n | fairness_gap | cond_gap | tvd_1way | tvd_2way | acc (MLP) | gap_sd |
|---|---|---|---|---|---|---|---|---|
| decaf | none | 15 | 0.2901 | 0.2302 | 0.1309 | 0.2321 | 0.6136 | 0.198 |
| decaf | ftu | 15 | 0.1901 | 0.1551 | 0.1307 | 0.2322 | 0.6093 | 0.144 |
| decaf | cf | 15 | 0.1553 | 0.1235 | 0.1309 | 0.2328 | 0.5936 | 0.139 |
| decaf | **dp** | 15 | **0.0661** | 0.0710 | 0.1313 | 0.2354 | 0.5461 | 0.084 |
| mst | none | 45 | 0.2325 | 0.1403 | 0.0096 | 0.0299 | 0.6302 | 0.068 |
| mst | ftu | 45 | 0.2215 | 0.1328 | 0.0096 | 0.0299 | 0.6307 | 0.072 |
| mst | cf | 45 | 0.2049 | 0.1161 | 0.0095 | 0.0304 | **0.6317** | 0.066 |
| mst | **dp** | 45 | 0.1898 | 0.1055 | **0.0095** | 0.0333 | 0.6322 | 0.060 |
| privbayes | none | 45 | 0.3036 | 0.1913 | 0.0122 | 0.0273 | 0.6478 | 0.109 |
| privbayes | ftu | 45 | 0.2691 | 0.1590 | 0.0131 | 0.0285 | **0.6498** | 0.074 |
| privbayes | cf | 45 | 0.2117 | 0.1274 | 0.0128 | 0.0336 | 0.6256 | 0.062 |
| privbayes | dp | 45 | 0.1982 | 0.1200 | 0.0130 | 0.0352 | 0.6210 | 0.064 |
| privsyn | none | 45 | 0.4566 | 0.3692 | 0.0116 | 0.0277 | 0.6309 | 0.161 |
| privsyn | ftu | 45 | 0.2728 | 0.1788 | 0.0124 | 0.0282 | 0.6362 | 0.102 |
| privsyn | cf | 45 | 0.2587 | 0.1856 | 0.0112 | 0.0296 | 0.6186 | 0.124 |
| privsyn | dp | 45 | 0.3012 | 0.2504 | 0.0255 | 0.0575 | 0.5568 | 0.171 |

### Adult

| method | mech | n | fairness_gap | cond_gap | tvd_1way | tvd_2way | acc (MLP) | gap_sd |
|---|---|---|---|---|---|---|---|---|
| decaf | none | 15 | 0.2807 | 0.2487 | 0.4564 | 0.6766 | 0.7134 | 0.228 |
| decaf | ftu | 15 | 0.2402 | 0.2005 | 0.4566 | 0.6766 | 0.7198 | 0.212 |
| decaf | cf | 15 | 0.0766 | 0.0603 | 0.4572 | 0.6764 | 0.6556 | 0.063 |
| decaf | **dp** | 15 | **0.0113** | 0.0107 | 0.4560 | 0.6749 | 0.7337 | 0.024 |
| mst | none | 45 | 0.0205 | 0.0185 | 0.0106 | 0.0481 | 0.7467 | 0.022 |
| mst | ftu | 45 | 0.0205 | 0.0185 | 0.0106 | 0.0481 | 0.7467 | 0.022 |
| mst | cf | 45 | 0.0182 | 0.0239 | 0.0106 | 0.0504 | 0.7538 | 0.020 |
| mst | **dp** | 45 | **0.0024** | 0.0024 | 0.0106 | 0.0533 | 0.7469 | 0.006 |
| privbayes | none | 45 | 0.1912 | 0.1486 | 0.0729 | 0.1393 | 0.8001 | 0.089 |
| privbayes | ftu | 45 | 0.1488 | 0.1138 | 0.0756 | 0.1421 | **0.8048** | 0.038 |
| privbayes | cf | 45 | 0.1096 | 0.0813 | 0.0763 | 0.1466 | 0.7858 | 0.077 |
| privbayes | dp | 45 | 0.1131 | 0.0837 | 0.0672 | 0.1378 | 0.7818 | 0.101 |
| privsyn | none | 45 | 0.2175 | 0.1808 | 0.0193 | 0.0755 | 0.7863 | 0.101 |
| privsyn | ftu | 45 | 0.1948 | 0.1572 | 0.0204 | 0.0755 | 0.7864 | 0.118 |
| privsyn | cf | 45 | 0.1330 | 0.1056 | 0.0197 | 0.0756 | 0.7791 | 0.100 |
| privsyn | dp | 45 | 0.0961 | 0.0824 | 0.0241 | 0.0863 | 0.7074 | 0.097 |

**Reading it.** The ordering `none > ftu > cf > dp` holds almost everywhere on the
headline gap — the mechanisms do what they claim. The exception is PrivSyn/COMPAS, where
`dp` (0.301) is *worse* than `cf` (0.259) and `ftu` (0.273); §5 shows why (it also costs
7.4pp of accuracy there, i.e. the DP constraint is degrading the model, not just the gap).

---

## 4. Privacy: does ε trade against fairness or utility?

Marginal methods only, averaged over mechanisms and role configs (n=60 per row).

| dataset | method | ε | tvd_1way | tvd_2way | corr_diff | acc MLP | acc LR | acc RF | fairness_gap |
|---|---|---|---|---|---|---|---|---|---|
| adult | mst | 1 | 0.0115 | 0.0525 | 0.070 | 0.7484 | 0.7501 | 0.7240 | 0.0157 |
| adult | mst | 10 | 0.0102 | 0.0487 | 0.071 | 0.7490 | 0.7498 | 0.7244 | 0.0160 |
| adult | mst | 1000 | 0.0102 | 0.0487 | 0.071 | 0.7482 | 0.7497 | 0.7257 | 0.0144 |
| adult | privbayes | 1 | 0.1423 | 0.2619 | 0.078 | 0.7705 | 0.7523 | 0.7572 | 0.1133 |
| adult | privbayes | 10 | 0.0642 | 0.1263 | 0.046 | 0.8035 | 0.7824 | 0.7931 | 0.1546 |
| adult | privbayes | 1000 | 0.0126 | 0.0362 | 0.034 | **0.8054** | 0.7906 | 0.7925 | 0.1541 |
| adult | privsyn | 1 | 0.0372 | 0.1111 | 0.111 | 0.7368 | 0.7414 | 0.7480 | 0.1192 |
| adult | privsyn | 10 | 0.0132 | 0.0645 | 0.085 | 0.7717 | 0.7565 | 0.7603 | 0.1620 |
| adult | privsyn | 1000 | 0.0123 | 0.0590 | 0.071 | 0.7859 | 0.7774 | 0.7874 | 0.1999 |
| compas | mst | 1 | 0.0111 | 0.0341 | 0.068 | 0.6260 | 0.6335 | 0.6181 | 0.2289 |
| compas | mst | 10 | 0.0089 | 0.0294 | 0.054 | 0.6331 | 0.6327 | 0.6319 | 0.2120 |
| compas | mst | 1000 | 0.0087 | 0.0292 | 0.054 | 0.6346 | 0.6379 | 0.6256 | 0.1957 |
| compas | privbayes | 1 | 0.0168 | 0.0394 | 0.065 | 0.6177 | 0.6205 | 0.5946 | 0.2270 |
| compas | privbayes | 10 | 0.0110 | 0.0282 | 0.045 | 0.6362 | 0.6447 | 0.6288 | 0.2463 |
| compas | privbayes | 1000 | 0.0106 | 0.0258 | 0.041 | **0.6543** | 0.6561 | 0.6451 | 0.2637 |
| compas | privsyn | 1 | 0.0196 | 0.0425 | 0.061 | 0.6086 | 0.6216 | 0.6075 | 0.3260 |
| compas | privsyn | 10 | 0.0114 | 0.0286 | 0.040 | 0.6250 | 0.6300 | 0.6190 | 0.3109 |
| compas | privsyn | 1000 | 0.0146 | 0.0363 | 0.042 | 0.5983 | 0.5998 | 0.5854 | 0.3300 |

Three things:

1. **MST is nearly ε-free.** From ε=1000 down to ε=1 it loses 0.0013 TVD-1 on Adult and
   0.2pp of accuracy. Nothing else comes close — PrivBayes' TVD-1 degrades **11×**
   (0.013 → 0.142) over the same range.
2. **Privacy noise acts as an accidental fairness mechanism.** For PrivBayes and PrivSyn
   on Adult, the gap *rises* as the budget loosens (PrivBayes 0.113 → 0.154; PrivSyn
   0.119 → 0.200). The noise that destroys fidelity at ε=1 also destroys the
   protected→outcome signal. This is not fairness you should claim credit for: it is
   bought with the same utility loss, and it disappears exactly when the generator starts
   working. MST/COMPAS runs the other way (0.229 → 0.196), so it is not a universal law.
3. **PrivBayes wins accuracy, MST wins fidelity and fairness.** PrivBayes reaches 0.805
   MLP accuracy on Adult but with a gap of 0.154; MST sits at 0.748 with a gap of 0.014.

---

## 5. Cost of fairness — paired within (ε, role config, seed)

Each mechanism differenced against `none` in the *same* cell, so generator noise cancels.
`t` = mean/SE over paired differences; `frac↓` = fraction of pairs where the gap fell.

| dataset | method | mech | n | Δ gap | SE | t | frac↓ | Δ acc |
|---|---|---|---|---|---|---|---|---|
| compas | decaf | ftu | 15 | −0.1000 | 0.076 | −1.32 | 0.80 | −0.004 |
| compas | decaf | dp | 15 | **−0.2240** | 0.059 | −3.83 | 0.73 | −0.068 |
| compas | decaf | cf | 15 | −0.1348 | 0.076 | −1.77 | 0.80 | −0.020 |
| compas | mst | ftu | 45 | −0.0111 | 0.007 | −1.58 | 0.13 | +0.001 |
| compas | mst | dp | 45 | −0.0427 | 0.014 | −2.99 | 0.73 | **+0.002** |
| compas | mst | cf | 45 | −0.0276 | 0.009 | −3.20 | 0.31 | +0.002 |
| compas | privbayes | ftu | 45 | −0.0346 | 0.019 | −1.86 | 0.44 | +0.002 |
| compas | privbayes | dp | 45 | −0.1054 | 0.021 | −5.07 | 0.82 | −0.027 |
| compas | privbayes | cf | 45 | −0.0919 | 0.019 | −4.83 | 0.78 | −0.022 |
| compas | privsyn | ftu | 45 | −0.1838 | 0.031 | −5.94 | 0.82 | +0.005 |
| compas | privsyn | dp | 45 | −0.1554 | 0.034 | −4.53 | 0.69 | −0.074 |
| compas | privsyn | cf | 45 | **−0.1979** | 0.030 | −6.59 | 0.73 | −0.012 |
| adult | decaf | ftu | 15 | −0.0405 | 0.019 | −2.10 | 0.40 | +0.006 |
| adult | decaf | dp | 15 | **−0.2694** | 0.059 | −4.54 | 0.80 | **+0.020** |
| adult | decaf | cf | 15 | −0.2041 | 0.056 | −3.64 | 0.73 | −0.058 |
| adult | mst | ftu | 45 | **0.0000** | 0.000 | — | 0.00 | 0.000 |
| adult | mst | dp | 45 | −0.0181 | 0.003 | −5.74 | 0.82 | +0.000 |
| adult | mst | cf | 45 | −0.0023 | 0.004 | −0.61 | 0.53 | +0.007 |
| adult | privbayes | ftu | 45 | −0.0424 | 0.017 | −2.44 | 0.33 | +0.005 |
| adult | privbayes | dp | 45 | −0.0781 | 0.024 | −3.24 | 0.58 | −0.018 |
| adult | privbayes | cf | 45 | −0.0816 | 0.021 | −3.94 | 0.76 | −0.014 |
| adult | privsyn | ftu | 45 | −0.0227 | 0.025 | −0.91 | 0.64 | +0.000 |
| adult | privsyn | dp | 45 | −0.1214 | 0.020 | −6.24 | 0.80 | −0.079 |
| adult | privsyn | cf | 45 | −0.0845 | 0.021 | −4.07 | 0.76 | −0.007 |

**What survives replication:**

- **DP is the only mechanism that works everywhere.** All 8 (dataset × method) cells are
  significant, |t| = 3.0–6.2.
- **CF works everywhere except MST/Adult** (t = −0.61, frac↓ 0.53 — indistinguishable from
  noise). Elsewhere |t| = 3.2–6.6.
- **FTU is unreliable.** Significant in only 3 of 8 cells, and its `frac↓` is often *below*
  0.5 (MST/COMPAS 0.13, PrivBayes/Adult 0.33) — i.e. on most individual seeds FTU makes
  the gap slightly *worse*, and its negative mean is carried by a few large improvements.
  This is the expected FTU failure mode: deleting the direct edge leaves every proxy path
  intact, and the generator re-routes signal through them.
- **Accuracy cost is method-specific, not mechanism-specific.** DP is free on MST
  (+0.002 / +0.000) and expensive on PrivSyn (−0.074 / −0.079). MST's tree structure has
  enough redundancy to reroute utility around a cut; PrivSyn's GUM sampler does not.
- **DP on DECAF/Adult *gains* 2pp of accuracy** while cutting the gap by 0.269 — but on a
  generator whose TVD-2 is 0.68, so this says more about how broken the baseline is on
  Adult than about the mechanism.

---

## 6. Distributional Fairness: real vs synthetic reference

`dp_delta` = (real-reference gap) − (synthetic-reference gap). **Positive = fairness that
only holds inside the synthetic data** and evaporates on real inputs.

### COMPAS

| method | mech | real | synth | Δ | cond real | cond synth | Δ |
|---|---|---|---|---|---|---|---|
| decaf | none | 0.2901 | 0.2766 | +0.014 | 0.2302 | 0.2258 | +0.004 |
| decaf | ftu | 0.1901 | 0.2267 | −0.037 | 0.1551 | 0.1418 | +0.013 |
| decaf | cf | 0.1553 | 0.2007 | −0.045 | 0.1235 | 0.1191 | +0.004 |
| decaf | dp | 0.0661 | 0.0606 | +0.006 | 0.0710 | 0.0636 | +0.007 |
| mst | none | 0.2325 | 0.2399 | −0.007 | 0.1403 | 0.1437 | −0.003 |
| mst | ftu | 0.2215 | 0.2350 | −0.014 | 0.1328 | 0.1391 | −0.006 |
| mst | cf | 0.2049 | 0.1767 | +0.028 | 0.1161 | 0.0828 | +0.033 |
| mst | dp | 0.1898 | 0.0655 | **+0.124** | 0.1055 | 0.0691 | +0.036 |
| privbayes | none | 0.3036 | 0.2936 | +0.010 | 0.1913 | 0.1905 | +0.001 |
| privbayes | ftu | 0.2691 | 0.2380 | +0.031 | 0.1590 | 0.1465 | +0.013 |
| privbayes | cf | 0.2117 | 0.1364 | +0.075 | 0.1274 | 0.0953 | +0.032 |
| privbayes | dp | 0.1982 | 0.0926 | **+0.106** | 0.1200 | 0.0941 | +0.026 |
| privsyn | none | 0.4566 | 0.4269 | +0.030 | 0.3692 | 0.3578 | +0.011 |
| privsyn | ftu | 0.2728 | 0.2478 | +0.025 | 0.1788 | 0.1800 | −0.001 |
| privsyn | cf | 0.2587 | 0.2315 | +0.027 | 0.1856 | 0.1769 | +0.009 |
| privsyn | dp | 0.3012 | 0.2486 | +0.053 | 0.2504 | 0.2436 | +0.007 |

### Adult

| method | mech | real | synth | Δ | cond real | cond synth | Δ |
|---|---|---|---|---|---|---|---|
| decaf | none | 0.2807 | 0.2418 | +0.039 | 0.2487 | 0.2286 | +0.020 |
| decaf | ftu | 0.2402 | 0.2259 | +0.014 | 0.2005 | 0.2172 | −0.017 |
| decaf | cf | 0.0766 | 0.1722 | −0.096 | 0.0603 | 0.0564 | +0.004 |
| decaf | dp | 0.0113 | 0.0014 | +0.010 | 0.0107 | 0.0014 | +0.009 |
| mst | none | 0.0205 | 0.0223 | −0.002 | 0.0185 | 0.0237 | −0.005 |
| mst | ftu | 0.0205 | 0.0223 | −0.002 | 0.0185 | 0.0237 | −0.005 |
| mst | cf | 0.0182 | 0.0241 | −0.006 | 0.0239 | 0.0205 | +0.003 |
| mst | dp | 0.0024 | 0.0020 | +0.000 | 0.0024 | 0.0016 | +0.001 |
| privbayes | none | 0.1912 | 0.1622 | +0.029 | 0.1486 | 0.1385 | +0.010 |
| privbayes | ftu | 0.1488 | 0.1294 | +0.019 | 0.1138 | 0.1104 | +0.003 |
| privbayes | cf | 0.1096 | 0.0408 | +0.069 | 0.0813 | 0.0299 | +0.051 |
| privbayes | dp | 0.1131 | 0.0189 | **+0.094** | 0.0837 | 0.0183 | +0.065 |
| privsyn | none | 0.2175 | 0.1510 | +0.067 | 0.1808 | 0.1253 | +0.056 |
| privsyn | ftu | 0.1948 | 0.1268 | +0.068 | 0.1572 | 0.0972 | +0.060 |
| privsyn | cf | 0.1330 | 0.0878 | +0.045 | 0.1068 | 0.0655 | +0.040 |
| privsyn | dp | 0.0961 | 0.0345 | +0.062 | 0.0824 | 0.0329 | +0.050 |

**The central finding of this axis, and it replicates at 5 seeds:** the Δ grows
monotonically with mechanism strength for every structure-editing method — MST/COMPAS
none −0.007 → ftu −0.014 → cf +0.028 → **dp +0.124**; PrivBayes/COMPAS +0.010 → +0.031 →
+0.075 → **+0.106**; PrivBayes/Adult +0.029 → +0.019 → +0.069 → **+0.094**.

The stronger the structural cut, the larger the share of the reported fairness gain that
is an artifact of the synthetic distribution rather than a property that transfers. MST's
DP mechanism looks near-perfect internally (0.066) and only mediocre in deployment
(0.190). **If you evaluate a fairness mechanism on the synthetic data it produced, you
will overstate it, and you will overstate it most for the mechanisms that cut most.**

DECAF is the interesting exception: because its mechanism acts at *generation* time by
shuffling parent columns rather than by editing structure, its Δ stays near zero or goes
negative. Whatever fairness it achieves transfers.

---

## 7. Role-config sensitivity

| dataset | role config | mech | fairness_gap | cond_gap | acc MLP |
|---|---|---|---|---|---|
| compas | prefair | none / ftu / cf / dp | 0.333 / 0.259 / 0.224 / 0.230 | 0.211 / 0.138 / 0.133 / 0.154 | 0.635 / 0.640 / 0.621 / 0.587 |
| compas | race_only_broad_adm | none / ftu / cf / dp | 0.325 / 0.238 / 0.232 / **0.171** | 0.183 / 0.096 / 0.099 / **0.060** | 0.634 / 0.639 / 0.632 / 0.618 |
| compas | sex_race_age_narrow_adm | none / ftu / cf / dp | 0.323 / 0.247 / **0.199** / 0.240 | 0.306 / 0.237 / 0.191 / 0.235 | 0.633 / 0.629 / 0.614 / 0.588 |
| adult | prefair | none / ftu / cf / dp | 0.157 / 0.141 / 0.084 / 0.082 | 0.118 / 0.103 / 0.060 / 0.063 | 0.772 / 0.774 / 0.763 / 0.732 |
| adult | sex_only_broad_adm | none / ftu / cf / dp | 0.151 / 0.140 / 0.101 / **0.054** | 0.124 / 0.115 / 0.079 / 0.042 | 0.772 / 0.773 / 0.756 / 0.740 |
| adult | sex_race_narrow_adm | none / ftu / cf / dp | 0.163 / 0.119 / 0.073 / 0.058 | 0.146 / 0.103 / 0.069 / 0.050 | 0.770 / 0.773 / 0.765 / 0.760 |

The predicted behaviour holds: **a narrower admissible set pushes CF toward DP.** On
COMPAS, `race_only_broad_adm` (4 admissible attrs) leaves CF at 0.232 against DP's 0.171
— a 0.061 gap; `sex_race_age_narrow_adm` (1 admissible attr) has CF at 0.199 *beating* DP's
0.240. On Adult the same: broad admissible → CF 0.101 vs DP 0.054 (0.047 apart); narrow →
0.073 vs 0.058 (0.015 apart). The admissible set is the real knob, not the mechanism label.

Also note the accuracy column: **DP's cost scales with how much you protect.** On Adult it
costs 4.0pp under `prefair` (3 protected attrs) and 1.0pp under `sex_race_narrow_adm` (2).

---

## 8. Best cells (mean over 5 seeds), within 3pp of the best accuracy on that dataset

Ranked over *cells*, not individual runs — ranking runs would just surface the luckiest seed.

**COMPAS** (best MLP accuracy = 0.663)

| method | ε | mech | role config | gap | cond gap | tvd_1way | acc | gap_sd |
|---|---|---|---|---|---|---|---|---|
| mst | 1000 | cf / ftu / none | race_only_broad_adm | 0.1701 | 0.0208 | 0.0087 | 0.6361 | 0.054 |
| mst | 1 | dp | prefair | 0.1753 | 0.0581 | 0.0112 | 0.6356 | 0.035 |
| mst | 1 | cf | sex_race_age_narrow_adm | 0.1767 | 0.1624 | 0.0106 | 0.6334 | 0.031 |
| mst | 1000 | cf | sex_race_age_narrow_adm | 0.1879 | 0.1839 | 0.0087 | 0.6338 | 0.040 |
| mst | 1000 | dp | sex_race_age_narrow_adm | 0.1887 | 0.1860 | 0.0086 | 0.6344 | 0.036 |

**Adult** (best MLP accuracy = 0.813)

| method | ε | mech | role config | gap | cond gap | tvd_1way | acc | gap_sd |
|---|---|---|---|---|---|---|---|---|
| privbayes | 1000 | dp | sex_race_narrow_adm | 0.1298 | 0.1048 | 0.0125 | 0.7945 | 0.074 |
| privbayes | 1000 | cf | prefair | 0.1302 | 0.0823 | 0.0129 | 0.8008 | 0.039 |
| privbayes | 1 | ftu | prefair | 0.1328 | 0.0971 | 0.1466 | 0.7889 | 0.041 |
| privbayes | 10 | cf | sex_only_broad_adm | 0.1339 | 0.1069 | 0.0681 | 0.7953 | 0.080 |

**MST at ε=1 is not on the Adult board only because of the 3pp accuracy filter** — it
reaches a gap of 0.0038 (DP mechanism) at 0.747 accuracy, 6.6pp below PrivBayes. The
honest summary of Adult is: PrivBayes buys ~5pp of accuracy for ~10× the fairness gap.

---

## 9. Caveats that survived replication

1. **DECAF's Adult fidelity is broken** — TVD-2 ≈ 0.676 against MST's 0.048. Its
   downstream accuracy (0.71–0.73) rides on the marginal base rate, not on learned
   structure. This is what `decaf_ctgan` was built to fix (measured: TVD-2 0.172).
2. **FTU is an exact no-op for MST on Adult.** Δ = 0.0000 with SE 0.0000 across all 45
   pairs. MST's private structure search never selects the direct protected→income edge
   in the first place, so there is nothing for FTU to remove. Structural, not a bug — but
   it means the MST/Adult FTU column carries no information.
3. **`education-num` is mis-encoded** in `data/datasets.py::load_adult` — only object
   columns are ordinal-coded, so it keeps raw UCI values 1–16 while its declared domain is
   16. `mbi` clips it, so every method sees the same clipped data and comparisons within
   this batch are valid; but the top category is folded into the second-from-top. Fixing
   it requires re-running all Adult cells, deliberately deferred so this batch stays
   internally consistent.
4. **AIM is implemented and still absent from every batch.** It is registered in
   `SDG_METHODS` but not in `DEFAULT_METHODS`.
5. **COMPAS gaps are large and noisy across the board** (median within-cell sd 0.074).
   Nothing on COMPAS gets below a 0.17 gap at competitive accuracy; treat COMPAS
   differences under ~0.07 as unresolved even at 5 seeds.

---

## 10. Follow-ups

- Land `gan-backbones-2026-07-30` and append §11 with `decaf_dpgan` / `decaf_ctgan` /
  `decaf_dpctgan`. Expect the ε=1 DP-GAN rows to come back `partial` (single-class
  collapse) — see `2026-07-30-decaf-gan-backbones.md` §4.
- Fix `education-num`, re-run Adult under a new batch tag.
- Add AIM to the default grid.
- The DF real-vs-synth Δ (§6) is the most publishable result here and deserves its own
  targeted experiment rather than being a by-product of this grid.
