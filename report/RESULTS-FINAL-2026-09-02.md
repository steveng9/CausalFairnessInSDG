# Final sweep results — `rf-2026-09-01`

Generated 2026-09-02, after the sweep completed. **Nothing in `manuscript/` was
edited to produce this** — the working copy here is stale relative to the
version being edited elsewhere. This file carries every number needed to update
the manuscript by hand.

- Merged data: `report/data/rf_2026_09_01.csv` — **1080 runs, 0 failed**,
  60 in each of the 18 dataset × generator blocks.
- Train-on-real reference: `report/data/real_baselines.csv`
- Regenerated figures: `manuscript/figs/results.pdf`, `manuscript/figs/radar.pdf`
- Ready-to-paste Table 1 body: `report/table1_body_FINAL.tex`

Sweep configuration: ε = 10 only for the private generators (non-private ones
run once), **random forest** as the downstream classifier, 5 seeds, 3 role
splits, 4 mechanisms.

---

## 1. The four `TKTK` macros

```latex
\newcommand{\CFAUCKEPT}{0.92}   % AUC under CF / AUC under no mechanism
\newcommand{\AUCMARG}{0.79}     % marginals-based TSTR AUC, Adult
\newcommand{\AUCDIFF}{0.81}     % causal diffusion TSTR AUC, Adult
\newcommand{\AUCREAL}{0.86}     % train-on-real AUC, Adult
```

Exact values: 0.915 (246 matched pairs, both datasets; Adult alone 0.871),
0.794 (n=45), 0.805 (n=30), 0.863 (RF; the MLP figure is 0.867).

---

## 2. Section 4.1 — utility, no-mechanism release (RF, train-on-synthetic / test-on-real)

**Adult** — train-on-real: acc 0.818, macro-F1 0.746, **AUC 0.863**; majority-class acc 0.747

| generator | acc | macro-F1 | AUC |
|---|---|---|---|
| MST | 0.731 | 0.607 | 0.742 |
| PrivBayes | 0.798 | 0.717 | 0.839 |
| PrivSyn | 0.777 | 0.684 | 0.801 |
| DECAF | 0.735 | 0.547 | 0.648 |
| +CTGAN | 0.788 | 0.674 | 0.810 |
| +DP-GAN | 0.746 | 0.427 | **0.427 — below chance** |
| +DP-CTGAN | 0.746 | 0.439 | 0.548 |
| +Diffusion | 0.789 | 0.691 | 0.819 |
| +DP-Diffusion | 0.779 | 0.629 | 0.791 |

**COMPAS** — train-on-real: acc 0.660, macro-F1 0.651, **AUC 0.697**; majority-class acc 0.548

| generator | acc | macro-F1 | AUC |
|---|---|---|---|
| MST | 0.637 | 0.627 | 0.649 |
| PrivBayes | 0.639 | 0.628 | 0.661 |
| PrivSyn | 0.643 | 0.634 | 0.676 |
| DECAF | 0.619 | 0.612 | 0.657 |
| +CTGAN | 0.599 | 0.582 | 0.633 |
| +DP-GAN | — | — | **collapsed, no AUC defined** |
| +DP-CTGAN | 0.493 | 0.430 | 0.509 |
| +Diffusion | 0.632 | 0.623 | 0.669 |
| +DP-Diffusion | 0.602 | 0.587 | 0.621 |

### This resolves M39, and Martine was right

Her objection was that the draft called out MST and DP-GAN as "trouble" while
DECAF, DP-CTGAN and DP-Diffusion also sat below baseline. On AUC the picture is
unambiguous and the call-outs change:

- **MST is no longer a problem.** Its accuracy looked like zero lift (0.731 vs a
  0.747 majority rate), but its AUC is 0.742 — a solidly useful release. The old
  complaint was an artefact of accuracy on an imbalanced table, exactly as
  suspected.
- **DP-GAN is the only genuine failure, and it is worse than the draft said.**
  Adult AUC 0.427 is *below chance*: the release teaches a classifier to rank
  backwards. On COMPAS it collapses to one outcome class and has no AUC at all.
- **DP-CTGAN is marginal** (0.548 Adult, 0.509 COMPAS) — near chance, not below.
- **DECAF, +Diffusion, +DP-Diffusion are all fine** on AUC (0.648–0.819).

So the honest sentence is that one generator family fails, not two, and the
failure is DP-SGD on an adversarial critic — not the marginals synthesizer.

M37 and M38 are answered by the train-on-real rows above.

---

## 3. Table 1 — matched pairs (paste `report/table1_body_FINAL.tex`)

| family | dataset | none gap | FTU | CF | DemPar |
|---|---|---|---|---|---|
| Marginals | Adult | 0.210 | −0.019 ± 0.022 | −0.086 ± 0.024 | −0.108 ± 0.027 |
| Marginals | COMPAS | 0.234 | −0.042 ± 0.026 | −0.056 ± 0.033 | −0.066 ± 0.034 |
| Causal GAN | Adult | 0.088 | −0.013 ± 0.017 | −0.068 ± 0.023 | −0.077 ± 0.025 |
| Causal GAN | COMPAS | 0.199 | −0.017 ± 0.025 | −0.020 ± 0.029 | −0.055 ± 0.034 |
| Diffusion | Adult | 0.159 | −0.001 ± 0.009 | −0.121 ± 0.010 | −0.135 ± 0.011 |
| Diffusion | COMPAS | 0.216 | −0.057 ± 0.037 | −0.054 ± 0.036 | −0.079 ± 0.042 |

### ⚠ The FTU claim has to change

**FTU is no longer distinguishable from zero in 4 of the 6 cells** — marginals/Adult,
GAN/Adult, GAN/COMPAS, and diffusion/Adult. It survives only on marginals/COMPAS
(−0.042 ± 0.026) and diffusion/COMPAS (−0.057 ± 0.037). In the previous sweep
FTU/marginals/Adult *was* significant (−0.022 ± 0.020), so any sentence asserting
that FTU reliably reduces the gap needs weakening.

CF and DemPar remain significant in every cell except CF on GAN/COMPAS
(−0.020 ± 0.029).

This is a **cleaner** result for the paper's thesis, not a worse one: cutting the
single direct edge (FTU) barely moves the gap, while cutting whole paths
(CF, DemPar) reliably does. It reinforces the existing argument about path-based
mechanisms rather than undercutting it.

### Why the gaps are larger than the old table

Old marginals/Adult baseline gap was 0.143; it is now 0.210. This is the RF
switch working as intended. The MLP mostly predicted the majority class
(macro-F1 0.431 on Adult), and a classifier that rarely predicts positive has an
artificially small parity gap. RF actually predicts the positive class
(macro-F1 0.607), so there is real disparity for the mechanism to remove.

---

## 4. Radar figure values

| axis | Marginals | Causal GAN | Causal diffusion |
|---|---|---|---|
| 1-way fidelity | 0.980 | 0.732 | 0.942 |
| 2-way fidelity | 0.947 | 0.603 | 0.885 |
| association structure | 0.951 | 0.883 | 0.908 |
| utility (AUC vs real) | 0.935 | 0.798 | 0.929 |
| fairness gain (CF) | 0.319 | 0.448 | 0.539 |
| utility kept under the cut | 0.950 | 0.916 | 0.854 |

Marginals win fidelity and utility; **diffusion wins the fairness lever** and
pays for it in utility kept. The causal GAN is dominated on five of six axes.
This supports the Discussion's recommendation as written.

---

## 5. Changes made to the plotting and merge code

- `scripts/merge_sweep_dbs.py` — **added deduplication**. Shards were restarted
  during the sweep (to move off the wrong GPU, and to re-split the remaining
  work), and a restart writes into a fresh database without seeing rows an
  earlier shard already produced for the same cell. Concatenating double-counted
  29 cells, which would have silently reweighted every pooled mean and CI in
  Table 1. Now deduped on the experimental cell, preferring a completed run and
  then the most recent.
- `manuscript/make_figs_v2.py` — AUC bars in panel (b) are now **signed**. They
  were clipped at zero length, so DP-GAN's below-chance 0.427 drew identically to
  a missing cell — the precise confusion M39 is about. Below-chance bars now grow
  leftwards past the chance line, x-limit widened to 0.41, and a cell with no
  AUC (COMPAS DP-GAN) is labelled `no AUC (collapsed)` rather than left blank.
- `sweeps/rf-2026-09-01/launch.sh` — fixed a bug where `dev_arg` was computed and
  never passed, so `--device` never reached the runner and torch defaulted onto
  GPU 0. Now pins `CUDA_VISIBLE_DEVICES` so the wrong GPU is unreachable rather
  than merely unrequested.

## 6. Still open

- `figs/porting.pdf` exists in the zip but is unplaced (Steven #34).
- No LaTeX toolchain on this machine, so nothing here has been compile-checked.
