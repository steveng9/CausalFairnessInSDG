# Post-submission TODOs

## Utility / baseline classifier
- **Baseline accuracy is weak in two cells.** MST on Adult sits at exactly zero
  lift over majority-class (0.747 vs 0.758). DP-GAN family collapses to a
  constant predictor in several ε=1 cells (168/360 runs across DP-GAN and DP-
  CTGAN combined). Currently disclosed in §4.1 and matched-pair analysis is
  robust to it, but a stronger baseline (or a different downstream task on
  Adult) would tighten the story.
- **RF-as-primary-classifier fairness sweep, deferred.** RF *accuracy* is
  already recorded per run (`downstream_accuracy_rf`), and switching the
  reference classifier does not rescue MST on Adult (RF 0.727 / MLP 0.747, both
  below majority). What it *would* change is the fairness side: `dp_gap__*` is
  only computed against the primary classifier, so an RF-based fairness pass
  requires re-running with `--classifier=rf`. Full sweep ≈ 6–12 h wall-clock
  even parallelised; a marginals-only variant (~810 runs, 1–2 h) would be a
  cheap sanity check. Not run for the submission.

## AIM
- AIM is currently a smoke-test at ε=1 with one seed per (dataset, mechanism)
  — 8 completed runs total. Mentioned in a footnote in §3; not pooled into
  Table 1. Extending AIM to the full sweep (3 ε × 5 seeds × 3 role splits × 2
  datasets × 4 mechanisms = 360 runs) would give it Table 1 representation.

## Figures / tables
- Alternative fair-vs-fidelity scatter (Δ 2-way TVD instead of Δ accuracy) is
  built at `figs/fig_fair_vs_fidelity.pdf`; not used in the paper. Consider
  including in an appendix or swapping if reviewers complain about accuracy.
- Edges-omitted table (Table 2 in the paper) counts baseline-graph edges the
  mechanism would forbid. The GAN/diffusion count is a deterministic cut on
  the fixed DAG; the marginals count is per baseline run averaged. Script:
  `scratchpad/compute_edges_omitted.py`.
