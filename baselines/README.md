# Baseline comparison: Angelozzi & Arcolezi (PETS 2026)

Their benchmark asks *where* in a differentially private synthetic-data pipeline a
fairness intervention should go, and answers with three downstream stages
(pre-processing, in-processing, post-processing) over eight aif360 mitigators.
Our mechanism is a fourth stage it does not have: inside the generator.

Everything here exists to make that a like-for-like comparison rather than two
papers' protocols placed side by side. **Their harness is unmodified.** Both arms
use their preprocessed data, their train/test splits, their classifier, their
MinMax scaling and their aif360 metric objects. The only difference between arms
is who produced the synthetic training table.

## Layout

| path | what |
|---|---|
| `dp-fair-intervention-benchmark/` | upstream clone (2.3 GB; ships pre-generated synthetic data, so the expensive half is skipped) |
| `install_env.sh` | builds the `dpfair` conda env (aif360 + fairlearn + inFairness + smartnoise, CPU torch) |
| `gen_ours.py` | fits our MST under each mechanism on *their* train split, writes into *their* directory layout |
| `run_bench.py` | one (dataset, arm, classifier) cell of their `benchmark()`; resumable via a `.done` marker |
| `launch_grid.sh` | fans the 28-cell grid across cores, nice 19 / ionice idle / 2 threads per process |
| `aggregate.py` | collects every cell into `output/tidy.csv` and prints the comparison |
| `logs/` | `grid.log` (one line per finished cell), `cells/*.log` (per-cell output) |

## Arms

- `mst`, `aim` — theirs, with all eight mitigators
- `ours-mst-{none,ftu,dp,cf}` — our MST with the causal cut inside the generator
- `ours-mst-none-halfeps` — control at eps/2

The half-budget control exists because of the one protocol difference between the
arms: their smartnoise MST is called with `preprocessor_eps = eps/2`, so half its
nominal budget goes on discovering the domain, while we treat the domain as public
and give the whole budget to the model. The control shows the difference is small
(Adult eps=1: 0.079 vs 0.073 parity gap), so it does not drive the comparison.

`ours-mst-none` is the other control: our MST with no mechanism, isolating our
private-pgm implementation from smartnoise's.

## Running

```bash
./install_env.sh                     # once, ~10 min
/home/golobs/miniconda3/envs/sdg/bin/python gen_ours.py \
    --datasets Adult,Compas --seeds 5,42,253,4112,32645 --eps 1,10 \
    --mechs none,ftu,dp,cf           # ~11 s per cell
JOBS=10 ./launch_grid.sh             # resumable; skips cells with a .done marker
/home/golobs/miniconda3/envs/dpfair/bin/python aggregate.py --classifier XGB
```

Seeds are the first five of their published twenty. XGB is their own pairing for
MST and runs ~10x faster than their saga/elasticnet logistic regression; the LR
half of the grid is the long pole.

## What we cannot run

FLIP (Hyrup et al. 2025) and PF-WGAN (Sarmin et al. 2025) publish no code, checked
against their arXiv pages. Reproducing either means reimplementing a method and
tuning it against our own, which is not a comparison worth reporting.
