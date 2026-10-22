#!/usr/bin/env bash
# FairPFN extension, 2026-10-22, GPU 0 only (GPU 1 stays free for Daniil),
# cores 16-23. New contexts since the 2026-10-11 baseline: every other
# generator's unmitigated release (decaf, decaf_ctgan, decaf_diffusion,
# decaf_dpgan / dpctgan / dpdiffusion at epsilon 1, 10, 1000, tabfairgan), and
# edited releases as contexts (--stack: decaf_tabpfn, decaf_diffusion and MST
# at epsilon 10, mechanisms dp and ftu, each role split), where the publisher
# and the recipient both act. 10,000-row contexts only. Inputs were
# re-exported first (export --synth-methods <all> --stack ...).
# Three processes, split by dataset, each appending to its own results file.
# Stop: kill this script, then the three fairpfn_baseline.py PIDs it prints.
cd "$(dirname "$0")/.."
PY=/home/golobs/miniconda3/envs/fairpfn/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
lane() {
  OMP_NUM_THREADS=2 taskset -c "$2" nice -n 19 $PY scripts/fairpfn_baseline.py run \
    --datasets "$1" --ctx-sizes 10000 --device cuda:0 --n-jobs 2 --results "results-$3.csv" \
    > "sweeps/fairpfn-ext-2026-10-22-$3.log" 2>&1
}
lane acsincome_ca,acsincome_tx,acspubcov_ca,acspubcov_tx 16-18 a & pa=$!
lane hmda_ca,hmda_tx,sbo,snake 19-21 b & pb=$!
lane cdc_diabetes,adult,compas 22-23 c & pc=$!
step "lanes a=$pa b=$pb c=$pc"
wait $pa; step "a exited ($?)"; wait $pb; step "b exited ($?)"; wait $pc; step "c exited ($?)"
step done
