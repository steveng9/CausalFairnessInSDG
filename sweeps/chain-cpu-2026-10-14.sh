#!/usr/bin/env bash
# CPU queue started 2026-10-13/14, cores 32-47, at most 14 processes, in this
# order, once the earlier CPU chain whose PID is given as $1 has exited
# (facct-eps-a; facct-eps-b is on hold in its sweep.json and returns at once):
#   0. facct-eps-a PrivBayes: the SNAKE and SBO shards, and every epsilon-0.1
#      run again (added 2026-10-15). The vendored `calculate_k` picked k up to
#      12 at epsilon 0.1, so those runs were wrong or never finished; the
#      fixed k is 1-2 and they take minutes. The old runs are archived in
#      sweeps/facct-eps-a-2026-10-13/invalid-privbayes-k-2026-10-15/.
#   1. Angelozzi & Arcolezi mitigators on the no-mechanism releases (the
#      downstream baseline) of every generator in facct-2026-10-09
#   2. MAMA-MIA on every saved table so far
#   3. torch causal families on the hand-drawn graphs, and TabFairGAN
#      (added 2026-10-14; no graph, about 30 core-hours) (facct-2026-10-09)
#   4. Angelozzi harness on MST's mechanism arms
#   5. DP causal families (DP-SGD GAN / CTGAN / diffusion), hand-drawn graphs
#   6. anything of facct-eps-a still missing (normally nothing after step 0)
#   7. Angelozzi harness: export again to pick up the new families, then their
#      no-mechanism arms
#   8. MAMA-MIA on what was added
#   9. facct-eps-b (epsilon 0.3, 3, 30, 300), then MAMA-MIA on it
#  10. Angelozzi harness on every remaining mechanism arm (the long tail:
#      about 20 minutes a cell), then one tidy table
# Stop: kill this script, then `pkill -f "sweep.py run --batch facct-"`,
# `pkill -f launch_facct_grid.sh`, `pkill -f mama_mia.py`.
cd "$(dirname "$0")/.."
WAIT_PID="$1"
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
PIN="taskset -c 32-47 nice -n 19"
MAIN=facct-2026-10-09
BATCHES=facct-2026-10-09,facct-fges-2026-10-13,facct-eps-a-2026-10-13,facct-cut050-2026-10-12,facct-cut025-2026-10-12,facct-cut075-2026-10-12
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
sweep() { $SDG scripts/sweep.py run --batch "$1" --workers "$2" --cpus 32-47 "${@:3}"; }
grid() { JOBS=14 CPUS=32-47 ONLY="$1" baselines/launch_facct_grid.sh; }
collect() { $PIN $SDG baselines/facct_bench.py collect >> baselines/logs/facct_export.log 2>&1; }
mia() { OMP_NUM_THREADS=1 $PIN $SDG scripts/mama_mia.py --batches "$1" --workers 14 --collect-to "report/data/facct/mama_mia${2:-}.csv" >> sweeps/mama_mia.log 2>&1; }

step "waiting for pid $WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
step "0 facct-eps-a PrivBayes, epsilon 0.1 again"; sweep facct-eps-a-2026-10-13 14 --methods privbayes
step "1 Angelozzi, no-mechanism arms";            grid ' [a-z_]+-none '; collect
step "2 MAMA-MIA";                                mia "$BATCHES"
step "3 torch families (hand graphs), TabFairGAN"; sweep $MAIN 14 --methods decaf,decaf_ctgan,decaf_diffusion,tabfairgan
step "4 Angelozzi, MST mechanism arms";           grid ' mst-'; collect
step "5 DP causal families";                      sweep $MAIN 14 --methods decaf_dpgan,decaf_dpctgan,decaf_dpdiffusion
step "6 facct-eps-a leftovers";                   sweep facct-eps-a-2026-10-13 4
step "7 Angelozzi, export again, new no-mechanism arms"
$PIN $SDG baselines/facct_bench.py export --batches $MAIN >> baselines/logs/facct_export.log 2>&1
grid ' [a-z_]+-none '; collect
step "8 MAMA-MIA";                                mia "$BATCHES"
step "9 facct-eps-b";                             sweep facct-eps-b-2026-10-13 14 --even-if-held
mia facct-eps-b-2026-10-13 _eps_b
step "10 Angelozzi, remaining mechanism arms";    grid '.'; collect
step "done"
