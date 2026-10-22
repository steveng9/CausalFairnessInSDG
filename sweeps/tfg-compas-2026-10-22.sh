#!/usr/bin/env bash
# COMPAS TabFairGAN re-run at 970 + 30 epochs, 2026-10-22 (sweep notes §21.2):
# eight batches (none and dp at lambda 2.2; dp at 0.02 ... 2.0), GPU 0, cores
# 4-7, four schedulers of one worker each. Then, on cores 4-7:
#   collect_results.py, tabfairgan_lambda.py (COMPAS now from these batches),
#   MAMA-MIA pooled and by group (mama_mia[_group]_tfgcompas.csv), and every
#   release_arms.py arm on them; the arms are collected once the
#   arms-2026-10-22b.sh lanes (PIDs given) are gone, so each arm_*.csv holds
#   both. The Angelozzi re-point is done by hand afterwards.
# Usage: tfg-compas-2026-10-22.sh [pid-to-wait-for-before-collecting-arms ...]
# Log: sweeps/tfg-compas-2026-10-22.log
# Stop: kill this script, then the sweep.py / mama_mia.py / release_arms.py
# PIDs under it (by PID, not pkill -f).
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
B=facct-tfg-compas
D=2026-10-22
chain() {
  for batch in "$@"; do
    step "start $batch"
    $SDG scripts/sweep.py run --batch "$batch" --workers 1 --cpus 4-7 --gpu 0
    step "end $batch ($?)"
  done
}
chain $B-$D $B-lam002-$D &
p1=$!
chain $B-lam005-$D $B-lam01-$D &
p2=$!
chain $B-lam02-$D $B-lam05-$D &
p3=$!
chain $B-lam10-$D $B-lam20-$D &
p4=$!
wait $p1 $p2 $p3 $p4
step "sweeps done"
for batch in $B-$D $B-lam{002,005,01,02,05,10,20}-$D; do
  $SDG scripts/sweep.py status --batch "$batch" | tail -n 2
done
$SDG scripts/collect_results.py | tail -n 3
$SDG scripts/tabfairgan_lambda.py
ALL=$(ls -d sweeps/$B-*$D | xargs -n1 basename | paste -sd,)
step "MAMA-MIA on $ALL"
OMP_NUM_THREADS=1 taskset -c 4-7 nice -n 19 $SDG scripts/mama_mia.py --batches "$ALL" \
  --workers 4 --collect-to report/data/facct/mama_mia_tfgcompas.csv
OMP_NUM_THREADS=1 taskset -c 4-7 nice -n 19 $SDG scripts/mama_mia.py --by-group --batches "$ALL" \
  --workers 4 --collect-to report/data/facct/mama_mia_group_tfgcompas.csv
ARMS="unaware recipients placebo placebo_split resample unaware_recipients resample_recipients"
for arm in $ARMS; do
  step "arm $arm"
  OMP_NUM_THREADS=1 taskset -c 4-7 nice -n 19 $SDG scripts/release_arms.py run --arm "$arm" \
    --workers 4 --batches "$ALL"
done
for pid in "$@"; do
  step "waiting for pid $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
for arm in $ARMS; do $SDG scripts/release_arms.py collect --arm "$arm"; done
step done
