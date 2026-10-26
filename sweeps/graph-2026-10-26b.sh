#!/usr/bin/env bash
# Graph-variant experiment, CPU lane, 2026-10-26 (sweep notes §23). Replaces lane G
# of graph-2026-10-26.sh, which ran decaf / decaf_ctgan / decaf_diffusion on GPU 0
# and saturated it (stopped after 15 min; 2 diffusion shards of the hand batch had
# finished on the GPU and are kept). Here those three run on CPU, 18 workers on
# cores 27-34 and 38-47 (35-37 are lane T's placebo cores), then score the placebo
# on cores 16-22, one batch at a time, while the lane moves on.
# Usage: graph-2026-10-26b.sh   (log: sweeps/graph-2026-10-26-C.log)
# Stop: kill this script, then its sweep.py / release_arms.py children by PID.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
METHODS=decaf,decaf_ctgan,decaf_diffusion
placebo_pid=""
for b in hand nodirect noproxy100 noproxy50 contested over fges1000 fges10000; do
  batch=facct-graph-$b-2026-10-26
  step "$batch"
  $SDG scripts/sweep.py run --batch "$batch" --methods $METHODS --workers 18 --cpus 27-34,38-47
  step "$batch placebo"
  [ -n "$placebo_pid" ] && wait "$placebo_pid"
  OMP_NUM_THREADS=1 taskset -c 16-22 nice -n 19 $SDG scripts/release_arms.py run --arm placebo \
    --batches "$batch" --methods $METHODS --workers 7 &
  placebo_pid=$!
done
[ -n "$placebo_pid" ] && wait "$placebo_pid"
step done
