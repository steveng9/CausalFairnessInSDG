#!/usr/bin/env bash
# Graph-variant experiment, 2026-10-26 (sweep notes §23). Same data and
# generators, different graphs: does a fairness cut mean the same thing when
# the graph changes? GPU 0 only (GPU 1 stays free for Daniil); cores 0-11
# left free. Lanes, each started separately:
#   T  cores 12-14, 3 workers:  decaf_tabpfn over the eight facct-graph-* batches
#   G  cores 15-22, 8 workers:  decaf, decaf_ctgan, decaf_diffusion over the same
#   D  cores 23-26, 4 workers:  decaf_dpdiffusion (eps 1, 10) over facct-graph-dp-*
# After each batch, lanes T and G score the label-scramble placebo on that
# batch's unedited releases (scripts/release_arms.py, cores 35-37 / 27-34),
# one batch at a time per lane, while the lane moves on to the next batch.
# Usage: graph-2026-10-26.sh T|G|D   (log: sweeps/graph-2026-10-26-<lane>.log)
# Stop: kill this script, then its sweep.py / release_arms.py children by PID.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
BATCHES="hand nodirect noproxy100 noproxy50 contested over fges1000 fges10000"
placebo_pid=""
placebo() {  # batch methods cpus workers
  [ -n "$placebo_pid" ] && wait "$placebo_pid"
  OMP_NUM_THREADS=1 taskset -c "$3" nice -n 19 $SDG scripts/release_arms.py run --arm placebo \
    --batches "$1" --methods "$2" --workers "$4" &
  placebo_pid=$!
}
lane() {  # methods cpus workers placebo_cpus placebo_workers
  for b in $BATCHES; do
    batch=facct-graph-$b-2026-10-26
    step "$batch"
    $SDG scripts/sweep.py run --batch "$batch" --methods "$1" --workers "$3" --cpus "$2" --gpu 0
    step "$batch placebo"
    placebo "$batch" "$1" "$4" "$5"
  done
  [ -n "$placebo_pid" ] && wait "$placebo_pid"
}
case "$1" in
  T) lane decaf_tabpfn 12-14 3 35-37 3 ;;
  G) lane decaf,decaf_ctgan,decaf_diffusion 15-22 8 27-34 8 ;;
  D) for b in hand nodirect noproxy100 over; do
       batch=facct-graph-dp-$b-2026-10-26
       step "$batch"
       $SDG scripts/sweep.py run --batch "$batch" --workers 4 --cpus 23-26 --gpu 0
     done ;;
  *) echo "usage: $0 T|G|D" >&2; exit 2 ;;
esac
step done
