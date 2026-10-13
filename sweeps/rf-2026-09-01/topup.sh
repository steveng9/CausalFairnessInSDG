#!/usr/bin/env bash
# Top up the DECAF / +CTGAN seeds that were cut to seed 0 when the sweep was
# squeezed to 8 cores. Runs as a queue: never more than MAXJOBS of these at
# once, and each one is pinned to the same $CPUS as the rest of the sweep, so
# total sweep CPU cannot rise above the cap however many shards are alive.
set -u
ROOT=/home/golobs/CausalFairnessInSDG
PY=/home/golobs/miniconda3/envs/sdg/bin/python
BATCH=rf-2026-09-01
OUT=$ROOT/sweeps/$BATCH
CPUS=40-47
MAXJOBS=3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

# Wait until the whole sweep -- not just this script's children -- is down to
# room for another shard. The long diffusion/DP-GAN shards are multithreaded,
# so we gate on measured CPU rather than on a process count.
wait_for_room () {
  while :; do
    local pids t=0 p a b
    pids=$(pgrep -f "batch $BATCH" || true)
    [ -z "$pids" ] && return 0
    declare -A A=()
    for p in $pids; do A[$p]=$(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null || echo 0); done
    sleep 10
    for p in $pids; do b=$(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null || echo 0); t=$((t + b - ${A[$p]:-0})); done
    # jiffies over 10s -> percent of one core
    [ $((t / 10)) -lt 620 ] && return 0
  done
}

run_shard () {  # name  dataset  method  seed
  local name=$1 ds=$2 m=$3 seed=$4
  nohup nice -n 10 taskset -c $CPUS $PY $ROOT/scripts/run_experiments.py \
      --batch $BATCH --datasets "$ds" --methods "$m" \
      --epsilons 10 --seeds $seed --classifier rf \
      --db-path $OUT/db/$name.db --synth-dir $OUT/synth/$name \
      --results-dir $OUT/results-$name \
      > $ROOT/logs/$BATCH-$name.log 2>&1 &
  echo "$(date +%H:%M:%S) launched $name pid=$!"
}

for seed in 1 2 3 4; do
  for spec in "compas decaf" "adult decaf" "adult decaf_ctgan"; do
    set -- $spec
    ds=$1 m=$2
    short=$(echo $m | sed 's/decaf_ctgan/ctgan/')
    name="$ds-$short-s$seed"
    n=$(sqlite3 $OUT/db/$name.db "select count(*) from runs" 2>/dev/null || echo 0)
    if [ "$n" -ge 12 ]; then echo "skip $name (already $n)"; continue; fi
    wait_for_room
    run_shard "$name" "$ds" "$m" "$seed"
    sleep 20
  done
done
wait
echo "$(date +%H:%M:%S) top-up queue drained"
