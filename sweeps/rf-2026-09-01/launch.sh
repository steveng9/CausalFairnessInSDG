#!/usr/bin/env bash
# Second sweep: epsilon=10 (private methods) / non-private once, all nine
# generators, random-forest downstream model, and -- unlike the first sweep --
# every synthetic table written to disk so a metric added later costs a read
# rather than a regeneration.
#
# Sharded by (dataset x generator family) into its own SQLite file: seven
# writers on one database is lock contention for no gain, and `merge_dbs.py`
# folds them back afterwards.
#
# CPU BUDGET. This box is shared and another user's job has priority, so the
# whole sweep is confined to CPUS (8 cores) with taskset. That is a hard
# ceiling the scheduler enforces, unlike `nice`, which only yields when the
# machine is already contended -- and unlike a thread count, which multiplies
# by however many shards happen to be running. Keep every launch inside it.
set -u
ROOT=/home/golobs/CausalFairnessInSDG
PY=/home/golobs/miniconda3/envs/sdg/bin/python
BATCH=rf-2026-09-01
GPU=1        # GPU 0 belongs to the other user; never touch it
OUT=$ROOT/sweeps/$BATCH
CPUS=32-47   # 16 cores of 48; cores 0-31 stay free for the other user on this box
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

run_shard () {  # name  datasets  methods  [device]
  local name=$1 ds=$2 methods=$3 device=${4:-}
  local dev_arg=""
  # CUDA_VISIBLE_DEVICES=$GPU makes physical GPU $GPU the only one this process
  # can see, so it is addressed as cuda:0 and GPU 0 is unreachable even if a
  # code path ignores --device. Belt and braces: the flag alone was not enough,
  # because it used to be computed here and never passed.
  [ -n "$device" ] && dev_arg="--device cuda:0"
  nohup nice -n 10 taskset -c $CPUS env CUDA_VISIBLE_DEVICES=$GPU \
      $PY $ROOT/scripts/run_experiments.py $dev_arg \
      --batch $BATCH --datasets "$ds" --methods "$methods" \
      --epsilons 10 --seeds 0,1,2,3,4 --classifier rf \
      --db-path $OUT/db/$name.db --synth-dir $OUT/synth/$name \
      --results-dir $OUT/results-$name \
      > $ROOT/logs/$BATCH-$name.log 2>&1 &
  echo "$name pid=$!"
}

run_shard compas-marg compas mst,privbayes,privsyn
run_shard adult-marg  adult  mst,privbayes,privsyn
run_shard compas-gan  compas decaf,decaf_ctgan,decaf_dpgan,decaf_dpctgan cuda:1
run_shard adult-gan-np  adult decaf,decaf_ctgan            cuda:1
run_shard adult-gan-dp  adult decaf_dpgan,decaf_dpctgan    cuda:1
run_shard compas-diff compas decaf_diffusion,decaf_dpdiffusion cuda:1
run_shard adult-diff  adult  decaf_diffusion,decaf_dpdiffusion cuda:1
