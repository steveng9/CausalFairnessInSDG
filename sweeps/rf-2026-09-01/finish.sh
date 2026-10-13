#!/usr/bin/env bash
# Final push. The two remaining method queues (DP-diffusion, DP-CTGAN) were
# running as two sequential shards, which used 200% of a 1600% budget -- the
# work was serialised, not CPU-bound. Split per seed so the budget is usable.
# Each shard gets its own SQLite file (one writer per db, no lock contention)
# and its own results dir; merge_sweep_dbs.py folds them back.
set -u
ROOT=/home/golobs/CausalFairnessInSDG
PY=/home/golobs/miniconda3/envs/sdg/bin/python
BATCH=rf-2026-09-01
OUT=$ROOT/sweeps/$BATCH
CPUS=32-47                    # 16 of 48 cores; 0-31 stay free for the other user
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

go () {  # name  methods  seed
  nohup nice -n 10 taskset -c $CPUS env CUDA_VISIBLE_DEVICES=1 \
      $PY $ROOT/scripts/run_experiments.py --device cuda:0 \
      --batch $BATCH --datasets adult --methods "$2" \
      --epsilons 10 --seeds $3 --classifier rf \
      --db-path $OUT/db/$1.db --synth-dir $OUT/synth/$1 \
      --results-dir $OUT/results-$1 \
      > $ROOT/logs/$BATCH-$1.log 2>&1 &
  echo "  $1 pid=$!"
}
for s in 0 1 2 3 4; do go "adult-dpdiff-s$s"  decaf_dpdiffusion $s; done
for s in 0 1 2 3 4; do go "adult-dpctgan-s$s" decaf_dpctgan     $s; done
