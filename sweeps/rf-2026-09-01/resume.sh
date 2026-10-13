#!/usr/bin/env bash
# Resume the two critical-path Adult shards on the corrected settings:
#   * GPU 1 only  -- CUDA_VISIBLE_DEVICES makes GPU 0 unreachable, not merely
#     unrequested, so the other user's GPU stays his even if a torch code path
#     ignores --device (which is exactly how we ended up on GPU 0).
#   * 16 cores    -- cores 32-47, leaving 0-31 (3200%) free on a 48-core box.
# Both runs resume by key from their existing databases; completed runs are
# skipped, so the restart costs only the run that was in flight.
set -u
ROOT=/home/golobs/CausalFairnessInSDG
PY=/home/golobs/miniconda3/envs/sdg/bin/python
BATCH=rf-2026-09-01
OUT=$ROOT/sweeps/$BATCH
CPUS=32-47
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

go () {  # name  dataset  methods
  nohup nice -n 10 taskset -c $CPUS env CUDA_VISIBLE_DEVICES=1 \
      $PY $ROOT/scripts/run_experiments.py --device cuda:0 \
      --batch $BATCH --datasets "$2" --methods "$3" \
      --epsilons 10 --seeds 0,1,2,3,4 --classifier rf \
      --db-path $OUT/db/$1.db --synth-dir $OUT/synth/$1 \
      --results-dir $OUT/results-$1 \
      >> $ROOT/logs/$BATCH-$1.log 2>&1 &
  echo "$(date -u +%H:%M:%S)Z resumed $1 pid=$!"
}
go adult-gan-dp adult decaf_dpgan,decaf_dpctgan
go adult-diff   adult decaf_diffusion,decaf_dpdiffusion
