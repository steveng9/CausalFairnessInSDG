#!/usr/bin/env bash
# TabFairGAN lambda grid, 2026-10-22: six batches (lambda 0.02, 0.05, 0.1,
# 0.2, 1.0, 2.0), mechanism dp only, CPU only on cores 24-47 (0-3 left free;
# 4-23 for the scoring jobs). One scheduler per batch, 4 workers each, all six
# at once; shard locks keep them apart.
# Stop: kill this script, then the six scheduler PIDs it logs (by PID, not pkill -f).
# Log: sweeps/tfg-lambda-2026-10-22.log (schedulers: sweeps/<batch>/scheduler-*.log)
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
pids=()
for tag in 002 005 01 02 10 20; do
  batch=facct-tfg-lam${tag}-2026-10-22
  step "start $batch"
  $SDG scripts/sweep.py run --batch "$batch" --workers 4 --cpus 24-47 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; step "scheduler $pid exited ($?)"; done
step done
