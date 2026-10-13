#!/usr/bin/env bash
# Privacy-axis batches for the DP marginal generators, on cores 32-47, started
# once the CPU scheduler whose PID is given as $1 (facct-fges torch shards) exits.
#   a: epsilon 0.1, 100        b: epsilon 0.3, 3, 30, 300
# Stop: kill this script, then `pkill -f "sweep.py run --batch facct-eps-"`.
cd "$(dirname "$0")/.."
WAIT_PID="$1"
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
step "waiting for pid $WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
for batch in facct-eps-a-2026-10-13 facct-eps-b-2026-10-13; do
  step "$batch"
  $SDG scripts/sweep.py run --batch "$batch" --workers 14 --cpus 32-47
done
step "done"
