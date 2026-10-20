#!/usr/bin/env bash
# CF re-run after commit 2379bb9 (DECAF Corollary 1 edge selection), started
# 2026-10-16 on GPU 0 and cores 4-15 (idle at the time). Batches run one after
# another; each holds only the CF arm of the datasets whose cut set changed.
# Stop: kill this script, then `pkill -f "sweep.py run --batch facct-cffix-"`.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
for b in hand fges cut050 cut025; do
  step "facct-cffix-$b"
  $SDG scripts/sweep.py run --batch facct-cffix-$b-2026-10-16 --workers 6 --cpus 4-15 --gpu 0
done
step done
