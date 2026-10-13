#!/usr/bin/env bash
# decaf_tabpfn on the discovered (FGES) graphs, one worker on GPU 1, started
# once the GPU 1 queue whose PID is given as $1 has exited.
# Stop: kill this script, then `pkill -f "sweep.py run --batch facct-fges.*--gpu 1"`.
cd "$(dirname "$0")/.."
WAIT_PID="$1"
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
step "waiting for pid $WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
step "decaf_tabpfn, discovered graphs"
$SDG scripts/sweep.py run --batch facct-fges-2026-10-13 --workers 1 --cpus 32-47 --gpu 1 --methods decaf_tabpfn
step "done"
