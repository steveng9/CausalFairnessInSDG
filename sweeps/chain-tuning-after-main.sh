#!/usr/bin/env bash
# Start the two torch epoch-tuning pilots once the main FAccT sweep's
# scheduler has exited, on the same 16 pinned cores -- so the pilots never add
# to the sweep's CPU footprint, they inherit it. Launched 2026-10-09.
# To run the pilots on a GPU instead, kill this script and start
#   python scripts/sweep.py run --batch facct-tune-e30-2026-10-09 --workers 4 --cpus 32-47 --gpu <N>
set -u
cd /home/golobs/CausalFairnessInSDG
PY=/home/golobs/miniconda3/envs/sdg/bin/python
while pgrep -f "sweep.py run --batch facct-2026-10-09" >/dev/null; do sleep 120; done
for e in 30 100; do
  B=facct-tune-e$e-2026-10-09
  $PY scripts/sweep.py run --batch $B --workers 14 --cpus 32-47 \
      > sweeps/$B/scheduler-$(date +%m%d-%H%M).log 2>&1
done
