#!/usr/bin/env bash
# Arms scored on saved releases, 2026-10-22 (scripts/release_arms.py; rows in
# sweeps/arms-2026-10-22/<arm>/, collected to report/data/facct/arm_<arm>.csv).
# CPU only. Lanes, each started separately (and each waiting on a PID, if
# given, until the job using its cores is gone):
#   1  cores 4-7:            unaware (every batch), then recipients (every batch)
#   2  cores 8-15 (wait):    placebo (main, FGES), then resample (FGES)
#   3  cores 24-47 (wait):   resample (main)
# Usage: arms-2026-10-22.sh <lane> [pid-to-wait-for]   (log: sweeps/arms-2026-10-22-<lane>.log)
# Stop: kill this script, then the release_arms.py PIDs under it.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
MAIN=facct-2026-10-09
FGES=facct-fges-2026-10-13
EPS=facct-eps-a-2026-10-13,facct-eps-b-2026-10-13,facct-eps-dpdiff-2026-10-18
CFFIX=facct-cffix-hand-2026-10-16,facct-cffix-fges-2026-10-16,facct-cffix-cut025-2026-10-16,facct-cffix-cut050-2026-10-16,facct-cffix-cut075-2026-10-17,facct-cffix-dpdiff-2026-10-18,facct-cffix-dpgan-2026-10-20
CUTS=facct-cut025-2026-10-12,facct-cut050-2026-10-12,facct-cut075-2026-10-12
arm() {  # arm cpus workers batches
  OMP_NUM_THREADS=1 taskset -c "$2" nice -n 19 $SDG scripts/release_arms.py run --arm "$1" \
    --workers "$3" --batches "$4"
  $SDG scripts/release_arms.py collect --arm "$1"
}
if [ -n "$2" ]; then
  step "waiting for pid $2"
  while kill -0 "$2" 2>/dev/null; do sleep 60; done
fi
case "$1" in
  1) arm unaware 4-7 4 "$MAIN,$FGES,$EPS"
     arm recipients 4-7 4 "$MAIN,$FGES,$CFFIX,$EPS,$CUTS" ;;
  2) arm placebo 8-15 8 "$MAIN,$FGES"
     arm resample 8-15 8 "$FGES" ;;
  3) arm resample 24-47 22 "$MAIN" ;;
  *) echo "usage: $0 1|2|3 [pid]" >&2; exit 2 ;;
esac
step done
