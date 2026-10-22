#!/usr/bin/env bash
# More arms on saved releases, 2026-10-22 evening (scripts/release_arms.py;
# sweep notes §21.2). CPU only; cores 0-3 left free. Lanes, each started
# separately:
#   1  cores 8-23:   placebo_split (main, FGES)
#   2  cores 24-47:  resample (eps-a, eps-b, eps-dpdiff), then
#                    resample_recipients and unaware_recipients (main, FGES)
# Usage: arms-2026-10-22b.sh <lane>   (log: sweeps/arms-2026-10-22b-<lane>.log)
# Stop: kill this script, then the release_arms.py PIDs under it (by PID).
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
MAIN=facct-2026-10-09
FGES=facct-fges-2026-10-13
EPS=facct-eps-a-2026-10-13,facct-eps-b-2026-10-13,facct-eps-dpdiff-2026-10-18
arm() {  # arm cpus workers batches
  OMP_NUM_THREADS=1 taskset -c "$2" nice -n 19 $SDG scripts/release_arms.py run --arm "$1" \
    --workers "$3" --batches "$4"
  $SDG scripts/release_arms.py collect --arm "$1"
}
case "$1" in
  1) arm placebo_split 8-23 15 "$MAIN,$FGES" ;;
  2) arm resample 24-47 22 "$EPS"
     arm resample_recipients 24-47 22 "$MAIN,$FGES"
     arm unaware_recipients 24-47 22 "$MAIN,$FGES" ;;
  *) echo "usage: $0 1|2" >&2; exit 2 ;;
esac
step done
