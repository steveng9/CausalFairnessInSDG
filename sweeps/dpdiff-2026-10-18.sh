#!/usr/bin/env bash
# DP-diffusion queue, 2026-10-18, GPU 0 only (GPU 1 stays free for Daniil),
# cores 4-15 (0-3 left free). Two lanes share GPU 0; shard locks keep them
# from colliding, and both end on the epsilon batch.
#   a: CF re-run (facct-cffix-dpdiff), then the new epsilons (facct-eps-dpdiff)
#   b: epoch checks (facct-tune-dpdiff-e30, -e300), then the new epsilons
# Usage: dpdiff-2026-10-18.sh a|b   (log: sweeps/dpdiff-2026-10-18-<lane>.log)
# Stop: kill both lane scripts, then `pkill -f "sweep.py run --batch facct-.*dpdiff"`.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $1"; }
case "$1" in
  a) CPUS=4-11; WORKERS=6; BATCHES="facct-cffix-dpdiff-2026-10-18 facct-eps-dpdiff-2026-10-18" ;;
  b) CPUS=12-15; WORKERS=3; BATCHES="facct-tune-dpdiff-e30-2026-10-18 facct-tune-dpdiff-e300-2026-10-18 facct-eps-dpdiff-2026-10-18" ;;
  *) echo "usage: $0 a|b" >&2; exit 2 ;;
esac
for batch in $BATCHES; do
  step "$batch"
  $SDG scripts/sweep.py run --batch "$batch" --workers $WORKERS --cpus $CPUS --gpu 0
done
step done
