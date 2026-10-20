#!/usr/bin/env bash
# CF re-run for the 0.75 partial cut (missed from sweeps/cffix-2026-10-16.sh),
# started 2026-10-17 on GPU 0 and cores 4-15, as that queue was.
# Stop: kill this script, then `pkill -f "sweep.py run --batch facct-cffix-cut075"`.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
echo "[$(date -u '+%m-%d %H:%M:%S')] facct-cffix-cut075"
$SDG scripts/sweep.py run --batch facct-cffix-cut075-2026-10-17 --workers 6 --cpus 4-15 --gpu 0
echo "[$(date -u '+%m-%d %H:%M:%S')] done"
