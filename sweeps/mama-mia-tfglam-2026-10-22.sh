#!/usr/bin/env bash
# MAMA-MIA (pooled and by group) on the TabFairGAN lambda grid, 2026-10-22,
# after the Angelozzi re-score frees cores 16-23. TabFairGAN has no structure
# to read, so only the marginal flavour runs (see scripts/mama_mia.py).
# Rows: report/data/facct/mama_mia_tfglam.csv, mama_mia_group_tfglam.csv.
# Usage: mama-mia-tfglam-2026-10-22.sh [pid-to-wait-for]
# Stop: kill this script, then the mama_mia.py PIDs it started.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
BATCHES=facct-tfg-lam002-2026-10-22,facct-tfg-lam005-2026-10-22,facct-tfg-lam01-2026-10-22,facct-tfg-lam02-2026-10-22,facct-tfg-lam10-2026-10-22,facct-tfg-lam20-2026-10-22
if [ -n "$1" ]; then
  step "waiting for pid $1"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
step start
OMP_NUM_THREADS=1 taskset -c 16-23 nice -n 19 $SDG scripts/mama_mia.py --batches "$BATCHES" \
  --workers 8 --collect-to report/data/facct/mama_mia_tfglam.csv
OMP_NUM_THREADS=1 taskset -c 16-23 nice -n 19 $SDG scripts/mama_mia.py --by-group --batches "$BATCHES" \
  --workers 8 --collect-to report/data/facct/mama_mia_group_tfglam.csv
step done
