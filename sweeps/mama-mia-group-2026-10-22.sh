#!/usr/bin/env bash
# MAMA-MIA by group, 2026-10-22: the same attacks as mama_mia.csv, scored
# within each protected attribute's two groups (holdout modal value vs the
# rest), on every batch with saved tables. CPU only, cores 8-15.
# Rows: report/data/facct/mama_mia_group.csv.
# Stop: kill this script, then the mama_mia.py PIDs it started.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
BATCHES=facct-2026-10-09,facct-fges-2026-10-13,facct-eps-a-2026-10-13,facct-eps-b-2026-10-13,facct-cut025-2026-10-12,facct-cut050-2026-10-12,facct-cut075-2026-10-12,facct-cffix-hand-2026-10-16,facct-cffix-fges-2026-10-16,facct-cffix-cut025-2026-10-16,facct-cffix-cut050-2026-10-16,facct-cffix-cut075-2026-10-17,facct-cffix-dpdiff-2026-10-18,facct-cffix-dpgan-2026-10-20,facct-eps-dpdiff-2026-10-18
echo "[$(date -u '+%m-%d %H:%M:%S')] start"
OMP_NUM_THREADS=1 taskset -c 8-15 nice -n 19 $SDG scripts/mama_mia.py --by-group --batches "$BATCHES" \
  --workers 8 --collect-to report/data/facct/mama_mia_group.csv
echo "[$(date -u '+%m-%d %H:%M:%S')] done"
