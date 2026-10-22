#!/usr/bin/env bash
# MAMA-MIA on the tables no attack has seen yet, 2026-10-22: the CF re-runs
# (facct-cffix-*, which supersede the causal families' CF rows of the
# original batches) and the DP-diffusion epsilons 0.1 / 3 / 100. CPU only,
# cores 4-23 (0-3 left free). Rows: report/data/facct/mama_mia_cffix.csv.
# Stop: kill this script, then `pkill -f mama_mia.py`.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
BATCHES=facct-cffix-hand-2026-10-16,facct-cffix-fges-2026-10-16,facct-cffix-cut025-2026-10-16,facct-cffix-cut050-2026-10-16,facct-cffix-cut075-2026-10-17,facct-cffix-dpdiff-2026-10-18,facct-cffix-dpgan-2026-10-20,facct-eps-dpdiff-2026-10-18
echo "[$(date -u '+%m-%d %H:%M:%S')] start"
OMP_NUM_THREADS=1 taskset -c 4-23 nice -n 19 $SDG scripts/mama_mia.py --batches "$BATCHES" \
  --workers 20 --collect-to report/data/facct/mama_mia_cffix.csv
echo "[$(date -u '+%m-%d %H:%M:%S')] done"
