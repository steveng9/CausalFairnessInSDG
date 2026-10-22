#!/usr/bin/env bash
# The 23 Angelozzi cells with no .done marker on 2026-10-21 (all DP-GAN /
# DP-CTGAN arms; each died inside the LFR mitigator without a traceback while
# 28 jobs ran). Retried with 6 jobs, then `collect`.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
GRID=baselines/facct_data/grid_missing_2026-10-21.txt JOBS=6 CPUS=16-47 baselines/launch_facct_grid.sh
$SDG baselines/facct_bench.py collect
echo "[$(date -u '+%m-%d %H:%M:%S')] done"
