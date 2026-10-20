#!/usr/bin/env bash
# Angelozzi mitigators re-scored on the CF re-run tables, 2026-10-20.
# `facct_bench.py export --merge` pointed the CF arms of decaf_tabpfn, decaf,
# decaf_ctgan, decaf_diffusion (facct-cffix-hand-2026-10-16) and
# decaf_dpdiffusion (facct-cffix-dpdiff-2026-10-18) at the new tables: 340
# cells, listed in facct_data/grid_cffix_2026-10-20.txt. Their old-rule outputs
# are in facct_output/invalid-cf-oldrule-2026-10-20/. Cores 16-47 (the grid
# is otherwise finished); then `collect` rewrites angelozzi_mitigators.csv.
# Stop: `pkill -f grid_cffix_2026-10-20`.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
GRID=baselines/facct_data/grid_cffix_2026-10-20.txt JOBS=28 CPUS=16-47 baselines/launch_facct_grid.sh
$SDG baselines/facct_bench.py collect
echo "[$(date -u '+%m-%d %H:%M:%S')] done"
