#!/usr/bin/env bash
# Angelozzi mitigators re-scored on TabFairGAN at its chosen lambda, 2026-10-22.
# `scripts/tabfairgan_lambda.py` picked lambda per dataset x role split from the
# facct-tfg-lam* grid by the release's own label gap on S;
# `facct_bench.py export --merge --methods tabfairgan --choose
# report/data/facct/tabfairgan_lambda_choice.csv` re-pointed 63 tabfairgan-dp@
# cells (ACSIncome and COMPAS keep the paper's lambda, so are unchanged), listed
# in facct_data/grid_tfglam_2026-10-22.txt. Their paper-lambda outputs are in
# facct_output/superseded-tfg-paperlambda-2026-10-22/ (not collected).
# Cores 16-23, 6 jobs; then `collect` rewrites angelozzi_mitigators.csv.
# Stop: kill this script, then the launch_facct_grid.sh it started, by PID.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
GRID=baselines/facct_data/grid_tfglam_2026-10-22.txt JOBS=6 CPUS=16-23 baselines/launch_facct_grid.sh
$SDG baselines/facct_bench.py collect
echo "[$(date -u '+%m-%d %H:%M:%S')] done"
