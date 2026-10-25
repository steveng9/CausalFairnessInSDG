#!/usr/bin/env bash
# Angelozzi mitigators re-scored on COMPAS TabFairGAN at 970 + 30 epochs,
# 2026-10-22 (sweep notes §21.2). `facct_bench.py export --merge --methods
# tabfairgan --datasets compas --batches <facct-tfg-compas-*> --choose
# report/data/facct/tabfairgan_lambda_choice.csv` re-pointed the 9 COMPAS
# tabfairgan-none / tabfairgan-dp@ cells (lambda 2.2, as before) to
# facct-tfg-compas-2026-10-22, listed in facct_data/grid_tfgcompas_2026-10-22.txt.
# Their 70-epoch outputs are in facct_output/superseded-tfgcompas-70epochs-2026-10-22/
# (not collected); manifests before: facct_data/backup-pre-tfgcompas-2026-10-22/.
# Cores 16-23, 6 jobs; then `collect` rewrites angelozzi_mitigators.csv.
# Stop: kill this script, then the launch_facct_grid.sh it started, by PID.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
GRID=baselines/facct_data/grid_tfgcompas_2026-10-22.txt JOBS=6 CPUS=16-23 baselines/launch_facct_grid.sh
$SDG baselines/facct_bench.py collect
echo "[$(date -u '+%m-%d %H:%M:%S')] done"
