#!/usr/bin/env bash
# Generate our four causal-fairness arms across both datasets, plus a
# half-budget control (`ours-mst-none-halfeps`) that isolates the one protocol
# difference between the arms: their smartnoise MST spends eps/2 on the
# preprocessor, ours treats the domain as public and spends it all on the model.
set -euo pipefail
cd /home/golobs/CausalFairnessInSDG/baselines
PY=/home/golobs/miniconda3/envs/sdg/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
SEEDS=5,42,253,4112,32645
EPS=1,10

nice -n 19 ionice -c3 "$PY" gen_ours.py --datasets Adult,Compas \
  --seeds "$SEEDS" --eps "$EPS" --mechs none,ftu,dp,cf 2>&1 | grep -vE "FutureWarning|df = df.groupby|UserWarning|warnings.warn"

nice -n 19 ionice -c3 "$PY" gen_ours.py --datasets Adult,Compas \
  --seeds "$SEEDS" --eps "$EPS" --mechs none --halve-eps 2>&1 | grep -vE "FutureWarning|df = df.groupby|UserWarning|warnings.warn"

echo "GEN COMPLETE"
