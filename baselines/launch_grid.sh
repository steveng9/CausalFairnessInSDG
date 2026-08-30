#!/usr/bin/env bash
# Fan the (dataset x synthesizer-arm x classifier) grid across cores.
#
# Each cell is one `run_bench.py` process writing into its own output dir and
# dropping a `.done` marker, so the grid is resumable: re-running this script
# picks up only what is missing. Politeness on this shared box: nice 19,
# ionice idle, 2 threads per process, and a concurrency cap well under the
# 48 cores.
set -uo pipefail
ROOT=/home/golobs/CausalFairnessInSDG/baselines
PY=/home/golobs/miniconda3/envs/dpfair/bin/python
JOBS="${JOBS:-10}"
SEEDS="${SEEDS:-5,42,253,4112,32645}"
EPS="${EPS:-1,10}"
# XGB first: it is ~10x faster than the saga/elasticnet LR, so the grid has a
# complete answer early even if the LR half is still going.
CLASSIFIERS="${CLASSIFIERS:-XGB LR}"

# Their two synthesizers, our four mechanisms, and the half-budget control
# that isolates the preprocessor_eps accounting difference between the arms.
ARMS="mst aim ours-mst-none ours-mst-ftu ours-mst-dp ours-mst-cf ours-mst-none-halfeps
ours-aim-none ours-aim-ftu ours-aim-dp ours-aim-cf"

mkdir -p "$ROOT/logs/cells" "$ROOT/output"

for CLF in $CLASSIFIERS; do
  for DS in Adult Compas; do
    for ARM in $ARMS; do
      [ -d "$ROOT/dp-fair-intervention-benchmark/data/$DS/$ARM" ] || continue
      echo "$DS $ARM $CLF"
    done
  done
done > "$ROOT/logs/grid.txt"

echo "$(date -Is) grid has $(wc -l < "$ROOT/logs/grid.txt") cells, JOBS=$JOBS"

xargs -a "$ROOT/logs/grid.txt" -P "$JOBS" -L 1 bash -c '
  set -- $0 $@
  DS=$1; ARM=$2; CLF=$3
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
         NUMEXPR_NUM_THREADS=2 PYTHONUNBUFFERED=1
  LOG='"$ROOT"'/logs/cells/${DS}__${ARM}__${CLF}.log
  nice -n 19 ionice -c3 '"$PY"' '"$ROOT"'/run_bench.py \
    --dataset "$DS" --dp-method "$ARM" --classifier "$CLF" \
    --seeds '"$SEEDS"' --eps '"$EPS"' --out '"$ROOT"'/output > "$LOG" 2>&1
  echo "$(date -Is) exit=$? ${DS} ${ARM} ${CLF}"
'

echo "$(date -Is) GRID COMPLETE"
