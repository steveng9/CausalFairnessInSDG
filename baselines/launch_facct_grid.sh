#!/usr/bin/env bash
# Fan the FAccT benchmark grid (facct_data/grid.txt, written by
# `facct_bench.py export`) across cores. Each cell drops a `.done` marker, so
# re-running picks up only what is missing. Shared machine: nice 19, ionice
# idle, 1 thread per process, confined to $CPUS. ONLY (a grep -E pattern over
# the "name arm classifier" lines) restricts the run, e.g. ONLY=' [a-z_]+-none '
# for the no-mechanism arms, which carry the downstream-mitigator baseline.
set -uo pipefail
ROOT=/home/golobs/CausalFairnessInSDG/baselines
PY=/home/golobs/miniconda3/envs/sdg/bin/python
JOBS="${JOBS:-14}"
CPUS="${CPUS:-32-47}"
GRID="${GRID:-$ROOT/facct_data/grid.txt}"
ONLY="${ONLY:-.}"
CELLS=$(mktemp)
grep -E -- "$ONLY" "$GRID" > "$CELLS"
mkdir -p "$ROOT/logs/facct_cells" "$ROOT/facct_output"
echo "$(date -Is) grid has $(wc -l < "$CELLS") cells (ONLY=$ONLY), JOBS=$JOBS CPUS=$CPUS"
xargs -a "$CELLS" -P "$JOBS" -L 1 bash -c '
  set -- $0 $@
  NAME=$1; TAG=$2; CLF=$3
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
         NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
  LOG='"$ROOT"'/logs/facct_cells/${NAME}__${TAG}__${CLF}.log
  taskset -c '"$CPUS"' nice -n 19 ionice -c3 '"$PY"' '"$ROOT"'/facct_bench.py cell "$NAME" "$TAG" "$CLF" > "$LOG" 2>&1
  echo "$(date -Is) exit=$? $NAME $TAG $CLF"
' >> "$ROOT/logs/facct_grid.log" 2>&1
rm -f "$CELLS"
echo "$(date -Is) GRID COMPLETE"
