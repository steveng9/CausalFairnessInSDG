#!/usr/bin/env bash
# Launch the experiment grid fully detached, so it survives logging out, and
# at low enough priority that it stays out of the way of other jobs on the
# shared box.
#
#   ./scripts/launch_experiments.sh [extra args passed to run_experiments.py]
#
# Politeness measures (this machine routinely runs at load ~32/48 from other
# users' jobs):
#   - nice 19 / ionice idle: the kernel hands our CPU and disk time straight
#     back whenever anything else wants it.
#   - 2 threads per BLAS/OMP/torch pool: without this, numpy and torch each
#     grab all 48 cores and thrash against everyone else's jobs. Must be set
#     before numpy/torch are imported, hence here rather than in Python.
#   - CPU only, no CUDA: DECAF benchmarked the same speed on CPU as on GPU
#     (17.8s vs 16.3s per Adult epoch), so there's no reason to occupy a GPU
#     someone else could use.
#
# The run is resumable: re-running this script skips configs that already
# have a done/partial row for the batch tag.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/golobs/miniconda3}"
CONDA_ENV="${CONDA_ENV:-sdg}"
BATCH="${BATCH:-overnight-2026-07-30}"

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/results"
LOG="$REPO_ROOT/logs/${BATCH}.log"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export TORCH_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# setsid detaches from this terminal's session, so the run keeps going after
# logout; nohup covers the SIGHUP; </dev/null stops it blocking on stdin.
setsid nohup nice -n 19 ionice -c3 \
  "$CONDA_BASE/envs/$CONDA_ENV/bin/python" \
  "$REPO_ROOT/scripts/run_experiments.py" --batch "$BATCH" "$@" \
  >>"$LOG" 2>&1 </dev/null &

PID=$!
disown "$PID" 2>/dev/null || true
sleep 2
echo "Launched batch '$BATCH' as PID $PID"
echo "  log:     $LOG"
echo "  report:  $REPO_ROOT/results/${BATCH}_report.md  (rewritten after every run)"
echo "  csv:     $REPO_ROOT/results/${BATCH}_runs.csv"
echo "  stop:    kill $PID"
