#!/usr/bin/env bash
# Add an AIM arm to the comparison, then evaluate it.
#
# Why: on Adult, MST synthetic data (theirs and ours alike) yields a classifier
# *below* the majority-class rate at both budgets, so every fairness number in
# that panel is measured on a table that taught the model nothing. AIM is the
# one synthesizer in the benchmark that clears the baseline on Adult, so an AIM
# arm is what makes the Adult comparison interpretable at all.
set -uo pipefail
ROOT=/home/golobs/CausalFairnessInSDG/baselines
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
       NUMEXPR_NUM_THREADS=2 PYTHONUNBUFFERED=1

nice -n 19 ionice -c3 /home/golobs/miniconda3/envs/sdg/bin/python \
  "$ROOT/gen_ours.py" --synth aim --datasets Adult,Compas \
  --seeds 5,42,253,4112,32645 --eps 1,10 --mechs none,ftu,dp,cf \
  2>&1 | grep -avE "FutureWarning|df = df.groupby|UserWarning|warnings.warn"

echo "$(date -Is) AIM GENERATION COMPLETE"

# launch_grid.sh discovers arms by directory, so the four new ones join the
# grid automatically; finished cells are skipped by their .done markers.
ARMS_EXTRA=1 JOBS=8 "$ROOT/launch_grid.sh"
