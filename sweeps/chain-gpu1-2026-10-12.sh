#!/usr/bin/env bash
# GPU 1 queue started 2026-10-12, one process at a time, in this order:
#   1. decaf_tabpfn on the datasets added to facct-2026-10-09 that day
#   2. held-out log-loss of TabPFN's conditionals on those datasets
#   3. partial-cut batch, strength 0.5
#   4. FairPFN baseline on the new datasets and the PrivBayes / PrivSyn releases
#      (waits for the CPU scheduler given as $1 to finish the marginal runs)
#   5. partial-cut batches, strengths 0.25 and 0.75
# Stop: kill this script, then `pkill -f "sweep.py run --batch facct-.*--gpu 1"`.
cd "$(dirname "$0")/.."
CPU_SCHEDULER_PID="$1"
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
TABPFN=/home/golobs/miniconda3/envs/sdg_tabpfn/bin/python
FAIRPFN=/home/golobs/miniconda3/envs/fairpfn/bin/python
ALL=acsincome_ca,acsincome_tx,acspubcov_ca,acspubcov_tx,cdc_diabetes,sbo,snake,hmda_ca,hmda_tx,adult,compas
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
sweep() { $SDG scripts/sweep.py run --batch "$1" --workers 1 --cpus 32-47 --gpu 1 --methods decaf_tabpfn; }

step "1 decaf_tabpfn, new datasets"; sweep facct-2026-10-09
step "2 conditionals"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 nice -n 19 taskset -c 32-47 $TABPFN -u \
  scripts/validate_tabpfn_conditionals.py --datasets hmda_ca,hmda_tx,adult,compas --seed 0 \
  --device cuda:0 --out sweeps/facct-2026-10-09/tabpfn_conditionals.csv \
  >> sweeps/facct-2026-10-09/tabpfn_conditionals.log 2>&1
step "3 cut strength 0.5"; sweep facct-cut050-2026-10-12
step "4 FairPFN (waiting for CPU scheduler $CPU_SCHEDULER_PID)"
while kill -0 "$CPU_SCHEDULER_PID" 2>/dev/null; do sleep 60; done
OMP_NUM_THREADS=4 nice -n 19 taskset -c 32-47 $SDG scripts/fairpfn_baseline.py export \
  --datasets $ALL --synth-methods mst,privbayes,privsyn,decaf_tabpfn \
  >> sweeps/facct-2026-10-09/fairpfn_baseline/export-1012.log 2>&1
CUDA_VISIBLE_DEVICES=1 nice -n 19 taskset -c 32-47 $FAIRPFN -u scripts/fairpfn_baseline.py run \
  >> sweeps/facct-2026-10-09/fairpfn_baseline/run.log 2>&1
nice -n 19 taskset -c 32-47 $SDG scripts/summarize_fairpfn_baseline.py \
  >> sweeps/facct-2026-10-09/fairpfn_baseline/export-1012.log 2>&1
step "5 cut strength 0.25"; sweep facct-cut025-2026-10-12
step "5 cut strength 0.75"; sweep facct-cut075-2026-10-12
step "done"
