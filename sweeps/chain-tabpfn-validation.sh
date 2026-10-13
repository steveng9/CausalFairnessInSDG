#!/usr/bin/env bash
# Waits for the decaf_tabpfn scheduler on GPU 1 to exit, then scores TabPFN's
# conditionals on held-out rows for the datasets not yet in
# sweeps/facct-2026-10-09/tabpfn_conditionals.csv. One GPU process at a time.
cd "$(dirname "$0")/.."
while pgrep -f "sweep.py run --batch facct-2026-10-09 .*--methods decaf_tabpfn" > /dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 nice -n 19 taskset -c 32-47 \
  /home/golobs/miniconda3/envs/sdg_tabpfn/bin/python scripts/validate_tabpfn_conditionals.py \
  --datasets acsincome_tx,snake,sbo,acspubcov_ca,acspubcov_tx,cdc_diabetes --seed 0 --device cuda:0 \
  --out sweeps/facct-2026-10-09/tabpfn_conditionals.csv
