#!/usr/bin/env bash
# Extra capacity started 2026-10-15, alongside sweeps/chain-cpu-2026-10-14.sh
# (cores 32-47), while the box was nearly idle. Shard locks keep the schedulers
# apart; the chain's later steps skip whatever these finish.
#   gpu1: DP causal families of facct-2026-10-09 (chain step 5), 1 worker on GPU 1
#   cpu : facct-eps-b-2026-10-13 (chain step 9), 10 workers on cores 20-31
#   gpu1b: 2 more workers on GPU 1 for the DP causal families, cores 17-19
#          (added 2026-10-16: shards are 3-4x faster on GPU 1 than on CPU, but one
#          single-threaded worker leaves it nearly idle; each uses < 1 GB)
# Stop: kill this script's children: `pkill -f "sweep.py run --batch facct-2026-10-09 .*--gpu 1"`,
# `pkill -f "sweep.py run --batch facct-eps-b-2026-10-13 .*--cpus 20-31"`.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
step() { echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
case "$1" in
  gpu1) step "gpu1: DP causal families"
        $SDG scripts/sweep.py run --batch facct-2026-10-09 --workers 1 --cpus 32-47 --gpu 1 \
          --methods decaf_dpgan,decaf_dpctgan,decaf_dpdiffusion ;;
  gpu1b) step "gpu1b: DP causal families, 2 more workers"
        $SDG scripts/sweep.py run --batch facct-2026-10-09 --workers 2 --cpus 17-19 --gpu 1 \
          --methods decaf_dpgan,decaf_dpctgan,decaf_dpdiffusion ;;
  cpu)  step "cpu 20-31: facct-eps-b"
        $SDG scripts/sweep.py run --batch facct-eps-b-2026-10-13 --workers 10 --cpus 20-31 --even-if-held ;;
esac
step "$1 done"
