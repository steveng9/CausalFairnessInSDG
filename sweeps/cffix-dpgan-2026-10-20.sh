#!/usr/bin/env bash
# CF re-run for the DP-GAN / DP-CTGAN DECAF families, 2026-10-20, GPU 0 only
# (GPU 1 free for Daniil), cores 4-15 (0-3 left free).
# Stop: kill this script, then `pkill -f "sweep.py run --batch facct-cffix-dpgan"`.
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
echo "[$(date -u '+%m-%d %H:%M:%S')] facct-cffix-dpgan"
$SDG scripts/sweep.py run --batch facct-cffix-dpgan-2026-10-20 --workers 10 --cpus 4-15 --gpu 0
echo "[$(date -u '+%m-%d %H:%M:%S')] done"
