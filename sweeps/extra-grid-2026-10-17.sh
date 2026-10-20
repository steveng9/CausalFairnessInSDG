#!/usr/bin/env bash
# Second Angelozzi launcher beside step 10 of sweeps/chain-cpu-2026-10-14.sh
# (cores 32-47), started 2026-10-17 on cores 16-31 while the box was half idle.
# It walks the same grid in reverse order; each cell skips itself once another
# launcher has written its `.done`, so the two meet in the middle with at most a
# few cells run twice (identical results).
# Stop: `pkill -f "grid_reverse_2026-10-17"`.
cd "$(dirname "$0")/.."
GRID=baselines/facct_data/grid_reverse_2026-10-17.txt JOBS=14 CPUS=16-31 baselines/launch_facct_grid.sh
