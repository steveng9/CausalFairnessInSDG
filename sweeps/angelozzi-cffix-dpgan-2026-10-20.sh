#!/usr/bin/env bash
# After facct-cffix-dpgan-2026-10-20 and the first Angelozzi CF re-score finish
# (PIDs $1 and $2): point the CF arms of decaf_dpgan / decaf_dpctgan at the new
# tables (export --merge), move their old-rule outputs aside, re-score, collect.
# Usage: angelozzi-cffix-dpgan-2026-10-20.sh <cffix-dpgan pid> <angelozzi-cffix pid>
cd "$(dirname "$0")/.."
SDG=/home/golobs/miniconda3/envs/sdg/bin/python
D=baselines/facct_data; OUT=baselines/facct_output
for p in "$1" "$2"; do while kill -0 "$p" 2>/dev/null; do sleep 60; done; done
echo "[$(date -u '+%m-%d %H:%M:%S')] export"
BK=$D/backup-pre-cffix-dpgan-2026-10-20; mkdir -p $BK
for m in $D/*/manifest.json; do n=$(basename $(dirname $m)); mkdir -p $BK/$n; cp $m $BK/$n/; done
$SDG baselines/facct_bench.py export --merge --batches facct-cffix-dpgan-2026-10-20 --methods decaf_dpgan,decaf_dpctgan
$SDG - "$BK" > $D/grid_cffix_dpgan_2026-10-20.txt <<'PY'
import glob, json, os, sys
bk = sys.argv[1]
for b in sorted(glob.glob(bk + "/*/manifest.json")):
    n = os.path.basename(os.path.dirname(b))
    old = json.load(open(b))["tags"]
    new = json.load(open(f"baselines/facct_data/{n}/manifest.json"))["tags"]
    for t in new:
        if old.get(t) != new[t]:
            print(n, t, "XGB")
PY
echo "[$(date -u '+%m-%d %H:%M:%S')] $(wc -l < $D/grid_cffix_dpgan_2026-10-20.txt) cells"
INV=$OUT/invalid-cf-oldrule-2026-10-20; mkdir -p $INV
while read name tag clf; do d="$OUT/${name}__${tag}__${clf}"; [ -d "$d" ] && mv "$d" $INV/; done < $D/grid_cffix_dpgan_2026-10-20.txt
GRID=$D/grid_cffix_dpgan_2026-10-20.txt JOBS=28 CPUS=16-47 baselines/launch_facct_grid.sh
$SDG baselines/facct_bench.py collect
echo "[$(date -u '+%m-%d %H:%M:%S')] done"
