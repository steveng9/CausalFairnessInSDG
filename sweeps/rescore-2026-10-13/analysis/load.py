import sys, json, sqlite3, glob, os
from pathlib import Path
import pandas as pd, numpy as np
R=Path('/home/golobs/CausalFairnessInSDG'); sys.path.insert(0,str(R/'scripts'))
from merge_sweep_dbs import dedupe, load_shard
OUT=Path('/home/golobs/.claude/jobs/8ac11971/tmp/an')
fr=[]
for b in sorted((R/'sweeps').glob('facct-*')):
    if not (b/'db').is_dir(): continue
    for p in sorted((b/'db').glob('*.db')):
        try:
            f=load_shard(p)
            if not f.empty:
                f['batch_dir']=b.name; fr.append(f)
        except Exception as e: print('skip',p.name,e)
d=pd.concat(fr,ignore_index=True)
d=pd.concat([dedupe(g) for _,g in d.groupby('batch_dir')],ignore_index=True)
print(d.shape); print(d.groupby(['batch_dir','status']).size())
d.to_parquet(OUT/'wide.parquet')
print([c for c in d.columns if '__' not in c])
