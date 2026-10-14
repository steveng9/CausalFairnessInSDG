import pandas as pd, numpy as np, os
os.chdir('/home/golobs/.claude/jobs/8ac11971/tmp/an')
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
F=pd.read_parquet('flips.parquet'); F['mech']=F.fairness_mechanism.fillna('real'); F['gen']=F.sdg_method.fillna('real')
F['inprot']=[(a in p) if isinstance(p,str) else False for a,p in zip(F.attr,F.protected_attrs)]
F['ad']=F.dscore.abs(); F['tl']=F.tab_lab.abs()
q=F[(F.batch_dir.isin(['real','facct-2026-10-09']))&((F.gen=='real')|F.inprot)]
print('FLIP TEST on real holdout (flip S: modal -> largest other value). share of predictions that change / mean |score shift|; in-table label gap')
print(q.groupby(['gen','mech']).agg(n=('key','size'),flip=('flip','mean'),absd=('ad','mean'),tab_label_gap=('tl','mean')).round(4).to_string())
print(q[q.gen.isin(['real','decaf_tabpfn','mst'])].pivot_table(index=['dataset','attr'],columns=['gen','mech'],values='flip').round(3).to_string())
