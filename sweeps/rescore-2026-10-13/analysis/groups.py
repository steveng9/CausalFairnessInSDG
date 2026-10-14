"""Per-run, per-attribute-value group metrics from re-scored holdout predictions."""
import os; os.chdir("/home/golobs/.claude/jobs/8ac11971/tmp/an")
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,'/home/golobs/CausalFairnessInSDG/src')
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from causal_fairness_sdg.data.datasets import DATASETS
R=Path('/home/golobs/CausalFairnessInSDG'); OUT=R/'sweeps/rescore-2026-10-13'
rows=[json.loads(l) for f in glob.glob(str(OUT/'rows-*.jsonl')) for l in open(f)]
J=pd.DataFrame(rows).drop_duplicates('key')
W=pd.read_parquet('wide.parquet')
meta=W[['batch_dir','run_id','sdg_method','fairness_mechanism','epsilon','role_config','protected_attrs','admissible_attrs']]
J=J.merge(meta,on=['batch_dir','run_id'],how='left')
uni={ds:sorted(set().union(*[set(json.loads(p)) for p in g.protected_attrs.unique()])) for ds,g in W.groupby('dataset')}
out=[]; flips=[]
for (ds,seed),g in J.groupby(['dataset','seed']):
    data,dom,roles=DATASETS[ds](); y=next(iter(roles.outcome))
    tr,ho=train_test_split(data,test_size=0.3,random_state=int(seed)); yt=ho[y].to_numpy()
    masks={}
    for a in uni[ds]:
        for v,n in ho[a].value_counts().items():
            if n>=150: masks[(a,str(int(v)))]=(ho[a]==v).to_numpy()
    for r in g.itertuples():
        p=np.load(OUT/'preds'/r.batch_dir/r.shard/f"run_{r.run_id.split(':')[1]}.npy").astype(float); h=p>0.5
        thr=np.quantile(p,1-yt.mean()); hm=p>thr   # matched to real base rate
        for (a,v),m in masks.items():
            ym=yt[m]
            out.append((r.key,a,v,int(m.sum()),ym.mean(),h[m].mean(),hm[m].mean(),p[m].mean(),
                        h[m][ym==1].mean() if (ym==1).any() else np.nan, h[m][ym==0].mean() if (ym==0).any() else np.nan,
                        (h[m]==ym).mean(), roc_auc_score(ym,p[m]) if 0<ym.mean()<1 else np.nan))
        for a,res in (r.flip or {}).items():
            ks=list(res)
            if len(ks)==2: flips.append((r.key,a,res[ks[1]]['flip'],res[ks[1]]['mean']-res[ks[0]]['mean'],res[ks[1]]['pos']-res[ks[0]]['pos'], (r.tab or {}).get(a,{}).get('lab')))
G=pd.DataFrame(out,columns=['key','attr','val','n','base','sel','sel_m','score','tpr','fpr','acc','auc_g'])
Fl=pd.DataFrame(flips,columns=['key','attr','flip','dscore','dpos','tab_lab'])
Jm=J.drop(columns=['flip','tab'])
G=G.merge(Jm,on='key'); Fl=Fl.merge(Jm,on='key')
G.to_parquet('groups.parquet'); Fl.to_parquet('flips.parquet'); print(G.shape,Fl.shape,J.batch_dir.value_counts().to_dict())
