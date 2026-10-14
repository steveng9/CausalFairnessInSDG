"""Re-score saved synthetic tables: fit the sweep's RF (random_state=0) on each
table, save its scores on the real holdout, and run an S-flip test.
Usage: rescore.py <worker> <n_workers>   (resumable; rows appended per run)"""
import sys, os, json, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,'/home/golobs/CausalFairnessInSDG/src')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from causal_fairness_sdg.data.datasets import DATASETS
R=Path('/home/golobs/CausalFairnessInSDG'); OUT=R/'sweeps/rescore-2026-10-13'
w,n=int(sys.argv[1]),int(sys.argv[2])
W=pd.read_parquet('/home/golobs/.claude/jobs/8ac11971/tmp/an/wide.parquet')
W=W[W.status=='done']
uni={}
for ds,g in W.groupby('dataset'):
    s=set()
    for p in g.protected_attrs.unique(): s|=set(json.loads(p))
    uni[ds]=sorted(s)
def prio(r):
    b=r.batch_dir; e=r.epsilon
    if b=='facct-2026-10-09': return 0 if (pd.isna(e) or e==1000) else (4 if e==1 else 5)
    if 'fges' in b: return 1
    if 'cut' in b: return 2
    if 'eps' in b: return 6
    return 9
W=W.assign(prio=[prio(r) for r in W.itertuples()])
W=W[W.prio<4].sort_values(['prio','batch_dir','shard','run_id']).reset_index(drop=True)
real=pd.DataFrame([dict(batch_dir='real',shard=f'{d}-real-s{s}',run_id=f'{d}-real-s{s}:0',dataset=d,seed=s) for d in sorted(uni) for s in range(5)])
T=pd.concat([real,W],ignore_index=True)
T=T.iloc[w::n]
if len(sys.argv)>3: T=T.iloc[::-1]
rowf=OUT/f'rows-{w}.jsonl'
done=set()
for f in OUT.glob('rows-*.jsonl'):
    for line in open(f):
        try: done.add(json.loads(line)['key'])
        except Exception: pass
cache={}
def get(ds,seed):
    k=(ds,seed)
    if k not in cache:
        cache.clear()
        data,dom,roles=DATASETS[ds]()
        tr,ho=train_test_split(data,test_size=0.3,random_state=int(seed))
        cache[k]=(tr,ho,next(iter(roles.outcome)))
    return cache[k]
for r in T.itertuples():
    key=f'{r.batch_dir}|{r.run_id}'
    if key in done: continue
    try:
        if r.batch_dir!='real' and (OUT/'preds'/r.batch_dir/r.shard/f"run_{r.run_id.split(':')[1]}.npy").exists(): continue
        tr,ho,y=get(r.dataset,r.seed)
        if r.batch_dir=='real': syn=tr
        else:
            rid=r.run_id.split(':')[1]
            p=R/'sweeps'/r.batch_dir/'synth'/r.shard/f'run_{rid}.parquet'
            if not p.exists(): continue
            syn=pd.read_parquet(p)
        if syn[y].nunique()<2: continue
        fc=[c for c in syn.columns if c!=y]
        m=RandomForestClassifier(random_state=0,n_jobs=1).fit(syn[fc],syn[y])
        ci=list(m.classes_).index(1)
        X=ho[fc]; pr=m.predict_proba(X)[:,ci]
        pd_=OUT/'preds'/r.batch_dir/r.shard; pd_.mkdir(parents=True,exist_ok=True)
        np.save(pd_/f"run_{r.run_id.split(':')[1]}.npy",pr.astype(np.float16))
        hard=(m.predict(X)==1)
        flip={}
        for a in uni[r.dataset]:
            vc=ho[a].value_counts(normalize=True); vals=[v for v,s in vc.items()][:2]
            res={}; base=None
            for v in vals:
                Xv=X.copy(); Xv[a]=v; pv=m.predict_proba(Xv)[:,ci]; hv=pv>0.5
                if base is None: base=(pv,hv)
                res[int(v)]=dict(mean=float(pv.mean()),pos=float(hv.mean()),flip=float((hv!=base[1]).mean()),absd=float(np.abs(pv-base[0]).mean()))
            flip[a]=res
        # in-table label gap per attribute (no classifier): P(Y=1|S!=mode)-P(Y=1|S=mode)
        tab={}
        for a in uni[r.dataset]:
            mo=ho[a].mode().iloc[0]; s1=syn[a]!=mo
            tab[a]=dict(lab=float(syn.loc[s1,y].mean()-syn.loc[~s1,y].mean()) if s1.any() and (~s1).any() else None)
        row=dict(key=key,batch_dir=r.batch_dir,shard=r.shard,run_id=r.run_id,dataset=r.dataset,seed=int(r.seed),
                 auc=float(roc_auc_score(ho[y],pr)),pos=float(hard.mean()),flip=flip,tab=tab,synth_pos=float(syn[y].mean()))
        with open(rowf,'a') as f: f.write(json.dumps(row)+'\n')
    except Exception as e:
        with open(OUT/f'err-{w}.log','a') as f: f.write(f'{key}\t{type(e).__name__}: {e}\n')
