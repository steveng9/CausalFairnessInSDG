import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
L=pd.read_parquet('long.parquet'); L['g']=L.dp_gap.abs(); L['sg']=L.s_dp_gap.abs(); L['auc']=L.downstream_auc_rf
M=L[L.batch_dir=='facct-2026-10-09'].assign(eps=lambda x:x.epsilon.fillna(-1))
F=L[L.batch_dir=='facct-fges-2026-10-13'].assign(eps=-1)
print('== in-table (synthref) gap vs real-holdout gap, mean over datasets/attrs/splits/seeds')
for nm,q in [('tabpfn hand',M[M.fam=='tabpfn']),('marg e1000',M[(M.fam=='marg')&(M.eps==1000)]),('marg e1',M[(M.fam=='marg')&(M.eps==1)]),('fges gan',F[F.fam=='gan']),('fges diff',F[F.fam=='diff'])]:
    print(nm); print(q.groupby('fairness_mechanism').agg(synth_gap=('sg','mean'),real_gap=('g','mean'),auc=('auc','mean'),n=('g','size')).round(3).to_string())
print('== per dataset, DemPar mechanism: synth gap / real gap / none real gap')
for nm,q in [('tabpfn',M[M.fam=='tabpfn']),('marg e1000',M[(M.fam=='marg')&(M.eps==1000)])]:
    t=q[q.fairness_mechanism.isin(['none','dp'])].pivot_table(index='dataset',columns='fairness_mechanism',values=['sg','g','auc']).round(3)
    print(nm); print(t.to_string())
# real baseline gap from fairpfn control rf real 10k
r=pd.read_csv('/home/golobs/CausalFairnessInSDG/sweeps/facct-2026-10-09/fairpfn_baseline/results.csv',low_memory=False)
print(r.source.unique(), r.model.unique(), r.ctx_size.unique())
rr=r[(r.source=='real')&(r.model=='rf')&(r.ctx_size==r.ctx_size.max())].groupby(['dataset','protected']).dp_gap.apply(lambda s:s.abs().mean()).rename('real')
b=M[M.fairness_mechanism=='none'].assign(col=lambda x:x.sdg_method+'@'+x.eps.astype(int).astype(str)).pivot_table(index=['dataset','attr'],columns='col',values='g')
b=b.join(rr.rename_axis(['dataset','attr']))
ratio=b.div(b.real,axis=0).drop(columns='real')
big=b[b.real>0.05]
print('== ratio synthetic-none gap / train-on-real gap, attrs with real gap>0.05'); print(pd.concat([big.real.round(3),ratio.loc[big.index].round(2)],axis=1).to_string())
print(ratio.loc[big.index].describe().round(2).to_string())
