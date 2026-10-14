import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
L=pd.read_parquet('long.parquet'); L['g']=L.dp_gap.abs(); L['sg']=L.s_dp_gap.abs(); L['auc']=L.downstream_auc_rf
r=pd.read_csv('/home/golobs/CausalFairnessInSDG/sweeps/facct-2026-10-09/fairpfn_baseline/results.csv',low_memory=False)
rr=r[(r.source=='real')&(r.model=='rf')&(r.ctx_size==10000)].groupby(['dataset','protected']).dp_gap.apply(lambda s:s.abs().mean()).rename('real').rename_axis(['dataset','attr'])
L=L.join(rr,on=['dataset','attr'])
F=L[L.batch_dir=='facct-fges-2026-10-13']
print('FGES none: AUC per dataset/method'); print(F[F.fairness_mechanism=='none'].drop_duplicates('run_id').pivot_table(index='dataset',columns='sdg_method',values='auc').round(3).to_string())
u=F[(F.fairness_mechanism=='none')&(F.real>0.05)].assign(ratio=lambda x:x.g/x.real)
print('FGES none: gap/real ratio'); print(u.pivot_table(index='dataset',columns='sdg_method',values='ratio').round(2).to_string()); print(u.groupby('sdg_method').ratio.describe().round(2).to_string())
key=['dataset','sdg_method','seed','role_config','attr']
n0=F[F.fairness_mechanism=='none'].set_index(key)[['g','auc','sg']].add_suffix('0')
P=F[F.fairness_mechanism!='none'].join(n0,on=key).dropna(subset=['g0'])
P['noop']=(P.sg-P.sg0).abs()<1e-12
print('share of exact no-op cells (in-table gap identical to none):'); print(P.groupby(['sdg_method','fairness_mechanism']).noop.mean().unstack().round(2).to_string())
b=P[P.real>0.05]
t=b.groupby(['sdg_method','fairness_mechanism']).apply(lambda q: pd.Series(dict(g0=q.g0.mean(),g=q.g.mean(),frac=1-q.g.sum()/q.g0.sum(),auc0=q.auc0.mean(),auc=q.auc.mean(),n=len(q)))).round(3)
print(t.to_string())
t=b[b.sdg_method=='decaf_diffusion'].groupby(['dataset','fairness_mechanism']).apply(lambda q: pd.Series(dict(g0=q.g0.mean(),g=q.g.mean(),auc0=q.auc0.mean(),auc=q.auc.mean()))).round(3)
print('diffusion on FGES, by dataset'); print(t.unstack().to_string())
# partial cut tabpfn
C=L[(L.fam=='tabpfn')&(L.graph=='hand')&L.batch_dir.str.contains('cut|facct-2026-10-09')]
C=C[C.dataset.isin(['acsincome_ca','acsincome_tx','snake','sbo','acspubcov_ca','acspubcov_tx','cdc_diabetes','hmda_ca','hmda_tx'])]
C=C.assign(s=np.where(C.fairness_mechanism=='none',0.0,C.cut))
C=C[~((C.fairness_mechanism=='none')&(C.batch_dir!='facct-2026-10-09'))]
b=C[C.real>0.05]
for mech in ['dp','cf','ftu']:
    q=pd.concat([b[b.fairness_mechanism=='none'],b[b.fairness_mechanism==mech]])
    print('partial cut',mech); print(q.pivot_table(index='dataset',columns='s',values=['g','auc']).round(3).to_string())
