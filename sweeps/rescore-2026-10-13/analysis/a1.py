import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
L=pd.read_parquet('long.parquet')
L['g']=L.dp_gap.abs(); L['c']=L.cdp_gap.abs(); L['auc']=L.downstream_auc_rf
M=L[(L.batch_dir=='facct-2026-10-09')]
M=M.assign(eps=M.epsilon.fillna(-1))
key=['dataset','sdg_method','eps','seed','role_config','attr']
none=M[M.fairness_mechanism=='none'].set_index(key)[['g','c','auc','dp_gap','tvd_2way']].add_suffix('0')
P=M[M.fairness_mechanism!='none'].join(none,on=key)
P['dg']=P.g-P.g0; P['dauc']=P.auc-P.auc0; P['dc']=P.c-P.c0
P.to_parquet('pairs_main.parquet')
# baseline: none gap per dataset/attr by method at eps 1000 / tabpfn
b=M[(M.fairness_mechanism=='none')&(M.eps.isin([-1,1000,1]))]
b=b.assign(col=b.sdg_method+'@'+b.eps.astype(int).astype(str))
print('UNMITIGATED |gap| (mean over seeds, splits)'); print(b.pivot_table(index=['dataset','attr'],columns='col',values='g').round(3).to_string())
print('UNMITIGATED AUC'); print(b.pivot_table(index=['dataset'],columns='col',values='auc').round(3).to_string())
# mechanism effects: tabpfn & marg@1000, by dataset, mechanism (mean over attr, split)
for nm,q in [('tabpfn',P[P.sdg_method=='decaf_tabpfn']),('marg eps1000',P[(P.fam=='marg')&(P.eps==1000)]),('marg eps1',P[(P.fam=='marg')&(P.eps==1)])]:
    t=q.groupby(['dataset','fairness_mechanism']).agg(g0=('g0','mean'),g=('g','mean'),c0=('c0','mean'),c=('c','mean'),auc0=('auc0','mean'),auc=('auc','mean')).round(3)
    print(nm); print(t.unstack().swaplevel(axis=1).sort_index(axis=1).to_string())
