import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
L=pd.read_parquet('long.parquet'); L['g']=L.dp_gap.abs(); L['sg']=L.s_dp_gap.abs(); L['auc']=L.downstream_auc_rf
E=L[L.batch_dir.isin(['facct-2026-10-09','facct-eps-a-2026-10-13'])&(L.fam=='marg')].copy()
r=pd.read_csv('/home/golobs/CausalFairnessInSDG/sweeps/facct-2026-10-09/fairpfn_baseline/results.csv',low_memory=False)
rr=r[(r.source=='real')&(r.model=='rf')&(r.ctx_size==10000)].groupby(['dataset','protected']).dp_gap.apply(lambda s:s.abs().mean()).rename('real').rename_axis(['dataset','attr'])
E=E.join(rr,on=['dataset','attr'])
big=E[E.real>0.05]
print('cells per eps/method (none):'); print(E[E.fairness_mechanism=='none'].groupby(['sdg_method','epsilon']).size().unstack().to_string())
print('== attrs with real gap>0.05: mean real-holdout gap by mechanism x eps, per method')
print(big.pivot_table(index=['sdg_method','fairness_mechanism'],columns='epsilon',values='g').round(3).to_string())
print('== same, AUC'); print(big.drop_duplicates(['run_id']).pivot_table(index=['sdg_method','fairness_mechanism'],columns='epsilon',values='auc').round(3).to_string())
print('== in-table gap'); print(big.pivot_table(index=['sdg_method','fairness_mechanism'],columns='epsilon',values='sg').round(3).to_string())
print('== tvd2'); print(big.drop_duplicates(['run_id']).pivot_table(index=['sdg_method','fairness_mechanism'],columns='epsilon',values='tvd_2way').round(3).to_string())
# matched-pair removal fraction
key=['dataset','sdg_method','epsilon','seed','role_config','attr']
n0=big[big.fairness_mechanism=='none'].set_index(key)[['g','auc']].add_suffix('0')
P=big[big.fairness_mechanism!='none'].join(n0,on=key).dropna(subset=['g0'])
t=P.groupby(['sdg_method','fairness_mechanism','epsilon']).apply(lambda q: pd.Series(dict(frac_removed=1-q.g.sum()/q.g0.sum(), dAUC=(q.auc-q.auc0).mean())))
print('== pooled fraction of gap removed'); print(t.frac_removed.unstack().round(2).to_string()); print('== dAUC'); print(t.dAUC.unstack().round(3).to_string())
# unmitigated ratio to real, by eps, per dataset for privbayes
u=big[big.fairness_mechanism=='none'].assign(ratio=lambda x:x.g/x.real)
print('== unmitigated gap / real gap, median over attrs'); print(u.pivot_table(index='sdg_method',columns='epsilon',values='ratio',aggfunc='median').round(2).to_string())
print(u.pivot_table(index=['sdg_method','dataset'],columns='epsilon',values='ratio',aggfunc='mean').round(2).to_string())
