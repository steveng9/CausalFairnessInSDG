import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
L=pd.read_parquet('long.parquet'); L['g']=L.dp_gap.abs(); L['auc']=L.downstream_auc_rf
r=pd.read_csv('/home/golobs/CausalFairnessInSDG/sweeps/facct-2026-10-09/fairpfn_baseline/results.csv',low_memory=False)
rr=r[(r.source=='real')&(r.model=='rf')&(r.ctx_size==10000)].groupby(['dataset','protected']).dp_gap.apply(lambda s:s.abs().mean()).rename('real').rename_axis(['dataset','attr'])
L=L.join(rr,on=['dataset','attr'])
D=L[((L.batch_dir=='facct-2026-10-09')&(L.epsilon.isna()|(L.epsilon==1000)))|(L.batch_dir=='facct-fges-2026-10-13')]
D=D[D.real>0.05].copy(); D['cell']=D.dataset+'/'+D.attr; D['gen']=D.sdg_method+'-'+D.graph
import statsmodels.formula.api as smf, statsmodels.api as sm
def eta(df,y,form):
    m=smf.ols(f'{y} ~ {form}',df).fit(); a=sm.stats.anova_lm(m,typ=2); return (a.sum_sq/a.sum_sq.sum()).round(3)
print('n',len(D)); 
print('gap, all mechanisms incl none'); print(eta(D,'g','C(cell)+C(fairness_mechanism)+C(gen)+C(role_config):C(dataset)+C(seed)').to_string())
print('gap, with interactions'); print(eta(D,'g','C(cell)+C(fairness_mechanism)*C(gen)+C(cell):C(gen)+C(seed)').to_string())
print('AUC'); print(eta(D.drop_duplicates('run_id'),'auc','C(dataset)+C(fairness_mechanism)*C(gen)+C(seed)').to_string())
# seed noise: sd across seeds of gap within cell vs mean mechanism effect
sd=D.groupby(['cell','gen','fairness_mechanism','role_config']).g.std().mean(); print('mean within-cell seed sd of gap:',round(sd,4))
# simple table: unmitigated gap by generator vs mitigated — range across generators of none gap, vs mean mechanism effect
n=D[D.fairness_mechanism=='none'].groupby(['cell','gen']).g.mean().unstack()
print('per-cell range across 7 generators of UNMITIGATED gap: mean',round((n.max(1)-n.min(1)).mean(),3),' as share of real gap:',round(((n.max(1)-n.min(1))/D.groupby('cell').real.first()).mean(),2))
for mech in ['ftu','cf','dp']:
    m=D[D.fairness_mechanism==mech].groupby(['cell','gen']).g.mean().unstack()
    print(mech,'mean reduction vs none (same generator):',round((n-m).mean().mean(),3))
