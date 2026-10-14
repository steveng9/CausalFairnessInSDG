import pandas as pd, numpy as np, os
os.chdir('/home/golobs/.claude/jobs/8ac11971/tmp/an')
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
G=pd.read_parquet('groups.parquet')
G['mech']=G.fairness_mechanism.fillna('real'); G['gen']=G.sdg_method.fillna('real'); G['role_config']=G.role_config.fillna('-'); G['eps']=G.epsilon.fillna(-1)
G['inprot']=[(a in p) if isinstance(p,str) else False for a,p in zip(G.attr,G.protected_attrs)]
R=G[G.batch_dir=='real']
modal=R.groupby(['dataset','attr','val']).n.mean().reset_index().sort_values('n').groupby(['dataset','attr']).val.last().rename('modal')
G=G.join(modal,on=['dataset','attr']); G['side']=np.where(G.val==G.modal,'maj','rest')
G['npos']=G.n*G.base
for c in ['sel','acc']: G['w_'+c]=G[c]*G.n
G['w_tpr']=G.tpr.fillna(0)*G.npos
k=['key','attr','side']
B=G.groupby(k)[['n','npos','w_sel','w_acc','w_tpr']].sum()
B['sel']=B.w_sel/B.n; B['acc']=B.w_acc/B.n; B['tpr']=B.w_tpr/B.npos
Wd=B[['sel','acc','tpr']].unstack('side'); Wd.columns=[f'{a}_{b}' for a,b in Wd.columns]; Wd=Wd.reset_index()
meta=G.drop_duplicates(['key','attr'])[['key','attr','dataset','seed','batch_dir','gen','mech','eps','role_config','inprot']]
Wd=Wd.merge(meta,on=['key','attr'])
real=Wd[Wd.gen=='real'].groupby(['dataset','attr'])[['sel_maj','sel_rest']].mean().add_prefix('r_')
Wd=Wd.join(real,on=['dataset','attr'])
adv=(Wd.r_sel_maj>=Wd.r_sel_rest)
for m in ['sel','tpr','acc']:
    Wd[m+'_adv']=np.where(adv,Wd[m+'_maj'],Wd[m+'_rest']); Wd[m+'_dis']=np.where(adv,Wd[m+'_rest'],Wd[m+'_maj'])
Wd['rgap']=(Wd.r_sel_maj-Wd.r_sel_rest).abs()
Wd.to_parquet('sides.parquet')
import sys
bd=sys.argv[1] if len(sys.argv)>1 else 'facct-2026-10-09'
S=Wd[(Wd.gen!='real')&(Wd.batch_dir==bd)&(Wd.rgap>0.05)]
key=['dataset','seed','gen','eps','role_config','attr']
cols=['sel_adv','sel_dis','tpr_adv','tpr_dis','acc_adv','acc_dis']
n0=S[S.mech=='none'].set_index(key)[cols].add_suffix('0')
P=S[S.mech!='none'].join(n0,on=key).dropna(subset=['sel_adv0'])
for c in cols: P['d_'+c]=P[c]-P[c+'0']
P['gap0']=P.sel_adv0-P.sel_dis0; P['gap']=P.sel_adv-P.sel_dis
A=dict(n=('key','size'),gap0=('gap0','mean'),gap=('gap','mean'),d_sel_adv=('d_sel_adv','mean'),d_sel_dis=('d_sel_dis','mean'),d_tpr_adv=('d_tpr_adv','mean'),d_tpr_dis=('d_tpr_dis','mean'),d_acc_adv=('d_acc_adv','mean'),d_acc_dis=('d_acc_dis','mean'),dis_up=('d_sel_dis',lambda s:(s>0.005).mean()),dis_down=('d_sel_dis',lambda s:(s<-0.005).mean()))
print(bd,'PROTECTED attrs, real gap>0.05: mean change vs matched none run'); print(P[P.inprot].groupby(['gen','mech']).agg(**A).round(3).to_string())
print('UNPROTECTED in that split (spillover)'); print(P[~P.inprot].groupby(['gen','mech']).agg(**A).round(3).to_string())
print('by dataset, protected, causal fams'); t=P[P.inprot&P.gen.str.startswith('decaf')].groupby(['dataset','mech']).agg(gap0=('gap0','mean'),gap=('gap','mean'),d_sel_adv=('d_sel_adv','mean'),d_sel_dis=('d_sel_dis','mean'),d_tpr_dis=('d_tpr_dis','mean')).round(3); print(t.unstack('mech').to_string())
