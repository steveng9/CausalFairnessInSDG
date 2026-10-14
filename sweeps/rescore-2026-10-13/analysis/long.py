import pandas as pd, numpy as np, re
d=pd.read_parquet('wide.parquet'); d=d[d.status=='done'].copy()
ms=['dp_gap','cdp_gap','tprb','tnrb','ctprb','ctnrb']
attrs=sorted({c.split('__',1)[1] for c in d.columns if c.startswith('dp_gap__')})
base=[c for c in d.columns if '__' not in c]
rows=[]
for a in attrs:
    sub=d[base].copy(); sub['attr']=a
    for m in ms:
        sub[m]=d.get(f'{m}__{a}'); sub['s_'+m]=d.get(f'synthref__{m}__{a}')
    rows.append(sub[sub.dp_gap.notna()|sub.s_dp_gap.notna()])
L=pd.concat(rows,ignore_index=True)
L['prot']=[a in p for a,p in zip(L.attr,L.protected_attrs)]
L['graph']=np.where(L.batch_dir.str.contains('fges'),'fges','hand')
L['cut']=L.batch_dir.map(lambda b: .25 if 'cut025' in b else .5 if 'cut050' in b else .75 if 'cut075' in b else 1.0)
fam={'mst':'marg','privbayes':'marg','privsyn':'marg','decaf':'gan','decaf_ctgan':'gan','decaf_dpgan':'gan','decaf_dpctgan':'gan','decaf_diffusion':'diff','decaf_dpdiffusion':'diff','decaf_tabpfn':'tabpfn'}
L['fam']=L.sdg_method.map(fam)
L.to_parquet('long.parquet')
print(L.shape, d.eval_reference.unique())
m=d[d.batch_dir=='facct-2026-10-09']
print(m.groupby(['dataset','role_config']).agg(prot=('protected_attrs','first'),adm=('admissible_attrs','first')).to_string())
x=L[L.batch_dir=='facct-2026-10-09']
print(x.groupby(['dataset','role_config','attr']).prot.first().unstack().fillna('-').to_string())
print(x.dp_gap.describe())
