import pandas as pd, numpy as np, os
os.chdir('/home/golobs/.claude/jobs/8ac11971/tmp/an')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
S = pd.read_parquet('sides.parquet'); S['gap'] = S.sel_adv - S.sel_dis
print('runs per batch/gen'); print(S.drop_duplicates('key').groupby(['batch_dir', 'gen']).size().to_string())
# F1 with the 35k real baseline
U = S[(S.mech == 'none') & (S.rgap > 0.05) & S.batch_dir.isin(['facct-2026-10-09', 'facct-fges-2026-10-13']) & (S.eps.isin([-1, 1000]))].copy()
U['genb'] = U.gen + np.where(U.batch_dir.str.contains('fges'), '@fges', '')
U['ratio'] = U.gap / U.rgap
c = U.groupby(['dataset', 'attr', 'genb']).ratio.mean().unstack()
print('F1: unmitigated gap / train-on-real gap (35k), per cell')
print(pd.concat([U.groupby(['dataset', 'attr']).rgap.first().round(3), c.round(2)], axis=1).to_string())
print(c.describe().round(2).to_string())
print('share of cells with ratio<0.67:', (c < 0.67).mean().round(2).to_dict())
print('share of cells with ratio>1.2:', (c > 1.2).mean().round(2).to_dict())
G = pd.read_parquet('groups.parquet'); G['mech'] = G.fairness_mechanism.fillna('real'); G['gen'] = G.sdg_method.fillna('real'); G['role_config'] = G.role_config.fillna('-')


def contrast(ds, attr, a, b, split=None):
    q = G[(G.dataset == ds) & (G.attr == attr) & G.val.isin([a, b]) & ((G.epsilon.isna()) | (G.epsilon == 1000)) & G.batch_dir.isin(['real', 'facct-2026-10-09'])]
    if split: q = q[q.role_config.isin(['-', split])]
    w = q.pivot_table(index=['key', 'gen', 'mech'], columns='val', values=['sel', 'sel_m', 'score', 'tpr', 'acc'])
    o = pd.DataFrame({m + '_gap': w[m][a] - w[m][b] for m in ['sel', 'sel_m', 'score', 'tpr']})
    o['acc_' + a] = w['acc'][a]; o['acc_' + b] = w['acc'][b]; o['sel_' + a] = w['sel'][a]; o['sel_' + b] = w['sel'][b]
    print(ds, attr, f'value {a} minus value {b}', split); print(o.groupby(['gen', 'mech']).mean().round(3).to_string())


contrast('hmda_ca', 'race', '0', '1', 'race_narrow_adm'); contrast('hmda_tx', 'race', '0', '1', 'race_narrow_adm')
contrast('compas', 'race', G[(G.dataset=='compas')&(G.attr=='race')].val.value_counts().index[0], G[(G.dataset=='compas')&(G.attr=='race')].val.value_counts().index[1], 'prefair')
