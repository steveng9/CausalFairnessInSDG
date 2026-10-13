import pandas as pd, numpy as np, matplotlib as mpl, os
mpl.use('Agg')
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,'figs'); os.makedirs(OUT,exist_ok=True)
CSV=os.path.join(HERE,os.pardir,'report','data','all_runs.csv')
mpl.rcParams.update({
 'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'font.size':7.0,'axes.labelsize':6.8,'axes.titlesize':7.4,'legend.fontsize':6.5,
 'xtick.labelsize':6.4,'ytick.labelsize':6.4,'axes.linewidth':0.5,
 'xtick.major.width':0.5,'ytick.major.width':0.5,'xtick.major.size':2,'ytick.major.size':2,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
 'grid.linewidth':0.4,'grid.color':'#d8d7d2','figure.dpi':200,'savefig.bbox':'tight',
 'savefig.pad_inches':0.012,'pdf.fonttype':42})

BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'
INK,INK2,MUT='#0b0b0b','#52514e','#8a8880'
FAMC={'marginal':BLUE,'gan':ORANGE,'diff':AQUA}
MECHC={'ftu':BLUE,'cf':ORANGE,'dp':AQUA}
FAMN={'marginal':'Marginals-based','gan':'Causal GAN','diff':'Causal diffusion'}
MECHN={'ftu':'FTU','cf':'CF','dp':'DemPar'}
MAJ={'adult':0.747,'compas':0.548}
FAM={'mst':'marginal','privbayes':'marginal','privsyn':'marginal',
     'decaf':'gan','decaf_ctgan':'gan','decaf_dpgan':'gan','decaf_dpctgan':'gan',
     'decaf_diffusion':'diff','decaf_dpdiffusion':'diff'}
NICE={'mst':'MST','privbayes':'PrivBayes','privsyn':'PrivSyn','decaf':'DECAF',
 'decaf_ctgan':'+CTGAN','decaf_dpgan':'+DP-GAN','decaf_dpctgan':'+DP-CTGAN',
 'decaf_diffusion':'+Diff','decaf_dpdiffusion':'+DP-Diff'}
ORDER=['mst','privbayes','privsyn','decaf','decaf_ctgan','decaf_dpgan','decaf_dpctgan',
       'decaf_diffusion','decaf_dpdiffusion']
fams=['marginal','gan','diff']; mechs=['ftu','cf','dp']; EPS=[1.0,10.0,1000.0]

r=pd.read_csv(CSV,low_memory=False)
r=r[~r.superseded.astype(bool)].copy()
r['acc']=r.downstream_accuracy_mlp; r['lift']=r.acc-r.dataset.map(MAJ)
r['family']=r.sdg_method.map(FAM)

# ============ single results figure: fidelity | utility | fairness-vs-eps ============
b=r[r.fairness_mechanism=='none']
agg=b.groupby(['dataset','sdg_method']).agg(tvd2=('tvd_2way','mean'),lift=('lift','mean')).reset_index()

KEY=['dataset','sdg_method','epsilon','role_config','seed']
piv=r.pivot_table(index=KEY,columns='fairness_mechanism',
                  values=['fairness_gap','acc'],dropna=False)
lv=piv.index.get_level_values
def pairs(mask,mech,metric='fairness_gap'):
    sel=piv[mask]; g0=sel[(metric,'none')]; g1=sel[(metric,mech)]
    m=g0.notna()&g1.notna(); d=(g1-g0)[m]
    return dict(n=int(m.sum()),delta=d.mean(),se=d.std(ddof=1)/np.sqrt(max(m.sum(),1)))

fig=plt.figure(figsize=(5.5,1.62))
gs=fig.add_gridspec(1,3,width_ratios=[1.22,1.0,0.92],wspace=0.16)
ax0,ax1,ax2=[fig.add_subplot(gs[0,i]) for i in range(3)]
y=np.arange(len(ORDER))
for ax,col,lab in zip([ax0,ax1],['tvd2','lift'],
        ['(a) 2-way TVD  (lower = truer)','(b) accuracy over baseline']):
    for ds,hatch,alpha in [('adult',None,1.0),('compas','////',0.5)]:
        d=agg[agg.dataset==ds].set_index('sdg_method').reindex(ORDER)
        off=-0.19 if ds=='adult' else 0.19
        ax.barh(y+off,d[col],height=0.34,color=[FAMC[FAM[m]] for m in ORDER],alpha=alpha,
                hatch=hatch,edgecolor='white',linewidth=0.45,zorder=3)
    ax.axvline(0,color=INK,lw=0.7,zorder=4)
    ax.set_yticks(y); ax.set_ylim(len(ORDER)-0.45,-0.55)
    ax.set_xlabel(lab,color=INK2,labelpad=1.5)
    ax.grid(axis='x',zorder=0);ax.set_axisbelow(True);ax.tick_params(colors=INK2)
ax0.set_yticklabels([NICE[m] for m in ORDER]); ax1.set_yticklabels([])
ax1.set_xlim(-0.145,0.125)

# Panel (c) shows CF only, coloured by FAMILY so that colour means the same
# thing in all three panels.  Only the two families present in every (eps,
# dataset) cell are drawn; the GAN family collapses at tight budgets.
for fam in ('marginal','diff'):
    ms=[m for m,f in FAM.items() if f==fam]
    for ds,ls,mk in [('adult','-','o'),('compas','--','s')]:
        ax2.plot(EPS,[pairs((lv('dataset')==ds)&(lv('epsilon')==e)
                            &(lv('sdg_method').isin(ms)),'cf')['delta'] for e in EPS],
                 ls,marker=mk,color=FAMC[fam],lw=1.1,ms=2.8,zorder=3)
ax2.axhline(0,color=INK,lw=0.7,zorder=4)
ax2.set_xscale('log');ax2.set_xticks(EPS);ax2.set_xticklabels(['1','10','1000']);ax2.minorticks_off()
ax2.set_xlabel(r'(c) CF: $\Delta$ parity gap vs. $\varepsilon$',color=INK2,labelpad=1.5)
ax2.yaxis.tick_right(); ax2.spines['right'].set_visible(True); ax2.spines['left'].set_visible(False)
ax2.grid(zorder=0);ax2.set_axisbelow(True);ax2.tick_params(colors=INK2)

h=[plt.Rectangle((0,0),1,1,color=FAMC[f]) for f in fams]+[
   plt.Rectangle((0,0),1,1,fc='#9d9b94',ec='white',lw=0.45),
   plt.Rectangle((0,0),1,1,fc='#9d9b94',ec='white',lw=0.45,alpha=0.5,hatch='////')]
lbl=[FAMN[f] for f in fams]+['Adult','COMPAS']
fig.legend(h,lbl,loc='lower center',ncol=5,bbox_to_anchor=(0.5,-0.235),
           handlelength=1.05,columnspacing=1.2,handletextpad=0.45,borderpad=0.1)
fig.savefig(f'{OUT}/results.pdf');plt.close(fig)

for stale in ('fidelity.pdf','fairness.pdf','privacy.pdf'):
    p=os.path.join(OUT,stale)
    if os.path.exists(p): os.remove(p)
print('ok')


# ============ figure 2: two ports, two potencies ============
# The paper's central claim is that porting the cut is not one mechanism applied
# nine times but two different mechanisms, and that the difference is visible in
# how much disparity each can actually remove.  Panel (a) is the structural
# reason; panel (b) is the consequence.
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def _box(ax,x,y,w,h,text,fc,ec,fs=6.3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.010,rounding_size=0.018',
                 linewidth=0.6,facecolor=fc,edgecolor=ec,zorder=3))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,
            zorder=4,linespacing=1.3)

def _arrow(ax,x0,y,x1):
    ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle='-|>',mutation_scale=6,
                 linewidth=0.7,color=MUT,zorder=3,shrinkA=0,shrinkB=0))

fig=plt.figure(figsize=(5.5,3.15))
gs=fig.add_gridspec(2,1,height_ratios=[1.0,0.90],hspace=0.30)
axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[1,0])

axa.set_xlim(0,1); axa.set_ylim(0,1.14); axa.axis('off')
TINT={'gan':'#fdeee7','diff':'#e6f6f0','marginal':'#e8f0fb'}
W,H=0.203,0.285
xs=[0.055,0.300,0.545]
axa.text(0.0,1.135,'(a) The same fairness definition, two different operations',
         fontsize=7.2,color=INK,va='top')
# --- top lane: DECAF-style families (a DAG is supplied) ---
yT=0.560
axa.text(0.0,yT+H+0.045,'Causal GAN / causal diffusion: the graph is given',
         fontsize=6.5,color=INK,weight='bold',va='bottom')
_box(axa,xs[0],yT,W,H,'expert-asserted\ncausal DAG',TINT['diff'],AQUA)
_box(axa,xs[1],yT,W,H,'train generator\nfactored on it',TINT['diff'],AQUA)
_box(axa,xs[2],yT,W,H,'cut edge when\nsampling',TINT['diff'],AQUA)
for i in range(2): _arrow(axa,xs[i]+W,yT+H/2,xs[i+1])
axa.text(xs[2]+W+0.020,yT+H/2,'after training:\npure post-processing',fontsize=6.0,
         color=INK2,va='center',linespacing=1.35)
# --- bottom lane: marginals-based families (no DAG exists) ---
yB=0.055
axa.text(0.0,yB+H+0.045,'Marginals-based: the graph is bought from the privacy budget',
         fontsize=6.5,color=INK,weight='bold',va='bottom')
_box(axa,xs[0],yB,W,H,'no DAG\nsupplied',TINT['marginal'],BLUE)
_box(axa,xs[1],yB,W,H,'private search\npicks a structure',TINT['marginal'],BLUE)
_box(axa,xs[2],yB,W,H,'drop candidate\nbefore measuring',TINT['marginal'],BLUE)
for i in range(2): _arrow(axa,xs[i]+W,yB+H/2,xs[i+1])
axa.text(xs[2]+W+0.020,yB+H/2,'inside the search:\nbinds a graph noise chose',fontsize=6.0,
         color=INK2,va='center',linespacing=1.35)

# --- panel (b): how much of the parity gap the cut actually removes ---
SF={'diff':(89,43),'gan':(75,32),'marginal':(29,22)}
order=['diff','gan','marginal']
x=np.arange(len(order))
for j,(ds,alpha,hatch) in enumerate([('Adult',1.0,None),('COMPAS',0.5,'////')]):
    off=-0.185 if j==0 else 0.185
    vals=[SF[f][j] for f in order]
    axb.bar(x+off,vals,width=0.33,color=[FAMC[f] for f in order],alpha=alpha,
            hatch=hatch,edgecolor='white',linewidth=0.5,zorder=3)
    for xi,v in zip(x+off,vals):
        axb.text(xi,v+2.5,f'{v}%',ha='center',fontsize=6.3,color=INK2,zorder=4)
axb.set_xticks(x); axb.set_xticklabels([FAMN[f] for f in order])
axb.set_ylim(0,108); axb.set_yticks([0,25,50,75,100])
axb.set_ylabel('% of parity gap removed',color=INK2,labelpad=2)
axb.grid(axis='y',zorder=0); axb.set_axisbelow(True); axb.tick_params(colors=INK2)
axb.set_title('(b) What the cut actually removes, where there is disparity to remove',
              fontsize=7.2,color=INK,loc='left',pad=5)
h=[plt.Rectangle((0,0),1,1,fc='#9d9b94',ec='white',lw=0.45),
   plt.Rectangle((0,0),1,1,fc='#9d9b94',ec='white',lw=0.45,alpha=0.5,hatch='////')]
axb.legend(h,['Adult','COMPAS'],loc='upper right',handlelength=1.05,
           handletextpad=0.45,borderpad=0.2,ncol=2,columnspacing=1.0)
fig.savefig(f'{OUT}/porting.pdf'); plt.close(fig)
print('wrote figs/porting.pdf')


# ============ figure: the three definitions as three edge sets ============
# One small DAG drawn three times.  Solid = edge the generator may use;
# struck-through red = edge the mechanism severs.  CF spares the admissible
# route S -> A -> Y, which is the whole point of the definition.
NODES={'S':(0.06,0.50),'A':(0.48,0.86),'Z':(0.48,0.14),'Y':(0.90,0.50)}
EDGES=[('S','A'),('S','Z'),('S','Y'),('A','Y'),('Z','Y')]
CUTS={'FTU':{('S','Y')},
      'DemPar':{('S','Y'),('S','A'),('S','Z')},
      'CF':{('S','Y'),('S','Z')}}
SUB={'FTU':'cuts the direct edge','DemPar':'cuts every $S\\rightsquigarrow Y$ path',
     'CF':'spares the admissible route'}
NCOL={'S':'#f6d9cc','A':'#d6ead9','Z':'#eceae4','Y':'#d3e2f7'}
NEC={'S':ORANGE,'A':AQUA,'Z':MUT,'Y':BLUE}
RED='#cf2f2f'
fig,axes=plt.subplots(1,3,figsize=(5.5,1.42))
for ax,mech in zip(axes,['FTU','CF','DemPar']):
    ax.set_xlim(-0.20,1.16); ax.set_ylim(-0.10,1.10); ax.axis('off')
    for u,v in EDGES:
        cut=(u,v) in CUTS[mech]
        x0,y0=NODES[u]; x1,y1=NODES[v]
        ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=6.5,
            lw=1.0 if cut else 0.8,color=RED if cut else INK2,zorder=2,
            shrinkA=8.5,shrinkB=8.5,alpha=0.95 if cut else 0.85,
            linestyle=(0,(2.2,1.4)) if cut else '-'))
        if cut:  # strike-through marker at the midpoint, rotated ~8 deg CCW
                 # from the previous (0.045, 0.055) baseline so the mark is
                 # more visibly non-parallel to diagonal edges (e.g. S->A).
            mx,my=(x0+x1)/2,(y0+y1)/2
            ax.plot([mx-0.037,mx+0.037],[my-0.061,my+0.061],color=RED,lw=1.2,zorder=5)
    for n,(x,y) in NODES.items():
        ax.add_patch(plt.Circle((x,y),0.085,facecolor=NCOL[n],edgecolor=NEC[n],
                                lw=0.8,zorder=3))
        ax.text(x,y,n,ha='center',va='center',fontsize=7.0,color=INK,zorder=4)
    ax.set_title(f'\\textbf{{{mech}}}' if False else mech,fontsize=7.6,color=INK,pad=1.5)
    ax.text(0.48,-0.085,SUB[mech],ha='center',fontsize=6.2,color=INK2)
fig.savefig(f'{OUT}/pruning.pdf'); plt.close(fig)
print('wrote figs/pruning.pdf')
