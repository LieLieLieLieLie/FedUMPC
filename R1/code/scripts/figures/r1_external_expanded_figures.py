"""Eight-controller comparison figures; original paper assets stay unchanged."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import argparse,inspect,json,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import r1_final_figures as f
import r1_experiments as r1
from r1_external_robustness import CONDITIONS

ROOT=R1_ROOT
DATA=artifact_path('results_r1_external_expanded')
PAPER=ROOT.parents[1]/'paper/IoTJ_FedUMPC/R1'
OUT=PAPER/'figures/Experiment/R1_ExternalExpanded'
NEW=['AEKF-compensated pilot','Static reachability-CBF pilot']
ORDER=f.SIX[:-1]+NEW+['FedUMPC']
f.SIX=ORDER;f.FINAL=DATA;f.OUT=OUT
f.COLORS.update(dict(zip(NEW,['#00DDDD','#4D4D4D'])))
f.ALIASES.update(dict(zip(NEW,['AEKF','CBF'])))
METRICS=['SR','SC','CR','CBS','MinClearance','MeanViolationDepth','MaxViolationDuration']

def prepare_common():
 DATA.mkdir(exist_ok=True)
 for regime in ['common','mismatch']:
  for kind in ['runs','routes']:
   old=pd.read_csv(artifact_path(f'results_r1_final_screened/guarded_{regime}_{kind}.csv'))
   new=pd.read_csv(artifact_path(f'results_r1_external_mechanism_pilots/{regime}_{kind}.csv'))
   merged=pd.concat([old,new],ignore_index=True)
   assert len(merged)==(80 if kind=='runs' else 320)
   merged.to_csv(DATA/f'guarded_{regime}_{kind}.csv',index=False)
 f.setup_style()
 for fun in [f.common_plant_figure,f.mismatch_figure]:
  source=inspect.getsource(fun).replace('ncol=3','ncol=4').replace('columnspacing=0.9','columnspacing=0.55').replace('columnspacing=0.7','columnspacing=0.55')
  exec(compile(source,'<expanded-figure>','exec'),f.__dict__)
  f.__dict__[fun.__name__]()

def prepare_robustness():
 old=pd.read_csv(artifact_path('results_r1_wind_only/six_condition_runs.csv'))
 new=[]
 for condition in ['Nominal']+list(CONDITIONS):
  for key,method in zip(['aekf','cbf'],NEW):
   if condition=='Nominal':
    frame=pd.read_csv(artifact_path('results_r1_external_mechanism_pilots/common_runs.csv'))
    frame=frame[frame.Method==method].copy()
   else:
    slug=condition.lower().replace(' ','_').replace('.','p')
    frame=pd.read_csv(DATA/f'{key}_{slug}_runs.csv')
   assert len(frame)==10
   frame['Condition']=condition;new.append(frame)
 runs=pd.concat([old,*new],ignore_index=True);runs['CBS']=f.cbs(runs)
 assert len(runs)==480
 rows=[]
 for (condition,method),g in runs.groupby(['Condition','Method'],sort=False):
  row={'Condition':condition,'Method':method}
  for metric in METRICS:
   v=g[metric].to_numpy()*(100 if metric in ['SR','CR'] else 1)
   lo,hi=r1.cluster_bootstrap(v)
   row.update({metric+'_mean':v.mean(),metric+'_sd':v.std(ddof=1),metric+'_lo':lo,metric+'_hi':hi})
  rows.append(row)
 summary=pd.DataFrame(rows)
 runs.to_csv(DATA/'eight_condition_runs.csv',index=False);summary.to_csv(DATA/'eight_condition_summary.csv',index=False)
 wins=[]
 for condition,g in summary.groupby('Condition',sort=False):
  fed=g[g.Method=='FedUMPC'].iloc[0];first=[]
  for m in METRICS:
   target=g[m+'_mean'].min() if m in ['CR','MeanViolationDepth','MaxViolationDuration'] else g[m+'_mean'].max()
   if np.isclose(fed[m+'_mean'],target,atol=1e-8,rtol=0):first.append(m)
  wins.append({'condition':condition,'first_places':first})
 (DATA/'expanded_rank_audit.json').write_text(json.dumps({'total_first_places':sum(len(x['first_places']) for x in wins),'cells':42,'conditions':wins},indent=2))
 robustness_figure(summary,runs)

def robustness_figure(summary,runs):
 f.setup_style();conditions=list(summary.Condition.drop_duplicates());xx=np.arange(6)
 fig,axs=plt.subplots(2,3,figsize=(3.5,5.2))
 for ax,m,label,letter,title in [(axs[0,0],'SR','SR (%) ↑','a','Success'),(axs[0,1],'SC','SC (%) ↑','b','Compliance'),(axs[0,2],'MinClearance','Clr. (m) ↑','c','Clr.')]:
  for method in ORDER:
   g=summary[summary.Method==method].set_index('Condition').loc[conditions]
   ax.plot(xx,g[m+'_mean'],color=f.COLORS[method],marker='o',lw=.9,ms=2.5)
   ax.fill_between(xx,g[m+'_lo'],g[m+'_hi'],color=f.COLORS[method],alpha=.06,lw=0)
  ax.set_ylabel(label,labelpad=1);f.panel(ax,letter,title)
 ax=axs[1,0];width=.09
 for j,method in enumerate(ORDER):
  g=runs[runs.Method==method].groupby('Condition').CR.agg(['mean','std']).loc[conditions]
  ax.bar(xx+(j-3.5)*width,4*g['mean'],width,color=f.COLORS[method],alpha=.75)
 ax.set_ylabel('Contacts (–) ↓',labelpad=1);f.panel(ax,'d','Contacts')
 for ax in [*axs[0],axs[1,0]]:
  ax.set_xticks(xx,['Nom.','N.10','N.25','B.50','Cmp.','Wind'],rotation=65,ha='right')
 ax=axs[1,1];matrix=[]
 for method in ORDER:
  g=summary[summary.Method==method].set_index('Condition').loc[conditions]
  matrix.append([min(g[m+'_mean'].iloc[1:]-g[m+'_mean'].iloc[0]) if m!='CR' else min(g[m+'_mean'].iloc[0]-g[m+'_mean'].iloc[1:]) for m in ['SR','SC','CR']])
 matrix=np.asarray(matrix);v=max(1,np.abs(matrix).max());ax.imshow(matrix,cmap=f.DIV_CMAP,vmin=-v,vmax=v,aspect='auto')
 ax.set_xticks(range(3),['SR','SC','−CR'],rotation=45,ha='right')
 ax.set_yticks(range(8),[f.ALIASES[m] if m!='FedUMPC' else 'Fed.' for m in ORDER],rotation=24)
 for i in range(8):
  for j in range(3):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center')
 f.panel(ax,'e','Worst loss')
 ax=axs[1,2]
 for method in ORDER:
  g=summary[summary.Method==method];avg=.5*(g.SR_mean.mean()+g.SC_mean.mean());worst=np.minimum(g.SR_mean,g.SC_mean).min()
  ax.scatter(avg,worst,s=20,color=f.COLORS[method],edgecolor='white',lw=.4)
 ax.set_xlabel('Utility (%) ↑',labelpad=1);ax.set_ylabel('Worst (%) ↑',labelpad=1);f.panel(ax,'f','Trade-off')
 handles=[mpl.lines.Line2D([],[],color=f.COLORS[m],label=f.ALIASES[m]) for m in ORDER]
 fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.52,.006),frameon=False,columnspacing=.5,handlelength=1.)
 fig.subplots_adjust(left=.15,right=.91,top=.94,bottom=.20,hspace=.70,wspace=1.02)
 fig.savefig(OUT/'fig8_robustness.pdf',format='pdf');plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['common','robust']);a=p.parse_args()
 if os.environ.get('FEDUMPC_ARTIFACTS_ROOT'):
  OUT=Path(os.environ['FEDUMPC_ARTIFACTS_ROOT'])/'figure_data_preview'
 DATA.mkdir(parents=True,exist_ok=True)
 OUT.mkdir(parents=True,exist_ok=True)
 if a.stage=='common':prepare_common()
 else:prepare_robustness()
