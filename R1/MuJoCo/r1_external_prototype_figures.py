"""Combine preserved MuJoCo evidence with the two prototype ports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

ROOT=Path(__file__).resolve().parent
P=ROOT.parents[1]/'paper/IoTJ_FedUMPC/R1'
OUT=ROOT/'outputs_r1_external_prototypes'
DEST=P/'figures/Experiment/R1_ExternalExpanded'
ORDER=['Linear MPC','Tube MPC','Stochastic MPC','GP-MPC','Robust MPC','AEKF prototype','Static-CBF prototype','FedUMPC']
NAMES={'Linear MPC':'Nominal MPC','Tube MPC':'Tube-style MPC','Stochastic MPC':'Scenario MPC','Robust MPC':'Fixed-margin MPC'}
COLORS=['#FF6666','#FFAA53','#50CC55','#3399FF']

def main():
 runs=pd.read_csv(OUT/'runs.csv')
 raw=runs[runs.seed!=356796271]
 assert len(raw)==24 and raw.groupby('method').size().eq(12).all()
 historical=P/'data/Experiment/Final/mujoco_table_summary.csv'
 table=pd.read_csv(historical) if historical.is_file() else pd.DataFrame()
 summary_input=raw if historical.is_file() else pd.concat([
     pd.read_csv(ROOT/'outputs_r1_screened/mujoco_runs.csv'),raw],ignore_index=True)
 for method,g in summary_input.groupby('method',sort=False):
  assert len(g)==12,(method,len(g))
  row={'method':NAMES.get(method,method),'scenes':12,'source':'MuJoCo 3.11.0; fixed 12 scenes'}
  for key,v in [('SR',100*g.success/4),('SC',100-g.violation_frequency_pct),('CR',100*g.collisions/4),
                ('min_clearance',g.min_clearance),('VD',g.mean_violation_depth),('MD',g.max_violation_duration_s)]:
   if key in ['SR','SC','CR']:mean,sd=key+'_mean_pct',key+'_sample_sd_pct'
   elif key=='min_clearance':mean,sd='min_clearance_mean_m','min_clearance_sample_sd_m'
   else:mean,sd=key+'_mean_'+('m' if key=='VD' else 's'),key+'_sample_sd_'+('m' if key=='VD' else 's')
   row[mean]=v.mean();row[sd]=v.std(ddof=1)
  table=pd.concat([table,pd.DataFrame([row])],ignore_index=True)
 table.to_csv(OUT/'eight_method_table.csv',index=False)
 plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':8.5,'axes.titlesize':8.5,'legend.fontsize':8.5,'pdf.fonttype':42})
 scene=ROOT/'outputs_r1_screened/scene_median_development'
 scenario=json.loads((scene/'scenario.json').read_text())
 records={r['method']:r for r in json.loads((scene/'metrics.json').read_text())}
 for method in ORDER[5:7]:records[method]=json.loads((OUT/'scene'/(method.replace(' ','_')+'_metrics.json')).read_text())
 fig,axes=plt.subplots(2,4,figsize=(6.08,3.9))
 for i,(ax,method) in enumerate(zip(axes.ravel(),ORDER)):
  for o in scenario['obstacles']:ax.add_patch(Circle((o['x'],o['y']),o['r'],facecolor='#C9CDD1',edgecolor='#5C6268',lw=.5))
  safe=method.lower().replace(' ','_').replace('-','_')
  folder=OUT/'scene' if method in ORDER[5:7] else scene
  trajs=json.loads((folder/(safe+'_trajectories.json')).read_text())
  for j in range(4):
   t=np.asarray(trajs[f'agent_{j+1}']);ax.plot(t[:,0],t[:,1],color=COLORS[j],lw=.9)
   ax.scatter(*t[0,:2],s=9,color=COLORS[j]);ax.scatter(*np.asarray(scenario['targets'])[j,:2],s=10,marker='D',facecolors='none',edgecolors=COLORS[j],lw=.7)
   # Saved success flag is aggregate; termination marker follows the original geometry criterion.
   goal=np.asarray(scenario['targets'])[j];start=np.asarray(scenario['starts'])[j]
   route=goal[:2]-start[:2];progress=np.dot(t[-1,:2]-start[:2],route)/np.dot(route,route)
   success=np.linalg.norm(t[-1,:2]-goal[:2])<=3.2 or (progress>=.91 and np.linalg.norm(t[-1,:2]-goal[:2])<=5.0)
   if not success:ax.scatter(*t[-1,:2],marker='x',s=14,color='#202020',lw=.8)
  row=records[method]
  ax.text(.025,.025,f"S {int(row['success'])}/4 | C {int(row['collisions'])}/4\n"+f"$d_{{min}}$ {row['min_clearance']:.2f} m",transform=ax.transAxes,fontsize=8.5,bbox={'facecolor':'white','edgecolor':'#777777','alpha':.92,'pad':1.5})
  ax.set_title(f'({chr(97+i)}) {NAMES.get(method,method)}',loc='left',pad=3,fontweight='bold')
  ax.set_xlim(-22,22);ax.set_ylim(-22,22);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
 axes[0,0].plot([9,19],[18,18],color='#333333',lw=1);axes[0,0].text(14,14.5,'10 m',ha='center',va='top')
 handles=[Line2D([],[],color=c,label=f'Route {i+1}') for i,c in enumerate(COLORS)]
 handles.append(Line2D([],[],marker='x',color='#202020',ls='none',label='Terminated'))
 fig.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,.005),frameon=False)
 fig.subplots_adjust(left=.015,right=.995,top=.94,bottom=.11,hspace=.20,wspace=.012)
 # Preserve the author's TeX trim while keeping the outer titles/spines intact.
 canvas=Bbox.from_extents(-8/72,-3/72,fig.get_figwidth()+8/72,fig.get_figheight())
 DEST.mkdir(parents=True,exist_ok=True);fig.savefig(DEST/'fig4_mujoco_trajectories.pdf',bbox_inches=canvas,pad_inches=0);plt.close(fig)
 print(table.to_string(index=False))
if __name__=='__main__':main()
