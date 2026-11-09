"""Summarize every new learning setting and render Figure 7 from its records."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import json,inspect
import pandas as pd
import numpy as np
import r1_final_figures as f
import r1_layout_figures_20261109 as layout

ROOT=R1_ROOT
D=artifact_path('results_r1_federation_validated_20261109')
P=ROOT.parents[1]/'paper/IoTJ_FedUMPC/R1'
OUT=P/'figures/Experiment/R1_Federation20261109'

def main():
 raw=pd.read_csv(D/'fleet_all_settings.csv')
 fields=['Mean_MSE','Sparse_client_MSE','Final_drift','Empirical_B']
 summary=raw.groupby(['N','Method'],sort=False)[fields].agg(['mean','std'])
 summary.columns=['_'.join(c) for c in summary.columns]
 summary=summary.reset_index();summary.to_csv(D/'fleet_summary_local_reference.csv',index=False)
 supplied_path=D/'author_fedavg_repeat.csv'
 supplied=pd.read_csv(supplied_path).set_index('N') if supplied_path.exists() else None
 if supplied is not None:assert list(supplied.index)==[4,16,32]
 # Preserve the raw-derived local summary. The reported view carries explicit
 # per-field provenance rather than fabricating external per-seed records.
 reported=summary.copy()
 reported['Data_poor_source']='local three-replication evaluation'
 reported['Empirical_B_source']='local three-replication evaluation'
 for n in ([] if supplied is None else [4,16,32]):
  mask=(reported.N==n)&(reported.Method=='FedAvg')
  reported.loc[mask,'Sparse_client_MSE_mean']=supplied.loc[n,'Data_poor_NMSE']
  reported.loc[mask,'Sparse_client_MSE_std']=supplied.loc[n,'Data_poor_NMSE_SD']
  reported.loc[mask,'Empirical_B_mean']=supplied.loc[n,'Empirical_B']
  reported.loc[mask,'Empirical_B_std']=supplied.loc[n,'Empirical_B_SD']
  reported.loc[mask,['Data_poor_source','Empirical_B_source']]='same-configuration updated aggregates; original sample SD confirmed by author'
 reported.to_csv(D/'fleet_summary.csv',index=False)
 reported.to_csv(D/'fleet_summary_reported.csv',index=False)
 comm=pd.read_csv(D/'r1_communication_stress.csv').groupby('Scenario',sort=False)[['Mean_MSE','Sparse_client_MSE','Round_traffic_MB']].agg(['mean','std'])
 comm.columns=['_'.join(c) for c in comm.columns];comm.to_csv(D/'communication_summary.csv')
 paired=[]
 for n in [4,16,32]:
  ref=raw[(raw.N==n)&(raw.Method=='Prox-FL')].set_index('Seed')
  for comparator in ['FedAvg','Prox-FL mu=0.1','Local Only','Centralized']:
   other=raw[(raw.N==n)&(raw.Method==comparator)].set_index('Seed').loc[ref.index]
   row={'N':n,'Comparator':comparator}
   for m in ['Mean_MSE','Sparse_client_MSE']:
    values=(ref[m]-other[m]).to_numpy();rng=np.random.default_rng(20261109)
    boot=values[rng.integers(0,3,(10000,3))].mean(axis=1)
    row.update({m+'_delta':values.mean(),m+'_lo':np.percentile(boot,2.5),m+'_hi':np.percentile(boot,97.5)})
   paired.append(row)
 paired_frame=pd.DataFrame(paired)
 paired_frame.to_csv(D/'paired_differences_local_reference.csv',index=False)
 # New marginal means and SDs do not determine the covariance of seed-paired
 # differences. Keep original paired records separately; omit an unavailable
 # updated FedAvg data-poor interval instead of reusing the old one.
 if supplied is not None:
  mask=paired_frame.Comparator=='FedAvg'
  paired_frame.loc[mask,['Sparse_client_MSE_delta','Sparse_client_MSE_lo','Sparse_client_MSE_hi']]=np.nan
 paired_frame.to_csv(D/'paired_differences.csv',index=False)
 selection=json.loads((D/'selection.json').read_text())
 report={'selected_mu':selection['mu'],'evaluation_seed_count':3,'evaluated_fleet_rows':len(raw),
         'communication_rows':len(pd.read_csv(D/'r1_communication_stress.csv')),
         'old_closed_loop_weights_replaced':False,'fedavg_uniformly_outperformed':False,
         'figure7_a_c_fedavg_source':('same-configuration updated means; original sample SD confirmed by author; updated paired data-poor inference unavailable' if supplied is not None else 'new per-replication results'),
         'figure7_b_fedavg_source':'original local per-replication records',
         'development_fedavg_reference':'original local matched development runs; unchanged'}
 (D/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 # Numerical table is generated from all five settings, including the original
 # coefficient on the same new evaluation seeds; no unfavorable rows are dropped.
 table=r'''\begin{table}[!ht]\centering\scriptsize
\caption{Learning-only evaluation: mean (sample SD). Selected Prox-FL uses $\mu=10^{-4}$. Dashes denote unavailable diagnostics. Closed-loop models remain unchanged.}
\label{tab:supp_fleet_validation}\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}\toprule
$N$&Method&Fleet NMSE&Data-poor NMSE&Drift&$B_{\mathrm{emp}}$\\\midrule
'''
 order=['Local Only','FedAvg','Prox-FL','Prox-FL mu=0.1','Centralized']
 for n in [4,16,32]:
  for j,m in enumerate(order):
   row=reported[(reported.N==n)&(reported.Method==m)].iloc[0]
   cells=[]
   for k,d in zip(fields,[5,5,3,3]):
    cells.append('--' if pd.isna(row[k+'_mean']) else f'{row[k+"_mean"]:.{d}f} ({row[k+"_std"]:.{d}f})')
   name={'Prox-FL':'Prox-FL, selected','Prox-FL mu=0.1':r'Prox-FL, $\mu=0.10$'}.get(m,m)
   table+=(str(n) if j==0 else '')+'&'+name+'&'+'&'.join(cells)+r'\\'+'\n'
  if n!=32:table+=r'\midrule'+'\n'
 table+=r'\bottomrule\end{tabular}\end{table}'+'\n'
 scores=pd.read_csv(D/'validation_scores.csv').sort_values('Mu')
 dev=pd.read_csv(D/'development.csv');baseline=dev[dev.Mu==0]
 table+=r'''\begin{table}[!ht]\centering\scriptsize
\caption{Coefficient selection at $N=16$ on two development seeds. The score in S8 is normalized to matched FedAvg ($\mu=0$).}
\label{tab:supp_mu_selection}\setlength{\tabcolsep}{5pt}
\begin{tabular}{rrrrr}\toprule
$\mu$&Validation score&Fleet NMSE&Data-poor NMSE&Drift\\\midrule
'''
 table+=f'0 & 1.00000 & {baseline.Mean_MSE.mean():.5f} & {baseline.Sparse_client_MSE.mean():.5f} & {baseline.Final_drift.mean():.3f}'+r'\\'+'\n'
 for _,row in scores.iterrows():
  table+=' & '.join([f'{row.Mu:g}',f'{row.Validation_score:.5f}',f'{row.Fleet_NMSE:.5f}',f'{row.Sparse_NMSE:.5f}',f'{row.Drift:.3f}'])+r'\\'+'\n'
 table+=r'\bottomrule\end{tabular}\end{table}'+'\n'
 (P/'sections/fleet_validation_tables_20261109.tex').write_text(table,encoding='utf-8')
 OUT.mkdir(exist_ok=True);layout.DEST=OUT;f.OUT=OUT;f.AUX=D;f.setup_style();f.save=layout.save_compact
 f.SUPPLIED_FEDAVG=supplied.to_dict(orient='index') if supplied is not None else None
 layout.execute(f.federation_figure,[('figsize=(3.5, 4.5)','figsize=(3.5, 4.1)'),('hspace=0.65','hspace=0.50'),
    ('grouped = frame.groupby("N").Sparse_client_MSE.agg(["mean", "std"])',
     'grouped = frame.groupby("N").Sparse_client_MSE.agg(["mean", "std"])\n        if method=="FedAvg" and SUPPLIED_FEDAVG is not None:\n            for n in [4,16,32]:\n                grouped.loc[n,"mean"]=SUPPLIED_FEDAVG[n]["Data_poor_NMSE"]\n                grouped.loc[n,"std"]=SUPPLIED_FEDAVG[n]["Data_poor_NMSE_SD"]'),
    ('image = ax.imshow(matrix, cmap=POS_CMAP',
     'if SUPPLIED_FEDAVG is not None:\n        for j,n in enumerate([4,16,32]):\n            matrix[methods.index("FedAvg"),j]=SUPPLIED_FEDAVG[n]["Empirical_B"]\n    image = ax.imshow(matrix, cmap=POS_CMAP'),
    ('marker="o", capsize=2','marker=("s" if method=="Prox-FL" else "o"), capsize=2'),
    ('marker="o", label=m','marker=("s" if m=="Prox-FL" else "o"), label=m')])()
 print(summary.to_string(index=False),flush=True)
 print('FLEET_FINALIZATION_COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
