"""Validation-selected proximal coefficient, followed by a prospectively frozen
learning-only evaluation. All methods/settings/replicates are retained. No
controller weights, historical records, or physical scenarios are overwritten.
"""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import argparse,hashlib,json,platform
import r1_experiments as r
import numpy as np
import pandas as pd
import torch
from config import Config
from federated import FedAvgClient,ProxFLClient

ROOT=R1_ROOT
OUT=artifact_path('results_r1_federation_validated_20261109')
GRID=[0.,.0001,.001,.01,.1]
def seeds(tag,n):
 return [int.from_bytes(hashlib.sha256(f'FedUMPC|fleet-validation|20261109|{tag}|{j}'.encode()).digest()[:4],'big')%(2**31-1) for j in range(n)]
DEV=seeds('development',2);TEST=seeds('evaluation',3)

def frozen_protocol():
 OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
 source={n:hashlib.sha256((source_file(n)).read_bytes()).hexdigest() for n in
         ['config.py','controllers.py','models.py','main.py','federated.py','r1_experiments.py','r1_federation_validation_20261109.py']}
 protocol={'purpose':'learning-only coefficient validation; not a replacement of closed-loop controller weights',
           'development_seeds':DEV,'evaluation_seeds_predeclared_before_development':TEST,
           'development_N':16,'evaluation_N':[4,16,32],'mu_grid':GRID,
           'selection':'minimum mean of fleet-NMSE/FedAvg and sparse-NMSE/FedAvg ratios on two development seeds, over positive mu; ties choose smaller mu',
           'rounds':12,'local_epochs':3,'batch_size':32,'base_LR':.004,'effective_Adam_LR':.0054,
           'centralized_epochs':45,'training':'same original non-IID generator and sample counts; original predictor architecture',
           'evaluation_queries_per_client':180,'device':'CPU','torch_threads':1,
           'communication_conditions':[['Synchronous full',1.,0],['50% participation',.5,0],['50% + two-round stale',.5,2]],
           'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'source_sha256':source}
 path=OUT/'protocol.json'
 if path.exists():check_protocol(json.loads(path.read_text()),protocol)
 else:path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 assert not set(DEV)&set(TEST)
 assert not (set(DEV)|set(TEST))&set(r.FEDERATION_SEEDS)
 return protocol

def settings(mu):
 return r.temporary_config(ROUNDS=12,LOCAL_EPOCHS=3,BATCH_SIZE=32,LR=.004,PROXIMAL_MU=mu,DEVICE='cpu')

def dev():
 path=OUT/'development.csv';done=r.completed_keys(path,['Seed','Mu'])
 for seed in DEV:
  data,holdout,meta=r.generate_fleet_data(16,seed)
  for mu in GRID:
   if (seed,mu) in done:continue
   print('[fleet development]',seed,'mu',mu,flush=True)
   with settings(mu):model,_,drift=r.train_federated(FedAvgClient if mu==0 else ProxFLClient,data,seed=seed,rounds=12)
   detail=r.evaluate_predictors([],{'Model':model},holdout,meta)
   detail.to_csv(OUT/f'development_detail_{seed}_{mu:g}.csv',index=False)
   r.save_model(OUT/'models'/f'dev_{seed}_{mu:g}.pt',model)
   r.append_row(path,{'Seed':seed,'Mu':mu,'Mean_MSE':detail.MSE.mean(),
                     'Sparse_client_MSE':detail[detail.Sparse].MSE.mean(),'Final_drift':drift})
 frame=pd.read_csv(path);assert len(frame)==len(GRID)*len(DEV)
 base=frame[frame.Mu==0].set_index('Seed')
 score=[]
 for mu in GRID[1:]:
  g=frame[frame.Mu==mu].set_index('Seed').loc[base.index]
  objective=.5*(g.Mean_MSE/base.Mean_MSE+g.Sparse_client_MSE/base.Sparse_client_MSE)
  score.append({'Mu':mu,'Validation_score':objective.mean(),'Fleet_NMSE':g.Mean_MSE.mean(),
                'Sparse_NMSE':g.Sparse_client_MSE.mean(),'Drift':g.Final_drift.mean()})
 result=pd.DataFrame(score).sort_values(['Validation_score','Mu'])
 result.to_csv(OUT/'validation_scores.csv',index=False)
 selection={'mu':float(result.iloc[0].Mu),'rule':frozen_protocol()['selection'],
            'development_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'evaluation_results_available_at_selection':(OUT/'fleet_all_settings.csv').exists()}
 dest=OUT/'selection.json'
 if dest.exists():
  recorded=json.loads(dest.read_text())
  # A resumed audit must preserve the original selection chronology, rather
  # than recomputing that historical fact after evaluation files now exist.
  assert all(recorded[k]==selection[k] for k in ['mu','rule','development_csv_sha256'])
  assert recorded['evaluation_results_available_at_selection'] is False
  selection=recorded
 else:dest.write_text(json.dumps(selection,indent=2),encoding='utf-8')
 print(result.to_string(index=False),flush=True)
 print('FROZEN_SELECTION',selection,flush=True)

def evaluate():
 selection=json.loads((OUT/'selection.json').read_text());mu=selection['mu']
 assert not selection['evaluation_results_available_at_selection']
 path=OUT/'fleet_all_settings.csv';done=r.completed_keys(path,['N','Seed','Method'])
 for n in [4,16,32]:
  for seed in TEST:
   data,holdout,meta=r.generate_fleet_data(n,seed)
   for method in ['Local Only','FedAvg','Prox-FL','Prox-FL mu=0.1','Centralized']:
    if (n,seed,method) in done:continue
    print('[fleet independent evaluation]',n,seed,method,flush=True)
    with settings(.1 if method=='Prox-FL mu=0.1' else mu):
     if method=='Local Only':models=r.train_local(data,seed=seed,rounds=12);model=None;drift=np.nan
     elif method=='Centralized':model=r.train_centralized(data,seed=seed,epochs=45);drift=np.nan
     else:model,_,drift=r.train_federated(FedAvgClient if method=='FedAvg' else ProxFLClient,data,seed=seed,rounds=12)
     detail=r.evaluate_predictors(models if model is None else [],{} if model is None else {method:model},holdout,meta)
     b=r.empirical_dissimilarity(model,data) if model is not None else np.nan
    if model is not None:r.save_model(OUT/'models'/f'eval_N{n}_{seed}_{method.replace(" ","_")}.pt',model)
    detail.to_csv(OUT/f'eval_detail_N{n}_{seed}_{method.replace(" ","_")}.csv',index=False)
    r.append_row(path,{'N':n,'Seed':seed,'Method':method,'Mu':.1 if method=='Prox-FL mu=0.1' else (mu if method=='Prox-FL' else 0.),
                      'Mean_MSE':detail.MSE.mean(),'Sparse_client_MSE':detail[detail.Sparse].MSE.mean(),
                      'Rich_client_MSE':detail[~detail.Sparse].MSE.mean(),'Final_drift':drift,'Empirical_B':b,
                      'Model_payload_MB':r.model_bytes(model)/1e6 if model is not None else np.nan})
 fleet=pd.read_csv(path);assert len(fleet)==45
 fleet[fleet.Method!='Prox-FL mu=0.1'].to_csv(OUT/'r1_fleet_scaling.csv',index=False)
 stress=OUT/'r1_communication_stress.csv';done=r.completed_keys(stress,['Scenario','Seed'])
 for seed in TEST:
  data,holdout,meta=r.generate_fleet_data(16,seed)
  for label,participation,staleness in frozen_protocol()['communication_conditions']:
   if (label,seed) in done:continue
   print('[fleet communication]',seed,label,flush=True)
   with settings(mu):model,_,drift=r.train_federated(ProxFLClient,data,seed=seed,rounds=12,participation=participation,staleness=staleness)
   detail=r.evaluate_predictors([],{'Prox-FL':model},holdout,meta)
   detail.to_csv(OUT/f'comm_detail_{seed}_{staleness}_{participation}.csv',index=False)
   r.append_row(stress,{'Scenario':label,'Seed':seed,'Mu':mu,'Mean_MSE':detail.MSE.mean(),
                       'Sparse_client_MSE':detail[detail.Sparse].MSE.mean(),'Final_drift':drift,
                       'Round_traffic_MB':2*16*participation*r.model_bytes(model)/1e6})
 print(fleet.groupby(['N','Method'])[['Mean_MSE','Sparse_client_MSE','Final_drift']].mean().to_string(),flush=True)
 print('FLEET_VALIDATED_COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','evaluation']);a=p.parse_args()
 torch.set_num_threads(1);Config.DEVICE='cpu';frozen_protocol()
 dev() if a.stage=='development' else evaluate()
