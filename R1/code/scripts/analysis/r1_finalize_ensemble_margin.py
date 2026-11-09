
from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=R1_ROOT
sys.path.insert(0,str(ROOT))
import r1_experiments as r1
from r1_ensemble_margin import EnsembleMarginMPC
from models import EnsemblePredictor
from config import Config
from utils import CrossingEnv
OUT=artifact_path('results_r1_ensemble_margin')
rows=[]; contrasts=[]
metrics=['SR','SC','CR','CBS','MinClearance','MeanViolationDepth','MaxViolationDuration']
for regime in ['common','mismatch']:
    new=pd.read_csv(OUT/f'ensemble_margin_{regime}_runs.csv')
    old=pd.read_csv(artifact_path(f'results_r1_final_screened/guarded_{regime}_runs.csv'))
    old=old[old.Method=='FedUMPC'].copy()
    assert len(new)==len(old)==10
    for f in [new,old]:
        sr=100*f.SR
        f['CBS']=2*sr*f.SC/(sr+f.SC)*(1-f.CR)
    for f in [new,old]:
        row={'Regime':regime,'Method':f.Method.iloc[0]}
        for m in metrics:
            scale=100 if m in ['SR','CR'] else 1
            row[m+'_mean']=f[m].mean()*scale
            row[m+'_sd']=f[m].std(ddof=1)*scale
        rows.append(row)
    a=old.set_index('Seed'); b=new.set_index('Seed')
    assert set(a.index)==set(b.index)
    row={'Regime':regime,'Reference':'FedUMPC','Comparator':'Ensemble-margin MPC'}
    for m in metrics:
        scale=100 if m in ['SR','CR'] else 1
        d=(a[m]-b.loc[a.index,m]).to_numpy()*scale
        lo,hi=r1.cluster_bootstrap(d)
        row[m+'_difference']=float(d.mean()); row[m+'_lo']=lo; row[m+'_hi']=hi
    contrasts.append(row)
pd.DataFrame(rows).to_csv(OUT/'comparison_summary.csv',index=False)
pd.DataFrame(contrasts).to_csv(OUT/'paired_scene_contrasts.csv',index=False)
# Smoke check: explicit margin changes when member disagreement changes.
Config.HORIZON=25
class Dummy:
    def __init__(self,v):self.v=v
    def predict_horizon(self,s,u):return np.zeros_like(s),np.full(len(s),self.v)
c=EnsembleMarginMPC(CrossingEnv(seed=1158625181,obstacle_density=3),Dummy(0))
state=np.array([-12.,-.5,0.,0.]); target=np.array([20.,2.,3.,0.])
g=np.zeros(50); base=c._cost(g,state,target)
c.ensemble=Dummy(.25); changed=c._cost(g,state,target)
assert changed>base,(base,changed)
(OUT/'margin_function_check.json').write_text(json.dumps({'zero_disagreement_score':float(base),'positive_disagreement_score':float(changed),'margin_response_confirmed':True},indent=2))
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False))
