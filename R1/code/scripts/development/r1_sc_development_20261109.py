"""Prospective SC-oriented development pilot; never overwrites paper endpoints.

Three previously designated development seeds, two predeclared conditions,
and two longer screening horizons versus the frozen controller. The only
changes are screening horizon and near-cost slack. No holdout outcome is read.
Do not insert this pilot as a replacement for the frozen manuscript evidence.
"""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import hashlib,json,functools
import pandas as pd
import r1_experiments as r1
from config import Config
from controllers import FedUMPCController

ROOT=R1_ROOT
OUT=artifact_path('results_r1_sc_development_20261109')
SEEDS=[11,23,37]
GRID=[('Frozen',10,.002),('Screen18_slack002',18,.02),('Screen25_slack002',25,.02)]
CONDITIONS={'Nominal':(0.,0.),'Compound':(.25,1.)}

def main():
 OUT.mkdir(exist_ok=True)
 original=json.loads((artifact_path('results_r1_final_screened/protocol.json')).read_text())
 assert not set(SEEDS)&set(original['heldout_seeds'])
 for name,h in original['frozen_source_sha256'].items():
  assert hashlib.sha256((source_file(name)).read_bytes()).hexdigest()==h
 protocol={'date':'2026-11-09','seeds':SEEDS,'conditions':CONDITIONS,'grid':GRID,
           'purpose':'SC-oriented development only, not confirmatory evaluation',
           'heldout_outcome_files_read':False,'same_candidate_generator':True,
           'selection_rule':'consider SC gains only if SR does not decrease and CR does not increase in either condition; a selected design requires a new prospective independent evaluation',
           'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'frozen_source_sha256':original['frozen_source_sha256']}
 path=OUT/'protocol.json'
 if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(protocol))
 else:path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 Config.HORIZON=25;Config.SIM_STEPS=200;Config.MC_SAMPLES=10
 Config.USE_SLSQP_REFINEMENT=False;Config.RESULTS_DIR=str(OUT);r1.OUT=OUT
 r1.run_simulation=functools.partial(r1.run_simulation,strict=True)
 model=r1.load_model(artifact_path('results_r1/models/prox.pt'))
 for condition,(noise,bias) in CONDITIONS.items():
  for name,horizon,slack in GRID:
   def make(env,h=horizon,s=slack):
    result=[FedUMPCController(env,model,screen_horizon=h) for _ in range(4)]
    for controller in result:controller.selection_cost_slack=s
    return result
   prefix=condition.lower()+'_'+name.lower()
   r1.run_panel(prefix,{name:make},SEEDS,noise=noise,disturbance=bias)
 frames=[]
 for condition in CONDITIONS:
  for name,_,_ in GRID:
   f=pd.read_csv(OUT/(condition.lower()+'_'+name.lower()+'_runs.csv'))
   assert len(f)==3
   f['Condition']=condition;frames.append(f)
 runs=pd.concat(frames,ignore_index=True)
 runs['CBS']=2*(100*runs.SR)*runs.SC/(100*runs.SR+runs.SC).clip(lower=1e-12)*(1-runs.CR)
 runs.to_csv(OUT/'all_runs.csv',index=False)
 summary=runs.groupby(['Condition','Method'],sort=False)[['SR','SC','CR','CBS','MinClearance']].mean()
 summary['SR']*=100;summary['CR']*=100
 summary.to_csv(OUT/'summary.csv')
 print(summary.to_string(),flush=True)
 print('SC_DEVELOPMENT_COMPLETE',flush=True)
if __name__=='__main__':main()
