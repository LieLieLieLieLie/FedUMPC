"""Extend all frozen disturbance conditions to the two exploratory controls."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import argparse,functools,hashlib,json
import r1_experiments as r1
from config import Config
from r1_external_mechanism_pilots import AEKFCompensatedPilot,StaticReachabilityCBFPilot,FROZEN

ROOT=R1_ROOT
OUT=artifact_path('results_r1_external_expanded')
CONDITIONS={'Process noise 0.10':(.10,0.),'Process noise 0.25':(.25,0.),
 'Actuation bias 0.50':(0.,.5),'Compound stress':(.25,1.),'Wind-only':(0.,1.)}
def main():
 p=argparse.ArgumentParser();p.add_argument('method',choices=['aekf','cbf']);a=p.parse_args()
 OUT.mkdir(exist_ok=True)
 for n,h in FROZEN['frozen_source_sha256'].items():assert hashlib.sha256((source_file(n)).read_bytes()).hexdigest()==h
 Config.HORIZON=25;Config.SIM_STEPS=200;Config.USE_SLSQP_REFINEMENT=False
 Config.RESULTS_DIR=str(OUT);r1.OUT=OUT
 r1.run_simulation=functools.partial(r1.run_simulation,strict=True)
 cls,name=(AEKFCompensatedPilot,'AEKF-compensated pilot') if a.method=='aekf' else (StaticReachabilityCBFPilot,'Static reachability-CBF pilot')
 protocol={'conditions':CONDITIONS,'seeds':FROZEN['heldout_seeds'],
  'method':name,'parameters':'unchanged from original mechanism pilot; no outcome-dependent retuning',
  'pilot_source_sha256':hashlib.sha256((source_file('r1_external_mechanism_pilots.py')).read_bytes()).hexdigest()}
 (OUT/f'{a.method}_protocol.json').write_text(json.dumps(protocol,indent=2))
 for condition,(noise,bias) in CONDITIONS.items():
  slug=condition.lower().replace(' ','_').replace('.','p')
  r1.run_panel(a.method+'_'+slug,{name:lambda e:[cls(e) for _ in range(4)]},FROZEN['heldout_seeds'],noise=noise,disturbance=bias)
 print('ROBUSTNESS_COMPLETE',a.method,flush=True)
if __name__=='__main__':main()
