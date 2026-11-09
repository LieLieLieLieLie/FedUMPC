"""Independent numerical reconstruction of the frozen scenario objective."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
import sys,json
from pathlib import Path
import numpy as np
ROOT=R1_ROOT
sys.path.insert(0,str(ROOT))
import r1_experiments as r1
from controllers import StochasticMPC
from config import Config
from utils import CrossingEnv
Config.HORIZON=25
Config.STOCHASTIC_N_SAMPLES=6
c=StochasticMPC(CrossingEnv(seed=1158625181,obstacle_density=3))
s=np.array([-20.,-2.,3.,0.]); target=np.array([20.,2.,3.,0.])
z=np.array([[0,0,0,0],[1,0,1,.5],[-1,0,-1,-.5],[0,1,1,-.5],[0,-1,-1,.5],[1,-1,0,0]],float)
errors=[]; costs=[]
for g in c._initial_guesses(s,target):
    u=g.reshape(25,2); parts=[]
    for row in z:
        x=s.copy(); states=[]
        for h in range(25):
            x=c.model.step(x,u[h])
            x=x+np.sqrt(h+1)*np.array([.05,.05,.30,.02])*row
            states.append(x.copy())
        states=np.array(states)
        parts.append(c._base_cost(states,u,target)+sum(c._obstacle_cost(x,1.35) for x in states))
    actual=c._cost(g,s,target)
    errors.append(abs(actual-np.mean(parts))); costs.append(actual)
assert max(errors)<1e-7
result={'six_vectors':z.tolist(),'weights':[1/6]*6,'candidate_count':len(costs),
 'max_cost_reconstruction_error':max(errors),'all_costs_finite':bool(np.isfinite(costs).all()),
 'units':['m','m','m/s','rad'], 'perturbation_application':'recursive, post-plant, sqrt(h+1) scaling',
 'six_point_mean':z.mean(axis=0).tolist(),
 'interpretation':'controlled deterministic scenario-cost average, not a Gaussian Monte Carlo or chance-constrained implementation',
 'important_limitation':'six-vector subset is not centered; recursive perturbations accumulate along horizon, including through speed and heading. Not a faithful canonical scenario-MPC reproduction.',
 'data_action':'all previous result files retained unchanged'}
out=artifact_path('results_r1_ensemble_margin/scenario_audit.json')
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
