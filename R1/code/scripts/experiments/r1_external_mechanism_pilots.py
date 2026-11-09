"""Exploratory mechanism adaptations, NOT reproductions of the two papers.

Full original equations/author code have not been obtained. Do not publish
these outputs under UA Safe Planner or Mismatch-aware RMPC method names.
All outcomes are retained, with no outcome-dependent parameter selection.
"""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import functools
import hashlib
import json
import platform
import numpy as np
import r1_experiments as r1
from config import Config
from controllers import LinearMPC

ROOT=R1_ROOT
OUT=artifact_path('results_r1_external_mechanism_pilots')
FROZEN=json.loads((artifact_path('results_r1_final_screened/protocol.json')).read_text())

class AEKFCompensatedPilot(LinearMPC):
    """Augmented additive-bias EKF with adaptive diagonal observation noise.

    Uses the executed full-state observation at the next action call, never
    true mass/drag. This is an offset-free nonlinear candidate MPC adaptation,
    NOT Zhao et al.'s parameter-varying path-following MPC implementation.
    """
    scale=np.array([1.6,1.6,.45,.25])

    def __init__(self,env):
        super().__init__(env)
        self.z=None
        self.last_state=None
        self.last_action=None
        self.P=np.diag(np.r_[np.ones(4)*.01,np.ones(4)*.01])
        self.Q=np.diag(np.r_[np.ones(4)*1e-6,np.ones(4)*1e-5])
        self.R=np.eye(4)*1e-4
        self.updates=0

    def _transition(self,z,u):
        state=z[:4]*self.scale
        pred=self.model.step(state,u)/self.scale+z[4:]
        return np.r_[pred,z[4:]]

    def get_action(self,state,target):
        if self.z is None:
            self.z=np.r_[state/self.scale,np.zeros(4)]
        else:
            prior=self._transition(self.z,self.last_action)
            A=np.zeros((8,8)); eps=1e-5
            for j in range(8):
                dx=np.zeros(8);dx[j]=eps
                A[:,j]=(self._transition(self.z+dx,self.last_action)-self._transition(self.z-dx,self.last_action))/(2*eps)
            P=A@self.P@A.T+self.Q
            innovation=state/self.scale-prior[:4]
            S=P[:4,:4]+self.R
            gain=np.linalg.solve(S,P[:4,:]).T
            self.z=prior+gain@innovation
            H=np.c_[np.eye(4),np.zeros((4,4))]
            IK=np.eye(8)-gain@H
            self.P=IK@P@IK.T+gain@self.R@gain.T
            # Positive diagonal projection of an innovation covariance estimate.
            estimate=np.diag(np.outer(innovation,innovation)-P[:4,:4])
            diag=np.clip(.98*np.diag(self.R)+.02*estimate,1e-8,.1)
            self.R=np.diag(diag)
            self.updates+=1
        action=super().get_action(state,target)
        self.last_action=action.copy();self.last_state=state.copy()
        return action

    def _rollout(self,u,state):
        current=state.copy();out=[]
        bias=np.zeros(4) if self.z is None else self.z[4:]*self.scale
        for a in u:
            current=self.model.step(current,a)+bias
            current[2]=max(0.,current[2]);out.append(current.copy())
        return np.asarray(out)

class StaticReachabilityCBFPilot(LinearMPC):
    """Finite-candidate discrete CBF filter with exact static obstacle sets.

    Static obstacles have zero motion/prediction covariance. This does not
    reproduce Guo et al.'s probabilistic moving-target reachability model,
    handling-stability constraints, or constrained continuous optimizer.
    """
    gamma=.2

    def _solve(self,state,target):
        guesses=self._initial_guesses(state,target)
        centers=np.array([[o['x'],o['y']] for o in self.env.obstacles])
        radii=np.array([o['r'] for o in self.env.obstacles])+Config.SAFETY_MARGIN
        scored=[]
        for g in guesses:
            u=g.reshape(self.H,2);states=self._rollout(u,state)
            xy=np.vstack([state[:2],states[:,:2]])
            barrier=np.sum((xy[:,None,:]-centers[None,:,:])**2,axis=2)-radii[None,:]**2
            residual=barrier[1:]-(1-self.gamma)*barrier[:-1]
            deficit=float(np.maximum(-residual/(2*radii[None,:]),0).max())
            cost=self._base_cost(states,u,target)+sum(self._obstacle_cost(s,Config.SAFETY_MARGIN) for s in states)
            scored.append((deficit,cost,g))
        feasible=[row for row in scored if row[0]<=1e-9]
        selected=min(feasible,key=lambda r:r[1]) if feasible else min(scored,key=lambda r:(r[0],r[1]))
        u_seq=np.clip(selected[2],self._lo,self._hi)
        self.u_prev=np.clip(np.roll(u_seq,-2),self._lo,self._hi)
        self.u_prev[-2:]=u_seq[-2:]
        return u_seq[:2]

def main():
    OUT.mkdir(exist_ok=True)
    for name,expected in FROZEN['frozen_source_sha256'].items():
        assert hashlib.sha256((source_file(name)).read_bytes()).hexdigest()==expected
    protocol={'status':'exploratory mechanism adaptations, not original-paper reproductions; not manuscript evidence',
        'original_papers':{'UA Safe Planner':'10.1109/TITS.2025.3554836','Mismatch-aware RMPC':'10.1016/j.conengprac.2024.106126'},
        'full_method_or_author_code_available':False,
        'seeds':FROZEN['heldout_seeds'],'H':25,'steps':200,
        'candidate_budget':'same 11/14 candidate generator; no continuous optimization',
        'aekf':{'state':'normalized state and additive increment bias (8D)',
                'scale':[1.6,1.6,.45,.25],'P0_diagonal':.01,
                'Q_state':1e-6,'Q_bias':1e-5,'R0':1e-4,
                'R_update_rate':.02,'R_diagonal_bounds':[1e-8,.1],
                'measurements':'observed executed state at next control call; no true mass/drag access'},
        'cbf':{'gamma':.2,'clearance_target_m':1.35,
               'target_reachable_set':'known static obstacle singleton; zero motion covariance',
               'condition':'h_next >= (1-gamma) h_previous at each predicted step',
               'infeasible_fallback':'minimum worst normalized CBF deficit; objective tie-break'},
        'parameter_selection':'declared before running pilots; no outcome-dependent tuning',
        'inclusion_rule':'retain all results; no original-method performance claim without full-method fidelity',
        'frozen_source_sha256':FROZEN['frozen_source_sha256'],
        'pilot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment':{'python':platform.python_version(),'numpy':np.__version__}}
    path=OUT/'protocol.json'
    if path.exists():
        check_protocol(json.loads(path.read_text()),protocol)
    else:path.write_text(json.dumps(protocol,indent=2))
    Config.HORIZON=25;Config.SIM_STEPS=200;Config.USE_SLSQP_REFINEMENT=False
    Config.RESULTS_DIR=str(OUT);r1.OUT=OUT
    r1.run_simulation=functools.partial(r1.run_simulation,strict=True)
    methods={'AEKF-compensated pilot':lambda e:[AEKFCompensatedPilot(e) for _ in range(4)],
             'Static reachability-CBF pilot':lambda e:[StaticReachabilityCBFPilot(e) for _ in range(4)]}
    for regime,kw in [('common',{}),('mismatch',{'vehicle_masses':[650.,850.,1150.,1350.],'vehicle_drags':[.03,.05,.08,.11]})]:
        runs,_=r1.run_panel(regime,methods,protocol['seeds'],**kw)
        r1.summarize_runs(runs,regime)
    print('PILOT_COMPLETE',flush=True)

if __name__=='__main__':main()
