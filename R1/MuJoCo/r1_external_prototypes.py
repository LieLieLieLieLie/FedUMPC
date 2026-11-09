"""MuJoCo ports of local mechanism prototypes, NOT source-paper algorithms."""
from pathlib import Path
import csv,json,hashlib
from dataclasses import asdict
import numpy as np
import run_six_algorithms as core

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs_r1_external_prototypes'
BASE=core.SamplingMPC
METHODS=['AEKF prototype','Static-CBF prototype']

class PrototypeMPC(BASE):
 def __init__(self,method,obstacles,bnn):
  super().__init__(method,obstacles,bnn)
  self.scale=np.array([1.6,1.6,.45,.25]);self.z=None
  self.P=np.eye(8)*.01;self.Q=np.diag(np.r_[np.ones(4)*1e-6,np.ones(4)*1e-5]);self.R=np.eye(4)*1e-4

 def transition(self,z,u):return np.r_[core.kinematic_step(z[:4]*self.scale,u)/self.scale+z[4:],z[4:]]

 def update_model(self,state,control,actual):
  if self.method!='AEKF prototype':return
  if self.z is None:self.z=np.r_[state/self.scale,np.zeros(4)]
  prior=self.transition(self.z,control);eps=1e-5;A=np.zeros((8,8))
  for j in range(8):
   dx=np.zeros(8);dx[j]=eps
   A[:,j]=(self.transition(self.z+dx,control)-self.transition(self.z-dx,control))/(2*eps)
  P=A@self.P@A.T+self.Q;innovation=actual/self.scale-prior[:4]
  innovation[3]=core.wrap_angle(innovation[3]*self.scale[3])/self.scale[3]
  K=np.linalg.solve(P[:4,:4]+self.R,P[:4,:]).T
  self.z=prior+K@innovation;H=np.c_[np.eye(4),np.zeros((4,4))];IK=np.eye(8)-K@H
  self.P=IK@P@IK.T+K@self.R@K.T
  self.R=np.diag(np.clip(.98*np.diag(self.R)+.02*(innovation**2-np.diag(P[:4,:4])),1e-8,.1))

 def _rollout(self,state,sequence):
  if self.method!='AEKF prototype':return super()._rollout(state,sequence)
  if self.z is None:self.z=np.r_[state/self.scale,np.zeros(4)]
  current=state.copy();states=[]
  for u in sequence:
   current=core.kinematic_step(current,u)+self.z[4:]*self.scale
   current[2]=max(0.,current[2]);states.append(current.copy())
  return np.asarray(states)

 def _evaluate(self,state,target,sequence):
  states=self._rollout(state,sequence)
  return self._base_cost(states,sequence,target)+self._obstacle_penalty(states,np.full(core.HORIZON,core.SAFETY_MARGIN),soft_weight=75.),None

 def action(self,state,target):
  if self.method=='AEKF prototype':return super().action(state,target)
  if self.route_start is None:self.route_start=state[:2].copy()
  candidates=self._candidates(state,target)
  centers=np.array([[o['x'],o['y']] for o in self.obstacles]);radii=np.array([o['r'] for o in self.obstacles])+core.SAFETY_MARGIN
  rows=[]
  for c in candidates:
   states=self._rollout(state,c);xy=np.vstack([state[:2],states[:,:2]])
   h=np.sum((xy[:,None,:]-centers[None,:,:])**2,axis=2)-radii[None,:]**2
   deficit=float(np.maximum(.8*h[:-1]-h[1:],0).max(axis=0).dot(np.zeros(len(radii)))) if False else float(np.max(np.maximum(.8*h[:-1]-h[1:],0)/(2*radii[None,:])))
   rows.append((deficit,self._evaluate(state,target,c)[0],c))
  feasible=[r for r in rows if r[0]<=1e-9]
  chosen=min(feasible,key=lambda r:r[1]) if feasible else min(rows,key=lambda r:(r[0],r[1]))
  self.prev_u=np.clip(chosen[2][0],[-core.MAX_STEER,-core.MAX_ACCEL],[core.MAX_STEER,core.MAX_ACCEL])
  return self.prev_u.copy()

def main():
 import mujoco
 assert mujoco.__version__=='3.11.0',mujoco.__version__
 OUT.mkdir(exist_ok=True);core.OUTPUT=OUT;core.SamplingMPC=PrototypeMPC
 parent=json.loads((ROOT/'outputs_r1_screened/protocol.json').read_text())
 protocol={'parent_seeds':parent['heldout_seeds'],'methods':METHODS,'mujoco':mujoco.__version__,
  'scope':'physical-engine ports of local mechanism prototypes; not source-paper reproductions',
  'parameters':'AEKF covariance parameters and CBF gamma unchanged from analytical pilots',
  'candidate_horizon':core.HORIZON,'soft_obstacle_weight':75.,'cbf_gamma':.2,
  'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['run_six_algorithms.py','r1_external_prototypes.py']}}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 raw=OUT/'runs.csv';done=set()
 if raw.exists():
  with raw.open(newline='') as f:done={(int(r['seed']),r['method']) for r in csv.DictReader(f)}
 for seed in parent['heldout_seeds']+[356796271]:
  obstacles=core.generate_obstacles(seed);xml=core.make_scene_xml(obstacles)
  for method in METHODS:
   if (seed,method) in done:continue
   print('[MuJoCo prototype]',method,seed,flush=True)
   _,trajs,metric=core.run_method(method,xml,obstacles,None,show_viewer=False)
   row={'seed':seed,**asdict(metric)};exists=raw.exists()
   with raw.open('a',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(row))
    if not exists:w.writeheader()
    w.writerow(row)
   if seed==356796271:
    d=OUT/'scene';d.mkdir(exist_ok=True);core.OUTPUT=d
    core.save_trajectory_data(method,trajs)
    (d/(method.replace(' ','_')+'_metrics.json')).write_text(json.dumps(asdict(metric),indent=2));core.OUTPUT=OUT
 print('MUJOCO_PROTOTYPES_COMPLETE',flush=True)
if __name__=='__main__':main()
