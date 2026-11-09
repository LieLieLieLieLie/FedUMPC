"""Keep all pilot results; never label them as the two published algorithms."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import r1_experiments as r1
from r1_external_mechanism_pilots import AEKFCompensatedPilot,StaticReachabilityCBFPilot
from utils import CrossingEnv
from config import Config

ROOT=R1_ROOT
OUT=artifact_path('results_r1_external_mechanism_pilots')
metrics=['SR','SC','CR','CBS','MinClearance','MeanViolationDepth','MaxViolationDuration']

def smoke_checks():
    Config.HORIZON=25
    env=CrossingEnv(seed=1158625181,obstacle_density=3)
    c=AEKFCompensatedPilot(env)
    c._solve=lambda state,target:np.array([0.,.2])
    state=np.array([-20.,-2.,3.,0.]);target=np.array([20.,-2.,3.,0.])
    bias=np.array([0.,0.,.03,0.])
    for _ in range(100):
        u=c.get_action(state,target)
        state=c.model.step(state,u)+bias
    estimated=c.z[4:]*c.scale
    assert np.linalg.norm(estimated-bias)<.005,(estimated,bias)
    assert np.linalg.eigvalsh(c.P).min()>-1e-10
    assert np.diag(c.R).min()>0
    # A distant obstacle provides feasible constant-state CBF candidates.
    env.obstacles=[{'x':15.,'y':15.,'r':1.}]
    f=StaticReachabilityCBFPilot(env)
    s=np.array([-20.,-2.,3.,0.]);a=f.get_action(s,target)
    sequence=np.r_[a,f.u_prev[:-2]].reshape(25,2)
    trajectory=f._rollout(sequence,s)
    xy=np.vstack([s[:2],trajectory[:,:2]])
    h=np.sum((xy-np.array([15.,15.]))**2,axis=1)-(1+Config.SAFETY_MARGIN)**2
    residual=h[1:]-(1-f.gamma)*h[:-1]
    assert residual.min()>=-1e-7
    return {'synthetic_true_bias':bias.tolist(),'estimated_bias':estimated.tolist(),
            'filter_covariance_positive_semidefinite':True,'observation_covariance_positive':True,
            'feasible_candidate_cbf_min_residual':float(residual.min()),
            'scope':'implementation sanity checks, not original-paper fidelity or realized-safety proofs'}

def main():
    rows=[];contrasts=[]
    for regime in ['common','mismatch']:
        runs=pd.read_csv(OUT/f'{regime}_runs.csv')
        routes=pd.read_csv(OUT/f'{regime}_routes.csv')
        assert len(runs)==20 and len(routes)==80
        assert runs.groupby('Method').size().eq(10).all()
        assert routes.groupby(['Method','Seed']).size().eq(4).all()
        assert np.isfinite(runs.select_dtypes(include='number').to_numpy()).all()
        ref=pd.read_csv(artifact_path(f'results_r1_final_screened/guarded_{regime}_runs.csv'))
        ref=ref[ref.Method=='FedUMPC'].copy()
        ref.to_csv(OUT/f'reference_fedumpc_{regime}_runs.csv',index=False)
        for frame in [runs,ref]:
            sr=100*frame.SR
            frame['CBS']=2*sr*frame.SC/(sr+frame.SC)*(1-frame.CR)
        combined=pd.concat([runs,ref],ignore_index=True)
        for method,f in combined.groupby('Method',sort=False):
            assert len(f)==10
            row={'Regime':regime,'Method':method}
            for m in metrics:
                scale=100 if m in ['SR','CR'] else 1
                row[m+'_mean']=f[m].mean()*scale
                row[m+'_sd']=f[m].std(ddof=1)*scale
            rows.append(row)
            if method=='FedUMPC':continue
            a=ref.set_index('Seed');b=f.set_index('Seed')
            assert set(a.index)==set(b.index)
            row={'Regime':regime,'Reference':'FedUMPC','Comparator':method}
            for m in metrics:
                scale=100 if m in ['SR','CR'] else 1
                diff=(a[m]-b.loc[a.index,m]).to_numpy()*scale
                lo,hi=r1.cluster_bootstrap(diff)
                row[m+'_difference']=float(diff.mean());row[m+'_lo']=lo;row[m+'_hi']=hi
            contrasts.append(row)
    pd.DataFrame(rows).to_csv(OUT/'all_pilot_comparisons.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'all_paired_scene_contrasts.csv',index=False)
    audit=smoke_checks()
    protocol=json.loads((OUT/'protocol.json').read_text())
    verify_source_hash('r1_external_mechanism_pilots.py',protocol['pilot_source_sha256'])
    assert all(hashlib.sha256((source_file(n)).read_bytes()).hexdigest()==h for n,h in protocol['frozen_source_sha256'].items())
    audit['complete_method_scene_records']=40
    audit['complete_route_records']=160
    audit['frozen_source_hashes_preserved']=True
    audit['pilot_source_matches_pre_run_protocol']=True
    audit['input_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*_runs.csv')}
    (OUT/'implementation_checks.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
