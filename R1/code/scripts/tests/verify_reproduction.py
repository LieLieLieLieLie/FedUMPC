"""Read-only source, route-cluster, dependency and repeatability checks."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash

METRICS=['SR','SC','CR','MinClearance','MeanViolationDepth','MaxViolationDuration']
def verify(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    record=json.loads((R1_ROOT/'scripts/relocation_verification.json').read_text())
    for name,digest in record['unchanged_algorithm_files'].items():
        assert source_digest(name)==digest,name
    modules={p.stem for p in R1_ROOT.glob('*.py')}|{p.stem for p in (R1_ROOT/'scripts').rglob('*.py')}
    for p in list(R1_ROOT.glob('*.py'))+list((R1_ROOT/'scripts').rglob('*.py')):
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    for folder in (R1_ROOT/'artifacts/current').rglob('*'):
        if folder.name=='protocol.json':
            data=json.loads(folder.read_text(encoding='utf-8'))
            for n,h in data.get('frozen_source_sha256',{}).items():verify_source_hash(n,h)
    table2=[]
    for regime in ['common','mismatch']:
        base=artifact_path('results_r1_external_expanded')
        runs=pd.read_csv(base/f'guarded_{regime}_runs.csv')
        routes=pd.read_csv(base/f'guarded_{regime}_routes.csv')
        assert len(runs)==80 and len(routes)==320
        assert not runs.duplicated(['Method','Seed']).any()
        assert not routes.duplicated(['Method','Seed','vehicle']).any()
        for _,row in runs.iterrows():
            r=routes[(routes.Method==row.Method)&(routes.Seed==row.Seed)]
            assert len(r)==4
            values={'SR':r.success.mean(),'CR':r.collision.mean(),
                    'SC':100*(1-r.violation_steps.sum()/r.steps.sum()),
                    'MinClearance':r.min_clearance.min(),
                    'MeanViolationDepth':(r.violation_steps*r.mean_violation_depth).sum()/max(1,r.violation_steps.sum()),
                    'MaxViolationDuration':r.max_violation_duration.max()}
            for name,value in values.items():assert np.isclose(value,row[name],rtol=0,atol=1e-8),(regime,row.Method,row.Seed,name)
        sr=100*runs.SR;runs['CBS']=2*sr*runs.SC/(sr+runs.SC).clip(lower=1e-12)*(1-runs.CR)
        g=runs.groupby('Method')[METRICS+['CBS']].mean();g.SR*=100;g.CR*=100
        g.insert(0,'Plant',regime);table2.append(g.reset_index())
    pd.concat(table2).to_csv(output/'table2_reconstructed.csv',index=False)
    main=artifact_path('results_r1_final_screened')
    ab=pd.read_csv(main/'final_mechanisms_runs.csv')
    assert set(['Full FedUMPC','No uncertainty response','No bounded trust','No hybrid clearance','No candidate screen'])<=set(ab.Method)
    sr=100*ab.SR;ab['CBS']=2*sr*ab.SC/(sr+ab.SC).clip(lower=1e-12)*(1-ab.CR)
    g=ab.groupby('Method')[METRICS+['CBS']].mean();g.SR*=100;g.CR*=100;g.to_csv(output/'table3_reconstructed.csv')
    mc=pd.read_csv(main/'r1_mc_closed_loop_runs.csv').groupby('M').agg(Mean_run_p95_ms=('ComputeP95Ms','mean'))
    mc.to_csv(output/'table4_raw_timing_reference.csv')
    # Summary corrections are preserved, not expanded into invented scene data.
    summary=pd.read_csv(main/'r1_mc_closed_loop_summary.csv').set_index('M')
    discrepancies={int(m):{'raw_mean':float(mc.loc[m,'Mean_run_p95_ms']),'reported':float(row.Mean_run_p95_ms)}
                   for m,row in summary.iterrows() if not np.isclose(mc.loc[m,'Mean_run_p95_ms'],row.Mean_run_p95_ms)}
    result={'source_algorithms_unchanged':6,'table2_verified_scenes':160,'table2_verified_routes':640,
            'all_python_files_parse':True,'table4_recorded_aggregate_corrections':discrepancies,
            'stored_model_files':len(list(artifact_path('results_r1/models').glob('*.pt'))),
            'python':sys.version,'numpy':np.__version__}
    (output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

def smoke(output,seeds=1):
    """Re-execute matched stored checkpoints; timing is never equality-tested."""
    import r1_experiments as exp
    from config import Config
    from controllers import FedUMPCController,LinearMPC
    output=Path(output).resolve()
    if output.exists():raise ValueError('Smoke output must be new to prevent a cached pass')
    output.mkdir(parents=True)
    protocol=json.loads((artifact_path('results_r1_final_screened')/'protocol.json').read_text())
    for n,h in protocol['frozen_source_sha256'].items():verify_source_hash(n,h)
    assert 1<=seeds<=10
    Config.SIM_STEPS=200;Config.HORIZON=25;Config.MC_SAMPLES=10;Config.USE_SLSQP_REFINEMENT=False
    exp.OUT=output;Config.RESULTS_DIR=str(output)
    model=exp.load_model(artifact_path('results_r1/models/prox.pt'))
    methods={'Nominal MPC':lambda e:[LinearMPC(e) for _ in range(4)],
             'FedUMPC':lambda e:[FedUMPCController(e,model) for _ in range(4)]}
    repeat,_=exp.run_panel('smoke_common',methods,protocol['heldout_seeds'][:seeds])
    reference=pd.read_csv(artifact_path('results_r1_final_screened/guarded_common_runs.csv'))
    both=repeat.merge(reference,on=['Method','Seed'],suffixes=('_repeat','_reference'),validate='one_to_one')
    checks=[]
    for _,row in both.iterrows():
        for metric in METRICS:
            a,b=row[metric+'_repeat'],row[metric+'_reference']
            checks.append({'Method':row.Method,'Seed':int(row.Seed),'Metric':metric,'repeat':float(a),
                           'reference':float(b),'absolute_difference':float(abs(a-b)),
                           'matches':bool(np.isclose(a,b,rtol=1e-5,atol=1e-6))})
    pd.DataFrame(checks).to_csv(output/'repeat_comparison.csv',index=False)
    assert all(x['matches'] for x in checks),'Inspect repeat_comparison.csv; numerical differences were retained'
    print('PASS: fresh scene replay matches six non-timing endpoints for both controllers.')
