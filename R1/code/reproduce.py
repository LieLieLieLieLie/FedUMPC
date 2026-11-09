"""Single entry point for R1 checks, isolated reruns, and figure generation."""
from pathlib import Path
import argparse
import fnmatch
import importlib
import json
import os
import runpy
import shutil
import sys

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
sys.dont_write_bytecode=True
for folder in ['experiments','analysis','figures','tests','development','legacy']:
    sys.path.append(str(ROOT/'scripts'/folder))

TASKS={
    'train-predictors':'train_predictors.py',
    'uncertainty-diagnostics':'uncertainty_diagnostics.py',
    'prepare-figure-data':'r1_external_expanded_figures.py',
    'common':'r1_final_evaluation.py', 'learners':'r1_final_learners.py',
    'ablation':'paper_ablation.py','uncertainty-controls':'r1_final_screened_interventions.py',
    'robustness':'r1_final_robustness.py','wind':'r1_wind_only.py',
    'wind-summary':'r1_finalize_wind_only.py','diagnostics':'r1_final_screened_diagnostics.py',
    'ensemble-margin':'r1_ensemble_margin.py','ensemble-summary':'r1_finalize_ensemble_margin.py',
    'prototypes':'r1_external_mechanism_pilots.py','prototype-summary':'r1_finalize_external_mechanism_pilots.py',
    'prototype-disturbances':'r1_external_robustness.py',
    'federation':'r1_federation_validation_20261109.py',
    'matched-ci':'r1_matched_predictor_ci.py','prediction-horizon':'r1_prediction_horizon.py',
    'prediction-horizon-reference':'prediction_horizon_reference.py',
    'scenario-audit':'r1_audit_scenario.py',
}
FRESH={
    'train-predictors':('results_r1',['models/*.pt','models/training_metadata.json']),
    'common':('results_r1_final_screened',['guarded_common*','guarded_mismatch*','seven_metric_scoreboard.csv']),
    'learners':('results_r1_final_screened',['guarded_learners*','guarded_learner_scoreboard.csv']),
    'ablation':('results_r1_final_screened',['final_mechanisms*','mechanism_table.csv']),
    'uncertainty-controls':('results_r1_final_screened',['final_uncertainty*','uncertainty_controls_table.csv','fixed_growth_margin.csv']),
    'robustness':('results_r1_final_screened',['final_robustness*']),
    'diagnostics':('results_r1_final_screened',['r1_mc_*','r1_candidate_latency.csv','r1_obstacle_sensitivity*']),
    'wind':('results_r1_wind_only',['wind_only_*.csv']),
    'prototypes':('results_r1_external_mechanism_pilots',['common_*.csv','mismatch_*.csv']),
    'prototype-disturbances':('results_r1_external_expanded',['aekf_*_runs.csv','aekf_*_routes.csv','cbf_*_runs.csv','cbf_*_routes.csv']),
    'ensemble-margin':('results_r1_ensemble_margin',['common_*.csv','mismatch_*.csv']),
    'federation':('results_r1_federation_validated_20261109',['development.csv','selection.json','fleet_all_settings.csv','r1_communication_stress.csv','models/*.pt']),
}

def workspace(path, task=None, fresh=False):
    """Copy records once; fresh evaluation omits only its own copied outputs."""
    from project_paths import _MAP
    path=Path(path).resolve()
    if path==ROOT or ROOT in path.parents or path in ROOT.parents:
        raise ValueError('Use an independent workspace outside the reference R1 tree')
    if path.exists():
        if fresh:raise ValueError('--fresh requires a new, nonexistent workspace')
        if not (path/'workspace.json').exists():raise ValueError('Unrecognized existing workspace')
    else:
        path.mkdir(parents=True)
        omit=FRESH.get(task) if fresh else None
        for source in (ROOT/'artifacts').rglob('*'):
            if not source.is_file():continue
            rel=source.relative_to(ROOT/'artifacts')
            if omit:
                oldname,patterns=omit
                sub=Path(_MAP['data'][oldname]).relative_to('artifacts')
                if rel.is_relative_to(sub) and any(fnmatch.fnmatch(rel.relative_to(sub).as_posix(),p) for p in patterns):continue
            dest=path/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        (path/'workspace.json').write_text(json.dumps({'source':str(ROOT),'task':task,'fresh':fresh},indent=2))
    os.environ['FEDUMPC_ARTIFACTS_ROOT']=str(path)
    return path

def figures(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if output==ROOT or ROOT in output.parents or output in ROOT.parents:
        raise ValueError('Choose an independent figure output directory')
    import r1_layout_figures_20261109 as layout
    from project_paths import artifact_path
    layout.FRESH_FLEET = not (artifact_path('results_r1')/'r1_fleet_scaling.csv').is_file()
    layout.DEST=output/'layout';layout.main()
    import r1_finalize_federation_validation_20261109 as federation
    federation.P=output; (output/'sections').mkdir(exist_ok=True)
    federation.OUT=output/'federation';federation.main()
    # Figure 4 comes from the retained physical-engine trajectories, not the
    # analytical plant. All generated tables/figures go to the isolated output.
    physical=ROOT.parent/'MuJoCo'
    if not (physical/'r1_external_prototype_figures.py').exists():
        physical=ROOT.parents[1]/'MuJoCo/R1'
    if (physical/'r1_external_prototype_figures.py').exists():
        spec=importlib.util.spec_from_file_location('mujoco_record_figures',physical/'r1_external_prototype_figures.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.OUT=output/'mujoco_records'
        shutil.copytree(physical/'outputs_r1_external_prototypes',module.OUT)
        module.DEST=output/'mujoco';module.DEST.mkdir()
        module.main()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    p=sub.add_parser('check');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('smoke');p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,default=1)
    p=sub.add_parser('run');p.add_argument('task',choices=TASKS);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--fresh',action='store_true');p.add_argument('--stage-args',nargs='*',default=[])
    p=sub.add_parser('figures');p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='list':
        print('\n'.join(f'{k:24s} {v}' for k,v in TASKS.items()));return
    if args.command in {'run','figures'}:workspace(args.workspace,getattr(args,'task',None),getattr(args,'fresh',False))
    # Install the experiment runtime's existing Windows compatibility shim first.
    import r1_experiments
    if args.command=='check':
        from verify_reproduction import verify
        verify(args.output)
        runpy.run_path(str(ROOT/'scripts/tests/r1_regression_tests.py'),run_name='__main__')
    elif args.command=='smoke':
        from verify_reproduction import smoke
        smoke(args.output,args.seeds)
    elif args.command=='figures':figures(args.output)
    else:
        from project_paths import source_file
        name=TASKS[args.task]
        if name in {'paper_ablation.py','train_predictors.py','uncertainty_diagnostics.py'}:
            path=ROOT/'scripts/experiments'/name
        elif name=='prediction_horizon_reference.py':
            path=ROOT/'scripts/analysis'/name
        else:path=source_file(name)
        sys.argv=[str(path),*args.stage_args]
        runpy.run_path(str(path),run_name='__main__')

if __name__=='__main__':main()
