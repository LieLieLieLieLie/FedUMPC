"""Supplement the frozen R1 robustness protocol with the original Wind condition.

The original Wind label denotes a constant +1 m/s^2 longitudinal acceleration
bias, with no Gaussian position-rate noise. No controller parameter is tuned.
"""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
import hashlib
import json
from pathlib import Path
import r1_experiments as exp
import r1_final_robustness as base
from config import Config

ROOT=R1_ROOT
OUT=artifact_path('results_r1_wind_only')

def main():
    base.final.verify()
    OUT.mkdir(exist_ok=True)
    record={
        'condition':'Wind-only', 'noise_std_m_per_s':0.0,
        'longitudinal_acceleration_bias_m_per_s2':1.0,
        'definition':'Original Wind proxy, not a lateral aerodynamic wind model',
        'parent_protocol':base.final.PROTOCOL,
        'methods':base.METHODS,
        'predictor_sha256':hashlib.sha256((artifact_path('results_r1/models/prox.pt')).read_bytes()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'controller_retuned':False,
    }
    p=OUT/'protocol.json'
    if p.exists():
        check_protocol(json.loads(p.read_text(encoding='utf-8')),record)
    else:p.write_text(json.dumps(record,indent=2),encoding='utf-8')
    Config.SIM_STEPS=200
    Config.HORIZON=25
    Config.MC_SAMPLES=10
    Config.USE_SLSQP_REFINEMENT=False
    Config.SEEDS=base.final.SEEDS
    Config.N_SEEDS=len(base.final.SEEDS)
    Config.RESULTS_DIR=str(OUT)
    exp.OUT=OUT
    prox=exp.load_model(artifact_path('results_r1/models/prox.pt'))
    factories=base.factories(prox)
    methods={name:factories[name] for name in base.METHODS}
    runs,routes=exp.run_panel('wind_only',methods,base.final.SEEDS,noise=0.0,disturbance=1.0)
    assert len(runs)==60 and len(routes)==240
    exp.summarize_runs(runs,'wind_only')
    exp.paired_contrasts(runs,routes,reference='FedUMPC',prefix='wind_only')
    print('Wind-only completed: 60 method-scenes / 240 routes.',flush=True)

if __name__=='__main__':main()
