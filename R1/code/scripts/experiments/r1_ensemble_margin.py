"""Independent reviewer-requested ensemble-disagreement margin control.

No FedUMPC trust, hybrid geometry, risk pricing, or candidate screen is used.
The original frozen sources, weights, scenes, and evaluation files are read-only.
"""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import r1_experiments as r1
from config import Config
from controllers import LinearMPC
from models import EnsemblePredictor

ROOT = R1_ROOT
OUT = artifact_path('results_r1_ensemble_margin')
PROTOCOL = json.loads((artifact_path('results_r1_final_screened/protocol.json')).read_text())

class EnsembleMarginMPC(LinearMPC):
    def __init__(self, env, ensemble):
        super().__init__(env)
        self.ensemble = ensemble

    def _cost(self, u_flat, state, target):
        u = np.asarray(u_flat).reshape(self.H, 2)
        states = self._rollout(u, state)
        queries = np.vstack([state, states[:-1]])
        _, variance = self.ensemble.predict_horizon(queries, u)
        uncertainty = np.sqrt(np.maximum(variance, 0.0))
        margin = Config.SAFETY_MARGIN + np.clip(
            Config.ROBUST_BETA * Config.UNCERTAINTY_MARGIN_SCALE
            * uncertainty * (1.0 + 0.15*np.arange(self.H)),
            0.0, Config.MAX_UNCERTAINTY_MARGIN)
        return self._base_cost(states, u, target) + sum(
            self._obstacle_cost(states[h], margin[h]) for h in range(self.H))

def main():
    OUT.mkdir(exist_ok=True)
    for name, expected in PROTOCOL['frozen_source_sha256'].items():
        assert hashlib.sha256((source_file(name)).read_bytes()).hexdigest() == expected
    Config.SIM_STEPS=200
    Config.HORIZON=25
    Config.USE_SLSQP_REFINEMENT=False
    Config.RESULTS_DIR=str(OUT)
    r1.OUT=OUT
    paths=[artifact_path('results_r1/models')/f'ensemble_{i}.pt' for i in range(3)]
    ensemble=EnsemblePredictor([r1.load_model(p) for p in paths])
    config={'seeds':PROTOCOL['heldout_seeds'], 'members':3,
            'model_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'base_margin_m':Config.SAFETY_MARGIN, 'beta':Config.ROBUST_BETA,
            'uncertainty_scale_m':Config.UNCERTAINTY_MARGIN_SCALE,
            'inflation_slope':0.15, 'cap_m':Config.MAX_UNCERTAINTY_MARGIN,
            'parameter_selection':'existing frozen margin map; no held-out tuning',
            'uncertainty':'root trace of unbiased sample covariance of three deterministic normalized member increments',
            'rollout':'nominal physics; no learned trust, hybrid clearance, or screening',
            'frozen_source_sha256':PROTOCOL['frozen_source_sha256']}
    (OUT/'protocol.json').write_text(json.dumps(config,indent=2))
    methods={'Ensemble-margin MPC':lambda env:[EnsembleMarginMPC(env,ensemble) for _ in range(4)]}
    for name,kwargs in [('common',{}),('mismatch',{'vehicle_masses':[650.,850.,1150.,1350.], 'vehicle_drags':[.03,.05,.08,.11]})]:
        runs,_=r1.run_panel('ensemble_margin_'+name,methods,config['seeds'],**kwargs)
        r1.summarize_runs(runs,'ensemble_margin_'+name)
    print('COMPLETE',flush=True)

if __name__=='__main__':
    main()
