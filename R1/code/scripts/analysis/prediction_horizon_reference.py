"""Reproduce the historical horizon diagnostic, explicitly distinct from Full.

The stored diagnostic was produced with trust uncertainty scale 0.55, retained
in the pre-residual source snapshot. The final controller uses 0.12. This
wrapper recreates that original diagnostic; it does not retune Full FedUMPC.
"""
from pathlib import Path
import json
import r1_experiments as exp
import r1_prediction_horizon
from project_paths import artifact_path

if __name__=='__main__':
    scale=0.55
    with exp.temporary_config(BNN_TRUST_UNC_SCALE=scale):
        r1_prediction_horizon.main()
    record={'diagnostic':'historical paper horizon diagnostic',
            'uncertainty_trust_scale':scale,'final_controller_scale':0.12,
            'source':'FedRMPC/revision_snapshots/R1_pre_residual_20261103/code/R1/config.py',
            'distinction':'not an evaluation using the final controller scale',
            'seed':20261106,'queries':128,'horizon':25}
    (artifact_path('results_r1')/'historical_horizon_protocol.json').write_text(json.dumps(record,indent=2))
