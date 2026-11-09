"""Held-out multi-step prediction diagnostics and an implemented-bound check."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
import r1_experiments as r
import numpy as np
import pandas as pd

def main():
    r.Config.set_mode('fast')
    _, _, model, _, _, _, _ = r.prepare_primary_models()
    rng = np.random.default_rng(20261106)
    r.reset_rng(20261106)
    n = 128
    initial = np.c_[rng.uniform(-12, 12, (n, 2)),
                    rng.uniform(2, 8, n), rng.uniform(-np.pi, np.pi, n)]
    controls = rng.uniform([-0.3, -1.0], [0.3, 1.0],
                           size=(n, r.Config.HORIZON, 2))
    masses = [650., 850., 1150., 1350.]
    drags = [.03, .05, .08, .11]
    truth_models = [r.VehicleModel(masses[i % 4], drags[i % 4]) for i in range(n)]
    physics = r.VehicleModel()
    nominal = initial.copy()
    hybrid = initial.copy()
    truth = initial.copy()
    bound = np.zeros(n)
    projection = np.zeros(n, dtype=bool)
    scale = model.delta_scale.detach().cpu().numpy()
    rows = []
    for h in range(r.Config.HORIZON):
        action = controls[:, h]
        nominal_next = np.asarray([physics.step(s, u)
                                   for s, u in zip(nominal, action)])
        truth = np.asarray([p.step(s, u)
                            for p, s, u in zip(truth_models, truth, action)])
        mean, variance = model.predict_horizon(hybrid, action)
        mean = np.clip(mean, -2., 2.)
        uncertainty = np.sqrt(np.maximum(variance, 0.))
        increment = nominal_next - nominal
        discrepancy = np.linalg.norm(mean - increment, axis=1)
        trust = np.clip(np.exp(-uncertainty / r.Config.BNN_TRUST_UNC_SCALE)
                        * np.exp(-discrepancy /
                                 (np.linalg.norm(increment, axis=1) + .2)),
                        0., .55)
        hybrid += (1. - trust[:, None]) * increment + trust[:, None] * mean
        projection |= np.any(np.abs(hybrid[:, :2]) > r.Config.WORLD_LIMIT, axis=1)
        hybrid[:, :2] = np.clip(hybrid[:, :2], -r.Config.WORLD_LIMIT,
                               r.Config.WORLD_LIMIT)
        bound += trust * discrepancy
        gap = np.linalg.norm(hybrid - nominal_next, axis=1)
        for i in range(n):
            rows.append(dict(Query=i, Client=i % 4 + 1, Horizon_step=h + 1,
                             Time_s=(h + 1) * r.Config.DT,
                             Physics_position_error_m=float(np.linalg.norm(
                                 truth[i, :2] - nominal_next[i, :2])),
                             Hybrid_position_error_m=float(np.linalg.norm(
                                 truth[i, :2] - hybrid[i, :2])),
                             Physics_normalized_error=float(np.linalg.norm(
                                 (truth[i] - nominal_next[i]) / scale)),
                             Hybrid_normalized_error=float(np.linalg.norm(
                                 (truth[i] - hybrid[i]) / scale)),
                             Uncertainty=float(uncertainty[i]),
                             Hybrid_disagreement=float(gap[i]),
                             Accumulated_disagreement_bound=float(bound[i]),
                             Projection_active=bool(projection[i])))
        nominal = nominal_next
    frame = pd.DataFrame(rows)
    interior = frame[~frame.Projection_active]
    assert np.all(interior.Hybrid_disagreement
                  <= interior.Accumulated_disagreement_bound + 1e-5)
    frame.to_csv(r.OUT / 'r1_prediction_horizon.csv', index=False)
    (frame.groupby(['Horizon_step', 'Time_s'], as_index=False)
     .agg(Physics_position_error_m=('Physics_position_error_m', 'mean'),
          Physics_position_error_sd=('Physics_position_error_m', 'std'),
          Hybrid_position_error_m=('Hybrid_position_error_m', 'mean'),
          Hybrid_position_error_sd=('Hybrid_position_error_m', 'std'),
          Mean_uncertainty=('Uncertainty', 'mean'))
     .to_csv(r.OUT / 'r1_prediction_horizon_summary.csv', index=False))
    print('Checked interior disagreement bound on', len(interior), 'query horizons')
    print(frame[frame.Horizon_step == r.Config.HORIZON][[
        'Physics_position_error_m', 'Hybrid_position_error_m']].mean())

if __name__ == '__main__':
    main()
