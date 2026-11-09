"""Post-evaluation predictor diagnosis; never retrains or selects a controller.

This is hypothesis-generating analysis of the frozen R1 checkpoints, not a new
confirmatory comparison. All outputs go to a separate diagnostic directory.
"""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import hashlib
import json

import r1_experiments as r
import numpy as np
import pandas as pd
import torch


SEED = 20261117
DEST = r.OUT / 'predictor_diagnosis'
COMPONENTS = [('x', 'm'), ('y', 'm'), ('v', 'm/s'), ('heading', 'rad')]


def physics_batch(states, controls, masses, drags):
    """Exactly the disturbance-free VehicleModel update, with vector inputs."""
    state = np.asarray(states, dtype=float)
    action = np.asarray(controls, dtype=float)
    out = state.copy()
    steer = np.clip(action[:, 0], -r.Config.MAX_STEER, r.Config.MAX_STEER)
    accel = np.clip(action[:, 1], -r.Config.MAX_ACCEL, r.Config.MAX_ACCEL)
    beta = np.arctan(.5 * np.tan(steer))
    out[:, 0] += r.Config.DT * state[:, 2] * np.cos(state[:, 3] + beta)
    out[:, 1] += r.Config.DT * state[:, 2] * np.sin(state[:, 3] + beta)
    out[:, 2] = np.maximum(0., state[:, 2] + r.Config.DT * (
        r.Config.MASS / np.asarray(masses) * accel - np.asarray(drags) * state[:, 2]))
    out[:, 3] += r.Config.DT * state[:, 2] / r.Config.L * np.sin(beta)
    return out


def prediction(model, states, controls, mode):
    if mode == 'MC10':
        output = []
        for start in range(0, len(states), 256):
            mean, _ = model.predict_horizon(states[start:start + 256],
                                            controls[start:start + 256],
                                            num_samples=10)
            output.append(mean)
        return np.concatenate(output)
    model.eval()
    with torch.no_grad():
        xu = torch.as_tensor(np.c_[states, controls], dtype=torch.float32,
                             device=r.Config.DEVICE)
        return model(xu).cpu().numpy()


def component_records(name, condition, mode, estimates, targets, scale):
    error = estimates - targets
    rows = []
    for index, (component, unit) in enumerate(COMPONENTS):
        values = error[:, index]
        rows.append(dict(Method=name, Condition=condition, Inference=mode,
                         Component=component, Unit=unit, Queries=len(values),
                         Bias=float(values.mean()),
                         RMSE=float(np.sqrt(np.mean(values ** 2))),
                         MAE=float(np.abs(values).mean()),
                         Normalized_RMSE=float(np.sqrt(np.mean(
                             (values / scale[index]) ** 2)))))
    return rows


def one_step(models, metadata, scale):
    rows = []
    n = 2048
    rng = np.random.default_rng(SEED)
    states = np.c_[rng.uniform(-12, 12, (n, 2)), rng.uniform(2, 8, n),
                   rng.uniform(-np.pi, np.pi, n)]
    clients = np.arange(n) % 4
    for condition in ['Training-client plants', 'Strong mismatch', 'Full actuator range']:
        if condition == 'Training-client plants':
            masses = np.asarray([metadata[str(i)]['mass_kg'] for i in clients])
            drags = np.asarray([metadata[str(i)]['drag'] for i in clients])
        else:
            masses = np.asarray([650., 850., 1150., 1350.])[clients]
            drags = np.asarray([.03, .05, .08, .11])[clients]
        low, high = (([-.6, -2.], [.6, 2.]) if condition == 'Full actuator range'
                     else ([-.3, -1.], [.3, 1.]))
        controls = rng.uniform(low, high, size=(n, 2))
        true_inc = physics_batch(states, controls, masses, drags) - states
        nominal_inc = physics_batch(states, controls, r.Config.MASS,
                                    r.Config.DRAG) - states
        # The evaluated plants differ only in the speed equation at one step.
        np.testing.assert_allclose(nominal_inc[:, [0, 1, 3]],
                                   true_inc[:, [0, 1, 3]], atol=1e-12)
        rows += component_records('Nominal physics', condition, 'Deterministic',
                                  nominal_inc, true_inc, scale)
        for name, model in models.items():
            for mode in ['Dropout disabled', 'MC10']:
                r.reset_rng(SEED + 1)
                if isinstance(model, list):
                    estimate = np.empty_like(true_inc)
                    for client, local_model in enumerate(model):
                        ix = clients == client
                        estimate[ix] = prediction(local_model, states[ix],
                                                  controls[ix], mode)
                else:
                    estimate = prediction(model, states, controls, mode)
                rows += component_records(name, condition, mode, estimate,
                                          true_inc, scale)
    result = pd.DataFrame(rows)
    result.to_csv(DEST / 'one_step_components.csv', index=False)
    return result


def horizon_probe(model):
    n = 256
    rng = np.random.default_rng(SEED + 2)
    initial = np.c_[rng.uniform(-10, 10, (n, 2)), rng.uniform(2, 7, n),
                    rng.uniform(-np.pi, np.pi, n)]
    controls = rng.uniform([-.3, -1.], [.3, 1.], size=(n, r.Config.HORIZON, 2))
    clients = np.arange(n) % 4
    masses = np.asarray([650., 850., 1150., 1350.])[clients]
    drags = np.asarray([.03, .05, .08, .11])[clients]
    rows = []
    for variant in ['Implemented frozen-nominal increments',
                    'State-consistent physics increments']:
        r.reset_rng(SEED + 3)
        truth = initial.copy()
        nominal = initial.copy()
        hybrid = initial.copy()
        for h in range(r.Config.HORIZON):
            u = controls[:, h]
            truth = physics_batch(truth, u, masses, drags)
            nominal_next = physics_batch(nominal, u, r.Config.MASS, r.Config.DRAG)
            if variant.startswith('Implemented'):
                increment = nominal_next - nominal
            else:
                increment = physics_batch(hybrid, u, r.Config.MASS,
                                          r.Config.DRAG) - hybrid
            mean, variance = model.predict_horizon(hybrid, u, num_samples=10)
            mean = np.clip(mean, -2., 2.)
            uncertainty = np.sqrt(np.maximum(variance, 0.))
            mismatch = np.linalg.norm(mean - increment, axis=1)
            trust = np.clip(np.exp(-uncertainty / r.Config.BNN_TRUST_UNC_SCALE)
                            * np.exp(-mismatch / (
                                np.linalg.norm(increment, axis=1) + .2)), 0., .55)
            hybrid += (1. - trust[:, None]) * increment + trust[:, None] * mean
            projection = np.any(np.abs(hybrid[:, :2]) > r.Config.WORLD_LIMIT, axis=1)
            hybrid[:, :2] = np.clip(hybrid[:, :2], -r.Config.WORLD_LIMIT,
                                   r.Config.WORLD_LIMIT)
            for query in range(n):
                rows.append(dict(Variant=variant, Query=query, Horizon=h + 1,
                    Time_s=(h + 1) * r.Config.DT,
                    Nominal_position_error_m=float(np.linalg.norm(
                        truth[query, :2] - nominal_next[query, :2])),
                    Hybrid_position_error_m=float(np.linalg.norm(
                        truth[query, :2] - hybrid[query, :2])),
                    Trust=float(trust[query]),
                    Trust_at_cap=bool(np.isclose(trust[query], .55, atol=1e-7)),
                    Uncertainty=float(uncertainty[query]),
                    Heading_outside_training_support=bool(abs(hybrid[query, 3]) > np.pi),
                    Scene_projection=bool(projection[query])))
            nominal = nominal_next
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST / 'horizon_geometry_probe.csv', index=False)
    summary = frame.groupby(['Variant', 'Horizon', 'Time_s'], as_index=False).agg(
        Nominal_error_m=('Nominal_position_error_m', 'mean'),
        Hybrid_error_m=('Hybrid_position_error_m', 'mean'),
        Mean_trust=('Trust', 'mean'), Trust_cap_fraction=('Trust_at_cap', 'mean'),
        Mean_uncertainty=('Uncertainty', 'mean'),
        Heading_outside_support_fraction=('Heading_outside_training_support', 'mean'),
        Projection_count=('Scene_projection', 'sum'))
    summary.to_csv(DEST / 'horizon_geometry_summary.csv', index=False)
    return summary


def main():
    r.Config.set_mode('fast')
    DEST.mkdir(exist_ok=True)
    required = [r.MODEL_DIR / f'{name}.pt' for name in
                ['prox', 'fedavg', 'central', 'local_0', 'local_1', 'local_2', 'local_3']]
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in required}
    models = {'Prox-FL': r.load_model(required[0]),
              'FedAvg': r.load_model(required[1]),
              'Centralized': r.load_model(required[2]),
              'Local': [r.load_model(path) for path in required[3:]]}
    metadata = json.loads((r.MODEL_DIR / 'training_metadata.json').read_text(
        encoding='utf-8'))['training_clients']
    scale = models['Prox-FL'].delta_scale.detach().cpu().numpy()
    # The vectorized diagnostic must reproduce the actual executed plant.
    check_states = np.array([[-3., 2., 5., -.4], [1., -2., 3., 2.8]])
    check_actions = np.array([[.25, -.8], [-.2, .6]])
    check_masses, check_drags = np.array([650., 1350.]), np.array([.03, .11])
    expected = np.array([r.VehicleModel(m, d).step(s, u) for s, u, m, d in
                         zip(check_states, check_actions, check_masses, check_drags)])
    np.testing.assert_allclose(physics_batch(check_states, check_actions,
        check_masses, check_drags), expected, atol=1e-12)
    components = one_step(models, metadata, scale)
    horizons = horizon_probe(models['Prox-FL'])
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in required}
    assert before == after, 'Diagnosis must not change trained checkpoints'
    findings = {'scope': 'Post-evaluation hypothesis-generating diagnosis, not confirmatory tests or tuning.',
        'seed': SEED, 'checkpoint_sha256': before,
        'one_step_queries_per_condition': 2048,
        'one_step_plant_identity': 'Mass and drag affect only the speed increment. Nominal position and heading increments are exact at a common initial state.',
        'training_generator': {'raw_steering_support': [-1, 1], 'applied_steering_limits': [-.6, .6],
            'acceleration_training_support': [-1, 1], 'acceleration_controller_limits': [-2, 2],
            'disturbance_mean_acceleration_m_s2': .2,
            'heading_label_wrap': 'No wrap in VehicleModel; no +/-2pi label discontinuity found.'},
        'horizon_final': horizons[horizons.Horizon == r.Config.HORIZON].to_dict(orient='records')}
    (DEST / 'diagnosis_metadata.json').write_text(json.dumps(findings, indent=2), encoding='utf-8')
    print(components[(components.Condition == 'Strong mismatch') &
                     components.Inference.isin(['Deterministic', 'MC10'])][[
                        'Method', 'Component', 'RMSE', 'Bias']].to_string(index=False))
    print(horizons[horizons.Horizon == r.Config.HORIZON].to_string(index=False))
    print('Checkpoints unchanged:', len(before))


if __name__ == '__main__':
    main()
