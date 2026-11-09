"""Audit matched-predictor route clusters and compute paired scene CIs.

Reads the frozen screened evaluation; never changes controller results.
Resamples ten paired scenes, keeping all four routes together. CBS is
computed within each scene before averaging, not from rounded summary rows.
"""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = R1_ROOT
SOURCE = artifact_path('results_r1_final_screened')
OUT = artifact_path('results_r1_matched_predictor_ci_20261109')
LABELS = {'Local UMPC': 'Local', 'FedAvg UMPC': 'FedAvg',
          'Ensemble UMPC': 'Ensemble', 'Centralized UMPC': 'Centralized',
          'FedUMPC': 'Prox-FL'}
METRICS = ['SR', 'SC', 'CR', 'CBS', 'MinClearance',
           'MeanViolationDepth', 'MaxViolationDuration']

def read(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

def save(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def truth(value):
    value = str(value).lower()
    assert value in {'true', 'false', '1', '0', '1.0', '0.0'}, value
    return value in {'true', '1', '1.0'}

def main():
    OUT.mkdir(exist_ok=True)
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    seeds = protocol['heldout_seeds']
    assert len(seeds) == len(set(seeds)) == 10
    scoreboard_path = SOURCE / 'guarded_learner_scoreboard.csv'
    scoreboard = {(r['Regime'], r['Method']): r for r in read(scoreboard_path)}
    paths = [SOURCE / 'protocol.json', scoreboard_path]
    scene_records, summary, route_audits = [], [], []
    for regime in ['common', 'mismatch']:
        runpath = SOURCE / f'guarded_learners_{regime}_runs.csv'
        routepath = SOURCE / f'guarded_learners_{regime}_routes.csv'
        paths += [runpath, routepath]
        runs, routes = read(runpath), read(routepath)
        assert len(runs) == 50 and len(routes) == 200
        assert len({(r['Method'], r['Seed']) for r in runs}) == 50
        assert len({(r['Method'], r['Seed'], r['vehicle']) for r in routes}) == 200
        for method, label in LABELS.items():
            sub = {int(r['Seed']): r for r in runs if r['Method'] == method}
            assert set(sub) == set(seeds)
            for seed in seeds:
                row = sub[seed]
                cluster = [r for r in routes if r['Method'] == method and int(r['Seed']) == seed]
                assert len(cluster) == 4 and len({r['vehicle'] for r in cluster}) == 4
                sr = np.mean([truth(r['success']) for r in cluster])
                cr = np.mean([truth(r['collision']) for r in cluster])
                # SC pools executed route steps within a scene, contact-inclusive.
                violations = sum(int(r['violation_steps']) for r in cluster)
                sc = 100 * (1 - violations / sum(int(r['steps']) for r in cluster))
                clearance = min(float(r['min_clearance']) for r in cluster)
                depth = sum(int(r['violation_steps']) * float(r['mean_violation_depth']) for r in cluster) / max(1, violations)
                duration = max(int(r['max_violation_duration']) for r in cluster)
                for name, value in [('SR', sr), ('CR', cr), ('SC', sc), ('MinClearance', clearance),
                                    ('MeanViolationDepth', depth), ('MaxViolationDuration', duration)]:
                    assert np.isclose(value, float(row[name]), atol=1e-9, rtol=0), (regime, method, seed, name, value, row[name])
                cbs = (2 * (sr * 100) * sc / max(1e-12, sr * 100 + sc)) * (1 - cr)
                record = {'Plant': regime, 'Predictor': label, 'Seed': seed,
                          'SR': sr * 100, 'SC': sc, 'CR': cr * 100, 'CBS': cbs}
                record.update({m: float(row[m]) for m in METRICS[4:]})
                scene_records.append(record)
                route_audits.append({'Plant': regime, 'Predictor': label, 'Seed': seed,
                                     'Routes': 4, 'all_six_route_derived_endpoints_verified': True})
            records = [r for r in scene_records if r['Plant'] == regime and r['Predictor'] == label]
            agg = {'Plant': regime, 'Predictor': label, 'Scenes': 10, 'Routes': 40}
            for metric in METRICS:
                vals = np.array([r[metric] for r in records])
                agg[metric + '_mean'] = float(vals.mean())
                agg[metric + '_sd'] = float(vals.std(ddof=1))
            old = scoreboard[('Common' if regime == 'common' else 'Heterogeneous', method)]
            for metric in ['SR', 'SC', 'CR', 'CBS']:
                assert np.isclose(agg[metric + '_mean'], float(old[metric]), atol=1e-9, rtol=0), (regime, method, metric)
            summary.append(agg)
    contrasts = []
    draws = np.random.default_rng(20261109).integers(0, 10, (10000, 10))
    for comparator in ['Local', 'FedAvg']:
        for metric in ['CBS', 'SR', 'CR']:
            a = {r['Seed']: r[metric] for r in scene_records if r['Plant'] == 'mismatch' and r['Predictor'] == 'Prox-FL'}
            b = {r['Seed']: r[metric] for r in scene_records if r['Plant'] == 'mismatch' and r['Predictor'] == comparator}
            differences = np.array([a[s] - b[s] for s in seeds])
            low, high = np.percentile(differences[draws].mean(axis=1), [2.5, 97.5])
            contrasts.append({'Reference': 'Prox-FL', 'Comparator': comparator,
                              'Metric': metric, 'Unit': 'percentage points',
                              'Difference': float(differences.mean()),
                              'CI95_low': float(low), 'CI95_high': float(high),
                              'Contains_zero': bool(low <= 0 <= high), 'Paired_scenes': 10})
    save(OUT / 'scene_endpoints.csv', scene_records)
    save(OUT / 'summary.csv', summary)
    save(OUT / 'paired_scene_cluster_ci.csv', contrasts)
    save(OUT / 'route_reconstruction_audit.csv', route_audits)
    metadata = {'n_bootstrap': 10000, 'bootstrap_seed': 20261109,
                'cluster': 'one scene seed, with all four vehicle routes retained together',
                'interval': 'marginal paired percentile bootstrap, no multiplicity adjustment',
                'seeds': seeds, 'scene_records': 100, 'routes_verified': 400,
                'CBS': 'scene-wise harmonic mean of SR_percent and SC_percent times (1-CR_fraction)',
                'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT / 'audit.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(contrasts, indent=2))

if __name__ == '__main__':
    main()
