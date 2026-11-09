"""Aggregate Wind evidence without modifying the five-condition source records."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol
import hashlib
import json
from pathlib import Path
import r1_experiments as exp
import numpy as np
import pandas as pd

ROOT=R1_ROOT
OLD=artifact_path('results_r1_final_screened')
OUT=artifact_path('results_r1_wind_only')
METRICS=['SR','SC','CR','CBS','MinClearance','MeanViolationDepth','MaxViolationDuration']
def score(df):
    sr=100*df.SR
    return 2*sr*df.SC/(sr+df.SC).clip(lower=1e-12)*(1-df.CR)
def main():
    runs=pd.read_csv(OUT/'wind_only_runs.csv')
    routes=pd.read_csv(OUT/'wind_only_routes.csv')
    assert not runs.duplicated(['Method','Seed']).any() and len(runs)==60
    assert not routes.duplicated(['Method','Seed','vehicle']).any() and len(routes)==240
    runs['CBS']=score(runs)
    rows=[]
    for method,g in runs.groupby('Method',sort=False):
        row={'Method':method}
        for metric in METRICS:
            vals=g[metric].to_numpy()*(100 if metric in ['SR','CR'] else 1)
            row[metric]=float(vals.mean())
            row[metric+'_sd']=float(vals.std(ddof=1))
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'wind_endpoints.csv',index=False)
    ref=runs[runs.Method=='FedUMPC'].set_index('Seed')
    contrasts=[]
    for method,g in runs.groupby('Method',sort=False):
        if method=='FedUMPC':continue
        g=g.set_index('Seed').loc[ref.index]
        row={'Comparator':method,'scenes':10}
        for metric in METRICS:
            delta=(ref[metric]-g[metric]).to_numpy()*(100 if metric in ['SR','CR'] else 1)
            lo,hi=exp.cluster_bootstrap(delta)
            row.update({metric:float(delta.mean()),metric+'_lo':lo,metric+'_hi':hi})
        contrasts.append(row)
    pd.DataFrame(contrasts).to_csv(OUT/'wind_scene_cluster_contrasts.csv',index=False)
    runs.insert(0,'Condition','Wind-only')
    old=pd.read_csv(OLD/'final_robustness_runs.csv')
    old['CBS']=score(old)
    combined=pd.concat([old,runs],ignore_index=True)
    combined.to_csv(OUT/'six_condition_runs.csv',index=False)
    summaries=[]
    for (condition,method),g in combined.groupby(['Condition','Method'],sort=False):
        row={'Condition':condition,'Method':method}
        for metric in METRICS:
            vals=g[metric].to_numpy()*(100 if metric in ['SR','CR'] else 1)
            lo,hi=exp.cluster_bootstrap(vals)
            row.update({metric+'_mean':float(vals.mean()),metric+'_std':float(vals.std(ddof=1)),
                        metric+'_lo':lo,metric+'_hi':hi})
        summaries.append(row)
    six=pd.DataFrame(summaries)
    six.to_csv(OUT/'six_condition_summary.csv',index=False)
    wins=[]
    for condition,g in six.groupby('Condition',sort=False):
        fed=g[g.Method=='FedUMPC'].iloc[0]
        best=[]
        for metric in METRICS:
            vals=g[metric+'_mean']
            target=vals.min() if metric in ['CR','MeanViolationDepth','MaxViolationDuration'] else vals.max()
            if np.isclose(fed[metric+'_mean'],target,atol=1e-8,rtol=0):best.append(metric)
        wins.append({'condition':condition,'first_place_endpoints':best})
    report={'six_condition_fed_first_places':sum(len(x['first_place_endpoints']) for x in wins),
            'endpoint_cells':42,'by_condition':wins,
            'original_five_condition_csv_sha256':hashlib.sha256((OLD/'final_robustness_runs.csv').read_bytes()).hexdigest(),
            'bootstrap_resamples':10000,'bootstrap_seed':20261103,
            'inference_unit':'paired scene, retaining all four routes',
            'wind_added_after_original_five_condition_evaluation':True}
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
