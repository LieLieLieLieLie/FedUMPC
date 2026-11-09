"""Re-evaluate only the five component-removal rows retained in Table III."""
import r1_experiments as exp
from config import Config
from controllers import FedUMPCController
from project_paths import artifact_path
from r1_final_mechanisms import NominalClearanceFedUMPC, verify, SEEDS

def main():
    verify()
    Config.SIM_STEPS=200;Config.HORIZON=25;Config.MC_SAMPLES=10
    Config.USE_SLSQP_REFINEMENT=False
    exp.OUT=artifact_path('results_r1_final_screened')
    Config.RESULTS_DIR=str(exp.OUT)
    model=exp.load_model(artifact_path('results_r1/models/prox.pt'))
    methods={
        'Full FedUMPC':lambda e:[FedUMPCController(e,model) for _ in range(4)],
        'No uncertainty response':lambda e:[FedUMPCController(e,model,uncertainty_aware=False) for _ in range(4)],
        'No bounded trust':lambda e:[FedUMPCController(e,model,use_model_trust=False) for _ in range(4)],
        'No hybrid clearance':lambda e:[NominalClearanceFedUMPC(e,model) for _ in range(4)],
        'No candidate screen':lambda e:[FedUMPCController(e,model,use_candidate_screen=False) for _ in range(4)],
    }
    runs,routes=exp.run_panel('final_mechanisms',methods,SEEDS)
    exp.summarize_runs(runs,'final_mechanisms')
    exp.paired_contrasts(runs,routes,reference='Full FedUMPC',prefix='final_mechanisms')
    sr=100*runs.SR
    runs['CBS']=2*sr*runs.SC/(sr+runs.SC).clip(lower=1e-12)*(1-runs.CR)
    table=runs.groupby('Method',sort=False)[['SR','SC','CR','CBS','MinClearance']].mean().reset_index()
    table.SR*=100;table.CR*=100
    table.to_csv(exp.OUT/'mechanism_table.csv',index=False)
    print(table.to_string(index=False))

if __name__=='__main__':main()
