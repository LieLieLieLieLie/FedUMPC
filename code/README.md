# FedRMPC

Official implementation of **FedRMPC: Federated Uncertainty-Aware Predictive Control for Heterogeneous Internet-of-Vehicles Systems**.

FedRMPC is a safety-oriented edge--cloud architecture that combines federated dynamics learning, MC-dropout epistemic-uncertainty estimation, and vehicle-local model predictive control. The physics model supplies the nominal geometric rollout, while the shared predictor contributes candidate-dependent uncertainty, learned-rollout disagreement, and adaptive static-obstacle risk shaping. The implementation evaluates independent four-vehicle rollouts under both common and physically heterogeneous plants; it does not implement coupled trajectory negotiation or certified safety.

## Highlights

- Federated MC-dropout dynamics learning evaluated using Local Only, FedAvg, MOON, and Prox-FL under a matched primary protocol; SCAFFOLD remains implemented but is excluded from the paper's main table because equal-budget method-specific tuning was unavailable.
- Candidate-batched MC-dropout evaluation that preserves recursive horizon uncertainty while reducing neural-network launch overhead.
- Baselines including Linear MPC, Tube MPC, Stochastic MPC, GP-MPC, and Robust MPC.
- Auditable five-seed scripts for common-plant and heterogeneous-plant control, robustness, ablation, sensitivity, gradient-noise, learner matching, partial participation, and runtime reporting.
- Publication-ready figures and spreadsheets exported automatically.

The benchmark evaluates static-obstacle avoidance and does not model dynamic vehicle-to-vehicle collision avoidance.

## Repository Structure

```text
.
├── config.py         # Global simulation, MPC, FL, BNN, DP, and plotting parameters
├── controllers.py    # MPC baselines and FedRMPC controller
├── federated.py      # Federated clients, server aggregation, and privacy utilities
├── main.py           # End-to-end experiment entry point
├── models.py         # Neural dynamics predictor with MC-dropout uncertainty
├── paper_experiments.py          # Main five-seed manuscript experiments
├── paper_extended_experiments.py # Extended ablation and diagnostic experiments
├── data_scarcity_experiment.py   # Local-data scarcity and unequal-coverage checks
├── participation_experiment.py   # Partial-client-participation stress test
├── regenerate_paper_figures.py   # Rebuild figures from saved experiment tables
├── utils.py          # Vehicle model, environment, metrics, and plotting utilities
├── requirements.txt  # Python dependencies
└── README.md
```

Generated outputs are written to `results/` by default and are intentionally ignored by Git.

## Installation

Python 3.8+ is recommended. A CUDA-enabled PyTorch installation is optional; the code also runs on CPU.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you need a CUDA-specific PyTorch wheel, install PyTorch first from the official selector, then install the remaining packages from `requirements.txt`.

## Usage

Run the default experiment pipeline:

```bash
python main.py
```

Run the larger full configuration:

```bash
python main.py --full
```

Reproduce the manuscript-facing five-seed experiments:

```bash
python paper_experiments.py
python paper_extended_experiments.py
python data_scarcity_experiment.py
```

The manuscript scripts write generated CSV and vector-PDF artifacts to `results/`.
The stored `exp_metadata.json` records the fixed seed list, hardware metadata,
and plant parameters. The published paper directory retains the exact tables
used for the submitted manuscript.

The default mode is intended for reproducible development runs. The full mode increases the horizon, number of seeds, federated rounds, local epochs, hidden dimension, and MC-dropout samples.

## Experiments

The pipeline in `main.py` executes:

1. Synthetic heterogeneous vehicle-data generation.
2. Federated training across multiple FL optimizers.
3. Closed-loop MPC comparison against classical and learning-based baselines.
4. Federated training-dynamics analysis.
5. Robustness evaluation under disturbances and model mismatch.
6. Matched ablation of per-vehicle local learning, federated learning, and uncertainty feedback.
7. Parameter-sensitivity sweeps.
8. Gradient-noise stress testing without claiming a formal privacy certificate.
9. Partial-client-participation and controller-call latency measurements.

## Citation

If this repository is useful for your research, please cite the paper:

```bibtex
@article{fedrmpc2026,
  title={FedRMPC: Federated Uncertainty-Aware Predictive Control for Heterogeneous Internet-of-Vehicles Systems},
  author={Zhou, Yuanqiang and Yin, Zilong and Zhu, Yan-Ran and Wang, Haoyu and Li, Dewei and Gao, Furong and Chen, Hong},
  year={2026}
}
```
