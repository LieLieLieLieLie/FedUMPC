# FedRMPC

Official implementation of **FedRMPC: Federated Robust Model Predictive Control via Uncertainty and Proximal Regularization**.

FedRMPC is a safety-oriented multi-vehicle control framework that combines federated learning, Bayesian neural-network dynamics modeling, epistemic uncertainty quantification, and robust model predictive control. The implementation evaluates a four-vehicle intersection scenario under heterogeneous dynamics, non-IID local data, disturbances, ablations, sensitivity sweeps, and differential-privacy noise.

## Highlights

- Federated BNN dynamics learning with FedAvg, FedProx, SCAFFOLD, MOON, DP-FedAvg, and DP-FedProx variants.
- Uncertainty-aware MPC using Monte Carlo dropout to penalize high-risk predictions and adapt safety margins.
- Baselines including Linear MPC, Tube MPC, Stochastic MPC, GP-MPC, and Robust MPC.
- End-to-end experiment pipeline for closed-loop comparison, FL convergence, robustness, ablation, sensitivity, privacy-utility, and runtime reporting.
- Publication-ready figures and spreadsheets exported automatically.

## Repository Structure

```text
.
├── config.py         # Global simulation, MPC, FL, BNN, DP, and plotting parameters
├── controllers.py    # MPC baselines and FedRMPC controller
├── federated.py      # Federated clients, server aggregation, and privacy utilities
├── main.py           # End-to-end experiment entry point
├── models.py         # Bayesian neural network with MC-dropout uncertainty
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

The default mode is intended for reproducible development runs. The full mode increases the horizon, number of seeds, federated rounds, local epochs, hidden dimension, and MC-dropout samples.

## Experiments

The pipeline in `main.py` executes:

1. Synthetic heterogeneous vehicle-data generation.
2. Federated training across multiple FL optimizers.
3. Closed-loop MPC comparison against classical and learning-based baselines.
4. Federated training-dynamics analysis.
5. Robustness evaluation under disturbances and model mismatch.
6. Ablation of federated learning and uncertainty regularization.
7. Parameter-sensitivity sweeps.
8. Privacy-utility comparison for DP-FedAvg and DP-FedProx.
9. Runtime summary exported to `results/timing_report.xlsx`.

## Citation

If this repository is useful for your research, please cite the paper:

```bibtex
@article{fedrmpc2026,
  title={FedRMPC: Federated Robust Model Predictive Control via Uncertainty and Proximal Regularization},
  author={Zhou, Yuanqiang and Yin, Zilong and Zhu, Yan-Ran and Wang, Haoyu and Li, Dewei and Gao, Furong and Chen, Hong},
  year={2026}
}
```

## License

This repository follows the license file included in the project.
