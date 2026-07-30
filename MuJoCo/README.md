# MuJoCo six-controller trajectory comparison

This folder reproduces the six closed-loop controller trajectories in a common
MuJoCo intersection scene:

1. Linear MPC
2. Tube MPC
3. Stochastic MPC
4. GP-MPC
5. Robust MPC
6. FedRMPC

The four vehicles use the same starts, goals, obstacle map, control period, and
MuJoCo plant for every controller. Vehicle-to-vehicle contacts are disabled to
match the paper's independent four-vehicle rollout protocol; vehicle-obstacle
contacts remain active. Each controller reads state from MuJoCo and its action
is converted to longitudinal/lateral force and yaw torque. The saved trail is
drawn into the MuJoCo render scene before pixel capture.

## Run

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' .\run_six_algorithms.py
```

Run one method for diagnosis:

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' .\run_six_algorithms.py --method 'FedRMPC'
```

Rebuild only the final 1×6 layout after typography changes:

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' .\run_six_algorithms.py --rebuild-figure
```

## Outputs

- `outputs/mujoco_six_trajectories_1x6.png`: publication-resolution 1×6 figure.
- `outputs/mujoco_six_trajectories_1x6.pdf`: vector container with MuJoCo raster panels.
- `outputs/mujoco_six_trajectories_1x6_updated.pdf`: rebuilt PDF fallback when a Windows previewer locks the standard PDF.
- `outputs/final_frames/*.png`: the six individual MuJoCo window captures.
- `outputs/*_trajectories.json`: state trajectories measured from MuJoCo.
- `outputs/metrics.json`: per-method success, collision, clearance, and step summary.
- `scene_seed123.xml`: exact MuJoCo scene used by the completed run.

The lightweight MC-dropout dynamics network is trained inside this runner with
four heterogeneous clients and proximal federated aggregation, avoiding an
extra PyTorch installation in the existing `E:\MuJoCo\runtime` environment.
