# MuJoCo Simulation for FedRMPC

This directory contains the cleaned MuJoCo 3.1.1 simulation used to produce the
paper's seed-123 cross-simulator deployment figure and an optional sequential
four-vehicle demonstration. The local working copy retains trajectory/metric records
and the six official-Viewer frames used to assemble the paper figure, whereas Git
excludes generated outputs so the public repository remains source-focused. The
scripts rebuild these artifacts from the documented seed and scenario.

# 基线对比与四车依次行动演示

本目录是合并后的唯一 MuJoCo 仿真目录，包含论文使用的六算法独立闭环对比（`run_six_algorithms.py`）、Fig. 4 官方 Viewer 帧采集脚本（`capture_official_viewer_frames.pyw`）、与场景障碍物对齐的热力图生成脚本（`generate_uncertainty_heatmap.py`），以及顺序调度演示（`run_sequential_vehicles.py`）。顺序演示中四辆车不再同时行动，而是严格按照下面的顺序运行：

```text
A车到达目的地（或因静态障碍物碰撞而终止）
    ↓
B车开始
    ↓
C车开始
    ↓
D车开始
```

等待中的车辆保持在起点不动，已经完成的车辆保持在终点。每辆车交接前保留约1秒静止画面，便于演示时看清车辆切换。

该版本用于直观演示顺序通行，不替换论文中原有的四车独立并行轨迹实验，也不作为新的多车协同实验结果。

## 1. 最简单的运行方法

直接双击：

```text
run_fedrmpc_sequential.cmd
```

第一次运行会执行顺序MuJoCo仿真并保存时间线，所需时间相对较长。以后再次运行会直接读取缓存并打开平滑回放窗口。

## 2. CMD运行方式

进入目录：

```bat
cd /d D:\PythonProject\MPC\FedRMPC\MuJoCo
```

运行FedRMPC四车顺序演示：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_sequential_vehicles.py"
```

不切换目录，使用完整路径：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "D:\PythonProject\MPC\FedRMPC\MuJoCo\run_sequential_vehicles.py"
```

强制重新执行顺序MuJoCo物理仿真：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_sequential_vehicles.py" --recompute
```

只重新计算并保存数据，不打开窗口：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_sequential_vehicles.py" --recompute --no-viewer
```

## 3. PowerShell运行方式

进入目录：

```powershell
Set-Location 'D:\PythonProject\MPC\FedRMPC\MuJoCo'
```

运行FedRMPC四车顺序演示：

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' '.\run_sequential_vehicles.py'
```

不切换目录，使用完整路径：

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' 'D:\PythonProject\MPC\FedRMPC\MuJoCo\run_sequential_vehicles.py'
```

强制重新执行顺序MuJoCo物理仿真：

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' '.\run_sequential_vehicles.py' --recompute
```

## 4. 窗口操作

窗口使用带左右菜单的MuJoCo传统Viewer：

- `R`：从A车开始重新播放；
- 空格：暂停或继续；
- `+`：逐级加速，最高8倍速；
- `-`：逐级减速，最低0.25倍速；
- 鼠标左键拖动：旋转视角；
- 鼠标右键拖动：平移视角；
- 鼠标滚轮：缩放；
- 关闭窗口：结束程序。

顺序回放结束后窗口会停留在最终状态。此时按`R`即可重新播放，无需重新输入命令。

启动时可以直接指定倍速。例如CMD以2倍速运行：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_sequential_vehicles.py" --speed 2
```

PowerShell以4倍速运行：

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' '.\run_sequential_vehicles.py' --speed 4
```

支持`0.25`、`0.5`、`1`、`2`、`4`、`8`等倍速。窗口打开后仍可用键盘`+/-`动态调整。

## 5. 运行其他控制算法

例如运行Tube MPC：

CMD：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_sequential_vehicles.py" --method "Tube MPC"
```

PowerShell：

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' '.\run_sequential_vehicles.py' --method 'Tube MPC'
```

可选算法名称为：

```text
Linear MPC
Tube MPC
Stochastic MPC
GP-MPC
Robust MPC
FedRMPC
```

## 6. 输出文件

结果保存在`MuJoCo\outputs`：

- `fedrmpc_sequential_timeline.json`：四车顺序运动的完整时间线；
- `fedrmpc_sequential_metrics.json`：每辆车的成功、静态障碍物碰撞、步数和最小安全距离；
- `scene_seed123.xml`：本版本使用的MuJoCo场景。

## 7. 说明

顺序演示直接复用本目录`run_six_algorithms.py`中的车辆质量、动力学、障碍物、控制器和目标位置。改变的只有车辆启动顺序：同一时刻最多只有一辆车运动，因此不会再出现两辆运动中的车辆在交叉口相互穿过的画面。

论文 Fig. 4 的六算法对比由 `run_six_algorithms.py --no-render` 重新运行，保存的是 MuJoCo body-state 轨迹与接触指标；`--no-render` 只跳过依赖 OpenGL 上下文的截图，不绕过 MuJoCo 物理仿真。随后，`capture_official_viewer_frames.pyw` 在 Windows 桌面图形会话中把这些保存结果加载到 MuJoCo 官方 Viewer，并生成论文2×3跨栏图所使用的六张视口截图。截图与轨迹数据分开生成，避免显示层影响控制结果。FedRMPC 的配置选择可由 `tune_fedrmpc.py` 复现，13 组候选和最终排序保存在 `outputs\fedrmpc_tuning_seed123.json`。本轮选中配置在固定 seed 123 场景中获得 4/4 路线成功、0 次静态障碍物接触和 1.267 m 最小净空。
