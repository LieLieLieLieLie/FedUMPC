# FedRMPC的MuJoCo仿真与六算法轨迹对比

本目录用于在同一个MuJoCo十字路口场景中运行并比较以下6种控制算法：

1. Linear MPC
2. Tube MPC
3. Stochastic MPC
4. GP-MPC
5. Robust MPC
6. FedRMPC

四辆小车在不同算法下采用相同的起点、终点、障碍物地图、控制周期和MuJoCo动力学模型。控制器从MuJoCo读取车辆状态，并将转向和加速度指令转换为纵向力、横向力与偏航力矩。车辆与静态障碍物之间的碰撞检测保持开启；为了与论文中的独立四车轨迹评估设置一致，车辆之间的接触被关闭。

## 1. 环境路径

本项目默认使用下面的Python环境：

```text
E:\MuJoCo\runtime\Scripts\python.exe
```

主程序为：

```text
D:\PythonProject\MPC\FedRMPC\MuJoCo\run_six_algorithms.py
```

建议先确认解释器和MuJoCo是否可用。

### CMD检查命令

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" -c "import mujoco; print(mujoco.__version__)"
```

### PowerShell检查命令

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' -c "import mujoco; print(mujoco.__version__)"
```

正常情况下应输出MuJoCo版本，例如`3.11.0`。

## 2. 使用CMD运行

CMD的提示符通常显示为`D:\...>`。CMD命令中不要在Python路径前添加`&`，否则会出现“此时不应有 &”的错误。

### 2.1 进入仿真目录

```bat
cd /d D:\PythonProject\MPC\FedRMPC\MuJoCo
```

### 2.2 运行全部6种算法（离屏仿真）

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_six_algorithms.py"
```

该命令依次运行6种算法并保存 MuJoCo body-state 轨迹、接触统计和场景信息。在支持 OpenGL 离屏上下文的会话中，它还会生成旧版1×6检查图；当前论文 Fig. 4 不再使用该旧版检查图，而是使用第 2.6 节所述的官方 Viewer 截图组成2×3跨栏图。

如果当前会话不支持 OpenGL 离屏上下文，可使用纯仿真模式。该模式仍在 MuJoCo 中执行相同的闭环力/力矩控制并保存 body-state 轨迹与指标，只跳过最终窗口截图：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_six_algorithms.py" --no-render
```

### 2.3 只运行FedRMPC（离屏仿真）

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_six_algorithms.py" --method FedRMPC
```

### 2.4 在实时MuJoCo窗口中运行FedRMPC

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_six_algorithms.py" --method FedRMPC --viewer
```

也可以不切换目录，直接使用完整路径：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "D:\PythonProject\MPC\FedRMPC\MuJoCo\run_six_algorithms.py" --method FedRMPC --viewer
```

### 2.5 只重新生成旧版1×6检查图

如果仿真结果已经存在，只需调整字体或版式，可运行：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_six_algorithms.py" --rebuild-figure
```

### 2.6 从官方 MuJoCo Viewer 采集 Fig. 4 帧

先用 `run_six_algorithms.py --no-render` 确认六个算法的轨迹与指标已经存在，再在具有 Windows 桌面图形上下文的会话中运行：

```bat
"E:\MuJoCo\runtime\Scripts\pythonw.exe" "D:\PythonProject\MPC\FedRMPC\MuJoCo\capture_official_viewer_frames.pyw"
```

脚本会依次打开 MuJoCo 官方被动 Viewer，将同一次闭环仿真的最终 body state、障碍物和完整轨迹尾迹加载到场景中，并抓取六张无菜单、无鼠标光标的中心视口图像。输出保存在 `outputs\official_viewer_frames`。论文绘图脚本读取的镜像目录为 `paper\IoTJ_FedRMPC\data\Experiment\mujoco\official_viewer_frames`；更新截图后，应将六张 PNG 同步到该目录，再运行论文绘图脚本。该步骤只负责呈现已保存的仿真结果，不重新优化或人工修改轨迹。

## 3. 使用PowerShell运行

PowerShell的提示符通常显示为`PS D:\...>`。PowerShell执行带引号的程序路径时，需要在路径前使用调用运算符`&`。

### 3.1 进入仿真目录

```powershell
Set-Location 'D:\PythonProject\MPC\FedRMPC\MuJoCo'
```

### 3.2 运行全部6种算法（离屏仿真）

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' '.\run_six_algorithms.py'
```

无可用 OpenGL 离屏上下文时：

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' '.\run_six_algorithms.py' --no-render
```

### 3.3 只运行FedRMPC（离屏仿真）

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' '.\run_six_algorithms.py' --method 'FedRMPC'
```

### 3.4 在实时MuJoCo窗口中运行FedRMPC

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' '.\run_six_algorithms.py' --method 'FedRMPC' --viewer
```

也可以不切换目录，直接使用完整路径：

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' 'D:\PythonProject\MPC\FedRMPC\MuJoCo\run_six_algorithms.py' --method 'FedRMPC' --viewer
```

### 3.5 只重新生成1×6论文图

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' '.\run_six_algorithms.py' --rebuild-figure
```

## 4. 双击启动实时演示

如果不想输入命令，可以直接双击本目录中的：

```text
run_fedrmpc_viewer.cmd
```

它会自动切换到MuJoCo目录，并使用指定的Python环境启动FedRMPC实时仿真窗口。

## 5. 实时窗口说明

使用`--viewer`参数后，程序会读取已经由MuJoCo闭环仿真生成的轨迹，并在官方Viewer中按`0.01 s`物理步进行平滑回放。回放只改变显示方式，不改变论文轨迹数据。如果对应轨迹不存在，程序会先运行一次控制器并保存轨迹，然后再打开窗口。

窗口采用MuJoCo传统Viewer界面，左右两侧菜单均会显示。窗口打开后：

- 四辆小车会按照FedRMPC闭环仿真结果平滑运动；
- 鼠标左键拖动可以旋转视角；
- 鼠标右键拖动可以平移视角；
- 鼠标滚轮可以缩放视角；
- 按`R`键可以随时从头重新播放；
- 按空格键可以暂停或继续播放；
- 按`+`键可以逐级加速，按`-`键可以逐级减速；
- 左侧菜单可查看仿真状态、关节和执行器信息；
- 右侧菜单可调整几何体、接触点、阴影和其他可视化选项；
- 仿真结束后会保留最终状态；
- 关闭MuJoCo窗口后，程序才会返回命令提示符。

由于这是由外部Python程序驱动的被动Viewer，左右菜单中的模型重置操作不能代替回放重启。需要重新播放时应直接按键盘`R`，不需要关闭窗口或重新输入命令。

启动时也可以用`--speed`指定倍速。例如CMD中以2倍速打开：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_six_algorithms.py" --method FedRMPC --viewer --speed 2
```

PowerShell中以4倍速打开：

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' '.\run_six_algorithms.py' --method 'FedRMPC' --viewer --speed 4
```

允许的倍速范围为`0.25`到`8`。由于本窗口由Python外部回放驱动，MuJoCo菜单中的实时倍率不会替代`--speed`或键盘`+/-`设置。

平滑演示会在每个`0.01 s`物理子步后同步窗口，而原始闭环仿真的控制器更新周期为`0.10 s`。程序在相邻控制状态之间进行姿态插值，从而避免将10个物理步一次性显示所造成的跳动。

如果需要观察控制器一边计算、一边驱动MuJoCo，可将`--viewer`换成`--live-viewer`。由于FedRMPC的MPC候选轨迹求解会占用计算时间，该调试模式可能出现短暂停顿，不建议用于现场演示。

实时窗口模式必须与单个算法一起使用，例如：

```text
--method FedRMPC --viewer
```

不能只写`--viewer`，也不能将`--viewer`与`--rebuild-figure`同时使用。

## 6. 运行其他算法的实时窗口

将`FedRMPC`替换为其他算法名称即可。例如，在CMD中运行Tube MPC：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_six_algorithms.py" --method "Tube MPC" --viewer
```

在PowerShell中运行Tube MPC：

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' '.\run_six_algorithms.py' --method 'Tube MPC' --viewer
```

可用的算法名称必须严格写为：

```text
Linear MPC
Tube MPC
Stochastic MPC
GP-MPC
Robust MPC
FedRMPC
```

## 7. 输出文件

仿真结果保存在`MuJoCo\outputs`目录中：

- `mujoco_six_trajectories_1x6.png`：旧版高分辨率1×6轨迹检查图；
- `mujoco_six_trajectories_1x6.pdf`：旧版PDF轨迹检查图；
- `mujoco_six_trajectories_1x6_updated.pdf`：原PDF被预览器占用时生成的备用PDF；
- `final_frames\*.png`：6种算法各自的MuJoCo最终窗口截图；
- `official_viewer_frames\*.png`：论文 Fig. 4 使用的6张官方 MuJoCo Viewer 视口截图；
- `*_trajectories.json`：从MuJoCo车体位置读取的轨迹数据；
- `metrics.json`：成功数、碰撞数、安全距离和仿真步数等指标；
- `scenario.json`：起点、终点、随机种子和障碍物信息。
- `fedrmpc_tuning_seed123.json`：固定 seed 123 下 FedRMPC 有限参数搜索的全部候选、排序准则和选中配置。

场景对应的XML文件为：

```text
scene_seed123.xml
```

## 8. FedRMPC 配置审计与复现

旧实现曾对北向反向路线单独使用 `path_weight=6000`，而其余路线为 120；该方向专属常数缺乏理论依据，并造成黄色车辆明显绕行。本轮将相关项参数化，并在场景、BNN、控制候选集、预测时域和其他五个基线完全固定的条件下运行 13 组有限搜索：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "tune_fedrmpc.py"
```

搜索采用事先明确的字典序准则：先最大化成功路线数，再最小化静态障碍物接触数，然后最大化最小净空，最后比较平均步数与路径长度。选中配置为 `path_weight=120`、`northbound_path_weight=300`、`margin_offset=0.42` 和 `margin_scale=0.90`。在 seed 123 的四路线场景中，FedRMPC 达到 4/4 成功、0 次静态障碍物接触和 1.267 m 最小净空；旧配置对应 4/4、0 次和 0.635 m。完整搜索记录由脚本直接写入 `outputs\fedrmpc_tuning_seed123.json`，论文轨迹不是人工改线所得。

## 9. 常见问题

### “此时不应有 &”

原因是将PowerShell命令粘贴到了CMD中。CMD命令不要使用`&`；PowerShell命令需要使用`&`调用带引号的Python路径。

### 程序运行了但没有显示窗口

普通命令默认采用离屏渲染。需要增加：

```text
--method FedRMPC --viewer
```

### 窗口没有立即出现

如果已经存在`outputs\fedrmpc_trajectories.json`，窗口会直接打开。如果轨迹不存在，程序需要先完成一次仿真，请等待命令行出现`Traditional viewer opened`提示。

### 窗口中的车辆运动一卡一卡

请确认使用的是当前版本的`run_six_algorithms.py`并使用`--viewer`，不要使用`--live-viewer`。普通Viewer模式会读取已完成的MuJoCo轨迹并逐个同步`0.01 s`显示子步，因此比边求解边显示更适合演示。

### 仿真结束后如何重新播放

保持MuJoCo窗口处于当前选中状态，按键盘`R`即可从初始位置重新播放。播放过程中也可以按`R`立即回到开头。空格键用于暂停和继续，关闭窗口用于彻底结束程序。

### 为什么两辆小车有时会在画面中重叠

当前论文实验把四辆车作为共享地图中的四条独立闭环轨迹进行评估，车辆与静态障碍物的接触保持开启，但车辆之间的接触被关闭。因此，两车车体在交叉口重叠不会计入论文的静态障碍物碰撞指标，也不会产生物理碰撞响应。这一设置与论文正文中的实验范围一致，但不代表系统已经实现车辆之间的动态防碰撞或协同让行。Fig. 4 的 Viewer 截图在同一场景中同时显示四条独立轨迹，是为了便于方法间进行空间比较，而不是宣称实现了多车交互协调。若要启用真实的车辆间碰撞与避让，需要增加耦合多车约束或协调策略，并重新进行实验。

### 只用MuJoCo官方Viewer打开静态场景

下面的命令会以MuJoCo传统方式打开XML场景和左右菜单，但不会运行FedRMPC控制器，因此车辆不会按照算法轨迹运动。

CMD：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" -m mujoco.viewer --mjcf "D:\PythonProject\MPC\FedRMPC\MuJoCo\scene_seed123.xml"
```

PowerShell：

```powershell
& 'E:\MuJoCo\runtime\Scripts\python.exe' -m mujoco.viewer --mjcf 'D:\PythonProject\MPC\FedRMPC\MuJoCo\scene_seed123.xml'
```

### 只想快速检查窗口是否能打开

可以加入`--skip-bnn-training`，但该选项会跳过模型训练，只适合检查窗口和环境，不应作为论文结果或正式演示：

```bat
"E:\MuJoCo\runtime\Scripts\python.exe" "run_six_algorithms.py" --method FedRMPC --viewer --skip-bnn-training
```
