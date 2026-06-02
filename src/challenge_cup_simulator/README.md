# Challenge Cup Simulator

本包提供挑战杯仿真赛使用的 Kuavo `biped_s52` MuJoCo 场景、模型与启动工具。

## 启动仿真

推荐通过选手任务模板启动，它会自动完成场景生成、完整性校验、随机场景初始化、反作弊监控和计时器启动：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
```

也可以直接启动 simulator launch 文件：

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch
```

`with_estimation` 默认是 `false`，这样即使工作空间里没有 `humanoid_estimation` 也能启动。
如果当前环境已包含该包，可以显式打开：

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch with_estimation:=true
```

CRAIC rosbag nodelet include 也是可选的，默认关闭。只有当前环境包含
`humanoid_interface` 时才建议打开：

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch use_rosbag_nodelet:=true
```

默认加载场景：

```text
models/biped_s52/xml/scene1.xml
```

切换场景：

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch scene_name:=scene2
```

## 场景生成

场景 XML 由 YAML 场景描述生成，流程与 CRAIC simulator 类似：

```bash
rosrun challenge_cup_simulator scene_builder.py
```

生成全部比赛场景：

```bash
rosrun challenge_cup_simulator scene_builder.py --all
```

也可以从源码目录直接运行：

```bash
python3 src/challenge_cup_simulator/utils/scene_builder.py
```

YAML 源文件：

```text
config/scenes/scene1.yaml
config/scenes/scene2.yaml
config/scenes/scene3.yaml
```

## 比赛计时器

推荐通过选手任务模板启动计时器：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --time-limit 120
```

仿真已经启动后，也可以单独运行计时器做本地查看或验证：

```bash
rosrun challenge_cup_simulator sim_timer.py --time-limit 120
rosrun challenge_cup_simulator sim_timer.py --time-limit 120 --no-gui
```

计时单位是秒，`--time-limit 120` 表示 120 秒。计时器使用 `/sensors_data_raw`
中的仿真时间作为计时基准，因此仿真暂停、卡顿或实时率变化不会改变比赛用时口径。
设置 `--time-limit` 后，到时会自动结束当前任务节点；不设置时长时只显示用时，不自动结束。
单独的 `sim_timer.py` 只是薄包装；正式启动链路由受保护启动器自动拉起计时器。
计时器窗口支持单次 `Stop Timer`，只冻结计时显示，便于比赛完成后由裁判查看用时。

## LiDAR

模拟 mid360 LiDAR 使用 PyPI 的 `mujoco-lidar` 包生成扫描模式。
启动前在 Docker 工作空间中安装依赖：

```bash
pip3 install --no-deps -r src/challenge_cup_simulator/requirements.txt
```

`--no-deps` 会跳过本任务不需要的可视化可选依赖。默认 launch 使用 CPU 后端，并发布
`/lidar/points`、`/lidar_imu` 和 `/mujoco/qpos`：

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch enable_lidar:=true lidar_backend:=cpu
```

## 场景概览

- `scene1`：快递包裹称重与摆放。
- `scene2`：三类零件分拣归档。
- `scene3`：SMT 料盘出库。
- 机器人默认从桌前起步，面向操作区域。
