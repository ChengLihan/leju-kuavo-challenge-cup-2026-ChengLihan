# Challenge Cup Simulator

`challenge_cup_simulator` 提供挑战杯仿真赛的 MuJoCo 场景、模型、传感器节点、夹爪仿真接口和底层 launch 文件。

选手和正式评测推荐从任务模板启动：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
```

该入口会自动完成：

- 场景 XML 生成；
- 受保护文件完整性校验；
- seed 对应的随机场景初始化；
- 物体摆放服务上锁；
- 反作弊监控；
- 比赛计时器启动。

## 底层调试启动

只调试 simulator 本身时，可以直接启动 launch：

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch scene_name:=scene1
```

可切换场景：

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch scene_name:=scene2
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch scene_name:=scene3
```

注意：直接 `roslaunch` 只加载静态 XML，不会执行 seed 随机摆放、完整性校验、反作弊监控和计时器。比赛流程请使用 `challenge_task.py`。

常用 launch 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `scene_name` | `scene1` | 加载 `scene1` / `scene2` / `scene3`。 |
| `raw_image` | `false` | 是否额外发布 raw 图像；默认只保留压缩图像，降低 CPU 压力。 |
| `mujoco_vsync` | `true` | MuJoCo 窗口垂直同步，采集图像时建议保持开启。 |
| `enable_lidar` | `true` | 是否启动 Mid360 LiDAR 仿真节点。 |
| `lidar_backend` | `cpu` | LiDAR 射线追踪后端。 |
| `lidar_output_type` | `pointcloud2` | `pointcloud2` / `livox` / `both`。 |
| `with_estimation` | `true` | 是否启动状态估计 nodelet。 |
| `use_rosbag_nodelet` | `false` | 是否加载 rosbag nodelet。 |

示例：

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch \
  scene_name:=scene2 raw_image:=false enable_lidar:=true lidar_backend:=cpu
```

## 场景生成

场景配置文件：

```text
config/scenes/scene1.yaml
config/scenes/scene2.yaml
config/scenes/scene3.yaml
```

重新生成全部静态基准场景：

```bash
rosrun challenge_cup_simulator scene_builder.py --all
```

生成结果位于：

```text
models/biped_s52/xml/scene1.xml
models/biped_s52/xml/scene2.xml
models/biped_s52/xml/scene3.xml
```

统一任务入口启动时还会生成临时文件：

```text
models/biped_s52/xml/_scene_<scene>_active.xml
```

该文件只保存基准场景，真实随机位置由受保护模块在运行时写入 MuJoCo 内存。

## 计时器

任务入口会自动启动计时器：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --time-limit 120
```

`--time-limit` 单位是秒。设置后，到时会结束当前任务节点；不设置时只显示用时，不自动结束。

仿真已经启动时，也可以单独查看计时器：

```bash
rosrun challenge_cup_simulator sim_timer.py --time-limit 120
rosrun challenge_cup_simulator sim_timer.py --time-limit 120 --no-gui
```

计时器使用 `/sensors_data_raw.sensor_time`，按仿真时间计时。窗口里的 `Stop Timer` 只冻结显示，便于裁判查看完成用时。

## LiDAR

Mid360 仿真节点默认随 launch 启动，发布：

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/lidar/points` | `sensor_msgs/PointCloud2` | 默认点云输出。 |
| `/livox/lidar` | `livox_ros_driver2/CustomMsg` | `lidar_output_type:=livox` 或 `both` 时发布。 |
| `/lidar_imu` | `sensor_msgs/Imu` | 与 LiDAR 时间对齐的 IMU。 |

如环境缺少依赖，在容器内安装：

```bash
pip3 install --no-deps -r src/challenge_cup_simulator/requirements.txt
```

## 主要脚本

| 文件 | 作用 |
| --- | --- |
| `utils/scene_builder.py` | 根据 YAML 生成 MuJoCo XML。 |
| `utils/challenge_sim_launcher.py` | 受保护启动器，供任务模板调用。 |
| `scripts/sim_timer.py` | 比赛计时器包装脚本。 |
| `utils/lidar_mid360_node.py` | Mid360 LiDAR 仿真节点。 |
| `scripts/sim_leju_claw_interface.py` | 仿真夹爪服务/话题桥接。 |
| `scripts/forbidden_topic_subscriber.py` | 维护侧测试反作弊监控的违规订阅脚本。 |

## 场景概览

- `scene1`：快递包裹称重与摆放。
- `scene2`：三类零件分拣归档。
- `scene3`：SMT 料盘出库。

更完整的比赛入口说明见仓库根目录 `readme.md` 和 `src/challenge_cup_task_template/README.md`。
