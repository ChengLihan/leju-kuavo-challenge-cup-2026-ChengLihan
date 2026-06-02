# 2026 挑战杯 Kuavo 仿真赛环境

本仓库包含 **2026 挑战杯 Kuavo 人形机器人仿真赛** 的 MuJoCo 仿真环境、三类任务场景、选手任务模板和常用接口。

推荐选手主要修改：

```text
src/challenge_cup_task_template/scripts/challenge_task.py
```

仿真环境、场景生成、随机化、反作弊监控和计时器由 `challenge_cup_simulator` 自动处理。

## 环境搭建

### 1. 导入 Docker 镜像

比赛镜像名：

```text
kuavo_challenge_cup_2026:latest
```

下载镜像压缩包：

```bash
wget https://kuavo.lejurobot.com/challenge_cup_2026/kuavo_challenge_cup_2026_latest.tar.gz
```

导入镜像：

```bash
docker load -i kuavo_challenge_cup_2026_latest.tar.gz
```

> 当前启动脚本不会自动拉取旧镜像；本地没有 `kuavo_challenge_cup_2026:latest`
> 时会直接报错。

### 2. 启动 Docker 容器

在仓库根目录执行：

```bash
./docker/run_with_gpu_for_challenge.sh
```

脚本会：

- 使用 `kuavo_challenge_cup_2026:latest` 镜像；
- 自动设置 `ROBOT_VERSION=52`；
- 将当前仓库挂载到容器内 `/root/kuavo_ws`；
- 使用 host 网络和本机 X11 显示 MuJoCo 窗口。

### 3. 编译（容器内执行）

```bash
cd /root/kuavo_ws
catkin config -DCMAKE_ASM_COMPILER=/usr/bin/as -DCMAKE_BUILD_TYPE=Release
source installed/setup.zsh
catkin build kuavo_msgs challenge_cup_simulator challenge_cup_task_template humanoid_controllers
source devel/setup.zsh
```

如果工作空间已经编译过，通常只需要：

```bash
source devel/setup.zsh
```

## 快速启动

推荐通过统一任务入口启动仿真。它会自动完成场景生成、完整性校验、随机场景初始化、反作弊监控、计时器启动和 ROS 节点初始化。

```bash
# 场景一：快递称重与摆放
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3

# 场景二：零件分拣归档
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3

# 场景三：SMT 料盘出库
rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
```

设置比赛限时：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --time-limit 120
```

`--time-limit` 的单位是秒。设置后，到时会自动结束当前任务节点；不设置时长时只显示用时，不自动结束。

不弹出计时器窗口：

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --time-limit 120 --no-timer-gui
```

计时器窗口中的 `Stop Timer` 只冻结计时显示，便于裁判在任务完成后查看用时，不会杀死任务。

## 接口文档

> 实物机器人接口可参考：[接口使用文档](https://kuavo.lejurobot.com/manual/basic_usage/kuavo-ros-control/docs/4%E5%BC%80%E5%8F%91%E6%8E%A5%E5%8F%A3/%E6%8E%A5%E5%8F%A3%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3/)，该文档只作参考。
> 挑战杯接口以下方仿真环境实际发布的话题、服务和本仓库说明为准。

常用控制、传感器、夹爪和任务入口接口见下表：

| 文档 | 内容 |
| --- | --- |
| [运动控制接口文档](docs/运动控制API.md) | 行走、手臂、头部、夹爪、传感器和状态话题/服务说明。 |
| [基础 ROS 话题说明](docs/readme_topics.md) | OCS2 状态/控制向量、核心话题命名和调试话题说明。 |
| [RL 控制框架 ROS 接口文档](docs/RL控制框架ROS接口文档.md) | 控制器切换、控制器状态查询和 RL 相关服务/话题。 |
| [选手任务模板说明](src/challenge_cup_task_template/README.md) | `challenge_task.py` 参数、seed、计时器、受保护启动流程。 |
| [仿真器说明](src/challenge_cup_simulator/README.md) | 底层 launch 参数、场景生成、LiDAR 和 simulator 维护脚本。 |

下方“本地常用话题与服务”只列比赛中最常用的接口速查；完整字段、消息格式和调用示例以接口文档为准。

## 比赛场景

### 场景一：快递称重与摆放

场景内设置桌面、称重区域、分拣箱和 4 个快递盒。机器人需要抓取快递、完成称重相关操作，并将快递放入指定区域。

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 0
```

### 场景二：零件分拣归档

场景内设置三类零件，共 6 个：

- `part_type_a_*`：A 类零件；
- `part_type_b_*`：B 类零件；
- `part_type_c_*`：C 类零件，螺丝刀模型。

机器人需要识别并抓取零件，将其放入对应收纳区域。

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 0
```

### 场景三：SMT 料盘出库

场景内设置 SMT 料盘货架、5 个可抓取料盘和出库区域。机器人需要从指定库位取出料盘并放置到出库区域。

```bash
rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 0
```

### 随机种子

`--seed` 用于选择场景实例。正式评测 seed 由组委会指定；选手本地可以用任意 seed 测试代码鲁棒性。

> 注意：挑战杯 seed 只用于场景实例随机化，不随机机器人初始位姿。具体随机规则由受保护模块处理，选手不需要依赖具体规则写死策略。

## 本地常用话题与服务

可通过以下命令查看当前实际话题：

```bash
rostopic list
rosservice list
```

### 控制接口

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/Twist` | 机器人底盘速度指令，常用 `linear.x`、`linear.y`、`angular.z`。 |
| `/kuavo_arm_traj` | `sensor_msgs/JointState` | 手臂轨迹控制接口，左臂 7 关节 + 右臂 7 关节。 |
| `/kuavo_arm_target_poses` | `kuavo_msgs/armTargetPoses` | 手臂目标轨迹接口，键盘控制脚本中使用。 |
| `/cmd_pose` | `geometry_msgs/Twist` | 躯干/姿态增量控制接口，部分调试脚本使用。 |

### 夹爪接口

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/control_robot_leju_claw` | `kuavo_msgs/controlLejuClaw` service | 二指夹爪服务接口。`position` 为闭合百分比，`0` 表示张开，`100` 表示闭合。 |
| `/leju_claw_command` | `kuavo_msgs/lejuClawCommand` | 二指夹爪命令话题。 |
| `/leju_claw_state` | `kuavo_msgs/lejuClawState` | 二指夹爪状态话题。 |
| `/gripper/state` | `sensor_msgs/JointState` | 仿真底层夹爪关节状态。 |

### 传感器接口

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/sensors_data_raw` | `kuavo_msgs/sensorsData` | 仿真原始传感器数据，包含 `sensor_time`、关节、IMU、末端执行器信息。 |
| `/lidar/points` | `sensor_msgs/PointCloud2` | Mid360 点云。 |
| `/livox/lidar` | `livox_ros_driver2/CustomMsg` | Livox 格式点云输出，供需要该格式的算法使用。 |
| `/lidar_imu` | `sensor_msgs/Imu` | 与 LiDAR 时间对齐的 IMU。 |
| `/tf`、`/tf_static` | `tf2_msgs/TFMessage` | TF 树。 |

### 相机接口

默认发布压缩图像话题：

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/cam_h/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 头部 RGB 摄像头。 |
| `/cam_l/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 左腕 RGB 摄像头。 |
| `/cam_r/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 右腕 RGB 摄像头。 |
| `/cam_h/depth/image_raw/compressedDepth` | `sensor_msgs/CompressedImage` | 头部深度图。 |
| `/cam_l/depth/image_rect_raw/compressedDepth` | `sensor_msgs/CompressedImage` | 左腕深度图。 |
| `/cam_r/depth/image_rect_raw/compressedDepth` | `sensor_msgs/CompressedImage` | 右腕深度图。 |
| `/cam_h/color/camera_info` | `sensor_msgs/CameraInfo` | 头部相机内参。 |
| `/cam_l/color/camera_info` | `sensor_msgs/CameraInfo` | 左腕相机内参。 |
| `/cam_r/color/camera_info` | `sensor_msgs/CameraInfo` | 右腕相机内参。 |

如果确实需要 raw 图像，可直接启动 simulator launch 时传入：

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch raw_image:=true
```

使用统一任务入口时默认保持压缩图像，以降低 CPU 压力。

### 禁止使用的上帝视角接口

以下接口属于仿真真值或内部初始化接口，比赛中禁止选手节点读取或调用：

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/mujoco/qpos` | topic | MuJoCo 全量状态真值。 |
| `/ground_truth/state` | topic | 真值状态。 |
| `/set_object_position` | service | 内部物体摆放服务。 |
| `/set_object_position_lock` | service | 内部物体摆放锁服务。 |

反作弊监控会检测非白名单节点订阅或调用这些接口；触发后会输出违规原因并杀掉对应 ROS 节点。

维护侧可用下面脚本测试监控是否生效：

```bash
rosrun challenge_cup_simulator forbidden_topic_subscriber.py
```

该脚本默认订阅 `/mujoco/qpos`，正常情况下应被反作弊监控杀掉。

## 场景生成与受保护初始化

场景 XML 由 YAML 配置生成：

```text
src/challenge_cup_simulator/config/scenes/scene1.yaml
src/challenge_cup_simulator/config/scenes/scene2.yaml
src/challenge_cup_simulator/config/scenes/scene3.yaml
```

手动重新生成场景：

```bash
rosrun challenge_cup_simulator scene_builder.py --all
```

统一任务入口启动时，会生成临时基准 XML：

```text
src/challenge_cup_simulator/models/biped_s52/xml/_scene_<scene>_active.xml
```

真实随机位置由受保护 `.so` 在运行时写入 MuJoCo 内存，不写入可读 XML 文件。

## 计时器

计时器使用 `/sensors_data_raw.sensor_time`，即仿真时间，不受电脑性能、仿真卡顿或实时率变化影响。

```bash
# 通过任务入口自动启动，120 秒到时结束任务
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3 --time-limit 120

# 已经启动仿真时，单独查看计时器
rosrun challenge_cup_simulator sim_timer.py --time-limit 120
```

## 选手代码位置

选手推荐在 `challenge_task.py` 中按场景分支实现任务逻辑：

```text
src/challenge_cup_task_template/scripts/challenge_task.py
```

模板脚本中已给出常用 publisher 和 TODO 区域。更多说明见：

```text
src/challenge_cup_task_template/README.md
src/challenge_cup_simulator/README.md
```

## 严禁事项

比赛中禁止以下行为；触发检测或人工确认后，可取消对应成绩：

1. 直接读取仿真真值，如 `/mujoco/qpos`、`/ground_truth/state`；
2. 调用内部物体摆放服务，如 `/set_object_position`、`/set_object_position_lock`；
3. 修改 `challenge_cup_simulator` 中的场景、模型、启动器、受保护模块或随机化相关文件；
4. 通过非物理方式移动物体、移动机器人或干预仿真状态；
5. 依赖正式评测 seed 的具体数值写死策略。

## 代码提交规范

> 本节需要根据组委会最终通知补充提交入口、截止时间和压缩包命名要求。

建议提交内容至少包含：

```text
参赛队伍名称/
└── challenge_cup_task_template/
    ├── package.xml
    ├── CMakeLists.txt
    ├── scripts/
    │   └── challenge_task.py
    └── README.md
```

如修改了其他功能包，请在提交包 README 中说明：

- 修改了哪些包；
- 修改原因；
- 编译方法；
- 运行方法；
- 是否引入额外依赖。

## 相关文档

- [选手任务模板说明](src/challenge_cup_task_template/README.md)
- [仿真器说明](src/challenge_cup_simulator/README.md)
- [基础 ROS 话题说明](docs/readme_topics.md)
- [运动控制接口文档](docs/运动控制API.md)
- [RL 控制框架 ROS 接口文档](docs/RL控制框架ROS接口文档.md)
