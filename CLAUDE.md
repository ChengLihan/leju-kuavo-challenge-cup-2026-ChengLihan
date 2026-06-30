# CLAUDE.md

## 项目概述

2026 挑战杯 Kuavo 人形机器人仿真赛 — Scene2 零件分拣归档（YOLO 方案）

- **仓库**: https://github.com/ChengLihan/leju-kuavo-challenge-cup-2026-ChengLihan
- **当前分支**: `scene2_yolo`（黄文珂的 YOLO 检测方案）
- **容器镜像**: `kuavo_challenge_cup_2026:latest`
- **容器名**: `kuavo_challenge_container_GPU_82d3b07d`
- **选手核心代码**: `src/challenge_cup_task_template/scripts/challenge_task.py`

## 环境搭建（已完成）

### Docker 镜像
```bash
wget https://kuavo.lejurobot.com/challenge_cup_2026/kuavo_challenge_cup_2026_latest.tar.gz
docker load -i kuavo_challenge_cup_2026_latest.tar.gz
```

### 容器内已安装的额外依赖
- torch 2.2.2+cu121 (GPU)
- torchvision 0.17.2
- ultralytics 8.4.80（与模型训练版本 8.4.70 兼容）
- ros-noetic-cv-bridge, ros-noetic-tf2-ros
- tqdm, thop, pandas, seaborn

### 系统库路径（已写入 ldconfig）
- `/opt/drake/lib`
- `/opt/ros/noetic/lib`
- `/root/kuavo_ws/devel/lib`
- `/root/kuavo_ws/installed/lib`

### 修复过的坑
- `libplantIK.so` 缺失 → 从 `humanoid-control/humanoid_arm_control/scripts/motion_capture_ik_packaged/lib/` 拷到 `manipulation_nodes/motion_capture_ik/lib/`
- 场景二 YOLO 模型 (`models/yolo/best.pt`) 用 ultralytics 8.4.70 训练，必须用 8.4.x 版本加载

## 日常启动流程

```bash
# 每次重启电脑后：
xhost +                                              # 允许 X11 连接
docker start kuavo_challenge_container_GPU_82d3b07d   # 启动容器
docker exec -it kuavo_challenge_container_GPU_82d3b07d zsh  # 进入容器
```

容器内：
```bash
cd /root/kuavo_ws
source installed/setup.zsh
source devel/setup.zsh
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3 --time-limit 120
```

## 代码结构

```
challenge_task.py  (~2500 行，主文件)
├── RobotBase        — 底层控制封装（行走 /cmd_vel、手臂 /kuavo_arm_traj、夹爪、头部）
├── Perception       — 视觉感知（YOLO 分割 + 深度相机 + TF 坐标变换）
├── Navigation       — 导航到目标位置
├── Manipulation     — 抓取/放置动作
├── Scene1Controller — 场景一逻辑
├── Scene2Controller — 场景二逻辑（当前焦点）
└── Scene3Controller — 场景三逻辑
```

### 辅助文件
- `scene2_yolo_detector.py` — YOLO 模型加载 + 推理
- `screwdriver_demo_detector.py` — 螺丝刀检测 demo
- `models/yolo/best.pt` — 训练好的 YOLOv8-seg 模型 (24MB)

## Scene2 管线
```
头部相机 → YOLO 分割检测(3类零件) → 深度图获取距离 → TF变换到机器人坐标系
  → 导航走过去 → 腕部相机二次校准 → 抓取 → 走到对应箱子 → 放置 → 下一个
```

三类零件 → 箱子映射：
| 零件 | 英文 | 箱子 |
|------|------|------|
| 螺丝刀 | screwdriver | purple_bin |
| 管夹 | pipe_clamp | blue_bin |
| 管接头 | pipe_fitting | orange_bin |

## 关键 ROS 接口

| 接口 | 类型 | 用途 |
|------|------|------|
| `/cmd_vel` | Twist | 行走速度控制 |
| `/kuavo_arm_traj` | JointState | 手臂轨迹（14 关节） |
| `/control_robot_leju_claw` | Service | 夹爪控制（0=张开, 100=闭合） |
| `/cam_h/color/image_raw/compressed` | CompressedImage | 头部 RGB 相机 |
| `/cam_h/depth/image_raw/compressedDepth` | CompressedImage | 头部深度图 |
| `/sensors_data_raw` | sensorsData | 关节/IMU 状态 |
| `/lidar/points` | PointCloud2 | 激光雷达点云 |
| `/humanoid_controller/switch_controller` | Service | MPC ↔ RL 控制器切换 |

## 严禁使用的接口（反作弊监控）
- `/mujoco/qpos` — MuJoCo 真值
- `/ground_truth/state` — 真值状态
- `/set_object_position` — 物体摆放服务

## 编译命令
```bash
# 只编译需要的包（跳过 motion_capture_ik 等有问题的）
catkin config --skiplist motion_capture_ik
catkin build challenge_cup_simulator challenge_cup_task_template
```

## pip 安装规范
- **必须用清华镜像**: `-i https://pypi.tuna.tsinghua.edu.cn/simple`
- Python 版本是 3.8（容器内），注意兼容性

## GPU
- RTX 4060 Max-Q / Mobile, Driver 595.71.05, CUDA 13.2
- torch 确认 CUDA 可用: `torch.cuda.is_available()` → True

## 其他分支
- `master` — 官方三场景模板（空 TODO）
- `scene2_yolo` — 当前分支
- `scene3_custom_dataset_capturer` — 场景三数据集采集
- `scene3_manual_decision_navigation` — 场景三手动导航
