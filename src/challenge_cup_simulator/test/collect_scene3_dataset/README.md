# Scene3 上层料盘局部技能数采器

该目录用于 Scene3 上层 SMT 料盘“抓取、抽出、腰间收纳”的离线数据采集。脚本默认从机器人已经到达货架前工作位开始；如果加 `--navigate-to-shelf` 或 `--full-run`，也可以先执行一段数采用的货架前进导航，再开始局部抓取。它不做上下层决策、不做完整出库放箱任务，也不是正式评测推理脚本。

## 入口

如果终端是 `zsh`，优先使用：

```bash
cd /home/cheng/kuavo_ws
source devel/setup.zsh
```

如果终端是 `bash`，使用：

```bash
cd /home/cheng/kuavo_ws
source devel/setup.bash
```

不要在 `zsh` 里 source `setup.bash`。如果看到 `/root/kuavo_ws/setup.sh` 之类路径错误，通常说明当前是 root shell 或 source 文件和 shell 类型不匹配。

打印固定 named-pose 专家轨迹：

```bash
python3 src/challenge_cup_simulator/test/collect_scene3_dataset/collect_scene3_tray_dataset.py \
  --seed 0 \
  --slot upper \
  --print-plan
```

连接已有 Scene3 仿真并采一条 rosbag：

```bash
python3 src/challenge_cup_simulator/test/collect_scene3_dataset/collect_scene3_tray_dataset.py \
  --seed 0 \
  --slot upper \
  --use-existing-sim \
  --record-rosbag \
  --repeat 1
```

默认局部模式不做导航，会在开始动手臂前停住，并提示你先把机器人导航到货架前工作位；确认底盘停止、头部能看到上层料盘后按 Enter 才会开始抓取。只有在你确信机器人已经在货架前时，才使用：

```bash
--assume-at-shelf
```

## 单终端全流程

如果希望采集脚本自己启动 Scene3、走到货架前、再抓取录包，可以用：

```bash
python3 src/challenge_cup_simulator/test/collect_scene3_dataset/collect_scene3_tray_dataset.py \
  --seed 0 \
  --slot upper \
  --full-run \
  --record-rosbag \
  --repeat 1 \
  --debug-run-once \
  --skip-pointcloud-precheck
```

`--full-run` 会默认启动 Scene3 仿真，并执行一段货架前进导航，再开始局部抓取技能。如果仿真已经启动，只想让脚本先走到货架前再抓取：

```bash
python3 src/challenge_cup_simulator/test/collect_scene3_dataset/collect_scene3_tray_dataset.py \
  --seed 0 \
  --slot upper \
  --use-existing-sim \
  --navigate-to-shelf \
  --record-rosbag \
  --repeat 1 \
  --debug-run-once \
  --skip-pointcloud-precheck
```

默认前进距离是 `1.00m`。如果距离还需要微调，可以覆盖：

```bash
--approach-shelf-distance 0.80
```

如果没有 `/state_estimate/base/pos_xyz`，脚本会退回按时间开环前进；也可以强制开环：

```bash
--nav-open-loop
```

调抓取时建议先保留失败包，并在自动前进后人工看一眼位置：

```bash
python3 src/challenge_cup_simulator/test/collect_scene3_dataset/collect_scene3_tray_dataset.py \
  --seed 0 \
  --slot upper \
  --full-run \
  --record-rosbag \
  --repeat 1 \
  --debug-run-once \
  --skip-pointcloud-precheck \
  --no-delete-failed-bag \
  --hold-seconds 3 \
  --confirm-after-navigation
```

`--confirm-after-navigation` 会在前进 1m 后暂停，确认机器人已经正对货架、右臂能到上层料盘后再按 Enter。`--no-delete-failed-bag` 会保留失败 bag，方便回放。

如果日志里显示的前进距离不是你想要的距离，例如：

```text
open-loop forward 1.35m
```

可以直接在命令行覆盖成 1m：

```bash
--approach-shelf-distance 1.00
```

如果一次运行后多个非目标料盘都明显移动，脚本会判失败：

```text
SCENE_DISTURBED:NON_TARGET_TRAY_MOVED
```

这通常说明底盘前进太多，或者手臂/夹爪进货架时撞到了货架和其他料盘。先把 `--approach-shelf-distance` 降低到 `1.00`、`0.90` 或 `0.80` 试，再调 `upper_tray_avoidance_high` 和 `upper_tray_pregrasp`。

成功检查使用“开始录包前”的 `/mujoco/qpos` 快照作为初始料盘位置，而不是 XML 里的静态 body pos。日志中应看到：

```text
initial_source: live_qpos_at_record_start
```

这样可以避免 Scene3 刚启动后的物理 settling 被误认为采集动作撞动了料盘。

只调试轨迹、不录包：

```bash
python3 src/challenge_cup_simulator/test/collect_scene3_dataset/collect_scene3_tray_dataset.py \
  --seed 0 \
  --slot upper \
  --use-existing-sim \
  --no-rosbag \
  --debug-run-once
```

只停在预抓取位观察，并打印目标料盘和右夹爪的相对距离：

```bash
python3 src/challenge_cup_simulator/test/collect_scene3_dataset/collect_scene3_tray_dataset.py \
  --seed 0 \
  --slot upper \
  --full-run \
  --no-rosbag \
  --debug-run-once \
  --skip-pointcloud-precheck \
  --preset-only \
  --hold-seconds 20
```

完整运行时会打印两行调参信息：

```text
[INFO] pregrasp target/gripper distance=...
[INFO] approach target/gripper distance=...
```

其中 `target_base` 是目标料盘在 `base_link` 下的位置，`gripper_base` 是右夹爪在 `base_link` 下的位置，`delta_target_minus_gripper` 是目标相对夹爪的三轴差值。当前如果 `approach` 距离还大于约 `0.30m`，说明 `upper_tray_edge_approach` 还没有贴到料盘前缘，不应进入正式录包。

默认已开启数采专用真值 IK：

```yaml
expert:
  truth_ik:
    enabled: true
```

开启后，`pregrasp / approach / extract / lift` 会使用 `/mujoco/qpos` 中的目标料盘位置生成 IK 目标，腰间收纳仍使用 named pose。这个逻辑只属于数采脚本，不能搬到正式评测推理脚本。若 IK 服务不可用，先确认仿真启动后存在：

```bash
rosservice list | grep /ik
```

抓取点相对目标料盘中心的偏移在 `configs/scene3_collect.yaml` 的 `expert.truth_ik.stage_offsets` 中调：

```yaml
stage_offsets:
  pregrasp:
    x: -0.12
    y: 0.00
    z: 0.02
  approach:
    x: -0.05
    y: 0.00
    z: 0.00
```

`x` 更小表示手更靠近机器人、更远离货架深处；`z` 更大表示手更高。

如果第一次调试时报：

```text
POINTCLOUD_EMPTY: not enough points in pregrasp ROI
```

说明录包前的 RGB-D ROI 点云预检没过。先确认机器人已经在货架前、头部能看到上层料盘、`/tf` 可用；调姿态时也可以先跳过点云预检，把动作和 rosbag 跑通：

```bash
python3 src/challenge_cup_simulator/test/collect_scene3_dataset/collect_scene3_tray_dataset.py \
  --seed 0 \
  --slot upper \
  --use-existing-sim \
  --record-rosbag \
  --repeat 1 \
  --debug-run-once \
  --skip-pointcloud-precheck
```

之后再调 `configs/scene3_roi.yaml` 里的 `extract_upper` 和 `waist_stow` 范围，或临时降低阈值：

```bash
--min-roi-points 50
```

如果失败时报：

```text
TRAY_NOT_EXTRACTED
```

说明成功检查发现目标料盘没有离开原位，默认目标是 `smt_tray_5`，最小抽出距离是 `0.10m`。错误日志会打印：

```text
target
initial_xyz / final_xyz
extract_distance_xy_m
min_extract_distance_m
tray_motion_xy_m
```

看诊断时按这个顺序判断：

- `smt_tray_5` 的 `extract_distance_xy_m` 接近 0：右夹爪没夹到目标料盘，优先调 `upper_tray_pregrasp` 和 `upper_tray_edge_approach`
- 其他上层料盘，例如 `smt_tray_3` 或 `smt_tray_4` 移动明显：抓错目标，改 `scene3_collect.yaml` 里的 `target_tray_name`，或把 right arm 的 y 方向姿态调回 `smt_tray_5`
- `smt_tray_5` 移动了但小于 `0.10m`：抽出不够，调 `upper_tray_extract_mid`、`upper_tray_extract_out`、`upper_tray_post_extract_clearance`
- 料盘掉落：调夹爪闭合参数，或让 `upper_tray_edge_approach` 更贴近前缘后再闭合
- 视觉里明显已经成功但检查失败：临时调低 `success.min_extract_distance_m`，或者确认 `success.target_tray_name` 和实际抓的料盘一致

成功检查还会要求目标料盘最终靠近活动夹爪，默认配置是：

```yaml
success:
  fail_on_non_target_tray_motion: true
  max_non_target_tray_motion_m: 0.03
  require_gripper_near_tray: true
  log_target_gripper_distance: true
  gripper_reference_frame: base_link
  active_gripper_frame: right_gripper_base
  gripper_tf_timeout_sec: 1.0
  max_gripper_tray_distance_m: 0.30
```

这样可以避免“料盘被碰出货架但没有被夹住”也保存为成功数据。如果确实夹住了但仍被判失败，看日志里的 `target_gripper_distance_m`，再把 `max_gripper_tray_distance_m` 适当调大。检查器会把 `/mujoco/qpos` 里的料盘位置转换到 `base_link`，再和 TF 里的 `right_gripper_base` 比距离。

批量 seed：

```bash
python3 src/challenge_cup_simulator/test/collect_scene3_dataset/collect_scene3_tray_dataset.py \
  --slot upper \
  --auto \
  --seed-start 0 \
  --seed-end 100 \
  --record-rosbag \
  --max-attempts-per-seed 5
```

## 文件职责

- `collect_scene3_tray_dataset.py`：主入口，编排 seed/retry、仿真连接、准备姿态、rosbag、专家轨迹、成功/失败清单。
- `scene3_tray_grasp_expert.py`：固定 named pose 专家。第一版只做 `upper_tray_pregrasp -> approach -> close -> extract -> lift -> waist_stow`。
- `scene3_observation_recorder.py`：订阅 RGB-D、相机内参、关节、夹爪，并发布 `stage`、`expert_action`、可选 xyzrgb ROI 点云。
- `scene3_pointcloud_utils.py`：解码 compressed RGB-D，反投影为 `[x,y,z,r,g,b]`，ROI 裁剪、固定点数采样、归一化。
- `scene3_rosbag_utils.py`：rosbag 启停、topic 等待、bag 频率检查、manifest 写入。
- `scene3_success_checker.py`：bag 质量检查和数采专用 `/mujoco/qpos` 真值检查。真值只在本目录使用。
- `scripts/convert_scene3_bag_to_il_dataset.py`：离线转换为 NPZ/HDF5 episode。

## 输出

默认输出到：

```text
bags/scene3_tray_stow/
```

成功 episode：

```text
scene3_seed_{seed}_upper_{timestamp}_attempt_{n}.bag
scene3_seed_{seed}_upper_{timestamp}_attempt_{n}.yaml
success_manifest.txt
```

失败 episode：

```text
failed_seeds.txt
```

失败时默认删除 bag，只保留 metadata 和 failed manifest。调试时可加：

```bash
--no-delete-failed-bag
```

## 需要调参的位置

`configs/scene3_named_poses.yaml` 中的关节角是第一版占位，继承了正式任务模板的 Scene3 姿态并补齐了 approach/extract/lift/stow。实际采集前需要在仿真中手调：

- `upper_tray_avoidance_high`
- `upper_tray_pregrasp`
- `upper_tray_edge_approach`
- `upper_tray_extract_mid`
- `upper_tray_extract_out`
- `upper_tray_post_extract_clearance`
- `upper_tray_lift`
- `upper_tray_shelf_clearance`
- `waist_stow_pose`

### 抓取参数怎么改

抓取参数主要在两个配置文件：

```text
configs/scene3_named_poses.yaml
configs/scene3_collect.yaml
```

`scene3_named_poses.yaml` 负责每个动作点的 14 个手臂关节角，顺序是：

```text
left  7 joints: l_arm_pitch, l_arm_roll, l_arm_yaw, l_forearm_pitch, l_hand_yaw, l_hand_pitch, l_hand_roll
right 7 joints: r_arm_pitch, r_arm_roll, r_arm_yaw, r_forearm_pitch, r_hand_yaw, r_hand_pitch, r_hand_roll
```

当前抓上层料盘默认用右臂，所以通常只需要调每个 pose 的 `right: [...]` 和 `duration`：

```yaml
upper_tray_pregrasp:
  right: [10, -20, 0, -80, 0, 20, 0]
  duration: 3.0
```

常用调参方向：

- 夹爪还没到料盘前缘：调 `upper_tray_pregrasp` 和 `upper_tray_edge_approach`
- 夹爪插得太深或撞货架：减小 `upper_tray_edge_approach` 的前伸程度，或增大 `upper_tray_avoidance_high` / `upper_tray_pregrasp` 的安全间隙
- 抽出时刮货架：调 `upper_tray_extract_mid`、`upper_tray_extract_out`、`upper_tray_post_extract_clearance`
- 抽出后转腰间时撞货架边：调 `upper_tray_shelf_clearance`
- 腰间收纳位置不稳：调 `waist_stow_pose`
- 动作太快：增大对应 pose 的 `duration`

夹爪参数在 `scene3_collect.yaml` 的 `expert.gripper`：

```yaml
expert:
  gripper:
    backend: joint_state
    open_position: 0.0
    close_position: 255.0
    leju_open_position: 10.0
    leju_close_position: 90.0
    close_wait_sec: 0.5
```

如果用 `/gripper/command`，主要调 `open_position / close_position`。如果切到 leju topic/service，主要调 `leju_open_position / leju_close_position`。

### 货架避障

货架避障已经加在 `scene3_collect.yaml` 的 `expert.shelf_avoidance`，默认开启：

```yaml
expert:
  shelf_avoidance:
    enabled: true
    pregrasp_waypoints:
      - upper_tray_avoidance_high
      - upper_tray_pregrasp
    approach_waypoints:
      - upper_tray_edge_approach
    extract_waypoints:
      - upper_tray_extract_mid
      - upper_tray_extract_out
      - upper_tray_post_extract_clearance
    lift_waypoints:
      - upper_tray_lift
    stow_waypoints:
      - upper_tray_shelf_clearance
      - waist_stow_pose
      - finish_hold_pose
```

避障逻辑是固定关节 waypoint 绕开货架：

```text
safe_home
-> scene3_ready_pose
-> upper_tray_avoidance_high        # 从货架外高位靠近，避免手臂扫到货架
-> upper_tray_pregrasp
-> upper_tray_edge_approach
-> close_gripper
-> upper_tray_extract_mid
-> upper_tray_extract_out
-> upper_tray_post_extract_clearance # 抽出后先离开货架边缘
-> upper_tray_lift
-> upper_tray_shelf_clearance        # 转腰间前先清货架立柱/边框
-> waist_stow_pose
-> finish_hold_pose
```

如果你想临时关闭避障 waypoint，改成：

```yaml
expert:
  shelf_avoidance:
    enabled: false
```

更推荐保持开启，只调 `scene3_named_poses.yaml` 里的避障姿态。尤其是 `upper_tray_post_extract_clearance` 和 `upper_tray_shelf_clearance`，它们是避免料盘/夹爪撞货架边缘的关键点。

当前脚本只采上层货架料盘，命令行 `--slot` 仅支持 `upper`。默认目标真值检查对象是 `smt_tray_5`，对应 Scene3 XML 中右侧上层料盘。如果要采其他上层料盘，改：

```yaml
scene:
  target_tray_name: smt_tray_3
success:
  target_tray_name: smt_tray_3
```

## 离线转换

```bash
python3 src/challenge_cup_simulator/test/collect_scene3_dataset/scripts/convert_scene3_bag_to_il_dataset.py \
  --rosbag-dir bags/scene3_tray_stow \
  --output-dataset-dir dataset/scene3_tray_stow \
  --format npz \
  --train-hz 10 \
  --chunk-size 20 \
  --point-num 1024
```

转换后的 episode 包含：

```text
images_head_rgb
points_head_xyzrgb
states_q_arm
states_q_gripper
states_stage_id
actions_observed_dq_arm
actions_expert_q_target
actions_expert_gripper_cmd
timestamp
```

ACT / Diffusion Policy 训练时可用 `states_* + images_head_rgb + points_head_xyzrgb + stage_id` 作为 observation，用 `actions_observed_dq_arm` 或 expert command delta 作为 action。

## 边界

允许在数采脚本中使用 `/mujoco/qpos`、active XML、generated layout 等真值做专家动作和成功检查。正式评测推理脚本不能读取这些真值，也不能依赖本目录的 `generated_layouts/`。
