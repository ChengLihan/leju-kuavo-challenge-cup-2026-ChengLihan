## Scene2 数采脚本说明

### 文件职责

- `scene2_data_collection_pipeline.py`：当前主入口。启动 Scene2 仿真、执行六个工件分拣、可选录制 rosbag，并检查分拣结果和相机频率。
- `scene2_part_grasp_ik.py`：抓取与交接逻辑。包含工件偏置、抓取姿态、窄边/另一头抓取判断、交接位姿配置、FK/IK 服务调用。
- `challenge_task.py`：数采专用仿真启动入口。pipeline 会直接调用这个脚本。
- `generated_layouts/`：Scene2 每个 seed 的工件位姿 YAML。记录六个工件实际摆到仿真里的 `world_xyz`、`quat_wxyz` 和绕 Z 轴 yaw，pipeline 和 IK 脚本按 seed 读取。

### 路径约定

数采流程启动 Scene2 时，`challenge_task.py` 会把当前 seed 的工件位姿写到：

```text
src/challenge_cup_simulator/test/collect_scene2_dataset/generated_layouts/
```

这里的“工件位姿”不是箱子/桌子/模型尺寸配置，而是 `challenge_secret` 根据 `scene2 + seed` 算出的六个待分拣工件真实摆放结果：

```text
part_type_a_1 / part_type_a_2
part_type_b_1 / part_type_b_2
part_type_c_1 / part_type_c_2
```

每个工件会记录世界坐标 `world_xyz`、四元数 `quat_wxyz`、以及从四元数反算的 `yaw_z_rad/yaw_z_deg`。`scene2_part_grasp_ik.py` 会用这些值把本地配置的 `grasp_offset_xyz_local` 转到世界系，并按工件实际朝向计算抓取姿态。

rosbag 默认输出到：

```text
bags/scene2/
```

可以通过 `--output-dir` 改写。

## 推荐入口

### 单个 seed 采集一个有效 rosbag

```bash
python3 src/challenge_cup_simulator/test/collect_scene2_dataset/scene2_data_collection_pipeline.py \
  --seed 0 \
  --record-rosbag \
  --repeat 1
```

`--repeat 1` 的含义是“直到成功保存 1 个有效 rosbag”。如果分拣失败、相机频率不达标或 rosbag 检查失败，会丢弃当前 bag 并重新尝试。

### 连续 seed 自动采集

```bash
python3 src/challenge_cup_simulator/test/collect_scene2_dataset/scene2_data_collection_pipeline.py \
  --record-rosbag \
  --auto \
  --seed-start 0 \
  --seed-end 100
```

每个 seed 采集 1 个有效 rosbag，默认最多尝试 5 次。5 次都失败则跳过该 seed，继续下一个。

### 指定抓取顺序

把某个工件放到第一个抓：

```bash
python3 src/challenge_cup_simulator/test/collect_scene2_dataset/scene2_data_collection_pipeline.py \
  --seed 0 \
  --record-rosbag \
  --first-pick part_type_c_1
```

把某个工件抓取顺序插到另一个工件前面：

```bash
python3 src/challenge_cup_simulator/test/collect_scene2_dataset/scene2_data_collection_pipeline.py \
  --seed 0 \
  --record-rosbag \
  --insert-before part_type_c_2 part_type_b_1
```

可选工件名：

```text
part_type_b_1
part_type_b_2
part_type_a_1
part_type_a_2
part_type_c_1
part_type_c_2
```

## `scene2_data_collection_pipeline.py` 选项

这是当前主数采脚本。

```bash
python3 src/challenge_cup_simulator/test/collect_scene2_dataset/scene2_data_collection_pipeline.py [选项]
```

### 基础选项

- `--seed SEED`：指定 Scene2 seed。不指定时随机生成一个 seed。
- `--output-dir DIR`：指定 rosbag 输出目录。默认是仓库下的 `bags/scene2/`。

### rosbag 选项

- `--record-rosbag`：开启 rosbag 录制。录制从机械臂第一次到工作位后开始，到六个工件分拣完成并最后回到工作位附近后停止。

开启录制后，脚本会检查：

- 六个工件是否都进入目标箱。
- 相机相关 topic 是否存在。
- 相机 topic 平均频率是否不低于 `25Hz`。

失败时会丢弃当前 rosbag。

### 抓取顺序选项

- `--first-pick OBJECT`：把指定工件提到第一个抓，其他工件仍按自动排序。
- `--insert-before MOVE_OBJECT BEFORE_OBJECT`：在自动排序结果上，把 `MOVE_OBJECT` 插到 `BEFORE_OBJECT` 前面。

如果同时使用 `--insert-before` 和 `--first-pick`，先执行插队，再把 `--first-pick` 指定工件提到第一位。

### 重复采集选项

- `--repeat N`：收集 `N` 个有效运行结果。失败不计数，会继续重跑直到达到 `N` 个有效结果。

示例：

```bash
python3 src/challenge_cup_simulator/test/collect_scene2_dataset/scene2_data_collection_pipeline.py \
  --seed 0 \
  --record-rosbag \
  --repeat 3
```

### 自动 seed 范围选项

- `--auto`：开启按 seed 范围自动采集模式。
- `--seed-start SEED`：自动模式起始 seed，包含该 seed。
- `--seed-end SEED`：自动模式结束 seed，包含该 seed。
- `--max-attempts-per-seed N`：每个 seed 最多尝试次数，默认 `5`。

示例：

```bash
python3 src/challenge_cup_simulator/test/collect_scene2_dataset/scene2_data_collection_pipeline.py \
  --record-rosbag \
  --auto \
  --seed-start 0 \
  --seed-end 100 \
  --max-attempts-per-seed 5
```

### 抓取与交接调参位置

pipeline 负责启动仿真、构建分拣 job、判断是否需要交接、调用抓取/放置流程。具体的抓取偏置、抓取姿态、交接位姿主要集中在：

```text
src/challenge_cup_simulator/test/collect_scene2_dataset/scene2_part_grasp_ik.py
```

### A/B/C 类工件抓取配置

A/B/C 六个工件的抓取配置在 `scene2_part_grasp_ik.py` 的 `OBJECT_PART_CONFIG` 里：

```text
part_type_a_1 / part_type_a_2
part_type_b_1 / part_type_b_2
part_type_c_1 / part_type_c_2
```

每个工件常用字段：

- `grasp_offset_xyz_local`：抓取偏置，表示“从工件原点到期望夹爪目标点”的 `[x, y, z]` 偏移量，单位是米，坐标系是工件自己的局部坐标系，不是世界坐标系，例如可以进入`src/challenge_cup_simulator/models/t_junction`，终端输入`meshlab T_junction.STL`可视化工件模型，并点开坐标系选项，查看工件自身的局部坐标系 。代码会先根据当前 seed 的 `quat_wxyz/yaw` 把这个局部偏置旋转到世界系，再加到该工件的 `world_xyz` 上。它按 `left/right` 和工件实际 `world_y` 的 `y_range` 分档，方便同一个工件在不同 Y 位置、不同作业手下用不同夹取点。
- `grasp_pose`：抓取时夹爪姿态。会叠加工件实际 yaw 和自动窄边 `±90°`。
- `lift_pose`：抓起后抬升姿态。若 `follow_grasp_narrow_edge=True`，会沿用抓取时选择的窄边方向。
- `place_pose`：放入箱子时的姿态。
- `use_opposite_arm`：在自动选手后强制换另一只手。
- `flip_auto_narrow_edge_grasp`：整体反转自动选择的窄边方向。

最终抓取点大致按下面链路得到：

```text
当前 seed 的工件 world_xyz
  + 选中的 grasp_offset_xyz_local 转成世界系后的偏移
  + pipeline 里的 WORLD_TO_EE_OFFSET_*
  => IK 目标点
```

如果只想调某个工件夹哪里，优先改对应工件的 `grasp_offset_xyz_local`。如果要整体平移所有抓取点，再改 `scene2_data_collection_pipeline.py` 里的 `WORLD_TO_EE_OFFSET_X/Y/Z`。

### 偏置选择规则

偏置选择逻辑在 `scene2_part_grasp_ik.py`：

- `_select_offset_from_y_bins()`
- `_grasp_offset_local_for_side()`
- `_apply_grasp_offset()`
- `get_object_world_xyz()`

规则：

1. 根据当前作业手 `active_arm`，先取 `grasp_offset_xyz_local["left"]` 或 `["right"]`。
2. 用工件实际 `world_y` 命中对应 `y_range`。
3. 如果当前手没有命中，会去另一只手的 offset 表里找同一个 `world_y` 区间。
4. 两边都没命中时，选当前手最近的 `y_range`。
5. 如果自动窄边方向算出的 `narrow_offset < 0`，会把 `offset_local[1]` 取反。

因此调 Y 方向偏置时要注意：配置里的局部 Y 值，运行时可能因为“另一头抓”被反号。

### 左右手交接配置

是否需要交接由 `scene2_data_collection_pipeline.py` 的 `_handoff_target_arm()` 判断：

```text
A 类进 sorting_bin_a：
  如果当前是左手抓，交给右手。

C 类进 sorting_bin_c：
  如果当前是右手抓，交给左手。

B 类当前不交接。
```

交接位置和姿态在 `scene2_part_grasp_ik.py` 的 `OBJECT_PART_CONFIG` 中：

```text
handoff_left_to_right
handoff_right_to_left
```

常用字段：

- `place_world_xyz_by_part_type`：当前手把工件送到交接点的位置。
- `place_quat_xyzw_by_part_type`：当前手交接释放时的姿态。
- `grasp_world_xyz_by_part_type`：目标手重新抓取的位置。
- `grasp_quat_xyzw_by_part_type`：目标手重新抓取时的姿态。

如果要改“中间左右手交接偏置”，主要改这两个字段：

```text
OBJECT_PART_CONFIG["handoff_left_to_right"]["place_world_xyz_by_part_type"]
OBJECT_PART_CONFIG["handoff_left_to_right"]["grasp_world_xyz_by_part_type"]

OBJECT_PART_CONFIG["handoff_right_to_left"]["place_world_xyz_by_part_type"]
OBJECT_PART_CONFIG["handoff_right_to_left"]["grasp_world_xyz_by_part_type"]
```

这些值是世界坐标语义，pipeline 后续还会通过 `_world_xyz_to_ee_xyz()` 加末端执行器偏置。

### 另一头抓判断

“从工件的另一头抓”的判断逻辑在 `scene2_part_grasp_ik.py`：

- `_select_narrow_edge_yaw_offset()`：在 `+90°` 和 `-90°` 两个窄边候选中，先选夹爪工具方向更朝机器人前方的一侧。
- `_part_should_grasp_other_narrow_edge_by_axis()`：按工件类型、位置和局部轴方向判断是否要翻到另一头。
- `_narrow_edge_yaw_offset_for_object()`：汇总自动选择、另一头抓判断、以及 `flip_auto_narrow_edge_grasp`。
- `get_object_grasp_quat_xyzw()`：最终生成抓取四元数。

当前规则：

- A 类：仅当 `object_y` 在 `[-0.2, 0.1]` 时判断。看工件局部 Y 轴在世界 XY 平面里的方向，目标方向是 `[1, 1]`，夹角阈值 `45°`。不满足对齐时翻到另一头抓。
- B 类：`object_y` 在 `[0.0, 0.2]` 时目标方向是 `[1, -1]`；`object_y` 在 `[-0.2, 0.0)` 时目标方向是 `[1, 1]`；夹角阈值 `45°`。不满足对齐时翻到另一头抓。
- C 类右手特殊规则：只对 C 类、右手、且 `object_y` 在 `[-0.3, 0.0]` 生效。把工件局部向量 `[-1, 0, 1]` 转到世界系，如果它和世界 `-Y` 的夹角小于等于 `20°`，就翻到另一头抓。

如果要改判断条件，改 `_part_should_grasp_other_narrow_edge_by_axis()`。如果只是临时强制某个工件整体反方向，改该工件配置里的：

```python
"flip_auto_narrow_edge_grasp": True
```

如果要关闭某类自动翻转，可以在 `_part_should_grasp_other_narrow_edge_by_axis()` 里让对应分支直接 `return False`。

