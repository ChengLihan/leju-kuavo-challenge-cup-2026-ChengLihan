## Scene1 数采脚本说明

### 文件职责

- `collect_scene1_handoff_dataset.py`：当前唯一主入口。启动 Scene1 仿真、把双臂带到身前预设位，然后对 4 个快递依次执行「右手抓取 → 放到称重区 → 二次夹起 → 交给左手 → 左手放进分拣箱」，可选录制 rosbag，并按「称重变黄」和「入箱」两道检查判定每个包裹是否成功。
- `challenge_secret.cpython-38-*.so`：场景密钥/位姿检查规格（`get_pose_check_spec`）的二进制实现，脚本运行时按 `scene1` 读取；本目录随脚本一起携带。

### 流程概览

一个 seed 一条龙（4 个快递 `parcel_1 ~ parcel_4` 顺序处理）：

```text
双臂预设抬手 (PRESET_POINTS_DEG, 段1外展张开 → 段2抬到身前对称预设位)
  └─ [录制起点] 抬到预设位后才开始录 rosbag
对每个 parcel：
  右手抓取        → 高位对齐 x/y/姿态，再下降到抓取高度，0x03 硬解夹紧(解不出回退 0x06)
  放到称重区      → 高位横移到称重点上方，下降释放；松手后判「变黄」(geom 角点全落入 + z 合理)
  二次夹起        → 实时读包裹落点，原姿态下降重新夹起
  交给左手        → 右手抬到交接高位，左手到接收点，右手松开、按 y 偏移退让
  左手放进箱子    → 抬起 → 箱上方预备 → 下放 → 松手；入箱检查(箱体内框 + margin)
最后回到 PRESET_POINTS_DEG 末点
```

### 路径约定

rosbag、成功清单、失败清单默认输出到（容器内）：

```text
/root/kuavo_ws/bags/scene1_handoff/
```

可用 `--output-dir` 改写。每条成功保留的 bag 命名为：

```text
scene1_seed_{seed}_{时间戳}.bag      # 例 scene1_seed_0_20260617_191306.bag
```

同目录下会维护两个 txt：

- `success_manifest.txt`：每条成功（bag 已保留）追加一行，`seed  时间  bag=...  parcels=...  run=i/N`。
- `failed_seeds.txt`：每条失败追加一行，`seed  时间  run=i/N  parcels=...  reason=...`。

## .so 切换（采集前 / 正式测试前）

采集要用本目录随附的 `.so`，正式测试要用原本的 `.so`。程序固定从 `challenge_cup_simulator/lib/`
按文件名加载，所以只能把文件**覆盖进 `lib/`**，不能放别处用绝对路径指。

采集前（先备份原本的，再换上采集用的）：

```bash
cd <你的工作区>/src/challenge_cup_simulator
cp -n lib/challenge_secret.cpython-38-x86_64-linux-gnu.so  lib/challenge_secret.cpython-38-x86_64-linux-gnu.so.bak
cp -f test/collect_scene1_dataset/challenge_secret.cpython-38-x86_64-linux-gnu.so  lib/challenge_secret.cpython-38-x86_64-linux-gnu.so
```

正式测试前（换回原本的）：

```bash
cd <你的工作区>/src/challenge_cup_simulator
cp -f lib/challenge_secret.cpython-38-x86_64-linux-gnu.so.bak  lib/challenge_secret.cpython-38-x86_64-linux-gnu.so
```

> 正式测试前务必换回原本的 `.so`。

## 推荐入口

> 下面命令里的脚本路径按本目录写；实际运行请在已经起好 ROS/MuJoCo 依赖的容器里执行。

### 单个 seed 采集一个有效 rosbag

```bash
python3 src/challenge_cup_simulator/test/collect_scene1_dataset/collect_scene1_handoff_dataset.py \
  --seed 0
```

默认就会录制 rosbag。录制从「机械臂抬到预设位之后」开始，到 4 个快递全部处理完。只有当 **每个包裹都通过称重 + 入箱检查** 时才保留该 bag；任一失败会丢弃当前 bag，并对同一 seed 重启重试，最多 `--max-seed-attempts` 次（默认 10），仍失败则记入 `failed_seeds.txt` 并跳过。

### 连续多个 seed 自动采集

```bash
python3 src/challenge_cup_simulator/test/collect_scene1_dataset/collect_scene1_handoff_dataset.py \
  --seed 0 \
  --count 200
```

`--count N` 表示从 `--seed` 起做 N 个「重启仿真 + 录一条有效 bag」的循环，seed 依次 `+1`（0,1,…,199）。每个 seed 内部仍按 `--max-seed-attempts` 重试。

### 只采指定包裹 / 调试不录包

```bash
# 只跑 parcel_3（可重复 --parcel 指定多个；不带则四个全跑）
python3 .../collect_scene1_handoff_dataset.py --seed 0 --parcel parcel_3 --no-rosbag --verify-object-pose

# 只看抓取/称重/箱子的目标点，不启动 ROS
python3 .../collect_scene1_handoff_dataset.py --seed 0 --print-plan

# 只跑预设抬手轨迹，停住观察（调预设/避障路径用）
python3 .../collect_scene1_handoff_dataset.py --preset-only --hold-seconds 15
```

## 选项

```bash
python3 src/challenge_cup_simulator/test/collect_scene1_dataset/collect_scene1_handoff_dataset.py [选项]
```

### 基础 / 批量

- `--seed SEED`：场景随机种子（布局 + 启动），默认 `0`。
- `--count N`：从 `--seed` 起做 N 个重启-录制循环，seed 递增，默认 `1`。
- `--output-dir DIR`：bag / 两个 txt 的输出目录，默认 `/root/kuavo_ws/bags/scene1_handoff`。
- `--failed-seeds-file FILE`：失败清单文件名（相对路径落在 `--output-dir` 下），默认 `failed_seeds.txt`。
- `--max-seed-attempts N`：录 bag 时同一 seed 失败后的重试次数，默认 `10`（仅录 bag 时生效）。

### 包裹选择

- `--parcel NAME`：只处理指定包裹，可重复。可选 `parcel_1 / parcel_2 / parcel_3 / parcel_4`；不指定则四个都跑。
- `--max-parcels N`：只处理选中的前 N 个。

### 录制与检查

- `--no-rosbag`：不录 bag，只跑运动（默认是录的）。
- `--debug-verify-object-pose` / `--verify-object-pose`：在 `--no-rosbag` 时也开启称重 + 入箱检查（录 bag 时自动开启）。
- `--weigh-dwell SEC`：包裹放上称重区后的停留时长，默认 `1.0`。
- `--headless`：本次运行设 `MUJOCO_HEADLESS=1`，不开窗口。
- `--use-existing-sim`：挂到一个已经在跑的 scene1 仿真上，不自己启动。

### 调试

- `--print-plan`：只打印 seed 推出的各目标点（抓取/称重/交接/放箱），不启动 ROS。
- `--no-realtime-pick`：关闭抓取前的实时目标刷新，退回 seed 计算出的布局。默认开启：每个包裹抓取前会从 `/mujoco/qpos` 实时读包裹实际落定的 x/y 再对准（抓取高度仍固定）。
- `--preset-only`：只跑预设抬手轨迹（move_home）后停住观察，不进入抓取。
- `--preset-waypoints N`：配合 `--preset-only`，只走前 N 个预设路点。
- `--hold-seconds SEC`：调试模式跑完后停住观察的时长，默认 `15`。

### 姿态覆盖（一般不用动，调试用）

各阶段夹爪欧拉角（yaw pitch roll，单位度，与 `arm_control.py` 对齐，两套依次叠加）：

```text
--right-ypr-deg / --right-second-ypr-deg              # 右手第一次抓取
--right-weigh-ypr-deg / --right-weigh-second-ypr-deg  # 右手称重释放
--right-regrasp-ypr-deg / --right-regrasp-second-ypr-deg  # 右手二次夹起
```

## 调参位置

所有可调参数集中在 `collect_scene1_handoff_dataset.py` 顶部，按流程编号分段。常用的几组：

### 1. 右手抓取

- `RIGHT_PICK_IK_Z`：最终抓取高度基准（绝对 IK z）。**抓太高/太低、称重 `full_inside=False` 优先调这里。**
- `RIGHT_PICK_OFFSET_FAR_ROW` / `RIGHT_PICK_OFFSET_NEAR_ROW` / `RIGHT_PICK_NEAR_FAR_Y_THRESHOLD`：按包裹实际 Y 分「远身 / 近身」两档的抓取微调 `[x, y, z]`。
- `RIGHT_PICK_OFFSET_BY_PARCEL`：给特定包裹名单独覆盖（优先级最高），如 `parcel_4`。
- `RIGHT_PICK_TRANSIT_IK_Z`：抓取前横移高度；`USE_HARD_FINAL_GRASP` / `RIGHT_GRASP_FINAL_CONSTRAINT_MODE`：最终一步是否用 0x03 硬解。

### 2. 称重 / 二次夹起

- `WEIGHING_CENTER_WORLD`：称重区世界中心（场景参考，对齐 scene1.xml 实测 `[-0.17,-0.56]`）。
- `WEIGH_RELEASE_IK`：释放点 IK（已做落点补偿）。`WEIGH_TRANSIT_IK_Z`：释放前高位。
- `WEIGH_REGRASP_IK`：二次夹起点（z 为重新夹起高度）。
- 「变黄」判定阈值：`WEIGHING_AREA_GEOM_NAME` + `WEIGH_FOOTPRINT_TOLERANCE`（与仿真 `mujoco_node.cc` 同源）；`WEIGH_CHECK_Z_RANGE` 只判 z 是否落回台面附近。

### 3. 左右手交接

- `LEFT_PRESET_2_IK`：左手等待位；`RIGHT_HANDOFF_TO_LEFT_IK` + `RIGHT_HANDOFF_TRANSIT_IK_Z`（及 `*_FALLBACK_IK_ZS`）：右手交接点与避障高位。
- `LEFT_HANDOFF_RECEIVE_IK`（及 `*_XZ_READY_IK`）：左手接收点；`RIGHT_HANDOFF_RELEASE_RETRACT_Y_OFFSET`：右手退让 y 偏移。

### 4. 放入箱子

- `BOX_DROP_IK`：左手放箱基准点。`BOX_DROP_OFFSET_BY_PARCEL`：每个包裹在基准上叠加的 `[x,y,z]`，默认排成 2×2。
- `BOX_DROP_IK_X_FALLBACK_DELTAS`：放置点 IK **硬失败**时把 x 往机器人侧收的回退表（没失败用原值）。
- 入箱检查：`BOX_BODY_NAME` / `BOX_INNER_SIZE_WORLD` / `BOX_CHECK_XY_MARGIN` / `BOX_CHECK_Z_MARGIN`。

### 预设抬手

- `PRESET_POINTS_DEG`：双臂预设路点（段1外展张开避障，段2抬到身前对称位，末点也是收尾回归点）；`PRESET_SEGMENT_TIME` / `PRESET_SETTLE_TIME` / `PRESET_SPLINE_TENSION` 控制插值。
