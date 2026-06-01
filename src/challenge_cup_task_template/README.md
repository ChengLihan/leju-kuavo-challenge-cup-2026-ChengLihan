# challenge_cup_task_template

挑战杯仿真赛**选手任务模板**功能包。职责划分参考 CRAIC 的 `craic_task_template`：

- `challenge_cup_simulator`：只管仿真环境 / 场景 / 模型；
- `challenge_cup_task_template`（本包）：只管选手的一键入口脚本。

## 快速开始

```bash
# 编译后 source 工作空间
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
rosrun challenge_cup_task_template challenge_task.py --scene scene3
```

该脚本会自动：

1. 调用受保护的 `challenge_sim_launcher.py` 生成静态场景 XML；
2. `roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch`（自带 roscore）启动仿真；
3. 通过 `challenge_secret*.so` 按 seed 计算运行时物体位置，并写入 MuJoCo 内存；
4. 锁定物体摆放服务并启动作弊监控；
5. 初始化 ROS 节点并等待 `/sensors_data_raw` 出现确认就绪；
6. 进入脚本里的 TODO 任务逻辑区，由选手填写。

退出（Ctrl+C）时通过 `atexit` 自动关闭 roslaunch 子进程。

## 文件结构

```
challenge_cup_task_template/        # 选手包（可编辑：写任务逻辑）
├── package.xml
├── CMakeLists.txt
├── README.md
└── scripts/
    └── challenge_task.py           # 三场景统一入口

challenge_cup_simulator/utils/      # 受保护包（选手不可改动）
└── challenge_sim_launcher.py       # 公共启动器：校验 + 生成场景 + roslaunch
```

> **公共启动器 `challenge_sim_launcher.py` 故意放在受保护的 `challenge_cup_simulator` 包内**，
> 而非本包，这样完整性校验无法被选手删改绕过。场景脚本通过 rospkg 定位并导入它。

## 关于 `--seed`（重要）

挑战杯的 seed 与 CRAIC **语义不同**：

- seed 不随机机器人初始位姿；
- scene1：4 个包裹在各自基准位附近做 y 方向随机抖动；
- scene2：6 个零件按 seed 打乱摆放；
- scene3：当前无随机化配置，保留 seed 参数用于接口一致性。

## 稳定控制参数（重要）

公共启动器在 `roslaunch` 时**显式带上**已验证过的稳定控制参数：

```
with_estimation:=true
wbc_frequency:=1000
sensor_frequency:=1000
```

这是为了避免一键启动复现此前 challenge launch 默认值被降级（关闭状态估计 / 频率减半）
导致的转向异常 / 摔倒问题。即使将来 launch 默认值再次被改动，本包也能保证启动行为稳定。

## 生成文件

每次启动会在 `challenge_cup_simulator/models/biped_s52/xml/` 下生成：

```
_scene_<scene>_active.xml
```

> 必须放在该 xml 目录、与原始 `sceneN.xml` 同级，因为 XML 里的 `include` / `mesh` / `texture`
> 都是相对该 XML 文件位置的相对路径，放到 `/tmp` 等其它目录会导致资源加载失败。

这些生成文件已在仓库根 `.gitignore` 中忽略（`_scene_*.xml`），不会污染 git。

## 完整性校验（防篡改）

每次启动时，launcher 会调用 `challenge_secret`（编译为 `.so` 的 Cython 模块，
对标 CRAIC 的 `craic_secret`）校验场景**源输入**和启动器是否被篡改：
`config/scenes/scene*.yaml` + `utils/scene_builder.py` + `utils/challenge_sim_launcher.py` +
`models/biped_s52/xml/biped_s52.xml`。

- `.so` 缺失 → 默认 `[FATAL]` 退出（fail-closed）；开发机如需放行，设
  `CHALLENGE_SECRET_ALLOW_MISSING=1` 降级为警告并继续；
- 校验不通过（文件被改）→ `[FATAL]` 退出，不启动仿真。

编译产物 `challenge_cup_simulator/lib/challenge_secret*.so` 随仿真包分发；
机密源码由组委会单独保存，**不**进选手分发仓库。

> 校验调用 `_verify_integrity()` 位于受保护的 `challenge_cup_simulator/utils/challenge_sim_launcher.py`，
> 选手无法改动该包，因此无法删改校验逻辑绕过。选手只能编辑本包（task_template）里的任务脚本。

## 与 CRAIC 的差异

挑战杯有**自己的** `challenge_secret`（见上），但仍**不**使用 CRAIC 的以下能力：

- `get_random_init_state`（随机机器人初始位姿）—— 挑战杯 seed 只做物体随机化；
- `GripperController`（CRAIC 夹爪类）—— 挑战杯请用 `/control_robot_leju_claw` 服务及
  `/leju_claw_command`、`/leju_claw_state` 话题，参考
  `challenge_cup_simulator/scripts/sim_leju_claw_interface.py`、`leju_claw_keyboard.py`。
