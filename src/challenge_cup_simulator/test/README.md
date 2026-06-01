# Scene2 Dataset Collection

这个目录里的 `collect_scene2_dataset.py` 用于采集场景二固定抓取数据。默认流程是：

1. 启动 `challenge_cup_simulator` 的场景二仿真。
2. 先控制头部低头并等待稳定。
3. 开始录制指定相机、手臂、夹爪话题。
4. 执行两次固定位置抓取，并放到中间收纳盒。
5. 切回手臂模式后等待 1 秒，停止 rosbag。
6. 关闭本次仿真；如果设置了 `--count`，再启动下一轮。

手臂实际控制走 `/kuavo_arm_target_poses`，但数据集只录 `/kuavo_arm_traj`。`/kuavo_arm_traj` 是 100Hz 的 `JointState` 线性插值轨迹，单位为角度，方便按连续 action 读取。

相机数据只录压缩图话题。`load_kuavo_mujoco_challenge.launch` 默认 `raw_image:=false`，避免额外 raw republish 占用 CPU；需要 raw 图调试时再显式传 `raw_image:=true`。脚本自动采集时也保持 `mujoco_vsync:=true`，画面更稳，30Hz 主要靠相机异步编码发布保证。

## 自动启动仿真并采集

在 Docker 或工作空间环境中先 source：

```bash
source ~/.zshrc
source /root/kuavo_ws/devel/setup.zsh
cd /root/kuavo_ws
```

采集 1 组：

```bash
python3 src/challenge_cup_simulator/test/collect_scene2_dataset.py --count 1
```

连续采集多组：

```bash
python3 src/challenge_cup_simulator/test/collect_scene2_dataset.py --count 10
```

脚本每一轮都会执行：

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch scene_name:=scene2 raw_image:=false mujoco_vsync:=true
```

所以 `--count` 不是在同一个仿真里重复抓取，而是“启动仿真、采一组、关闭仿真、再启动下一组”。

## 使用已启动的仿真调试

调试动作时可以自己先启动仿真：

```bash
roslaunch challenge_cup_simulator load_kuavo_mujoco_challenge.launch scene_name:=scene2 raw_image:=false
```

然后另开终端执行：

```bash
python3 src/challenge_cup_simulator/test/collect_scene2_dataset.py --use-existing-sim
```

`--use-existing-sim` 不会关闭你手动启动的仿真，也不能和 `--count > 1` 一起使用。

## 输出位置

默认输出到：

```text
bags/scene2/scene2_时间戳.bag
```

`--count > 1` 时文件名会带 run 序号，例如：

```text
bags/scene2/scene2_20260601_103000_run_001.bag
bags/scene2/scene2_20260601_103107_run_002.bag
```

输出目录只保留 `.bag` 文件，不再生成每组子目录、`metadata.json` 或日志文件。
