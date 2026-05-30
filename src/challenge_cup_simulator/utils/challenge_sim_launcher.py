#!/usr/bin/env python3
"""
挑战杯仿真环境自动启动器（公共逻辑）

本模块**故意放在 challenge_cup_simulator 包内**（选手不可改动的受保护包），
而不是选手任务模板包里——这样完整性校验 _verify_integrity() 无法被选手删改绕过。
选手脚本（challenge_cup_task_template）只负责 import 并调用：

    from challenge_sim_launcher import ChallengeSimLauncher

    launcher = ChallengeSimLauncher(scene="scene2", seed=0)
    launcher.start(node_name="scene2_sorting")   # 校验 + 生成场景XML + roslaunch + 初始化ROS节点 + 等待就绪
    # ... 选手任务逻辑 ...
    launcher.stop()                               # 退出时关闭仿真（已注册 atexit，通常无需手动调用）

要点：
  - 启动前用 challenge_secret(.so) 校验场景源文件完整性（防篡改），失败则拒绝启动。
  - seed 仅用于 scene_builder 的“构建期物体随机化”，不随机机器人初始位姿。
    当前只有 scene2 配置了 shuffleable_parts；scene1/scene3 传 seed 基本无效果。
  - 生成的场景 XML 写入 challenge_cup_simulator 包内的 xml 目录（不是 /tmp），
    因为 XML 里的 include/mesh/texture 都是相对该 XML 文件位置的相对路径，放别处会加载失败。
  - roslaunch 时显式带上已验证过的稳定控制参数
    （with_estimation:=true / wbc_frequency:=1000 / sensor_frequency:=1000），
    避免一键启动复现此前 challenge launch 默认值被降级导致的转向异常/摔倒问题。
"""

import atexit
import os
import signal
import subprocess
import sys
import time

VALID_SCENES = ("scene1", "scene2", "scene3")
SIM_PKG = "challenge_cup_simulator"
LAUNCH_FILE = "load_kuavo_mujoco_challenge.launch"

# 已验证过的稳定控制参数：必须显式传给 roslaunch，
# 即使将来 launch 默认值被改动，也能保证一键启动行为稳定。
STABLE_CONTROL_ARGS = {
    "with_estimation": "true",
    "wbc_frequency": "1000",
    "sensor_frequency": "1000",
}


class ChallengeSimLauncher:
    """挑战杯仿真环境自动启动器"""

    def __init__(self, scene, seed=0, robot_version=None):
        """
        Args:
            scene: "scene1" / "scene2" / "scene3"
            seed: 随机种子（仅影响 scene_builder 构建期物体随机化，默认 0）
            robot_version: 机器人版本号（默认 52）。挑战杯只发布 biped_s52，
                故固定默认 52，且**不**从 ROBOT_VERSION 环境变量读取——
                否则选手可借环境变量切到无哈希基线的版本，绕过完整性校验。
        """
        if scene not in VALID_SCENES:
            print(f"[FATAL] challenge_sim_launcher: 非法场景 '{scene}'，"
                  f"只支持 {VALID_SCENES}")
            sys.exit(1)
        self.scene = scene
        self.seed = seed
        self.robot_version = robot_version or 52
        self._launch_proc = None
        self._sim_pkg_path = None
        self._scene_file = None

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def start(self, node_name="challenge_task", timeout=120):
        """
        启动仿真环境并等待就绪。

        流程：
          1. 定位 challenge_cup_simulator 包路径；
          2. 校验场景源文件完整性（challenge_secret，防篡改）；
          3. 调用 scene_builder.py 生成带 seed 的场景 XML（写入包内 xml 目录）；
          4. roslaunch（自带 roscore），显式带稳定控制参数 + sceneFile 绝对路径；
          5. 等待 roscore 就绪并初始化 ROS 节点；
          6. 等待 /sensors_data_raw 出现，确认仿真就绪。
        """
        self._sim_pkg_path = self._find_sim_pkg_path()
        self._verify_integrity()
        self._scene_file = self._build_scene_xml()

        cmd = [
            "roslaunch", SIM_PKG, LAUNCH_FILE,
            f"scene_name:={self.scene}",
            f"sceneFile:={self._scene_file}",
            f"robot_version:={self.robot_version}",
        ]
        cmd += [f"{k}:={v}" for k, v in STABLE_CONTROL_ARGS.items()]

        print(f"[INFO] challenge_sim_launcher: 启动仿真: {' '.join(cmd)}")
        self._launch_proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
        # 注册退出清理，确保 Ctrl+C 或异常退出时关闭仿真
        atexit.register(self.stop)

        # 等待 roscore 就绪后初始化 ROS 节点
        self._wait_for_roscore(timeout=30)
        import rospy
        rospy.init_node(node_name, anonymous=True)

        rospy.loginfo("challenge_sim_launcher: 等待仿真就绪...")
        self._wait_for_sim(timeout)
        rospy.loginfo("challenge_sim_launcher: 仿真环境就绪。")

    def stop(self):
        """关闭仿真环境（终止整个 roslaunch 进程组）"""
        if self._launch_proc is not None:
            print("[INFO] challenge_sim_launcher: 关闭仿真环境...")
            try:
                os.killpg(os.getpgid(self._launch_proc.pid), signal.SIGTERM)
                self._launch_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self._launch_proc.pid), signal.SIGKILL)
                self._launch_proc.wait(timeout=5)
            except Exception:
                pass
            self._launch_proc = None

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #
    def _find_sim_pkg_path(self):
        """定位 challenge_cup_simulator 包路径（优先 rospkg，回退到源码相对路径）"""
        try:
            import rospkg
            return rospkg.RosPack().get_path(SIM_PKG)
        except Exception:
            # 回退：本脚本位于 challenge_cup_simulator/utils/，
            # 包根目录即上一级。
            here = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.abspath(os.path.join(here, ".."))
            if os.path.isfile(os.path.join(candidate, "package.xml")):
                return candidate
            print(f"[FATAL] challenge_sim_launcher: 找不到 {SIM_PKG} 包路径，"
                  f"请确认已 source 工作空间。")
            sys.exit(1)

    def _verify_integrity(self):
        """校验场景源文件完整性（challenge_secret，防篡改）。

        行为（默认 fail-closed）：
          - .so 缺失 -> [FATAL] 退出，禁止启动；
            仅当显式设置环境变量 CHALLENGE_SECRET_ALLOW_MISSING=1（开发环境）时，
            才降级为打印警告并继续。
          - .so 存在但校验不通过（文件被篡改）-> [FATAL] 退出，不启动仿真。
        """
        lib_dir = os.path.join(self._sim_pkg_path, "lib")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        try:
            from challenge_secret import verify_source_files
        except ImportError:
            if os.environ.get("CHALLENGE_SECRET_ALLOW_MISSING") == "1":
                print("[WARN] challenge_sim_launcher: 未找到 challenge_secret 模块（.so 缺失），"
                      "因 CHALLENGE_SECRET_ALLOW_MISSING=1 跳过完整性校验（仅开发环境可接受）。")
                return
            print("[FATAL] challenge_sim_launcher: 未找到 challenge_secret 模块（.so 缺失），"
                  "禁止启动仿真。开发环境如需放行，请设置 CHALLENGE_SECRET_ALLOW_MISSING=1。")
            sys.exit(1)

        passed, messages = verify_source_files(self._sim_pkg_path, self.robot_version)
        for msg in messages:
            print(f"[{'INFO' if passed else 'ERROR'}] challenge_sim_launcher: {msg}")
        if not passed:
            print("[FATAL] challenge_sim_launcher: 场景源文件校验失败，禁止启动仿真！")
            sys.exit(1)
        print("[INFO] challenge_sim_launcher: 场景源文件校验通过。")

    def _build_scene_xml(self):
        """调用 scene_builder.py 生成带 seed 的场景 XML，返回其绝对路径。

        生成文件放在 simulator 包内的 xml 目录，与原始 sceneN.xml 同级，
        以保证 XML 中 include/mesh/texture 的相对路径仍然有效。
        """
        builder = os.path.join(self._sim_pkg_path, "utils", "scene_builder.py")
        config = os.path.join(self._sim_pkg_path, "config", "scenes", f"{self.scene}.yaml")
        xml_dir = os.path.join(self._sim_pkg_path, "models", "biped_s52", "xml")
        out_path = os.path.join(xml_dir, f"_scene_{self.scene}_seed_{self.seed}.xml")

        for path in (builder, config):
            if not os.path.isfile(path):
                print(f"[FATAL] challenge_sim_launcher: 缺少文件 {path}")
                sys.exit(1)

        cmd = [
            sys.executable, builder, config,
            "--seed", str(self.seed),
            "-o", out_path,
        ]
        print(f"[INFO] challenge_sim_launcher: 生成场景: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0 or not os.path.isfile(out_path):
            print(f"[FATAL] challenge_sim_launcher: 场景生成失败 ({self.scene}, seed={self.seed})")
            sys.exit(1)
        print(f"[INFO] challenge_sim_launcher: 场景 XML -> {out_path}")
        return out_path

    def _wait_for_roscore(self, timeout):
        """等待 roscore 就绪"""
        import xmlrpc.client
        master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
        start = time.time()
        while time.time() - start < timeout:
            try:
                master = xmlrpc.client.ServerProxy(master_uri)
                master.getSystemState("")
                return
            except Exception:
                time.sleep(0.5)
        print(f"[WARN] challenge_sim_launcher: 等待 roscore 超时 ({timeout}s)")

    def _wait_for_sim(self, timeout):
        """等待仿真关键 topic /sensors_data_raw 出现"""
        import rospy
        start = time.time()
        while time.time() - start < timeout:
            if rospy.is_shutdown():
                return
            try:
                topics = [t[0] for t in rospy.get_published_topics()]
                if "/sensors_data_raw" in topics:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        rospy.logwarn("challenge_sim_launcher: 等待仿真就绪超时 (%ds)，继续执行...", timeout)
