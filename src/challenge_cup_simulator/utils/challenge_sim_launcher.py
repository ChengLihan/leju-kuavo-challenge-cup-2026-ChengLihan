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
  - seed 用于 challenge_secret(.so) 的运行时场景实例初始化。
  - scene_builder 只生成静态基准 XML，真实随机位置由仿真启动后写入 MuJoCo 内存。
  - 生成的基准 XML 写入 challenge_cup_simulator 包内的 xml 目录（不是 /tmp），
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
# 必须随机化的场景：运行时摆放任一环节失败均 fail-closed 退出，禁止用基准位悄悄跑错。
SCENES_REQUIRING_PLACEMENT = ("scene1", "scene2")
SIM_PKG = "challenge_cup_simulator"
LAUNCH_FILE = "load_kuavo_mujoco_challenge.launch"

# 已验证过的稳定控制参数：必须显式传给 roslaunch，
# 即使将来 launch 默认值被改动，也能保证一键启动行为稳定。
STABLE_CONTROL_ARGS = {
    "with_estimation": "true",
    "wbc_frequency": "1000",
    "sensor_frequency": "1000",
}


def _eval_mode():
    """评测模式：CHALLENGE_EVAL_MODE=1 时，日志不暴露 seed 和摆放后的真实坐标。
    本地调试默认关闭（保留 seed/坐标方便排查）。"""
    return os.environ.get("CHALLENGE_EVAL_MODE") == "1"


class ChallengeSimLauncher:
    """挑战杯仿真环境自动启动器"""

    def __init__(self, scene, seed=0, robot_version=None):
        """
        Args:
            scene: "scene1" / "scene2" / "scene3"
            seed: 随机种子（正式评测由组委会注入；scene1/scene2 用于运行时物体布局）
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
        self._cheat_monitor_proc = None
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
          3. 调用 scene_builder.py 生成静态场景 XML（写入包内 xml 目录）；
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

        # 运行时按 .so 计算的布局摆放物体（反作弊 B+C：真实位置不落进可读文件）
        self._place_objects()
        self._start_cheat_monitor()

    def stop(self):
        """关闭仿真环境（终止整个 roslaunch 进程组）"""
        if self._cheat_monitor_proc is not None:
            print("[INFO] challenge_sim_launcher: 关闭反作弊监控...")
            try:
                os.killpg(os.getpgid(self._cheat_monitor_proc.pid), signal.SIGTERM)
                self._cheat_monitor_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self._cheat_monitor_proc.pid), signal.SIGKILL)
                self._cheat_monitor_proc.wait(timeout=3)
            except Exception:
                pass
            self._cheat_monitor_proc = None

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
        """调用 scene_builder.py 生成静态场景 XML，返回其绝对路径。

        生成文件放在 simulator 包内的 xml 目录，与原始 sceneN.xml 同级，
        以保证 XML 中 include/mesh/texture 的相对路径仍然有效。
        """
        builder = os.path.join(self._sim_pkg_path, "utils", "scene_builder.py")
        config = os.path.join(self._sim_pkg_path, "config", "scenes", f"{self.scene}.yaml")
        xml_dir = os.path.join(self._sim_pkg_path, "models", "biped_s52", "xml")
        # 文件名不带 seed：避免从文件名泄露评测 seed（真实物体位置也不写进该文件）。
        out_path = os.path.join(xml_dir, f"_scene_{self.scene}_active.xml")

        for path in (builder, config):
            if not os.path.isfile(path):
                print(f"[FATAL] challenge_sim_launcher: 缺少文件 {path}")
                sys.exit(1)

        # --no-randomize：生成静态基准场景，不把随机位置写进 XML；
        # 真实位置在仿真就绪后由 _place_objects() 经 set_object_position 运行时摆放。
        cmd = [
            sys.executable, builder, config,
            "--seed", str(self.seed),
            "--no-randomize",
            "-o", out_path,
        ]
        # 评测模式不在日志里暴露 seed
        display_cmd = list(cmd)
        if _eval_mode():
            for i, tok in enumerate(display_cmd):
                if tok == "--seed" and i + 1 < len(display_cmd):
                    display_cmd[i + 1] = "***"
        print(f"[INFO] challenge_sim_launcher: 生成场景: {' '.join(display_cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0 or not os.path.isfile(out_path):
            seed_disp = "***" if _eval_mode() else self.seed
            print(f"[FATAL] challenge_sim_launcher: 场景生成失败 ({self.scene}, seed={seed_disp})")
            sys.exit(1)
        print(f"[INFO] challenge_sim_launcher: 场景 XML -> {out_path}")
        return out_path

    def _fail_placement(self, msg, required):
        """需要随机化的场景(scene1/scene2)摆放失败 -> fail-closed 退出；否则告警。"""
        import rospy
        if required:
            rospy.logfatal("challenge_sim_launcher: %s（场景 %s 必须随机化，禁止启动）",
                           msg, self.scene)
            self.stop()
            sys.exit(1)
        rospy.logwarn("challenge_sim_launcher: %s（场景 %s 无需随机化，跳过）", msg, self.scene)

    def _lock_object_position(self, required):
        """摆放完成后锁定 set_object_position：任务进行中禁止再挪物体（含 scene3 的静态物体）。
        必须随机化的场景若上锁失败 -> fail-closed（否则场景在可篡改状态下继续运行）。"""
        import rospy
        from std_srvs.srv import SetBool
        try:
            rospy.wait_for_service("set_object_position_lock", timeout=5)
            resp = rospy.ServiceProxy("set_object_position_lock", SetBool)(True)
            if not getattr(resp, "success", False):
                raise RuntimeError(getattr(resp, "message", "lock returned success=false"))
            rospy.loginfo("challenge_sim_launcher: 已锁定 set_object_position。")
        except Exception as exc:
            if required:
                rospy.logfatal("challenge_sim_launcher: 锁定 set_object_position 失败: %s"
                               "（必须随机化场景，禁止在可篡改状态下运行）", exc)
                self.stop()
                sys.exit(1)
            rospy.logwarn("challenge_sim_launcher: 锁定 set_object_position 失败: %s", exc)

    def _place_objects(self):
        """仿真就绪后，按 challenge_secret(.so) 计算的布局用 set_object_position 运行时摆放物体。
        真实位置只在运行时设置，不写进任何可读场景文件（反作弊 B+C）。
        scene1/scene2 必须随机化 -> 任一环节失败均 fail-closed 退出；scene3 无随机化则跳过。
        无论哪种场景，最后都锁定 set_object_position，禁止任务进行中再篡改物体。"""
        import rospy

        required = self.scene in SCENES_REQUIRING_PLACEMENT

        lib_dir = os.path.join(self._sim_pkg_path, "lib")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        try:
            from challenge_secret import get_object_layout
        except ImportError:
            return self._fail_placement("未找到 challenge_secret，无法运行时摆放", required)

        layout = get_object_layout(self.scene, self.seed)
        if not layout:
            if required:
                return self._fail_placement("场景需要随机布局但 .so 返回空", True)
            rospy.loginfo("challenge_sim_launcher: 场景 %s 无运行时布局，跳过摆放。", self.scene)
            self._lock_object_position(required)
            return

        from kuavo_msgs.srv import SetObjectPosition
        from std_srvs.srv import SetBool
        from geometry_msgs.msg import Point, Quaternion

        try:
            rospy.wait_for_service("set_object_position", timeout=15)
        except rospy.ROSException:
            return self._fail_placement("set_object_position 服务不可用", required)
        set_pos = rospy.ServiceProxy("set_object_position", SetObjectPosition)

        # 暂停仿真，避免摆放过程被看到瞬移；拿不到 sim_start 也不致命
        sim_ctrl = None
        try:
            rospy.wait_for_service("sim_start", timeout=5)
            sim_ctrl = rospy.ServiceProxy("sim_start", SetBool)
            sim_ctrl(False)
        except Exception:
            sim_ctrl = None

        all_ok = True
        for name, od in layout.items():
            pos = od["pos"]
            qw, qx, qy, qz = od["quat"]
            try:
                resp = set_pos(
                    object_name=name,
                    position=Point(x=pos[0], y=pos[1], z=pos[2]),
                    orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
                    randomize=False,
                    x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0, z_min=0.0, z_max=0.0,
                )
                if resp.success:
                    if _eval_mode():
                        rospy.loginfo("challenge_sim_launcher: 摆放 %s 完成", name)
                    else:
                        rospy.loginfo("challenge_sim_launcher: 摆放 %s -> (%.3f, %.3f, %.3f)",
                                      name, pos[0], pos[1], pos[2])
                else:
                    all_ok = False
                    rospy.logerr("challenge_sim_launcher: 摆放 %s 失败: %s", name, resp.message)
            except Exception as exc:
                all_ok = False
                rospy.logerr("challenge_sim_launcher: 摆放 %s 异常: %s", name, exc)

        if sim_ctrl is not None:
            try:
                sim_ctrl(True)
            except Exception:
                pass

        if not all_ok:
            return self._fail_placement("部分物体摆放失败", required)

        # 摆放成功后再上锁
        self._lock_object_position(required)
        rospy.loginfo("challenge_sim_launcher: 运行时摆放完成。")

    def _start_cheat_monitor(self):
        """摆放并上锁后启动 .so 内的监控：选手碰上帝视角接口即杀节点。"""
        import rospy

        lib_dir = os.path.join(self._sim_pkg_path, "lib")
        code = (
            "import sys; "
            "sys.path.insert(0, sys.argv[1]); "
            "import challenge_secret; "
            "challenge_secret.run_cheat_monitor()"
        )
        cmd = [sys.executable, "-c", code, lib_dir]
        self._cheat_monitor_proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
        time.sleep(0.5)
        if self._cheat_monitor_proc.poll() is not None:
            rospy.logfatal("challenge_sim_launcher: 反作弊监控启动失败，禁止继续运行。")
            self.stop()
            sys.exit(1)
        rospy.loginfo("challenge_sim_launcher: 反作弊监控已启动。")

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
