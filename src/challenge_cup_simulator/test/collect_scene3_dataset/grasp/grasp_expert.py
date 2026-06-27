"""Scene3 upper-tray pregrasp + close expert (uses local arm_ik.ArmIK)."""
import math, os, yaml
from .arm_controller import ArmTrajHold
from .arm_ik import ArmIK
from .gripper_controller import GripperController
from .mujoco_utils import (build_qpos_and_body_maps, pose_from_qpos,
                           pose_from_body_freejoint, yaw_from_quat_wxyz,
                           world_to_base_xyz)
from .named_poses import load_poses, rad_to_deg
from .ros_utils import REPO_ROOT, wait_publisher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认夹爪姿态 RPY (度数): 手爪水平朝下 → pitch = -90°
DEFAULT_EULER_DEG = [0.0, -90.0, 0.0]  # [roll, pitch, yaw]


def euler_deg_to_quat_xyzw(roll_deg, pitch_deg, yaw_deg):
    """欧拉角 (degrees, Z-Y-X / RPY) → 夹爪四元数 [x, y, z, w] (base_link 系).

    表示的是末端执行器在 base_link 系下的实际空间姿态, 不是关节角度。

    roll  = 绕 X 轴 (前后翻转)
    pitch = 绕 Y 轴 (上下俯仰)
    yaw   = 绕 Z 轴 (左右偏转)
    """
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return [x, y, z, w]


def _parse_orientation(raw):
    """从 yaml 解析夹爪空间姿态 → 四元数 [x,y,z,w] (base_link 系)。

    支持两种格式:
      quat:      [x, y, z, w]  直接四元数 (base_link 系)
      euler_deg: [roll, pitch, yaw]  欧拉角 RPY 度 (base_link 系)
    都不写则返回默认姿态 (朝下)。
    """
    if raw is None:
        return euler_deg_to_quat_xyzw(*DEFAULT_EULER_DEG)
    if isinstance(raw, dict):
        if "quat" in raw:
            return [float(v) for v in raw["quat"]]
        if "euler_deg" in raw:
            r, p, y = [float(v) for v in raw["euler_deg"]]
            return euler_deg_to_quat_xyzw(r, p, y)
    return euler_deg_to_quat_xyzw(*DEFAULT_EULER_DEG)


class GraspExpert:
    def __init__(self, params_path=None, pose_config_path=None):
        with open(params_path or os.path.join(SCRIPT_DIR, "configs", "grasp_params.yaml")) as f:
            self.params = yaml.safe_load(f) or {}
        self.poses = load_poses(pose_config_path)
        self._tray = self.params.get("target", {}).get("tray_name", "smt_tray_4")
        self._arm = self.params.get("target", {}).get("active_arm", "right")
        self.arm_hold = None
        self.gripper = None
        self.arm_ik = None
        self._arm_pub = None
        self._safe_home_rad = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def setup(self, timeout=20.0):
        import rospy
        from sensor_msgs.msg import JointState

        self._arm_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
        wait_publisher(self._arm_pub, timeout)
        pub = rospy.Publisher("/gripper/command", JointState, queue_size=10)
        wait_publisher(pub, timeout)

        self.arm_hold = ArmTrajHold(self._arm_pub, self._read_deg(timeout), 100.0)
        self.arm_hold.start()
        self.gripper = GripperController(pub, 100.0)
        self.gripper.start()

        self.arm_ik = ArmIK(
            read_joints_cb=lambda: self._read_rad(20.0),
            arm_hold=self.arm_hold,
        )

    def shutdown(self):
        if self.arm_hold:
            self.arm_hold.stop()
        if self.gripper:
            self.gripper.stop()

    # ------------------------------------------------------------------
    # misc actions
    # ------------------------------------------------------------------

    def set_arm_ext(self, timeout=20.0):
        import rospy
        from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

        rospy.wait_for_service("/arm_traj_change_mode", timeout=timeout)
        r = rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(
            changeArmCtrlModeRequest(control_mode=2))
        if not r.result:
            raise RuntimeError("arm mode rejected")

    def set_head(self, y=0.0, p=-15.0):
        import rospy
        from kuavo_msgs.msg import robotHeadMotionData

        pub = rospy.Publisher("/robot_head_motion_data", robotHeadMotionData, queue_size=10)
        wait_publisher(pub, 5)
        m = robotHeadMotionData()
        m.joint_data = [float(y), float(p)]
        for _ in range(5):
            pub.publish(m)
            rospy.sleep(0.1)
        rospy.sleep(0.4)

    def open_gripper(self):
        self.gripper.open()

    def close_gripper(self):
        import rospy
        self.gripper.close(self._arm)
        rospy.sleep(0.5)

    # ------------------------------------------------------------------
    # task-level stages
    # ------------------------------------------------------------------

    def prepare(self):
        self.set_arm_ext()
        self.set_head()
        self.open_gripper()
        self._move_named("safe_home")
        import rospy
        rospy.sleep(0.5)
        self._safe_home_rad = self._read_rad(5.0)

    def run_pregrasp(self):
        return self._ik("pregrasp")

    def safe_stop(self):
        import rospy
        from geometry_msgs.msg import Twist

        try:
            p = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
            for _ in range(3):
                p.publish(Twist())
                rospy.sleep(0.05)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # IK (uses local ArmIK)
    # ------------------------------------------------------------------

    def _ik(self, stage):
        import rospy

        t = self._ik_target(stage)
        lo = (list(self._safe_home_rad[:7])
              if self._safe_home_rad else list(self._read_rad(20.0)[:7]))

        try:
            if t.get("waypoints"):
                # trajectory 模式: 途经多个 waypoint
                self.arm_ik.move_along_trajectory(
                    waypoints=t["waypoints"],
                    arm=self._arm,
                    locked_other_arm_joints=lo,
                    constraint_mode=0x06,
                    pos_cost_weight=2.0,
                    settle_time=0.2,
                )
            else:
                # 单步模式: 直接到目标
                self.arm_ik.move_to(
                    target_pos=t["pos"],
                    target_quat=t["quat"],
                    arm=self._arm,
                    duration=float(t["duration"]),
                    settle_time=0.2,
                    locked_other_arm_joints=lo,
                    constraint_mode=0x06,
                    pos_cost_weight=2.0,
                )
        except RuntimeError as e:
            pm = math.sqrt(sum(float(v) ** 2 for v in t["pos"]))
            rospy.logwarn("IK %s failed: pos=%s dist=%.3fm | %s", stage,
                          [round(float(v), 3) for v in t["pos"]], pm, e)
            raise RuntimeError(f"IK_FAILED {stage}: {e}") from e

    def _ik_target(self, stage):
        o = self.params.get("ik_stages", {}).get(stage, {})
        import rospy
        from std_msgs.msg import Float64MultiArray

        xml = os.path.join(REPO_ROOT, "src", "challenge_cup_simulator", "models",
                           "biped_s52", "xml", "_scene_scene3_active.xml")
        qm, _, bj = build_qpos_and_body_maps(xml)
        q = list(rospy.wait_for_message("/mujoco/qpos", Float64MultiArray, timeout=20.0).data)
        tw = pose_from_qpos(q, qm, self._tray)
        bx, bq = pose_from_body_freejoint(q, bj, "base_link")
        tb = world_to_base_xyz(tw, bx, yaw_from_quat_wxyz(bq))

        # 默认姿态从 yaml 读取
        default_quat = _parse_orientation(o.get("orientation"))
        pos = [
            tb[0] + float(o.get("offset_x", -0.20)),
            tb[1] + float(o.get("offset_y", 0.0)),
            tb[2] + float(o.get("offset_z", 0.08)),
        ]
        rospy.loginfo("IK %s: pos=%s (base_link) | quat=%s", stage,
                      [round(float(v), 3) for v in pos],
                      [round(float(v), 4) for v in default_quat])
        result = {"pos": pos, "quat": default_quat,
                  "duration": float(o.get("duration", 2.0))}

        # 途经点
        raw_wps = o.get("waypoints") or []
        if raw_wps:
            wps = []
            for wp in raw_wps:
                if not isinstance(wp, dict):
                    continue
                wp_quat = _parse_orientation(wp.get("orientation")) if "orientation" in wp else None
                wps.append({
                    "pos": [
                        tb[0] + float(wp.get("offset_x", 0.0)),
                        tb[1] + float(wp.get("offset_y", 0.0)),
                        tb[2] + float(wp.get("offset_z", 0.0)),
                    ],
                    "quat": wp_quat,  # None 时由 arm_ik 自动继承
                    "duration": float(wp.get("duration", 1.0)),
                })
            result["waypoints"] = wps
            rospy.loginfo("IK %s: trajectory mode, %d waypoints", stage, len(wps))
        return result

    # ------------------------------------------------------------------
    # named-pose motion (joint-space interpolation)
    # ------------------------------------------------------------------

    def _move_named(self, name):
        import rospy

        pose = self.poses[name]
        start = self._read_deg(5.0)
        target = list(pose.joints_deg)
        n = max(1, int(round(pose.duration * 100.0)))
        r = rospy.Rate(100.0)
        for i in range(n + 1):
            if rospy.is_shutdown():
                break
            a = i / n
            self.arm_hold.set_degrees(
                [start[j] + (target[j] - start[j]) * a for j in range(14)])
            if i < n:
                r.sleep()
        rospy.sleep(0.15)

    # ------------------------------------------------------------------
    # joint reading
    # ------------------------------------------------------------------

    def _read_deg(self, to):
        import rospy
        from kuavo_msgs.msg import sensorsData

        j = list(rospy.wait_for_message("/sensors_data_raw", sensorsData,
                                        timeout=float(to)).joint_data.joint_q)
        return rad_to_deg(j[13:27]) if len(j) >= 27 else rad_to_deg(j[12:26])

    def _read_rad(self, to):
        import rospy
        from kuavo_msgs.msg import sensorsData

        j = list(rospy.wait_for_message("/sensors_data_raw", sensorsData,
                                        timeout=float(to)).joint_data.joint_q)
        return [float(v) for v in j[13:27]] if len(j) >= 27 else [float(v) for v in j[12:26]]
