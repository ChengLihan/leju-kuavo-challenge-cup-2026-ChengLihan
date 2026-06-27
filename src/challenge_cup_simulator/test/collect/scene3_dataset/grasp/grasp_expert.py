"""Scene3 upper-tray pregrasp + close expert (simplified)."""
import math, os, sys, yaml
from .arm_controller import ArmTrajHold
from .gripper_controller import GripperController
from .named_poses import load_poses, rad_to_deg
from .ros_utils import wait_publisher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE2_IK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "collect_scene2_dataset"))

class GraspExpert:
    def __init__(self, params_path=None, pose_config_path=None):
        with open(params_path or os.path.join(SCRIPT_DIR, "configs", "grasp_params.yaml")) as f:
            self.params = yaml.safe_load(f) or {}
        self.poses = load_poses(pose_config_path)
        self._tray = self.params.get("target", {}).get("tray_name", "smt_tray_4")
        self._arm = self.params.get("target", {}).get("active_arm", "right")
        self.arm_hold = None; self.gripper = None; self._arm_pub = None
        self._safe_home_rad = None

    def setup(self, timeout=20.0):
        import rospy; from sensor_msgs.msg import JointState
        self._arm_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10); wait_publisher(self._arm_pub, timeout)
        pub = rospy.Publisher("/gripper/command", JointState, queue_size=10); wait_publisher(pub, timeout)
        self.arm_hold = ArmTrajHold(self._arm_pub, self._read_deg(timeout), 100.0); self.arm_hold.start()
        self.gripper = GripperController(pub, 100.0); self.gripper.start()

    def shutdown(self):
        if self.arm_hold: self.arm_hold.stop()
        if self.gripper: self.gripper.stop()

    def set_arm_ext(self, timeout=20.0):
        import rospy; from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
        rospy.wait_for_service("/arm_traj_change_mode", timeout=timeout)
        r = rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)(changeArmCtrlModeRequest(control_mode=2))
        if not r.result: raise RuntimeError("arm mode rejected")

    def set_head(self, y=0.0, p=-15.0):
        import rospy; from kuavo_msgs.msg import robotHeadMotionData
        pub = rospy.Publisher("/robot_head_motion_data", robotHeadMotionData, queue_size=10); wait_publisher(pub, 5)
        m = robotHeadMotionData(); m.joint_data = [float(y), float(p)]
        for _ in range(5): pub.publish(m); rospy.sleep(0.1)
        rospy.sleep(0.4)

    def open_gripper(self): self.gripper.open()

    def close_gripper(self):
        import rospy; self.gripper.close(self._arm); rospy.sleep(0.5)

    def prepare(self):
        self.set_arm_ext(); self.set_head(); self.open_gripper()
        self._move_named("safe_home")
        import rospy; rospy.sleep(0.5); self._safe_home_rad = self._read_rad(5.0)

    def run_pregrasp(self): return self._ik("pregrasp")

    def safe_stop(self):
        import rospy; from geometry_msgs.msg import Twist
        try:
            p = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
            for _ in range(3): p.publish(Twist()); rospy.sleep(0.05)
        except: pass

    def _move_named(self, name):
        import rospy; pose = self.poses[name]; start = self._read_deg(5.0); target = list(pose.joints_deg)
        n = max(1, int(round(pose.duration * 100.0))); r = rospy.Rate(100.0)
        for i in range(n + 1):
            if rospy.is_shutdown(): break
            a = i / n; self.arm_hold.set_degrees([start[j] + (target[j] - start[j]) * a for j in range(14)])
            if i < n: r.sleep()
        rospy.sleep(0.15)

    def _ik(self, stage):
        import rospy
        if SCENE2_IK_DIR not in sys.path: sys.path.insert(0, SCENE2_IK_DIR)
        from scene2_part_grasp_ik import GraspRuntime, move_arm_ik_once
        t = self._ik_target(stage)
        rt = GraspRuntime(world_to_ee_offset_x=0.0, world_to_ee_offset_y_left=0.0, world_to_ee_offset_y_right=0.0,
            world_to_ee_offset_z=0.0, pre_grasp_z_offset=0.0, grasp_position_tolerance=0.08,
            orientation_tolerance_rad=math.radians(70.0), gripper_close_time=0.5, timeout=20.0,
            move_time=float(t["duration"]), settle_time=0.2, ik_mode_pos_hard_ori_hard=0x06,
            read_current_arm_joints_cb=lambda: self._read_rad(20.0),
            execute_arm_motion_cb=self._ik_motion,
            publish_arm_gripper_close_cb=lambda _: self.close_gripper(),
            sleep_cb=lambda s: rospy.sleep(s), loginfo_cb=lambda m,*a: rospy.loginfo(m,*a),
            logwarn_cb=lambda m,*a: rospy.logwarn(m,*a))
        cur = self._read_rad(20.0)
        lo = list(self._safe_home_rad[:7]) if self._safe_home_rad else list(cur[:7])
        try:
            move_arm_ik_once(runtime=rt, active_arm=self._arm, active_pos=t["pos"],
                locked_other_arm_joints=lo, active_quat=t["quat"], label=f"scene3_{stage}",
                constraint_mode=0x06, pos_cost_weight=2.0, move_time=float(t["duration"]), settle_time=0.2)
        except RuntimeError as e:
            pm = math.sqrt(sum(float(v)**2 for v in t["pos"]))
            rospy.logwarn("IK %s failed: pos=%s dist=%.3fm | %s", stage,
                          [round(float(v),3) for v in t["pos"]], pm, e)
            raise RuntimeError(f"IK_FAILED {stage}: {e}") from e

    def _ik_target(self, stage):
        o = self.params.get("ik_stages", {}).get(stage, {})
        import rospy; from std_msgs.msg import Float64MultiArray
        from scene3_success_checker import (build_qpos_and_body_maps, pose_from_qpos,
            pose_from_body_freejoint, yaw_from_quat_wxyz, world_to_base_xyz)
        from .ros_utils import REPO_ROOT
        xml = os.path.join(REPO_ROOT, "src", "challenge_cup_simulator", "models", "biped_s52", "xml",
                           "_scene_scene3_active.xml")
        qm, _, bj = build_qpos_and_body_maps(xml)
        q = list(rospy.wait_for_message("/mujoco/qpos", Float64MultiArray, timeout=20.0).data)
        tw = pose_from_qpos(q, qm, self._tray)
        bx, bq = pose_from_body_freejoint(q, bj, "base_link")
        tb = world_to_base_xyz(tw, bx, yaw_from_quat_wxyz(bq))
        pos = [tb[0] + float(o.get("offset_x", -0.20)), tb[1] + float(o.get("offset_y", 0.0)),
               tb[2] + float(o.get("offset_z", 0.08))]
        qt = [0.0, -0.70682518, 0.0, 0.70738827]
        rospy.loginfo("IK %s: pos=%s quat=%s", stage, [round(float(v),3) for v in pos],
                      [round(float(v),4) for v in qt])
        return {"pos": pos, "quat": qt, "duration": float(o.get("duration", 2.0))}

    def _ik_motion(self, sd, td, mt, st):
        import rospy
        s, t = [float(v) for v in sd], [float(v) for v in td]
        n = max(1, int(round(mt * 100.0))); r = rospy.Rate(100.0)
        for i in range(n + 1):
            if rospy.is_shutdown(): break
            a = i / n; self.arm_hold.set_degrees([s[j] + (t[j] - s[j]) * a for j in range(14)])
            if i < n: r.sleep()
        rospy.sleep(float(st))

    def _read_deg(self, to):
        import rospy; from kuavo_msgs.msg import sensorsData
        j = list(rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=float(to)).joint_data.joint_q)
        return rad_to_deg(j[13:27]) if len(j) >= 27 else rad_to_deg(j[12:26])

    def _read_rad(self, to):
        import rospy; from kuavo_msgs.msg import sensorsData
        j = list(rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=float(to)).joint_data.joint_q)
        return [float(v) for v in j[13:27]] if len(j) >= 27 else [float(v) for v in j[12:26]]
