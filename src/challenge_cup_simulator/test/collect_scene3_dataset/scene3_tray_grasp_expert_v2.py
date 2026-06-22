#!/usr/bin/env python3
"""
scene3_tray_grasp_expert_v2.py — Refactored named-pose + truth-IK grasp expert.

Delegates low-level classes to core/ subpackage:
  - core.named_poses        → NamedPose, load_named_poses, DEFAULT_JOINT_ORDER
  - core.arm_controller     → ArmTrajHold
  - core.gripper_controller → JointStateGripperHold, LejuClawCommandClient
  - core.grasp_params       → UPPER_PARAMS, TRAY_POSITIONS (reference only)
  - core.config_loader      → resolve_pose_config (re-exported)

Interface is identical to the original Scene3TrayGraspExpert.
"""

import math
import os
import sys
import threading

from core.arm_controller import ArmTrajHold
from core.config_loader import resolve_pose_config
from core.gripper_controller import JointStateGripperHold, LejuClawCommandClient
from core.named_poses import (
    ARM_JOINT_NAMES,
    DEFAULT_JOINT_ORDER,
    NamedPose,
    load_named_poses,
    rad_to_deg,
)
from scene3_rosbag_utils import wait_for_connection
from scene3_success_checker import Scene3SuccessChecker

# Truth IK: imported at runtime from collect_scene2_dataset
SCENE2_IK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "collect_scene2_dataset"))
IK_MODE_THREE_POINT_MIXED = 0x06


# ──────────────────────────── Expert Class ────────────────────────────

class Scene3TrayGraspExpert:
    """Orchestrates arm/gripper motion for Scene3 upper-tray grasp extraction."""

    def __init__(self, cfg, pose_config_path, observer=None):
        self.cfg = cfg
        self.observer = observer
        self.joint_order, self.poses = load_named_poses(pose_config_path)
        self.active_arm = cfg.get("robot", {}).get("active_arm", "right")
        self.expert_cfg = cfg.get("expert", {})
        self.avoid_cfg = self.expert_cfg.get("shelf_avoidance", {})
        self.truth_ik_cfg = self.expert_cfg.get("truth_ik", {})
        self.gripper_cfg = self.expert_cfg.get("gripper", {})
        self.arm_hold = None
        self.gripper_hold = None
        self.leju_client = LejuClawCommandClient(self.gripper_cfg, self.active_arm)
        self._arm_pub = None
        self._target_pub = None

    # ── Plan / introspection ─────────────────────────────────────────

    def print_plan(self):
        print("Scene3 upper-tray expert plan (v2):")
        if self.truth_ik_cfg.get("enabled", False):
            print("  truth_ik  → enabled: pregrasp/approach/extract/lift from live /mujoco/qpos + IK")
        for stage, pose_name in self.plan_steps():
            pose = self.poses[pose_name]
            print(f"  {stage:14s} → {pose_name:24s} {pose.duration:.2f}s  {pose.joints_deg}")

    def plan_steps(self):
        steps = [("reset", "safe_home")]
        if "scene3_ready_pose" in self.poses:
            steps.append(("pregrasp", "scene3_ready_pose"))
        steps.extend(("pregrasp", n) for n in self._waypoints("pregrasp_waypoints", ["upper_tray_pregrasp"]))
        steps.extend(("approach", n) for n in self._waypoints("approach_waypoints", ["upper_tray_edge_approach"]))
        steps.extend(("extract", n) for n in self._waypoints("extract_waypoints", ["upper_tray_extract_mid", "upper_tray_extract_out"]))
        steps.extend(("lift",    n) for n in self._waypoints("lift_waypoints",    ["upper_tray_lift"]))
        steps.extend(("stow",    n) for n in self._waypoints("stow_waypoints",    ["waist_stow_pose", "finish_hold_pose"]))
        return steps

    # ── ROS setup / teardown ─────────────────────────────────────────

    def setup_ros(self, timeout=20.0):
        import rospy
        from kuavo_msgs.msg import armTargetPoses
        from sensor_msgs.msg import JointState

        self._arm_pub = rospy.Publisher(
            self.expert_cfg.get("arm_command_topic", "/kuavo_arm_traj"),
            JointState, queue_size=10,
        )
        wait_for_connection(self._arm_pub, timeout)

        self._target_pub = rospy.Publisher(
            self.expert_cfg.get("arm_target_topic", "/kuavo_arm_target_poses"),
            armTargetPoses, queue_size=10,
        )

        initial = self._read_current_arm_degrees(timeout)
        self.arm_hold = ArmTrajHold(self._arm_pub, initial,
                                     hz=self.expert_cfg.get("arm_command_hz", 100.0))
        self.arm_hold.start()

        if self.gripper_cfg.get("backend", "joint_state") == "joint_state":
            pub = rospy.Publisher(
                self.gripper_cfg.get("joint_command_topic", "/gripper/command"),
                JointState, queue_size=10,
            )
            wait_for_connection(pub, timeout)
            self.gripper_hold = JointStateGripperHold(pub, self.gripper_cfg,
                                                       hz=self.gripper_cfg.get("command_hz", 100.0))
            self.gripper_hold.start()
        else:
            self.leju_client.setup(timeout)

    def shutdown(self):
        if self.arm_hold is not None:
            self.arm_hold.stop()
        if self.gripper_hold is not None:
            self.gripper_hold.stop()

    # ── Arm mode helpers ─────────────────────────────────────────────

    def set_arm_mode_external(self, timeout=20.0):
        import rospy
        from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

        service = self.expert_cfg.get("arm_mode_service", "/arm_traj_change_mode")
        rospy.wait_for_service(service, timeout=timeout)
        proxy = rospy.ServiceProxy(service, changeArmCtrlMode)
        req = changeArmCtrlModeRequest()
        req.control_mode = int(self.expert_cfg.get("external_control_mode", 2))
        resp = proxy(req)
        if not resp.result:
            raise RuntimeError(f"{service} rejected external control: {resp.message}")

    def restore_arm_mode_if_needed(self, timeout=10.0):
        if not self.expert_cfg.get("restore_arm_mode_on_success", False):
            return
        import rospy
        from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

        service = self.expert_cfg.get("arm_mode_service", "/arm_traj_change_mode")
        rospy.wait_for_service(service, timeout=timeout)
        proxy = rospy.ServiceProxy(service, changeArmCtrlMode)
        req = changeArmCtrlModeRequest()
        req.control_mode = int(self.expert_cfg.get("auto_swing_mode", 1))
        resp = proxy(req)
        if not resp.result:
            rospy.logwarn("scene3 collect: failed to restore arm mode: %s", resp.message)

    # ── Head / Gripper ───────────────────────────────────────────────

    def publish_head_target(self, timeout=20.0):
        import rospy
        from kuavo_msgs.msg import robotHeadMotionData

        robot_cfg = self.cfg.get("robot", {})
        topic = self.cfg.get("topics", {}).get("head_command", "/robot_head_motion_data")
        pub = rospy.Publisher(topic, robotHeadMotionData, queue_size=10)
        wait_for_connection(pub, timeout)
        msg = robotHeadMotionData()
        msg.joint_data = [
            float(robot_cfg.get("head_yaw_deg", 0.0)),
            float(robot_cfg.get("head_pitch_deg", -15.0)),
        ]
        for _ in range(5):
            pub.publish(msg)
            rospy.sleep(0.1)
        rospy.sleep(0.4)

    def open_gripper(self):
        if self.gripper_hold is not None:
            self.gripper_hold.open()
            return True
        return self.leju_client.open()

    def close_gripper(self):
        import rospy

        if self.gripper_hold is not None:
            self.gripper_hold.close(self.active_arm)
            ok = True
        else:
            ok = self.leju_client.close()
        rospy.sleep(float(self.gripper_cfg.get("close_wait_sec", 0.5)))
        if self.observer is not None:
            q = self.arm_hold.current_degrees() if self.arm_hold is not None else [0.0] * 14
            self.observer.publish_expert_action("close_gripper", q, 1.0)
        return ok

    # ── Robot preparation ────────────────────────────────────────────

    def prepare_robot(self):
        self.set_arm_mode_external()
        self.publish_head_target()
        self.open_gripper()
        self.move_to_named_pose("safe_home", stage="pregrasp")
        if "scene3_ready_pose" in self.poses:
            self.move_to_named_pose("scene3_ready_pose", stage="pregrasp")
        self.move_to_pregrasp()
        self.zero_hand_roll()

    def zero_hand_roll(self, duration=0.8):
        import rospy

        if self.arm_hold is None:
            return
        current = self._read_current_arm_degrees(timeout=5.0)
        target = list(current)
        target[13] = 0.0  # r_hand_roll → index 13
        hz = float(self.expert_cfg.get("arm_command_hz", 100.0))
        steps = max(1, int(round(float(duration) * hz)))
        rate = rospy.Rate(hz)
        for step in range(steps + 1):
            if rospy.is_shutdown():
                break
            alpha = float(step) / float(steps)
            point = [current[i] + (target[i] - current[i]) * alpha for i in range(14)]
            self.arm_hold.set_degrees(point)
            if step < steps:
                rate.sleep()
        rospy.sleep(0.15)

    # ── High-level motion stages ─────────────────────────────────────

    def move_to_pregrasp(self):
        if self._truth_ik_enabled():
            return self._move_truth_ik_stage("pregrasp")
        return self._move_waypoints("pregrasp_waypoints", ["upper_tray_pregrasp"], stage="pregrasp")

    def approach_tray_edge(self):
        if self._truth_ik_enabled():
            return self._move_truth_ik_stage("approach")
        return self._move_waypoints("approach_waypoints", ["upper_tray_edge_approach"], stage="approach")

    def extract_tray(self):
        if self._truth_ik_enabled():
            self._move_truth_ik_stage("extract_mid")
            self._move_truth_ik_stage("extract_out")
            return True
        return self._move_waypoints("extract_waypoints",
                                     ["upper_tray_extract_mid", "upper_tray_extract_out"],
                                     stage="extract")

    def lift_tray(self):
        if self._truth_ik_enabled():
            return self._move_truth_ik_stage("lift")
        return self._move_waypoints("lift_waypoints", ["upper_tray_lift"], stage="lift")

    def move_to_waist_stow(self):
        return self._move_waypoints("stow_waypoints",
                                     ["waist_stow_pose", "finish_hold_pose"],
                                     stage="stow")

    # ── Named-pose motion ────────────────────────────────────────────

    def _waypoints(self, key, default):
        if not self.avoid_cfg.get("enabled", True):
            return [n for n in default if n in self.poses]
        configured = self.avoid_cfg.get(key, default)
        waypoints = [str(n) for n in configured if str(n) in self.poses]
        if not waypoints:
            raise RuntimeError(f"no valid expert waypoint configured for {key}")
        return waypoints

    def _move_waypoints(self, key, default, stage):
        for name in self._waypoints(key, default):
            self.move_to_named_pose(name, stage=stage)
        return True

    def move_to_named_pose(self, name, stage=None):
        import rospy

        if name not in self.poses:
            raise RuntimeError(f"unknown named pose: {name}")
        if self.arm_hold is None:
            raise RuntimeError("expert.setup_ros() must be called before motion")

        pose = self.poses[name]
        start = self._read_current_arm_degrees(timeout=5.0)
        target = list(pose.joints_deg)

        if self.expert_cfg.get("publish_arm_target_poses", False):
            self._publish_arm_target(target, pose.duration)

        hz = float(self.expert_cfg.get("arm_command_hz", 100.0))
        steps = max(1, int(round(float(pose.duration) * hz)))
        rate = rospy.Rate(hz)
        for step in range(steps + 1):
            if rospy.is_shutdown():
                break
            alpha = float(step) / float(steps)
            point = [start[i] + (target[i] - start[i]) * alpha for i in range(14)]
            self.arm_hold.set_degrees(point)
            if self.observer is not None and step % max(1, steps // 10) == 0:
                self.observer.publish_expert_action(stage or name, point,
                                                     self._gripper_cmd_value(stage))
            if step < steps:
                rate.sleep()
        rospy.sleep(0.15)
        return True

    # ── Truth IK (runtime import from collect_scene2_dataset) ────────

    def _truth_ik_enabled(self):
        return bool(self.truth_ik_cfg.get("enabled", False))

    def _move_truth_ik_stage(self, stage):
        if SCENE2_IK_DIR not in sys.path:
            sys.path.insert(0, SCENE2_IK_DIR)
        from scene2_part_grasp_ik import GraspRuntime, move_arm_ik_once

        target = self._truth_ik_stage_target(stage)
        runtime = GraspRuntime(
            world_to_ee_offset_x=0.0,
            world_to_ee_offset_y_left=0.0,
            world_to_ee_offset_y_right=0.0,
            world_to_ee_offset_z=0.0,
            pre_grasp_z_offset=0.0,
            grasp_position_tolerance=float(self.truth_ik_cfg.get("position_tolerance_m", 0.08)),
            orientation_tolerance_rad=math.radians(float(self.truth_ik_cfg.get("orientation_tolerance_deg", 70.0))),
            gripper_close_time=float(self.gripper_cfg.get("close_wait_sec", 0.5)),
            timeout=float(self.truth_ik_cfg.get("timeout_sec", 20.0)),
            move_time=float(target["duration"]),
            settle_time=float(self.truth_ik_cfg.get("settle_time", 0.2)),
            ik_mode_pos_hard_ori_hard=int(self.truth_ik_cfg.get("constraint_mode", IK_MODE_THREE_POINT_MIXED)),
            read_current_arm_joints_cb=lambda: self._read_current_arm_radians(
                timeout=float(self.truth_ik_cfg.get("timeout_sec", 20.0))),
            execute_arm_motion_cb=self._execute_ik_motion,
            publish_arm_gripper_close_cb=lambda _arm: self.close_gripper(),
            sleep_cb=self._ros_sleep,
            loginfo_cb=self._ros_loginfo,
            logwarn_cb=self._ros_logwarn,
        )

        current = self._read_current_arm_radians(timeout=float(self.truth_ik_cfg.get("timeout_sec", 20.0)))
        locked_other = list(current[:7]) if self.active_arm == "right" else list(current[7:14])

        active_pos = target["pos"]
        active_quat = target["quat"]
        constraint_mode = int(self.truth_ik_cfg.get("constraint_mode", IK_MODE_THREE_POINT_MIXED))
        ik_label = f"scene3_{stage}"

        try:
            move_arm_ik_once(
                runtime=runtime, active_arm=self.active_arm,
                active_pos=active_pos, locked_other_arm_joints=locked_other,
                active_quat=active_quat, label=ik_label,
                constraint_mode=constraint_mode,
                pos_cost_weight=float(self.truth_ik_cfg.get("pos_cost_weight", 2.0)),
                move_time=float(target["duration"]),
                settle_time=float(self.truth_ik_cfg.get("settle_time", 0.2)),
            )
        except RuntimeError as exc:
            pos_mag = math.sqrt(sum(float(v) ** 2 for v in active_pos))
            rospy.logwarn("truth_ik %s IK failed: pos=%s quat=%s dist=%.3fm | %s",
                          stage, [round(float(v), 4) for v in active_pos],
                          [round(float(v), 4) for v in active_quat], pos_mag, exc)
            if pos_mag > 0.70:
                rospy.logwarn("IK target >0.70m from base_link. "
                              "Try: --named-pose-mode, reduce stage_offsets, "
                              "or move robot closer (--approach-shelf-distance).")
            raise RuntimeError(
                f"IK_FAILED: /ik/two_arm_hand_pose_cmd_srv failed for stage={stage} "
                f"active_arm={self.active_arm} "
                f"pos={[round(float(v), 3) for v in active_pos]} "
                f"quat={[round(float(v), 4) for v in active_quat]} "
                f"dist_from_origin={pos_mag:.3f}m | original: {exc}"
            ) from exc
        return True

    def _truth_ik_stage_target(self, stage):
        checker = Scene3SuccessChecker(self.cfg)
        info = checker.measure_target_gripper_distance(
            timeout=float(self.truth_ik_cfg.get("timeout_sec", 20.0)))
        target_base = info["target_base_xyz"]
        offsets = self.truth_ik_cfg.get("stage_offsets", {})
        default_offsets = {
            "pregrasp":    {"x": -0.06, "y": 0.0, "z": 0.08, "duration": 2.0},
            "approach":    {"x": -0.02, "y": 0.0, "z": 0.06, "duration": 1.2},
            "lift":        {"x":  0.00, "y": 0.0, "z": 0.12, "duration": 0.8},
            "extract_mid": {"x": -0.12, "y": 0.0, "z": 0.12, "duration": 1.2},
            "extract_out": {"x": -0.22, "y": 0.0, "z": 0.12, "duration": 1.4},
        }
        stage_offset = dict(default_offsets.get(stage, {}))
        stage_offset.update(offsets.get(stage, {}))
        pos = [
            float(target_base[0]) + float(stage_offset.get("x", 0.0)),
            float(target_base[1]) + float(stage_offset.get("y", 0.0)),
            float(target_base[2]) + float(stage_offset.get("z", 0.0)),
        ]
        quat = [float(v) for v in self.truth_ik_cfg.get("quat_xyzw",
                                                          [0.0, -0.70682518, 0.0, 0.70738827])]
        duration = float(stage_offset.get("duration", self.truth_ik_cfg.get("move_time", 1.4)))
        import rospy
        rospy.loginfo("truth_ik %s: target_base=%s ik_pos=%s quat=%s",
                      stage, [round(float(v), 3) for v in target_base],
                      [round(float(v), 3) for v in pos],
                      [round(float(v), 4) for v in quat])
        return {"pos": pos, "quat": quat, "duration": duration}

    def _execute_ik_motion(self, start_degrees, target_degrees, move_time, settle):
        import rospy

        if self.arm_hold is None:
            raise RuntimeError("expert.setup_ros() must be called before IK motion")
        start = [float(v) for v in start_degrees]
        target = [float(v) for v in target_degrees]
        hz = float(self.expert_cfg.get("arm_command_hz", 100.0))
        steps = max(1, int(round(float(move_time) * hz)))
        rate = rospy.Rate(hz)
        for step in range(steps + 1):
            if rospy.is_shutdown():
                break
            alpha = float(step) / float(steps)
            point = [start[i] + (target[i] - start[i]) * alpha for i in range(14)]
            self.arm_hold.set_degrees(point)
            if self.observer is not None and step % max(1, steps // 10) == 0:
                self.observer.publish_expert_action("truth_ik", point,
                                                     self._gripper_cmd_value("truth_ik"))
            if step < steps:
                rate.sleep()
        rospy.sleep(float(settle))

    def safe_stop(self):
        import rospy
        from geometry_msgs.msg import Twist

        try:
            pub = rospy.Publisher(self.cfg.get("topics", {}).get("cmd_vel", "/cmd_vel"),
                                   Twist, queue_size=10)
            msg = Twist()
            for _ in range(3):
                pub.publish(msg)
                rospy.sleep(0.05)
        except Exception:
            pass

    # ── Internal helpers ─────────────────────────────────────────────

    def _publish_arm_target(self, degrees, duration):
        from kuavo_msgs.msg import armTargetPoses

        msg = armTargetPoses()
        msg.times = [float(duration)]
        msg.values = [float(v) for v in degrees]
        self._target_pub.publish(msg)

    def _read_current_arm_degrees(self, timeout):
        import rospy
        from kuavo_msgs.msg import sensorsData

        msg = rospy.wait_for_message(
            self.cfg.get("topics", {}).get("sensors", "/sensors_data_raw"),
            sensorsData, timeout=float(timeout))
        joint_q = list(msg.joint_data.joint_q)
        if len(joint_q) >= 27:
            return rad_to_deg(joint_q[13:27])
        if len(joint_q) >= 26:
            return rad_to_deg(joint_q[12:26])
        raise RuntimeError(f"/sensors_data_raw joint_q has {len(joint_q)} values")

    def _read_current_arm_radians(self, timeout):
        import rospy
        from kuavo_msgs.msg import sensorsData

        msg = rospy.wait_for_message(
            self.cfg.get("topics", {}).get("sensors", "/sensors_data_raw"),
            sensorsData, timeout=float(timeout))
        joint_q = list(msg.joint_data.joint_q)
        if len(joint_q) >= 27:
            return [float(v) for v in joint_q[13:27]]
        if len(joint_q) >= 26:
            return [float(v) for v in joint_q[12:26]]
        raise RuntimeError(f"/sensors_data_raw joint_q has {len(joint_q)} values")

    @staticmethod
    def _ros_sleep(seconds):
        import rospy
        rospy.sleep(float(seconds))

    @staticmethod
    def _ros_loginfo(message, *args):
        import rospy
        rospy.loginfo(message, *args)

    @staticmethod
    def _ros_logwarn(message, *args):
        import rospy
        rospy.logwarn(message, *args)

    @staticmethod
    def _gripper_cmd_value(stage):
        return 1.0 if stage in ("close_gripper", "extract", "lift", "stow") else 0.0
