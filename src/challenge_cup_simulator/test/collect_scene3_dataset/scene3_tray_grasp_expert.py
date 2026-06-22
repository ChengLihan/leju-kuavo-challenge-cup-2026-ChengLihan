#!/usr/bin/env python3
"""Fixed named-pose expert for upper-tray extraction and waist stow in Scene3."""

import math
import os
import sys
import threading
from dataclasses import dataclass

import yaml

from scene3_success_checker import Scene3SuccessChecker
from scene3_rosbag_utils import wait_for_connection


ARM_JOINT_NAMES = [f"arm_joint_{i}" for i in range(1, 15)]
SCENE2_IK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "collect_scene2_dataset"))
IK_MODE_THREE_POINT_MIXED = 0x06
DEFAULT_JOINT_ORDER = [
    "l_arm_pitch",
    "l_arm_roll",
    "l_arm_yaw",
    "l_forearm_pitch",
    "l_hand_yaw",
    "l_hand_pitch",
    "l_hand_roll",
    "r_arm_pitch",
    "r_arm_roll",
    "r_arm_yaw",
    "r_forearm_pitch",
    "r_hand_yaw",
    "r_hand_pitch",
    "r_hand_roll",
]


@dataclass
class NamedPose:
    name: str
    joints_deg: list
    duration: float
    description: str = ""


def load_named_poses(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    poses = {}
    for name, pose_data in (data.get("poses") or {}).items():
        if "joints" in pose_data:
            joints = list(pose_data["joints"])
        else:
            joints = list(pose_data.get("left", [0] * 7)) + list(pose_data.get("right", [0] * 7))
        if len(joints) != 14:
            raise ValueError(f"pose {name} has {len(joints)} joints, expected 14")
        poses[name] = NamedPose(
            name=name,
            joints_deg=[float(v) for v in joints],
            duration=float(pose_data.get("duration", 2.0)),
            description=str(pose_data.get("description", "")),
        )
    return data.get("joint_order", DEFAULT_JOINT_ORDER), poses


def rad_to_deg(values):
    return [math.degrees(float(v)) for v in values]


class ArmTrajHold:
    def __init__(self, pub, degrees_list, hz=100.0):
        self._pub = pub
        self._hz = float(hz)
        self._degrees = self._validate(degrees_list)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="scene3_arm_traj_hold", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_degrees(self, degrees_list):
        with self._lock:
            self._degrees = self._validate(degrees_list)

    def current_degrees(self):
        with self._lock:
            return list(self._degrees)

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState

        rate = rospy.Rate(self._hz)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            with self._lock:
                degrees = list(self._degrees)
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ARM_JOINT_NAMES
            msg.position = degrees
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break

    @staticmethod
    def _validate(degrees_list):
        values = [float(v) for v in degrees_list]
        if len(values) != 14:
            raise ValueError(f"arm command has {len(values)} joints, expected 14")
        return values


class JointStateGripperHold:
    def __init__(self, pub, cfg, hz=100.0):
        self._pub = pub
        self._hz = float(hz)
        self._cfg = cfg
        self._left = float(cfg.get("open_position", 0.0))
        self._right = float(cfg.get("open_position", 0.0))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="scene3_gripper_hold", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def open(self):
        value = float(self._cfg.get("open_position", 0.0))
        with self._lock:
            self._left = value
            self._right = value

    def close(self, arm):
        open_value = float(self._cfg.get("open_position", 0.0))
        close_value = float(self._cfg.get("close_position", 255.0))
        with self._lock:
            self._left = close_value if arm == "left" else open_value
            self._right = close_value if arm == "right" else open_value

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState

        rate = rospy.Rate(self._hz)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            with self._lock:
                left = self._left
                right = self._right
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ["left_gripper_joint", "right_gripper_joint"]
            msg.position = [left, right]
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break


class LejuClawCommandClient:
    def __init__(self, cfg, active_arm):
        self.cfg = cfg
        self.active_arm = active_arm
        self.backend = cfg.get("backend", "joint_state")
        self._pub = None
        self._srv = None

    def setup(self, timeout):
        import rospy

        if self.backend == "leju_topic":
            from kuavo_msgs.msg import lejuClawCommand

            self._pub = rospy.Publisher(
                self.cfg.get("leju_command_topic", "/leju_claw_command"),
                lejuClawCommand,
                queue_size=10,
            )
            wait_for_connection(self._pub, timeout)
        elif self.backend == "leju_service":
            from kuavo_msgs.srv import controlLejuClaw

            service = self.cfg.get("leju_service", "/control_robot_leju_claw")
            rospy.wait_for_service(service, timeout=timeout)
            self._srv = rospy.ServiceProxy(service, controlLejuClaw)

    def open(self):
        return self._send(float(self.cfg.get("leju_open_position", 10.0)), both=True)

    def close(self):
        return self._send(float(self.cfg.get("leju_close_position", 90.0)), both=False)

    def _send(self, position, both=False):
        if self.backend not in ("leju_topic", "leju_service"):
            return True
        from kuavo_msgs.msg import endEffectorData

        names = ["left_claw", "right_claw"] if both else [f"{self.active_arm}_claw"]
        data = endEffectorData()
        data.name = names
        data.position = [float(position)] * len(names)
        data.velocity = [float(self.cfg.get("velocity", 90.0))] * len(names)
        data.effort = [float(self.cfg.get("effort", 1.0))] * len(names)
        if self.backend == "leju_topic":
            from kuavo_msgs.msg import lejuClawCommand

            msg = lejuClawCommand()
            msg.data = data
            self._pub.publish(msg)
            return True
        if self.backend == "leju_service":
            from kuavo_msgs.srv import controlLejuClawRequest

            req = controlLejuClawRequest()
            req.data = data
            resp = self._srv(req)
            return bool(resp.success)
        return True


class Scene3TrayGraspExpert:
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

    def print_plan(self):
        print("Scene3 upper-tray expert plan:")
        if self.truth_ik_cfg.get("enabled", False):
            print("  truth_ik      -> enabled: pregrasp/approach/extract/lift from live /mujoco/qpos + IK")
        for stage, pose_name in self.plan_steps():
            pose = self.poses[pose_name]
            print(f"  {stage:14s} -> {pose_name:24s} {pose.duration:.2f}s  {pose.joints_deg}")
        print(f"  {'close_gripper':14s} -> active_arm={self.active_arm}")

    def plan_steps(self):
        steps = [("reset", "safe_home")]
        if "scene3_ready_pose" in self.poses:
            steps.append(("pregrasp", "scene3_ready_pose"))
        steps.extend(("pregrasp", name) for name in self._waypoints("pregrasp_waypoints", ["upper_tray_pregrasp"]))
        steps.extend(("approach", name) for name in self._waypoints("approach_waypoints", ["upper_tray_edge_approach"]))
        steps.extend(("extract", name) for name in self._waypoints("extract_waypoints", ["upper_tray_extract_mid", "upper_tray_extract_out"]))
        steps.extend(("lift", name) for name in self._waypoints("lift_waypoints", ["upper_tray_lift"]))
        steps.extend(("stow", name) for name in self._waypoints("stow_waypoints", ["waist_stow_pose", "finish_hold_pose"]))
        return steps

    def setup_ros(self, timeout=20.0):
        import rospy
        from sensor_msgs.msg import JointState
        from kuavo_msgs.msg import armTargetPoses

        self._arm_pub = rospy.Publisher(
            self.expert_cfg.get("arm_command_topic", "/kuavo_arm_traj"),
            JointState,
            queue_size=10,
        )
        wait_for_connection(self._arm_pub, timeout)
        self._target_pub = rospy.Publisher(
            self.expert_cfg.get("arm_target_topic", "/kuavo_arm_target_poses"),
            armTargetPoses,
            queue_size=10,
        )
        initial = self._read_current_arm_degrees(timeout)
        self.arm_hold = ArmTrajHold(
            self._arm_pub,
            initial,
            hz=self.expert_cfg.get("arm_command_hz", 100.0),
        )
        self.arm_hold.start()

        if self.gripper_cfg.get("backend", "joint_state") == "joint_state":
            pub = rospy.Publisher(
                self.gripper_cfg.get("joint_command_topic", "/gripper/command"),
                JointState,
                queue_size=10,
            )
            wait_for_connection(pub, timeout)
            self.gripper_hold = JointStateGripperHold(
                pub,
                self.gripper_cfg,
                hz=self.gripper_cfg.get("command_hz", 100.0),
            )
            self.gripper_hold.start()
        else:
            self.leju_client.setup(timeout)

    def shutdown(self):
        if self.arm_hold is not None:
            self.arm_hold.stop()
        if self.gripper_hold is not None:
            self.gripper_hold.stop()

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
        rospy.loginfo("scene3 collect: arm mode -> external: %s", resp.message)

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

    def prepare_robot(self):
        self.set_arm_mode_external()
        self.publish_head_target()
        self.open_gripper()
        self.move_to_named_pose("safe_home", stage="pregrasp")
        if "scene3_ready_pose" in self.poses:
            self.move_to_named_pose("scene3_ready_pose", stage="pregrasp")
        self.move_to_pregrasp()

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
        return self._move_waypoints(
            "extract_waypoints",
            ["upper_tray_extract_mid", "upper_tray_extract_out"],
            stage="extract",
        )

    def lift_tray(self):
        if self._truth_ik_enabled():
            return self._move_truth_ik_stage("lift")
        return self._move_waypoints("lift_waypoints", ["upper_tray_lift"], stage="lift")

    def move_to_waist_stow(self):
        return self._move_waypoints(
            "stow_waypoints",
            ["waist_stow_pose", "finish_hold_pose"],
            stage="stow",
        )

    def _waypoints(self, key, default):
        if not self.avoid_cfg.get("enabled", True):
            return [name for name in default if name in self.poses]
        configured = self.avoid_cfg.get(key, default)
        waypoints = [str(name) for name in configured if str(name) in self.poses]
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
        steps = max(1, int(round(float(pose.duration) * float(self.expert_cfg.get("arm_command_hz", 100.0)))))
        rate = rospy.Rate(float(self.expert_cfg.get("arm_command_hz", 100.0)))
        for step in range(steps + 1):
            if rospy.is_shutdown():
                break
            alpha = float(step) / float(steps)
            point = [start[i] + (target[i] - start[i]) * alpha for i in range(14)]
            self.arm_hold.set_degrees(point)
            if self.observer is not None and step % max(1, steps // 10) == 0:
                self.observer.publish_expert_action(stage or name, point, self._gripper_cmd_value(stage))
            if step < steps:
                rate.sleep()
        rospy.sleep(0.15)
        return True

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
                timeout=float(self.truth_ik_cfg.get("timeout_sec", 20.0))
            ),
            execute_arm_motion_cb=self._execute_ik_motion,
            publish_arm_gripper_close_cb=lambda _arm: self.close_gripper(),
            sleep_cb=self._ros_sleep,
            loginfo_cb=self._ros_loginfo,
            logwarn_cb=self._ros_logwarn,
        )
        current = self._read_current_arm_radians(timeout=float(self.truth_ik_cfg.get("timeout_sec", 20.0)))
        locked_other = list(current[:7]) if self.active_arm == "right" else list(current[7:14])
        move_arm_ik_once(
            runtime=runtime,
            active_arm=self.active_arm,
            active_pos=target["pos"],
            locked_other_arm_joints=locked_other,
            active_quat=target["quat"],
            label=f"scene3_{stage}",
            constraint_mode=int(self.truth_ik_cfg.get("constraint_mode", IK_MODE_THREE_POINT_MIXED)),
            pos_cost_weight=float(self.truth_ik_cfg.get("pos_cost_weight", 2.0)),
            move_time=float(target["duration"]),
            settle_time=float(self.truth_ik_cfg.get("settle_time", 0.2)),
        )
        return True

    def _truth_ik_stage_target(self, stage):
        checker = Scene3SuccessChecker(self.cfg)
        info = checker.measure_target_gripper_distance(
            timeout=float(self.truth_ik_cfg.get("timeout_sec", 20.0))
        )
        target_base = info["target_base_xyz"]
        offsets = self.truth_ik_cfg.get("stage_offsets", {})
        default_offsets = {
            "pregrasp": {"x": -0.12, "y": 0.0, "z": 0.02, "duration": 2.0},
            "approach": {"x": -0.05, "y": 0.0, "z": 0.00, "duration": 1.2},
            "extract_mid": {"x": -0.16, "y": 0.0, "z": 0.00, "duration": 1.0},
            "extract_out": {"x": -0.28, "y": 0.0, "z": 0.02, "duration": 1.2},
            "lift": {"x": -0.28, "y": 0.0, "z": 0.10, "duration": 1.0},
        }
        stage_offset = dict(default_offsets.get(stage, {}))
        stage_offset.update(offsets.get(stage, {}))
        pos = [
            float(target_base[0]) + float(stage_offset.get("x", 0.0)),
            float(target_base[1]) + float(stage_offset.get("y", 0.0)),
            float(target_base[2]) + float(stage_offset.get("z", 0.0)),
        ]
        quat = [float(v) for v in self.truth_ik_cfg.get("quat_xyzw", [0.0, -0.70682518, 0.0, 0.70738827])]
        duration = float(stage_offset.get("duration", self.truth_ik_cfg.get("move_time", 1.4)))
        print(
            "[INFO] truth_ik {stage}: target_base={target} ik_pos={pos} quat={quat}".format(
                stage=stage,
                target=[round(float(v), 3) for v in target_base],
                pos=[round(float(v), 3) for v in pos],
                quat=[round(float(v), 4) for v in quat],
            )
        )
        return {"pos": pos, "quat": quat, "duration": duration}

    def _execute_ik_motion(self, start_degrees, target_degrees, move_time, settle):
        import rospy

        if self.arm_hold is None:
            raise RuntimeError("expert.setup_ros() must be called before IK motion")
        start = [float(v) for v in start_degrees]
        target = [float(v) for v in target_degrees]
        steps = max(1, int(round(float(move_time) * float(self.expert_cfg.get("arm_command_hz", 100.0)))))
        rate = rospy.Rate(float(self.expert_cfg.get("arm_command_hz", 100.0)))
        for step in range(steps + 1):
            if rospy.is_shutdown():
                break
            alpha = float(step) / float(steps)
            point = [start[i] + (target[i] - start[i]) * alpha for i in range(14)]
            self.arm_hold.set_degrees(point)
            if self.observer is not None and step % max(1, steps // 10) == 0:
                self.observer.publish_expert_action("truth_ik", point, self._gripper_cmd_value("truth_ik"))
            if step < steps:
                rate.sleep()
        rospy.sleep(float(settle))

    def safe_stop(self):
        import rospy
        from geometry_msgs.msg import Twist

        try:
            pub = rospy.Publisher(self.cfg.get("topics", {}).get("cmd_vel", "/cmd_vel"), Twist, queue_size=10)
            msg = Twist()
            for _ in range(3):
                pub.publish(msg)
                rospy.sleep(0.05)
        except Exception:
            pass

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
            sensorsData,
            timeout=float(timeout),
        )
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
            sensorsData,
            timeout=float(timeout),
        )
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


def resolve_pose_config(script_dir, cfg):
    pose_cfg = cfg.get("expert", {}).get("pose_config", "configs/scene3_named_poses.yaml")
    if os.path.isabs(pose_cfg):
        return pose_cfg
    return os.path.abspath(os.path.join(script_dir, pose_cfg))
