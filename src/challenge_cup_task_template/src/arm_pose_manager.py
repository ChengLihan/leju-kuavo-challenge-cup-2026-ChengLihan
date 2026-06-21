#!/usr/bin/env python3
"""
手臂姿态管理模块：加载 named poses 并通过 /kuavo_arm_traj 发布。
"""

import os
import rospy
import yaml
import numpy as np


class ArmPoseManager:
    ARM_JOINT_NAMES = [
        "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
        "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
        "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
        "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
    ]

    def __init__(self, iface, config_path=None):
        self.iface = iface
        self._poses = {}

        if config_path is None:
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(pkg_dir, "config", "scene3_named_poses.yaml")

        self._load_poses(config_path)

    def _load_poses(self, config_path):
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            poses = data.get("poses", {})
            for name, pose_data in poses.items():
                left = pose_data.get("left", [0] * 7)
                right = pose_data.get("right", [0] * 7)
                self._poses[name] = {
                    "positions": left + right,
                    "duration": pose_data.get("duration", 2.0),
                    "description": pose_data.get("description", ""),
                }
            rospy.loginfo("ArmPoseManager: loaded %d poses from %s", len(self._poses), config_path)
        except Exception as e:
            rospy.logerr("ArmPoseManager: failed to load poses: %s", e)
            self._set_default_poses()

    def _set_default_poses(self):
        defaults = {
            "safe_home": [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0, 0],
            "upper_tray_pregrasp": [20, 0, 0, -30, 0, 0, 0, 10, -20, 0, -80, 0, 20, 0],
            "tray_transport": [10, 0, 0, -40, 0, 0, 0, 10, 0, 0, -60, 0, 10, 0],
            "box_preplace": [10, 0, 0, -40, 0, 0, 0, 20, -10, 0, -50, 0, 20, 0],
            "box_release": [10, 0, 0, -40, 0, 0, 0, 20, -10, 0, -70, 0, 0, 0],
            "box_after_release": [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0, 0],
            "finish_home": [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0, 0],
        }
        for name, pos in defaults.items():
            self._poses[name] = {"positions": pos, "duration": 2.0, "description": "default"}
        rospy.logwarn("ArmPoseManager: using default hardcoded poses")

    def get_pose(self, name):
        return self._poses.get(name)

    def move_to_named_pose(self, name):
        pose = self.get_pose(name)
        if pose is None:
            rospy.logerr("ArmPoseManager: unknown pose '%s'", name)
            return False

        positions = pose["positions"]
        duration = pose["duration"]

        rospy.loginfo("ArmPoseManager: moving to '%s' (%.1fs)", name, duration)

        steps = max(1, int(duration / 0.05))
        sleep_dt = duration / steps

        current = self.iface.get_arm_joint_positions()
        if current and len(current) == len(self.ARM_JOINT_NAMES):
            start_pos = [current.get(n, 0.0) for n in self.ARM_JOINT_NAMES]
        else:
            start_pos = [0.0] * len(self.ARM_JOINT_NAMES)

        for step in range(1, steps + 1):
            alpha = float(step) / steps
            interp = [(1.0 - alpha) * s + alpha * t for s, t in zip(start_pos, positions)]
            self.iface.publish_arm_joints(self.ARM_JOINT_NAMES, interp)
            rospy.sleep(sleep_dt)

        self.iface.publish_arm_joints(self.ARM_JOINT_NAMES, positions)
        rospy.sleep(duration * 0.3)
        return True

    def move_to_safe_home(self):
        return self.move_to_named_pose("safe_home")

    def move_to_upper_tray_pregrasp(self):
        return self.move_to_named_pose("upper_tray_pregrasp")

    def move_to_transport_pose(self):
        return self.move_to_named_pose("tray_transport")

    def move_to_box_preplace(self):
        return self.move_to_named_pose("box_preplace")

    def move_to_box_release(self):
        return self.move_to_named_pose("box_release")

    def move_to_finish_pose(self):
        return self.move_to_named_pose("finish_home")

    def hold_current_pose(self):
        current = self.iface.get_arm_joint_positions()
        if current:
            positions = [current.get(n, 0.0) for n in self.ARM_JOINT_NAMES]
            self.iface.publish_arm_joints(self.ARM_JOINT_NAMES, positions)
