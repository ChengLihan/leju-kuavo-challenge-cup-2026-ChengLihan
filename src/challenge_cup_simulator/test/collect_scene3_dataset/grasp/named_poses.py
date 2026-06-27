"""
grasp/named_poses.py — Named pose loading from YAML.
"""
import math
import os
from dataclasses import dataclass

import yaml

JOINT_ORDER = [
    "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
    "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
    "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
    "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POSE_PATH = os.path.join(SCRIPT_DIR, "configs", "named_poses.yaml")


@dataclass
class NamedPose:
    name: str
    joints_deg: list
    duration: float = 2.0


def load_poses(path=None):
    path = path or DEFAULT_POSE_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    poses = {}
    for name, pose_data in (data.get("poses") or {}).items():
        if "joints" in pose_data:
            joints = list(pose_data["joints"])
        else:
            joints = list(pose_data.get("left", [0] * 7)) + list(pose_data.get("right", [0] * 7))
        if len(joints) != 14:
            raise ValueError(f"pose '{name}' has {len(joints)} joints, expected 14")
        poses[name] = NamedPose(
            name=name,
            joints_deg=[float(v) for v in joints],
            duration=float(pose_data.get("duration", 2.0)),
        )
    return poses


def rad_to_deg(values):
    return [math.degrees(float(v)) for v in values]
