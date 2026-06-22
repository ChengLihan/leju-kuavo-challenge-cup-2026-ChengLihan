"""
core/named_poses.py — Named pose dataclass and YAML loading.

Supports both "joints" (flat 14) and "left"/"right" (split 7+7) formats.
"""

import math
from dataclasses import dataclass, field

import yaml

DEFAULT_JOINT_ORDER = [
    "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
    "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
    "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
    "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
]

ARM_JOINT_NAMES = [f"arm_joint_{i}" for i in range(1, 15)]


@dataclass
class NamedPose:
    name: str
    joints_deg: list
    duration: float = 2.0
    description: str = ""


def load_named_poses(path):
    """Load named poses from a YAML file.

    Returns (joint_order, poses_dict) where poses_dict maps name → NamedPose.
    """
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
            description=str(pose_data.get("description", "")),
        )

    return data.get("joint_order", DEFAULT_JOINT_ORDER), poses


def rad_to_deg(values):
    """Convert a list of radian values to degrees."""
    return [math.degrees(float(v)) for v in values]
