"""
grasp/named_poses.py
"""
import math, os, yaml
from dataclasses import dataclass

JOINT_ORDER = ["l_arm_pitch","l_arm_roll","l_arm_yaw","l_forearm_pitch","l_hand_yaw","l_hand_pitch","l_hand_roll",
               "r_arm_pitch","r_arm_roll","r_arm_yaw","r_forearm_pitch","r_hand_yaw","r_hand_pitch","r_hand_roll"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class NamedPose:
    name: str; joints_deg: list; duration: float = 2.0

def load_poses(path=None):
    path = path or os.path.join(SCRIPT_DIR, "configs", "named_poses.yaml")
    data = yaml.safe_load(open(path, encoding="utf-8")) or {}
    poses = {}
    for name, d in (data.get("poses") or {}).items():
        if "joints" in d: j = list(d["joints"])
        else: j = list(d.get("left", [0]*7)) + list(d.get("right", [0]*7))
        assert len(j) == 14
        poses[name] = NamedPose(name=name, joints_deg=[float(v) for v in j], duration=float(d.get("duration", 2.0)))
    return poses

def rad_to_deg(v): return [math.degrees(float(x)) for x in v]
