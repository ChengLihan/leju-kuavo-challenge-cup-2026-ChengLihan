"""
core/__init__.py — restructured scene3 data collection subpackage.

Modules:
  config_loader   — YAML loading, deep_update, config resolution
  named_poses     — NamedPose dataclass, pose loading
  arm_controller  — ArmTrajHold (arm trajectory publisher thread)
  gripper_controller — JointStateGripperHold, LejuClawCommandClient
  grasp_params    — calculated shelf distance / squat / bend params
"""

from .config_loader import deep_update, load_yaml, dump_yaml, ensure_repo_relative, resolve_pose_config
from .named_poses import NamedPose, load_named_poses, DEFAULT_JOINT_ORDER, ARM_JOINT_NAMES
from .arm_controller import ArmTrajHold
from .gripper_controller import JointStateGripperHold, LejuClawCommandClient
from .grasp_params import GraspParams, UPPER_PARAMS, LOWER_PARAMS
