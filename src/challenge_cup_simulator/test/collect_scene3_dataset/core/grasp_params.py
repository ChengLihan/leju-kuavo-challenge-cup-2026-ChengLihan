"""
core/grasp_params.py — Calculated grasp parameters for Scene3 tray picking.

Derived from:
  - biped_s52 URDF/MuJoCo kinematic chain (forward kinematics)
  - scene3.xml tray ground-truth positions
  - named arm poses (upper_tray_pregrasp, lower_tray_pregrasp, etc.)
  - body squat / bend modeling

Values computed by scripts/calc_scene3_grasp_params.py and verified against
existing configs (scene3_collect.yaml, scene3_lower_tray_pick.yaml).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GraspParams:
    """Recommended grasp parameters for a shelf level."""
    shelf: str                     # "upper" or "lower"
    approach_shelf_distance: float  # recommended forward drive distance (m)
    base_to_tray_x: float          # base_link X → tray center X distance (m)
    shoulder_z_standing: float     # shoulder height above ground (m)
    tray_z: float                  # tray center Z above ground (m)
    squat_delta: float             # squat from nominal (m, negative = lower)
    bend_deg: float                # body forward bend angle (degrees)
    note: str


# === Upper shelf (z=1.15m) ===
UPPER_PARAMS = GraspParams(
    shelf="upper",
    approach_shelf_distance=1.00,   # verified against scene3_collect.yaml
    base_to_tray_x=0.85,            # base_link X → tray center X
    shoulder_z_standing=1.214,      # shoulder Z when standing upright
    tray_z=1.15,                    # upper tray center Z
    squat_delta=0.0,                # no squat needed
    bend_deg=0.0,                   # no bend needed
    note="Upper tray only 6cm below shoulder — stand upright, arm reaches easily.",
)

# === Lower shelf (z=0.75m) ===
LOWER_PARAMS = GraspParams(
    shelf="lower",
    approach_shelf_distance=1.35,   # verified against scene3_lower_tray_pick.yaml
    base_to_tray_x=1.30,            # need to stand further back
    shoulder_z_standing=1.214,      # shoulder Z when standing upright
    tray_z=0.75,                    # lower tray center Z
    squat_delta=-0.25,              # squat 25 cm (base_z: 0.82→0.57)
    bend_deg=20.0,                  # bend forward 20°
    note="Lower tray 46cm below shoulder — must squat 25cm + bend 20° + stand back.",
)

# Tray world positions from scene3.xml
TRAY_POSITIONS = {
    "smt_tray_1": (0.983, 0.35, 0.75),
    "smt_tray_2": (0.983, 0.15, 0.75),
    "smt_tray_3": (0.983, 0.05, 1.15),
    "smt_tray_4": (0.983, 0.25, 1.15),
    "smt_tray_5": (0.983, -0.15, 1.15),
}

# Robot kinematic constants
BASE_Z_NOMINAL = 0.82              # base_link Z above ground (standing)
SHOULDER_Z_NOMINAL = 1.214         # shoulder Z above ground (standing)
SHOULDER_Y_BASE = -0.2527          # right shoulder Y in base_link frame
ARM_UPPER_LEN = 0.284              # upper arm length (shoulder→elbow)
ARM_FOREARM_LEN = 0.116            # forearm + wrist + gripper length (elbow→EE)
ARM_TOTAL_LEN = 0.400              # total arm length when straight
