"""
grasp/arm_ik.py — Arm IK solver with full trajectory support.

Features:
  - solve(target_pos, target_quat, arm) → IK-computed 14-element joint angles (rad)
  - move_to(target_pos, target_quat, arm) → single-step move with pose control
  - move_along_trajectory(waypoints, arm) → multi-waypoint trajectory with per-point
    position and optional orientation

Usage example::

    from .arm_controller import ArmTrajHold
    from .arm_ik import ArmIK

    arm_ik = ArmIK(read_joints_cb=read_rad, arm_hold=arm_hold)

    # single target
    arm_ik.move_to(
        target_pos=[0.5, 0.0, 0.9],
        target_quat=[0.0, -0.70682518, 0.0, 0.70738827],
        arm="right",
        duration=2.0,
    )

    # trajectory with waypoints
    waypoints = [
        {"pos": [0.5, 0.0, 0.8],            "duration": 1.5},
        {"pos": [0.5, 0.0, 1.0],
         "quat": [0.0, -0.707, 0.0, 0.707], "duration": 1.0},
        {"pos": [0.4, 0.0, 1.0],            "duration": 0.5},
    ]
    arm_ik.move_along_trajectory(waypoints, arm="right")
"""
import math
import threading
from dataclasses import dataclass, field

USE_CUSTOM_IK_PARAM = True
JOINT_ANGLES_AS_Q0 = True


@dataclass
class IKSolverConfig:
    """Solver-level parameters."""
    timeout: float = 20.0
    constraint_mode: int = 0x06
    pos_cost_weight: float = 2.0


@dataclass
class Waypoint:
    """Single waypoint in a trajectory."""
    pos: list       # [x, y, z] – required
    quat: list = None  # [x, y, z, w] – optional; interpolated if absent
    duration: float = 1.0  # seconds to reach this waypoint from the previous one


@dataclass
class Trajectory:
    """Ordered series of waypoints."""
    arm: str = "right"
    waypoints: list = field(default_factory=list)
    settle_time: float = 0.2
    ik_config: IKSolverConfig = field(default_factory=IKSolverConfig)
    locked_other_arm_joints: list = None  # 7-element rad list or None


class ArmIK:
    """Solve arm IK and execute motions along trajectories.

    Integrates with ROS IK services under the hood.  Callers must provide
    two hooks so this class stays transport-agnostic:

    :param read_joints_cb: () → 14-element joint angles in **radians**.
    :param arm_hold:        ``ArmTrajHold`` instance for publishing commands.
    """

    def __init__(self, read_joints_cb=None, arm_hold=None):
        self._read_joints = read_joints_cb
        self._arm_hold = arm_hold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, target_pos, target_quat, arm="right",
              current_joints=None, locked_other_arm_joints=None,
              constraint_mode=0x06, pos_cost_weight=2.0, timeout=20.0):
        """Solve IK for a single end-effector pose.

        Returns 14-element joint angles in **radians**.
        """
        if current_joints is None:
            current_joints = self._read_rad()

        cfg = IKSolverConfig(
            timeout=float(timeout),
            constraint_mode=int(constraint_mode),
            pos_cost_weight=float(pos_cost_weight),
        )
        return _call_single_arm_ik(
            current_joints=[float(v) for v in current_joints],
            active_arm=arm,
            active_pos=[float(v) for v in target_pos],
            active_quat=list(target_quat) if target_quat is not None else None,
            locked_other_arm_joints=(
                [float(v) for v in locked_other_arm_joints]
                if locked_other_arm_joints is not None else None
            ),
            cfg=cfg,
        )

    def move_to(self, target_pos, target_quat, arm="right",
                duration=2.0, settle_time=0.2,
                locked_other_arm_joints=None,
                constraint_mode=0x06, pos_cost_weight=2.0):
        """Move end-effector to a single target pose (one IK step)."""
        current = self._read_rad()
        ik_q = self.solve(
            target_pos=target_pos,
            target_quat=target_quat,
            arm=arm,
            current_joints=current,
            locked_other_arm_joints=locked_other_arm_joints,
            constraint_mode=constraint_mode,
            pos_cost_weight=pos_cost_weight,
        )
        start_q, target_deg14 = self._prepare_joint_cmd(
            current, ik_q, arm, locked_other_arm_joints,
        )
        self._execute_motion(start_q, target_deg14, float(duration), float(settle_time))

    def move_along_trajectory(self, waypoints, arm="right",
                              locked_other_arm_joints=None,
                              constraint_mode=0x06, pos_cost_weight=2.0,
                              settle_time=0.2):
        """Execute a full trajectory defined by waypoints.

        Each waypoint is a dict: ``{"pos": [x,y,z], "quat": [x,y,z,w],
        "duration": 1.0}``.  ``pos`` is required; ``quat`` and ``duration``
        are optional.

        The solver pre-computes IK for every waypoint, then runs them in
        sequence with smooth joint-space interpolation.
        """
        import rospy

        cfg = IKSolverConfig(
            constraint_mode=int(constraint_mode),
            pos_cost_weight=float(pos_cost_weight),
        )
        other_lock = (
            [float(v) for v in locked_other_arm_joints]
            if locked_other_arm_joints is not None else None
        )

        wp_parsed = _normalize_waypoints(waypoints)

        # Pre-compute IK for every waypoint
        joint_targets = []  # list of 14-rad lists
        current_joints = self._read_rad()
        for i, wp in enumerate(wp_parsed):
            if wp.quat is None:
                quat = _resolve_quat(wp_parsed, i, current_joints, arm, cfg)
            else:
                quat = list(wp.quat)
            ik_q = _call_single_arm_ik(
                current_joints=list(current_joints),
                active_arm=arm,
                active_pos=[float(v) for v in wp.pos],
                active_quat=quat,
                locked_other_arm_joints=other_lock,
                cfg=cfg,
            )
            joint_targets.append(ik_q)
            current_joints = ik_q  # chain IK seeds for continuity

        # Execute waypoints sequentially
        current = self._read_rad()
        for i, wp in enumerate(wp_parsed):
            ik_q = joint_targets[i]
            start_deg, target_deg = self._prepare_joint_cmd(
                current, ik_q, arm, other_lock,
            )
            rospy.loginfo("arm_ik waypoint %d/%d: pos=%s dur=%.1fs",
                          i + 1, len(wp_parsed),
                          [round(float(v), 3) for v in wp.pos],
                          float(wp.duration))
            self._execute_motion(
                start_deg, target_deg, float(wp.duration),
                float(settle_time) if i == len(wp_parsed) - 1 else 0.0,
            )
            current = ik_q

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_rad(self):
        if self._read_joints is None:
            raise RuntimeError("read_joints_cb not set")
        j = [float(v) for v in self._read_joints()]
        if len(j) != 14:
            raise ValueError(f"expected 14 joints, got {len(j)}")
        return j

    def _prepare_joint_cmd(self, current_rad, target_rad, arm, locked_other):
        """Build start (degrees) and target (degrees) 14-element lists."""
        start = list(current_rad)
        if locked_other is not None:
            if arm == "left":
                start[7:14] = locked_other
            elif arm == "right":
                start[:7] = locked_other
        return _rad_to_deg(start), _rad_to_deg(target_rad)

    def _execute_motion(self, start_deg, target_deg, duration, settle_time):
        """Interpolate in joint space and publish via ArmTrajHold."""
        import rospy

        s = [float(v) for v in start_deg]
        t = [float(v) for v in target_deg]
        n = max(1, int(round(duration * 100.0)))
        r = rospy.Rate(100.0)
        for i in range(n + 1):
            if rospy.is_shutdown():
                break
            a = i / n
            self._arm_hold.set_degrees(
                [s[j] + (t[j] - s[j]) * a for j in range(14)]
            )
            if i < n:
                r.sleep()
        if settle_time > 0:
            rospy.sleep(float(settle_time))


# ==========================================================================
# IK service helpers (self-contained – no dependency on scene2)
# ==========================================================================

def _make_ik_param(constraint_mode, pos_cost_weight):
    from kuavo_msgs.msg import ikSolveParam

    p = ikSolveParam()
    p.major_optimality_tol = 1e-3
    p.major_feasibility_tol = 1e-3
    p.minor_feasibility_tol = 1e-3
    p.major_iterations_limit = 100
    p.oritation_constraint_tol = 1e-3
    p.pos_constraint_tol = 1e-3
    p.pos_cost_weight = float(pos_cost_weight)
    p.constraint_mode = int(constraint_mode)
    return p


def _call_fk(joint_angles_rad, timeout):
    import rospy
    from kuavo_msgs.srv import fkSrv

    rospy.wait_for_service("/ik/fk_srv", timeout=float(timeout))
    resp = rospy.ServiceProxy("/ik/fk_srv", fkSrv)(list(joint_angles_rad))
    if not resp.success:
        raise RuntimeError("/ik/fk_srv returned success=false")
    return resp.hand_poses


def _call_two_hands_ik(current_rad14, left_pos, right_pos,
                       left_quat, right_quat, constraint_mode,
                       pos_cost_weight, timeout):
    import rospy
    from kuavo_msgs.msg import twoArmHandPoseCmd
    from kuavo_msgs.srv import twoArmHandPoseCmdSrv

    req = twoArmHandPoseCmd()
    req.ik_param = _make_ik_param(int(constraint_mode), float(pos_cost_weight))
    req.use_custom_ik_param = USE_CUSTOM_IK_PARAM
    req.joint_angles_as_q0 = JOINT_ANGLES_AS_Q0

    cur = [float(v) for v in current_rad14]
    req.hand_poses.left_pose.joint_angles = cur[:7]
    req.hand_poses.right_pose.joint_angles = cur[7:14]
    req.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    req.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    req.hand_poses.left_pose.pos_xyz = [float(v) for v in left_pos]
    req.hand_poses.left_pose.quat_xyzw = [float(v) for v in left_quat]
    req.hand_poses.right_pose.pos_xyz = [float(v) for v in right_pos]
    req.hand_poses.right_pose.quat_xyzw = [float(v) for v in right_quat]

    rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=float(timeout))
    resp = rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)(req)
    if not resp.success:
        raise RuntimeError(
            "IK failed: " + getattr(resp, "error_reason", "")
        )

    q_arm = list(resp.q_arm) if hasattr(resp, "q_arm") else []
    if len(q_arm) >= 14:
        return q_arm[:14]

    left_res = list(resp.hand_poses.left_pose.joint_angles)
    right_res = list(resp.hand_poses.right_pose.joint_angles)
    if len(left_res) == 7 and len(right_res) == 7:
        return left_res + right_res

    raise RuntimeError("IK response did not contain arm joints")


def _call_single_arm_ik(current_joints, active_arm, active_pos,
                        active_quat, locked_other_arm_joints, cfg):
    cur = [float(v) for v in current_joints]
    fk = _call_fk(cur, cfg.timeout)

    if active_arm == "left":
        right_lock = (list(locked_other_arm_joints)
                      if locked_other_arm_joints is not None
                      else list(cur[7:14]))
        q0 = cur[:7] + right_lock
        lock_fk = _call_fk(q0, cfg.timeout)
        lq = list(active_quat) if active_quat is not None else list(
            fk.left_pose.quat_xyzw)
        ik14 = _call_two_hands_ik(
            current_rad14=q0,
            left_pos=list(active_pos),
            right_pos=list(lock_fk.right_pose.pos_xyz),
            left_quat=lq,
            right_quat=list(lock_fk.right_pose.quat_xyzw),
            constraint_mode=cfg.constraint_mode,
            pos_cost_weight=cfg.pos_cost_weight,
            timeout=cfg.timeout,
        )
        return ik14[:7] + right_lock

    if active_arm == "right":
        left_lock = (list(locked_other_arm_joints)
                     if locked_other_arm_joints is not None
                     else list(cur[:7]))
        q0 = left_lock + cur[7:14]
        lock_fk = _call_fk(q0, cfg.timeout)
        rq = list(active_quat) if active_quat is not None else list(
            fk.right_pose.quat_xyzw)
        ik14 = _call_two_hands_ik(
            current_rad14=q0,
            left_pos=list(lock_fk.left_pose.pos_xyz),
            right_pos=list(active_pos),
            left_quat=list(lock_fk.left_pose.quat_xyzw),
            right_quat=rq,
            constraint_mode=cfg.constraint_mode,
            pos_cost_weight=cfg.pos_cost_weight,
            timeout=cfg.timeout,
        )
        return left_lock + ik14[7:14]

    raise ValueError(f"unknown arm: {active_arm}")


# ==========================================================================
# Waypoint / trajectory helpers
# ==========================================================================

def _normalize_waypoints(waypoints):
    """Convert raw dict waypoints into ``Waypoint`` objects."""
    parsed = []
    for wp in waypoints:
        if isinstance(wp, Waypoint):
            parsed.append(wp)
            continue
        pos = [float(v) for v in wp["pos"]]
        quat = (
            [float(v) for v in wp["quat"]]
            if wp.get("quat") is not None else None
        )
        dur = float(wp.get("duration", 1.0))
        parsed.append(Waypoint(pos=pos, quat=quat, duration=dur))
    return parsed


def _resolve_quat(waypoints, idx, current_joints, arm, cfg):
    """Determine quaternion for a waypoint that did not provide one.

    Heuristic:
    1. Use the quat of the **next** waypoint that has one.
    2. If none ahead, use the quat of the **previous** wp that had one.
    3. Fallback: keep current FK quat.
    """
    # Forward search
    for i in range(idx + 1, len(waypoints)):
        if waypoints[i].quat is not None:
            return list(waypoints[i].quat)

    # Backward search
    for i in range(idx - 1, -1, -1):
        if waypoints[i].quat is not None:
            return list(waypoints[i].quat)

    # Fallback: current FK end-effector orientation
    fk = _call_fk(current_joints, cfg.timeout)
    pose = fk.left_pose if arm == "left" else fk.right_pose
    return list(pose.quat_xyzw)


# ==========================================================================
# Utilities
# ==========================================================================

def _rad_to_deg(values):
    return [math.degrees(float(v)) for v in values]


def write_trajectory_yaml(waypoints, filepath, arm="right"):
    """Persist a trajectory definition to a YAML file for reuse."""
    import os
    import yaml

    def _serialize(wp):
        d = {"pos": [float(v) for v in wp.pos], "duration": float(wp.duration)}
        if wp.quat is not None:
            d["quat"] = [float(v) for v in wp.quat]
        return d

    data = {
        "arm": arm,
        "waypoints": [_serialize(w) for w in _normalize_waypoints(waypoints)],
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def load_trajectory_yaml(filepath):
    """Load a trajectory definition from YAML."""
    import yaml

    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    wp_list = data.get("waypoints", [])
    return Trajectory(
        arm=data.get("arm", "right"),
        waypoints=_normalize_waypoints(wp_list),
    )
