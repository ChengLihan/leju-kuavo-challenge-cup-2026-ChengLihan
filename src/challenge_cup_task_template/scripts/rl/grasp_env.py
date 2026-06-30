#!/usr/bin/env python3
"""
MuJoCo Gymnasium env for Kuavo grasping RL — DEPLOYMENT-REALISTIC redesign.

Key principle: observation space must match exactly what is available in ROS deployment.
  - NO joint velocity (ROS sensors_data_raw has it but TF frame timing is unreliable)
  - NO joint torque/actuator force (not exposed over ROS topics)
  - NO gripper force feedback (sim_leju_claw_interface only reports position %)
  - YES: joint angles, FK target angles, object position (from YOLO), EE position (from TF)

Design:
  - Agent outputs 7-DOF residual correction (±5° in training, ±3° at deployment)
  - Gripper is SCRIPTED (close to 85%) — force feedback unavailable, RL can't learn it
  - Single-step or multi-step correction at the approach point, then scripted grasp
"""

import os, math
import numpy as np
from typing import Dict, Tuple, Optional, Any

import mujoco
from mujoco import mjtObj

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# ── Paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
_SCENE2_XML = os.path.join(
    _PROJECT_ROOT, "src/challenge_cup_simulator/models/biped_s52/xml/scene2.xml")

# ── Constants ───────────────────────────────────────────────────────────
LEG_L_STAND = [0.006, 0.0, -0.503, 0.862, -0.36, -0.006]
LEG_R_STAND = [-0.007, 0.0, -0.502, 0.861, -0.36, 0.007]

ARM_DEFAULT_LEFT  = [0.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ARM_DEFAULT_RIGHT = [0.0, -60.0, 0.0, 0.0, 0.0, 0.0, 0.0]

JOINT_LIMITS_DEG = {
    "right": {"arm_pitch":(-137,34),"arm_roll":(-84,20),"arm_yaw":(-26,90),
              "forearm":(-150,0),"hand_yaw":(-90,90),"hand_pitch":(-40,40),"hand_roll":(-40,75)},
    "left":  {"arm_pitch":(-137,34),"arm_roll":(-20,84),"arm_yaw":(-90,26),
              "forearm":(-150,0),"hand_yaw":(-90,90),"hand_pitch":(-40,40),"hand_roll":(-75,40)},
}
JOINT_NAMES_ORDER = ["arm_pitch","arm_roll","arm_yaw","forearm","hand_yaw","hand_pitch","hand_roll"]

OBJECT_NAMES = ["part_type_a_1","part_type_a_2","part_type_b_1","part_type_b_2","part_type_c_1","part_type_c_2"]

SIM_STEPS_PER_ACTION = 8        # 125Hz control (PD stable tested)
TABLE_Z = 0.815
MAX_RESIDUAL_RAD = np.deg2rad(5.0)  # ±5°
MAX_EPISODE_STEPS = 15          # Agent has 15 steps to correct approach before auto-grasp
GRASP_TRIGGER_STEP = 10         # Force grasp at step 10 (gives ~8 RL correction steps before grasp)


class GraspEnv(gym.Env):
    """Kuavo grasping RL env — deployment-realistic observations.

    Observation: Box(23,) =
      [arm_qpos(7), fk_target(7), obj_to_ee_vec(3),
       ee_pos(3), gripper_q(1), part_onehot(2)]

    Action: Box(7,) = residual arm joint correction (±5° rad)

    Reward: dense EE-to-object distance + sparse lift bonus
    """

    metadata = {"render_modes": [None]}

    def __init__(self, arm="right", render_mode=None, difficulty=0.0, scene_xml=None):
        super().__init__()
        self.arm = arm
        self.difficulty = difficulty
        self.scene_xml = scene_xml or _SCENE2_XML

        if not os.path.exists(self.scene_xml):
            raise FileNotFoundError(f"Scene XML not found: {self.scene_xml}")

        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.001
        self.sim_steps_per_action = SIM_STEPS_PER_ACTION

        self._resolve_indices()

        # ── Observation space (23-dim, deployment-realistic) ──
        self.obs_dim = 23  # 7+7+3+3+1+2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # ── Action space (7-dim residual only) ──
        self.action_space = spaces.Box(
            low=-MAX_RESIDUAL_RAD, high=MAX_RESIDUAL_RAD,
            shape=(7,), dtype=np.float32)

        # ── Internal state ──
        self.step_count = 0
        self.max_steps = MAX_EPISODE_STEPS
        self.fk_target_joints = np.zeros(7, dtype=np.float32)
        self.gripper_target_pct = 0.0
        self.target_obj_name = ""
        self.target_obj_body_id = -1
        self.part_type_idx = 0
        self._obj_initial_z = TABLE_Z
        self._was_lifted = False
        self._grasp_triggered = False
        self.current_arm_target = np.zeros(7, dtype=np.float32)

        # ── PD gains for arm motors (torque actuators) ──
        self.arm_kp = np.array([80,80,60,80,20,20,20], dtype=np.float64)
        self.arm_kd = np.array([4,4,3,4,1.5,1.5,1.5], dtype=np.float64)
        self.arm_torque_limits = np.array([66,75,57,75,14.1,14.1,14.1], dtype=np.float64)

    # ── MuJoCo Index Resolution ─────────────────────────────────────

    def _resolve_indices(self):
        m = self.model
        side = "right" if self.arm == "right" else "left"
        al = self.arm[0]

        arm_joint_names = [f"zarm_{al}{i}_joint" for i in range(1,8)]
        self.arm_qpos_ids = [m.jnt_qposadr[mujoco.mj_name2id(m,mjtObj.mjOBJ_JOINT,n)] for n in arm_joint_names]
        self.arm_dof_ids  = [m.jnt_dofadr[mujoco.mj_name2id(m,mjtObj.mjOBJ_JOINT,n)] for n in arm_joint_names]

        motor_names = [f"zarm_{al}{i}_joint_motor" for i in range(1,8)]
        self.arm_actuator_ids = [mujoco.mj_name2id(m,mjtObj.mjOBJ_ACTUATOR,n) for n in motor_names]

        self.gripper_driver_qpos_id = m.jnt_qposadr[mujoco.mj_name2id(m,mjtObj.mjOBJ_JOINT,f"{side}_right_driver_joint")]
        self.gripper_driver_dof_id  = m.jnt_dofadr[mujoco.mj_name2id(m,mjtObj.mjOBJ_JOINT,f"{side}_right_driver_joint")]
        self.gripper_actuator_id = mujoco.mj_name2id(m,mjtObj.mjOBJ_ACTUATOR,f"{side}_fingers_actuator")
        self.ee_body_id = mujoco.mj_name2id(m,mjtObj.mjOBJ_BODY,f"zarm_{al}7_link")
        self.pinch_site_id = mujoco.mj_name2id(m,mjtObj.mjOBJ_SITE,f"{side}_pinch")
        self.base_id = mujoco.mj_name2id(m,mjtObj.mjOBJ_BODY,"base_link")

        leg_l = [f"leg_l{i}_joint" for i in range(1,7)]
        leg_r = [f"leg_r{i}_joint" for i in range(1,7)]
        self.leg_l_qpos_ids = [m.jnt_qposadr[mujoco.mj_name2id(m,mjtObj.mjOBJ_JOINT,n)] for n in leg_l]
        self.leg_r_qpos_ids = [m.jnt_qposadr[mujoco.mj_name2id(m,mjtObj.mjOBJ_JOINT,n)] for n in leg_r]
        self.waist_qpos_id = m.jnt_qposadr[mujoco.mj_name2id(m,mjtObj.mjOBJ_JOINT,"waist_yaw_joint")]
        self.head_yaw_qpos_id = m.jnt_qposadr[mujoco.mj_name2id(m,mjtObj.mjOBJ_JOINT,"zhead_1_joint")]
        self.head_pitch_qpos_id = m.jnt_qposadr[mujoco.mj_name2id(m,mjtObj.mjOBJ_JOINT,"zhead_2_joint")]

        ol = 'l' if self.arm == "right" else 'r'
        self.other_arm_qpos_ids = [m.jnt_qposadr[mujoco.mj_name2id(m,mjtObj.mjOBJ_JOINT,f"zarm_{ol}{i}_joint")] for i in range(1,8)]
        os = "left" if self.arm == "right" else "right"
        self.other_gripper_driver_qpos_id = m.jnt_qposadr[mujoco.mj_name2id(m,mjtObj.mjOBJ_JOINT,f"{os}_right_driver_joint")]
        self.other_gripper_actuator_id = mujoco.mj_name2id(m,mjtObj.mjOBJ_ACTUATOR,f"{os}_fingers_actuator")

        self.object_body_ids = {n: mujoco.mj_name2id(m,mjtObj.mjOBJ_BODY,n) for n in OBJECT_NAMES}

        limits = JOINT_LIMITS_DEG[self.arm]
        self.joint_low_rad  = np.array([np.deg2rad(limits[k][0]) for k in JOINT_NAMES_ORDER], dtype=np.float32)
        self.joint_high_rad = np.array([np.deg2rad(limits[k][1]) for k in JOINT_NAMES_ORDER], dtype=np.float32)

        # ── FK table for deployment-matching baseline ──
        self._load_fk_table()

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._was_lifted = False
        self._grasp_triggered = False
        self.gripper_target_pct = 0.0

        self._reset_robot_pose()
        self._randomize_objects()

        # CRITICAL: call mj_forward BEFORE reading xpos (object qpos was just set)
        mujoco.mj_forward(self.model, self.data)

        obj_idx = self.np_random.integers(0, len(OBJECT_NAMES))
        self.target_obj_name = OBJECT_NAMES[obj_idx]
        self.target_obj_body_id = self.object_body_ids[self.target_obj_name]
        self.part_type_idx = 0 if "part_type_a" in self.target_obj_name else (1 if "part_type_b" in self.target_obj_name else 2)

        obj_true = self._get_object_position()
        self._obj_initial_z = obj_true[2]

        # FK target from NOISY object position (simulates YOLO → FK pipeline)
        noise_std_m = 0.015 + self.difficulty * 0.035
        self._obj_noisy = obj_true + self.np_random.normal(0, noise_std_m, 3).astype(np.float32)
        self.fk_target_joints = self._compute_fk_baseline(self._obj_noisy)
        self._obj_true = obj_true

        # Current arm = FK target + joint-level noise (control error)
        joint_noise_std_deg = 1.0 + self.difficulty * 3.0
        joint_noise_rad = self.np_random.normal(0, np.deg2rad(joint_noise_std_deg), 7).astype(np.float32)
        noisy_joints = np.clip(self.fk_target_joints + joint_noise_rad, self.joint_low_rad, self.joint_high_rad)
        self._set_arm_joints(noisy_joints)
        self.current_arm_target = self.fk_target_joints.copy()

        # Open gripper
        self._set_gripper_ctrl(0.0)

        # Forward again with noisy arm joints
        mujoco.mj_forward(self.model, self.data)

        obs = self._build_obs()
        return obs, {}

    def _reset_robot_pose(self):
        d = self.data
        for i,v in enumerate(LEG_L_STAND): d.qpos[self.leg_l_qpos_ids[i]] = v
        for i,v in enumerate(LEG_R_STAND): d.qpos[self.leg_r_qpos_ids[i]] = v
        d.qpos[self.waist_qpos_id] = 0.0
        d.qpos[self.head_yaw_qpos_id] = 0.0
        d.qpos[self.head_pitch_qpos_id] = 0.349

        arm_default = ARM_DEFAULT_LEFT if self.arm == "left" else ARM_DEFAULT_RIGHT
        for i,deg in enumerate(arm_default): d.qpos[self.arm_qpos_ids[i]] = np.deg2rad(deg)
        other_default = ARM_DEFAULT_RIGHT if self.arm == "left" else ARM_DEFAULT_LEFT
        for i,deg in enumerate(other_default): d.qpos[self.other_arm_qpos_ids[i]] = np.deg2rad(deg)

        d.qpos[self.gripper_driver_qpos_id] = 0.0
        d.qpos[self.other_gripper_driver_qpos_id] = 0.0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0

    def _randomize_objects(self):
        """Randomize all 6 objects on the table."""
        rng = self.np_random
        half_x = 0.10 * (0.2 + 0.8 * self.difficulty)
        half_y = 0.35 * (0.3 + 0.7 * self.difficulty)
        yaw_range = np.pi * self.difficulty

        for name in OBJECT_NAMES:
            body_id = self.object_body_ids[name]
            qpos_start = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
            x = rng.uniform(-0.25 - half_x, -0.25 + half_x)
            y = rng.uniform(-half_y, half_y)
            yaw = rng.uniform(-yaw_range, yaw_range)
            qw, qz = np.cos(yaw/2), np.sin(yaw/2)
            self.data.qpos[qpos_start:qpos_start+7] = [x, y, TABLE_Z + 0.005, qw, 0, 0, qz]
            dof_start = self.model.jnt_dofadr[self.model.body_jntadr[body_id]]
            self.data.qvel[dof_start:dof_start+6] = 0.0

    # ── Step ─────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        residual = np.asarray(action, dtype=np.float32).flatten()

        # 1. Apply arm correction via PD
        new_target = np.clip(self.current_arm_target + residual, self.joint_low_rad, self.joint_high_rad)
        self.current_arm_target = new_target
        for i, cid in enumerate(self.arm_actuator_ids):
            pos_err = new_target[i] - self.data.qpos[self.arm_qpos_ids[i]]
            vel = self.data.qvel[self.arm_dof_ids[i]]
            torque = np.clip(self.arm_kp[i]*pos_err - self.arm_kd[i]*vel,
                            -self.arm_torque_limits[i], self.arm_torque_limits[i])
            self.data.ctrl[cid] = torque

        # 2. Gripper: scripted close to 85% at grasp trigger step
        self.step_count += 1
        if self.step_count >= GRASP_TRIGGER_STEP and not self._grasp_triggered:
            self._grasp_triggered = True
            self._set_gripper_ctrl(85.0)
        elif not self._grasp_triggered:
            self._set_gripper_ctrl(0.0)

        # 3. Simulate
        for _ in range(self.sim_steps_per_action):
            mujoco.mj_step(self.model, self.data)

        # 4. Observation
        obs = self._build_obs()

        # 5. Reward
        reward = self._compute_reward()

        # 6. Check lift (base_link-relative Z)
        obj_rel_z = self.object_position[2]
        if obj_rel_z > self._obj_initial_z + 0.03:
            self._was_lifted = True

        # 7. Termination (object must be above table by 8cm in base_link frame)
        terminated = self._was_lifted and obj_rel_z > 0.08
        truncated = self.step_count >= self.max_steps

        info = {"obj_z": float(obj_rel_z), "lifted": self._was_lifted,
                "ee_dist": float(np.linalg.norm(self.ee_position - self.object_position))}
        return obs, float(reward), terminated, truncated, info

    # ── Observation (23-dim, deployment-realistic) ──────────────────

    def _build_obs(self):
        """Build observation matching ROS deployment data.

        - arm_qpos: from sensor_data_raw topic (same as MuJoCo qpos)
        - fk_target: FK table lookup (same .npy as deployment)
        - ee_pos: MuJoCo pinch site → deployment gets same from TF (zarm_X7_end_effector frame)
        - obj_pos: noisy (YOLO-like error) → deployment uses YOLO + depth
        """
        d = self.data
        arm_qpos = np.array([d.qpos[i] for i in self.arm_qpos_ids], dtype=np.float32)

        # EE position: MuJoCo pinch site = deployment TF frame (zarm_X7_end_effector)
        ee_pos = self.ee_position

        # Object position = noisy (simulates YOLO detection error)
        obj_noisy = self._obj_noisy

        gripper_q = float(d.qpos[self.gripper_driver_qpos_id])

        part_onehot = np.zeros(2, dtype=np.float32)
        if self.part_type_idx < 2:
            part_onehot[self.part_type_idx] = 1.0

        obs = np.concatenate([
            arm_qpos,                # 7  — current joint angles (rad)
            self.fk_target_joints,   # 7  — FK target from table lookup (rad)
            obj_noisy - ee_pos,      # 3  — relative vector (noisy YOLO → TF EE)
            ee_pos,                  # 3  — EE position from TF (MuJoCo pinch = TF frame)
            [gripper_q],             # 1  — gripper driver position (rad)
            part_onehot,             # 2  — part class
        ]).astype(np.float32)
        return obs

    def _estimate_ee_position(self, joints_deg):
        """Geometric FK for EE position — IDENTICAL to challenge_task.py version."""
        shoulder_z = 0.394
        shoulder_y = -0.255 if self.arm == "right" else 0.255

        pitch = math.radians(joints_deg[0])
        forearm = math.radians(joints_deg[3])
        roll = math.radians(joints_deg[1])

        L1, L2 = 0.32, 0.35
        ee_x = L1 * math.cos(pitch) + L2 * math.cos(pitch + forearm)
        ee_z_s = -(L1 * math.sin(pitch) + L2 * math.sin(pitch + forearm))
        ee_y = shoulder_y + (L1 + L2) * math.sin(roll) * 0.2

        return np.array([ee_x, ee_y, shoulder_z + ee_z_s - 0.17], dtype=np.float32)

    # ── Reward ───────────────────────────────────────────────────────

    def _compute_reward(self):
        # Use base_link-relative Z for both obj_z and initial_z (consistent!)
        obj_rel = self.object_position  # base_link-relative
        obj_z = obj_rel[2]
        init_z = self._obj_initial_z  # already base_link-relative

        dist = np.linalg.norm(self.ee_position - obj_rel)
        lift = max(0.0, obj_z - init_z - 0.005)

        reward = 0.0
        reward -= dist * 3.0
        reward += lift * 8.0
        if obj_z > init_z + 0.05:
            reward += 30.0
        reward -= np.linalg.norm(self.current_arm_target - self.fk_target_joints) * 0.15
        for i in range(7):
            q = float(self.data.qpos[self.arm_qpos_ids[i]])
            margin = min(q - self.joint_low_rad[i], self.joint_high_rad[i] - q)
            if margin < 0.1:
                reward -= (0.1 - margin) * 15.0
        return reward

    # ── Position helpers ─────────────────────────────────────────────

    # URDF end_effector offset (matches zarm_X7_end_effector TF frame)
    EE_LOCAL = np.array([0.0, 0.0, -0.17], dtype=np.float64)

    @property
    def ee_position(self):
        """EE position = zarm_X7_link + R * (0,0,-0.17) — matches URDF/TF frame."""
        r7_world = self.data.xpos[self.ee_body_id]
        R = self.data.xmat[self.ee_body_id].reshape(3, 3)
        ee_world = r7_world + R @ self.EE_LOCAL
        return (ee_world - self.data.xpos[self.base_id]).astype(np.float32)

    @property
    def object_position(self):
        return self.data.xpos[self.target_obj_body_id].copy() - self.data.xpos[self.base_id].copy()

    def _get_object_position(self):
        return self.object_position

    # ── FK Table Lookup (matches deployment exactly) ──────────────────

    def _load_fk_table(self):
        """Load FK table V2 — same .npy files used by deployment Scene2Controller.

        FK tables live in main repo scripts/ (alongside challenge_task.py).
        When training from a worktree, we cross back to the main repo.
        """
        fname = f"fk_table_{self.arm}_v2.npy"
        # Worktree layout: .claude/worktrees/xxx/src/.../
        # Main repo:     src/.../
        # From worktree root, go up 3 to main repo root.
        # From main repo root, go up 0 (same as _PROJECT_ROOT).
        candidates = [
            os.path.join(_SCRIPT_DIR, "..", fname),         # rl/../ (scripts/ dir)
            os.path.join(_PROJECT_ROOT, "..", "..", "..",    # worktree→main repo
                         "src", "challenge_cup_task_template", "scripts", fname),
            os.path.join(_PROJECT_ROOT, "src",               # main repo directly
                         "challenge_cup_task_template", "scripts", fname),
        ]
        for path in candidates:
            if os.path.exists(path):
                self._fk_table = np.load(path)
                self._fk_positions = self._fk_table[:, 7:10].copy()
                self._fk_joints_deg = self._fk_table[:, :7].copy()
                return
        raise FileNotFoundError(f"FK table not found: {fname}. Tried: {candidates}")

    def _compute_fk_baseline(self, target_pos):
        """Nearest-neighbor lookup in FK table V2 — matches deployment FKLookup."""
        # Weighted distance: XY error matters more than Z (deployment uses similar weighting)
        tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
        pos = self._fk_positions
        dists = np.sqrt(
            3.0 * (pos[:, 0] - tx) ** 2 +
            3.0 * (pos[:, 1] - ty) ** 2 +
            1.0 * (pos[:, 2] - tz) ** 2
        )
        best = int(np.argmin(dists))
        joints_deg = self._fk_joints_deg[best].copy()
        # Clamp to limits
        limits = JOINT_LIMITS_DEG[self.arm]
        for i, key in enumerate(JOINT_NAMES_ORDER):
            joints_deg[i] = np.clip(joints_deg[i], limits[key][0], limits[key][1])
        return np.deg2rad(joints_deg).astype(np.float32)

    # ── Low-level control ────────────────────────────────────────────

    def _set_arm_joints(self, joints_rad):
        for i, qid in enumerate(self.arm_qpos_ids):
            self.data.qpos[qid] = float(joints_rad[i])

    def _set_gripper_ctrl(self, close_pct):
        self.gripper_target_pct = close_pct
        self.data.ctrl[self.gripper_actuator_id] = np.clip(close_pct * 2.55, 0, 255)

    def close(self):
        pass


# ── Registration ────────────────────────────────────────────────────
try:
    gym.register(id="KuavoGrasp-v1", entry_point="rl.grasp_env:GraspEnv", max_episode_steps=MAX_EPISODE_STEPS)
except Exception:
    pass
