#!/usr/bin/env python3
"""Reward functions for Kuavo grasping RL.

Two reward components:
  1. Residual RL: rewards bringing EE close to object and lifting it
  2. Gripper RL: rewards appropriate force application and secure grasp
"""

import numpy as np


class RewardCalculator:
    """Computes combined reward for residual RL + gripper RL."""

    def __init__(self, env):
        self.env = env

    def compute(self) -> float:
        """Compute total reward for current simulation state."""
        reward = 0.0

        # ── A. Residual RL rewards ──
        reward += self._ee_proximity_reward()
        reward += self._lift_reward()
        reward += self._grasp_success_bonus()
        reward += self._smoothness_penalty()
        reward += self._joint_limit_penalty()

        # ── B. Gripper RL rewards ──
        reward += self._contact_detection_reward()
        reward += self._force_appropriateness_reward()
        reward += self._excessive_force_penalty()
        reward += self._gripper_efficiency_reward()

        return reward

    # ── Residual RL components ────────────────────────────────────

    def _ee_proximity_reward(self) -> float:
        """Penalize distance between end effector and target object."""
        dist = np.linalg.norm(
            self.env.ee_position - self.env.object_position
        )
        return -dist * 2.0  # -0.1 at 5cm, -0.2 at 10cm

    def _lift_reward(self) -> float:
        """Reward lifting the object above table."""
        obj_z = self.env.data.xpos[self.env.target_obj_body_id][2]
        lift = max(0.0, obj_z - self.env.object_initial_z - 0.005)
        return lift * 5.0  # +0.25 at 5cm lift

    def _grasp_success_bonus(self) -> float:
        """Large sparse bonus for successful grasp (object lifted >5cm)."""
        obj_z = self.env.data.xpos[self.env.target_obj_body_id][2]
        if obj_z > self.env.object_initial_z + 0.05:
            return 20.0  # ~40x the per-step EE reward
        return 0.0

    def _smoothness_penalty(self) -> float:
        """Small penalty for large residual corrections (encourage smooth)."""
        residual_norm = np.linalg.norm(self.env.last_residual)
        return -residual_norm * 0.2

    def _joint_limit_penalty(self) -> float:
        """Penalty for approaching joint limits."""
        penalty = 0.0
        for i in range(7):
            q = float(self.env.data.qpos[self.env.arm_qpos_ids[i]])
            lo = float(self.env.joint_low_rad[i])
            hi = float(self.env.joint_high_rad[i])
            margin = min(q - lo, hi - q)
            if margin < 0.1:  # within 0.1 rad of limit
                penalty += (0.1 - margin) * 10.0
        return -penalty

    # ── Gripper RL components ─────────────────────────────────────

    def _contact_detection_reward(self) -> float:
        """Small reward for detecting contact with object."""
        force = abs(self.env.gripper_force)
        if force > 0.05:
            return 0.2
        return 0.0

    def _force_appropriateness_reward(self) -> float:
        """
        Reward for applying appropriate force based on part type.
        Different parts need different grip strengths.
        """
        force = abs(self.env.gripper_force)
        part_idx = self.env.part_type_idx
        ideal_forces = {0: 1.5, 1: 2.5, 2: 2.0}
        ideal = ideal_forces.get(part_idx, 2.0)
        force_error = abs(force - ideal)
        return -force_error * 0.3

    def _excessive_force_penalty(self) -> float:
        """Heavy penalty for applying too much force (risk of crushing)."""
        force = abs(self.env.gripper_force)
        part_idx = self.env.part_type_idx
        max_forces = {0: 3.0, 1: 4.0, 2: 3.5}
        max_f = max_forces.get(part_idx, 3.5)
        if force > max_f:
            return -(force - max_f) * 2.0
        return 0.0

    def _gripper_efficiency_reward(self) -> float:
        """Small reward for securing object with minimal grip percentage."""
        force = abs(self.env.gripper_force)
        close_pct = self.env.gripper_target_pct
        if force > 0.05 and close_pct < 90:
            return (90 - close_pct) / 90.0 * 0.3
        return 0.0
