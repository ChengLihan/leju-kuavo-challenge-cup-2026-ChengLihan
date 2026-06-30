#!/usr/bin/env python3
"""Domain randomization for Kuavo grasping RL environment.

Handles:
  - Object position randomization on the table surface
  - Object yaw orientation randomization
  - Object selection for multi-object scenes
  - Difficulty scaling (curriculum learning support)
"""

import numpy as np
from typing import Optional


class DomainRandomizer:
    """Randomizes graspable objects on the table for each episode."""

    def __init__(self, model, data, env):
        self.model = model
        self.data = data
        self.env = env

        # Table bounds (objects sit on table top at z≈0.815)
        # Objects are placed in scene2.xml at specific positions
        # We randomize within reasonable range on the table
        self.table_x_center = -0.25
        self.table_y_center = 0.0
        self.table_half_x = 0.10   # ±10cm from center
        self.table_half_y = 0.35   # ±35cm from center
        self.table_z = 0.815       # table top + object height offset

        # Per-object-type z offset (objects have different thicknesses)
        self.object_z_offsets = {
            "part_type_a": 0.005,   # steel pipe fastener (thin)
            "part_type_b": 0.005,   # T-junction
            "part_type_c": 0.005,   # screwdriver
        }

        # Object freejoint qpos addresses (first 7 elements of each object)
        self._obj_qpos_starts = {}
        for name in self.env.object_body_ids:
            body_id = self.env.object_body_ids[name]
            # Freejoint: qpos_addr gives start of 7-element group (x,y,z,qw,qx,qy,qz)
            self._obj_qpos_starts[name] = self.model.jnt_qposadr[
                self.model.body_jntadr[body_id]
            ]

    def randomize_objects(
        self,
        rng: np.random.Generator,
        difficulty: float = 0.0,
    ):
        """
        Randomize all 6 objects on the table.

        Args:
            rng: NumPy random generator
            difficulty: 0.0 (easy, narrow range) → 1.0 (hard, full range)
        """
        # Expand randomization range with difficulty
        half_x = self.table_half_x * (0.2 + 0.8 * difficulty)
        half_y = self.table_half_y * (0.3 + 0.7 * difficulty)
        yaw_range = np.pi * difficulty  # 0 → π radians

        for name in OBJECT_NAMES:
            if name not in self._obj_qpos_starts:
                continue

            # Random XY on table
            x = rng.uniform(
                self.table_x_center - half_x,
                self.table_x_center + half_x,
            )
            y = rng.uniform(
                self.table_y_center - half_y,
                self.table_y_center + half_y,
            )

            # Random yaw rotation
            yaw = rng.uniform(-yaw_range, yaw_range)

            # Convert yaw to quaternion (rotation about world Z)
            qw = np.cos(yaw / 2.0)
            qx = 0.0
            qy = 0.0
            qz = np.sin(yaw / 2.0)

            # Object type determines z placement
            z = self.table_z
            for prefix, offset in self.object_z_offsets.items():
                if name.startswith(prefix):
                    z += offset
                    break

            # Write qpos
            start = self._obj_qpos_starts[name]
            self.data.qpos[start + 0] = x
            self.data.qpos[start + 1] = y
            self.data.qpos[start + 2] = z
            self.data.qpos[start + 3] = qw
            self.data.qpos[start + 4] = qx
            self.data.qpos[start + 5] = qy
            self.data.qpos[start + 6] = qz

            # Zero object velocity
            dof_addr = self.model.jnt_dofadr[
                self.model.body_jntadr[self.env.object_body_ids[name]]
            ]
            for i in range(6):
                self.data.qvel[dof_addr + i] = 0.0

    def get_object_pose(self, name: str) -> np.ndarray:
        """Get object world position (x, y, z)."""
        body_id = self.env.object_body_ids[name]
        return self.data.xpos[body_id].copy()


OBJECT_NAMES = [
    "part_type_a_1", "part_type_a_2",
    "part_type_b_1", "part_type_b_2",
    "part_type_c_1", "part_type_c_2",
]
