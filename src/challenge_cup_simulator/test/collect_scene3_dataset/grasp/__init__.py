"""
grasp/__init__.py — Scene3 upper-tray grasp.

Provides:
  - ArmIK         – IK solver with pose + trajectory support
  - Waypoint      – single trajectory waypoint
  - Trajectory    – ordered series of waypoints
  - GraspExpert   – high-level grasp behaviour
  - ShelfNavigator – open-loop shelf approach
"""
from .arm_ik import ArmIK, Waypoint, Trajectory, write_trajectory_yaml, load_trajectory_yaml
from .grasp_expert import GraspExpert
from .navigation import ShelfNavigator
