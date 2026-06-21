#!/usr/bin/env python3
"""
LiDAR 感知模块：解析 /lidar/points PointCloud2，提取货架/箱子距离和偏航。
替代深度相机方案，使用 360° LiDAR 进行障碍物检测和导航。
"""

import rospy
import numpy as np

_LIDAR_DTYPE = np.dtype([
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("intensity", "<f4"),
    ("time", "<f4"),
    ("ring", "<u2"),
    ("_pad", "<u2"),
])
_LIDAR_POINT_STEP = 24


class LidarPerception:
    def __init__(self, iface, config):
        self.iface = iface
        self.config = config

        nav = config.get("navigation", {})
        self._min_obstacle_z = nav.get("min_obstacle_z", 0.10)
        self._max_obstacle_z = nav.get("max_obstacle_z", 1.80)

        self._shelf_cone_half = np.deg2rad(nav.get("lidar_shelf_cone_half_deg", 30))
        self._box_cone_half = np.deg2rad(nav.get("lidar_box_cone_half_deg", 60))

        self._self_radius = nav.get("lidar_self_radius", 0.25)
        self._self_z_max = nav.get("lidar_self_z_max", 1.5)
        self._self_min_x = nav.get("lidar_self_min_x", 0.15)

    def get_points_xyz(self):
        msg = self.iface.lidar
        if msg is None:
            return None

        if hasattr(msg, "data") is False or len(msg.data) == 0:
            return None

        try:
            raw = msg.data
            n = len(raw) // _LIDAR_POINT_STEP
            if n == 0:
                return None
            arr = np.frombuffer(raw, dtype=_LIDAR_DTYPE, count=n)
            return np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
        except Exception:
            return None

    def filter_points(self, points, x_range=None, y_range=None, z_range=None):
        if points is None or len(points) == 0:
            return points

        mask = np.ones(len(points), dtype=bool)
        if x_range is not None:
            mask &= (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        if y_range is not None:
            mask &= (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        if z_range is not None:
            mask &= (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])

        return points[mask]

    def get_front_cone_distance(self, cone_half_rad=None):
        points = self.get_points_xyz()
        if points is None:
            return None, None

        if cone_half_rad is None:
            cone_half_rad = self._shelf_cone_half

        mask = (
            (points[:, 0] > 0)
            & (np.abs(np.arctan2(points[:, 1], points[:, 0])) < cone_half_rad)
        )
        pts = points[mask]
        if len(pts) < 5:
            return None, None

        pts = pts[
            (pts[:, 2] > self._min_obstacle_z)
            & (pts[:, 2] < self._max_obstacle_z)
        ]
        if len(pts) < 5:
            return None, None

        dist = np.median(pts[:, 0])
        yaw_error = np.arctan2(np.median(pts[:, 1]), dist) if dist > 0 else 0.0
        return float(dist), float(yaw_error)

    def get_shelf_distance(self):
        dist, yaw = self.get_front_cone_distance(self._shelf_cone_half)
        if dist is not None:
            rospy.loginfo_throttle(3.0, "LidarPerception: shelf dist=%.3f m, yaw_err=%.3f rad", dist, yaw)
        return dist

    def get_shelf_yaw_error(self):
        points = self.get_points_xyz()
        if points is None:
            return 0.0

        cone_half = self._shelf_cone_half
        left_mask = (
            (points[:, 0] > 0)
            & (np.arctan2(points[:, 1], points[:, 0]) > 0)
            & (np.arctan2(points[:, 1], points[:, 0]) < cone_half)
        )
        right_mask = (
            (points[:, 0] > 0)
            & (np.arctan2(points[:, 1], points[:, 0]) < 0)
            & (np.arctan2(points[:, 1], points[:, 0]) > -cone_half)
        )

        left = points[left_mask]
        right = points[right_mask]

        left = left[(left[:, 2] > self._min_obstacle_z) & (left[:, 2] < self._max_obstacle_z)]
        right = right[(right[:, 2] > self._min_obstacle_z) & (right[:, 2] < self._max_obstacle_z)]

        if len(left) < 3 or len(right) < 3:
            return 0.0

        d_left = np.median(left[:, 0])
        d_right = np.median(right[:, 0])
        return float(np.arctan2(d_right - d_left, 0.3))

    def get_box_distance(self):
        points = self.get_points_xyz()
        if points is None:
            rospy.logwarn_throttle(3.0, "LidarPerception: no points for box")
            return None

        points = self._remove_self_points(points)

        cone_half = np.deg2rad(60)
        mask = (
            (points[:, 0] > 0.5)
            & (np.abs(np.arctan2(points[:, 1], points[:, 0])) < cone_half)
        )
        pts = points[mask]
        pts = pts[
            (pts[:, 2] > self._min_obstacle_z)
            & (pts[:, 2] < self._max_obstacle_z)
        ]

        if len(pts) < 5:
            mask = (
                (points[:, 0] > 0.8)
                & (np.abs(np.arctan2(points[:, 1], points[:, 0])) < np.deg2rad(80))
            )
            pts = points[mask]
            pts = pts[
                (pts[:, 2] > self._min_obstacle_z)
                & (pts[:, 2] < self._max_obstacle_z)
            ]

        if len(pts) < 5:
            rospy.logwarn_throttle(3.0,
                "LidarPerception: no valid box pts after self-filter (total=%d, cone=%d)",
                len(points), len(pts))
            return None

        pts_x = np.sort(pts[:, 0])
        n = len(pts_x)
        dist = float(np.median(pts_x[int(n * 0.3):int(n * 0.7)]))
        rospy.loginfo_throttle(3.0, "LidarPerception: box dist=%.3f m (%d pts)", dist, len(pts))
        return dist

    def _remove_self_points(self, points):
        body_mask = ~(
            (np.sqrt(points[:, 0]**2 + points[:, 1]**2) < self._self_radius)
            & (np.abs(points[:, 2]) < self._self_z_max)
        )

        tray_mask = ~(
            (points[:, 0] > self._self_min_x) & (points[:, 0] < 0.7)
            & (np.abs(points[:, 1]) < 0.40)
            & (points[:, 2] > 0.4) & (points[:, 2] < 1.3)
        )

        return points[body_mask & tray_mask]

    def find_obstacle_in_front(self, min_dist=0.1, max_dist=5.0):
        points = self.get_points_xyz()
        if points is None:
            return None

        cone_half = np.deg2rad(45)
        mask = (
            (points[:, 0] > min_dist)
            & (points[:, 0] < max_dist)
            & (np.abs(np.arctan2(points[:, 1], points[:, 0])) < cone_half)
        )
        pts = points[mask]
        pts = pts[
            (pts[:, 2] > self._min_obstacle_z)
            & (pts[:, 2] < self._max_obstacle_z)
        ]
        if len(pts) < 5:
            return None
        return float(np.median(pts[:, 0]))

    def has_front_obstacle(self, threshold=3.0, min_points=10):
        points = self.get_points_xyz()
        if points is None:
            return False

        cone_half = np.deg2rad(60)
        mask = (
            (points[:, 0] > 0.2)
            & (points[:, 0] < threshold)
            & (np.abs(np.arctan2(points[:, 1], points[:, 0])) < cone_half)
        )
        pts = points[mask]
        pts = pts[
            (pts[:, 2] > self._min_obstacle_z)
            & (pts[:, 2] < self._max_obstacle_z)
        ]
        return len(pts) >= min_points

    def get_front_point_count(self):
        points = self.get_points_xyz()
        if points is None:
            return 0

        cone_half = np.deg2rad(60)
        mask = (
            (points[:, 0] > 0.2)
            & (points[:, 0] < 5.0)
            & (np.abs(np.arctan2(points[:, 1], points[:, 0])) < cone_half)
        )
        pts = points[mask]
        pts = pts[
            (pts[:, 2] > self._min_obstacle_z)
            & (pts[:, 2] < self._max_obstacle_z)
        ]
        return len(pts)
