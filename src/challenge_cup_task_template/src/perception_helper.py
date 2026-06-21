#!/usr/bin/env python3
"""
感知辅助模块：深度距离估计、ROI裁剪、点云生成。
使用头部 RGB-D 和 CameraInfo 进行货架/箱子的距离与偏航估计。
"""

import rospy
import numpy as np
import cv2


class PerceptionHelper:
    def __init__(self, iface, config):
        self.iface = iface
        self.config = config

        self._min_depth = config.get("min_valid_depth", 0.15)
        self._max_depth = config.get("max_valid_depth", 3.00)

        roi_s = config.get("shelf_roi", {})
        self.shelf_roi = (
            roi_s.get("u_min", 200), roi_s.get("u_max", 440),
            roi_s.get("v_min", 100), roi_s.get("v_max", 330),
        )

        roi_b = config.get("box_roi", {})
        self.box_roi = (
            roi_b.get("u_min", 180), roi_b.get("u_max", 460),
            roi_b.get("v_min", 120), roi_b.get("v_max", 360),
        )

    def sensors_ready(self):
        return self.iface.all_topics_ready()

    def wait_for_sensor_ready(self, timeout=10.0):
        return self.iface.wait_for_all_topics(timeout)

    def _decode_depth(self, compressed_depth_msg):
        if compressed_depth_msg is None:
            return None
        try:
            raw_data = compressed_depth_msg.data
            if isinstance(raw_data, str):
                raw_data = raw_data.encode('latin-1')

            buf = bytearray(raw_data)

            png_start = buf.find(b'\x89PNG')
            if png_start > 0:
                buf = buf[png_start:]

            arr = np.asarray(buf, dtype=np.uint8)
            depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

            if depth is None:
                buf2 = bytearray(raw_data)
                if len(buf2) > 12:
                    buf2 = buf2[12:]
                arr2 = np.asarray(buf2, dtype=np.uint8)
                depth = cv2.imdecode(arr2, cv2.IMREAD_UNCHANGED)

            return depth
        except Exception:
            return None

    def estimate_distance_roi(self, depth_img, roi=None):
        if depth_img is None:
            return None

        if roi is None:
            roi = self.shelf_roi

        u_min, u_max, v_min, v_max = roi
        h, w = depth_img.shape[:2]
        u_min = max(0, min(u_min, w - 1))
        u_max = max(u_min + 1, min(u_max, w))
        v_min = max(0, min(v_min, h - 1))
        v_max = max(v_min + 1, min(v_max, h))

        region = depth_img[v_min:v_max, u_min:u_max].astype(np.float32)

        valid = region[(region > self._min_depth * 1000) & (region < self._max_depth * 1000)]
        if len(valid) < 50:
            return None

        return np.median(valid) / 1000.0

    def estimate_shelf_distance(self):
        depth = self.iface.head_depth
        if depth is None:
            rospy.logwarn_throttle(3.0, "Perception: head_depth msg is None")
            return None
        depth_img = self._decode_depth(depth)
        if depth_img is None:
            rospy.logwarn_throttle(3.0, "Perception: depth decode returned None (shape check)")
            return None
        return self.estimate_distance_roi(depth_img, self.shelf_roi)

    def estimate_box_distance(self):
        depth = self.iface.head_depth
        if depth is None:
            rospy.logwarn_throttle(3.0, "Perception: head_depth msg is None (box)")
            return None
        depth_img = self._decode_depth(depth)
        if depth_img is None:
            rospy.logwarn_throttle(3.0, "Perception: depth decode returned None (box)")
            return None
        return self.estimate_distance_roi(depth_img, self.box_roi)

    def estimate_shelf_yaw_error(self):
        depth = self.iface.head_depth
        if depth is None:
            return 0.0

        depth_img = self._decode_depth(depth)
        if depth_img is None:
            return 0.0

        h, w = depth_img.shape[:2]
        mid = w // 2
        half = 60

        left_roi = (max(0, mid - half - 40), max(0, mid - half),
                     self.shelf_roi[2], self.shelf_roi[3])
        right_roi = (min(w, mid + half), min(w, mid + half + 40),
                      self.shelf_roi[2], self.shelf_roi[3])

        d_left = self.estimate_distance_roi(depth_img, left_roi)
        d_right = self.estimate_distance_roi(depth_img, right_roi)

        if d_left is None or d_right is None:
            return 0.0

        return np.arctan2(d_right - d_left, 0.15)

    def build_head_xyzrgb_cloud(self, max_points=1024):
        rgb_msg = self.iface.head_rgb
        depth_msg = self.iface.head_depth
        info_msg = self.iface.camera_info

        if rgb_msg is None or depth_msg is None or info_msg is None:
            return None

        try:
            raw_rgb = np.frombuffer(rgb_msg.data, dtype=np.uint8)
            rgb = cv2.imdecode(raw_rgb, cv2.IMREAD_COLOR)
            depth = self._decode_depth(depth_msg)
        except Exception:
            return None

        if rgb is None or depth is None:
            return None

        fx = info_msg.K[0]
        fy = info_msg.K[4]
        cx = info_msg.K[2]
        cy = info_msg.K[5]

        h, w = depth.shape[:2]
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h))

        depth_m = depth.astype(np.float32) / 1000.0
        valid = (depth_m > self._min_depth) & (depth_m < self._max_depth)

        vs, us = np.where(valid)
        if len(vs) > max_points:
            indices = np.random.choice(len(vs), max_points, replace=False)
            vs = vs[indices]
            us = us[indices]

        z = depth_m[vs, us]
        x = (us - cx) * z / fx
        y = (vs - cy) * z / fy

        rgb_vals = rgb[vs, us, :3].astype(np.float32) / 255.0

        points = np.stack([x, y, z, rgb_vals[:, 2], rgb_vals[:, 1], rgb_vals[:, 0]], axis=-1)
        return points

    def crop_scene3_roi(self, points):
        if points is None or len(points) == 0:
            return None

        mask = (
            (points[:, 0] > -0.2) & (points[:, 0] < 2.5)
            & (points[:, 1] > -1.5) & (points[:, 1] < 1.5)
            & (points[:, 2] > 0.1) & (points[:, 2] < 2.0)
        )
        return points[mask]
