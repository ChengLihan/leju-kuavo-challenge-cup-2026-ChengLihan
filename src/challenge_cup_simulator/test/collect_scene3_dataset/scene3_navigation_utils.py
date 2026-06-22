#!/usr/bin/env python3
"""Small shelf-approach navigator used only by the Scene3 dataset collector."""

import math
import threading
import time

import numpy as np

from scene3_rosbag_utils import wait_for_connection


class Scene3ShelfNavigator:
    """Move the base from Scene3 start/near-start pose to the shelf-front work pose.

    This is intentionally narrow: it only approaches the shelf and then stops.
    It does not decide target level, place the tray, or implement the final task policy.
    """

    def __init__(self, cfg):
        import rospy
        from geometry_msgs.msg import Twist
        from std_msgs.msg import Float64MultiArray

        self.cfg = cfg
        self.nav_cfg = cfg.get("navigation", {})
        topics = cfg.get("topics", {})
        self.cmd_vel_topic = topics.get("cmd_vel", "/cmd_vel")
        self.pos_topic = self.nav_cfg.get("base_position_topic", "/state_estimate/base/pos_xyz")
        self.yaw_topic = self.nav_cfg.get("base_yaw_topic", "/state_estimate/base/angular_zyx")

        self._lock = threading.Lock()
        self._latest_pos = None
        self._latest_yaw = None
        self._last_pos_time = None
        self._last_yaw_time = None

        self._cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        rospy.Subscriber(self.pos_topic, Float64MultiArray, self._cb_pos, queue_size=1)
        rospy.Subscriber(self.yaw_topic, Float64MultiArray, self._cb_yaw, queue_size=1)

    def _cb_pos(self, msg):
        import rospy

        data = list(msg.data)
        if len(data) >= 3:
            with self._lock:
                self._latest_pos = np.asarray(data[:3], dtype=np.float64)
                self._last_pos_time = rospy.Time.now()

    def _cb_yaw(self, msg):
        import rospy

        data = list(msg.data)
        if len(data) >= 3:
            with self._lock:
                self._latest_yaw = float(data[2])
                self._last_yaw_time = rospy.Time.now()

    def wait_ready(self, timeout=5.0):
        import rospy

        try:
            wait_for_connection(self._cmd_pub, min(float(timeout), 5.0))
        except Exception as exc:
            rospy.logwarn("Scene3ShelfNavigator: /cmd_vel has no subscriber yet: %s", exc)
        start = time.time()
        while time.time() - start < float(timeout) and not rospy.is_shutdown():
            if self.get_position() is not None:
                return True
            rospy.sleep(0.1)
        return False

    def get_position(self):
        import rospy

        with self._lock:
            pos = None if self._latest_pos is None else self._latest_pos.copy()
            stamp = self._last_pos_time
        if pos is None or stamp is None:
            return None
        if (rospy.Time.now() - stamp).to_sec() > 2.0:
            return None
        return pos

    def get_yaw(self):
        import rospy

        with self._lock:
            yaw = self._latest_yaw
            stamp = self._last_yaw_time
        if yaw is None or stamp is None:
            return None
        if (rospy.Time.now() - stamp).to_sec() > 2.0:
            return None
        return float(yaw)

    def stop(self, count=5):
        import rospy
        from geometry_msgs.msg import Twist

        msg = Twist()
        for _ in range(int(count)):
            self._cmd_pub.publish(msg)
            rospy.sleep(0.05)

    def approach_shelf(self):
        import rospy

        distance = float(self.nav_cfg.get("approach_shelf_distance", 1.00))
        timeout = float(self.nav_cfg.get("approach_timeout_sec", 20.0))
        max_speed = float(self.nav_cfg.get("max_forward_speed_shelf", 0.12))
        open_loop = bool(self.nav_cfg.get("force_open_loop", False))

        self.stop(count=3)
        has_position = self.wait_ready(timeout=min(timeout, 5.0))
        if open_loop or not has_position:
            if self.nav_cfg.get("require_position", False):
                raise RuntimeError("NAVIGATION_FAILED: no base position estimate for shelf approach")
            return self._approach_open_loop(distance, max_speed, timeout)
        return self._approach_closed_loop(distance, max_speed, timeout)

    def _approach_open_loop(self, distance, speed, timeout):
        import rospy
        from geometry_msgs.msg import Twist

        speed = max(0.02, min(float(speed), 0.20))
        duration = min(float(timeout), max(0.5, float(distance) / speed))
        rospy.logwarn(
            "Scene3ShelfNavigator: no fresh position estimate; open-loop forward %.2fm at %.2fm/s for %.1fs",
            distance,
            speed,
            duration,
        )
        msg = Twist()
        msg.linear.x = speed
        rate = rospy.Rate(20)
        end_time = time.time() + duration
        while time.time() < end_time and not rospy.is_shutdown():
            self._cmd_pub.publish(msg)
            rate.sleep()
        self.stop()
        rospy.sleep(float(self.nav_cfg.get("settle_after_nav_sec", 1.0)))
        return True

    def _approach_closed_loop(self, distance, max_speed, timeout):
        import rospy
        from geometry_msgs.msg import Twist

        start_pos = self.get_position()
        start_yaw = self.get_yaw()
        if start_pos is None:
            return self._approach_open_loop(distance, max_speed, timeout)

        rospy.loginfo(
            "Scene3ShelfNavigator: closed-loop shelf approach %.2fm, start_pos=%s",
            distance,
            [round(v, 4) for v in start_pos[:2]],
        )
        start_time = time.time()
        last_pos = start_pos.copy()
        stuck_count = 0
        rate = rospy.Rate(20)
        while time.time() - start_time < float(timeout) and not rospy.is_shutdown():
            cur_pos = self.get_position()
            if cur_pos is None:
                rospy.sleep(0.05)
                continue
            traveled = float(np.linalg.norm(cur_pos[:2] - start_pos[:2]))
            remaining = float(distance) - traveled
            if remaining <= float(self.nav_cfg.get("approach_done_tolerance_m", 0.03)):
                rospy.loginfo("Scene3ShelfNavigator: shelf approach done, traveled=%.3fm", traveled)
                self.stop()
                rospy.sleep(float(self.nav_cfg.get("settle_after_nav_sec", 1.0)))
                return True

            if np.allclose(cur_pos[:2], last_pos[:2], atol=0.005):
                stuck_count += 1
                if stuck_count > 80:
                    raise RuntimeError(f"NAVIGATION_FAILED: base position appears stuck at {cur_pos[:2]}")
            else:
                stuck_count = 0
                last_pos = cur_pos.copy()

            msg = Twist()
            msg.linear.x = self._approach_speed(remaining, max_speed)
            cur_yaw = self.get_yaw()
            if start_yaw is not None and cur_yaw is not None:
                yaw_error = self._normalize_angle(cur_yaw - start_yaw)
                msg.angular.z = float(np.clip(-2.0 * yaw_error, -0.10, 0.10))
            self._cmd_pub.publish(msg)
            rate.sleep()

        self.stop()
        raise RuntimeError(f"NAVIGATION_FAILED: shelf approach timeout after {timeout:.1f}s")

    @staticmethod
    def _approach_speed(remaining, max_speed):
        if remaining < 0.20:
            return max(0.03, min(float(max_speed), remaining * 0.35))
        return max(0.03, min(float(max_speed), 0.20))

    @staticmethod
    def _normalize_angle(value):
        return (float(value) + math.pi) % (2.0 * math.pi) - math.pi
