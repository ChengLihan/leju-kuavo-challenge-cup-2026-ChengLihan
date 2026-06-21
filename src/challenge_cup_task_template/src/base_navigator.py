#!/usr/bin/env python3
"""
底盘导航模块：通过 /cmd_vel 控制机器人。
使用 state_estimate/base/pos_xyz 做位置闭环，angular_zyx 做 yaw 闭环。
"""

import rospy
import numpy as np


class BaseNavigator:
    def __init__(self, iface, perception, config):
        self.iface = iface
        self.perception = perception
        self.nav_config = config.get("navigation", {})
        self.timeouts = config.get("timeouts", {})
        self._pos_source = None
        self._last_fastlio_pos = None
        self._last_fastlio_time = None

    def set_position_source(self, source):
        self._pos_source = source

    def stop(self):
        self.iface.stop_base()

    def _get_pos(self):
        if self._pos_source is not None:
            pos = self._pos_source.get_position()
            fallback_pos = self.iface.get_base_position()
            if pos is None:
                return fallback_pos
            ref = self._last_fastlio_pos
            if ref is None:
                ref = np.zeros(3)
            if not np.allclose(pos, ref, atol=0.001):
                self._last_fastlio_pos = pos.copy()
                self._last_fastlio_time = rospy.Time.now()
                return pos
            if self._last_fastlio_time is not None:
                dt = (rospy.Time.now() - self._last_fastlio_time).to_sec()
                if dt > 1.5:
                    rospy.logwarn_throttle(2.0, "BaseNavigator: FastLIO stale %.1fs, fallback SE", dt)
                    return fallback_pos
            return pos
        return self.iface.get_base_position()

    def _get_yaw(self):
        return self.iface.base_yaw

    def _dist2d(self, a, b):
        return np.linalg.norm(a[:2] - b[:2])

    def move_forward(self, distance, timeout=None, max_speed=None):
        if timeout is None:
            timeout = max(10.0, distance / 0.06 + 5.0)
        if max_speed is None:
            max_speed = self.nav_config.get("max_forward_speed_shelf", 0.12)

        start_pos = self._get_pos()
        start_yaw = self._get_yaw()
        if start_pos is None:
            rospy.logerr("BaseNavigator: no position estimate, cannot move_forward")
            return False

        rospy.loginfo("BaseNavigator: move_forward %.2f m, start_pos=%s, start_yaw=%.3f",
                       distance, start_pos[:2], start_yaw)

        start_time = rospy.Time.now()
        rate = rospy.Rate(10)
        last_pos = start_pos.copy()
        stuck_count = 0

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout:
                rospy.logerr("BaseNavigator: move_forward timeout (%.1fs)", elapsed)
                self.stop()
                return False

            cur = self._get_pos()
            cur_yaw = self._get_yaw()
            if cur is None:
                rospy.sleep(0.1)
                continue

            traveled = self._dist2d(cur, start_pos)

            if np.allclose(cur[:2], last_pos[:2], atol=0.01):
                stuck_count += 1
                if stuck_count > 40:
                    rospy.logerr("BaseNavigator: position stuck, pos=%s", cur[:2])
                    self.stop()
                    return False
            else:
                stuck_count = 0
                last_pos = cur.copy()

            remaining = distance - traveled

            if remaining <= 0.03:
                rospy.loginfo("BaseNavigator: move_forward done, traveled=%.3f m", traveled)
                self.stop()
                rospy.sleep(1.0)
                return True

            if remaining < 0.2:
                speed = max(0.03, remaining * 0.2)
            else:
                speed = max_speed

            yaw_error = self._normalize_angle(cur_yaw - start_yaw)
            angular_z = np.clip(-2.0 * yaw_error, -0.10, 0.10)

            self.iface.publish_cmd_vel(speed, 0.0, angular_z)
            rate.sleep()

        self.stop()
        return False

    @staticmethod
    def _normalize_angle(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def move_backward(self, distance=0.50, timeout=None):
        if timeout is None:
            timeout = max(12.0, distance / 0.03 + 5.0)
        max_speed = self.nav_config.get("max_backward_speed", 0.10)

        start_pos = self._get_pos()
        if start_pos is None:
            rospy.logerr("BaseNavigator: no position estimate, cannot move_backward")
            return False

        rospy.loginfo("BaseNavigator: move_backward %.2f m, start_pos=%s", distance, start_pos[:2])

        start_time = rospy.Time.now()
        rate = rospy.Rate(10)
        last_pos = start_pos.copy()
        stuck_count = 0

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout:
                cur = self._get_pos()
                traveled = self._dist2d(cur, start_pos) if cur is not None else 0
                rospy.logerr("BaseNavigator: move_backward timeout (%.1fs, traveled=%.3f)", elapsed, traveled)
                self.stop()
                return False

            cur = self._get_pos()
            if cur is None:
                rospy.sleep(0.1)
                continue

            traveled = self._dist2d(cur, start_pos)
            remaining = distance - traveled

            if remaining <= 0.03:
                rospy.loginfo("BaseNavigator: move_backward done, traveled=%.3f m", traveled)
                self.stop()
                rospy.sleep(1.0)
                return True

            if np.allclose(cur[:2], last_pos[:2], atol=0.01):
                stuck_count += 1
                if stuck_count > 40:
                    rospy.logerr("BaseNavigator: stuck during backward, pos=%s", cur[:2])
                    self.stop()
                    return False
            else:
                stuck_count = 0
                last_pos = cur.copy()

            speed = min(remaining * 0.3, max_speed)
            speed = max(speed, 0.04)
            self.iface.publish_cmd_vel(-speed, 0.0, 0.0)
            rate.sleep()

        self.stop()
        return False

    def turn_by_yaw_delta(self, target_yaw_delta, tolerance=None):
        if tolerance is None:
            tolerance = self.nav_config.get("box_yaw_tolerance", 0.10)

        max_yaw_speed = self.nav_config.get("max_yaw_speed", 0.25)
        timeout = self.timeouts.get("turn_to_box", 25.0)

        start_yaw = self._get_yaw()
        start_time = rospy.Time.now()

        rospy.loginfo("BaseNavigator: turning %.2f rad (%.0f°), start_yaw=%.3f",
                       target_yaw_delta, np.rad2deg(target_yaw_delta), start_yaw)

        direction = 1.0 if target_yaw_delta > 0 else -1.0
        target_accumulated = abs(target_yaw_delta)
        time_estimate = target_accumulated / max_yaw_speed * 1.4

        rate = rospy.Rate(20)
        prev_yaw = start_yaw
        accumulated = 0.0

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()

            if elapsed > timeout:
                if accumulated > target_accumulated * 0.7:
                    rospy.logwarn("BaseNavigator: turn timeout, 70%% done, accepting")
                    break
                rospy.logerr("BaseNavigator: turn timeout (elapsed=%.1fs, acc=%.2f/%.2f)",
                             elapsed, accumulated, target_accumulated)
                self.stop()
                return False

            current_yaw = self._get_yaw()
            delta = current_yaw - prev_yaw
            delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
            accumulated += delta * direction
            prev_yaw = current_yaw

            if accumulated >= target_accumulated - tolerance:
                rospy.loginfo("BaseNavigator: turn done, acc=%.3f rad (%.0f°)",
                               accumulated, np.rad2deg(accumulated))
                break

            if accumulated < target_accumulated * 0.3:
                speed = max_yaw_speed * 0.5
            else:
                speed = max_yaw_speed

            angular_z = direction * speed
            self.iface.publish_cmd_vel(0.0, 0.0, angular_z)
            rate.sleep()

        self.stop()
        rospy.sleep(1.0)
        return True

    def turn_to_box(self):
        ok = self.turn_by_yaw_delta(self.nav_config.get("box_turn_yaw_rad", 3.1416))
        if ok:
            rospy.sleep(2.0)
            self.stop()
        return ok
