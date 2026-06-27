"""
grasp/navigation.py — Shelf-approach navigator via /cmd_vel.
"""
import threading
import time
import numpy as np

from .ros_utils import wait_publisher


class ShelfNavigator:
    """Open-loop forward drive to approach the shelf."""

    def __init__(self, approach_distance=1.3, speed=0.15, timeout=20.0):
        self._distance = float(approach_distance)
        self._speed = max(0.02, min(float(speed), 0.20))
        self._timeout = float(timeout)

    def approach(self):
        import rospy
        from geometry_msgs.msg import Twist

        pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        try:
            wait_publisher(pub, 5.0)
        except Exception:
            rospy.logwarn("/cmd_vel has no subscriber yet")

        # Stop first
        for _ in range(3):
            pub.publish(Twist())
            rospy.sleep(0.05)

        duration = min(self._timeout, max(0.5, self._distance / self._speed))
        rospy.loginfo("open-loop forward %.2fm at %.2fm/s for %.1fs",
                      self._distance, self._speed, duration)

        msg = Twist()
        msg.linear.x = self._speed
        end_time = time.time() + duration
        rate = rospy.Rate(20)
        while time.time() < end_time and not rospy.is_shutdown():
            pub.publish(msg)
            rate.sleep()

        # Stop
        for _ in range(5):
            pub.publish(Twist())
            rospy.sleep(0.05)
        rospy.sleep(1.0)
        rospy.loginfo("navigation complete")
