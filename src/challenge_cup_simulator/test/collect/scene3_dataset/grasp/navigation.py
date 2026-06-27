"""
grasp/navigation.py
"""
import time

class ShelfNavigator:
    def __init__(self, approach_distance=1.3, speed=0.15, timeout=20.0):
        self._d, self._s, self._t = float(approach_distance), max(0.02, min(float(speed), 0.20)), float(timeout)

    def approach(self):
        import rospy
        from geometry_msgs.msg import Twist
        pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        for _ in range(3): pub.publish(Twist()); rospy.sleep(0.05)
        dur = min(self._t, max(0.5, self._d / self._s))
        rospy.loginfo("open-loop %.2fm @ %.2fm/s for %.1fs", self._d, self._s, dur)
        msg = Twist(); msg.linear.x = self._s
        end = time.time() + dur; rate = rospy.Rate(20)
        while time.time() < end and not rospy.is_shutdown(): pub.publish(msg); rate.sleep()
        for _ in range(5): pub.publish(Twist()); rospy.sleep(0.05)
        rospy.sleep(1.0)
