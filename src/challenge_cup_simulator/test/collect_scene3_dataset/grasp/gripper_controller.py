"""
grasp/gripper_controller.py — Gripper open/close via JointState topic.
"""
import threading


class GripperController:
    """Publishes left/right gripper positions."""

    def __init__(self, pub, hz=100.0):
        self._pub = pub
        self._hz = float(hz)
        self._left = 0.0   # open
        self._right = 0.0  # open
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def open(self):
        with self._lock:
            self._left = 0.0
            self._right = 0.0

    def close(self, arm="right"):
        with self._lock:
            self._left = 255.0 if arm == "left" else 0.0
            self._right = 255.0 if arm == "right" else 0.0

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState

        rate = rospy.Rate(self._hz)
        while not self._stop.is_set() and not rospy.is_shutdown():
            with self._lock:
                l, r = self._left, self._right
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ["left_gripper_joint", "right_gripper_joint"]
            msg.position = [l, r]
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break
