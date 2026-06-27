"""
grasp/gripper_controller.py
"""
import threading

class GripperController:
    def __init__(self, pub, hz=100.0):
        self._pub, self._hz = pub, float(hz)
        self._l, self._r = 0.0, 0.0
        self._lock, self._stop, self._thread = threading.Lock(), threading.Event(), None

    def start(self):
        if self._thread is not None: return
        self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=1.0)

    def open(self):
        with self._lock: self._l, self._r = 0.0, 0.0

    def close(self, arm="right"):
        with self._lock:
            self._l = 255.0 if arm == "left" else 0.0
            self._r = 255.0 if arm == "right" else 0.0

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState
        rate = rospy.Rate(self._hz)
        while not self._stop.is_set() and not rospy.is_shutdown():
            with self._lock: l, r = self._l, self._r
            msg = JointState(); msg.header.stamp = rospy.Time.now(); msg.name = ["left_gripper_joint", "right_gripper_joint"]; msg.position = [l, r]
            try: self._pub.publish(msg); rate.sleep()
            except rospy.ROSException: break
