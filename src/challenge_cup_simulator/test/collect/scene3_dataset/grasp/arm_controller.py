"""
grasp/arm_controller.py
"""
import threading

ARM_JOINT_NAMES = [f"arm_joint_{i}" for i in range(1, 15)]

class ArmTrajHold:
    def __init__(self, pub, degrees_list, hz=100.0):
        self._pub, self._hz = pub, float(hz)
        self._degrees = self._v(degrees_list)
        self._lock, self._stop, self._thread = threading.Lock(), threading.Event(), None

    def start(self):
        if self._thread is not None: return
        self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=1.0)

    def set_degrees(self, v):
        with self._lock: self._degrees = self._v(v)

    def current_degrees(self):
        with self._lock: return list(self._degrees)

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState
        rate = rospy.Rate(self._hz)
        while not self._stop.is_set() and not rospy.is_shutdown():
            with self._lock: d = list(self._degrees)
            msg = JointState(); msg.header.stamp = rospy.Time.now(); msg.name = list(ARM_JOINT_NAMES); msg.position = d
            try: self._pub.publish(msg); rate.sleep()
            except rospy.ROSException: break

    @staticmethod
    def _v(v): vv = [float(x) for x in v]; assert len(vv) == 14; return vv
