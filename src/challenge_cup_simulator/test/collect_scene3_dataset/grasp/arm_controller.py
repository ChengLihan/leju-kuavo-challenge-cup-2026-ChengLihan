"""
grasp/arm_controller.py — Arm trajectory holder thread.
"""
import threading

ARM_JOINT_NAMES = [f"arm_joint_{i}" for i in range(1, 15)]


class ArmTrajHold:
    """Continuously publishes 14-element arm JointState at fixed rate."""

    def __init__(self, pub, degrees_list, hz=100.0):
        self._pub = pub
        self._hz = float(hz)
        self._degrees = self._validate(degrees_list)
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

    def set_degrees(self, values):
        with self._lock:
            self._degrees = self._validate(values)

    def current_degrees(self):
        with self._lock:
            return list(self._degrees)

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState

        rate = rospy.Rate(self._hz)
        while not self._stop.is_set() and not rospy.is_shutdown():
            with self._lock:
                deg = list(self._degrees)
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = list(ARM_JOINT_NAMES)
            msg.position = deg
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break

    @staticmethod
    def _validate(values):
        v = [float(x) for x in values]
        if len(v) != 14:
            raise ValueError(f"expected 14 joints, got {len(v)}")
        return v
