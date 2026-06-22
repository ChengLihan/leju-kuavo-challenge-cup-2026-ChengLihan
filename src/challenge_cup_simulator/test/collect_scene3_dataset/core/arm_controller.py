"""
core/arm_controller.py — Arm trajectory holder.

Publishes a 14-element JointState on /kuavo_arm_traj at a fixed rate.
The joint order follows ARM_JOINT_NAMES (arm_joint_1 .. arm_joint_14).
"""

import threading

from .named_poses import ARM_JOINT_NAMES


class ArmTrajHold:
    """Continuously publishes a fixed arm joint state at a given frequency.

    Thread-safe: degrees can be hot-swapped via set_degrees() while running.
    """

    def __init__(self, pub, degrees_list, hz=100.0):
        self._pub = pub
        self._hz = float(hz)
        self._degrees = self._validate(degrees_list)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="arm_traj_hold", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_degrees(self, degrees_list):
        with self._lock:
            self._degrees = self._validate(degrees_list)

    def current_degrees(self):
        with self._lock:
            return list(self._degrees)

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState

        rate = rospy.Rate(self._hz)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            with self._lock:
                degrees = list(self._degrees)
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = list(ARM_JOINT_NAMES)
            msg.position = degrees
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break

    @staticmethod
    def _validate(degrees_list):
        values = [float(v) for v in degrees_list]
        if len(values) != 14:
            raise ValueError(f"arm command has {len(values)} joints, expected 14")
        return values
