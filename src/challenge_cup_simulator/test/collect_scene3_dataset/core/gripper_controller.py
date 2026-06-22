"""
core/gripper_controller.py — Gripper control backends.

Two backends:
  1. JointStateGripperHold — publishes /gripper/command JointState
  2. LejuClawCommandClient — publishes /leju_claw_command or calls service
"""

import threading


class JointStateGripperHold:
    """Continuously publishes left/right gripper positions as JointState."""

    def __init__(self, pub, cfg, hz=100.0):
        self._pub = pub
        self._hz = float(hz)
        self._cfg = cfg
        self._left = float(cfg.get("open_position", 0.0))
        self._right = float(cfg.get("open_position", 0.0))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gripper_hold", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def open(self):
        value = float(self._cfg.get("open_position", 0.0))
        with self._lock:
            self._left = value
            self._right = value

    def close(self, arm):
        open_val = float(self._cfg.get("open_position", 0.0))
        close_val = float(self._cfg.get("close_position", 255.0))
        with self._lock:
            self._left = close_val if arm == "left" else open_val
            self._right = close_val if arm == "right" else open_val

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState

        rate = rospy.Rate(self._hz)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            with self._lock:
                left, right = self._left, self._right
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ["left_gripper_joint", "right_gripper_joint"]
            msg.position = [left, right]
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break


class LejuClawCommandClient:
    """Send Leju-format claw open/close commands via topic or service."""

    def __init__(self, cfg, active_arm):
        self.cfg = cfg
        self.active_arm = active_arm
        self.backend = cfg.get("backend", "joint_state")
        self._pub = None
        self._srv = None

    def setup(self, timeout):
        import rospy

        if self.backend == "leju_topic":
            from kuavo_msgs.msg import lejuClawCommand
            self._pub = rospy.Publisher(
                self.cfg.get("leju_command_topic", "/leju_claw_command"),
                lejuClawCommand, queue_size=10,
            )
            # wait_for_connection is in the parent package
            import importlib
            rosbag_utils = importlib.import_module("scene3_rosbag_utils")
            rosbag_utils.wait_for_connection(self._pub, timeout)
        elif self.backend == "leju_service":
            from kuavo_msgs.srv import controlLejuClaw
            service = self.cfg.get("leju_service", "/control_robot_leju_claw")
            rospy.wait_for_service(service, timeout=timeout)
            self._srv = rospy.ServiceProxy(service, controlLejuClaw)

    def open(self):
        return self._send(float(self.cfg.get("leju_open_position", 10.0)), both=True)

    def close(self):
        return self._send(float(self.cfg.get("leju_close_position", 90.0)), both=False)

    def _send(self, position, both=False):
        if self.backend not in ("leju_topic", "leju_service"):
            return True

        from kuavo_msgs.msg import endEffectorData

        names = ["left_claw", "right_claw"] if both else [f"{self.active_arm}_claw"]
        data = endEffectorData()
        data.name = names
        data.position = [float(position)] * len(names)
        data.velocity = [float(self.cfg.get("velocity", 90.0))] * len(names)
        data.effort = [float(self.cfg.get("effort", 1.0))] * len(names)

        if self.backend == "leju_topic":
            from kuavo_msgs.msg import lejuClawCommand
            msg = lejuClawCommand()
            msg.data = data
            self._pub.publish(msg)
            return True

        if self.backend == "leju_service":
            from kuavo_msgs.srv import controlLejuClawRequest
            req = controlLejuClawRequest()
            req.data = data
            resp = self._srv(req)
            return bool(resp.success)

        return True
