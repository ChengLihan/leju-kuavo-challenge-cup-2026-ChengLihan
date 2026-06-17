#!/usr/bin/env python3
"""Expose simulated gripper state and service through the real Leju claw API."""

import threading

import rospy
from kuavo_msgs.msg import endEffectorData, lejuClawCommand, lejuClawState
from kuavo_msgs.srv import controlLejuClaw, controlLejuClawResponse
from sensor_msgs.msg import JointState


CLAW_NAMES = ["left_claw", "right_claw"]
GRIPPER_JOINT_NAMES = ["left_gripper_joint", "right_gripper_joint"]
DRIVER_JOINT_MAX = 0.8


def clamp(value, low, high):
    return max(low, min(high, float(value)))


def driver_position_to_percent(position):
    return clamp(position / DRIVER_JOINT_MAX * 100.0, 0.0, 100.0)


class SimLejuClawInterface:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_percent = [0.0, 0.0]
        self._target_percent = [0.0, 0.0]
        self._last_velocity = [0.0, 0.0]
        self._last_effort = [0.0, 0.0]
        self._has_state = False

        self._command_pub = rospy.Publisher("/leju_claw_command", lejuClawCommand, queue_size=10)
        self._state_pub = rospy.Publisher("/leju_claw_state", lejuClawState, queue_size=10)
        rospy.Subscriber("/gripper/state", JointState, self._gripper_state_cb)
        rospy.Subscriber("/leju_claw_command", lejuClawCommand, self._topic_command_cb)
        self._service = rospy.Service("/control_robot_leju_claw", controlLejuClaw, self._service_cb)
        self._timer = rospy.Timer(rospy.Duration(0.02), self._publish_state)

        rospy.loginfo("sim_leju_claw_interface: /control_robot_leju_claw, /leju_claw_state ready")

    def _gripper_state_cb(self, msg):
        with self._lock:
            for i, name in enumerate(msg.name):
                if name not in GRIPPER_JOINT_NAMES:
                    continue
                side = GRIPPER_JOINT_NAMES.index(name)
                if i < len(msg.position):
                    self._last_percent[side] = driver_position_to_percent(msg.position[i])
                if i < len(msg.velocity):
                    self._last_velocity[side] = msg.velocity[i] / DRIVER_JOINT_MAX * 100.0
                if i < len(msg.effort):
                    self._last_effort[side] = msg.effort[i]
            self._has_state = True

    def _extract_targets(self, data):
        if len(data.name) != len(data.position):
            raise ValueError("name and position arrays must have the same length")

        targets = list(self._target_percent)
        seen = set()
        for i, name in enumerate(data.name):
            if name not in CLAW_NAMES:
                raise ValueError("name must be 'left_claw' or 'right_claw'")
            side = CLAW_NAMES.index(name)
            seen.add(side)
            targets[side] = clamp(data.position[i], 0.0, 100.0)

        if len(seen) == 0:
            raise ValueError("no valid claw target in request")
        return targets

    def _set_targets(self, targets):
        with self._lock:
            self._target_percent = list(targets)

    def _publish_command(self, targets):
        msg = lejuClawCommand()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "leju_claw"
        msg.data = endEffectorData()
        msg.data.name = CLAW_NAMES
        msg.data.position = [clamp(targets[0], 0.0, 100.0), clamp(targets[1], 0.0, 100.0)]
        self._command_pub.publish(msg)
        self._set_targets(targets)

    def _service_cb(self, req):
        try:
            targets = self._extract_targets(req.data)
        except ValueError as exc:
            return controlLejuClawResponse(success=False, message=str(exc))

        self._publish_command(targets)
        return controlLejuClawResponse(success=True, message="success")

    def _topic_command_cb(self, msg):
        try:
            targets = self._extract_targets(msg.data)
        except ValueError as exc:
            rospy.logwarn("sim_leju_claw_interface: invalid /leju_claw_command: %s", exc)
            return
        self._set_targets(targets)

    def _publish_state(self, _event):
        with self._lock:
            current = list(self._last_percent)
            target = list(self._target_percent)
            velocity = list(self._last_velocity)
            effort = list(self._last_effort)
            has_state = self._has_state

        msg = lejuClawState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "leju_claw"
        msg.data = endEffectorData()
        msg.data.name = CLAW_NAMES
        msg.data.position = current
        msg.data.velocity = velocity
        msg.data.effort = effort
        if not has_state:
            msg.state = [lejuClawState.kUnknown, lejuClawState.kUnknown]
        else:
            msg.state = [
                lejuClawState.kReached if abs(current[i] - target[i]) < 3.0 else lejuClawState.kMoving
                for i in range(2)
            ]
        self._state_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("sim_leju_claw_interface")
    SimLejuClawInterface()
    rospy.spin()
