#!/usr/bin/env python3
"""Keyboard teleop for the simulated Leju two-finger claw.

Uses the real robot API:
  /control_robot_leju_claw
  /leju_claw_state

Command convention follows kuavo_msgs/controlLejuClaw:
  0   = open
  100 = closed
"""

import select
import sys
import termios
import threading
import tty

import rospy
from kuavo_msgs.msg import lejuClawState
from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest


CLAW_NAMES = ["left_claw", "right_claw"]


HELP = """
Leju claw keyboard control

  o : open both
  c : close both
  h : half open

  q / a : left open / close
  e / d : right open / close
  [ / ] : both -/+ step

  s : print state
  ? : help
  Ctrl-C : quit
"""


def clamp(value):
    return max(0.0, min(100.0, float(value)))


class ClawKeyboard:
    def __init__(self):
        self._lock = threading.Lock()
        self._position = [0.0, 0.0]
        self._state = [lejuClawState.kUnknown, lejuClawState.kUnknown]
        self._old_settings = termios.tcgetattr(sys.stdin)

        self._state_sub = rospy.Subscriber("/leju_claw_state", lejuClawState, self._state_cb)
        rospy.loginfo("waiting for /control_robot_leju_claw ...")
        rospy.wait_for_service("/control_robot_leju_claw")
        self._srv = rospy.ServiceProxy("/control_robot_leju_claw", controlLejuClaw)
        rospy.loginfo("connected to /control_robot_leju_claw")

    def _state_cb(self, msg):
        with self._lock:
            self._state = list(msg.state)
            if len(msg.data.position) >= 2:
                self._position = [clamp(msg.data.position[0]), clamp(msg.data.position[1])]

    def _get_key(self):
        tty.setraw(sys.stdin.fileno())
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        key = sys.stdin.read(1) if ready else ""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
        return key

    def _call(self, left, right):
        left = clamp(left)
        right = clamp(right)
        req = controlLejuClawRequest()
        req.data.name = CLAW_NAMES
        req.data.position = [left, right]
        req.data.velocity = [80.0, 80.0]
        req.data.effort = [1.0, 1.0]
        try:
            res = self._srv(req)
        except rospy.ServiceException as exc:
            rospy.logerr("control_robot_leju_claw failed: %s", exc)
            return
        if not res.success:
            rospy.logwarn("control_robot_leju_claw rejected command: %s", res.message)
            return
        with self._lock:
            self._position = [left, right]
        rospy.loginfo("target: left=%.1f right=%.1f", left, right)

    def _print_state(self):
        with self._lock:
            pos = list(self._position)
            state = list(self._state)
        rospy.loginfo("state: left=%.1f right=%.1f raw_state=%s", pos[0], pos[1], state)

    def run(self):
        print(HELP)
        step = float(rospy.get_param("~step", 10.0))
        try:
            while not rospy.is_shutdown():
                key = self._get_key()
                if not key:
                    continue

                with self._lock:
                    left, right = self._position

                if key == "\x03":
                    break
                if key == "?":
                    print(HELP)
                elif key == "o":
                    self._call(0.0, 0.0)
                elif key == "c":
                    self._call(100.0, 100.0)
                elif key == "h":
                    self._call(50.0, 50.0)
                elif key == "q":
                    self._call(0.0, right)
                elif key == "a":
                    self._call(100.0, right)
                elif key == "e":
                    self._call(left, 0.0)
                elif key == "d":
                    self._call(left, 100.0)
                elif key == "[":
                    self._call(left - step, right - step)
                elif key == "]":
                    self._call(left + step, right + step)
                elif key == "s":
                    self._print_state()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)


if __name__ == "__main__":
    rospy.init_node("leju_claw_keyboard")
    ClawKeyboard().run()
