#!/usr/bin/env python3
"""
Scene3 决策与导航控制器：SMT 上层料盘出库状态机。
使用 state_estimate 位置估计做闭环导航。

流程：
  INIT -> APPROACH_SHELF -> PREGRASP_UPPER -> EXTRACT_UPPER
  -> TRANSPORT_POSE -> RETREAT -> TURN_TO_BOX -> APPROACH_BOX
  -> PREPLACE -> PLACE -> FINISH
"""

import os
import sys
import rospy
import numpy as np
import yaml


S0_INIT = "init"
S1_APPROACH_SHELF = "approach_shelf"
S2_PREGRASP_UPPER = "pregrasp_upper"
S3_EXTRACT_UPPER = "extract_upper"
S4_TRANSPORT_POSE = "transport_pose"
S5_RETREAT = "retreat"
S6_TURN_TO_BOX = "turn_to_box"
S7_APPROACH_BOX = "approach_box"
S8_PREPLACE = "preplace"
S9_PLACE = "place"
S10_FINISH = "finish"
S_FAIL = "fail"


def _resolve_config_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
    )


def _ensure_src_path():
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def load_yaml(filename):
    path = os.path.join(_resolve_config_dir(), filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Scene3TaskController:
    def __init__(self, nav_config_path=None, pose_config_path=None):
        _ensure_src_path()

        if nav_config_path is None:
            nav_config_path = os.path.join(_resolve_config_dir(), "scene3_nav.yaml")
        if pose_config_path is None:
            pose_config_path = os.path.join(_resolve_config_dir(), "scene3_named_poses.yaml")

        self.nav_config = load_yaml("scene3_nav.yaml")
        self.pose_config_path = pose_config_path

        from ros_interfaces import InterfaceManager
        from base_navigator import BaseNavigator
        from arm_pose_manager import ArmPoseManager
        from skill_bridge import SkillBridge
        from fastlio_localization import FastLIOBridge

        self.iface = InterfaceManager()
        self.fastlio = FastLIOBridge()
        self.nav = BaseNavigator(self.iface, None, self.nav_config)
        self.arm = ArmPoseManager(self.iface, pose_config_path)
        self.skill = SkillBridge(self.iface, None)

        self.nav.set_position_source(self.fastlio)

        self.state = S0_INIT
        self._fail_msg = ""

    def run(self):
        try:
            while not rospy.is_shutdown():
                if self.state == S0_INIT:
                    ok = self._do_init()
                    self.state = S1_APPROACH_SHELF if ok else S_FAIL

                elif self.state == S1_APPROACH_SHELF:
                    ok = self._do_approach_shelf()
                    self.state = S2_PREGRASP_UPPER if ok else S_FAIL

                elif self.state == S2_PREGRASP_UPPER:
                    ok = self._do_pregrasp_upper()
                    self.state = S3_EXTRACT_UPPER if ok else S_FAIL

                elif self.state == S3_EXTRACT_UPPER:
                    ok = self._do_extract_upper()
                    self.state = S4_TRANSPORT_POSE if ok else S_FAIL

                elif self.state == S4_TRANSPORT_POSE:
                    ok = self._do_transport_pose()
                    self.state = S5_RETREAT if ok else S_FAIL

                elif self.state == S5_RETREAT:
                    ok = self._do_retreat()
                    self.state = S6_TURN_TO_BOX if ok else S_FAIL

                elif self.state == S6_TURN_TO_BOX:
                    ok = self._do_turn_to_box()
                    self.state = S7_APPROACH_BOX if ok else S_FAIL

                elif self.state == S7_APPROACH_BOX:
                    ok = self._do_approach_box()
                    self.state = S8_PREPLACE if ok else S_FAIL

                elif self.state == S8_PREPLACE:
                    ok = self._do_preplace()
                    self.state = S9_PLACE if ok else S_FAIL

                elif self.state == S9_PLACE:
                    ok = self._do_place()
                    self.state = S10_FINISH if ok else S_FAIL

                elif self.state == S10_FINISH:
                    self._do_finish()
                    rospy.loginfo("Scene3TaskController: task completed successfully")
                    return True

                elif self.state == S_FAIL:
                    self._do_emergency_stop()
                    rospy.logerr("Scene3TaskController: task FAILED - %s", self._fail_msg)
                    return False

        except rospy.ROSInterruptException:
            self._do_emergency_stop()
            return False
        except Exception as e:
            rospy.logerr("Scene3TaskController: unexpected error: %s", e)
            self._do_emergency_stop()
            return False

    def _do_init(self):
        rospy.loginfo("[init] Initializing Scene3...")

        if not self.iface.wait_for_all_topics(timeout=self.nav_config["timeouts"]["init"]):
            self._fail_msg = "topics not ready"
            return False

        self.iface.stop_base()
        rospy.sleep(0.5)

        if not self.iface.switch_arm_control_mode(mode=2):
            rospy.logwarn("[init] arm mode switch failed, continuing...")
        rospy.sleep(0.5)

        self.iface.open_claw()
        rospy.sleep(0.8)

        if not self.arm.move_to_safe_home():
            self._fail_msg = "safe_home failed"
            return False

        rospy.sleep(0.5)

        rospy.loginfo("[init] Waiting for FastLIO to initialize...")
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            pos = self.fastlio.get_position()
            if pos is not None and np.linalg.norm(pos) > 1e-6:
                rospy.loginfo("[init] FastLIO ready, pos=%s", pos[:2])
                break
            if (rospy.Time.now() - start).to_sec() > 15.0:
                rospy.logwarn("[init] FastLIO init timeout, using state_estimate fallback")
                self.nav.set_position_source(None)
                break
            rospy.sleep(0.3)

        pos = self.nav._get_pos()
        yaw = self.nav._get_yaw()
        rospy.loginfo("[init] Complete. pos=%s, yaw=%.2f",
                       pos[:2] if pos is not None else "N/A", yaw)
        return True

    def _do_approach_shelf(self):
        rospy.loginfo("[approach_shelf] Walking forward %.2f m...",
                       self.nav_config["navigation"]["approach_shelf_distance"])
        rospy.sleep(0.3)
        return self.nav.move_forward(
            distance=self.nav_config["navigation"]["approach_shelf_distance"],
            timeout=self.nav_config["timeouts"]["approach_shelf"],
        )

    def _do_pregrasp_upper(self):
        rospy.loginfo("[pregrasp_upper] Moving to pregrasp pose...")
        rospy.sleep(0.5)
        if not self.arm.move_to_upper_tray_pregrasp():
            return False
        self.iface.open_claw()
        rospy.sleep(0.5)
        return True

    def _do_extract_upper(self):
        rospy.loginfo("[extract_upper] Running extract policy...")
        ok = self.skill.run_extract_upper_policy(
            timeout=self.nav_config["timeouts"]["extract"]
        )
        if not ok:
            ok = self._recover_extract_once()
        return ok

    def _recover_extract_once(self):
        rospy.logwarn("[extract_upper] Recovery attempt...")
        self.iface.stop_base()
        self.iface.open_claw()
        rospy.sleep(0.5)
        self.arm.move_to_upper_tray_pregrasp()
        rospy.sleep(0.5)
        return self.skill.run_extract_upper_policy(
            timeout=self.nav_config["timeouts"]["extract"]
        )

    def _do_transport_pose(self):
        rospy.loginfo("[transport_pose] Moving to transport pose...")
        rospy.sleep(0.3)
        if not self.arm.move_to_transport_pose():
            return False
        rospy.sleep(0.5)
        return True

    def _do_retreat(self):
        rospy.loginfo("[retreat] Walking backward %.2f m...",
                       self.nav_config["navigation"]["retreat_distance"])
        rospy.sleep(0.3)
        return self.nav.move_backward(
            distance=self.nav_config["navigation"]["retreat_distance"],
            timeout=self.nav_config["timeouts"]["retreat"],
        )

    def _do_turn_to_box(self):
        rospy.loginfo("[turn_to_box] Turning 180 degrees...")
        rospy.sleep(0.3)
        return self.nav.turn_to_box()

    def _do_approach_box(self):
        rospy.loginfo("[approach_box] Walking forward %.2f m...",
                       self.nav_config["navigation"]["approach_box_distance"])
        rospy.sleep(0.3)
        return self.nav.move_forward(
            distance=self.nav_config["navigation"]["approach_box_distance"],
            timeout=self.nav_config["timeouts"]["approach_box"],
        )

    def _do_preplace(self):
        rospy.loginfo("[preplace] Moving to box preplace pose...")
        rospy.sleep(0.5)
        if not self.arm.move_to_box_preplace():
            return False
        rospy.sleep(0.5)
        return True

    def _do_place(self):
        rospy.loginfo("[place] Placing tray...")
        return self._rule_based_place()

    def _rule_based_place(self):
        if not self.arm.move_to_box_release():
            return False
        rospy.sleep(0.5)
        self.iface.open_claw()
        rospy.sleep(0.8)
        self.arm.move_to_named_pose("box_after_release")
        rospy.sleep(0.5)
        return True

    def _do_finish(self):
        rospy.loginfo("[finish] Task complete.")
        self.iface.stop_base()
        self.iface.open_claw()
        rospy.sleep(0.5)
        self.arm.move_to_finish_pose()
        rospy.sleep(0.5)
        rospy.loginfo("[finish] Done.")

    def _do_emergency_stop(self):
        rospy.logerr("Scene3TaskController: EMERGENCY STOP")
        self.iface.stop_base()
        self.arm.hold_current_pose()
