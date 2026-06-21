#!/usr/bin/env python3
"""
技能桥接模块：调用抓取/抽盘策略和放置策略的接口。
当前为 mock 实现，后续可接入真实模仿学习策略。
"""

import rospy


class SkillBridge:
    def __init__(self, iface, perception):
        self.iface = iface
        self.perception = perception

    def run_extract_upper_policy(self, timeout=20.0):
        rospy.loginfo("SkillBridge: extract_upper_policy (MOCK) - simulating tray extraction")

        rospy.sleep(3.0)

        success = True
        if success:
            closed = self.iface.is_claw_closed("right")
            rospy.loginfo("SkillBridge: extract mock done, claw_closed=%s", closed)
            return True

        return False

    def run_place_policy(self, timeout=10.0):
        rospy.loginfo("SkillBridge: place_policy (MOCK) - simulating tray placement")

        rospy.sleep(2.0)

        success = True
        if success:
            rospy.loginfo("SkillBridge: place mock done")
            return True

        return False

    def build_extract_observation(self):
        if self.perception is None:
            return {}
        points = self.perception.build_head_xyzrgb_cloud(max_points=1024)
        points = self.perception.crop_scene3_roi(points)
        arm_q = self.iface.get_arm_joint_positions()
        return {
            "head_xyzrgb_points": points,
            "q_arm": arm_q,
            "stage_id": "extract_upper",
        }

    def build_place_observation(self):
        if self.perception is None:
            return {}
        points = self.perception.build_head_xyzrgb_cloud(max_points=1024)
        points = self.perception.crop_scene3_roi(points)
        arm_q = self.iface.get_arm_joint_positions()
        return {
            "head_xyzrgb_points": points,
            "q_arm": arm_q,
            "stage_id": "place_upper",
        }

    def verify_tray_extracted(self):
        if self.iface.is_claw_closed("right"):
            rospy.loginfo("SkillBridge: tray extraction verified (claw closed)")
            return True
        rospy.logwarn("SkillBridge: tray extraction NOT verified (claw open)")
        return False
