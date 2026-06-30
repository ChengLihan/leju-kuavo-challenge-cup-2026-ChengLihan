#!/usr/bin/env python3
"""
训练专用: 发布物体真值 TF 帧。

原理: 订阅 /sensors_data_raw (合法接口) 获取机器人关节角,
      在本地 MuJoCo 副本中设置关节角 → mj_forward → 读取物体 body 在 base_link 下的位置,
      发布为 TF 帧。

完全不碰 /mujoco/qpos (禁区), 训练/数据采集阶段使用, 比赛时不启动。

用法:
  rosrun challenge_cup_task_template publish_object_tf.py
"""

import rospy
import tf2_ros
import geometry_msgs.msg
import numpy as np
import mujoco
import os

XML_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../challenge_cup_simulator/models/biped_s52/xml/biped_s52.xml"
)

# 物体 body 名 → 逻辑名
OBJECT_BODIES = {
    "part_type_a_1": "pipe_fitting_1",
    "part_type_a_2": "pipe_fitting_2",
    "part_type_b_1": "pipe_clamp_1",
    "part_type_b_2": "pipe_clamp_2",
    "part_type_c_1": "screwdriver_1",
    "part_type_c_2": "screwdriver_2",
}

# MuJoCo joint qpos 索引 (来自之前探索)
# qpos: free(7) + leg_l(6) + leg_r(6) + waist(1) + arm_l(7) + gripper(8) + arm_r(7) + head(2)
# sensors joint_q: leg_l(6) + leg_r(6) + waist(1) + arm_l(7) + arm_r(7) + head(2)  = 29
# 映射: sensors[i] → qpos[7+i]  for i in 0..28,  但 gripper 8 维在 sensors 里没有
LEG_L_QPOS  = list(range(7, 13))    # qpos 7-12
LEG_R_QPOS  = list(range(13, 19))   # qpos 13-18
WAIST_QPOS  = 19
HEAD_QPOS   = [42, 43]
# sensors 索引
LEG_L_SENS  = list(range(0, 6))
LEG_R_SENS  = list(range(6, 12))
WAIST_SENS  = 12
ARM_L_SENS  = list(range(13, 20))
ARM_R_SENS  = list(range(20, 27))
HEAD_SENS   = [27, 28]


class ObjectTFPublisher:
    def __init__(self):
        rospy.init_node("object_tf_publisher", anonymous=True)

        rospy.loginfo("加载 MuJoCo: %s", XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        # 找 base_link body
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        rospy.loginfo("base_link body_id=%d", self.base_id)

        # 预设所有物体 body ID
        self.obj_ids = {}
        for obj_name, logical in OBJECT_BODIES.items():
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if bid >= 0:
                self.obj_ids[obj_name] = (bid, logical)
                rospy.loginfo("  %s → %s  body_id=%d", obj_name, logical, bid)
            else:
                rospy.logwarn("找不到: %s", obj_name)

        self.br = tf2_ros.TransformBroadcaster()

        from kuavo_msgs.msg import sensorsData
        self.sub = rospy.Subscriber("/sensors_data_raw", sensorsData, self._cb, queue_size=1)
        rospy.loginfo("object_tf_publisher 就绪 (%d 物体)", len(self.obj_ids))

    def _cb(self, msg):
        try:
            q = list(msg.joint_data.joint_q)
        except Exception:
            return

        if len(q) < 29:
            return  # 还没初始化完

        # ── 把传感器关节同步到本地 MuJoCo 副本 ──
        for i, (si, qi) in enumerate(zip(LEG_L_SENS, LEG_L_QPOS)):
            if si < len(q): self.data.qpos[qi] = q[si]
        for si, qi in zip(LEG_R_SENS, LEG_R_QPOS):
            if si < len(q): self.data.qpos[qi] = q[si]
        if WAIST_SENS < len(q):
            self.data.qpos[WAIST_QPOS] = q[WAIST_SENS]
        for si, qi in zip(ARM_L_SENS, range(20, 27)):
            if si < len(q): self.data.qpos[qi] = q[si]
        for si, qi in zip(ARM_R_SENS, range(35, 42)):
            if si < len(q): self.data.qpos[qi] = q[si]
        for si, qi in zip(HEAD_SENS, HEAD_QPOS):
            if si < len(q): self.data.qpos[qi] = q[si]

        mujoco.mj_forward(self.model, self.data)
        base_pos = self.data.xpos[self.base_id].copy() if self.base_id >= 0 else np.zeros(3)
        stamp = rospy.Time.now()

        tfs = []
        for obj_name, (bid, logical) in self.obj_ids.items():
            world_pos = self.data.xpos[bid].copy()
            rel = world_pos - base_pos
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "base_link"
            t.child_frame_id = logical
            t.transform.translation.x = rel[0]
            t.transform.translation.y = rel[1]
            t.transform.translation.z = rel[2]
            t.transform.rotation.w = 1.0
            tfs.append(t)

        if tfs:
            self.br.sendTransform(tfs)


if __name__ == "__main__":
    ObjectTFPublisher()
    rospy.spin()
