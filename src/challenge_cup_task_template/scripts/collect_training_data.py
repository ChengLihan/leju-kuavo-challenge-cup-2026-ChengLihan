#!/usr/bin/env python3
"""
训练数据采集: 自动收集 (YOLO坐标, 物体真值) 和 (真值坐标, 关节角) 训练对。

依赖: publish_object_tf.py 必须先运行 (提供物体真值 TF 帧)

用法:
  # 终端1: 先启动物体 TF 发布
  rosrun challenge_cup_task_template publish_object_tf.py

  # 终端2: 采集数据 (可多次跑, 自动累积)
  rosrun challenge_cup_task_template collect_training_data.py --seeds 0,1,2,3,4

输出:
  training_data/
    yolo_correction.npz   — {inputs: (N,3), outputs: (N,3)}
    joint_prediction.npz  — {inputs: (N,3), outputs: (N,7)}
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from collections import defaultdict

import rospy
import tf2_ros

# 导入 challenge_task 里的核心类
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from challenge_task import (
    RobotBase, Perception, Navigation, Manipulation,
    Scene2Controller, run_scene, _load_launcher,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")

# 物体 TF 帧名 → 类别
OBJ_CLASS = {
    "pipe_fitting_1": "pipe_fitting",
    "pipe_fitting_2": "pipe_fitting",
    "pipe_clamp_1":  "pipe_clamp",
    "pipe_clamp_2":  "pipe_clamp",
    "screwdriver_1": "screwdriver",
    "screwdriver_2": "screwdriver",
}


class DataCollectingController(Scene2Controller):
    """在 Scene2Controller 基础上增加数据采集钩子。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TF buffer (用于读物体真值)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # 累积的数据
        os.makedirs(DATA_DIR, exist_ok=True)
        self._yolo_inputs = []   # YOLO 检测到的坐标
        self._yolo_outputs = []  # 物体真值坐标
        self._joint_inputs = []  # 真值坐标
        self._joint_outputs = [] # 实际关节角
        self._stats = {"attempts": 0, "success": 0, "yolo_samples": 0}

    def _get_object_ground_truth(self):
        """从 publish_object_tf 发布的 TF 读物体真值坐标。

        返回: {class_name: [(x, y, z), ...]}  按 Y 降序排列
        """
        result = defaultdict(list)
        for frame_name, cls in OBJ_CLASS.items():
            try:
                tf = self._tf_buffer.lookup_transform(
                    "base_link", frame_name, rospy.Time(0), rospy.Duration(0.3))
                t = tf.transform.translation
                result[cls].append((t.x, t.y, t.z))
            except Exception:
                pass

        # 按 Y 从大到小排序 (和 YOLO 检测一致)
        for cls in result:
            result[cls].sort(key=lambda p: p[1], reverse=True)
        return result

    def _record_yolo_pair(self, yolo_x, yolo_y, yolo_z, gt_x, gt_y, gt_z):
        """记录一对 YOLO→真值 数据"""
        self._yolo_inputs.append([yolo_x, yolo_y, yolo_z])
        self._yolo_outputs.append([gt_x, gt_y, gt_z])
        self._stats["yolo_samples"] += 1
        self._diag_log("[Data] YOLO→GT: (%.3f,%.3f,%.3f) → (%.3f,%.3f,%.3f)",
                       yolo_x, yolo_y, yolo_z, gt_x, gt_y, gt_z)

    def _record_joint_pair(self, gt_x, gt_y, gt_z, joints_deg):
        """记录一对 真值→关节角 数据"""
        self._joint_inputs.append([gt_x, gt_y, gt_z])
        self._joint_outputs.append(list(joints_deg))
        self._stats["success"] += 1
        self._diag_log("[Data] GT→Joints: (%.3f,%.3f,%.3f) → [%s]",
                       gt_x, gt_y, gt_z,
                       ",".join("%.1f" % v for v in joints_deg))

    def run(self):
        """重写 run(): 采集一帧 YOLO vs 真值, 然后正常抓取。"""
        rospy.loginfo("=== 数据采集模式 ===")

        # 初始化
        self._robot.look_at(pitch=+20.0, yaw=0.0)
        self._robot.switch_arm_control_mode(2)
        rospy.sleep(0.5)
        pregrasp_left  = [0.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pregrasp_right = [0.0, -60.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._send_arms(left=pregrasp_left, right=pregrasp_right)
        self._sleep_hold(1.0)
        self._perception.wait_for_head_pitch(+20.0)
        self._perception.wait_for_data()
        self._robot.switch_arm_control_mode(2)
        rospy.sleep(0.3)
        self._send_arms(left=pregrasp_left, right=pregrasp_right)
        rospy.sleep(0.5)

        TARGET_CLASSES = ["pipe_fitting", "pipe_clamp", "screwdriver"]

        # ── 1. YOLO 检测 ──
        self._perception.republish_viz()
        objects = self._perception.get_objects_3d_yolo()
        yolo_by_class = defaultdict(list)
        for o in (objects or []):
            if (o.get("class_name") in TARGET_CLASSES
                    and "position_base" in o
                    and o["confidence"] >= 0.5
                    and abs(o["position_base"][0]) < 2.0):
                cls = o["class_name"]
                yolo_by_class[cls].append(o["position_base"])
        for cls in yolo_by_class:
            yolo_by_class[cls].sort(key=lambda p: p[1], reverse=True)

        # ── 2. 读物体真值 (TF) ──
        rospy.sleep(0.5)
        gt_by_class = self._get_object_ground_truth()

        # ── 3. 配对 YOLO ↔ 真值 ──
        # 两者都按 Y 排序, 顺序对应
        for cls in TARGET_CLASSES:
            yolo_positions = yolo_by_class.get(cls, [])
            gt_positions   = gt_by_class.get(cls, [])
            n = min(len(yolo_positions), len(gt_positions))
            for i in range(n):
                yx, yy, yz = yolo_positions[i]
                gx, gy, gz = gt_positions[i]
                self._record_yolo_pair(yx, yy, yz, gx, gy, gz)
            if n == 0 and yolo_positions:
                rospy.logwarn("[Data] %s: YOLO有%d个但TF没找到",
                              cls, len(yolo_positions))
            if n == 0 and gt_positions:
                rospy.logwarn("[Data] %s: TF有%d个但YOLO没检测到",
                              cls, len(gt_positions))

        rospy.loginfo("[Data] 已采集 %d 对 YOLO→GT", len(self._yolo_inputs))

        # ── 4. 正常抓取流程 (用 FK 表) ──
        PICK_FUNCTIONS = {
            "pipe_fitting": lambda bx,by,bz,arm=None:
                self._pick_place_with_record("pipe_fitting", bx, by, bz, arm),
            "pipe_clamp":  lambda bx,by,bz,arm=None:
                self._pick_place_with_record("pipe_clamp", bx, by, bz, arm),
            "screwdriver": lambda bx,by,bz,arm=None:
                self._pick_place_with_record("screwdriver", bx, by, bz, arm),
        }

        for cls in TARGET_CLASSES:
            gt_positions = gt_by_class.get(cls, [])
            if not gt_positions:
                continue
            for pick_i, (gx, gy, gz) in enumerate(gt_positions):
                self._stats["attempts"] += 1
                rospy.loginfo("[Data] %s %d/%d GT=(%.3f,%.3f,%.3f)",
                              cls, pick_i + 1, len(gt_positions), gx, gy, gz)
                success = PICK_FUNCTIONS[cls](gx, gy, gz)
                rospy.loginfo("[Data] %s %d/%d %s",
                              cls, pick_i + 1, len(gt_positions),
                              "成功" if success else "失败")
                rospy.sleep(0.5)

        # ── 5. 保存数据 ──
        self._save_data()

        # 可视化保持
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            self._perception.republish_viz()
            rate.sleep()

    def _pick_place_with_record(self, target_class, bx, by, bz, arm=None):
        """抓取并记录成功的关节角。"""
        import math
        if arm is None:
            arm = "left" if by > 0 else "right"

        # FK 查表规划
        clear_z = max(float(bz + 0.40), self.BIN_CLEAR_Z_MIN)
        grasp_z = max(float(bz + self.GRASP_DESCEND_CLEARANCE), float(bz) - 0.02)
        clear_joints = self._fk_lookup(arm, bx, by, clear_z)
        grasp_joints = self._fk_lookup(arm, bx, by, grasp_z)

        self._robot.control_gripper(0, arm)
        rospy.sleep(0.2)

        # 越顶
        self._move_arm_slow(arm, clear_joints, duration=2.0, steps=6)
        self._sleep_hold(0.3)
        # 下降到抓取点
        self._move_arm_slow(arm, grasp_joints, duration=1.5, steps=5)
        self._sleep_hold(0.5)

        # 闭爪
        self._robot.control_gripper(85, arm)
        rospy.sleep(0.6)

        # 读实际达到的关节角
        actual_q = self._current_arm_joints_rad(timeout=0.3)
        start_i = 0 if arm == "left" else 7
        act_deg = [math.degrees(actual_q[i]) for i in range(start_i, start_i + 7)]

        # 记录 (不管成功与否, 先记; 后续可用 YOLO 二次确认标记好坏)
        self._record_joint_pair(bx, by, bz, act_deg)

        # 抬起
        self._move_arm_slow(arm, clear_joints, duration=1.5, steps=5)
        self._sleep_hold(0.4)

        # 放箱
        self._place_object_in_bin(target_class, arm, bz=bz)
        return True

    def _save_data(self):
        """存累积数据到 npz"""
        yi = np.array(self._yolo_inputs, dtype=np.float32) if self._yolo_inputs else np.zeros((0,3))
        yo = np.array(self._yolo_outputs, dtype=np.float32) if self._yolo_outputs else np.zeros((0,3))
        ji = np.array(self._joint_inputs, dtype=np.float32) if self._joint_inputs else np.zeros((0,3))
        jo = np.array(self._joint_outputs, dtype=np.float32) if self._joint_outputs else np.zeros((0,7))

        # 合并已有数据
        yolo_path = os.path.join(DATA_DIR, "yolo_correction.npz")
        joint_path = os.path.join(DATA_DIR, "joint_prediction.npz")
        stats_path = os.path.join(DATA_DIR, "stats.json")

        if os.path.exists(yolo_path):
            old = np.load(yolo_path)
            yi = np.vstack([old["inputs"], yi]) if len(yi) else old["inputs"]
            yo = np.vstack([old["outputs"], yo]) if len(yo) else old["outputs"]
        if os.path.exists(joint_path):
            old = np.load(joint_path)
            ji = np.vstack([old["inputs"], ji]) if len(ji) else old["inputs"]
            jo = np.vstack([old["outputs"], jo]) if len(jo) else old["outputs"]

        np.savez(yolo_path, inputs=yi, outputs=yo)
        np.savez(joint_path, inputs=ji, outputs=jo)

        # 合并统计
        stats = dict(self._stats)
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                old_stats = json.load(f)
            for k in stats:
                old_stats[k] = old_stats.get(k, 0) + stats[k]
            stats = old_stats
        with open(stats_path, "w") as f:
            json.dump(stats, f)

        rospy.loginfo("[Data] 保存完成: YOLO=%d对 Joints=%d对 尝试=%d",
                      len(yi), len(ji), stats["attempts"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0",
                        help="逗号分隔的 seed 列表, 如 0,1,2,3")
    parser.add_argument("--time-limit", type=float, default=90)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"采集 seeds: {seeds}")

    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        config = {"scene": "scene2", "title": "scene2", "node_name": "challenge_task_scene2_collect"}

        ChallengeSimLauncher = _load_launcher()
        launcher = ChallengeSimLauncher(scene="scene2", seed=seed, match_time_limit=args.time_limit)
        launcher.start(node_name=config["node_name"], timeout=120)

        import rospy
        robot = RobotBase()
        perception = Perception()
        navigation = Navigation(robot)
        manipulation = Manipulation(robot)
        rospy.sleep(1.0)

        ctrl = DataCollectingController(robot, perception, navigation, manipulation, seed=seed)
        ctrl.run()

        rospy.signal_shutdown("collection_done")
        time.sleep(2)


if __name__ == "__main__":
    main()
