#!/usr/bin/env python3
"""
离线 FK 采样建表: 纯 MuJoCo FK, 替代 IK。

关键: MuJoCo FK 已验证与仿真 TF 一致(误差 <2cm)!
      IK 服务 FK 与仿真 TF 不一致(误差 20cm+)。

输出: fk_table_left.npy, fk_table_right.npy
  每行: [arm_pitch, arm_roll, arm_yaw, forearm, end_eff_X, Y, Z]
  (前 4 列=关节角度度, 后 3 列=base_link 下的末端坐标米)
"""

import mujoco
import numpy as np
import os, time

XML = os.path.join(os.path.dirname(__file__),
                    "../../challenge_cup_simulator/models/biped_s52/xml/biped_s52.xml")

# MuJoCo qpos 索引
L_START = 20  # zarm_l1..l7 = qpos[20:27]
R_START = 35  # zarm_r1..r7 = qpos[35:42]

PIDX, RIDX, YIDX, FIDX = 0, 1, 2, 3
HYAW, HPIT, HROL = 4, 5, 6

# 只采前 3 关节 (决定 XY), forearm 固定
PITCH_RANGE = np.arange(-60, 35, 4)     # 24 levels
ROLL_RANGE  = np.arange(10, 92, 4)      # 21 levels (降至10°才能过中线)
YAW_RANGE   = np.arange(-45, 46, 4)     # 23 levels
FORE_SAMPLES = [-130, -110, -95, -80, -65, -50, -35]  # 7 levels

# 手腕固定 (指向下方)
WRIST = {
    "left":  [82.0, 26.0, -11.0],
    "right": [90.0, 40.0,  0.0],
}

# 末端局部偏移 (URDF: zarm_l7_end_effector offset from zarm_l7_link)
EE_LOCAL = np.array([0.0, 0.0, -0.17])


def build_table():
    print(f"Loading: {XML}")
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)

    # ── 设置站立姿态 (精确的传感器初值) ──
    leg_l = [0.006, 0.0, -0.503, 0.862, -0.36, -0.006]
    leg_r = [-0.007, 0.0, -0.502, 0.861, -0.36, 0.007]
    for i, v in enumerate(leg_l): data.qpos[7+i] = v
    for i, v in enumerate(leg_r): data.qpos[13+i] = v
    data.qpos[19] = 0.0   # waist_yaw
    data.qpos[42] = 0.0   # head_yaw
    data.qpos[43] = 0.35  # head_pitch

    l7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "zarm_l7_link")
    r7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "zarm_r7_link")
    base_id = 1  # base_link

    total = len(PITCH_RANGE) * len(ROLL_RANGE) * len(YAW_RANGE) * len(FORE_SAMPLES)
    print(f"Total samples: {total}")
    t0 = time.time()

    left_rows, right_rows = [], []

    for pitch in PITCH_RANGE:
        for roll in ROLL_RANGE:
            for yaw in YAW_RANGE:
                for fore in FORE_SAMPLES:
                    # Left arm
                    data.qpos[L_START+PIDX] = np.deg2rad(pitch)
                    data.qpos[L_START+RIDX] = np.deg2rad(roll)
                    data.qpos[L_START+YIDX] = np.deg2rad(yaw)
                    data.qpos[L_START+FIDX] = np.deg2rad(fore)
                    data.qpos[L_START+HYAW] = np.deg2rad(WRIST["left"][0])
                    data.qpos[L_START+HPIT] = np.deg2rad(WRIST["left"][1])
                    data.qpos[L_START+HROL] = np.deg2rad(WRIST["left"][2])

                    # Right arm (ROLL取负, 镜像左臂)
                    data.qpos[R_START+PIDX] = np.deg2rad(pitch)
                    data.qpos[R_START+RIDX] = np.deg2rad(-roll)
                    data.qpos[R_START+YIDX] = np.deg2rad(-yaw)
                    data.qpos[R_START+FIDX] = np.deg2rad(fore)
                    data.qpos[R_START+HYAW] = np.deg2rad(WRIST["right"][0])
                    data.qpos[R_START+HPIT] = np.deg2rad(WRIST["right"][1])
                    data.qpos[R_START+HROL] = np.deg2rad(WRIST["right"][2])

                    mujoco.mj_forward(model, data)
                    base_pos = data.xpos[base_id]

                    # 左臂末端 = zarm_l7 + R * (0,0,-0.17), 在 base_link 系
                    Rl = data.xmat[l7_id].reshape(3, 3)
                    ee_l = data.xpos[l7_id] + Rl @ EE_LOCAL - base_pos
                    left_rows.append([pitch, roll, yaw, fore,
                                     ee_l[0], ee_l[1], ee_l[2]])

                    # 右臂
                    Rr = data.xmat[r7_id].reshape(3, 3)
                    ee_r = data.xpos[r7_id] + Rr @ EE_LOCAL - base_pos
                    right_rows.append([pitch, roll, yaw, fore,
                                      ee_r[0], ee_r[1], ee_r[2]])

    dt = time.time() - t0
    left_arr  = np.array(left_rows,  dtype=np.float32)
    right_arr = np.array(right_rows, dtype=np.float32)
    print(f"Done in {dt:.1f}s ({total/dt:.0f} pts/s)")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(out_dir, "fk_table_left.npy"),  left_arr)
    np.save(os.path.join(out_dir, "fk_table_right.npy"), right_arr)

    print(f"\nLeft  EndEff XYZ range: X=[{left_arr[:,4].min():.2f},{left_arr[:,4].max():.2f}] "
          f"Y=[{left_arr[:,5].min():.2f},{left_arr[:,5].max():.2f}] "
          f"Z=[{left_arr[:,6].min():.2f},{left_arr[:,6].max():.2f}]")
    print(f"Right EndEff XYZ range: X=[{right_arr[:,4].min():.2f},{right_arr[:,4].max():.2f}] "
          f"Y=[{right_arr[:,5].min():.2f},{right_arr[:,5].max():.2f}] "
          f"Z=[{right_arr[:,6].min():.2f},{right_arr[:,6].max():.2f}]")

    # 验证: IK grasp cmd 的点
    test_cmd = [21.5, 84.0, -11.1, -95.2]
    dists = np.sqrt(((left_arr[:,:4] - test_cmd)**2).sum(axis=1))
    nearest = left_arr[np.argmin(dists)]
    print(f"\nVerification: cmd={test_cmd}")
    print(f"  Nearest table: joints={nearest[:4].tolist()} ee=({nearest[4]:.3f},{nearest[5]:.3f},{nearest[6]:.3f})")
    print(f"  Expected (sim TF): (0.369, 0.591, 0.165)")


if __name__ == "__main__":
    build_table()
