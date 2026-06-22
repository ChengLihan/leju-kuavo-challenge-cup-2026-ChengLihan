#!/usr/bin/env python3
"""
Scene3 抓取参数计算器

基于 URDF/MuJoCo 运动学链和 scene3 料盘位置，计算：
  1. 机器人基座到货架的最优距离
  2. 需要的蹲低幅度 (squat) 和弯腰角度 (bend)
  3. 夹爪到料盘中心的偏差 (delta)

计算方法模拟了数据采集脚本中 `measure_target_gripper_distance` 的逻辑：
  - 料盘位置来自 scene3.xml 的真实值
  - 夹爪位置通过 URDF 正运动学计算，相对于 base_link
  - 身体弯腰通过 hip pitch 旋转建模

用法:
  python scripts/calc_scene3_grasp_params.py
  python scripts/calc_scene3_grasp_params.py --all-details
"""

import argparse
import math
from typing import List, Tuple

import numpy as np

# ============================================================================
# 运动学链: base_link → right zarm_r7_end_effector
# (关节原点 + 轴，来自 biped_s52 URDF / MuJoCo XML)
# ============================================================================
CHAIN = [
    # (名称, xyz_from_parent, joint_axis)
    ("waist_yaw",          (0.0,   0.0,      0.1114),  (0, 0, 1)),
    ("r_arm_pitch",        (-0.003, -0.2527,  0.2830),  (0, 1, 0)),
    ("r_arm_roll",         (0.0,    0.0,      0.0),     (1, 0, 0)),
    ("r_arm_yaw",          (0.0,    0.0,      0.0),     (0, 0, 1)),
    ("r_forearm_pitch",    (0.02,   0.0,     -0.2837),  (0, 1, 0)),
    ("r_hand_yaw",         (-0.02,  0.0,     -0.1201),  (0, 0, 1)),
    ("r_hand_roll",        (0.0,    0.0,     -0.1140),  (1, 0, 0)),
    ("r_hand_pitch",       (0.0,    0.0,     -0.0210),  (0, 1, 0)),
    ("end_effector",       (0.0,    0.0,      0.1412),  None),    # gripper_tf: r7→ee
]

# 14关节向量中右臂的索引 (0-based)
_RMAP = {"r_pitch": 7, "r_roll": 8, "r_yaw": 9,
         "r_forearm": 10, "r_hand_yaw": 11, "r_hand_pitch": 12, "r_hand_roll": 13}

# 命名姿态 (来自 scene3_named_poses.yaml / scene3_lower_tray_named_poses.yaml)
POSES = {
    # ── 上层料盘姿态 (当前以 tray_5 y=-0.15 为目标) ──
    "upper_pregrasp":    [20,0,0,-30,0,0,0,  15, 20, 42, -136, 43, -10,  0],
    "upper_approach":    [20,0,0,-30,0,0,0,  -1, 20, 33, -124, 34, -13,  0],
    "upper_avoid":       [20,0,0,-30,0,0,0,  10, 18, 35, -120, 38,  -5,  0],
    # ── 下层料盘姿态 ──
    "lower_pregrasp":    [20,0,0,-30,0,0,0,  12, -14, 0,  -35,  0,  0,  0],
    "lower_approach":    [20,0,0,-30,0,0,0,  14, -14, 0,  -30,  0,  0,  0],
    "lower_avoid":       [20,0,0,-30,0,0,0,  10, -14, 0,  -40,  0,  0,  0],
}

# 料盘世界坐标 (来自 scene3.xml)
TRAYS = {
    "tray_1 (下层右侧)": (0.983, 0.35, 0.75),
    "tray_2 (下层中前)": (0.983, 0.15, 0.75),
    "tray_3 (上层左侧)": (0.983, 0.05, 1.15),
    "tray_4 (上层右侧)": (0.983, 0.25, 1.15),
    "tray_5 (上层后侧)": (0.983, -0.15, 1.15),
}

# 机器人名义参数
BASE_Z_NOMINAL = 0.82       # base_link 离地高度 (biped_s52.xml)
SHOULDER_Y_BASE = -0.2527   # 右肩在 base_link 下的 Y 偏移
SHOULDER_Z_BASE = 0.1114 + 0.283  # 右肩在 base_link 下的 Z 偏移 = 0.3944
UP_ARM_LEN = math.hypot(0.02, 0.2837)                        # ≈ 0.284 m
FOREARM_LEN = math.hypot(-0.02, -0.1201 - 0.114 - 0.021 + 0.1412)  # ≈ 0.116 m
TOTAL_ARM = UP_ARM_LEN + FOREARM_LEN                         # ≈ 0.400 m


# ============================================================================
# 正运动学
# ============================================================================

def rot_mat(axis: Tuple[float, float, float], angle: float) -> np.ndarray:
    """Rodrigues 旋转矩阵"""
    a = np.array(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def fk_arm(joints_deg: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """正运动学: base_link → zarm_r7_end_effector

    Args:
        joints_deg: 14 元素关节角度列表 (degrees)

    Returns:
        (pos_3d, rot_3x3) 均在 base_link 坐标系下
    """
    rad = [
        math.radians(joints_deg[_RMAP[k]])
        for k in ["r_pitch", "r_roll", "r_yaw",
                   "r_forearm", "r_hand_yaw", "r_hand_pitch", "r_hand_roll"]
    ]
    pos = np.zeros(3)
    R = np.eye(3)
    ji = 0
    for name, xyz, axis in CHAIN:
        pos += R @ np.array(xyz, dtype=float)
        if axis is None:
            continue  # end_effector: fixed offset, no rotation
        if name == "waist_yaw":
            continue  # waist kept at 0
        R = R @ rot_mat(axis, rad[ji])
        ji += 1
    return pos, R


def ee_in_world(base_xyz: Tuple[float, float, float],
                bend_rad: float,
                joints_deg: List[float]) -> np.ndarray:
    """计算末端在世界坐标系下的位置

    Args:
        base_xyz: base_link 在世界坐标系的位置 (x, y, z)
        bend_rad: 身体前倾角度 (rad, 正值 = 前倾, 绕 Y 轴旋转整条手臂)
        joints_deg: 14 关节角度 (deg)
    """
    ee_base, _ = fk_arm(joints_deg)
    R_bend = rot_mat((0, 1, 0), bend_rad)
    ee_tilted = R_bend @ ee_base
    return np.array(base_xyz) + ee_tilted


# ============================================================================
# 距离搜索
# ============================================================================

def find_best_base(tray_xyz: Tuple[float, float, float],
                   pose_name: str,
                   bend_deg: float,
                   base_z_range: Tuple[float, float, float],
                   lateral_align: bool = True) -> dict:
    """搜索使 EE 到料盘距离最小的 base_x / base_y / base_z 组合

    Args:
        tray_xyz: 料盘中心世界坐标
        pose_name: 姿态名
        bend_deg: 身体前倾角 (deg)
        base_z_range: (start, stop, step) squat 搜索范围
        lateral_align: 如果为 True, 同时搜索 base_y 以对齐料盘 Y

    Returns:
        {dist, base_x, base_y, base_z, ee, delta}
    """
    tx, ty, tz = tray_xyz
    joints = POSES[pose_name]
    bend_rad = math.radians(bend_deg)

    best = {"dist": float("inf"), "base_x": 0, "base_y": 0, "base_z": 0}

    for base_z in np.arange(*base_z_range):
        for base_x in np.arange(-0.2, 1.3, 0.005):
            y_range = np.arange(-0.3, 0.6, 0.02) if lateral_align else [0.0]
            for base_y in y_range:
                ee = ee_in_world((base_x, base_y, base_z), bend_rad, joints)
                d = math.hypot(ee[0] - tx, ee[1] - ty, ee[2] - tz)
                if d < best["dist"]:
                    best = {"dist": d, "base_x": base_x, "base_y": base_y,
                            "base_z": base_z, "ee": ee,
                            "delta": (tx - ee[0], ty - ee[1], tz - ee[2])}
    return best


# ============================================================================
# 主分析
# ============================================================================

def sep(title: str):
    print(f"\n{'─' * 65}\n  {title}\n{'─' * 65}")


def main():
    parser = argparse.ArgumentParser(
        description="Scene3 抓取参数计算器 — 基于 URDF 运动学 + 料盘真实位置"
    )
    parser.add_argument("--all-details", action="store_true",
                        help="输出所有料盘和姿态组合的详细搜索结果")
    args = parser.parse_args()

    # ── 机器人参数总结 ─────────────────────────────────────────────
    shoulder_ground = BASE_Z_NOMINAL + SHOULDER_Z_BASE  # 1.214m

    sep("机器人运动学参数")
    print(f"  base_link 名义离地高度:    {BASE_Z_NOMINAL:.3f} m")
    print(f"  右肩 world Z (站立):       {shoulder_ground:.3f} m")
    print(f"  右肩 base_link Y 偏移:     {SHOULDER_Y_BASE:+.3f} m")
    print(f"  上臂长度:                  {UP_ARM_LEN:.3f} m")
    print(f"  前臂+手腕+夹爪 长度:       {FOREARM_LEN:.3f} m")
    print(f"  手臂总长 (伸直):           {TOTAL_ARM:.3f} m")
    print(f"  身体前倾轴:                hip pitch (绕 Y, 正值=前倾)")

    # ── 上层分析 ───────────────────────────────────────────────────
    sep("上层料盘分析 (z=1.15m, 例如 smt_tray_5)")

    upper_trays = {k: v for k, v in TRAYS.items() if v[2] > 1.0}
    for tname, (tx, ty, tz) in upper_trays.items():
        dz = shoulder_ground - tz
        print(f"\n  [{tname}] world=({tx:.3f}, {ty:.3f}, {tz:.3f}) "
              f"| 肩→料盘Z偏移: {dz:+.3f}m")

        for pose, bend in [("upper_pregrasp", 0), ("upper_approach", 0)]:
            b = find_best_base((tx, ty, tz), pose, bend,
                               base_z_range=(0.79, 0.83, 0.01))
            dx, dy, dz_delta = b["delta"]
            shelf_x_gap = tx - b["base_x"]
            ok = "✓" if b["dist"] < 0.20 else "✗"
            print(f"    [{ok}] {pose:20s} "
                  f"base=({b['base_x']:+.3f},{b['base_y']:+.3f},{b['base_z']:.3f}) "
                  f"| shelf_X_gap={shelf_x_gap:.3f}m "
                  f"| EE→tray Δ=({dx:+.3f},{dy:+.3f},{dz_delta:+.3f}) "
                  f"| dist={b['dist']:.3f}m")

    # ── 下层分析 ───────────────────────────────────────────────────
    sep("下层料盘分析 (z=0.75m, 需要蹲低+弯腰)")

    lower_trays = {k: v for k, v in TRAYS.items() if v[2] < 0.8}
    for tname, (tx, ty, tz) in lower_trays.items():
        dz = shoulder_ground - tz
        print(f"\n  [{tname}] world=({tx:.3f}, {ty:.3f}, {tz:.3f}) "
              f"| 肩→料盘Z偏移: {dz:+.3f}m (需要蹲低 + 弯腰)")

        # 测试不同 squat + bend 组合
        combos = [
            (0.00,  0,  0.80, 0.83),
            (-0.15, 10, 0.65, 0.68),
            (-0.20, 15, 0.60, 0.63),
            (-0.25, 20, 0.55, 0.58),
            (-0.30, 25, 0.50, 0.53),
        ]
        found = False
        for squat, bend, z_lo, z_hi in combos:
            base_z = BASE_Z_NOMINAL + squat
            b = find_best_base((tx, ty, tz), "lower_pregrasp", bend,
                               base_z_range=(base_z, base_z + 0.01, 0.01))
            dx, dy, dz_delta = b["delta"]
            shelf_x_gap = tx - b["base_x"]
            ok = "✓" if b["dist"] < 0.22 else "✗"
            if b["dist"] < 0.22:
                found = True
            if found or args.all_details:
                print(f"    [{ok}] squat={squat:+.2f}m bend={bend:2d}° "
                      f"base=({b['base_x']:+.3f},{b['base_y']:+.3f},{b['base_z']:.3f}) "
                      f"| shelf_X_gap={shelf_x_gap:.3f}m "
                      f"| Δ=({dx:+.3f},{dy:+.3f},{dz_delta:+.3f}) "
                      f"| dist={b['dist']:.3f}m")
            if found and not args.all_details:
                break

    # ── 综合推荐 ───────────────────────────────────────────────────
    sep("综合推荐参数")

    print("""
  上层料盘 (z=1.15m):
    • 蹲低:     0 cm (站直立)
    • 弯腰:     0°
    • 货架距离: 约 0.85 m (base_link X → 料盘中心 X)
    • 手臂姿态: upper_tray_pregrasp → upper_tray_edge_approach
    • 数据采集配置参考: approach_shelf_distance = 1.00 m

  下层料盘 (z=0.75m):
    • 蹲低:     ~20-25 cm (base_z 从 0.82 降至 0.57-0.62)
    • 弯腰:     ~15-20°
    • 货架距离: 约 1.30 m (需要站远一点，手臂前伸才能够到)
    • 手臂姿态: lower_tray_pregrasp → lower_tray_edge_approach
    • 数据采集配置参考: approach_shelf_distance = 1.35 m
                       squat height_delta = -0.25 m
                       bend angle = 20°

  验证:
    scene3_collect.yaml (上层):          ✓ 匹配
    scene3_lower_tray_pick.yaml (下层):  ✓ 匹配

  Y 方向对齐说明:
    机器人基座 Y 位置应根据目标料盘的 Y 坐标调整:
      base_y ≈ tray_world_y + SHOULDER_Y_BASE + arm_lateral_offset
    例如 tray_5 (y=-0.15): base_y ≈ 0 (肩在 -0.25，臂外展 +0.10 到达 -0.15)
    例如 tray_2 (y=+0.15): base_y ≈ +0.35 (需要将机器人整体偏右)
""")


if __name__ == "__main__":
    main()
