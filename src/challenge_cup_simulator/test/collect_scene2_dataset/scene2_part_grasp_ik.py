#!/usr/bin/env python3
"""Scene2 抓取执行模块：包含目标构造、IK 服务调用与抓取动作。"""

import math
import os
from dataclasses import dataclass
import numpy as np


USE_CUSTOM_IK_PARAM = True
JOINT_ANGLES_AS_Q0 = True




@dataclass
class GraspRuntime:
    """抓取流程运行时参数与回调集合。

    主 pipeline 负责 ROS topic 发布、当前关节读取、sleep/log 等环境相关操作；
    本模块只负责计算目标位姿、调用 FK/IK 服务并组织抓取动作。
    """

    # 坐标偏移
    world_to_ee_offset_x: float
    world_to_ee_offset_y_left: float
    world_to_ee_offset_y_right: float
    world_to_ee_offset_z: float
    # 抓取参数
    pre_grasp_z_offset: float
    grasp_position_tolerance: float
    orientation_tolerance_rad: float
    gripper_close_time: float
    # IK 运动参数
    timeout: float
    move_time: float
    settle_time: float
    ik_mode_pos_hard_ori_hard: int
    # 运动/状态回调
    read_current_arm_joints_cb: callable
    execute_arm_motion_cb: callable
    publish_arm_gripper_close_cb: callable
    sleep_cb: callable
    loginfo_cb: callable
    logwarn_cb: callable

def euler_to_rotation_matrix(yaw_adaptive=0, pitch_adaptive=0, roll_adaptive=0,
                            yaw_manual=0, pitch_manual=0, roll_manual=0):
    """
    欧拉角(Z-Y-X顺序) → 旋转矩阵
    参数:
        yaw (float):   绕Z轴旋转角度（弧度）
        pitch (float): 绕Y轴旋转角度（弧度）
        roll (float):  绕X轴旋转角度（弧度）
    返回:
        np.ndarray: 3x3旋转矩阵
    """
    # 计算三角函数值
    cy, sy = np.cos(yaw_adaptive), np.sin(yaw_adaptive)
    cp, sp = np.cos(pitch_adaptive), np.sin(pitch_adaptive)
    
    R = np.array([
        [cy * cp,   -sy,        cy * sp],
        [sy * cp,    cy,        sy * sp],
        [-sp,        0,         cp     ]
    ])

    # 存在自定义参数 需要二次旋转
    if yaw_manual or pitch_manual or roll_manual:

        # 初始化为单位矩阵
        R_manual = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        if abs(yaw_manual) > 0.01:
            c, s = np.cos(yaw_manual), np.sin(yaw_manual)
            R_manual = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R_manual

        if abs(pitch_manual) > 0.01:
            c, s = np.cos(pitch_manual), np.sin(pitch_manual)
            R_manual = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ R_manual

        if abs(roll_manual) > 0.01:
            c, s = np.cos(roll_manual), np.sin(roll_manual)
            R_manual = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ R_manual

        return R @ R_manual
    # 不存在自定义参数,直接输出旋转矩阵
    else :
        return R

def rotation_matrix_to_quaternion(R):
    """旋转矩阵转四元数，输出 [x, y, z, w]。"""
    trace = np.trace(R)
    if trace > 0:
        w = math.sqrt(trace + 1.0) / 2.0
        x = (R[2, 1] - R[1, 2]) / (4.0 * w)
        y = (R[0, 2] - R[2, 0]) / (4.0 * w)
        z = (R[1, 0] - R[0, 1]) / (4.0 * w)
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j = (i + 1) % 3
        k = (j + 1) % 3
        t = np.zeros(4)
        t[i] = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) / 2.0
        if abs(t[i]) < 1e-9:
            return [0.0, 0.0, 0.0, 1.0]
        t[j] = (R[i, j] + R[j, i]) / (4.0 * t[i])
        t[k] = (R[i, k] + R[k, i]) / (4.0 * t[i])
        t[3] = (R[k, j] - R[j, k]) / (4.0 * t[i])
        x, y, z, w = t
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / norm, y / norm, z / norm, w / norm]

def euler_to_quaternion_via_matrix(yaw_adaptive=0, pitch_adaptive=0, roll_adaptive=0,
                                    yaw_manual=0, pitch_manual=0, roll_manual=0):
    """
    欧拉角 → 旋转矩阵 → 四元数
    参数:
        yaw (float):   绕Z轴旋转角度(弧度)
        pitch (float): 绕Y轴旋转角度(弧度)
        roll (float):  绕X轴旋转角度(弧度)
    返回:
        np.ndarray: 四元数 [x, y, z, w]
    """
    R = euler_to_rotation_matrix(yaw_adaptive, pitch_adaptive, roll_adaptive,
                                yaw_manual, pitch_manual, roll_manual)
    return rotation_matrix_to_quaternion(R)


# 工件抓取配置：world_xyz 是兜底值；实际运行优先读取 generated_layouts 里的 seed 布局。
# pose 配置字段顺序与 euler_to_quaternion_via_matrix() 一致；
# yaw_adaptive 是叠加在工件自身 Z 轴角度上的附加量。
OBJECT_PART_CONFIG = {
    "part_type_a_1": {
        "world_xyz": [-0.235, -0.33, 0.835],
        "use_opposite_arm": False,
        "flip_auto_narrow_edge_grasp": False,
        "grasp_offset_xyz_local": {
            "left": [
                {"y_range": [0.0, 0.1], "xyz": [0.02, -0.005, 0.05]},
                {"y_range": [0.1, 0.2], "xyz": [0.02, -0.005, 0.05]},
                {"y_range": [0.2, 0.3], "xyz": [0.02, -0.015, 0.05]},
                {"y_range": [0.3, 0.4], "xyz": [0.02, -0.005, 0.05]},
            ],
            "right": [
                {"y_range": [0.0, -0.1], "xyz": [0.02, -0.005, 0.05]},
                {"y_range": [-0.1, -0.2], "xyz": [0.02, -0.005, 0.05]},
                {"y_range": [-0.2, -0.3], "xyz": [0.012, -0.03, 0.04]},
                {"y_range": [-0.3, -0.4], "xyz": [0.015, -0.025, 0.05]},
            ],
        },
        "grasp_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 4, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
        "lift_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0, "follow_grasp_narrow_edge": True},
        "place_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
    },
    "part_type_a_2": {
        "world_xyz": [-0.24, -0.25, 0.803],
        "use_opposite_arm": False,
        "flip_auto_narrow_edge_grasp": False,
        "grasp_offset_xyz_local": {
            "left": [
                {"y_range": [0.0, 0.1], "xyz": [0.02, -0.02, 0.05]},
                {"y_range": [0.1, 0.2], "xyz": [0.02, 0.025, 0.05]},
                {"y_range": [0.2, 0.3], "xyz": [0.02, 0.015, 0.05]},
                {"y_range": [0.3, 0.4], "xyz": [0.02, 0.015, 0.05]},
            ],
            "right": [
                {"y_range": [0.0, -0.1], "xyz": [0.015, -0.020, 0.05]},
                {"y_range": [-0.1, -0.2], "xyz": [0.02, -0.005, 0.05]},
                {"y_range": [-0.2, -0.3], "xyz": [0.03, -0.005, 0.05]},
                {"y_range": [-0.3, -0.4], "xyz": [0.04, 0.015, 0.05]},
            ],
        },
        "grasp_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 4, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
        "lift_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0, "follow_grasp_narrow_edge": True},
        "place_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
    },
    "part_type_b_1": {
        "world_xyz": [-0.24, -0.06, 0.835],
        "use_opposite_arm": False,
        "flip_auto_narrow_edge_grasp": False,
        "grasp_offset_xyz_local": {
            "left": [
                {"y_range": [0.0, 0.1], "xyz": [0.02, 0.0, 0.05]},
                {"y_range": [0.1, 0.2], "xyz": [0.02, 0.01, 0.05]},
                {"y_range": [0.2, 0.3], "xyz": [0.03, 0.01, 0.05]},
                {"y_range": [0.3, 0.4], "xyz": [0.02, -0.02, 0.05]},
            ],
            "right": [
                {"y_range": [0.0, -0.1], "xyz": [0.03, 0.0, 0.05]},
                {"y_range": [-0.1, -0.2], "xyz": [0.02, 0.01, 0.05]},
                {"y_range": [-0.2, -0.3], "xyz": [0.03, 0.01, 0.05]},
                {"y_range": [-0.3, -0.4], "xyz": [0.02, 0.0, 0.05]},
            ],
        },
        "grasp_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 4, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
        "lift_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0, "follow_grasp_narrow_edge": True},
        "place_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
    },
    "part_type_b_2": {
        "world_xyz": [-0.24, 0.02, 0.835],
        "use_opposite_arm": False,
        "flip_auto_narrow_edge_grasp": False,
        "grasp_offset_xyz_local": {
            "left": [
                {"y_range": [0.0, 0.1], "xyz": [0.01, 0.0, 0.05]},
                {"y_range": [0.1, 0.2], "xyz": [0.02, 0.01, 0.05]},
                {"y_range": [0.2, 0.3], "xyz": [0.03, 0.01, 0.05]},
                {"y_range": [0.3, 0.4], "xyz": [0.03, 0.01, 0.05]},
            ],
            "right": [
                {"y_range": [0.0, -0.1], "xyz": [0.01, -0.01, 0.05]},
                {"y_range": [-0.1, -0.2], "xyz": [0.02, 0.01, 0.05]},
                {"y_range": [-0.2, -0.3], "xyz": [0.03, 0.01, 0.05]},
                {"y_range": [-0.3, -0.4], "xyz": [0.03, 0.01, 0.05]},
            ],
        },
        "grasp_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 4, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
        "lift_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0, "follow_grasp_narrow_edge": True},
        "place_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
    },
    "part_type_c_1": {
        "world_xyz": [-0.25, 0.19, 0.835],
        "use_opposite_arm": False,
        "flip_auto_narrow_edge_grasp": False,
        "grasp_offset_xyz_local": {
            "left": [
                {"y_range": [0.0, 0.1], "xyz": [0.03, -0.04, -0.02]},
                {"y_range": [0.1, 0.2], "xyz": [0.03, -0.04, -0.02]},
                {"y_range": [0.2, 0.3], "xyz": [0.03, -0.04, -0.02]},
                {"y_range": [0.3, 0.4], "xyz": [0.03, -0.04, -0.02]},
            ],
            "right": [
                {"y_range": [0.0, -0.1], "xyz": [0.01, -0.04, -0.03]},
                {"y_range": [-0.1, -0.2], "xyz": [0.02, -0.04, -0.01]},
                {"y_range": [-0.2, -0.3], "xyz": [0.04, -0.04, -0.01]},
                {"y_range": [-0.3, -0.4], "xyz": [0.03, -0.04, -0.01]},
            ],
        },
        "grasp_pose": {"yaw_adaptive": -math.pi / 4, "pitch_adaptive": -math.pi / 4, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
        "lift_pose": {"yaw_adaptive": -math.pi / 4, "pitch_adaptive": -math.pi / 4, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0, "follow_grasp_narrow_edge": True},
        "place_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
    },
    "part_type_c_2": {
        "world_xyz": [-0.25, 0.33, 0.835],
        "use_opposite_arm": False,
        "flip_auto_narrow_edge_grasp": False,
        "grasp_offset_xyz_local": {
            "left": [
                {"y_range": [0.0, 0.1], "xyz": [0.04, -0.04, -0.01]},
                {"y_range": [0.1, 0.2], "xyz": [0.03, -0.04, -0.01]},
                {"y_range": [0.2, 0.3], "xyz": [0.03, -0.04, -0.02]},
                {"y_range": [0.3, 0.4], "xyz": [0.04, -0.04, -0.02]},
            ],
            "right": [
                {"y_range": [0.0, -0.1], "xyz": [0.02, -0.04, 0.01]},
                {"y_range": [-0.1, -0.2], "xyz": [0.02, -0.04, -0.01]},
                {"y_range": [-0.2, -0.3], "xyz": [0.04, 0.025, -0.02]},
                {"y_range": [-0.3, -0.4], "xyz": [0.03, -0.05, -0.02]},
            ],
        },
        "grasp_pose": {"yaw_adaptive": -math.pi / 4, "pitch_adaptive": -math.pi / 4, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
        "lift_pose": {"yaw_adaptive": -math.pi / 4, "pitch_adaptive": -math.pi / 4, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0, "follow_grasp_narrow_edge": True},
        "place_pose": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": math.pi / 2, "pitch_manual": 0.0, "roll_manual": 0.0},
    },
    "handoff_left_to_right": {
        "place_world_xyz_by_part_type": {"part_type_a": [-0.30, 0.0, 0.885], "part_type_c": [-0.25, 0.0, 0.885]},
        "place_quat_xyzw_by_part_type": {"part_type_a": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": -3*math.pi / 4, "pitch_manual": 0.0, "roll_manual": -math.pi / 2}, "part_type_c": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": -2*math.pi / 3, "pitch_manual": 0.0, "roll_manual": -math.pi / 2}},
        "grasp_world_xyz_by_part_type": {"part_type_a": [-0.25, 0.0, 0.850], "part_type_c": [-0.25, 0.0, 0.850]},
        "grasp_quat_xyzw_by_part_type": {"part_type_a": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": 2*math.pi / 3, "pitch_manual": 0.0, "roll_manual": math.pi / 2}, "part_type_c": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": 2*math.pi / 3, "pitch_manual": 0.0, "roll_manual": math.pi / 2}},
    },
    "handoff_right_to_left": {
        "place_world_xyz_by_part_type": {"part_type_a": [-0.30, 0.0, 0.885], "part_type_c": [-0.26, 0.0, 0.895]},
        "place_quat_xyzw_by_part_type": {"part_type_a": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": 3*math.pi / 4, "pitch_manual": 0.0, "roll_manual": math.pi / 2}, "part_type_c": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": 3*math.pi / 4, "pitch_manual": 0.0, "roll_manual": math.pi / 2}},
        "grasp_world_xyz_by_part_type": {"part_type_a": [-0.25, 0.0, 0.850], "part_type_c": [-0.25, 0.0, 0.845]},
        "grasp_quat_xyzw_by_part_type": {"part_type_a": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": -2*math.pi / 3, "pitch_manual": 0.0, "roll_manual": -math.pi / 2}, "part_type_c": {"yaw_adaptive": 0.0, "pitch_adaptive": -math.pi / 2, "roll_adaptive": 0.0, "yaw_manual": -2*math.pi / 3, "pitch_manual": 0.0, "roll_manual": -math.pi / 2}},
    },
}

_LAYOUT_SEED = None
_LAYOUT_PATH = None
_LAYOUT_CACHE = None
_LAYOUT_CACHE_KEY = None
_LAYOUT_WARNED_KEYS = set()
GRASP_NARROW_EDGE_YAW_OFFSET = math.pi / 2


def configure_scene2_layout(seed=None, path=None):
    """设置本次 scene2 抓取要读取的 seed 布局文件。"""
    global _LAYOUT_SEED, _LAYOUT_PATH, _LAYOUT_CACHE, _LAYOUT_CACHE_KEY
    _LAYOUT_SEED = None if seed is None else int(seed)
    _LAYOUT_PATH = path
    _LAYOUT_CACHE = None
    _LAYOUT_CACHE_KEY = None


def _default_layout_path():
    if _LAYOUT_PATH:
        return _LAYOUT_PATH
    seed = _LAYOUT_SEED
    if seed is None:
        raw_seed = os.environ.get("SCENE2_LAYOUT_SEED")
        if raw_seed not in (None, ""):
            try:
                seed = int(raw_seed)
            except ValueError:
                seed = None
    if seed is None:
        return None
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "generated_layouts",
        f"scene2_seed_{seed}_parts.yaml",
    )


def _parse_float_list(text):
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"expected YAML inline list, got: {text}")
    body = text[1:-1].strip()
    if not body:
        return []
    return [float(item.strip()) for item in body.split(",")]


def _load_generated_layout():
    global _LAYOUT_CACHE, _LAYOUT_CACHE_KEY
    path = _default_layout_path()
    if not path:
        return {}

    cache_key = (path, os.path.getmtime(path) if os.path.exists(path) else None)
    if _LAYOUT_CACHE is not None and _LAYOUT_CACHE_KEY == cache_key:
        return _LAYOUT_CACHE

    if not os.path.isfile(path):
        _warn_once(path, f"scene2 layout yaml not found, fallback to static config: {path}")
        _LAYOUT_CACHE = {}
        _LAYOUT_CACHE_KEY = cache_key
        return _LAYOUT_CACHE

    parts = {}
    current_name = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
                current_name = stripped[:-1]
                parts[current_name] = {}
                continue
            if current_name is None or not line.startswith("    ") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            value = value.strip()
            if key in ("world_xyz", "quat_wxyz"):
                parts[current_name][key] = _parse_float_list(value)
            elif key in ("yaw_z_rad", "yaw_z_deg"):
                parts[current_name][key] = float(value)

    _LAYOUT_CACHE = parts
    _LAYOUT_CACHE_KEY = cache_key
    return _LAYOUT_CACHE


def _warn_once(key, message):
    if key in _LAYOUT_WARNED_KEYS:
        return
    _LAYOUT_WARNED_KEYS.add(key)
    print(f"[WARN] scene2_part_grasp_ik: {message}")


def _layout_for_object(object_name):
    return _load_generated_layout().get(object_name, {})


def _yaw_from_quat_wxyz(quat_wxyz):
    qw, qx, qy, qz = [float(v) for v in quat_wxyz]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm > 1e-9:
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _rotate_vector_by_quat_wxyz(vector, quat_wxyz):
    qw, qx, qy, qz = [float(v) for v in quat_wxyz]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm > 1e-9:
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    x, y, z = [float(v) for v in vector]
    # v' = v + 2*w*(q_vec x v) + 2*(q_vec x (q_vec x v))
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return [
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    ]


def _rotate_vector_by_yaw(vector, yaw_z):
    x, y, z = [float(v) for v in vector]
    c = math.cos(float(yaw_z))
    s = math.sin(float(yaw_z))
    return [c * x - s * y, s * x + c * y, z]


def _object_local_offset_to_world(object_name, offset_xyz_local):
    layout = _layout_for_object(object_name)
    if "quat_wxyz" in layout:
        return _rotate_vector_by_quat_wxyz(offset_xyz_local, layout["quat_wxyz"])
    return _rotate_vector_by_yaw(offset_xyz_local, get_object_yaw_z_rad(object_name))


def _base_object_world_xyz(object_name):
    layout = _layout_for_object(object_name)
    if "world_xyz" in layout:
        return list(layout["world_xyz"])
    return list(OBJECT_PART_CONFIG[object_name]["world_xyz"])


def _y_in_range(y_value, y_range):
    if not y_range or len(y_range) < 2:
        return False
    a, b = float(y_range[0]), float(y_range[1])
    lo, hi = min(a, b), max(a, b)
    return lo <= float(y_value) <= hi


def _select_offset_from_y_bins(offset_cfg, active_arm, object_y):
    if not isinstance(offset_cfg, dict):
        return offset_cfg
    bins = offset_cfg.get(active_arm) or []
    for item in bins:
        if _y_in_range(object_y, item.get("y_range")):
            return item.get("xyz", [0.0, 0.0, 0.0])

    other_arm = "right" if active_arm == "left" else "left"
    for item in offset_cfg.get(other_arm) or []:
        if _y_in_range(object_y, item.get("y_range")):
            return item.get("xyz", [0.0, 0.0, 0.0])

    if not bins:
        return [0.0, 0.0, 0.0]

    # y 超出配置区间时，选最近的区间，避免没有 offset 可用。
    return min(
        bins,
        key=lambda item: abs(
            float(object_y)
            - 0.5 * (float(item.get("y_range", [0.0, 0.0])[0]) + float(item.get("y_range", [0.0, 0.0])[1]))
        ),
    ).get("xyz", [0.0, 0.0, 0.0])


def _grasp_offset_local_for_side(object_name, active_arm):
    offset_cfg = OBJECT_PART_CONFIG[object_name].get("grasp_offset_xyz_local", [0.0, 0.0, 0.0])
    object_y = _base_object_world_xyz(object_name)[1]
    offset_local = _select_offset_from_y_bins(offset_cfg, active_arm, object_y)
    offset_local = [float(v) for v in offset_local]
    if len(offset_local) >= 2:
        pose_cfg = OBJECT_PART_CONFIG[object_name]["grasp_pose"]
        object_yaw_z = get_object_yaw_z_rad(object_name)
        narrow_offset = _narrow_edge_yaw_offset_for_object(
            object_name,
            pose_cfg,
            object_yaw_z,
            active_arm,
        )
        if narrow_offset < 0.0:
            offset_local[1] = -offset_local[1]
    return offset_local


def _apply_grasp_offset(object_name, base_world_xyz, active_arm):
    offset_local = _grasp_offset_local_for_side(object_name, active_arm)
    if not offset_local or all(abs(float(v)) < 1e-9 for v in offset_local):
        return list(base_world_xyz)
    offset_world = _object_local_offset_to_world(object_name, offset_local)
    return [float(base_world_xyz[i]) + offset_world[i] for i in range(3)]


def get_object_yaw_z_rad(object_name):
    layout = _layout_for_object(object_name)
    if "yaw_z_rad" in layout:
        return float(layout["yaw_z_rad"])
    if "quat_wxyz" in layout:
        return _yaw_from_quat_wxyz(layout["quat_wxyz"])
    return 0.0


def _manual_yaw_for_arm(pose_cfg, active_arm=None):
    yaw_manual = float(pose_cfg.get("yaw_manual", 0.0))
    if active_arm == "left":
        yaw_manual = -yaw_manual
    return yaw_manual


def _pose_matrix_from_config(pose_cfg, object_yaw_z, active_arm=None, adaptive_yaw_offset=0.0):
    yaw_manual = _manual_yaw_for_arm(pose_cfg, active_arm)
    return euler_to_rotation_matrix(
        yaw_adaptive=float(object_yaw_z)
        + float(pose_cfg.get("yaw_adaptive", pose_cfg.get("yaw_adaptive_bias", 0.0)))
        + float(adaptive_yaw_offset),
        pitch_adaptive=float(pose_cfg.get("pitch_adaptive", 0.0)),
        roll_adaptive=float(pose_cfg.get("roll_adaptive", 0.0)),
        yaw_manual=yaw_manual,
        pitch_manual=float(pose_cfg.get("pitch_manual", 0.0)),
        roll_manual=float(pose_cfg.get("roll_manual", 0.0)),
    )


def _tool_forward_x_score(pose_cfg, object_yaw_z, active_arm, adaptive_yaw_offset):
    matrix = _pose_matrix_from_config(pose_cfg, object_yaw_z, active_arm, adaptive_yaw_offset)
    # 夹爪初始工具方向为局部 -Z；用世界系 x 分量在两种窄边候选中选一头。
    forward_world = -(matrix @ np.array([0.0, 0.0, 1.0]))
    return float(forward_world[0])


def _select_narrow_edge_yaw_offset(pose_cfg, object_yaw_z, active_arm):
    candidates = (GRASP_NARROW_EDGE_YAW_OFFSET, -GRASP_NARROW_EDGE_YAW_OFFSET)
    return max(
        candidates,
        key=lambda offset: _tool_forward_x_score(pose_cfg, object_yaw_z, active_arm, offset),
    )


def _object_local_y_axis_world_xy(object_name, object_yaw_z):
    layout = _layout_for_object(object_name)
    if "quat_wxyz" in layout:
        axis = _rotate_vector_by_quat_wxyz([0.0, 1.0, 0.0], layout["quat_wxyz"])
        return float(axis[0]), float(axis[1])
    return -math.sin(float(object_yaw_z)), math.cos(float(object_yaw_z))


def _object_local_vector_world(object_name, object_yaw_z, vector):
    layout = _layout_for_object(object_name)
    if "quat_wxyz" in layout:
        return _rotate_vector_by_quat_wxyz(vector, layout["quat_wxyz"])
    return _rotate_vector_by_yaw(vector, object_yaw_z)


def _part_should_grasp_other_narrow_edge_by_axis(object_name, object_yaw_z, active_arm):
    """判断是否需要翻到窄边另一头抓取。

    A/B 类按工件局部 Y 轴在世界 XY 平面里的方向判断。
    C 类右手抓取是特殊规则：仅当 world Y 在 [-0.3, 0.0] 时，
    用工件局部 -X 与 +Z 之间的 45 度轴 `[-1, 0, 1]` 对齐世界 -Y
    的程度判断；夹角 20 度以内时翻到另一头抓。
    """

    object_y = float(_base_object_world_xyz(object_name)[1])
    if object_name.startswith("part_type_c") and active_arm == "right" and -0.3 <= object_y <= 0.0:
        axis = _object_local_vector_world(object_name, object_yaw_z, [-1.0, 0.0, 1.0])
        axis_x, axis_y, axis_z = [float(v) for v in axis]
        axis_norm = math.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
        if axis_norm <= 1e-9:
            return False
        cos_axis_angle = -axis_y / axis_norm
        return cos_axis_angle >= math.cos(math.radians(20.0))

    if object_name.startswith("part_type_a") and -0.2 <= object_y <= 0.1:
        target_x, target_y = 1.0, 1.0
        angle_threshold_deg = 45.0
    elif object_name.startswith("part_type_b") and 0.0 <= object_y <= 0.2:
        target_x, target_y = 1.0, -1.0
        angle_threshold_deg = 45.0
    elif object_name.startswith("part_type_b") and -0.2 <= object_y < 0.0:
        target_x, target_y = 1.0, 1.0
        angle_threshold_deg = 45.0
    else:
        return False

    axis_x, axis_y = _object_local_y_axis_world_xy(object_name, object_yaw_z)
    axis_norm = math.sqrt(axis_x * axis_x + axis_y * axis_y)
    target_norm = math.sqrt(target_x * target_x + target_y * target_y)
    if axis_norm <= 1e-9:
        return True

    cos_axis_angle = (axis_x * target_x + axis_y * target_y) / (axis_norm * target_norm)
    cos_axis_angle = abs(cos_axis_angle)
    is_aligned = cos_axis_angle >= math.cos(math.radians(angle_threshold_deg))
    return not is_aligned


def _narrow_edge_yaw_offset_for_object(object_name, pose_cfg, object_yaw_z, active_arm):
    auto_offset = _select_narrow_edge_yaw_offset(pose_cfg, object_yaw_z, active_arm)
    if _part_should_grasp_other_narrow_edge_by_axis(object_name, object_yaw_z, active_arm):
        auto_offset = -auto_offset
    if OBJECT_PART_CONFIG[object_name].get("flip_auto_narrow_edge_grasp", False):
        return -auto_offset
    return auto_offset


def _pose_config_to_quat_xyzw(pose_cfg, object_yaw_z, active_arm=None, adaptive_yaw_offset=0.0):
    yaw_manual = _manual_yaw_for_arm(pose_cfg, active_arm)
    return euler_to_quaternion_via_matrix(
        yaw_adaptive=float(object_yaw_z)
        + float(pose_cfg.get("yaw_adaptive", pose_cfg.get("yaw_adaptive_bias", 0.0)))
        + float(adaptive_yaw_offset),
        pitch_adaptive=float(pose_cfg.get("pitch_adaptive", 0.0)),
        roll_adaptive=float(pose_cfg.get("roll_adaptive", 0.0)),
        yaw_manual=yaw_manual,
        pitch_manual=float(pose_cfg.get("pitch_manual", 0.0)),
        roll_manual=float(pose_cfg.get("roll_manual", 0.0)),
    )


def _quat_config_to_xyzw(quat_cfg, active_arm=None):
    if quat_cfg is None:
        return None
    if isinstance(quat_cfg, dict):
        return _pose_config_to_quat_xyzw(
            quat_cfg,
            0.0,
            active_arm=active_arm,
        )
    return [float(v) for v in quat_cfg]


def _rad_to_deg(point):
    return [math.degrees(float(v)) for v in point]


def _axis_error(actual, desired):
    return math.sqrt(sum((float(actual[i]) - float(desired[i])) ** 2 for i in range(3)))


def _quat_angle_error(actual_xyzw, desired_xyzw):
    dot = sum(float(actual_xyzw[i]) * float(desired_xyzw[i]) for i in range(4))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


def _make_ik_param_like_example(constraint_mode, pos_cost_weight):
    # 参数与 example_ik_srv.py 保持一致，仅额外补充 constraint_mode。
    from kuavo_msgs.msg import ikSolveParam

    param = ikSolveParam()
    param.major_optimality_tol = 1e-3
    param.major_feasibility_tol = 1e-3
    param.minor_feasibility_tol = 1e-3
    param.major_iterations_limit = 100
    param.oritation_constraint_tol = 1e-3
    param.pos_constraint_tol = 1e-3
    param.pos_cost_weight = float(pos_cost_weight)
    param.constraint_mode = int(constraint_mode)
    return param


def _call_fk(joint_angles, timeout):
    import rospy
    from kuavo_msgs.srv import fkSrv

    rospy.wait_for_service("/ik/fk_srv", timeout=timeout)
    response = rospy.ServiceProxy("/ik/fk_srv", fkSrv)(list(joint_angles))
    if not response.success:
        raise RuntimeError("/ik/fk_srv returned success=false")
    return response.hand_poses


def _call_two_hands_ik(
    runtime: GraspRuntime,
    current_joint_values,
    left_pos,
    right_pos,
    left_quat,
    right_quat,
    constraint_mode,
    pos_cost_weight,
):
    import rospy
    from kuavo_msgs.msg import twoArmHandPoseCmd
    from kuavo_msgs.srv import twoArmHandPoseCmdSrv

    request = twoArmHandPoseCmd()
    request.ik_param = _make_ik_param_like_example(constraint_mode, pos_cost_weight)
    request.use_custom_ik_param = USE_CUSTOM_IK_PARAM
    request.joint_angles_as_q0 = JOINT_ANGLES_AS_Q0

    # 与 arm_keyboard_control.py 对齐：使用当前/锁定后的关节角作为 IK 初值 q0。
    request.hand_poses.left_pose.joint_angles = list(current_joint_values[:7])
    request.hand_poses.right_pose.joint_angles = list(current_joint_values[7:])
    request.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]

    request.hand_poses.left_pose.pos_xyz = list(left_pos)
    request.hand_poses.left_pose.quat_xyzw = list(left_quat)
    request.hand_poses.right_pose.pos_xyz = list(right_pos)
    request.hand_poses.right_pose.quat_xyzw = list(right_quat)
    rospy.loginfo(
        "scene2 ik request: left_pos=%s left_quat=%s right_pos=%s right_quat=%s",
        [round(float(v), 6) for v in request.hand_poses.left_pose.pos_xyz],
        [round(float(v), 6) for v in request.hand_poses.left_pose.quat_xyzw],
        [round(float(v), 6) for v in request.hand_poses.right_pose.pos_xyz],
        [round(float(v), 6) for v in request.hand_poses.right_pose.quat_xyzw],
    )

    rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=runtime.timeout)
    response = rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)(request)
    if not response.success:
        raise RuntimeError(
            "/ik/two_arm_hand_pose_cmd_srv failed: "
            + getattr(response, "error_reason", "")
            + f" left={list(left_pos)} right={list(right_pos)}"
        )
    q_arm = list(response.q_arm) if hasattr(response, "q_arm") else []
    left_result = list(response.hand_poses.left_pose.joint_angles)
    right_result = list(response.hand_poses.right_pose.joint_angles)
    q_arm_deg = [round(math.degrees(float(v)), 3) for v in q_arm] if q_arm else []
    left_result_deg = [round(math.degrees(float(v)), 3) for v in left_result]
    right_result_deg = [round(math.degrees(float(v)), 3) for v in right_result]
    rospy.loginfo(
        "scene2 ik result(deg): q_arm=%s left_joint_angles=%s right_joint_angles=%s",
        q_arm_deg,
        left_result_deg,
        right_result_deg,
    )

    if len(q_arm) >= 14:
        return q_arm[:14]
    if len(left_result) == 7 and len(right_result) == 7:
        return left_result + right_result
    raise RuntimeError("IK response did not contain arm joints")


def _call_single_arm_ik(
    runtime: GraspRuntime,
    current_joint_values,
    active_arm,
    active_pos,
    active_quat,
    locked_other_arm_joints,
    constraint_mode,
    pos_cost_weight,
):
    current_fk = _call_fk(current_joint_values, runtime.timeout)

    if active_arm == "left":
        right_lock = (
            [float(v) for v in locked_other_arm_joints]
            if locked_other_arm_joints is not None
            else list(current_joint_values[7:])
        )
        q0 = list(current_joint_values[:7]) + right_lock
        lock_fk = _call_fk(q0, runtime.timeout)
        left_quat = list(active_quat) if active_quat is not None else list(current_fk.left_pose.quat_xyzw)
        ik_full = _call_two_hands_ik(
            runtime=runtime,
            current_joint_values=q0,
            left_pos=list(active_pos),
            right_pos=list(lock_fk.right_pose.pos_xyz),
            left_quat=left_quat,
            right_quat=list(lock_fk.right_pose.quat_xyzw),
            constraint_mode=constraint_mode,
            pos_cost_weight=pos_cost_weight,
        )
        return list(ik_full[:7]) + right_lock

    if active_arm == "right":
        left_lock = (
            [float(v) for v in locked_other_arm_joints]
            if locked_other_arm_joints is not None
            else list(current_joint_values[:7])
        )
        q0 = left_lock + list(current_joint_values[7:])
        lock_fk = _call_fk(q0, runtime.timeout)
        right_quat = list(active_quat) if active_quat is not None else list(current_fk.right_pose.quat_xyzw)
        ik_full = _call_two_hands_ik(
            runtime=runtime,
            current_joint_values=q0,
            left_pos=list(lock_fk.left_pose.pos_xyz),
            right_pos=list(active_pos),
            left_quat=list(lock_fk.left_pose.quat_xyzw),
            right_quat=right_quat,
            constraint_mode=constraint_mode,
            pos_cost_weight=pos_cost_weight,
        )
        return left_lock + list(ik_full[7:])

    raise ValueError(f"unknown arm: {active_arm}")


def _measure_hand_pose(runtime: GraspRuntime, arm):
    q = runtime.read_current_arm_joints_cb()
    poses = _call_fk(q, runtime.timeout)
    pose = poses.left_pose if arm == "left" else poses.right_pose
    return list(pose.pos_xyz), list(pose.quat_xyzw)


def _move_arm_ik_once(
    runtime: GraspRuntime,
    active_arm,
    active_pos,
    locked_other_arm_joints,
    active_quat,
    label,
    constraint_mode,
    pos_cost_weight,
    move_time,
    settle_time,
):
    current = runtime.read_current_arm_joints_cb()
    ik_q = _call_single_arm_ik(
        runtime=runtime,
        current_joint_values=current,
        active_arm=active_arm,
        active_pos=active_pos,
        active_quat=active_quat,
        locked_other_arm_joints=locked_other_arm_joints,
        constraint_mode=constraint_mode,
        pos_cost_weight=pos_cost_weight,
    )
    cmd14 = _rad_to_deg(ik_q)
    start_q = list(current)
    if locked_other_arm_joints is not None:
        locked = [float(v) for v in locked_other_arm_joints]
        if active_arm == "left":
            start_q[7:14] = locked
        elif active_arm == "right":
            start_q[:7] = locked
    runtime.execute_arm_motion_cb(
        _rad_to_deg(start_q),
        cmd14,
        float(move_time),
        float(settle_time),
    )

    actual, actual_quat = _measure_hand_pose(runtime, active_arm)
    pos_err = _axis_error(actual, active_pos)
    quat_err = _quat_angle_error(actual_quat, active_quat) if active_quat is not None else None
    runtime.loginfo_cb(
        "scene2 grasp: %s %s-hand IK actual=%s pos_err=%.4f m quat_err=%s",
        label,
        active_arm,
        [round(v, 4) for v in actual],
        pos_err,
        "%.1fdeg" % math.degrees(quat_err) if quat_err is not None else "n/a",
    )
    return pos_err, quat_err, actual, actual_quat, cmd14


def _world_xyz_to_ee_xyz(runtime: GraspRuntime, world_xyz, arm):
    y_offset = (
        runtime.world_to_ee_offset_y_left if arm == "left" else runtime.world_to_ee_offset_y_right
    )
    return [
        float(world_xyz[0]) + runtime.world_to_ee_offset_x,
        float(world_xyz[1]) + y_offset,
        float(world_xyz[2]) + runtime.world_to_ee_offset_z,
    ]


def get_object_world_xyz(object_name, active_arm=None):
    if object_name not in OBJECT_PART_CONFIG:
        raise ValueError(f"unsupported object type: {object_name}")
    base_world_xyz = _base_object_world_xyz(object_name)
    if active_arm is None:
        active_arm = "left" if float(base_world_xyz[1]) >= 0.0 else "right"
    return _apply_grasp_offset(object_name, base_world_xyz, active_arm)


def get_object_grasp_quat_xyzw(object_name, active_arm=None):
    if object_name not in OBJECT_PART_CONFIG:
        raise ValueError(f"unsupported object type: {object_name}")
    if active_arm is None:
        active_arm = get_object_arm(object_name)
    object_yaw_z = get_object_yaw_z_rad(object_name)
    pose_cfg = OBJECT_PART_CONFIG[object_name]["grasp_pose"]
    adaptive_yaw_offset = _narrow_edge_yaw_offset_for_object(
        object_name,
        pose_cfg,
        object_yaw_z,
        active_arm,
    )
    return _pose_config_to_quat_xyzw(
        pose_cfg,
        object_yaw_z,
        active_arm=active_arm,
        adaptive_yaw_offset=adaptive_yaw_offset,
    )


def get_object_grasp_yaw_debug(object_name, active_arm=None):
    if object_name not in OBJECT_PART_CONFIG:
        raise ValueError(f"unsupported object type: {object_name}")
    if active_arm is None:
        active_arm = get_object_arm(object_name)
    object_yaw_z = get_object_yaw_z_rad(object_name)
    pose_cfg = OBJECT_PART_CONFIG[object_name]["grasp_pose"]
    adaptive_yaw_offset = _narrow_edge_yaw_offset_for_object(
        object_name,
        pose_cfg,
        object_yaw_z,
        active_arm,
    )
    return object_yaw_z, adaptive_yaw_offset


def get_object_lift_quat_xyzw(object_name, active_arm=None):
    if object_name not in OBJECT_PART_CONFIG:
        raise ValueError(f"unsupported object type: {object_name}")
    if active_arm is None:
        active_arm = get_object_arm(object_name)
    object_yaw_z = get_object_yaw_z_rad(object_name)
    pose_cfg = OBJECT_PART_CONFIG[object_name]["lift_pose"]
    adaptive_yaw_offset = 0.0
    if pose_cfg.get("follow_grasp_narrow_edge", False):
        grasp_pose_cfg = OBJECT_PART_CONFIG[object_name]["grasp_pose"]
        adaptive_yaw_offset = _narrow_edge_yaw_offset_for_object(
            object_name,
            grasp_pose_cfg,
            object_yaw_z,
            active_arm,
        )
    return _pose_config_to_quat_xyzw(
        pose_cfg,
        object_yaw_z,
        active_arm=active_arm,
        adaptive_yaw_offset=adaptive_yaw_offset,
    )


def get_object_place_quat_xyzw(object_name, active_arm=None):
    if object_name not in OBJECT_PART_CONFIG:
        raise ValueError(f"unsupported object type: {object_name}")
    if active_arm is None:
        active_arm = get_object_arm(object_name)
    object_yaw_z = get_object_yaw_z_rad(object_name)
    return _pose_config_to_quat_xyzw(
        OBJECT_PART_CONFIG[object_name]["place_pose"],
        object_yaw_z,
        active_arm=active_arm,
    )


def get_handoff_transition_config(handoff_name, active_arm=None):
    if handoff_name not in OBJECT_PART_CONFIG:
        raise ValueError(f"unsupported handoff transition: {handoff_name}")
    cfg = dict(OBJECT_PART_CONFIG[handoff_name])
    # Handoff poses are manually tuned per arm; use the Euler values literally
    # instead of applying the normal left-arm yaw mirroring.
    cfg["place_quat_xyzw_by_part_type"] = {
        part_type: _quat_config_to_xyzw(quat_cfg, active_arm=None)
        for part_type, quat_cfg in cfg.get("place_quat_xyzw_by_part_type", {}).items()
    }
    cfg["grasp_quat_xyzw_by_part_type"] = {
        part_type: _quat_config_to_xyzw(quat_cfg, active_arm=None)
        for part_type, quat_cfg in cfg.get("grasp_quat_xyzw_by_part_type", {}).items()
    }
    return cfg


def get_object_arm(object_name):
    if object_name not in OBJECT_PART_CONFIG:
        raise ValueError(f"unsupported object type: {object_name}")
    world_xyz = _base_object_world_xyz(object_name)
    object_y = float(world_xyz[1])
    if object_name.startswith("part_type_a") and -0.2 <= object_y <= 0.1:
        auto_arm = "right"
    elif object_name.startswith("part_type_c") and -0.2 <= object_y <= 0.0:
        auto_arm = "left"
    else:
        auto_arm = "left" if object_y >= 0.0 else "right"
    if OBJECT_PART_CONFIG[object_name].get("use_opposite_arm", False):
        return "right" if auto_arm == "left" else "left"
    return auto_arm


def _run_grasp_sequence(
    runtime: GraspRuntime,
    object_name: str,
    world_xyz,
    active_arm: str,
    locked_other_arm_joints,
    pre_grasp_z_offset: float,
    grasp_quat_xyzw=None,
):
    grasp_target = _world_xyz_to_ee_xyz(runtime, world_xyz, active_arm)
    pre_grasp_target = [grasp_target[0], grasp_target[1], grasp_target[2] + float(pre_grasp_z_offset)]
    _ = active_arm
    grasp_quat = (
        list(grasp_quat_xyzw)
        if grasp_quat_xyzw is not None
        else get_object_grasp_quat_xyzw(object_name, active_arm=active_arm)
    )
    object_yaw_z, narrow_edge_offset = get_object_grasp_yaw_debug(
        object_name,
        active_arm=active_arm,
    )

    runtime.loginfo_cb(
        "scene2 grasp: %s %s-hand yaw=%.1fdeg narrow_offset=%+.1fdeg pre-grasp=%s",
        object_name,
        active_arm,
        math.degrees(object_yaw_z),
        math.degrees(narrow_edge_offset),
        [round(v, 4) for v in pre_grasp_target],
    )
    _move_arm_ik_once(
        runtime=runtime,
        active_arm=active_arm,
        active_pos=pre_grasp_target,
        locked_other_arm_joints=locked_other_arm_joints,
        active_quat=grasp_quat,
        label=f"{object_name}_pre_grasp",
        constraint_mode=runtime.ik_mode_pos_hard_ori_hard,
        pos_cost_weight=1.0,
        move_time=runtime.move_time,
        settle_time=runtime.settle_time,
    )
    runtime.sleep_cb(0.5)

    runtime.loginfo_cb(
        "scene2 grasp: %s %s-hand descend=%s",
        object_name,
        active_arm,
        [round(v, 4) for v in grasp_target],
    )
    pos_err, quat_err, _actual, _actual_quat, _cmd14 = _move_arm_ik_once(
        runtime=runtime,
        active_arm=active_arm,
        active_pos=grasp_target,
        locked_other_arm_joints=locked_other_arm_joints,
        active_quat=grasp_quat,
        label=f"{object_name}_grasp",
        constraint_mode=runtime.ik_mode_pos_hard_ori_hard,
        pos_cost_weight=1.0,
        move_time=runtime.move_time,
        settle_time=runtime.settle_time,
    )
    success = bool(
        pos_err <= runtime.grasp_position_tolerance and quat_err <= runtime.orientation_tolerance_rad
    )
    if not success:
        runtime.logwarn_cb(
            "scene2 grasp: %s tolerance miss xyz_err=%.4f/%.4f quat_err=%.1fdeg/%.1fdeg",
            object_name,
            pos_err,
            runtime.grasp_position_tolerance,
            math.degrees(quat_err),
        )

    runtime.publish_arm_gripper_close_cb(active_arm)
    runtime.sleep_cb(runtime.gripper_close_time)
    return success


def grasp_part_type_a(runtime: GraspRuntime, object_name, world_xyz, active_arm, locked_other_arm_joints, grasp_quat_xyzw=None):
    """A 类工件抓取：上方 -> 下降 -> 夹紧。"""
    return _run_grasp_sequence(
        runtime,
        object_name,
        world_xyz,
        active_arm,
        locked_other_arm_joints,
        pre_grasp_z_offset=runtime.pre_grasp_z_offset,
        grasp_quat_xyzw=grasp_quat_xyzw,
    )


def grasp_part_type_b(runtime: GraspRuntime, object_name, world_xyz, active_arm, locked_other_arm_joints, grasp_quat_xyzw=None):
    """B 类工件抓取：上方 -> 下降 -> 夹紧。"""
    return _run_grasp_sequence(
        runtime,
        object_name,
        world_xyz,
        active_arm,
        locked_other_arm_joints,
        pre_grasp_z_offset=runtime.pre_grasp_z_offset,
        grasp_quat_xyzw=grasp_quat_xyzw,
    )


def grasp_part_type_c(runtime: GraspRuntime, object_name, world_xyz, active_arm, locked_other_arm_joints, grasp_quat_xyzw=None):
    """C 类工件抓取：上方 -> 下降 -> 夹紧。"""
    return _run_grasp_sequence(
        runtime,
        object_name,
        world_xyz,
        active_arm,
        locked_other_arm_joints,
        pre_grasp_z_offset=runtime.pre_grasp_z_offset,
        grasp_quat_xyzw=grasp_quat_xyzw,
    )


def execute_part_grasp(runtime: GraspRuntime, object_name, world_xyz, active_arm, locked_other_arm_joints, grasp_quat_xyzw=None):
    """输入工件名和位置，执行对应抓取函数并返回是否成功。"""
    if object_name.startswith("part_type_a"):
        return grasp_part_type_a(runtime, object_name, world_xyz, active_arm, locked_other_arm_joints, grasp_quat_xyzw=grasp_quat_xyzw)
    if object_name.startswith("part_type_b"):
        return grasp_part_type_b(runtime, object_name, world_xyz, active_arm, locked_other_arm_joints, grasp_quat_xyzw=grasp_quat_xyzw)
    if object_name.startswith("part_type_c"):
        return grasp_part_type_c(runtime, object_name, world_xyz, active_arm, locked_other_arm_joints, grasp_quat_xyzw=grasp_quat_xyzw)
    raise ValueError(f"unsupported object type for grasp: {object_name}")


def rad_to_deg(point):
    """公开角度转换，供主流程复用。"""
    return _rad_to_deg(point)


def measure_hand_pose(runtime: GraspRuntime, arm):
    """公开末端位姿测量，供主流程复用。"""
    return _measure_hand_pose(runtime, arm)


def move_arm_ik_once(
    runtime: GraspRuntime,
    active_arm,
    active_pos,
    locked_other_arm_joints,
    active_quat,
    label,
    constraint_mode,
    pos_cost_weight,
    move_time,
    settle_time,
):
    """公开单次 IK 执行，供主流程复用。"""
    return _move_arm_ik_once(
        runtime=runtime,
        active_arm=active_arm,
        active_pos=active_pos,
        locked_other_arm_joints=locked_other_arm_joints,
        active_quat=active_quat,
        label=label,
        constraint_mode=constraint_mode,
        pos_cost_weight=pos_cost_weight,
        move_time=move_time,
        settle_time=settle_time,
    )
