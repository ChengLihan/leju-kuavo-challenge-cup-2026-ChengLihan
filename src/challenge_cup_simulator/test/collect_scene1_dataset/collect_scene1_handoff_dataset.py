#!/usr/bin/env python3
"""采集场景1数据：右手称重，左手放箱。

本脚本为**单文件独立**：不依赖 collect_scene2_dataset。通用的 ROS/IK/手臂运动原语
已内联在文件顶部「内联通用原语」分区，场景1专属的启动、布局、抓取、递手和放置逻辑在其后。
"""

import argparse
import datetime as _datetime
import math
import os
import signal
import subprocess
import sys
import threading
import time

# =============================================================================
# 内联通用原语（原 collect_scene2_dataset 的基础设施，内联以使本脚本单文件独立运行）
# 只含 scene1 实际用到的常量/函数/类；scene1 已自有的（GripperCommandHold、
# _start_scene_launch、_start_rosbag、_now_tag 等）不在此重复。
# =============================================================================

# --- ROS / 启动 / 录制相关常量 ---
TOPIC_TIMEOUT = 20.0
LAUNCH_TIMEOUT = 120.0
RECORD_SETTLE_TIME = 2.0
POST_ARM_MODE_RECORD_TIME = 1.0
GRIPPER_COMMAND_HZ = 100.0

# --- 手臂控制相关常量 ---
ARM_JOINT_NAMES = ["arm_joint_" + str(i) for i in range(1, 15)]
ARM_MODE_EXTERNAL_CONTROL = 2
ARM_MODE_AUTO_SWING = 1
ARM_MODE_SERVICE = "/arm_traj_change_mode"
ARM_TARGET_POSES_TOPIC = "/kuavo_arm_target_poses"
ARM_TRAJ_TOPIC = "/kuavo_arm_traj"
ARM_TRAJ_HZ = 100.0
ARM_MOVE_TIME = 2.0
ARM_SETTLE_TIME = 0.3

# --- 头部 / IK 模式常量 ---
HEAD_TARGET = [0.0, 20.0]
HEAD_SETTLE_TIME = 0.8
IK_MODE_POS_HARD_ORI_SOFT = 0x02
IK_MODE_POS_HARD_ORI_HARD = 0x03
IK_MODE_THREE_POINT_MIXED = 0x06


def _load_challenge_sim_launcher():
    sim_utils = _find_package_subdir("utils")
    if sim_utils not in sys.path:
        sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher

    return ChallengeSimLauncher


def _init_ros_node(node_name):
    import rospy

    if not rospy.core.is_initialized():
        rospy.init_node(node_name, anonymous=True)


def _terminate_process_group(proc, sig, timeout):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), sig)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(
                f"[WARN] roslaunch process group {proc.pid} did not exit after SIGKILL",
                file=sys.stderr,
            )
    except ProcessLookupError:
        return


def _wait_for_topics(topics, timeout):
    import rospy

    required = list(dict.fromkeys(topics))
    start = time.time()
    while time.time() - start < timeout and not rospy.is_shutdown():
        published = {name for name, _ in rospy.get_published_topics()}
        missing = [topic for topic in required if topic not in published]
        if not missing:
            return []
        time.sleep(0.5)
    return missing


def _wait_for_connection(pub, timeout):
    import rospy

    start = time.time()
    while pub.get_num_connections() == 0 and time.time() - start < timeout and not rospy.is_shutdown():
        rospy.sleep(0.2)
    if pub.get_num_connections() == 0:
        raise RuntimeError(f"topic {pub.name} has no subscriber")


def _set_arm_mode(mode, timeout):
    import rospy
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

    rospy.wait_for_service(ARM_MODE_SERVICE, timeout=timeout)
    proxy = rospy.ServiceProxy(ARM_MODE_SERVICE, changeArmCtrlMode)
    request = changeArmCtrlModeRequest()
    request.control_mode = mode
    response = proxy(request)
    if not response.result:
        raise RuntimeError(f"{ARM_MODE_SERVICE} rejected mode {mode}: {response.message}")
    rospy.loginfo("scene1 handoff: arm mode -> %s: %s", mode, response.message)


def _publish_head_target(timeout):
    import rospy
    from kuavo_msgs.msg import robotHeadMotionData

    pub = rospy.Publisher("/robot_head_motion_data", robotHeadMotionData, queue_size=10)
    _wait_for_connection(pub, timeout)

    msg = robotHeadMotionData()
    msg.joint_data = list(HEAD_TARGET)
    for _ in range(5):
        if rospy.is_shutdown():
            break
        pub.publish(msg)
        rospy.sleep(0.1)
    rospy.sleep(HEAD_SETTLE_TIME)


class ArmTrajHold:
    def __init__(self, pub, degrees_list, hz=ARM_TRAJ_HZ):
        self._pub = pub
        self._hz = float(hz)
        self._degrees = self._validate_degrees(degrees_list)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="arm_traj_hold", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_degrees(self, degrees_list):
        degrees = self._validate_degrees(degrees_list)
        with self._lock:
            self._degrees = degrees

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState

        rate = rospy.Rate(self._hz)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            with self._lock:
                degrees = list(self._degrees)
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ARM_JOINT_NAMES
            msg.position = degrees
            self._pub.publish(msg)
            rate.sleep()

    @staticmethod
    def _validate_degrees(degrees_list):
        degrees = [float(v) for v in degrees_list]
        if len(degrees) != len(ARM_JOINT_NAMES):
            raise ValueError(f"arm traj command has {len(degrees)} joints, expected {len(ARM_JOINT_NAMES)}")
        return degrees


def _start_arm_traj_hold(timeout):
    import rospy
    from sensor_msgs.msg import JointState

    pub = rospy.Publisher(ARM_TRAJ_TOPIC, JointState, queue_size=10)
    _wait_for_connection(pub, timeout)
    initial_degrees = _rad_to_deg(_read_current_arm_joints(timeout))
    hold = ArmTrajHold(pub, initial_degrees)
    hold.start()
    return hold


def _publish_arm_target_poses(target_pub, degrees_list, move_time):
    from kuavo_msgs.msg import armTargetPoses

    msg = armTargetPoses()
    msg.times = [float(move_time)]
    msg.values = [float(v) for v in degrees_list]
    target_pub.publish(msg)


def _publish_arm_traj_interpolation(arm_hold, start_degrees, target_degrees, duration):
    import rospy

    start = [float(v) for v in start_degrees]
    target = [float(v) for v in target_degrees]
    if len(start) != len(target):
        raise ValueError(f"arm traj interpolation length mismatch: {len(start)} != {len(target)}")

    steps = max(1, int(round(float(duration) * ARM_TRAJ_HZ)))
    rate = rospy.Rate(ARM_TRAJ_HZ)
    for step in range(steps + 1):
        if rospy.is_shutdown():
            break
        alpha = float(step) / float(steps)
        point = [start[i] + (target[i] - start[i]) * alpha for i in range(len(target))]
        arm_hold.set_degrees(point)
        if step < steps:
            rate.sleep()


def _execute_arm_motion(target_pub, arm_hold, start_degrees, target_degrees, move_time, settle):
    import rospy

    _publish_arm_target_poses(target_pub, target_degrees, move_time)
    _publish_arm_traj_interpolation(arm_hold, start_degrees, target_degrees, move_time)
    rospy.sleep(settle)


def _move_arm_to(target_pub, arm_hold, degrees_list, move_time=ARM_MOVE_TIME, settle=ARM_SETTLE_TIME):
    start = _rad_to_deg(_read_current_arm_joints(TOPIC_TIMEOUT))
    _execute_arm_motion(target_pub, arm_hold, start, degrees_list, move_time, settle)


def _axis_error(actual, desired, axes):
    return math.sqrt(
        sum((actual[i] - desired[i]) ** 2 for i, enabled in enumerate(axes) if enabled)
    )


def _quat_angle_error(actual_xyzw, desired_xyzw):
    dot = sum(float(actual_xyzw[i]) * float(desired_xyzw[i]) for i in range(4))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


def _rad_to_deg(point):
    return [math.degrees(v) for v in point]


def _read_current_arm_joints(timeout):
    import rospy
    from kuavo_msgs.msg import sensorsData

    msg = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=timeout)
    joint_q = list(msg.joint_data.joint_q)
    if len(joint_q) >= 27:
        return joint_q[13:27]
    if len(joint_q) >= 26:
        return joint_q[12:26]
    raise RuntimeError(f"/sensors_data_raw joint_q has {len(joint_q)} values")


def _make_ik_param(constraint_mode=IK_MODE_POS_HARD_ORI_SOFT, pos_cost_weight=0.0):
    from kuavo_msgs.msg import ikSolveParam

    param = ikSolveParam()
    param.major_optimality_tol = 1e-3
    param.major_feasibility_tol = 1e-3
    param.minor_feasibility_tol = 1e-3
    param.major_iterations_limit = 500
    param.oritation_constraint_tol = 1e-3
    param.pos_constraint_tol = 1e-3
    param.pos_cost_weight = float(pos_cost_weight)
    param.constraint_mode = constraint_mode
    return param


def _call_fk(joint_angles, timeout):
    import rospy
    from kuavo_msgs.srv import fkSrv

    rospy.wait_for_service("/ik/fk_srv", timeout=timeout)
    response = rospy.ServiceProxy("/ik/fk_srv", fkSrv)(joint_angles)
    if not response.success:
        raise RuntimeError("/ik/fk_srv returned success=false")
    return response.hand_poses
# =============================================================================
# 内联通用原语结束
# =============================================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_package_subdir(name):
    """从脚本目录逐级向上找包内的 name 子目录(utils / lib)，返回第一个存在的。
    脚本可能位于 <pkg>/test/ 或 <pkg>/test/collect_scene1_dataset/ 等不同深度，
    向上搜索避免硬编码相对层级(挪动目录后也能找到正确位置)。"""
    here = SCRIPT_DIR
    for _ in range(6):
        cand = os.path.join(here, name)
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return os.path.join(SCRIPT_DIR, name)


SCENE_NAME = "scene1"
DEFAULT_OUTPUT_DIR = "/root/kuavo_ws/bags/scene1_handoff"
DEFAULT_FAILED_SEEDS_FILE = "failed_seeds.txt"
DEFAULT_SUCCESS_MANIFEST_FILE = "success_manifest.txt"
DEFAULT_MAX_SEED_ATTEMPTS = 10
MUJOCO_QPOS_TOPIC = "/mujoco/qpos"

# Bag 采集话题。位置检查会在采集节点内临时读取 /mujoco/qpos，但不会把真值 topic 写进 bag。
DEFAULT_TOPICS = [
    "/gripper/command",
    "/gripper/state",
    "/sensors_data_raw",
    "/joint_cmd",
    "/kuavo_arm_traj",
    "/cam_h/color/image_raw/compressed",
    "/cam_l/color/image_raw/compressed",
    "/cam_r/color/image_raw/compressed",
    "/cam_h/depth/image_raw/compressedDepth",
    "/cam_r/depth/image_rect_raw/compressedDepth",
    "/cam_l/depth/image_rect_raw/compressedDepth",
]


def _matmul3(a, b):
    return [
        [
            a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def _rotation_matrix_to_quat_xyzw(matrix):
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        w = math.sqrt(trace + 1.0) / 2.0
        x = (matrix[2][1] - matrix[1][2]) / (4.0 * w)
        y = (matrix[0][2] - matrix[2][0]) / (4.0 * w)
        z = (matrix[1][0] - matrix[0][1]) / (4.0 * w)
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        x = math.sqrt(max(0.0, 1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])) / 2.0
        y = (matrix[0][1] + matrix[1][0]) / (4.0 * x)
        z = (matrix[0][2] + matrix[2][0]) / (4.0 * x)
        w = (matrix[2][1] - matrix[1][2]) / (4.0 * x)
    elif matrix[1][1] > matrix[2][2]:
        y = math.sqrt(max(0.0, 1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])) / 2.0
        x = (matrix[0][1] + matrix[1][0]) / (4.0 * y)
        z = (matrix[1][2] + matrix[2][1]) / (4.0 * y)
        w = (matrix[0][2] - matrix[2][0]) / (4.0 * y)
    else:
        z = math.sqrt(max(0.0, 1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])) / 2.0
        x = (matrix[0][2] + matrix[2][0]) / (4.0 * z)
        y = (matrix[1][2] + matrix[2][1]) / (4.0 * z)
        w = (matrix[1][0] - matrix[0][1]) / (4.0 * z)

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / norm, y / norm, z / norm, w / norm]


def _quat_from_ypr_deg(first_ypr_deg, second_ypr_deg=None):
    yaw, pitch, _roll = [math.radians(float(v)) for v in first_ypr_deg]
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    matrix = [
        [cy * cp, -sy, cy * sp],
        [sy * cp, cy, sy * sp],
        [-sp, 0.0, cp],
    ]

    if second_ypr_deg is not None:
        manual_yaw, manual_pitch, manual_roll = [math.radians(float(v)) for v in second_ypr_deg]
        manual = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        if abs(manual_yaw) > 0.01:
            c, s = math.cos(manual_yaw), math.sin(manual_yaw)
            manual = _matmul3([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], manual)
        if abs(manual_pitch) > 0.01:
            c, s = math.cos(manual_pitch), math.sin(manual_pitch)
            manual = _matmul3([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], manual)
        if abs(manual_roll) > 0.01:
            c, s = math.cos(manual_roll), math.sin(manual_roll)
            manual = _matmul3([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], manual)
        matrix = _matmul3(matrix, manual)

    return _rotation_matrix_to_quat_xyzw(matrix)


# ===== 坐标与场景基础参数 =====
# world -> IK/base 坐标偏移。由场景2固定抓取脚本反推得到：
# fixed_grasp_ik - challenge_secret.get_object_layout("scene2", 0)["part_type_b_*"]["pos"]。
WORLD_TO_IK_OFFSET = [0.565966, -0.013886, -0.923783]
# 随机数种子只影响下面这些包裹的初始位置；称重区和箱子点位都是固定场景点位。
PARCEL_NAMES = ["parcel_1", "parcel_2", "parcel_3", "parcel_4"]
PARCEL_CENTER_ON_TABLE_Z = 0.95

# ===== 1. 右手到桌面抓快递 =====
# 最终抓取高度基准,绝对 IK z(其余微调叠加在此)。
RIGHT_PICK_IK_Z = -0.03
# 抓取偏移按包裹实际 Y 分区(借鉴 scene2 的 y-bin):右臂在"近身侧"(y 较大/靠机器人)够不深、
# 窄边对中会漂,需 +y 对正并压深;"远身侧"够得深用基准。基于实时读到的真实 y 选区 → 对任意 seed 通用。
# offset = [沿长边 x 微调, 沿窄边 y 微调(对正夹取窄边), z 深度微调]。
RIGHT_PICK_OFFSET_FAR_ROW = [-0.03, 0.0, 0.0]       # y <= 阈值(远身)
RIGHT_PICK_OFFSET_NEAR_ROW = [-0.03, 0.02, -0.02]   # y > 阈值(近身,实测 z重叠 0.036 咬实)
RIGHT_PICK_NEAR_FAR_Y_THRESHOLD = -0.20
# 仍可给特定包裹名单独覆盖(优先级最高);默认按上面的 Y 分区。
RIGHT_PICK_OFFSET_BY_PARCEL = {
    # parcel_4(近 y + 远 x 角落,最难够)基线 -0.03 会抓到长边端头(x重叠低至 0.008)。
    # 受可达性限制不能大幅 +x(+0.02 会 0x03 IK 失败),给温和 -0.01 把抓点往长边中段拉一点。
    "parcel_4": [-0.01, 0.02, -0.02],
}
RIGHT_PICK_FIRST_YPR_DEG = [0, -90, 0.0]
RIGHT_PICK_SECOND_YPR_DEG = [90.0, 0.0, 0.0]
# 从当前/预设位置去第一次抓取点时的横移高度，绝对 IK z。
# 右手先在这个高度对齐抓取点 x/y 和四元素，最后才下降到 right_pick_ik[2] 真正夹取。
RIGHT_PICK_TRANSIT_IK_Z = 0.120
# 夹住包裹后，从最终抓取点向上抬起的相对高度。
LIFT_Z_OFFSET = 0.300
# 第一段对齐 y/z 时，当前右手 x 可能太靠近身体，带着抓取姿态会进入 IK 边界。
# 这个安全 x 只用于非接触过渡点；后续仍会再走到真实抓取上方和抓取高度。
RIGHT_PICK_YZ_ALIGN_SAFE_IK_X = 0.184
# 抓取/抬升阶段也统一使用 0x06 三点混合，避免不同阶段 IK 模式切换带来不可重复误差。
RIGHT_PICK_IK_CONSTRAINT_MODE = IK_MODE_THREE_POINT_MIXED
RIGHT_PICK_POS_COST_WEIGHT = 0.0
RIGHT_PICK_IK_MAJOR_ITERATIONS_LIMIT = 500
# 最终抓取那一步：先到预抓取位暖启动后，右手单手用 0x03(位姿都硬)精确对正夹爪，解不出再回退 0x06。
# 左手此步不参与（保持当前 preset_2 关节），避免左手不可达朝向连累右手硬解。
USE_HARD_FINAL_GRASP = True
RIGHT_GRASP_FINAL_CONSTRAINT_MODE = IK_MODE_POS_HARD_ORI_HARD
# ===== 2. 右手把快递放到称重区，再二次夹起 =====
# 称重区中心 world（对齐 scene1.xml 实测的称重区中心 [-0.17,-0.56]，原 [-0.10,-0.56] 偏到了区域边缘）。
WEIGHING_CENTER_WORLD = [-0.17, -0.560, PARCEL_CENTER_ON_TABLE_Z]

# 2.1 平移：夹起并抬高后，直接高位横移到称重释放点上方。
# WEIGH_TRANSIT_IK_Z 是释放前高位；下一步只下降 z 到 2.2 释放点。
WEIGH_TRANSIT_IK_Z = 0.326217

# 2.2 释放：到称重区上方后，下降到这个 IK 点并打开右手夹爪。
# x/y 已做落点补偿——实测包裹脱手后会沿 -x 方向滑一小段，故释放点取得比期望落点更靠 +x，
# 让包裹最终停在称重区内；y 落点基本对正。具体落点以“变黄判定”日志为准。
WEIGH_RELEASE_IK = [0.396, -0.574, 0.146217]
RIGHT_WEIGH_RELEASE_FIRST_YPR_DEG = [0, -100, 0.0]
RIGHT_WEIGH_RELEASE_SECOND_YPR_DEG = [90.0, 0.0, 0.0]
# 到达释放点后先等一段时间，让手臂和快递稳定，再打开右手夹爪。
# 这个等待发生在松开夹爪之前，和下面 WEIGH_DWELL 的“松开后称重等待”不是一回事。
WEIGH_RELEASE_SETTLE_BEFORE_OPEN = 1.5
# 2.1 上方中间点只用于拆路径，不需要每到一个中间点都停顿。
WEIGH_TRANSIT_SETTLE_TIME = 0.0
# 右手松开后，z 只要求包裹落回桌面/称重台附近高度，避免把“还挂在夹爪上”误判为称重成功。
# （xy 是否落入称重区由 _assert_parcel_on_weighing_area 用 geom 角点判定，见下方“变黄判定”。）
WEIGH_CHECK_Z_RANGE = [0.80, 1.12]
# 称重区/包裹角点判定（与仿真 mujoco_node.cc 变黄逻辑同源）：geom 名 + 角点容差。
WEIGHING_AREA_GEOM_NAME = "weighing_area_0p2m_square"
WEIGH_FOOTPRINT_TOLERANCE = 0.002  # 与 C++ checkParcelFootprint kTolerance 一致

# 2.3 二次抓取：称重等待后，右手沿用 2.2 释放高度对齐这个点的 x/y 和姿态，
# 再只下降 z 到这个 IK 点重新夹起快递；这里不再抬到 WEIGH_TRANSIT_IK_Z。
WEIGH_REGRASP_IK = [0.396, -0.574, -0.04]
RIGHT_WEIGH_REGRASP_FIRST_YPR_DEG = [0, -60, 0.0]
RIGHT_WEIGH_REGRASP_SECOND_YPR_DEG = [90.0, 0.0, 0.0]
WEIGH_DWELL = 1.0

# ===== 3. 左手先到等待位，右手把快递递给左手 =====
# 姿态参数完全对齐 arm_control.py：第一套/第二套都是 yaw/pitch/roll，单位 degree。

# 3.2 左手第二等待点：称重完成前停在这里，准备去接右手快递。
LEFT_PRESET_2_IK = [0.313, 0.239, 0.282]
LEFT_PRESET_2_FIRST_YPR_DEG = [-146.440, 4.966, 0.0]
LEFT_PRESET_2_SECOND_YPR_DEG = [0.0, 0.0, 96.580]
LEFT_PRESET_2_QUAT_XYZW = _quat_from_ypr_deg(
    LEFT_PRESET_2_FIRST_YPR_DEG,
    LEFT_PRESET_2_SECOND_YPR_DEG,
)

# 3.3 右手交接点：右手先到固定高位对齐 x/y 和四元素，再下降到交接点等待左手靠近。
RIGHT_HANDOFF_TO_LEFT_IK = [0.246, -0.044645, 0.3016983]
# 3.3 右手交接前高位 z：先抬高避开桌面其它快递盒，再对齐 x/y 和四元素。
# 最终交接高度看 RIGHT_HANDOFF_TO_LEFT_IK[2]。
RIGHT_HANDOFF_TRANSIT_IK_Z = 0.40
# 3.3 高位优先使用上面的避障高度；个别启动状态下右手从称重区直接到 0.40
# 可能不可解，按这个列表降级重试，不改最终交接高度。
RIGHT_HANDOFF_TRANSIT_FALLBACK_IK_ZS = [0.37, 0.35]
RIGHT_HANDOFF_TO_LEFT_FIRST_YPR_DEG = [-0.839, -100.0, 0.0]
RIGHT_HANDOFF_TO_LEFT_SECOND_YPR_DEG = [90.0, -20.0, 90.0]
RIGHT_HANDOFF_TO_LEFT_QUAT_XYZW = _quat_from_ypr_deg(
    RIGHT_HANDOFF_TO_LEFT_FIRST_YPR_DEG,
    RIGHT_HANDOFF_TO_LEFT_SECOND_YPR_DEG,
)

# 3.4 左手接收点：左手先在侧方对齐接收点 x/z 和四元素，再只沿 y 接近右手。
LEFT_HANDOFF_RECEIVE_XZ_READY_IK = [0.266, 0.139, 0.2216983]
LEFT_HANDOFF_RECEIVE_IK = [0.266, 0.04645, 0.2216983]
LEFT_HANDOFF_RECEIVE_FIRST_YPR_DEG = [-0.839, -100.0, 0.0]
LEFT_HANDOFF_RECEIVE_SECOND_YPR_DEG = [-90.0, -0.0, -90.0]
LEFT_HANDOFF_RECEIVE_QUAT_XYZW = _quat_from_ypr_deg(
    LEFT_HANDOFF_RECEIVE_FIRST_YPR_DEG,
    LEFT_HANDOFF_RECEIVE_SECOND_YPR_DEG,
)

# 3.5 右手退让：左手夹住、右手松开后，右手按这个 y 偏移退开，再让左手去箱子。
# 运行时会从当前右手 FK 位置退让，不强行回到理论 3.3 高度；负 y 是当前右手外侧方向。
RIGHT_HANDOFF_RELEASE_RETRACT_Y_OFFSET = -0.300

# ===== 4. 左手把快递放入箱子 =====
# sorting_box_0p4_0p3_0p3 的世界坐标原点定义在 scene1.yaml 中。
# 箱子不跟随随机数种子变化，只有包裹初始位置会变。
# 箱内放置点按 scene1.yaml 的箱子真实位置计算。
# 箱子原点 world=[0.10, 0.29, 0.83]，内尺寸 0.40x0.30x0.30。
# 箱子中心换算到 IK 的 x=0.665966，当前左臂够不到；这里取靠机器人侧的箱内点作为放置基准。
BOX_DROP_IK = [0.605966, 0.226114, 0.556217]
# 左臂够箱子在工作空间边缘，放置点 IK 偶尔解不出(out of workspace)。
# 仅在 IK 失败时，按此表把放置点 x 依次往机器人侧收(减小 x → 更易达)重试；没失败就用原值、不动。
# 收太多包裹会落到箱近沿外(入箱检查会判失败)，故幅度有限。
BOX_DROP_IK_X_FALLBACK_DELTAS = [-0.03, -0.06]
# 每个包裹放入箱子时，在 BOX_DROP_IK 基准点上叠加的偏移。
# 格式：[x, y, z]，单位 m，坐标轴与 IK/base 一致；z 可用于单独微调放置高度。
# 默认排成 2x2：x 不往更远处加，避免左臂够不到；y 分左右两列。
BOX_DROP_OFFSET_BY_PARCEL = {
    "parcel_1": [0.0, -0.01, 0.0],
    "parcel_2": [0.0, 0.04, 0.0],
    "parcel_3": [0.04, -0.03, 0.0],
    "parcel_4": [0.04, 0.03, 0.0],
}
BOX_BODY_NAME = "sorting_box_0p4_0p3_0p3"
BOX_INNER_SIZE_WORLD = [0.40, 0.30, 0.30]
BOX_FLOOR_THICKNESS = 0.016
# 左手松开后，包裹中心必须在箱体实际 body 的内框附近。
# margin 给仿真弹跳/包裹尺寸留余量，但不会允许落到箱子外很远。
BOX_CHECK_XY_MARGIN = 0.035
BOX_CHECK_Z_MARGIN = 0.055
LEFT_BOX_DROP_FIRST_YPR_DEG = [-0.328, -100.935, 0.0]
LEFT_BOX_DROP_SECOND_YPR_DEG = [-90.0, 0.0, 0.369]
LEFT_BOX_DROP_QUAT_XYZW = _quat_from_ypr_deg(
    LEFT_BOX_DROP_FIRST_YPR_DEG,
    LEFT_BOX_DROP_SECOND_YPR_DEG,
)
PLACE_APPROACH_Z_OFFSET = 0.060
PLACE_DWELL = 0.8
# 放箱上方中间点只用于避障拆路径，不需要每到一个上方点都停顿。
BOX_TRANSIT_SETTLE_TIME = 0.0

# ===== 运动时序与求解容差 =====
SCENE1_ARM_MOVE_TIME = 1
SCENE1_ARM_SETTLE_TIME = 0.8
HANDOFF_MOVE_TIME = 2.2
PICK_ALIGN_MOVE_TIME = 1.2
PICK_GRASP_MOVE_TIME = 1.4
PLACE_MOVE_TIME = 2.2
SCENE1_IK_CONSTRAINT_MODE = IK_MODE_THREE_POINT_MIXED  # 0x06，与 arm_control.py 保持一致
SCENE1_IK_POS_COST_WEIGHT = 1.0
SCENE1_IK_MAJOR_ITERATIONS_LIMIT = 100
HANDOFF_POS_COST_WEIGHT = SCENE1_IK_POS_COST_WEIGHT
# 关节轨迹分段上限。值越小越稳但越慢；14deg 在保持分段平滑的同时减少大量小段等待。
MAX_JOINT_STEP_DEG = 14.0
# 单个分段的最短执行时间。原来固定 0.35s 会让大角度动作即使调小 move_time 也快不起来。
MIN_JOINT_SEGMENT_TIME = 0.1
TRANSIT_POS_TOLERANCE = 0.070
PRESET_POS_TOLERANCE = 0.005
CONTACT_POS_TOLERANCE = PRESET_POS_TOLERANCE
HANDOFF_POS_TOLERANCE = CONTACT_POS_TOLERANCE
CONTACT_RETRY_COUNT = 2
CONTACT_RETRY_SETTLE_TIME = 0.3
TRANSIT_ORI_TOLERANCE_RAD = math.radians(30.0)
HANDOFF_ORI_TOLERANCE_RAD = math.radians(20.0)
# 运行时 FK 到位误差只用于诊断和打印 warning，不用于中断流程；
# 这类检查不能提高控制精度，真正的硬错误仍然是 IK 服务失败或 ROS 节点异常。
RAISE_ON_RUNTIME_POSE_ERROR = False

# ===== 等收敛（替代固定 settle）=====
# 仿真按墙钟时间跑、无 /clock，控制器在固定 settle 窗口内能否跟到位取决于本次 CPU 负载，
# 这是“同 seed 偶发失败”的根因。开启后：命令下发→轮询 FK，直到末端运动停稳或超时才返回，
# 给够时间且不空等。对实时率抖动免疫。
# 注意：必须等“位置+姿态都停稳”，不能位置一进容差就早退——否则像 right_yz_align 这种
# “位置先到、姿态还在大幅转动”的动作会在转到一半时被放行，把姿态没到位的坏构型交给
# 下一步精确 IK，导致 IK FAILED。
WAIT_FOR_CONVERGENCE = True
CONVERGE_TIMEOUT = 3.0                      # 收敛等待最长时间（s，墙钟）
CONVERGE_POLL_DT = 0.1                      # 轮询间隔（s）
CONVERGE_STABLE_EPS = 0.002                 # 连续两次末端位移 < 2mm 视为位置停稳
CONVERGE_STABLE_ORI_EPS = math.radians(1.0) # 连续两次末端姿态变化 < 1° 视为姿态停稳
CONVERGE_STABLE_HITS = 3                     # 位置+姿态连续达到停稳次数后判定停稳

# ===== 夹爪命令 =====
RIGHT_GRIPPER_OPEN = 0.0
LEFT_GRIPPER_OPEN = 0.0
RIGHT_GRIPPER_CLOSE = 255.0
LEFT_GRIPPER_CLOSE = 255.0
GRIPPER_OPEN_STATE_TOLERANCE = 0.05
GRIPPER_OPEN_WAIT_TIMEOUT = 3.0
GRIPPER_CLOSE_HOLD_TIME = 0.7

# 双臂预设抬手动作(路点 + 线性插值)。开场把双臂带到身前的对称预设位:
#   段1: 双肩外展张开(避免小臂蹭躯干/桌子的安全路径)
#   段2: 收外展 + 弯肘 + 肩旋,把双小臂抬到身前的对称预设位
PRESET_POINTS_DEG = [
    [20, 0, 0, -30, 0, 0, 0, 20, 0, 0, -30, 0, 0, 0],          # 起点
    [20, 90, 0, -55, 0, 0, 0, 20, -90, 0, -55, 0, 0, 0],       # 外展峰(段1终)
    [20, 60, 0, -75, 0, 0, 0, 20, -60, 0, -75, 0, 0, 0],       # 外展峰(段1终)
    [29.89, 30.67, 29.889, -139.1, -59.33, 0, 0,
     29.89, -30.67, -29.889, -139.1, 59.33, 0, 0],              # 末端预设(段2终)
    [29.89, 10.67, 9.889, -139.1, -59.33, 0, 0,
     29.89, -10.67, -9.889, -139.1, 59.33, 0, 0],    
]
PRESET_SEGMENT_TIME = 1.92   # 相邻路点之间的插值时长(秒)
PRESET_SETTLE_TIME = 0.5     # 抵达末端预设后的稳定时间
PRESET_SPLINE_TENSION = 0.0  # 样条张力: 0=最圆滑(Catmull-Rom), 越接近 1 越接近直线(收紧/抑制过冲)


def _cardinal_point(p0, p1, p2, p3, s, tension):
    """p1→p2 段的 Cardinal/Catmull-Rom 样条插值(s∈[0,1])。tension=0 即 Catmull-Rom。"""
    c = (1.0 - tension) * 0.5
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    out = []
    for i in range(len(p1)):
        m1 = c * (p2[i] - p0[i])
        m2 = c * (p3[i] - p1[i])
        out.append(h00 * p1[i] + h10 * m1 + h01 * p2[i] + h11 * m2)
    return out


def _publish_preset_spline(arm_hold, points, seg_time):
    """对全部路点做 Catmull-Rom 样条(C1 连续,精确经过每个路点)+ 全程缓入缓出,
    连续流式下发 /kuavo_arm_traj。相邻路点之间速度连续、不停顿,起止平滑加减速。"""
    import rospy

    pts = [[float(v) for v in p] for p in points]
    n = len(pts)
    if n == 0:
        return
    if n == 1:
        arm_hold.set_degrees(pts[0])
        return
    segments = n - 1
    total_steps = max(1, int(round(segments * float(seg_time) * ARM_TRAJ_HZ)))
    rate = rospy.Rate(ARM_TRAJ_HZ)
    for step in range(total_steps + 1):
        if rospy.is_shutdown():
            return
        u = step / float(total_steps)
        eased = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)  # smootherstep 缓入缓出
        pos = eased * segments
        seg = min(int(pos), segments - 1)
        s = pos - seg
        p1 = pts[seg]
        p2 = pts[seg + 1]
        p0 = pts[seg - 1] if seg - 1 >= 0 else p1
        p3 = pts[seg + 2] if seg + 2 < n else p2
        arm_hold.set_degrees(_cardinal_point(p0, p1, p2, p3, s, PRESET_SPLINE_TENSION))
        if step < total_steps:
            rate.sleep()
    arm_hold.set_degrees(pts[-1])


def _move_through_preset(arm_pub, arm_hold, max_points=None):
    """把双臂运动到身前对称预设位,再进入抓取等后续步骤。

    以当前姿态为起点,对全部路点做样条插值 + 缓入缓出,连续流式下发,
    相邻路点之间不停顿、速度连续 → 顺滑无顿挫。
    max_points: 只执行前 N 个路点(调试用,例如 N=2 只走"外展张开"那一段)。
    """
    import rospy

    points = PRESET_POINTS_DEG
    if max_points is not None:
        points = points[: int(max_points)]
    start = _rad_to_deg(_read_current_arm_joints(TOPIC_TIMEOUT))
    _publish_preset_spline(arm_hold, [start] + list(points), PRESET_SEGMENT_TIME)
    rospy.sleep(PRESET_SETTLE_TIME)


def _hold_for_observation(seconds):
    """停在当前姿态供观察。arm_hold 线程持续维持姿态;到时或 Ctrl-C 退出。"""
    import rospy

    rospy.loginfo("scene1 debug: 保持姿态 %.1fs 供观察 (Ctrl-C 提前退出)", float(seconds))
    deadline = rospy.Time.now() + rospy.Duration(float(seconds))
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        rospy.sleep(0.2)

_TF_LISTENER = None
_MUJOCO_MODEL = None
_MUJOCO_DATA = None
_MUJOCO_MODEL_PATH = None
_POSE_CHECK_SPEC = None
_PROGRESS_LOCK = threading.Lock()
_PROGRESS = {
    "seed": None,
    "run_index": None,
    "run_total": None,
    "attempt_index": None,
    "attempt_total": None,
    "stage": "startup",
    "job": None,
}


def _now_tag():
    return _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _set_progress(**updates):
    with _PROGRESS_LOCK:
        _PROGRESS.update(updates)


def _progress_reason(prefix):
    with _PROGRESS_LOCK:
        snapshot = dict(_PROGRESS)
    parts = [str(prefix)]
    for key in ("seed", "run_index", "run_total", "attempt_index", "attempt_total", "stage", "job"):
        value = snapshot.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _world_to_ik(world_xyz):
    return [float(world_xyz[i]) + WORLD_TO_IK_OFFSET[i] for i in range(3)]


def _with_z_offset(pos_xyz, offset):
    return [float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]) + float(offset)]


def _with_ik_z(pos_xyz, ik_z):
    return [float(pos_xyz[0]), float(pos_xyz[1]), float(ik_z)]


def _right_pick_offset_for_parcel(parcel_name, source_world=None):
    # 优先 per-parcel 覆盖;否则按包裹实际 Y 分区(近身/远身)选偏移。source_world=None 时退远身基准。
    if parcel_name in RIGHT_PICK_OFFSET_BY_PARCEL:
        offset = list(RIGHT_PICK_OFFSET_BY_PARCEL[parcel_name])
    elif source_world is not None and float(source_world[1]) > RIGHT_PICK_NEAR_FAR_Y_THRESHOLD:
        offset = list(RIGHT_PICK_OFFSET_NEAR_ROW)
    else:
        offset = list(RIGHT_PICK_OFFSET_FAR_ROW)
    if len(offset) == 2:
        offset.append(0.0)
    if len(offset) != 3:
        raise ValueError(f"{parcel_name} right pick offset expects [x, y, z_delta], got {offset}")
    return [float(offset[0]), float(offset[1]), float(offset[2])]


def _right_pick_ik_from_source_world(parcel_name, source_world):
    source_ik = _world_to_ik(source_world)
    offset = _right_pick_offset_for_parcel(parcel_name, source_world)
    return [
        source_ik[0] + offset[0],
        source_ik[1] + offset[1],
        RIGHT_PICK_IK_Z + offset[2],
    ]


def _right_pick_pre_ik(right_pick_ik):
    return _with_ik_z(right_pick_ik, RIGHT_PICK_TRANSIT_IK_Z)


def _right_pick_lift_ik(right_pick_ik):
    return _with_z_offset(right_pick_ik, LIFT_Z_OFFSET)


def _right_pick_yz_align_ik(current_right_ik, right_pick_pre_ik):
    # 先把 y 和姿态拉到抓取上方附近，x 保持在当前值和安全 x 之间。
    # z 不在这一步提前降到抓取横移高度：第二个及后续包裹会从右手退让位回来，
    # 如果一边回 y 一边降到低 z，IK 很容易落到不可达/差分支。
    # 下一步 right_x_to_pick_pre 再统一去真实抓取上方高度。
    current_x = float(current_right_ik[0])
    target_x = float(right_pick_pre_ik[0])
    if current_x <= target_x:
        align_x = min(target_x, max(current_x, RIGHT_PICK_YZ_ALIGN_SAFE_IK_X))
    else:
        align_x = max(target_x, min(current_x, RIGHT_PICK_YZ_ALIGN_SAFE_IK_X))
    align_z = max(float(current_right_ik[2]), float(right_pick_pre_ik[2]))
    return [align_x, float(right_pick_pre_ik[1]), align_z]


def _left_box_pre_ik(box_ik):
    return _with_z_offset(box_ik, PLACE_APPROACH_Z_OFFSET)


def _left_box_raise_before_xy_ik(current_left_ik, box_ik):
    # 放箱前先在当前位置抬到箱口上方高度，再横移到箱内上方，避免低位横移撞箱子。
    box_pre = _left_box_pre_ik(box_ik)
    return [
        float(current_left_ik[0]),
        float(current_left_ik[1]),
        float(box_pre[2]),
    ]


def _box_drop_offset_for_parcel(parcel_name):
    offset = list(BOX_DROP_OFFSET_BY_PARCEL.get(parcel_name, [0.0, 0.0, 0.0]))
    if len(offset) == 2:
        offset.append(0.0)
    if len(offset) != 3:
        raise ValueError(f"{parcel_name} box drop offset expects [x, y, z], got {offset}")
    return [float(offset[0]), float(offset[1]), float(offset[2])]


def _box_drop_ik_for_parcel(parcel_name):
    offset = _box_drop_offset_for_parcel(parcel_name)
    return [
        float(BOX_DROP_IK[0]) + offset[0],
        float(BOX_DROP_IK[1]) + offset[1],
        float(BOX_DROP_IK[2]) + offset[2],
    ]


def _get_tf_listener():
    global _TF_LISTENER
    if _TF_LISTENER is None:
        import rospy
        import tf

        _TF_LISTENER = tf.TransformListener()
        rospy.sleep(0.3)
    return _TF_LISTENER


def _lookup_eef_tf(side, timeout=1.0):
    import rospy

    listener = _get_tf_listener()
    frames = {
        "left": ("zarm_l7_end_effector", "eef_left"),
        "right": ("zarm_r7_end_effector", "eef_right"),
    }[side]
    last_error = None
    for frame in frames:
        try:
            listener.waitForTransform("base_link", frame, rospy.Time(0), rospy.Duration(timeout))
            pos, quat = listener.lookupTransform("base_link", frame, rospy.Time(0))
            return list(pos), list(quat), f"tf:{frame}"
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"TF lookup failed for {side} end effector: {last_error}")


def _actual_pose_for_side(fk_poses, side):
    # IK 目标和 FK 服务使用同一套 Drake 末端定义。夹爪 TF 里也发布了
    # zarm_*7_end_effector，但它带有夹爪工具偏移，不能直接拿来判定 IK 是否到位。
    pose = _pose_for_side(fk_poses, side)
    return list(pose.pos_xyz), list(pose.quat_xyzw), "fk"


def _log_tf_pose_offset(label, side, fk_pos, desired_pos):
    import rospy

    try:
        tf_pos, _, tf_source = _lookup_eef_tf(side, timeout=0.2)
    except Exception as exc:
        rospy.logwarn("scene1 handoff: %s %s TF diagnostic skipped: %s", label, side, exc)
        return

    tf_vs_fk = _axis_error(tf_pos, fk_pos, (True, True, True))
    if desired_pos is None:
        rospy.loginfo(
            "scene1 handoff: %s %s %s offset_from_fk=%.4f m tf_actual=%s fk_actual=%s",
            label,
            side,
            tf_source,
            tf_vs_fk,
            [round(v, 4) for v in tf_pos],
            [round(v, 4) for v in fk_pos],
        )
        return

    tf_target_err = _axis_error(tf_pos, desired_pos, (True, True, True))
    rospy.loginfo(
        "scene1 handoff: %s %s %s offset_from_fk=%.4f m tf_target_err=%.4f m tf_actual=%s fk_actual=%s",
        label,
        side,
        tf_source,
        tf_vs_fk,
        tf_target_err,
        [round(v, 4) for v in tf_pos],
        [round(v, 4) for v in fk_pos],
    )


def _mujoco_model_and_data():
    global _MUJOCO_MODEL, _MUJOCO_DATA, _MUJOCO_MODEL_PATH

    import rospy
    import mujoco

    model_path = rospy.get_param("legged_robot_scene_param", None)
    if not model_path:
        raise RuntimeError("ROS param legged_robot_scene_param is missing; cannot check object pose")
    model_path = os.path.abspath(str(model_path))
    if _MUJOCO_MODEL is None or _MUJOCO_MODEL_PATH != model_path:
        if not os.path.isfile(model_path):
            raise RuntimeError(f"MuJoCo scene XML not found: {model_path}")
        _MUJOCO_MODEL = mujoco.MjModel.from_xml_path(model_path)
        _MUJOCO_DATA = mujoco.MjData(_MUJOCO_MODEL)
        _MUJOCO_MODEL_PATH = model_path
    return _MUJOCO_MODEL, _MUJOCO_DATA


def _mujoco_body_world_pos(body_name, timeout=1.0):
    import mujoco
    import rospy
    from std_msgs.msg import Float64MultiArray

    model, data = _mujoco_model_and_data()
    msg = rospy.wait_for_message(MUJOCO_QPOS_TOPIC, Float64MultiArray, timeout=timeout)
    qpos = list(msg.data)
    if len(qpos) < model.nq:
        raise RuntimeError(
            f"{MUJOCO_QPOS_TOPIC} has {len(qpos)} qpos values, model expects at least {model.nq}"
        )
    data.qpos[:model.nq] = qpos[:model.nq]
    mujoco.mj_forward(model, data)
    try:
        return [float(v) for v in data.body(str(body_name)).xpos]
    except KeyError as exc:
        raise RuntimeError(f"MuJoCo body not found for pose check: {body_name}") from exc


def _scene1_pose_check_spec():
    global _POSE_CHECK_SPEC
    if _POSE_CHECK_SPEC is not None:
        return _POSE_CHECK_SPEC

    # 只有 box 子表会被读取（见 _assert_parcel_in_box）；称重判定走 geom 角点，不用 spec。
    fallback = {
        "box": {
            "body_name": BOX_BODY_NAME,
            "inner_size_world": list(BOX_INNER_SIZE_WORLD),
            "floor_thickness": BOX_FLOOR_THICKNESS,
            "xy_margin": BOX_CHECK_XY_MARGIN,
            "z_margin": BOX_CHECK_Z_MARGIN,
        },
    }
    try:
        challenge_secret = _load_challenge_secret()
        getter = getattr(challenge_secret, "get_pose_check_spec", None)
        spec = getter(SCENE_NAME) if getter is not None else None
        _POSE_CHECK_SPEC = spec or fallback
    except Exception:
        _POSE_CHECK_SPEC = fallback
    return _POSE_CHECK_SPEC


def _assert_parcel_on_weighing_area(parcel_name):
    """称重成功判定：与仿真“称重区变黄”逻辑完全同源。

    仿真里 mujoco_node.cc::updateScene1WeighingAreaIndicator 的判定（变黄=success）：
      对每个包裹算其 geom 的真实 8 个角点（含实际旋转 geom_xmat），
      若全部 (x,y) 落入称重区 geom 的世界方块 [center±half]（容差 0.002）即 full_inside；
      success = (恰好 1 个包裹 full_inside+静止+脱离夹爪) 且 (恰好 1 个包裹与区域 overlap)。
    本检查复刻其几何核心 full_inside：区域中心/半边直接读 MuJoCo 里 weighing_area geom 的
    真实 xpos/size（不再用硬编码常量，也不是官方 spec 的 [-0.10,-0.56]±0.18），
    包裹用真实角点判定。静止/脱手/唯一性由流水线保证（释放并停稳后调用、逐个处理）。
    """
    import mujoco
    import rospy
    from std_msgs.msg import Float64MultiArray

    model, data = _mujoco_model_and_data()
    msg = rospy.wait_for_message(MUJOCO_QPOS_TOPIC, Float64MultiArray, timeout=TOPIC_TIMEOUT)
    qpos = list(msg.data)
    if len(qpos) < model.nq:
        raise RuntimeError(
            f"{MUJOCO_QPOS_TOPIC} has {len(qpos)} qpos values, model expects at least {model.nq}"
        )
    data.qpos[:model.nq] = qpos[:model.nq]
    mujoco.mj_forward(model, data)

    area_geom = data.geom(WEIGHING_AREA_GEOM_NAME)
    area_size = model.geom(WEIGHING_AREA_GEOM_NAME).size
    ax, ay = float(area_geom.xpos[0]), float(area_geom.xpos[1])
    ahx, ahy = float(area_size[0]), float(area_size[1])
    min_x, max_x = ax - ahx, ax + ahx
    min_y, max_y = ay - ahy, ay + ahy

    parcel_geom_name = f"{parcel_name}_geom"
    pg = data.geom(parcel_geom_name)
    center = [float(v) for v in pg.xpos]
    mat = [float(v) for v in pg.xmat]  # row-major 3x3，与 C++ geom_xmat 一致
    size = [float(v) for v in model.geom(parcel_geom_name).size]

    tol = float(WEIGH_FOOTPRINT_TOLERANCE)
    full_inside = True
    pminx, pmaxx, pminy, pmaxy = 1e9, -1e9, 1e9, -1e9
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                lx, ly, lz = sx * size[0], sy * size[1], sz * size[2]
                wx = center[0] + mat[0] * lx + mat[1] * ly + mat[2] * lz
                wy = center[1] + mat[3] * lx + mat[4] * ly + mat[5] * lz
                pminx, pmaxx = min(pminx, wx), max(pmaxx, wx)
                pminy, pmaxy = min(pminy, wy), max(pmaxy, wy)
                if wx < min_x - tol or wx > max_x + tol or wy < min_y - tol or wy > max_y + tol:
                    full_inside = False

    z_min, z_max = [float(v) for v in WEIGH_CHECK_Z_RANGE]
    z_ok = z_min <= center[2] <= z_max
    rospy.loginfo(
        "scene1 handoff: %s weighing(变黄判定) area=[%.4f,%.4f]±[%.3f,%.3f] parcel_xy=[%.4f,%.4f] "
        "footprint_x=[%.4f,%.4f] footprint_y=[%.4f,%.4f] full_inside=%s z=%.4f z_ok=%s",
        parcel_name, ax, ay, ahx, ahy, center[0], center[1],
        pminx, pmaxx, pminy, pmaxy, full_inside, center[2], z_ok,
    )
    if not (full_inside and z_ok):
        raise RuntimeError(
            f"{parcel_name} not fully on weighing area (变黄判定 full_inside={full_inside} z_ok={z_ok}): "
            f"parcel_xy=[{center[0]:.4f},{center[1]:.4f}] footprint x=[{pminx:.4f},{pmaxx:.4f}] "
            f"y=[{pminy:.4f},{pmaxy:.4f}] area=[{min_x:.4f},{max_x:.4f}]x[{min_y:.4f},{max_y:.4f}] z={center[2]:.4f}"
        )


def _assert_parcel_in_box(parcel_name):
    import rospy

    spec = _scene1_pose_check_spec().get("box", {})
    parcel_pos = _mujoco_body_world_pos(parcel_name, timeout=TOPIC_TIMEOUT)
    box_body_name = spec.get("body_name", BOX_BODY_NAME)
    box_pos = _mujoco_body_world_pos(box_body_name, timeout=TOPIC_TIMEOUT)
    inner_size = spec.get("inner_size_world", BOX_INNER_SIZE_WORLD)
    xy_margin = float(spec.get("xy_margin", BOX_CHECK_XY_MARGIN))
    z_margin = float(spec.get("z_margin", BOX_CHECK_Z_MARGIN))
    floor_thickness = float(spec.get("floor_thickness", BOX_FLOOR_THICKNESS))
    half_x = float(inner_size[0]) / 2.0 + xy_margin
    half_y = float(inner_size[1]) / 2.0 + xy_margin
    z_min = float(box_pos[2]) + floor_thickness - z_margin
    z_max = float(box_pos[2]) + float(inner_size[2]) + z_margin
    dx = abs(parcel_pos[0] - box_pos[0])
    dy = abs(parcel_pos[1] - box_pos[1])
    inside = dx <= half_x and dy <= half_y and z_min <= parcel_pos[2] <= z_max
    rospy.loginfo(
        "scene1 handoff: %s box pose check parcel=%s box=%s dxy=[%.4f, %.4f] half=[%.4f, %.4f] z_range=[%.4f, %.4f]",
        parcel_name,
        [round(v, 4) for v in parcel_pos],
        [round(v, 4) for v in box_pos],
        dx,
        dy,
        half_x,
        half_y,
        z_min,
        z_max,
    )
    if not inside:
        raise RuntimeError(
            f"{parcel_name} box check failed: parcel={parcel_pos}, box={box_pos}, "
            f"dx={dx:.4f}, dy={dy:.4f}, allowed_half={[half_x, half_y]}, z_range={[z_min, z_max]}"
        )


def _parse_float_list(value, expected, name):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        text = str(value).strip()
        if text.lower() in ("", "none", "current"):
            return None
        values = text.replace(",", " ").split()
    result = [float(v) for v in values]
    if len(result) != expected:
        raise ValueError(f"{name} expects {expected} floats, got {len(result)}: {value}")
    return result


def _load_challenge_secret():
    lib_dir = _find_package_subdir("lib")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    import challenge_secret

    return challenge_secret


def _load_scene1_layout(seed):
    challenge_secret = _load_challenge_secret()
    layout = challenge_secret.get_object_layout(SCENE_NAME, int(seed))
    missing = [name for name in PARCEL_NAMES if name not in layout]
    if missing:
        raise RuntimeError(f"scene1 layout missing parcels: {', '.join(missing)}")
    return layout


def _start_scene_launch(seed):
    ChallengeSimLauncher = _load_challenge_sim_launcher()
    launcher = ChallengeSimLauncher(
        scene=SCENE_NAME,
        seed=seed,
        match_time_limit=0,
        timer_gui=False,
    )
    launcher.start(node_name="scene1_handoff_dataset_collector", timeout=LAUNCH_TIMEOUT)
    return launcher


def _stop_scene_launch(launcher):
    if launcher is not None:
        launcher.stop()


class GripperCommandHold:
    def __init__(self, pub, hz=GRIPPER_COMMAND_HZ):
        self._pub = pub
        self._hz = float(hz)
        self._left_cmd = float(LEFT_GRIPPER_OPEN)
        self._right_cmd = float(RIGHT_GRIPPER_OPEN)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="scene1_gripper_hold", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_open(self):
        self.set_command(LEFT_GRIPPER_OPEN, RIGHT_GRIPPER_OPEN)

    def set_left_open(self):
        self.set_command(LEFT_GRIPPER_OPEN, self._right())

    def set_right_open(self):
        self.set_command(self._left(), RIGHT_GRIPPER_OPEN)

    def set_left_closed(self):
        self.set_command(LEFT_GRIPPER_CLOSE, self._right())

    def set_right_closed(self):
        self.set_command(self._left(), RIGHT_GRIPPER_CLOSE)

    def set_left_closed_right_open(self):
        self.set_command(LEFT_GRIPPER_CLOSE, RIGHT_GRIPPER_OPEN)

    def set_both_closed(self):
        self.set_command(LEFT_GRIPPER_CLOSE, RIGHT_GRIPPER_CLOSE)

    def set_command(self, left_cmd, right_cmd):
        with self._lock:
            self._left_cmd = float(left_cmd)
            self._right_cmd = float(right_cmd)

    def _left(self):
        with self._lock:
            return self._left_cmd

    def _right(self):
        with self._lock:
            return self._right_cmd

    def _run(self):
        import rospy
        from sensor_msgs.msg import JointState

        rate = rospy.Rate(self._hz)
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            with self._lock:
                left_cmd = self._left_cmd
                right_cmd = self._right_cmd
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ["left_gripper_joint", "right_gripper_joint"]
            msg.position = [left_cmd, right_cmd]
            self._pub.publish(msg)
            rate.sleep()


def _start_gripper_hold(timeout):
    import rospy
    from sensor_msgs.msg import JointState

    pub = rospy.Publisher("/gripper/command", JointState, queue_size=10)
    _wait_for_connection(pub, timeout)
    hold = GripperCommandHold(pub)
    hold.start()
    return hold


def _pose_for_side(fk_poses, side):
    return fk_poses.left_pose if side == "left" else fk_poses.right_pose


def _set_pose_target(pose, pos_xyz, quat_xyzw):
    pose.pos_xyz = list(pos_xyz)
    pose.quat_xyzw = list(quat_xyzw)


def _make_scene1_ik_param(
    constraint_mode=SCENE1_IK_CONSTRAINT_MODE,
    pos_cost_weight=SCENE1_IK_POS_COST_WEIGHT,
    major_iterations_limit=SCENE1_IK_MAJOR_ITERATIONS_LIMIT,
):
    param = _make_ik_param(
        constraint_mode=constraint_mode,
        pos_cost_weight=pos_cost_weight,
    )
    param.major_iterations_limit = int(major_iterations_limit)
    return param


def _response_arm_joints(response):
    left_result = list(response.hand_poses.left_pose.joint_angles)
    right_result = list(response.hand_poses.right_pose.joint_angles)
    if len(left_result) == 7 and len(right_result) == 7:
        return left_result + right_result
    if len(response.q_arm) >= 14:
        return list(response.q_arm[:14])
    raise RuntimeError("IK response did not contain 14 arm joints")


def _response_one_hand_joints(response, current_joint_values, side):
    current = list(current_joint_values[:14])
    if len(current) != 14:
        raise RuntimeError(f"current arm joints did not contain 14 values: {len(current)}")

    if side == "left":
        target_result = list(response.hand_poses.left_pose.joint_angles)
        if len(target_result) != 7 and len(response.q_arm) >= 7:
            target_result = list(response.q_arm[:7])
        if len(target_result) == 7:
            return target_result + current[7:]
    elif side == "right":
        target_result = list(response.hand_poses.right_pose.joint_angles)
        if len(target_result) != 7 and len(response.q_arm) >= 14:
            target_result = list(response.q_arm[7:14])
        if len(target_result) == 7:
            return current[:7] + target_result
    else:
        raise ValueError(f"unknown hand side: {side}")

    raise RuntimeError(f"IK response did not contain 7 {side} arm joints")


def _execute_joint_motion_chunked(arm_pub, arm_hold, start_degrees, target_degrees, move_time, settle_time):
    start = [float(v) for v in start_degrees]
    target = [float(v) for v in target_degrees]
    if len(start) != len(target):
        raise ValueError(f"arm command length mismatch: {len(start)} != {len(target)}")

    max_delta = max(abs(target[i] - start[i]) for i in range(len(target)))
    segments = max(1, int(math.ceil(max_delta / MAX_JOINT_STEP_DEG)))
    segment_time = max(float(MIN_JOINT_SEGMENT_TIME), float(move_time) / float(segments))
    previous = start
    for segment in range(1, segments + 1):
        alpha = float(segment) / float(segments)
        point = [start[i] + (target[i] - start[i]) * alpha for i in range(len(target))]
        _execute_arm_motion(arm_pub, arm_hold, previous, point, segment_time, 0.0)
        previous = point
    import rospy

    rospy.sleep(settle_time)


def _log_and_guard_pose(
    label,
    side,
    actual_pos,
    desired_pos,
    actual_quat,
    desired_quat,
    actual_source,
    max_pos_err,
    max_quat_err,
    raise_on_error=True,
):
    import rospy

    pos_err = None
    quat_err = None
    if desired_pos is not None:
        pos_err = _axis_error(actual_pos, desired_pos, (True, True, True))
    if desired_quat is not None:
        quat_err = _quat_angle_error(actual_quat, desired_quat)

    if pos_err is not None and quat_err is not None:
        rospy.loginfo(
            "scene1 handoff: %s %s source=%s actual=%s pos_err=%.4f m quat_err=%.1f deg",
            label,
            side,
            actual_source,
            [round(v, 4) for v in actual_pos],
            pos_err,
            math.degrees(quat_err),
        )
    elif pos_err is not None:
        rospy.loginfo(
            "scene1 handoff: %s %s source=%s actual=%s pos_err=%.4f m",
            label,
            side,
            actual_source,
            [round(v, 4) for v in actual_pos],
            pos_err,
        )

    if max_pos_err is not None and pos_err is not None and pos_err > max_pos_err:
        if not raise_on_error:
            rospy.logwarn(
                "scene1 handoff: %s %s position over tolerance: err=%.4f m > %.4f m",
                label,
                side,
                pos_err,
                max_pos_err,
            )
            return
        raise RuntimeError(
            f"{label} {side} position did not converge: err={pos_err:.4f} m > {max_pos_err:.4f} m"
        )
    if max_quat_err is not None and quat_err is not None and quat_err > max_quat_err:
        if not raise_on_error:
            rospy.logwarn(
                "scene1 handoff: %s %s orientation over tolerance: err=%.1f deg > %.1f deg",
                label,
                side,
                math.degrees(quat_err),
                math.degrees(max_quat_err),
            )
            return
        raise RuntimeError(
            f"{label} {side} orientation did not converge: "
            f"err={math.degrees(quat_err):.1f} deg > {math.degrees(max_quat_err):.1f} deg"
        )


def _pose_error_exceeds(actual_pos, desired_pos, actual_quat, desired_quat, max_pos_err, max_quat_err):
    if max_pos_err is not None and desired_pos is not None:
        pos_err = _axis_error(actual_pos, desired_pos, (True, True, True))
        if pos_err > max_pos_err:
            return True
    if max_quat_err is not None and desired_quat is not None:
        quat_err = _quat_angle_error(actual_quat, desired_quat)
        if quat_err > max_quat_err:
            return True
    return False


def _motion_needs_retry(
    fk,
    left_pos,
    left_quat,
    right_pos,
    right_quat,
    max_left_pos_err,
    max_left_quat_err,
    max_right_pos_err,
    max_right_quat_err,
):
    if left_pos is not None or left_quat is not None:
        actual_pos, actual_quat, _ = _actual_pose_for_side(fk, "left")
        if _pose_error_exceeds(
            actual_pos,
            left_pos,
            actual_quat,
            left_quat,
            max_left_pos_err,
            max_left_quat_err,
        ):
            return True
    if right_pos is not None or right_quat is not None:
        actual_pos, actual_quat, _ = _actual_pose_for_side(fk, "right")
        if _pose_error_exceeds(
            actual_pos,
            right_pos,
            actual_quat,
            right_quat,
            max_right_pos_err,
            max_right_quat_err,
        ):
            return True
    return False


def _settle_eef_poses(fk, left_pos, right_pos):
    """取本次受约束侧的末端 (位置, 姿态)，用于判断运动是否停稳（位置和姿态都要停）。"""
    poses = []
    if left_pos is not None:
        p = _actual_pose_for_side(fk, "left")
        poses.append((p[0], p[1]))
    if right_pos is not None:
        p = _actual_pose_for_side(fk, "right")
        poses.append((p[0], p[1]))
    return poses


def _poses_max_delta(prev_poses, cur_poses):
    """相邻两次采样间，各受约束侧末端的最大位置位移(m)与最大姿态变化(rad)。"""
    max_pos = 0.0
    max_ori = 0.0
    for (p0, q0), (p1, q1) in zip(prev_poses, cur_poses):
        max_pos = max(max_pos, math.dist(p0, p1))
        max_ori = max(max_ori, _quat_angle_error(q0, q1))
    return max_pos, max_ori


def _wait_for_pose_settled(
    label,
    left_pos,
    left_quat,
    right_pos,
    right_quat,
    max_left_pos_err,
    max_left_quat_err,
    max_right_pos_err,
    max_right_quat_err,
    timeout=CONVERGE_TIMEOUT,
):
    """轮询 FK，等末端运动真正停稳（位置+姿态都不再变）后再返回，替代固定 settle。返回最后一次 fk。

    不在“位置进容差”就早退：对 right_yz_align 这种“位置先到、姿态还在大幅转”的动作，早退会把
    姿态没到位的坏构型交给下一步精确 IK 而失败。改为等运动停止——连续多次位置位移<EPS 且姿态
    变化<ORI_EPS 视为停稳。停稳或 timeout 才返回。负载轻早停、负载重多等，对实时率抖动免疫。
    """
    import rospy

    deadline = rospy.Time.now() + rospy.Duration(float(timeout))
    fk = _call_fk(_read_current_arm_joints(TOPIC_TIMEOUT), TOPIC_TIMEOUT)
    prev = _settle_eef_poses(fk, left_pos, right_pos)
    stable_hits = 0
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        rospy.sleep(CONVERGE_POLL_DT)
        fk = _call_fk(_read_current_arm_joints(TOPIC_TIMEOUT), TOPIC_TIMEOUT)
        cur = _settle_eef_poses(fk, left_pos, right_pos)
        moved_pos, moved_ori = _poses_max_delta(prev, cur)
        prev = cur
        if moved_pos < CONVERGE_STABLE_EPS and moved_ori < CONVERGE_STABLE_ORI_EPS:
            stable_hits += 1
            if stable_hits >= CONVERGE_STABLE_HITS:
                within = not _motion_needs_retry(
                    fk, left_pos, left_quat, right_pos, right_quat,
                    max_left_pos_err, max_left_quat_err, max_right_pos_err, max_right_quat_err,
                )
                rospy.loginfo(
                    "scene1 handoff: %s settled (%s)",
                    label,
                    "within tolerance" if within else "out of tolerance",
                )
                return fk
        else:
            stable_hits = 0
    rospy.logwarn("scene1 handoff: %s converge wait timed out after %.1fs", label, float(timeout))
    return fk


def _validate_ik_solution(
    label,
    ik_q,
    left_pos,
    left_quat,
    right_pos,
    right_quat,
    max_left_pos_err,
    max_left_quat_err,
    max_right_pos_err,
    max_right_quat_err,
):
    if not any(
        value is not None
        for value in (
            max_left_pos_err,
            max_left_quat_err,
            max_right_pos_err,
            max_right_quat_err,
        )
    ):
        return

    fk = _call_fk(ik_q, TOPIC_TIMEOUT)
    if left_pos is not None or left_quat is not None:
        pose = _pose_for_side(fk, "left")
        _log_and_guard_pose(
            f"{label}_ikfk",
            "left",
            list(pose.pos_xyz),
            left_pos,
            list(pose.quat_xyzw),
            left_quat,
            "ik_fk",
            max_left_pos_err,
            max_left_quat_err,
            raise_on_error=False,
        )
    if right_pos is not None or right_quat is not None:
        pose = _pose_for_side(fk, "right")
        _log_and_guard_pose(
            f"{label}_ikfk",
            "right",
            list(pose.pos_xyz),
            right_pos,
            list(pose.quat_xyzw),
            right_quat,
            "ik_fk",
            max_right_pos_err,
            max_right_quat_err,
            raise_on_error=False,
        )


def _new_two_arm_request(current_joint_values, fk_poses, constraint_mode,
                         pos_cost_weight, major_iterations_limit, joint_angles_as_q0):
    """构造 twoArmHandPoseCmd：两手先锁在当前 FK 位姿，调用方再覆盖要动的那只手。"""
    from kuavo_msgs.msg import twoArmHandPoseCmd

    request = twoArmHandPoseCmd()
    request.use_custom_ik_param = True
    request.joint_angles_as_q0 = bool(joint_angles_as_q0)
    request.ik_param = _make_scene1_ik_param(
        constraint_mode=constraint_mode,
        pos_cost_weight=pos_cost_weight,
        major_iterations_limit=major_iterations_limit,
    )
    request.hand_poses.left_pose.joint_angles = list(current_joint_values[:7])
    request.hand_poses.right_pose.joint_angles = list(current_joint_values[7:])
    request.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    _set_pose_target(request.hand_poses.left_pose,
                     fk_poses.left_pose.pos_xyz, fk_poses.left_pose.quat_xyzw)
    _set_pose_target(request.hand_poses.right_pose,
                     fk_poses.right_pose.pos_xyz, fk_poses.right_pose.quat_xyzw)
    return request


def _invoke_two_arm_ik(request, timeout, err_detail):
    """调用双臂 IK 服务并校验 success，返回 response。"""
    import rospy
    from kuavo_msgs.srv import twoArmHandPoseCmdSrv

    rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=timeout)
    response = rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)(request)
    if not response.success:
        raise RuntimeError(
            "/ik/two_arm_hand_pose_cmd_srv failed: "
            + getattr(response, "error_reason", "") + " " + err_detail
        )
    return response


def _call_hands_ik(
    current_joint_values,
    timeout,
    left_pos=None,
    left_quat=None,
    right_pos=None,
    right_quat=None,
    constraint_mode=SCENE1_IK_CONSTRAINT_MODE,
    pos_cost_weight=SCENE1_IK_POS_COST_WEIGHT,
    major_iterations_limit=SCENE1_IK_MAJOR_ITERATIONS_LIMIT,
):
    fk_poses = _call_fk(current_joint_values, timeout)
    request = _new_two_arm_request(current_joint_values, fk_poses, constraint_mode,
                                   pos_cost_weight, major_iterations_limit, True)
    cl, cr = fk_poses.left_pose, fk_poses.right_pose
    _set_pose_target(request.hand_poses.left_pose,
                     left_pos if left_pos is not None else cl.pos_xyz,
                     left_quat if left_quat is not None else cl.quat_xyzw)
    _set_pose_target(request.hand_poses.right_pose,
                     right_pos if right_pos is not None else cr.pos_xyz,
                     right_quat if right_quat is not None else cr.quat_xyzw)
    response = _invoke_two_arm_ik(request, timeout, f"left_pos={left_pos} right_pos={right_pos}")
    return _response_arm_joints(response)


def _call_one_hand_ik(
    current_joint_values,
    timeout,
    side,
    pos_xyz,
    quat_xyzw,
    constraint_mode=SCENE1_IK_CONSTRAINT_MODE,
    pos_cost_weight=SCENE1_IK_POS_COST_WEIGHT,
    major_iterations_limit=SCENE1_IK_MAJOR_ITERATIONS_LIMIT,
    joint_angles_as_q0=True,
):
    if side not in ("left", "right"):
        raise ValueError(f"unknown hand side: {side}")
    fk_poses = _call_fk(current_joint_values, timeout)
    request = _new_two_arm_request(current_joint_values, fk_poses, constraint_mode,
                                   pos_cost_weight, major_iterations_limit, joint_angles_as_q0)
    cur = fk_poses.left_pose if side == "left" else fk_poses.right_pose
    target_pose = request.hand_poses.left_pose if side == "left" else request.hand_poses.right_pose
    _set_pose_target(target_pose,
                     pos_xyz if pos_xyz is not None else cur.pos_xyz,
                     quat_xyzw if quat_xyzw is not None else cur.quat_xyzw)
    response = _invoke_two_arm_ik(request, timeout, f"side={side} pos={pos_xyz}")
    return _response_one_hand_joints(response, current_joint_values, side)


def _execute_ik_motion(
    arm_pub,
    arm_hold,
    left_pos=None,
    left_quat=None,
    right_pos=None,
    right_quat=None,
    label="ik",
    constraint_mode=SCENE1_IK_CONSTRAINT_MODE,
    pos_cost_weight=SCENE1_IK_POS_COST_WEIGHT,
    move_time=ARM_MOVE_TIME,
    settle_time=ARM_SETTLE_TIME,
    max_left_pos_err=None,
    max_left_quat_err=None,
    max_right_pos_err=None,
    max_right_quat_err=None,
    max_left_ikfk_pos_err=None,
    max_left_ikfk_quat_err=None,
    max_right_ikfk_pos_err=None,
    max_right_ikfk_quat_err=None,
    keep_other_hand_joints=True,
    ik_major_iterations_limit=SCENE1_IK_MAJOR_ITERATIONS_LIMIT,
):
    import rospy

    retries = CONTACT_RETRY_COUNT if any(
        value is not None
        for value in (
            max_left_pos_err,
            max_left_quat_err,
            max_right_pos_err,
            max_right_quat_err,
        )
    ) else 0
    cmd14 = None
    fk = None
    for attempt in range(retries + 1):
        current = _read_current_arm_joints(TOPIC_TIMEOUT)
        left_targeted = left_pos is not None or left_quat is not None
        right_targeted = right_pos is not None or right_quat is not None
        if keep_other_hand_joints and left_targeted and not right_targeted:
            ik_q = _call_one_hand_ik(
                current,
                TOPIC_TIMEOUT,
                "left",
                left_pos,
                left_quat,
                constraint_mode=constraint_mode,
                pos_cost_weight=pos_cost_weight,
                major_iterations_limit=ik_major_iterations_limit,
            )
        elif keep_other_hand_joints and right_targeted and not left_targeted:
            ik_q = _call_one_hand_ik(
                current,
                TOPIC_TIMEOUT,
                "right",
                right_pos,
                right_quat,
                constraint_mode=constraint_mode,
                pos_cost_weight=pos_cost_weight,
                major_iterations_limit=ik_major_iterations_limit,
            )
        else:
            ik_q = _call_hands_ik(
                current,
                TOPIC_TIMEOUT,
                left_pos=left_pos,
                left_quat=left_quat,
                right_pos=right_pos,
                right_quat=right_quat,
                constraint_mode=constraint_mode,
                pos_cost_weight=pos_cost_weight,
                major_iterations_limit=ik_major_iterations_limit,
            )
        _validate_ik_solution(
            label,
            ik_q,
            left_pos,
            left_quat,
            right_pos,
            right_quat,
            max_left_ikfk_pos_err,
            max_left_ikfk_quat_err,
            max_right_ikfk_pos_err,
            max_right_ikfk_quat_err,
        )
        cmd14 = _rad_to_deg(ik_q)
        # “等停稳”只用于精确步（设了容差 -> retries>0：抓取/放置/交接/二次夹取），保证下一步精确 IK
        # 从停稳构型起算；过渡动作（无容差）不等停稳、直接流过去，避免每步“走到→停死→再走”的顿挫。
        converging = WAIT_FOR_CONVERGENCE and retries > 0
        # 等收敛时把 chunked 末尾的固定 settle 设为 0，由轮询统一接管，避免重复空等。
        chunk_settle = 0.0 if converging else (
            settle_time if attempt == retries else CONTACT_RETRY_SETTLE_TIME
        )
        _execute_joint_motion_chunked(
            arm_pub,
            arm_hold,
            _rad_to_deg(current),
            cmd14,
            move_time,
            chunk_settle,
        )

        if converging:
            fk = _wait_for_pose_settled(
                label,
                left_pos,
                left_quat,
                right_pos,
                right_quat,
                max_left_pos_err,
                max_left_quat_err,
                max_right_pos_err,
                max_right_quat_err,
            )
        else:
            fk = _call_fk(_read_current_arm_joints(TOPIC_TIMEOUT), TOPIC_TIMEOUT)
        if not _motion_needs_retry(
            fk,
            left_pos,
            left_quat,
            right_pos,
            right_quat,
            max_left_pos_err,
            max_left_quat_err,
            max_right_pos_err,
            max_right_quat_err,
        ):
            break
        if attempt < retries:
            rospy.logwarn(
                "scene1 handoff: %s not converged after attempt %d/%d, recomputing IK from current joints",
                label,
                attempt + 1,
                retries + 1,
            )

    if left_pos is not None or left_quat is not None:
        actual_pos, actual_quat, actual_source = _actual_pose_for_side(fk, "left")
        _log_and_guard_pose(
            label,
            "left",
            actual_pos,
            left_pos,
            actual_quat,
            left_quat,
            actual_source,
            max_left_pos_err,
            max_left_quat_err,
            raise_on_error=RAISE_ON_RUNTIME_POSE_ERROR,
        )
        _log_tf_pose_offset(label, "left", actual_pos, left_pos)
    if right_pos is not None or right_quat is not None:
        actual_pos, actual_quat, actual_source = _actual_pose_for_side(fk, "right")
        _log_and_guard_pose(
            label,
            "right",
            actual_pos,
            right_pos,
            actual_quat,
            right_quat,
            actual_source,
            max_right_pos_err,
            max_right_quat_err,
            raise_on_error=RAISE_ON_RUNTIME_POSE_ERROR,
        )
        _log_tf_pose_offset(label, "right", actual_pos, right_pos)
    return cmd14


def _move_hand(
    side,
    arm_pub,
    arm_hold,
    pos_xyz,
    quat_xyzw,
    label,
    constraint_mode=SCENE1_IK_CONSTRAINT_MODE,
    pos_cost_weight=SCENE1_IK_POS_COST_WEIGHT,
    move_time=SCENE1_ARM_MOVE_TIME,
    settle_time=SCENE1_ARM_SETTLE_TIME,
    max_pos_err=None,
    max_quat_err=None,
    max_ikfk_pos_err=None,
    max_ikfk_quat_err=None,
    keep_other_hand_joints=True,
    ik_major_iterations_limit=SCENE1_IK_MAJOR_ITERATIONS_LIMIT,
):
    kwargs = {
        "left_pos": pos_xyz if side == "left" else None,
        "left_quat": quat_xyzw if side == "left" else None,
        "right_pos": pos_xyz if side == "right" else None,
        "right_quat": quat_xyzw if side == "right" else None,
    }
    return _execute_ik_motion(
        arm_pub,
        arm_hold,
        label=label,
        constraint_mode=constraint_mode,
        pos_cost_weight=pos_cost_weight,
        move_time=move_time,
        settle_time=settle_time,
        max_left_pos_err=max_pos_err if side == "left" else None,
        max_left_quat_err=max_quat_err if side == "left" else None,
        max_right_pos_err=max_pos_err if side == "right" else None,
        max_right_quat_err=max_quat_err if side == "right" else None,
        max_left_ikfk_pos_err=max_ikfk_pos_err if side == "left" else None,
        max_left_ikfk_quat_err=max_ikfk_quat_err if side == "left" else None,
        max_right_ikfk_pos_err=max_ikfk_pos_err if side == "right" else None,
        max_right_ikfk_quat_err=max_ikfk_quat_err if side == "right" else None,
        keep_other_hand_joints=keep_other_hand_joints,
        ik_major_iterations_limit=ik_major_iterations_limit,
        **kwargs,
    )


def _move_both_hands(
    arm_pub,
    arm_hold,
    left_pos,
    left_quat,
    right_pos,
    right_quat,
    label,
    constraint_mode=SCENE1_IK_CONSTRAINT_MODE,
    pos_cost_weight=SCENE1_IK_POS_COST_WEIGHT,
    move_time=SCENE1_ARM_MOVE_TIME,
    settle_time=SCENE1_ARM_SETTLE_TIME,
    max_left_pos_err=None,
    max_left_quat_err=None,
    max_right_pos_err=None,
    max_right_quat_err=None,
    max_left_ikfk_pos_err=None,
    max_left_ikfk_quat_err=None,
    max_right_ikfk_pos_err=None,
    max_right_ikfk_quat_err=None,
    ik_major_iterations_limit=SCENE1_IK_MAJOR_ITERATIONS_LIMIT,
):
    return _execute_ik_motion(
        arm_pub,
        arm_hold,
        left_pos=left_pos,
        left_quat=left_quat,
        right_pos=right_pos,
        right_quat=right_quat,
        label=label,
        constraint_mode=constraint_mode,
        pos_cost_weight=pos_cost_weight,
        move_time=move_time,
        settle_time=settle_time,
        max_left_pos_err=max_left_pos_err,
        max_left_quat_err=max_left_quat_err,
        max_right_pos_err=max_right_pos_err,
        max_right_quat_err=max_right_quat_err,
        max_left_ikfk_pos_err=max_left_ikfk_pos_err,
        max_left_ikfk_quat_err=max_left_ikfk_quat_err,
        max_right_ikfk_pos_err=max_right_ikfk_pos_err,
        max_right_ikfk_quat_err=max_right_ikfk_quat_err,
        ik_major_iterations_limit=ik_major_iterations_limit,
    )


# _move_hand_precise / _move_both_hands_precise 历史上与无后缀版完全等价（"precise" 仅在调用点
# 表意“这是精度步骤”），原本是逐参数转发的样板。这里降为别名：保留语义名和所有调用点不变，
# 但去掉 ~80 行重复转发。如需让精度步骤真正不同（更紧容差/更多迭代），再单独实现即可。
_move_hand_precise = _move_hand
_move_both_hands_precise = _move_both_hands


def _move_left_with_locked_right_joints(
    arm_pub,
    arm_hold,
    left_pos,
    left_quat,
    locked_right_degrees,
    label,
    constraint_mode=SCENE1_IK_CONSTRAINT_MODE,
    pos_cost_weight=SCENE1_IK_POS_COST_WEIGHT,
    move_time=SCENE1_ARM_MOVE_TIME,
    settle_time=SCENE1_ARM_SETTLE_TIME,
    max_left_ikfk_pos_err=None,
    max_left_ikfk_quat_err=None,
):
    """左手运动时，右手保持给定关节角，不再参与双臂 IK 重算。"""
    locked_right = [float(v) for v in locked_right_degrees]
    if len(locked_right) != 7:
        raise ValueError(f"locked_right_degrees expects 7 values, got {len(locked_right)}")

    current = _read_current_arm_joints(TOPIC_TIMEOUT)
    current_for_ik = list(current[:14])
    # 左手 IK 的 q0 也使用锁定右手，否则求解器会把右手下垂后的 FK 姿态当成约束。
    current_for_ik[7:14] = [math.radians(v) for v in locked_right]
    try:
        ik_q = _call_one_hand_ik(
            current_for_ik,
            TOPIC_TIMEOUT,
            "left",
            left_pos,
            left_quat,
            constraint_mode=constraint_mode,
            pos_cost_weight=pos_cost_weight,
            major_iterations_limit=SCENE1_IK_MAJOR_ITERATIONS_LIMIT,
            joint_angles_as_q0=True,
        )
    except RuntimeError as exc:
        import rospy

        rospy.logwarn(
            "scene1 handoff: %s IK failed from current q0, retrying without q0 lock: %s",
            label,
            exc,
        )
        ik_q = _call_one_hand_ik(
            current_for_ik,
            TOPIC_TIMEOUT,
            "left",
            left_pos,
            left_quat,
            constraint_mode=constraint_mode,
            pos_cost_weight=pos_cost_weight,
            major_iterations_limit=max(SCENE1_IK_MAJOR_ITERATIONS_LIMIT, 500),
            joint_angles_as_q0=False,
        )
    _validate_ik_solution(
        label,
        ik_q,
        left_pos,
        left_quat,
        None,
        None,
        max_left_ikfk_pos_err,
        max_left_ikfk_quat_err,
        None,
        None,
    )

    cmd14 = _rad_to_deg(ik_q)
    cmd14[7:14] = locked_right
    _execute_joint_motion_chunked(
        arm_pub,
        arm_hold,
        _rad_to_deg(current),
        cmd14,
        move_time,
        settle_time,
    )

    fk = _call_fk(_read_current_arm_joints(TOPIC_TIMEOUT), TOPIC_TIMEOUT)
    left_actual_pos, left_actual_quat, left_source = _actual_pose_for_side(fk, "left")
    _log_and_guard_pose(
        label,
        "left",
        left_actual_pos,
        left_pos,
        left_actual_quat,
        left_quat,
        left_source,
        None,
        None,
        raise_on_error=False,
    )
    _log_tf_pose_offset(label, "left", left_actual_pos, left_pos)

    right_actual_pos, right_actual_quat, right_source = _actual_pose_for_side(fk, "right")
    _log_and_guard_pose(
        f"{label}_right_locked",
        "right",
        right_actual_pos,
        None,
        right_actual_quat,
        None,
        right_source,
        None,
        None,
        raise_on_error=False,
    )
    _log_tf_pose_offset(f"{label}_right_locked", "right", right_actual_pos, None)
    return cmd14


def _parcel_jobs(seed, selected_names):
    layout = _load_scene1_layout(seed)
    jobs = []
    for name in selected_names:
        if name not in layout:
            raise ValueError(f"unknown parcel {name}; available: {', '.join(PARCEL_NAMES)}")
        source_world = list(layout[name]["pos"])
        jobs.append(
            {
                "object": name,
                "source_world": source_world,
                "source_ik": _world_to_ik(source_world),
                "right_pick_offset": _right_pick_offset_for_parcel(name, source_world),
                "right_pick_ik": _right_pick_ik_from_source_world(name, source_world),
                # 称重区和箱子是固定场景点，不从 seed layout 里取。
                "weigh_ik": list(WEIGH_RELEASE_IK),
                "box_drop_offset": _box_drop_offset_for_parcel(name),
                "box_ik": _box_drop_ik_for_parcel(name),
            }
        )
    return jobs


def _right_grasp_descend(arm_pub, arm_hold, target_ik, right_quat, label):
    """最终抓取：右手单手解。先用 0x03(位姿都硬)精确对正夹爪；IK 解不出再回退 0x06 三点式保底。
    左手不参与本步(保持当前关节)，避免其不可达朝向连累右手硬解；依赖上一步预抓取位的暖启动。"""
    import rospy

    common = dict(
        pos_cost_weight=RIGHT_PICK_POS_COST_WEIGHT,
        move_time=PICK_GRASP_MOVE_TIME,
        settle_time=SCENE1_ARM_SETTLE_TIME,
        max_pos_err=TRANSIT_POS_TOLERANCE,
        max_ikfk_pos_err=TRANSIT_POS_TOLERANCE,
        ik_major_iterations_limit=RIGHT_PICK_IK_MAJOR_ITERATIONS_LIMIT,
    )
    try:
        return _move_hand("right", arm_pub, arm_hold, target_ik, right_quat,
                          f"{label}_right_grasp_hard",
                          constraint_mode=RIGHT_GRASP_FINAL_CONSTRAINT_MODE, **common)
    except RuntimeError as exc:
        rospy.logwarn(
            "scene1 handoff: %s 0x03 hard grasp IK failed (%s); fall back to 0x06 three-point",
            label, exc,
        )
        return _move_hand("right", arm_pub, arm_hold, target_ik, right_quat,
                          f"{label}_right_grasp_fallback",
                          constraint_mode=RIGHT_PICK_IK_CONSTRAINT_MODE, **common)


def _left_wait_and_right_pick_from(
    arm_pub,
    arm_hold,
    gripper_hold,
    target_ik,
    right_quat,
    label,
    carry_quat=None,
):
    import rospy

    right_pre = _right_pick_pre_ik(target_ik)
    current = _read_current_arm_joints(TOPIC_TIMEOUT)
    current_right = _call_fk(current, TOPIC_TIMEOUT).right_pose.pos_xyz
    right_yz_align = _right_pick_yz_align_ik(current_right, right_pre)

    gripper_hold.set_right_open()
    # 左臂全程保持在 move_home 的预设位不动；pick/称重只动右手，
    # 左手要到交接阶段(右手把包裹递给左手)才开始动。
    _move_hand(
        "right",
        arm_pub,
        arm_hold,
        right_yz_align,
        right_quat,
        f"{label}_right_yz_align",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
        move_time=PICK_ALIGN_MOVE_TIME,
        settle_time=SCENE1_ARM_SETTLE_TIME,
        max_pos_err=TRANSIT_POS_TOLERANCE,
    )
    _move_hand(
        "right",
        arm_pub,
        arm_hold,
        right_pre,
        right_quat,
        f"{label}_right_x_to_pick_pre",
        constraint_mode=RIGHT_PICK_IK_CONSTRAINT_MODE,
        pos_cost_weight=RIGHT_PICK_POS_COST_WEIGHT,
        move_time=SCENE1_ARM_MOVE_TIME,
        settle_time=SCENE1_ARM_SETTLE_TIME,
        max_pos_err=TRANSIT_POS_TOLERANCE,
        ik_major_iterations_limit=RIGHT_PICK_IK_MAJOR_ITERATIONS_LIMIT,
    )
    if USE_HARD_FINAL_GRASP:
        # 左臂保持在预设位不动；右手单手暖启动硬解抓取(左手关节就地锁定)，失败自动回退三点式。
        _right_grasp_descend(arm_pub, arm_hold, target_ik, right_quat, label)
    else:
        _move_both_hands_precise(
            arm_pub,
            arm_hold,
            left_pos=LEFT_PRESET_2_IK,
            left_quat=LEFT_PRESET_2_QUAT_XYZW,
            right_pos=target_ik,
            right_quat=right_quat,
            label=f"{label}_left_preset_2_right_grasp",
            constraint_mode=RIGHT_PICK_IK_CONSTRAINT_MODE,
            pos_cost_weight=RIGHT_PICK_POS_COST_WEIGHT,
            move_time=PICK_GRASP_MOVE_TIME,
            settle_time=SCENE1_ARM_SETTLE_TIME,
            max_left_pos_err=TRANSIT_POS_TOLERANCE,
            max_left_quat_err=TRANSIT_ORI_TOLERANCE_RAD,
            max_right_pos_err=TRANSIT_POS_TOLERANCE,
            max_right_ikfk_pos_err=TRANSIT_POS_TOLERANCE,
            ik_major_iterations_limit=RIGHT_PICK_IK_MAJOR_ITERATIONS_LIMIT,
        )
    gripper_hold.set_right_closed()
    rospy.sleep(GRIPPER_CLOSE_HOLD_TIME)
    # 抬起：笛卡尔竖直直线上抬(保持抓取 x/y，只升 z)，且一步直接升到"搬运高度"
    # (= 后续横移高度 WEIGH_TRANSIT_IK_Z)。同时把朝向在这段竖直抬起里切到"搬运/释放朝向"
    # carry_quat，这样接下来的横移段全程"同一四元素 + 同一高度"纯水平移动，最平滑。
    lift_quat = carry_quat if carry_quat is not None else right_quat
    lift_ik = _with_ik_z(target_ik, WEIGH_TRANSIT_IK_Z)
    _move_right_cartesian_to(
        arm_pub,
        arm_hold,
        lift_ik,
        lift_quat,
        f"{label}_lift_straight_up",
        n_points=4,
    )
    return lift_ik


def _weigh_transit_ik(weigh_ik):
    return _with_ik_z(weigh_ik, WEIGH_TRANSIT_IK_Z)


def _weigh_release_pre_ik(weigh_ik):
    # 释放前高位点：x/y 与释放点一致，z 保持称重平移高度。
    # 在这个点先切到释放四元素，下一步再只下降 z。
    return _weigh_transit_ik(weigh_ik)


def _weigh_regrasp_ik(_weigh_ik):
    return list(WEIGH_REGRASP_IK)


def _weigh_regrasp_pre_ik(weigh_ik):
    # 二次抓取前点：x/y 与二次抓取点一致，z 沿用 2.2 释放高度。
    # 2.2 已经把包裹放在称重区；这里不再额外抬手，只切到二次抓取姿态后下降。
    regrasp = _weigh_regrasp_ik(weigh_ik)
    return _with_ik_z(regrasp, weigh_ik[2])


def _right_handoff_xy_ori_align_ik():
    # 右手交接前过渡点：先抬到固定高位，再对齐交接点 x/y 和四元素。
    # 下一步再只对齐 RIGHT_HANDOFF_TO_LEFT_IK 的最终 z。
    return _right_handoff_xy_ori_align_ik_at_z(RIGHT_HANDOFF_TRANSIT_IK_Z)


def _right_handoff_xy_ori_align_ik_at_z(ik_z):
    return [
        float(RIGHT_HANDOFF_TO_LEFT_IK[0]),
        float(RIGHT_HANDOFF_TO_LEFT_IK[1]),
        float(ik_z),
    ]


def _right_handoff_release_retract_ik():
    # 右手松开后保持交接高度和姿态，只沿 y 退让，避免左手去箱子时撞右手。
    return [
        float(RIGHT_HANDOFF_TO_LEFT_IK[0]),
        float(RIGHT_HANDOFF_TO_LEFT_IK[1]) + float(RIGHT_HANDOFF_RELEASE_RETRACT_Y_OFFSET),
        float(RIGHT_HANDOFF_TO_LEFT_IK[2]),
    ]


def _current_right_handoff_release_retract_pose():
    # 交接受力后右手可能会比理论 3.3 点低。退让时基于当前 FK，
    # 只改 y，不再把右手拉回理论高度，避免松开后出现第二次下坠/回弹。
    current = _read_current_arm_joints(TOPIC_TIMEOUT)
    right_pose = _call_fk(current, TOPIC_TIMEOUT).right_pose
    retract_pos = [
        float(right_pose.pos_xyz[0]),
        float(right_pose.pos_xyz[1]) + float(RIGHT_HANDOFF_RELEASE_RETRACT_Y_OFFSET),
        float(right_pose.pos_xyz[2]),
    ]
    return retract_pos, list(right_pose.quat_xyzw)


def _move_right_cartesian_to(arm_pub, arm_hold, target_ik, quat, label,
                             n_points=6, seg_time=0.4,
                             constraint_mode=IK_MODE_POS_HARD_ORI_HARD,
                             settle_time=SCENE1_ARM_SETTLE_TIME):
    """右手沿笛卡尔直线移动到 target_ik:沿直线等分采样 → 逐点单手IK预解 → 复用样条连续流式下发。

    每个采样点位置在直线上(位置硬约束),所以末端真正走直线、z 不塌陷,
    避免两端点之间关节插值导致的中途下沉/低空横扫其它包裹。左臂关节全程保持不动。
    """
    import rospy

    current = _read_current_arm_joints(TOPIC_TIMEOUT)
    start_ik = list(_call_fk(current, TOPIC_TIMEOUT).right_pose.pos_xyz)
    end = [float(v) for v in target_ik]
    joint_wps = [_rad_to_deg(current)]
    warm = current
    for k in range(1, int(n_points) + 1):
        a = float(k) / float(n_points)
        pt = [start_ik[i] + (end[i] - start_ik[i]) * a for i in range(3)]
        ik_q = _call_one_hand_ik(
            warm, TOPIC_TIMEOUT, "right", pt, quat,
            constraint_mode=constraint_mode,
            pos_cost_weight=RIGHT_PICK_POS_COST_WEIGHT,
            major_iterations_limit=SCENE1_IK_MAJOR_ITERATIONS_LIMIT,
        )
        joint_wps.append(_rad_to_deg(ik_q))
        warm = ik_q
    _publish_preset_spline(arm_hold, joint_wps, seg_time)
    rospy.sleep(float(settle_time))
    rospy.loginfo("scene1 handoff: %s 笛卡尔等高横移完成 (%d 段)", label, int(n_points))


def _move_right_to_weigh_release_pre(
    arm_pub,
    arm_hold,
    weigh_ik,
    release_quat,
    carry_quat=None,
):
    # 抬起后沿笛卡尔直线等高(略上行)直达称重释放点正上方,z 全程不塌陷、不低空横扫其它包裹;
    # 下一步 right_weigh_down 再只竖直下降放到称重区 → 整体"↑ 平移 ↓"门字形。
    # 横移保持"搬运/抓取朝向"(top-down) carry_quat:对任意可达位置都好解(不像释放前倾朝向在
    # 近排高位解不出),且姿态硬→四元素恒定;释放朝向只在 right_weigh_down 那一步切。
    transit_quat = carry_quat if carry_quat is not None else release_quat
    _move_right_cartesian_to(
        arm_pub,
        arm_hold,
        _weigh_release_pre_ik(weigh_ik),
        transit_quat,
        "right_weigh_transit",
    )


def _right_weigh_and_regrasp(
    arm_pub,
    arm_hold,
    gripper_hold,
    parcel_name,
    weigh_ik,
    release_quat,
    regrasp_quat,
    dwell,
    verify_object_pose=False,
    carry_quat=None,
):
    import rospy

    _move_right_to_weigh_release_pre(
        arm_pub,
        arm_hold,
        weigh_ik,
        release_quat,
        carry_quat=carry_quat,
    )
    _move_hand_precise(
        "right",
        arm_pub,
        arm_hold,
        weigh_ik,
        release_quat,
        "right_weigh_down",
    )
    rospy.loginfo("scene1 handoff: right hand at weighing release point, settle %.2fs before opening", WEIGH_RELEASE_SETTLE_BEFORE_OPEN)
    rospy.sleep(float(WEIGH_RELEASE_SETTLE_BEFORE_OPEN))
    gripper_hold.set_right_open()
    rospy.loginfo("scene1 handoff: parcel released on weighing area, dwell %.2fs", dwell)
    rospy.sleep(float(dwell))
    if verify_object_pose:
        _assert_parcel_on_weighing_area(parcel_name)

    # 二次夹取实时对准:读包裹在称重区的真实落点(称重放置有抖动),瞄那里夹,沿用二次夹取
    # 高度 z=WEIGH_REGRASP_IK[2]。这样不管称重落得正不正,二次夹取都对准实际包裹、夹得牢、
    # 交接不掉包。读失败退回固定点。对任意 seed 通用(称重区固定但落点随抓取/负载抖)。
    try:
        regrasp_world = _mujoco_body_world_pos(parcel_name, timeout=TOPIC_TIMEOUT)
        regrasp_ik = _world_to_ik(regrasp_world)
        regrasp_ik[2] = float(WEIGH_REGRASP_IK[2])
        rospy.loginfo(
            "scene1 handoff: %s 二次夹取实时对准 world=%s -> ik=%s",
            parcel_name, [round(v, 4) for v in regrasp_world], [round(v, 4) for v in regrasp_ik],
        )
    except Exception as exc:
        regrasp_ik = list(WEIGH_REGRASP_IK)
        rospy.logwarn(
            "scene1 handoff: %s 二次夹取实时读位失败(%s),退回固定点 %s",
            parcel_name, exc, [round(v, 4) for v in regrasp_ik],
        )
    regrasp_pre_ik = _with_ik_z(regrasp_ik, weigh_ik[2])

    regrasp_pre_cmd14 = _move_hand(
        "right",
        arm_pub,
        arm_hold,
        regrasp_pre_ik,
        regrasp_quat,
        "right_regrasp_xy_ori_align",
        settle_time=0.0,
    )
    _move_hand_precise(
        "right",
        arm_pub,
        arm_hold,
        regrasp_ik,
        regrasp_quat,
        "right_regrasp_from_weigh",
    )
    gripper_hold.set_right_closed()
    rospy.sleep(GRIPPER_CLOSE_HOLD_TIME)
    # 高位二次抓取点刚刚已经 IK 成功过。夹住包裹后物理 FK/姿态会有小偏差，
    # 此时重新 IK 可能失败；直接复用刚才的高位关节解，更稳定地抬回同一位置。
    current = _read_current_arm_joints(TOPIC_TIMEOUT)
    _execute_joint_motion_chunked(
        arm_pub,
        arm_hold,
        _rad_to_deg(current),
        regrasp_pre_cmd14,
        SCENE1_ARM_MOVE_TIME,
        SCENE1_ARM_SETTLE_TIME,
    )
    fk = _call_fk(_read_current_arm_joints(TOPIC_TIMEOUT), TOPIC_TIMEOUT)
    actual_pos, actual_quat, actual_source = _actual_pose_for_side(fk, "right")
    _log_and_guard_pose(
        "right_lift_from_weigh",
        "right",
        actual_pos,
        regrasp_pre_ik,
        actual_quat,
        regrasp_quat,
        actual_source,
        None,
        None,
        raise_on_error=False,
    )
    _log_tf_pose_offset("right_lift_from_weigh", "right", actual_pos, regrasp_pre_ik)


def _move_to_recorded_handoff_presets(arm_pub, arm_hold, gripper_hold):
    import rospy

    gripper_hold.set_left_open()
    handoff_transit_zs = [RIGHT_HANDOFF_TRANSIT_IK_Z] + list(RIGHT_HANDOFF_TRANSIT_FALLBACK_IK_ZS)
    for transit_z in handoff_transit_zs:
        try:
            current = _read_current_arm_joints(TOPIC_TIMEOUT)
            current_right_pose = _call_fk(current, TOPIC_TIMEOUT).right_pose
            right_handoff_raise = [
                float(current_right_pose.pos_xyz[0]),
                float(current_right_pose.pos_xyz[1]),
                float(transit_z),
            ]
            _move_hand_precise(
                "right",
                arm_pub,
                arm_hold,
                right_handoff_raise,
                list(current_right_pose.quat_xyzw),
                "right_handoff_raise_before_xy",
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
                move_time=SCENE1_ARM_MOVE_TIME,
                settle_time=SCENE1_ARM_SETTLE_TIME,
                max_ikfk_pos_err=HANDOFF_POS_TOLERANCE,
                max_ikfk_quat_err=HANDOFF_ORI_TOLERANCE_RAD,
            )
            _move_hand_precise(
                "right",
                arm_pub,
                arm_hold,
                _right_handoff_xy_ori_align_ik_at_z(transit_z),
                RIGHT_HANDOFF_TO_LEFT_QUAT_XYZW,
                "right_handoff_xy_ori_align",
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
                move_time=HANDOFF_MOVE_TIME,
                settle_time=SCENE1_ARM_SETTLE_TIME,
                max_ikfk_pos_err=HANDOFF_POS_TOLERANCE,
                max_ikfk_quat_err=HANDOFF_ORI_TOLERANCE_RAD,
            )
            if transit_z != RIGHT_HANDOFF_TRANSIT_IK_Z:
                rospy.logwarn(
                    "scene1 handoff: right_handoff_xy_ori_align fallback transit_z=%.3f",
                    transit_z,
                )
            break
        except RuntimeError as exc:
            if transit_z == handoff_transit_zs[-1]:
                raise
            rospy.logwarn(
                "scene1 handoff: right_handoff_raise/xy_ori_align z=%.3f failed, retry lower: %s",
                transit_z,
                exc,
            )
    right_handoff_cmd14 = _move_hand_precise(
        "right",
        arm_pub,
        arm_hold,
        RIGHT_HANDOFF_TO_LEFT_IK,
        RIGHT_HANDOFF_TO_LEFT_QUAT_XYZW,
        "right_handoff_to_left",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
        move_time=HANDOFF_MOVE_TIME,
        settle_time=SCENE1_ARM_SETTLE_TIME,
        max_ikfk_pos_err=HANDOFF_POS_TOLERANCE,
        max_ikfk_quat_err=HANDOFF_ORI_TOLERANCE_RAD,
    )
    # 右手夹着包裹到 3.3 后 FK/TF 可能被负载压低；不要把下沉后的实际关节
    # 重新当作锁定目标，否则左手接近时右手会被“锁”在更低的位置。
    locked_right_degrees = list(right_handoff_cmd14[7:14])
    rospy.loginfo("scene1 handoff: target right handoff joints locked before left receive")
    _move_left_with_locked_right_joints(
        arm_pub,
        arm_hold,
        left_pos=LEFT_HANDOFF_RECEIVE_XZ_READY_IK,
        left_quat=LEFT_HANDOFF_RECEIVE_QUAT_XYZW,
        locked_right_degrees=locked_right_degrees,
        label="left_receive_xz_ready_keep_right",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
        move_time=SCENE1_ARM_MOVE_TIME,
        settle_time=SCENE1_ARM_SETTLE_TIME,
        max_left_ikfk_pos_err=HANDOFF_POS_TOLERANCE,
        max_left_ikfk_quat_err=HANDOFF_ORI_TOLERANCE_RAD,
    )
    _move_left_with_locked_right_joints(
        arm_pub,
        arm_hold,
        left_pos=LEFT_HANDOFF_RECEIVE_IK,
        left_quat=LEFT_HANDOFF_RECEIVE_QUAT_XYZW,
        locked_right_degrees=locked_right_degrees,
        label="left_receive_from_right_y_keep_right",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
        move_time=HANDOFF_MOVE_TIME,
        settle_time=SCENE1_ARM_SETTLE_TIME,
        max_left_ikfk_pos_err=HANDOFF_POS_TOLERANCE,
        max_left_ikfk_quat_err=HANDOFF_ORI_TOLERANCE_RAD,
    )
    gripper_hold.set_left_closed()
    rospy.sleep(GRIPPER_CLOSE_HOLD_TIME)
    gripper_hold.set_right_open()
    rospy.sleep(GRIPPER_CLOSE_HOLD_TIME)
    right_retract_pos, right_retract_quat = _current_right_handoff_release_retract_pose()
    _move_hand(
        "right",
        arm_pub,
        arm_hold,
        right_retract_pos,
        right_retract_quat,
        "right_retract_after_handoff_release",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
        move_time=SCENE1_ARM_MOVE_TIME,
        settle_time=SCENE1_ARM_SETTLE_TIME,
    )


def _wait_for_gripper_open(side, timeout=GRIPPER_OPEN_WAIT_TIMEOUT):
    import rospy
    from sensor_msgs.msg import JointState

    joint_name = f"{side}_gripper_joint"
    deadline = rospy.Time.now() + rospy.Duration(float(timeout))
    last_position = None
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        try:
            msg = rospy.wait_for_message("/gripper/state", JointState, timeout=0.2)
        except rospy.ROSException:
            continue
        if joint_name not in msg.name:
            continue
        index = msg.name.index(joint_name)
        if index >= len(msg.position):
            continue
        last_position = float(msg.position[index])
        if abs(last_position - LEFT_GRIPPER_OPEN) <= GRIPPER_OPEN_STATE_TOLERANCE:
            rospy.loginfo(
                "scene1 handoff: %s gripper open confirmed, state=%.4f",
                side,
                last_position,
            )
            return True

    rospy.logwarn(
        "scene1 handoff: %s gripper open not confirmed within %.1fs, last_state=%s",
        side,
        float(timeout),
        "unknown" if last_position is None else f"{last_position:.4f}",
    )
    return False


def _open_left_gripper_at_box_drop(gripper_hold, parcel_name, verify_object_pose=False):
    import rospy

    gripper_hold.set_left_open()
    rospy.loginfo("scene1 handoff: left gripper opened at box drop point")
    _wait_for_gripper_open("left")
    rospy.sleep(PLACE_DWELL)
    if verify_object_pose:
        _assert_parcel_in_box(parcel_name)


def _left_drop_in_box(
    arm_pub,
    arm_hold,
    gripper_hold,
    parcel_name,
    box_ik,
    left_quat,
    retract=True,
    verify_object_pose=False,
):
    import rospy

    # 左臂够箱子在工作空间边缘，放置预备/下放点偶尔 IK 解不出(out of workspace)。
    # 仅在 IK 失败时把放置点 x 往机器人侧收一点(减小 x → 更易达)重试；首个 delta=0 即原值，
    # 没失败就用原值、什么都不改。收得太多会落到箱近沿外(入箱检查会判失败)，故 delta 表幅度有限。
    box_x_deltas = [0.0] + list(BOX_DROP_IK_X_FALLBACK_DELTAS)
    for i, dx in enumerate(box_x_deltas):
        box_ik_try = [float(box_ik[0]) + float(dx), float(box_ik[1]), float(box_ik[2])]
        try:
            current = _read_current_arm_joints(TOPIC_TIMEOUT)
            current_left = _call_fk(current, TOPIC_TIMEOUT).left_pose.pos_xyz
            box_raise = _left_box_raise_before_xy_ik(current_left, box_ik_try)
            box_pre = _left_box_pre_ik(box_ik_try)
            _move_hand(
                "left", arm_pub, arm_hold, box_raise, left_quat,
                "left_box_raise_before_xy",
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
                move_time=SCENE1_ARM_MOVE_TIME, settle_time=BOX_TRANSIT_SETTLE_TIME,
            )
            _move_hand(
                "left", arm_pub, arm_hold, box_pre, left_quat,
                "left_box_pre",
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
                move_time=PLACE_MOVE_TIME, settle_time=BOX_TRANSIT_SETTLE_TIME,
            )
            _move_hand(
                "left", arm_pub, arm_hold, box_ik_try, left_quat,
                "left_box_drop",
                constraint_mode=IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
                move_time=PLACE_MOVE_TIME,
            )
            if dx != 0.0:
                rospy.logwarn(
                    "scene1 handoff: left box place 用减小 x 回退成功 x_delta=%.3f box_ik=%s",
                    dx, [round(v, 4) for v in box_ik_try],
                )
            break
        except RuntimeError as exc:
            if i == len(box_x_deltas) - 1:
                raise
            rospy.logwarn(
                "scene1 handoff: left box place IK 失败 x_delta=%.3f (%s)；减小 x 重试", dx, exc,
            )
    _open_left_gripper_at_box_drop(gripper_hold, parcel_name, verify_object_pose=verify_object_pose)
    if not retract:
        rospy.loginfo("scene1 handoff: stopping at box drop point, no retract/home/auto-swing")
        return
    _move_hand(
        "left",
        arm_pub,
        arm_hold,
        LEFT_PRESET_2_IK,
        LEFT_PRESET_2_QUAT_XYZW,
        "left_box_retract_to_preset_2",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=HANDOFF_POS_COST_WEIGHT,
        move_time=PLACE_MOVE_TIME,
        settle_time=SCENE1_ARM_SETTLE_TIME,
        max_pos_err=TRANSIT_POS_TOLERANCE,
        max_quat_err=TRANSIT_ORI_TOLERANCE_RAD,
    )


class BehaviorAction:
    def __init__(self, name, action):
        self.name = name
        self._action = action

    def tick(self, blackboard):
        import rospy

        _set_progress(stage=self.name)
        rospy.loginfo("scene1 behavior: start %s", self.name)
        result = self._action(blackboard)
        if result is False:
            rospy.logwarn("scene1 behavior: failed %s", self.name)
            return False
        rospy.loginfo("scene1 behavior: success %s", self.name)
        return True


class BehaviorSequence:
    def __init__(self, name, children):
        self.name = name
        self.children = list(children)

    def tick(self, blackboard):
        import rospy

        rospy.loginfo("scene1 behavior: enter %s", self.name)
        for child in self.children:
            if not child.tick(blackboard):
                rospy.logwarn("scene1 behavior: abort %s at %s", self.name, child.name)
                return False
        rospy.loginfo("scene1 behavior: leave %s", self.name)
        return True


def _refresh_job_from_live_pose(job, args):
    """抓取前用 /mujoco/qpos 实时回读包裹真实 world 位置，重算抓取点。

    种子布局值是“理论摆放位”，物体落桌静置后可能微动；开启 --realtime-pick 后
    这里读运行时真值刷新 source_world/source_ik/right_pick_ik，让抓取跟随实际静置位。
    抓取高度 right_pick_ik[2] 仍由固定 RIGHT_PICK_IK_Z + 偏移决定（不取物体 z）。
    回读失败时保留原种子值并告警，不打断采集。
    """
    import rospy

    name = job["object"]
    try:
        live_world = _mujoco_body_world_pos(name, timeout=TOPIC_TIMEOUT)
    except Exception as exc:
        rospy.logwarn(
            "scene1 handoff: %s realtime pose readback failed (%s); fall back to seed layout %s",
            name,
            exc,
            [round(v, 4) for v in job["source_world"]],
        )
        return job

    seed_world = job["source_world"]
    job["source_world"] = list(live_world)
    job["source_ik"] = _world_to_ik(live_world)
    job["right_pick_ik"] = _right_pick_ik_from_source_world(name, live_world)
    rospy.loginfo(
        "scene1 handoff: %s realtime pick refresh seed_world=%s -> live_world=%s (dxy=%.4f, dz=%.4f)",
        name,
        [round(v, 4) for v in seed_world],
        [round(v, 4) for v in live_world],
        math.hypot(live_world[0] - seed_world[0], live_world[1] - seed_world[1]),
        live_world[2] - seed_world[2],
    )
    return job


def _run_scene1_job_behavior(arm_pub, arm_hold, gripper_hold, jobs, args, blackboard, index, job):
    import rospy

    _set_progress(stage=f"{job['object']}: start", job=job["object"])
    if getattr(args, "realtime_pick", False):
        job = _refresh_job_from_live_pose(job, args)
    right_pick_quat = args.right_pick_quat_xyzw
    right_weigh_release_quat = args.right_weigh_release_quat_xyzw
    right_weigh_regrasp_quat = args.right_weigh_regrasp_quat_xyzw

    rospy.loginfo(
        "scene1 handoff: job %d/%d %s source_world=%s source_ik=%s right_pick_offset=%s right_pick_ik=%s weigh_ik=%s box_drop_offset=%s box_ik=%s",
        index,
        len(jobs),
        job["object"],
        [round(v, 4) for v in job["source_world"]],
        [round(v, 4) for v in job["source_ik"]],
        [round(v, 4) for v in job["right_pick_offset"]],
        [round(v, 4) for v in job["right_pick_ik"]],
        [round(v, 4) for v in job["weigh_ik"]],
        [round(v, 4) for v in job["box_drop_offset"]],
        [round(v, 4) for v in job["box_ik"]],
    )
    _set_progress(stage=f"{job['object']}: pick", job=job["object"])
    _left_wait_and_right_pick_from(
        arm_pub,
        arm_hold,
        gripper_hold,
        job["right_pick_ik"],
        right_pick_quat,
        job["object"],
        carry_quat=right_pick_quat,
    )

    _set_progress(stage=f"{job['object']}: weigh_and_regrasp", job=job["object"])
    _right_weigh_and_regrasp(
        arm_pub,
        arm_hold,
        gripper_hold,
        job["object"],
        job["weigh_ik"],
        right_weigh_release_quat,
        right_weigh_regrasp_quat,
        args.weigh_dwell,
        verify_object_pose=args.debug_verify_object_pose,
        carry_quat=right_pick_quat,
    )
    rospy.loginfo(
        "scene1 handoff: weighing regrasp complete; moving to recorded handoff presets"
    )
    _set_progress(stage=f"{job['object']}: handoff", job=job["object"])
    _move_to_recorded_handoff_presets(arm_pub, arm_hold, gripper_hold)
    place_quat = LEFT_BOX_DROP_QUAT_XYZW

    is_last_job = index == len(jobs)
    _set_progress(stage=f"{job['object']}: box_drop", job=job["object"])
    _left_drop_in_box(
        arm_pub,
        arm_hold,
        gripper_hold,
        job["object"],
        job["box_ik"],
        place_quat,
        retract=not (blackboard["leave_at_box_drop"] and is_last_job),
        verify_object_pose=args.debug_verify_object_pose,
    )
    return True


def _build_scene1_behavior_tree(arm_pub, arm_hold, gripper_hold, jobs, args, on_preset_done=None):
    blackboard = {
        "leave_at_box_drop": True,
    }

    def _open_grippers(_blackboard):
        gripper_hold.set_open()
        return True

    def _move_home(_blackboard):
        # 把双臂运动到预设位,再进入抓取等后续步骤。
        _move_through_preset(arm_pub, arm_hold)
        return True

    def _finish(_blackboard):
        # 最后运动完成后，用 arm_target_poses 把双臂送回预设位（PRESET_POINTS_DEG 末点）。
        # 这样一条 demo 起于预设、终于预设，首尾姿态一致。
        gripper_hold.set_open()
        _move_arm_to(arm_pub, arm_hold, PRESET_POINTS_DEG[-1])
        return True

    children = [
        BehaviorAction("open_grippers", _open_grippers),
        BehaviorAction("move_home", _move_home),
    ]
    if on_preset_done is not None:
        # 到达预设位之后、开始抓取正逆解之前才起录 rosbag（不录启动/抬手过程）。
        children.append(BehaviorAction("start_record_after_preset", lambda _bb: on_preset_done()))
    for index, job in enumerate(jobs, start=1):
        children.append(
            BehaviorAction(
                f"process_{job['object']}",
                lambda blackboard, index=index, job=job: _run_scene1_job_behavior(
                    arm_pub,
                    arm_hold,
                    gripper_hold,
                    jobs,
                    args,
                    blackboard,
                    index,
                    job,
                ),
            )
        )
    children.append(BehaviorAction("finish", _finish))
    return BehaviorSequence("scene1_handoff_root", children), blackboard


def _run_scene1_motion(arm_pub, arm_hold, gripper_hold, jobs, args, on_preset_done=None):
    tree, blackboard = _build_scene1_behavior_tree(
        arm_pub, arm_hold, gripper_hold, jobs, args, on_preset_done=on_preset_done
    )
    if not tree.tick(blackboard):
        raise RuntimeError("scene1 behavior tree failed")
    return blackboard["leave_at_box_drop"]


def _start_rosbag(bag_path, topics):
    if bag_path is None:
        return None, []
    cmd = ["rosbag", "record", "-O", bag_path] + topics
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, cmd


def _seed_for_run(args, run_index):
    return int(args.seed) + int(run_index) - 1


def _bag_output_paths(bag_path):
    if bag_path is None:
        return []
    # rosbag records to *.active first, then renames to *.bag on clean SIGINT.
    return [bag_path, f"{bag_path}.active"]


def _unlock_path_for_host(path):
    if not path or not os.path.exists(path):
        return
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        try:
            parent = os.path.dirname(os.path.abspath(path)) or "."
            parent_stat = os.stat(parent)
            os.chown(path, parent_stat.st_uid, parent_stat.st_gid)
        except Exception as exc:
            print(f"[WARN] scene1 bag permission chown skipped for {path}: {exc}", file=sys.stderr)
    try:
        os.chmod(path, 0o666)
    except Exception as exc:
        print(f"[WARN] scene1 bag permission chmod skipped for {path}: {exc}", file=sys.stderr)


def _unlock_bag_outputs(bag_path):
    for path in _bag_output_paths(bag_path):
        _unlock_path_for_host(path)


def _delete_bag_outputs(bag_path):
    deleted = []
    for path in _bag_output_paths(bag_path):
        if not path or not os.path.exists(path):
            continue
        try:
            os.remove(path)
            deleted.append(path)
        except Exception as exc:
            print(f"[WARN] scene1 failed bag delete skipped for {path}: {exc}", file=sys.stderr)
    if deleted:
        print(f"[INFO] scene1 handoff: deleted incomplete bag output: {', '.join(deleted)}")


def _append_failed_seed(args, run_index, run_total, reason, seed=None, attempts=None):
    failed_seeds_file = getattr(args, "failed_seeds_file", None)
    if not failed_seeds_file:
        return
    try:
        os.makedirs(os.path.dirname(failed_seeds_file), exist_ok=True)
        seed = int(args.seed if seed is None else seed)
        parcels = ",".join(getattr(args, "parcels", None) or [])
        attempts_text = "" if attempts is None else f"\tattempts={int(attempts)}"
        line = (
            f"{seed}\t{_datetime.datetime.now().isoformat(timespec='seconds')}"
            f"\trun={int(run_index)}/{int(run_total)}\tparcels={parcels}{attempts_text}\treason={reason}\n"
        )
        with open(failed_seeds_file, "a", encoding="utf-8") as handle:
            handle.write(line)
        _unlock_path_for_host(failed_seeds_file)
    except Exception as exc:
        print(f"[WARN] scene1 failed seed log skipped: {exc}", file=sys.stderr)


def _append_success_manifest(args, bag_path, run_index, run_total, seed=None, attempts=None):
    """成功采集清单：每条成功（bag 已保留）的 run 追加一行，便于事后核对采了哪些、多少条。"""
    manifest_file = getattr(args, "success_manifest_file", None)
    if not manifest_file:
        return
    try:
        os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
        seed = int(args.seed if seed is None else seed)
        bag_name = os.path.basename(bag_path) if bag_path else "(no-bag)"
        parcels = ",".join(args.parcels)
        attempts_text = "" if attempts is None else f"\tattempts={int(attempts)}"
        line = (
            f"{seed}\t{_datetime.datetime.now().isoformat(timespec='seconds')}"
            f"\tbag={bag_name}\tparcels={parcels}\trun={int(run_index)}/{int(run_total)}{attempts_text}\n"
        )
        with open(manifest_file, "a", encoding="utf-8") as handle:
            handle.write(line)
        _unlock_path_for_host(manifest_file)
    except Exception as exc:
        print(f"[WARN] scene1 success manifest log skipped: {exc}", file=sys.stderr)


def _run_once(args, run_index=1, run_total=1):
    if args.headless:
        os.environ["MUJOCO_HEADLESS"] = "1"
    _set_progress(
        seed=int(args.seed),
        run_index=int(run_index),
        run_total=int(run_total),
        attempt_index=int(getattr(args, "attempt_index", 1)),
        attempt_total=int(getattr(args, "attempt_total", 1)),
        stage="run_start",
        job=None,
    )

    # bag 命名：{scene}_seed_{seed}_{时间戳}.bag（不补零、不带 handoff/run/attempt 后缀）。
    # 同一 seed 的重试是顺序进行且失败 bag 会被删，时间戳到秒已足够区分，不会冲突。
    run_tag = f"{SCENE_NAME}_seed_{int(args.seed)}_{_now_tag()}"
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    try:
        os.chmod(output_dir, 0o777)
    except Exception as exc:
        print(f"[WARN] scene1 output dir chmod skipped for {output_dir}: {exc}", file=sys.stderr)

    bag_path = None if args.no_rosbag else os.path.join(output_dir, f"{run_tag}.bag")
    topics = list(DEFAULT_TOPICS)
    jobs = _parcel_jobs(args.seed, args.parcels)

    launch_proc = None
    bag_proc = None
    gripper_hold = None
    arm_hold = None
    arm_mode_changed = False
    status = "failed"
    failure_reason = "unknown"
    interrupted = False
    try:
        if args.use_existing_sim:
            _set_progress(stage="attach_existing_sim")
            _init_ros_node("scene1_handoff_dataset_collector")
            missing = _wait_for_topics(["/sensors_data_raw"], LAUNCH_TIMEOUT)
            if missing:
                raise RuntimeError("existing simulation is not ready: " + ", ".join(missing))
        else:
            _set_progress(stage="start_scene_launch")
            launch_proc = _start_scene_launch(args.seed)

        _set_progress(stage="wait_required_topics")
        topic_wait_timeout = TOPIC_TIMEOUT if args.use_existing_sim else LAUNCH_TIMEOUT
        missing_topics = _wait_for_topics(["/sensors_data_raw"], topic_wait_timeout)
        if missing_topics:
            raise RuntimeError("required topics missing: " + ", ".join(missing_topics))

        _set_progress(stage="prepare_ros_publishers")
        _publish_head_target(TOPIC_TIMEOUT)
        gripper_hold = _start_gripper_hold(TOPIC_TIMEOUT)
        arm_hold = _start_arm_traj_hold(TOPIC_TIMEOUT)

        _set_progress(stage="set_external_control")
        _set_arm_mode(ARM_MODE_EXTERNAL_CONTROL, timeout=TOPIC_TIMEOUT)
        arm_mode_changed = True
        from kuavo_msgs.msg import armTargetPoses
        import rospy

        arm_pub = rospy.Publisher(ARM_TARGET_POSES_TOPIC, armTargetPoses, queue_size=10)
        _wait_for_connection(arm_pub, TOPIC_TIMEOUT)

        def _start_record_after_preset():
            # 机器人抬到预设位(move_home)之后、开始抓取正逆解之前才起录 rosbag，
            # 这样 bag 只含从预设姿态开始的任务动作，不录启动与抬手过程。
            nonlocal bag_proc
            if bag_path is not None:
                _set_progress(stage="start_rosbag")
                bag_proc, _ = _start_rosbag(bag_path, topics)
                time.sleep(1.0 + RECORD_SETTLE_TIME)
            return True

        _set_progress(stage="scene1_motion")
        if getattr(args, "preset_only", False):
            # 调试模式: 只跑预设抬手轨迹, 然后停住观察, 不进入抓取流程。
            gripper_hold.set_open()
            _move_through_preset(
                arm_pub, arm_hold, max_points=getattr(args, "preset_waypoints", None)
            )
            _hold_for_observation(getattr(args, "hold_seconds", 15.0))
            keep_external_control = True
        else:
            keep_external_control = _run_scene1_motion(
                arm_pub, arm_hold, gripper_hold, jobs, args,
                on_preset_done=_start_record_after_preset,
            )
        if keep_external_control:
            rospy.loginfo("scene1 handoff: keeping external control after box drop")
            arm_mode_changed = False
        else:
            _set_arm_mode(ARM_MODE_AUTO_SWING, timeout=TOPIC_TIMEOUT)
            arm_mode_changed = False

        if bag_path is not None:
            _set_progress(stage="post_record_wait")
            time.sleep(POST_ARM_MODE_RECORD_TIME)
        status = "success"
        failure_reason = ""
        _set_progress(stage="success")
    except KeyboardInterrupt:
        interrupted = True
        failure_reason = _progress_reason("KeyboardInterrupt")
        print(f"[WARN] scene1 handoff dataset interrupted: {failure_reason}", file=sys.stderr)
        raise
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        raise
    finally:
        _terminate_process_group(bag_proc, signal.SIGINT, timeout=10)
        if status == "success":
            if bag_path is not None:
                _unlock_bag_outputs(bag_path)
                _append_success_manifest(args, bag_path, run_index, run_total)
        else:
            if bag_path is not None:
                _delete_bag_outputs(bag_path)
            if interrupted or not getattr(args, "defer_failed_seed_log", False):
                _append_failed_seed(args, run_index, run_total, failure_reason)
        if gripper_hold is not None:
            gripper_hold.stop()
        if arm_hold is not None:
            arm_hold.stop()
        if arm_mode_changed:
            try:
                _set_arm_mode(ARM_MODE_AUTO_SWING, timeout=TOPIC_TIMEOUT)
            except Exception as exc:
                print(f"[WARN] failed to restore arm mode: {exc}")
        _stop_scene_launch(launch_proc)
        print(f"[INFO] scene1 handoff dataset run {status}: {bag_path or '(rosbag disabled)'}")

    return 0 if status == "success" else 1


def _child_args_for_run(args, run_index, run_total, attempt_index=1, attempt_total=1):
    run_seed = _seed_for_run(args, run_index)
    child_args = [
        sys.executable,
        os.path.abspath(__file__),
        "--output-dir", args.output_dir,
        "--failed-seeds-file", args.failed_seeds_file,
        "--seed", str(run_seed),
        "--count", "1",
        "--run-index", str(run_index),
        "--run-total", str(run_total),
        "--attempt-index", str(attempt_index),
        "--attempt-total", str(attempt_total),
        "--max-seed-attempts", str(args.max_seed_attempts),
        "--defer-failed-seed-log",
        "--weigh-dwell", str(args.weigh_dwell),
        "--right-ypr-deg", args.right_ypr_deg,
        "--right-second-ypr-deg", args.right_second_ypr_deg,
        "--right-weigh-ypr-deg", args.right_weigh_ypr_deg,
        "--right-weigh-second-ypr-deg", args.right_weigh_second_ypr_deg,
        "--right-regrasp-ypr-deg", args.right_regrasp_ypr_deg,
        "--right-regrasp-second-ypr-deg", args.right_regrasp_second_ypr_deg,
    ]
    for parcel in args.parcels:
        child_args += ["--parcel", parcel]
    if args.headless:
        child_args.append("--headless")
    if args.no_rosbag:
        child_args.append("--no-rosbag")
    if args.debug_verify_object_pose:
        child_args.append("--verify-object-pose")
    return child_args


def _run_child_attempt(child_args):
    proc = subprocess.Popen(child_args)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                return proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    proc.terminate()
                    return proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                return 130
        return proc.returncode if proc.returncode is not None else 130


def _is_interrupt_returncode(returncode):
    return returncode in (130, -signal.SIGINT)


def _run_count(args):
    if args.count < 1:
        raise ValueError("--count must be >= 1")
    max_attempts = int(args.max_seed_attempts) if not args.no_rosbag else 1
    if args.defer_failed_seed_log or (args.count == 1 and max_attempts == 1):
        return _run_once(args, args.run_index, args.run_total)
    if args.use_existing_sim:
        raise ValueError("batch/retry runs cannot be used with --use-existing-sim")

    failed = []
    for index in range(1, args.count + 1):
        run_seed = _seed_for_run(args, index)
        _set_progress(
            seed=run_seed,
            run_index=index,
            run_total=args.count,
            attempt_index=None,
            attempt_total=max_attempts,
            stage="seed_start",
            job=None,
        )
        print(
            f"[INFO] scene1 handoff dataset batch {index}/{args.count}: "
            f"seed={run_seed} start, max_attempts={max_attempts}"
        )
        last_reason = "unknown"
        for attempt in range(1, max_attempts + 1):
            _set_progress(
                seed=run_seed,
                run_index=index,
                run_total=args.count,
                attempt_index=attempt,
                attempt_total=max_attempts,
                stage="subprocess_attempt",
                job=None,
            )
            print(
                f"[INFO] scene1 handoff dataset seed={run_seed} "
                f"attempt {attempt}/{max_attempts} start"
            )
            returncode = _run_child_attempt(
                _child_args_for_run(args, index, args.count, attempt, max_attempts)
            )
            if returncode == 0:
                print(
                    f"[INFO] scene1 handoff dataset seed={run_seed} "
                    f"attempt {attempt}/{max_attempts} complete"
                )
                break
            if _is_interrupt_returncode(returncode):
                print(
                    f"[WARN] scene1 handoff dataset interrupted at seed={run_seed} "
                    f"attempt {attempt}/{max_attempts}; exiting",
                    file=sys.stderr,
                )
                return 130
            last_reason = f"exit_code={returncode} at attempt {attempt}/{max_attempts}"
            print(
                f"[ERROR] scene1 handoff dataset seed={run_seed} "
                f"attempt {attempt}/{max_attempts} failed with code {returncode}",
                file=sys.stderr,
            )
        else:
            failed.append(run_seed)
            _append_failed_seed(
                args,
                index,
                args.count,
                last_reason,
                seed=run_seed,
                attempts=max_attempts,
            )
            print(
                f"[ERROR] scene1 handoff dataset seed={run_seed} skipped after "
                f"{max_attempts} failed attempts; continuing",
                file=sys.stderr,
            )
            continue
        print(f"[INFO] scene1 handoff dataset batch {index}/{args.count}: seed={run_seed} complete")
    if failed:
        print(
            f"[ERROR] scene1 handoff dataset batch complete with failed seeds: "
            f"{', '.join(str(seed) for seed in failed)}",
            file=sys.stderr,
        )
        print(f"[ERROR] failed seed log: {args.failed_seeds_file}", file=sys.stderr)
        return 1
    return 0


def _rounded(values, digits=4):
    return [round(float(value), digits) for value in values]


def _print_plan(args):
    jobs = _parcel_jobs(args.seed, args.parcels)
    print(f"scene={SCENE_NAME} seed={args.seed}")
    print(f"output_dir={args.output_dir}")
    print(f"failed_seeds_file={args.failed_seeds_file}")
    print(f"verify_object_pose={args.debug_verify_object_pose}")
    print(f"max_seed_attempts={args.max_seed_attempts}  # 仅录 bag 时生效；失败会重启同一 seed")
    if args.count > 1:
        print(f"batch_seed_range={args.seed}..{_seed_for_run(args, args.count)} count={args.count}")
    print(f"world_to_ik_offset={_rounded(WORLD_TO_IK_OFFSET)}")
    print("姿态参数：")
    print(f"  right_pick_first_ypr_deg={_rounded(args.right_ypr_deg_xyz, 3)}")
    print(f"  right_pick_second_ypr_deg={_rounded(args.right_second_ypr_deg_xyz, 3)}")
    print(f"  right_weigh_release_first_ypr_deg={_rounded(args.right_weigh_release_ypr_deg_xyz, 3)}")
    print(f"  right_weigh_release_second_ypr_deg={_rounded(args.right_weigh_release_second_ypr_deg_xyz, 3)}")
    print(f"  right_weigh_regrasp_first_ypr_deg={_rounded(args.right_weigh_regrasp_ypr_deg_xyz, 3)}")
    print(f"  right_weigh_regrasp_second_ypr_deg={_rounded(args.right_weigh_regrasp_second_ypr_deg_xyz, 3)}")
    print(f"  left_preset_2_first_ypr_deg={_rounded(LEFT_PRESET_2_FIRST_YPR_DEG, 3)}")
    print(f"  left_preset_2_second_ypr_deg={_rounded(LEFT_PRESET_2_SECOND_YPR_DEG, 3)}")
    print(f"  right_handoff_first_ypr_deg={_rounded(RIGHT_HANDOFF_TO_LEFT_FIRST_YPR_DEG, 3)}")
    print(f"  right_handoff_second_ypr_deg={_rounded(RIGHT_HANDOFF_TO_LEFT_SECOND_YPR_DEG, 3)}")
    print(f"  left_receive_first_ypr_deg={_rounded(LEFT_HANDOFF_RECEIVE_FIRST_YPR_DEG, 3)}")
    print(f"  left_receive_second_ypr_deg={_rounded(LEFT_HANDOFF_RECEIVE_SECOND_YPR_DEG, 3)}")
    print(f"  left_box_drop_first_ypr_deg={_rounded(LEFT_BOX_DROP_FIRST_YPR_DEG, 3)}")
    print(f"  left_box_drop_second_ypr_deg={_rounded(LEFT_BOX_DROP_SECOND_YPR_DEG, 3)}")
    print("称重区调试参数：")
    print(f"  right_lift_z_offset={LIFT_Z_OFFSET}")
    print(f"  weighing_center_world={_rounded(WEIGHING_CENTER_WORLD)}  # 场景参考，不直接作为执行目标")
    print(f"  weigh_release_ik={_rounded(WEIGH_RELEASE_IK)}")
    print(f"  weigh_regrasp_ik={_rounded(WEIGH_REGRASP_IK)}")
    print(f"  weigh_transit_ik_z={WEIGH_TRANSIT_IK_Z}  # 2.1 称重释放前预备高度")
    print("高度参数：")
    print(f"  place_approach_z_offset={PLACE_APPROACH_Z_OFFSET}")
    print(f"  right_pick_transit_ik_z={RIGHT_PICK_TRANSIT_IK_Z}  # 从当前/预设位置到抓取上方的横移高度")
    print(f"  right_pick_ik_z={RIGHT_PICK_IK_Z}  # 最终抓取高度")
    print(f"  right_pick_offset far_row={_rounded(RIGHT_PICK_OFFSET_FAR_ROW)} "
          f"near_row={_rounded(RIGHT_PICK_OFFSET_NEAR_ROW)} "
          f"near_far_y_thresh={RIGHT_PICK_NEAR_FAR_Y_THRESHOLD}  # [x, y, z微调]")
    print("  right_pick_offset_by_parcel(实际解析):")
    for name in PARCEL_NAMES:
        print(f"    {name}: {_rounded(_right_pick_offset_for_parcel(name))}")
    print("运动时序：")
    print(f"  scene1_arm_move_time={SCENE1_ARM_MOVE_TIME}")
    print(f"  pick_align_move_time={PICK_ALIGN_MOVE_TIME}  # 抓取前预设/安全对齐")
    print(f"  pick_grasp_move_time={PICK_GRASP_MOVE_TIME}  # 右手下降到包裹抓取点")
    print(f"  handoff_move_time={HANDOFF_MOVE_TIME}  # 交接相关动作")
    print(f"  place_move_time={PLACE_MOVE_TIME}")
    print(f"  weigh_transit_settle_time={WEIGH_TRANSIT_SETTLE_TIME}  # 2.1 上方中间点停顿")
    print(f"  weigh_release_settle_before_open={WEIGH_RELEASE_SETTLE_BEFORE_OPEN}  # 2.2 松夹爪前稳定")
    print(f"  box_transit_settle_time={BOX_TRANSIT_SETTLE_TIME}  # 放箱上方中间点停顿")
    print(f"  place_dwell={PLACE_DWELL}  # 左手开夹爪后等待")
    print(f"  max_joint_step_deg={MAX_JOINT_STEP_DEG}")
    print(f"  min_joint_segment_time={MIN_JOINT_SEGMENT_TIME}")
    print("IK 参数：")
    print(f"  constraint_mode=0x{SCENE1_IK_CONSTRAINT_MODE:02x}")
    print(f"  pos_cost_weight={SCENE1_IK_POS_COST_WEIGHT}")
    print(f"  major_iterations_limit={SCENE1_IK_MAJOR_ITERATIONS_LIMIT}")
    print("固定点位：")
    print(f"  left_preset_2_ik={_rounded(LEFT_PRESET_2_IK)}  # 左手第二等待/回收点")
    print(f"  weigh_release_ik={_rounded(WEIGH_RELEASE_IK)}  # 2.2 释放")
    print(f"  weigh_regrasp_ik={_rounded(WEIGH_REGRASP_IK)}  # 2.3 二次抓取")
    print(f"  right_handoff_ik={_rounded(RIGHT_HANDOFF_TO_LEFT_IK)}  # 右手递给左手")
    print(f"  right_handoff_transit_ik_z={RIGHT_HANDOFF_TRANSIT_IK_Z}  # 3.3 右手交接前高位")
    print(f"  left_receive_ik={_rounded(LEFT_HANDOFF_RECEIVE_IK)}  # 左手接右手快递")
    print(f"  right_handoff_release_retract_nominal_ik={_rounded(_right_handoff_release_retract_ik())}  # 3.5 名义退让点，运行时按当前 FK 退")
    print(f"  right_handoff_release_retract_y_offset={RIGHT_HANDOFF_RELEASE_RETRACT_Y_OFFSET}")
    print(f"  box_drop_ik={_rounded(BOX_DROP_IK)}  # 左手放箱基准点")
    print("  box_drop_offset_by_parcel:")
    for name in PARCEL_NAMES:
        print(f"    {name}: {_rounded(_box_drop_offset_for_parcel(name))}")
    print("执行顺序：")
    for job in jobs:
        right_pick_pre = _right_pick_pre_ik(job["right_pick_ik"])
        right_pick_lift = _right_pick_lift_ik(job["right_pick_ik"])
        weigh_release_pre = _weigh_release_pre_ik(job["weigh_ik"])
        weigh_regrasp = _weigh_regrasp_ik(job["weigh_ik"])
        weigh_regrasp_pre = _weigh_regrasp_pre_ik(job["weigh_ik"])
        right_handoff_xy_ori_align = _right_handoff_xy_ori_align_ik()
        right_handoff_release_retract = _right_handoff_release_retract_ik()
        box_pre = _left_box_pre_ik(job["box_ik"])
        print(f"  {job['object']}:")
        print(
            f"    放箱偏移: offset={_rounded(job['box_drop_offset'])}, "
            f"box_ik={_rounded(job['box_ik'])}"
        )
        print(
            f"    1 左臂保持预设位不动；右手按当前 FK 做 y 安全对齐并保持较高 z: "
            f"right=[safe_x, {right_pick_pre[1]:.4f}, max(current_z, {right_pick_pre[2]:.4f})]"
        )
        print(f"    1.5 右手横向到抓取上方: right={_rounded(right_pick_pre)}")
        print(
            f"    2 右手下降夹快递（左臂仍保持预设位）: right={_rounded(job['right_pick_ik'])}"
        )
        print(f"    3 右手夹紧后抬高: right={_rounded(right_pick_lift)}")
        print(
            f"    4 右手先在当前 x/y 对到称重搬运高度，再保持该高度横移到释放点上方并切释放四元素: "
            f"right=[当前x, 当前y, {WEIGH_TRANSIT_IK_Z:.4f}] -> {_rounded(weigh_release_pre)}"
        )
        print(f"    5 右手只下降 z 放到称重区并等待: right={_rounded(job['weigh_ik'])}")
        print(
            f"    6 右手沿用 2.2 释放高度对齐二次抓取 x/y 和四元素，再只下降 z 夹起快递: "
            f"right={_rounded(weigh_regrasp_pre)} -> {_rounded(weigh_regrasp)}"
        )
        print(
            f"    7 右手先在当前 x/y 原地升到 z={RIGHT_HANDOFF_TRANSIT_IK_Z:.3f}，"
            f"再高位横移对齐交接 x/y 和四元素，最后只对准最终 z: "
            f"right=[当前x, 当前y, {RIGHT_HANDOFF_TRANSIT_IK_Z:.4f}] -> {_rounded(right_handoff_xy_ori_align)} -> "
            f"{_rounded(RIGHT_HANDOFF_TO_LEFT_IK)}"
        )
        print(
            f"    8 左手先对齐 x/z 和姿态，再沿 y 接近并夹住快递: "
            f"left={_rounded(LEFT_HANDOFF_RECEIVE_XZ_READY_IK)} -> {_rounded(LEFT_HANDOFF_RECEIVE_IK)}"
        )
        print(f"    9 右手松开后沿 y 退让: 名义 right={_rounded(right_handoff_release_retract)}，运行时按当前右手 FK 退")
        print(
            f"    10 左手先在当前 x/y 原地抬到箱口预备高度并切放箱姿态，"
            f"再横移到箱子上方: left=[当前x, 当前y, {box_pre[2]:.4f}] -> {_rounded(box_pre)}"
        )
        print(f"    11 左手只下降 z 放箱并打开夹爪: left={_rounded(job['box_ik'])}")
        print(
            f"    12 左手直接回第二等待点: "
            f"left={_rounded(LEFT_PRESET_2_IK)}"
        )
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect scene1 right-weigh left-place handoff rosbag datasets.")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of simulation restart-and-record cycles to run. With count > 1, seeds increment from --seed.",
    )
    parser.add_argument("--run-index", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--run-total", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--attempt-index", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--attempt-total", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--defer-failed-seed-log", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for bag files.")
    parser.add_argument(
        "--failed-seeds-file",
        default=None,
        help="Txt file for failed seeds. Relative paths are resolved under --output-dir.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Scene random seed used for layout and launch.")
    parser.add_argument(
        "--max-seed-attempts",
        type=int,
        default=DEFAULT_MAX_SEED_ATTEMPTS,
        help="When recording rosbag, retry the same seed this many times before logging and skipping it.",
    )
    parser.add_argument("--parcel", dest="parcels", action="append", choices=PARCEL_NAMES, help="Parcel to process; repeatable.")
    parser.add_argument("--max-parcels", type=int, default=None, help="Process only the first N selected parcels.")
    parser.add_argument("--weigh-dwell", type=float, default=WEIGH_DWELL, help="Seconds to leave the parcel on the weighing area.")
    parser.add_argument("--right-ypr-deg", default=" ".join(str(v) for v in RIGHT_PICK_FIRST_YPR_DEG),
                        help="Right pick first Euler set, same as arm_control.py: yaw pitch roll in degrees.")
    parser.add_argument("--right-second-ypr-deg", default=" ".join(str(v) for v in RIGHT_PICK_SECOND_YPR_DEG),
                        help="Right pick second/manual Euler set, same as arm_control.py: yaw pitch roll in degrees.")
    parser.add_argument("--right-weigh-ypr-deg", default=" ".join(str(v) for v in RIGHT_WEIGH_RELEASE_FIRST_YPR_DEG),
                        help="Right weighing release first Euler set: yaw pitch roll in degrees.")
    parser.add_argument("--right-weigh-second-ypr-deg", default=" ".join(str(v) for v in RIGHT_WEIGH_RELEASE_SECOND_YPR_DEG),
                        help="Right weighing release second/manual Euler set: yaw pitch roll in degrees.")
    parser.add_argument("--right-regrasp-ypr-deg", default=" ".join(str(v) for v in RIGHT_WEIGH_REGRASP_FIRST_YPR_DEG),
                        help="Right weighing regrasp first Euler set: yaw pitch roll in degrees.")
    parser.add_argument("--right-regrasp-second-ypr-deg", default=" ".join(str(v) for v in RIGHT_WEIGH_REGRASP_SECOND_YPR_DEG),
                        help="Right weighing regrasp second/manual Euler set: yaw pitch roll in degrees.")
    parser.add_argument("--headless", action="store_true", help="Set MUJOCO_HEADLESS=1 for this run.")
    parser.add_argument("--use-existing-sim", action="store_true", help="Attach to an already running scene1 simulation.")
    parser.add_argument("--no-rosbag", action="store_true", help="Run the motion without recording a bag.")
    parser.add_argument(
        "--debug-verify-object-pose",
        "--verify-object-pose",
        dest="debug_verify_object_pose",
        action="store_true",
        help=(
            "Enable pose checks when --no-rosbag is used. Bag recording enables this automatically: "
            "only runs where every parcel passes weighing and box checks keep the bag. "
            "The qpos topic is not added to rosbag topics."
        ),
    )
    parser.add_argument("--print-plan", action="store_true", help="Print seed-derived target points without launching ROS.")
    parser.add_argument(
        "--no-realtime-pick",
        dest="realtime_pick",
        action="store_false",
        help=(
            "Disable live grasp-target refresh and fall back to the seed-computed layout. "
            "By default each parcel grasp target is refreshed from live /mujoco/qpos right "
            "before picking (tracks the object's actual settled x/y; grasp height stays fixed)."
        ),
    )
    parser.set_defaults(realtime_pick=True)
    # ===== 调试: 单独调预设轨迹 =====
    parser.add_argument(
        "--preset-only",
        action="store_true",
        help="只跑预设抬手轨迹(move_home),然后停住观察,不进入抓取流程。",
    )
    parser.add_argument(
        "--preset-waypoints",
        type=int,
        default=None,
        help="配合 --preset-only: 只执行前 N 个预设路点(例如 2 = 只走外展张开那段)。",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=15.0,
        help="调试模式跑完后停住观察的时长(秒)。",
    )
    return parser


def _parse_ypr_arg(args, attr, name):
    value = _parse_float_list(getattr(args, attr), 3, name)
    setattr(args, f"{attr}_xyz", value)
    return value


def _normalize_args(args):
    args.output_dir = os.path.abspath(args.output_dir)
    if args.failed_seeds_file is None:
        args.failed_seeds_file = os.path.join(args.output_dir, DEFAULT_FAILED_SEEDS_FILE)
    elif not os.path.isabs(args.failed_seeds_file):
        args.failed_seeds_file = os.path.abspath(os.path.join(args.output_dir, args.failed_seeds_file))
    else:
        args.failed_seeds_file = os.path.abspath(args.failed_seeds_file)
    # 成功采集清单与 failed_seeds 同目录（output_dir），子进程同样 output_dir → 同一份清单。
    args.success_manifest_file = os.path.join(args.output_dir, DEFAULT_SUCCESS_MANIFEST_FILE)
    if args.max_seed_attempts < 1:
        raise ValueError("--max-seed-attempts must be >= 1")
    if args.attempt_index < 1 or args.attempt_total < 1:
        raise ValueError("internal attempt counters must be >= 1")
    # 录 bag 就必须做位置检查：只有所有包裹称重检查和入箱检查都通过，bag 才会保留。
    # --verify-object-pose 只用于 --no-rosbag 调试时手动开启同一套检查。
    if not args.no_rosbag:
        args.debug_verify_object_pose = True
    if args.parcels is None:
        args.parcels = list(PARCEL_NAMES)
    if args.max_parcels is not None:
        if args.max_parcels < 1:
            raise ValueError("--max-parcels must be >= 1")
        args.parcels = args.parcels[: args.max_parcels]
    args.right_ypr_deg_xyz = _parse_ypr_arg(
        args,
        "right_ypr_deg",
        "--right-ypr-deg",
    )
    args.right_second_ypr_deg_xyz = _parse_ypr_arg(
        args,
        "right_second_ypr_deg",
        "--right-second-ypr-deg",
    )
    args.right_weigh_release_ypr_deg_xyz = _parse_ypr_arg(
        args,
        "right_weigh_ypr_deg",
        "--right-weigh-ypr-deg",
    )
    args.right_weigh_release_second_ypr_deg_xyz = _parse_ypr_arg(
        args,
        "right_weigh_second_ypr_deg",
        "--right-weigh-second-ypr-deg",
    )
    args.right_weigh_regrasp_ypr_deg_xyz = _parse_ypr_arg(
        args,
        "right_regrasp_ypr_deg",
        "--right-regrasp-ypr-deg",
    )
    args.right_weigh_regrasp_second_ypr_deg_xyz = _parse_ypr_arg(
        args,
        "right_regrasp_second_ypr_deg",
        "--right-regrasp-second-ypr-deg",
    )
    args.right_pick_quat_xyzw = _quat_from_ypr_deg(
        args.right_ypr_deg_xyz,
        args.right_second_ypr_deg_xyz,
    )
    args.right_weigh_release_quat_xyzw = _quat_from_ypr_deg(
        args.right_weigh_release_ypr_deg_xyz,
        args.right_weigh_release_second_ypr_deg_xyz,
    )
    args.right_weigh_regrasp_quat_xyzw = _quat_from_ypr_deg(
        args.right_weigh_regrasp_ypr_deg_xyz,
        args.right_weigh_regrasp_second_ypr_deg_xyz,
    )
    return args


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        args = _normalize_args(args)
        if args.print_plan:
            return _print_plan(args)
        return _run_count(args)
    except KeyboardInterrupt:
        print(
            f"[WARN] scene1 handoff dataset interrupted by Ctrl+C: {_progress_reason('KeyboardInterrupt')}",
            file=sys.stderr,
        )
        return 130
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
