#!/usr/bin/env python3
"""Scene2 自动分拣数采主流程。

职责：
1. 启动指定 seed 的 Scene2 仿真，并读取 challenge_task.py 导出的随机布局。
2. 按配置顺序完成六个工件抓取、换手、放置。
3. 可选录制 rosbag，并在结束后检查分拣结果和相机话题频率。
4. 支持 repeat/auto 模式，失败的 rosbag 会丢弃，不计入有效采集数量。
"""

import argparse
import datetime as _datetime
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

from scene2_part_grasp_ik import (
    GraspRuntime,
    configure_scene2_layout,
    execute_part_grasp,
    get_handoff_transition_config,
    get_object_arm,
    get_object_lift_quat_xyzw,
    get_object_world_xyz,
    measure_hand_pose,
    move_arm_ik_once,
    rad_to_deg,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
SCENE_NAME = "scene2"
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "bags", SCENE_NAME)

# challenge_task.py 每次启动 scene2 时会生成 active XML，成功检查直接读这个 XML 的 qpos 地址。
ACTIVE_SCENE_XML = os.path.join(
    REPO_ROOT,
    "src",
    "challenge_cup_simulator",
    "models",
    "biped_s52",
    "xml",
    "_scene_scene2_active.xml",
)
CHALLENGE_TASK_SCRIPT = os.path.join(SCRIPT_DIR, "challenge_task.py")

ARM_JOINT_NAMES = ["arm_joint_" + str(i) for i in range(1, 15)]
# 训练用 rosbag 录制话题。相机话题后续会做最低频率检查，避免低帧率数据混入训练集。
ROSBAG_TOPICS = [
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
ROSBAG_CAMERA_TOPIC_KEYWORD = "cam"
ROSBAG_CAMERA_MIN_HZ = 25.0

PART_TO_BIN = {
    "part_type_a_1": "sorting_bin_a",
    "part_type_a_2": "sorting_bin_a",
    "part_type_b_1": "sorting_bin_b",
    "part_type_b_2": "sorting_bin_b",
    "part_type_c_1": "sorting_bin_c",
    "part_type_c_2": "sorting_bin_c",
}
PART_NAMES = list(PART_TO_BIN.keys())
BIN_NAMES = ["sorting_bin_a", "sorting_bin_b", "sorting_bin_c"]

# ---------------------------------------------------------------------------
# Joint-space poses (degrees). Order: left arm joint 1-7, right arm joint 8-14.
# ---------------------------------------------------------------------------
# Arms raised to the sides before moving down to WORK_POSE (avoids hitting table).
SIDE_LIFT_JOINTS_DEG = [
    20, 70, 0, -35, 0, 0, 0,
    20, -70, 0, -35, 0, 0, 0,
]

# Both hands ~10 cm above the table, level, one on each side.
WORK_POSE_JOINTS_DEG = [
    30, 10, 10, -120, -60, 0, 0,
    30, -10, -10, -120, 60, 0, 0,
]

HOME_JOINTS_DEG = [
    20, 0, 0, -30, 0, 0, 0,
    20, 0, 0, -30, 0, 0, 0,
]

# 放置阶段不用 IK：只配置活动臂 7 个关节角（degree）。
# 约定：sorting_bin_a 在机器人右侧，sorting_bin_b 在中间，sorting_bin_c 在机器人左侧。
# 非活动臂统一锁到 WORK_POSE_JOINTS_DEG 的对应半边，避免实时状态噪声造成抖动。
PLACE_ACTIVE_ARM_JOINTS_DEG = {
    "right": {
        # 右手能放右侧箱和中间箱。
        "sorting_bin_a": [-30, -10, -10, -80, 70, 0, 0],
        "sorting_bin_b": [-30, 0, 30, -80, 70, 0, 0],
    },
    "left": {
        # 左手能放左侧箱和中间箱。
        "sorting_bin_b": [-30, 0, -30, -80, -70, 0, 0],
        "sorting_bin_c": [-30, 10, 10, -80, -70, 0, 0],
    },
}

# 世界坐标到末端目标坐标的经验偏移（由 scene2 的 B 类工件手工标定得到）。
# 注意左右手在 y 方向通常需要镜像偏移，避免目标过于靠近中线导致 IK 不可达。
WORLD_TO_EE_OFFSET_X = 0.566
WORLD_TO_EE_OFFSET_Y_RIGHT = -0.014
WORLD_TO_EE_OFFSET_Y_LEFT = 0.014
WORLD_TO_EE_OFFSET_Z = -0.923783

# 预设放置位姿（箱子上方）：位置与四元数可提前写死并手动调整。
PRESET_PLACE_TARGETS_BY_BIN = {
    "sorting_bin_a": [
        [0.565486, -0.443608, 0.174811],
    ],
    "sorting_bin_b": [
        [0.565486, -0.013608, 0.174811],
    ],
    "sorting_bin_c": [
        [0.565486, 0.416392, 0.174811],
    ],
}

# 放置点 x 方向统一偏移（负数=更靠近机器人，不必到箱子正上方）。
PLACE_TARGET_X_BIAS = -0.030

# Filled after the robot reaches the initial work pose via FK.
_WORK_POSE = {
    "left_xyz": None,
    "right_xyz": None,
    "left_quat": None,
    "right_quat": None,
}

# Lift clearance after grasp; increase to avoid table/bin collisions in transfer.
LIFT_Z_OFFSET = 0.30
PRE_GRASP_APPROACH_Z_OFFSET = 0.1
HANDOFF_APPROACH_Z_OFFSET = 0.2
PLACE_DWELL = 0.4

HEAD_TARGET = [0.0, 20.0]
HEAD_SETTLE_TIME = 0.4
ARM_MODE_EXTERNAL_CONTROL = 2
ARM_MODE_AUTO_SWING = 1
ARM_MODE_SERVICE = "/arm_traj_change_mode"
ARM_TARGET_POSES_TOPIC = "/kuavo_arm_target_poses"
ARM_TRAJ_TOPIC = "/kuavo_arm_traj"
ARM_TRAJ_HZ = 100.0
ARM_MOVE_TIME = 1.4
ARM_SETTLE_TIME = 0.15
ORIENTATION_TOLERANCE_RAD = math.radians(60.0) #20
GRASP_POSITION_TOLERANCE = 0.1 #0.012
IK_MODE_POS_HARD_ORI_SOFT = 0x02
IK_MODE_POS_HARD_ORI_HARD = 0x03
IK_MODE_THREE_POINT_MIXED = 0x06
THREE_POINT_WEIGHT = 2.0
FAST_GRASP_SETTLE_HOLD = 0.8
GRIPPER_CLOSE_TIME = 0.6
TOPIC_TIMEOUT = 20.0
LAUNCH_TIMEOUT = 120.0
RIGHT_GRIPPER_OPEN = 0.0
LEFT_GRIPPER_OPEN = 0.0
RIGHT_GRIPPER_CLOSE = 255.0
LEFT_GRIPPER_CLOSE = 255.0
GRIPPER_COMMAND_HZ = 100.0

# 分拣顺序：
# 1) Middle pair first (part_type_b_1 / part_type_b_2)
# 2) Side pairs after middle (A side then C side)
SORTING_OBJECT_ORDER = [
    "part_type_b_1",
    "part_type_b_2",
    "part_type_a_1",
    "part_type_a_2",
    "part_type_c_1",
    "part_type_c_2",
]

OBJECT_TO_BIN = {
    "part_type_a_1": "sorting_bin_a",
    "part_type_a_2": "sorting_bin_a",
    "part_type_b_1": "sorting_bin_b",
    "part_type_b_2": "sorting_bin_b",
    "part_type_c_1": "sorting_bin_c",
    "part_type_c_2": "sorting_bin_c",
}


def _now_tag():
    return _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class RosbagRecorder:
    def __init__(self, enabled, bag_path, topics):
        self.enabled = bool(enabled)
        self.bag_path = bag_path
        self.topics = list(topics)
        self.proc = None
        self.keep = False

    def start(self):
        if not self.enabled or self.proc is not None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.bag_path)), exist_ok=True)
        cmd = [
            "rosbag",
            "record",
            "--buffsize=0",
            "--chunksize=4096",
            "-O",
            self.bag_path,
        ] + self.topics
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        print(f"[INFO] rosbag recording started: {self.bag_path}")

    def stop(self):
        if self.proc is None:
            return
        _terminate_process_group(self.proc, signal.SIGINT, timeout=10)
        self.proc = None
        print(f"[INFO] rosbag recording stopped: {self.bag_path}")

    def mark_keep(self):
        self.keep = True

    def discard(self):
        if not self.enabled:
            return
        self.stop()
        for path in (self.bag_path, self.bag_path + ".active"):
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"[INFO] discarded rosbag: {path}")
            except OSError as exc:
                print(f"[WARN] failed to remove rosbag {path}: {exc}")


def _check_rosbag_camera_frequency(
    bag_path,
    min_hz=ROSBAG_CAMERA_MIN_HZ,
    topic_keyword=ROSBAG_CAMERA_TOPIC_KEYWORD,
):
    if not os.path.exists(bag_path):
        print(f"[WARN] rosbag camera frequency check failed: missing bag {bag_path}")
        return False

    try:
        import rosbag
    except Exception as exc:
        print(f"[WARN] rosbag camera frequency check failed: cannot import rosbag: {exc}")
        return False

    try:
        with rosbag.Bag(bag_path, "r") as bag:
            topic_info = bag.get_type_and_topic_info()[1]
            camera_topics = [
                topic
                for topic in ROSBAG_TOPICS
                if topic_keyword in topic
            ]
            if not camera_topics:
                print("[WARN] rosbag camera frequency check failed: no camera topics found")
                return False
            missing_topics = [topic for topic in camera_topics if topic not in topic_info]
            if missing_topics:
                print(
                    "[WARN] rosbag camera frequency check failed: missing topics "
                    + ", ".join(missing_topics)
                )
                return False

            stats = {
                topic: {"count": 0, "first": None, "last": None}
                for topic in camera_topics
            }
            for topic, _msg, timestamp in bag.read_messages(topics=camera_topics):
                stamp = timestamp.to_sec()
                item = stats[topic]
                item["count"] += 1
                if item["first"] is None:
                    item["first"] = stamp
                item["last"] = stamp
    except Exception as exc:
        print(f"[WARN] rosbag camera frequency check failed: {exc}")
        return False

    failed = []
    summaries = []
    for topic in camera_topics:
        item = stats[topic]
        count = item["count"]
        first = item["first"]
        last = item["last"]
        if count < 2 or first is None or last is None or last <= first:
            failed.append(f"{topic}=insufficient({count})")
            continue
        hz = float(count - 1) / float(last - first)
        summaries.append(f"{topic}={hz:.2f}Hz")
        if hz < float(min_hz):
            failed.append(f"{topic}={hz:.2f}Hz")

    if summaries:
        print("[INFO] rosbag camera hz: " + ", ".join(summaries))
    if failed:
        print(
            f"[WARN] rosbag camera frequency check failed: min required {min_hz:.1f}Hz; "
            + ", ".join(failed)
        )
        return False

    print(f"[INFO] rosbag camera frequency check passed: all camera topics >= {min_hz:.1f}Hz")
    return True


def _world_xyz_to_ee_xyz(world_xyz, arm):
    y_offset = WORLD_TO_EE_OFFSET_Y_LEFT if arm == "left" else WORLD_TO_EE_OFFSET_Y_RIGHT
    return [
        world_xyz[0] + WORLD_TO_EE_OFFSET_X,
        world_xyz[1] + y_offset,
        world_xyz[2] + WORLD_TO_EE_OFFSET_Z,
    ]


def _place_target_xyz(bin_name, slot_index):
    targets = PRESET_PLACE_TARGETS_BY_BIN[bin_name]
    target = list(targets[0])
    target[0] += PLACE_TARGET_X_BIAS
    return target


def _arm_for_object(object_name):
    return get_object_arm(object_name)


def _handoff_target_arm(job):
    if job["bin"] == "sorting_bin_a" and job["arm"] == "left":
        return "right"
    if job["bin"] == "sorting_bin_c" and job["arm"] == "right":
        return "left"
    return None


def _handoff_key(source_arm, target_arm):
    if source_arm == "left" and target_arm == "right":
        return "handoff_left_to_right"
    if source_arm == "right" and target_arm == "left":
        return "handoff_right_to_left"
    raise ValueError(f"unsupported handoff: {source_arm} -> {target_arm}")


def _handoff_config(source_arm, target_arm):
    return get_handoff_transition_config(_handoff_key(source_arm, target_arm), active_arm=source_arm)


def _part_type_key(object_name):
    for part_type in ("part_type_a", "part_type_b", "part_type_c"):
        if object_name.startswith(part_type):
            return part_type
    return None


def _handoff_value_by_part_type(cfg, by_part_type_key, object_name):
    part_type = _part_type_key(object_name)
    values_by_part_type = cfg.get(by_part_type_key) or {}
    if part_type not in values_by_part_type:
        raise RuntimeError(f"no handoff {by_part_type_key} configured for {object_name}")
    return values_by_part_type[part_type]


def _resolve_handoff_world_xyz(world_xyz, fallback_z):
    target = list(world_xyz)
    if target[2] is None:
        target[2] = fallback_z
    return [float(v) for v in target]


def _handoff_place_world_xyz(job, target_arm):
    cfg = _handoff_config(job["arm"], target_arm)
    return _resolve_handoff_world_xyz(
        _handoff_value_by_part_type(
            cfg,
            "place_world_xyz_by_part_type",
            job["object"],
        ),
        job["world_xyz"][2],
    )


def _handoff_grasp_world_xyz(job, target_arm):
    cfg = _handoff_config(job["arm"], target_arm)
    return _resolve_handoff_world_xyz(
        _handoff_value_by_part_type(
            cfg,
            "grasp_world_xyz_by_part_type",
            job["object"],
        ),
        job["world_xyz"][2],
    )


def _handoff_release_quat_xyzw(job, target_arm):
    quat = _handoff_value_by_part_type(
        _handoff_config(job["arm"], target_arm),
        "place_quat_xyzw_by_part_type",
        job["object"],
    )
    return list(quat) if quat is not None else None


def _handoff_regrasp_quat_xyzw(job, target_arm):
    quat = _handoff_value_by_part_type(
        get_handoff_transition_config(
            _handoff_key(job["arm"], target_arm),
            active_arm=target_arm,
        ),
        "grasp_quat_xyzw_by_part_type",
        job["object"],
    )
    return list(quat) if quat is not None else None


def _order_sorting_jobs(jobs):
    order_index = {name: idx for idx, name in enumerate(SORTING_OBJECT_ORDER)}
    # 先清理中间/可直接入箱的工件，再做需要放到过渡区的换手工件，减少中间区域堆叠。
    return sorted(
        jobs,
        key=lambda job: (
            1 if _handoff_target_arm(job) else 0,
            abs(float(job["world_xyz"][1])),
            order_index.get(job["object"], 999),
        ),
    )


def _prioritize_first_pick(jobs, first_pick):
    if not first_pick:
        return jobs
    for idx, job in enumerate(jobs):
        if job["object"] == first_pick:
            return [job] + jobs[:idx] + jobs[idx + 1 :]
    raise ValueError(f"unknown first pick object: {first_pick}")


def _insert_pick_before(jobs, insert_before):
    """在自动排序结果上做一次局部插队：把 A 工件移动到 B 工件前面。"""
    if not insert_before:
        return jobs
    moving_object, before_object = insert_before
    if moving_object == before_object:
        raise ValueError("--insert-before requires two different objects")

    moving_job = None
    remaining = []
    for job in jobs:
        if job["object"] == moving_object:
            moving_job = job
        else:
            remaining.append(job)
    if moving_job is None:
        raise ValueError(f"unknown insert object: {moving_object}")

    for idx, job in enumerate(remaining):
        if job["object"] == before_object:
            return remaining[:idx] + [moving_job] + remaining[idx:]
    raise ValueError(f"unknown insert-before target object: {before_object}")


def _build_sorting_jobs(first_pick=None, insert_before=None):
    jobs = []
    slot_count_by_bin = {}
    for object_name in SORTING_OBJECT_ORDER:
        bin_name = OBJECT_TO_BIN[object_name]
        slot_index = slot_count_by_bin.get(bin_name, 0)
        slot_count_by_bin[bin_name] = slot_index + 1
        arm = _arm_for_object(object_name)
        world_xyz = get_object_world_xyz(object_name, active_arm=arm)
        # 每个 job 在这里固化“抓取目标 + 放置目标 + 使用哪只手”。
        jobs.append(
            {
                "object": object_name,
                "bin": bin_name,
                "arm": arm,
                "world_xyz": list(world_xyz),
                "grasp": _world_xyz_to_ee_xyz(world_xyz, arm),
                "place": _place_target_xyz(bin_name, slot_index),
            }
        )
    jobs = _order_sorting_jobs(jobs)
    jobs = _insert_pick_before(jobs, insert_before)
    return _prioritize_first_pick(jobs, first_pick)

def _start_challenge_task(seed):
    cmd = [
        sys.executable,
        CHALLENGE_TASK_SCRIPT,
        "--scene",
        SCENE_NAME,
        "--seed",
        str(seed),
        "--no-timer-gui",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, cmd


def _wait_for_roscore(proc, timeout):
    import xmlrpc.client

    master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"challenge_task exited early with code {proc.returncode}")
        try:
            master = xmlrpc.client.ServerProxy(master_uri)
            code, _message, _state = master.getSystemState("/scene2_sorting_pipeline_wait")
            if code == 1:
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"roscore did not become ready within {timeout:.1f}s")


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
                f"[WARN] process group {proc.pid} did not exit after SIGKILL",
                file=sys.stderr,
            )
    except ProcessLookupError:
        return


def _pkill_ros_processes():
    try:
        subprocess.run(
            ["sudo", "-S", "pkill", "-A", "-f", "ros|mujoco|MuJoCo"],
            input=" \n",
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        print("[INFO] executed final cleanup: sudo pkill -A -f 'ros|mujoco|MuJoCo'")
    except subprocess.TimeoutExpired:
        print("[WARN] final cleanup timed out: sudo pkill -A -f 'ros|mujoco|MuJoCo'")
    except Exception as exc:
        print(f"[WARN] final cleanup failed: {exc}")


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


def _joint_qpos_size(elem):
    if elem.tag == "freejoint":
        return 7
    if elem.tag != "joint":
        return 0
    joint_type = elem.get("type", "hinge")
    if joint_type == "free":
        return 7
    if joint_type == "ball":
        return 4
    return 1


def _iter_xml_children_with_includes(path):
    root = ET.parse(path).getroot()
    base_dir = os.path.dirname(path)
    for child in list(root):
        if child.tag == "include":
            include_path = child.get("file")
            if include_path:
                resolved = include_path
                if not os.path.isabs(resolved):
                    resolved = os.path.join(base_dir, resolved)
                yield from _iter_xml_children_with_includes(os.path.abspath(resolved))
            continue
        yield child, base_dir


def _walk_body_for_qpos(body, qpos_addr, qpos_map):
    for child in list(body):
        size = _joint_qpos_size(child)
        if size:
            name = child.get("name")
            if name:
                qpos_map[name] = qpos_addr
            qpos_addr += size
    for child in list(body):
        if child.tag == "body":
            qpos_addr = _walk_body_for_qpos(child, qpos_addr, qpos_map)
    return qpos_addr


def _build_qpos_map(xml_path):
    qpos_addr = 0
    qpos_map = {}
    for child, _base_dir in _iter_xml_children_with_includes(xml_path):
        if child.tag != "worldbody":
            continue
        for body in list(child):
            if body.tag == "body":
                qpos_addr = _walk_body_for_qpos(body, qpos_addr, qpos_map)
    return qpos_map


def _pose_from_qpos(qpos, qpos_map, name):
    if name not in qpos_map:
        raise KeyError(f"freejoint '{name}' not found in XML qpos map")
    addr = qpos_map[name]
    if addr + 2 >= len(qpos):
        raise IndexError(f"qpos for '{name}' address {addr} outside qpos length {len(qpos)}")
    return [float(qpos[addr]), float(qpos[addr + 1]), float(qpos[addr + 2])]


def _part_in_bin(part_pos, bin_pos, xy_tolerance=(0.12, 0.13)):
    return (
        abs(part_pos[0] - bin_pos[0]) <= float(xy_tolerance[0])
        and abs(part_pos[1] - bin_pos[1]) <= float(xy_tolerance[1])
    )


def _check_scene2_sorted_success(timeout=TOPIC_TIMEOUT):
    import rospy
    from std_msgs.msg import Float64MultiArray

    qpos_map = _build_qpos_map(ACTIVE_SCENE_XML)
    msg = rospy.wait_for_message("/mujoco/qpos", Float64MultiArray, timeout=float(timeout))
    qpos = list(msg.data)
    positions = {name: _pose_from_qpos(qpos, qpos_map, name) for name in BIN_NAMES + PART_NAMES}
    failures = []
    for part_name, bin_name in PART_TO_BIN.items():
        if not _part_in_bin(positions[part_name], positions[bin_name]):
            failures.append((part_name, bin_name, positions[part_name], positions[bin_name]))
    if failures:
        for part_name, bin_name, part_pos, bin_pos in failures:
            rospy.logwarn(
                "scene2 sorting check failed: %s not in %s part=%s bin=%s",
                part_name,
                bin_name,
                [round(v, 4) for v in part_pos],
                [round(v, 4) for v in bin_pos],
            )
        return False
    rospy.loginfo("scene2 sorting check passed: all parts are in expected bins")
    return True


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
    rospy.loginfo("scene2 sorting: arm mode -> %s: %s", mode, response.message)


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
        self._thread = threading.Thread(target=self._run, name="gripper_command_hold", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_open(self):
        self._set_command(LEFT_GRIPPER_OPEN, RIGHT_GRIPPER_OPEN)

    def set_right_closed(self):
        self._set_command(LEFT_GRIPPER_OPEN, RIGHT_GRIPPER_CLOSE)

    def set_left_closed(self):
        self._set_command(LEFT_GRIPPER_CLOSE, RIGHT_GRIPPER_OPEN)

    def _set_command(self, left_cmd, right_cmd):
        with self._lock:
            self._left_cmd = float(left_cmd)
            self._right_cmd = float(right_cmd)

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
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break


def _start_gripper_hold(timeout):
    import rospy
    from sensor_msgs.msg import JointState

    pub = rospy.Publisher("/gripper/command", JointState, queue_size=10)
    _wait_for_connection(pub, timeout)
    hold = GripperCommandHold(pub)
    hold.start()
    return hold


def _publish_gripper_open(gripper_hold):
    gripper_hold.set_open()


def _publish_arm_gripper_close(gripper_hold, arm):
    if arm == "left":
        gripper_hold.set_left_closed()
        return
    if arm == "right":
        gripper_hold.set_right_closed()
        return
    raise ValueError(f"unknown arm: {arm}")


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
            try:
                self._pub.publish(msg)
                rate.sleep()
            except rospy.ROSException:
                break

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
    initial_degrees = rad_to_deg(_read_current_arm_joints(timeout))
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

    #_publish_arm_target_poses(target_pub, target_degrees, move_time)
    _publish_arm_traj_interpolation(arm_hold, start_degrees, target_degrees, move_time)
    rospy.sleep(settle)


def _locked_motion_start_degrees(active_arm=None, locked_other_arm_joints=None):
    start = list(_read_current_arm_joints(TOPIC_TIMEOUT))
    if locked_other_arm_joints is not None:
        locked = [float(v) for v in locked_other_arm_joints]
        if active_arm == "left":
            start[7:14] = locked
        elif active_arm == "right":
            start[:7] = locked
        else:
            raise ValueError(f"unknown arm: {active_arm}")
    return rad_to_deg(start)


def _move_arm_to(
    target_pub,
    arm_hold,
    degrees_list,
    move_time=ARM_MOVE_TIME,
    settle=ARM_SETTLE_TIME,
    active_arm=None,
    locked_other_arm_joints=None,
):
    start = _locked_motion_start_degrees(active_arm, locked_other_arm_joints)
    _execute_arm_motion(target_pub, arm_hold, start, degrees_list, move_time, settle)


def _require_work_pose():
    if _WORK_POSE["left_xyz"] is None or _WORK_POSE["right_xyz"] is None:
        raise RuntimeError("initial work pose is not initialized")
    return _WORK_POSE


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


def _capture_work_pose_from_fk(grasp_runtime):
    left_xyz, left_quat = measure_hand_pose(grasp_runtime, "left")
    right_xyz, right_quat = measure_hand_pose(grasp_runtime, "right")
    _WORK_POSE["left_xyz"] = left_xyz
    _WORK_POSE["right_xyz"] = right_xyz
    _WORK_POSE["left_quat"] = left_quat
    _WORK_POSE["right_quat"] = right_quat


def _move_to_work_pose_joints(target_pub, arm_hold, active_arm=None, locked_other_arm_joints=None):
    import rospy

    rospy.loginfo("scene2 sorting: return to work pose (joint angles)")
    _move_arm_to(
        target_pub,
        arm_hold,
        WORK_POSE_JOINTS_DEG,
        move_time=ARM_MOVE_TIME,
        settle=FAST_GRASP_SETTLE_HOLD,
        active_arm=active_arm,
        locked_other_arm_joints=locked_other_arm_joints,
    )


def _enter_work_pose(target_pub, arm_hold, grasp_runtime):
    import rospy

    rospy.loginfo("scene2 sorting: side lift -> work pose (joint angles)")
    _move_arm_to(
        target_pub,
        arm_hold,
        SIDE_LIFT_JOINTS_DEG,
        move_time=ARM_MOVE_TIME,
        settle=ARM_SETTLE_TIME,
    )
    _move_to_work_pose_joints(target_pub, arm_hold)
    _capture_work_pose_from_fk(grasp_runtime)
    rospy.loginfo(
        "scene2 sorting: work pose locked left=%s right=%s",
        [round(v, 4) for v in _WORK_POSE["left_xyz"]],
        [round(v, 4) for v in _WORK_POSE["right_xyz"]],
    )


def _place_active_arm_joints(active_arm, bin_name):
    try:
        joints = PLACE_ACTIVE_ARM_JOINTS_DEG[active_arm][bin_name]
    except KeyError:
        raise RuntimeError(f"no joint-space place pose configured for {active_arm} hand -> {bin_name}")
    if len(joints) != 7:
        raise ValueError(f"place pose for {active_arm} hand -> {bin_name} must have 7 joints")
    return [float(v) for v in joints]


def _fixed_work_pose_other_arm_joints(active_arm):
    if active_arm == "left":
        other_deg = WORK_POSE_JOINTS_DEG[7:14]
    elif active_arm == "right":
        other_deg = WORK_POSE_JOINTS_DEG[:7]
    else:
        raise ValueError(f"unknown arm: {active_arm}")
    return [math.radians(float(v)) for v in other_deg]


def _compose_single_arm_place_joints(active_arm, active_joints_deg, locked_other_arm_joints):
    other_deg = rad_to_deg(locked_other_arm_joints)
    if active_arm == "left":
        return list(active_joints_deg) + other_deg
    if active_arm == "right":
        return other_deg + list(active_joints_deg)
    raise ValueError(f"unknown arm: {active_arm}")


def _pick_part_absolute(arm_pub, arm_hold, gripper_hold, job, locked_other_arm_joints, hold_time, grasp_runtime):
    import rospy

    active_arm = job["arm"]
    grasp_ok = execute_part_grasp(
        grasp_runtime,
        job["object"],
        job["world_xyz"],
        active_arm,
        locked_other_arm_joints,
        grasp_quat_xyzw=job.get("grasp_quat"),
    )
    if not grasp_ok:
        rospy.logwarn("scene2 sorting: %s grasp returned failed", job["object"])

    # 抓住后先上抬，避免移动到箱子上方时扫到桌面/箱沿。
    grasp_target = list(job["grasp"])
    lift_target = [grasp_target[0], grasp_target[1], grasp_target[2] + LIFT_Z_OFFSET]
    held_quat = job.get("lift_quat") or get_object_lift_quat_xyzw(job["object"], active_arm=active_arm)
    lift_err, _lift_quat_err, _lift_actual, _lift_quat, _lift_cmd14 = move_arm_ik_once(
        runtime=grasp_runtime,
        active_arm=active_arm,
        active_pos=lift_target,
        locked_other_arm_joints=locked_other_arm_joints,
        active_quat=held_quat,
        label=f"{job['object']}_lift",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=2.0,
        move_time=ARM_MOVE_TIME,
        settle_time=hold_time,
    )
    rospy.loginfo("scene2 sorting: %s lift finished xyz_err=%.4f m", job["object"], lift_err)
    return bool(grasp_ok)


def _transfer_part_to_handoff(arm_pub, arm_hold, gripper_hold, job, target_arm, locked_other_arm_joints, hold_time, grasp_runtime):
    import rospy

    active_arm = job["arm"]
    handoff_place_world_xyz = _handoff_place_world_xyz(job, target_arm)
    handoff_target = _world_xyz_to_ee_xyz(handoff_place_world_xyz, active_arm)
    handoff_above_target = [
        handoff_target[0],
        handoff_target[1],
        handoff_target[2] + HANDOFF_APPROACH_Z_OFFSET,
    ]
    held_quat = (
        _handoff_release_quat_xyzw(job, target_arm)
        or get_object_lift_quat_xyzw(job["object"], active_arm=active_arm)
    )
    rospy.loginfo(
        "scene2 sorting: %s handoff %s -> %s at world=%s",
        job["object"],
        active_arm,
        target_arm,
        [round(v, 4) for v in handoff_place_world_xyz],
    )
    move_arm_ik_once(
        runtime=grasp_runtime,
        active_arm=active_arm,
        active_pos=handoff_above_target,
        locked_other_arm_joints=locked_other_arm_joints,
        active_quat=held_quat,
        label=f"{job['object']}_handoff_above_{active_arm}_to_{target_arm}",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=2.0,
        move_time=ARM_MOVE_TIME,
        settle_time=hold_time,
    )
    drop_err, _quat_err, actual, _actual_quat, _cmd14 = move_arm_ik_once(
        runtime=grasp_runtime,
        active_arm=active_arm,
        active_pos=handoff_target,
        locked_other_arm_joints=locked_other_arm_joints,
        active_quat=held_quat,
        label=f"{job['object']}_handoff_{active_arm}_to_{target_arm}",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=2.0,
        move_time=ARM_MOVE_TIME,
        settle_time=hold_time,
    )
    rospy.loginfo(
        "scene2 sorting: %s handoff release actual=%s xyz_err=%.4f m",
        job["object"],
        [round(v, 4) for v in actual],
        drop_err,
    )
    _publish_gripper_open(gripper_hold)
    rospy.sleep(PLACE_DWELL)
    move_arm_ik_once(
        runtime=grasp_runtime,
        active_arm=active_arm,
        active_pos=handoff_above_target,
        locked_other_arm_joints=locked_other_arm_joints,
        active_quat=held_quat,
        label=f"{job['object']}_handoff_retract_{active_arm}_to_{target_arm}",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=2.0,
        move_time=ARM_MOVE_TIME,
        settle_time=hold_time,
    )
    _move_to_work_pose_joints(
        arm_pub,
        arm_hold,
        active_arm=active_arm,
        locked_other_arm_joints=locked_other_arm_joints,
    )
    return _handoff_grasp_world_xyz(job, target_arm)


def _make_regrasp_job(job, target_arm, handoff_world_xyz):
    regrasp = dict(job)
    regrasp["arm"] = target_arm
    regrasp["world_xyz"] = list(handoff_world_xyz)
    regrasp["grasp"] = _world_xyz_to_ee_xyz(handoff_world_xyz, target_arm)
    regrasp["handoff_from"] = job["arm"]
    handoff_quat = _handoff_regrasp_quat_xyzw(job, target_arm)
    if handoff_quat is not None:
        regrasp["grasp_quat"] = handoff_quat
        regrasp["lift_quat"] = handoff_quat
    return regrasp


def _place_part_absolute(arm_pub, arm_hold, gripper_hold, job, locked_other_arm_joints, hold_time, grasp_runtime):
    import rospy

    active_arm = job["arm"]
    active_place_joints = _place_active_arm_joints(active_arm, job["bin"])
    target_joints = _compose_single_arm_place_joints(
        active_arm,
        active_place_joints,
        locked_other_arm_joints,
    )
    rospy.loginfo(
        "scene2 sorting: %s %s-hand joint-space place -> %s active_joints=%s",
        job["object"],
        active_arm,
        job["bin"],
        [round(v, 2) for v in active_place_joints],
    )
    _move_arm_to(
        arm_pub,
        arm_hold,
        target_joints,
        move_time=ARM_MOVE_TIME,
        settle=hold_time,
        active_arm=active_arm,
        locked_other_arm_joints=locked_other_arm_joints,
    )
    rospy.loginfo(
        "scene2 sorting: %s place release by joint pose, open %s gripper",
        job["object"],
        active_arm,
    )
    _publish_gripper_open(gripper_hold)
    rospy.sleep(PLACE_DWELL)

    # 放置后直接回到 WORK_POSE_JOINTS_DEG，不走中间回位点。
    _move_to_work_pose_joints(
        arm_pub,
        arm_hold,
        active_arm=active_arm,
        locked_other_arm_joints=locked_other_arm_joints,
    )


def _run_pick_place_jobs(arm_pub, arm_hold, gripper_hold, jobs, hold_time, grasp_runtime):
    import rospy

    for index, job in enumerate(jobs, start=1):
        rospy.loginfo(
            "scene2 sorting: job %d/%d %s (%s hand) -> %s grasp=%s place=%s",
            index,
            len(jobs),
            job["object"],
            job["arm"],
            job["bin"],
            [round(v, 4) for v in job["grasp"]],
            [round(v, 4) for v in job["place"]],
        )
        _publish_gripper_open(gripper_hold)
        active_arm = job["arm"]
        # 非作业臂锁定到固定工作姿态，避免实时状态噪声造成另一侧抖动。
        locked_other_arm_joints = _fixed_work_pose_other_arm_joints(active_arm)
        grasp_ok = _pick_part_absolute(
            arm_pub,
            arm_hold,
            gripper_hold,
            job,
            locked_other_arm_joints,
            hold_time,
            grasp_runtime,
        )
        if not grasp_ok:
            raise RuntimeError(f"grasp failed for {job['object']}")
        handoff_arm = _handoff_target_arm(job)
        if handoff_arm is not None:
            handoff_world_xyz = _transfer_part_to_handoff(
                arm_pub,
                arm_hold,
                gripper_hold,
                job,
                handoff_arm,
                locked_other_arm_joints,
                hold_time,
                grasp_runtime,
            )
            regrasp_job = _make_regrasp_job(job, handoff_arm, handoff_world_xyz)
            rospy.loginfo(
                "scene2 sorting: %s regrasp from handoff using %s hand",
                job["object"],
                handoff_arm,
            )
            locked_other_arm_joints = _fixed_work_pose_other_arm_joints(handoff_arm)
            grasp_ok = _pick_part_absolute(
                arm_pub,
                arm_hold,
                gripper_hold,
                regrasp_job,
                locked_other_arm_joints,
                hold_time,
                grasp_runtime,
            )
            if not grasp_ok:
                raise RuntimeError(f"handoff regrasp failed for {job['object']}")
            _place_part_absolute(
                arm_pub,
                arm_hold,
                gripper_hold,
                regrasp_job,
                locked_other_arm_joints,
                hold_time,
                grasp_runtime,
            )
            continue
        _place_part_absolute(
            arm_pub,
            arm_hold,
            gripper_hold,
            job,
            locked_other_arm_joints,
            hold_time,
            grasp_runtime,
        )


def _run_sorting_pipeline(gripper_hold, arm_hold, recorder=None, first_pick=None, insert_before=None):
    import rospy
    from kuavo_msgs.msg import armTargetPoses

    arm_pub = rospy.Publisher(ARM_TARGET_POSES_TOPIC, armTargetPoses, queue_size=10)
    _wait_for_connection(arm_pub, TOPIC_TIMEOUT)

    grasp_runtime = GraspRuntime(
        world_to_ee_offset_x=WORLD_TO_EE_OFFSET_X,
        world_to_ee_offset_y_left=WORLD_TO_EE_OFFSET_Y_LEFT,
        world_to_ee_offset_y_right=WORLD_TO_EE_OFFSET_Y_RIGHT,
        world_to_ee_offset_z=WORLD_TO_EE_OFFSET_Z,
        pre_grasp_z_offset=PRE_GRASP_APPROACH_Z_OFFSET,
        grasp_position_tolerance=GRASP_POSITION_TOLERANCE,
        orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD,
        gripper_close_time=GRIPPER_CLOSE_TIME,
        timeout=TOPIC_TIMEOUT,
        move_time=ARM_MOVE_TIME,
        settle_time=ARM_SETTLE_TIME,
        ik_mode_pos_hard_ori_hard=IK_MODE_THREE_POINT_MIXED,
        read_current_arm_joints_cb=lambda: _read_current_arm_joints(TOPIC_TIMEOUT),
        execute_arm_motion_cb=lambda start_degrees, target_degrees, move_time, settle: _execute_arm_motion(
            arm_pub,
            arm_hold,
            start_degrees,
            target_degrees,
            move_time,
            settle,
        ),
        publish_arm_gripper_close_cb=lambda arm: _publish_arm_gripper_close(gripper_hold, arm),
        sleep_cb=rospy.sleep,
        loginfo_cb=rospy.loginfo,
        logwarn_cb=rospy.logwarn,
    )

    _publish_gripper_open(gripper_hold)
    _enter_work_pose(arm_pub, arm_hold, grasp_runtime)
    if recorder is not None:
        recorder.start()
    pick_place_jobs = _build_sorting_jobs(first_pick=first_pick, insert_before=insert_before)
    rospy.loginfo(
        "scene2 sorting: pick order=%s",
        " -> ".join(job["object"] for job in pick_place_jobs),
    )
    _run_pick_place_jobs(
        arm_pub,
        arm_hold,
        gripper_hold,
        pick_place_jobs,
        FAST_GRASP_SETTLE_HOLD,
        grasp_runtime,
    )
    rospy.loginfo("scene2 sorting: all jobs finished, return work -> side lift -> home")
    _move_to_work_pose_joints(arm_pub, arm_hold)
    if recorder is not None:
        recorder.stop()
    _move_arm_to(arm_pub, arm_hold, SIDE_LIFT_JOINTS_DEG)
    _move_arm_to(arm_pub, arm_hold, HOME_JOINTS_DEG)
    rospy.loginfo("scene2 sorting: all jobs finished, retracted to home")


def _run_once(args):
    if args.headless:
        os.environ["MUJOCO_HEADLESS"] = "1"

    seed = args.seed if args.seed is not None else random.randint(0, 9999)
    configure_scene2_layout(seed=seed)
    os.environ["SCENE2_LAYOUT_SEED"] = str(seed)
    print(f"[INFO] scene2 sorting pipeline seed={seed}")

    launch_proc = None
    gripper_hold = None
    arm_hold = None
    recorder = None
    arm_mode_changed = False
    status = "failed"
    try:
        if args.use_existing_sim:
            _init_ros_node("scene2_sorting_pipeline")
            missing = _wait_for_topics(["/sensors_data_raw"], LAUNCH_TIMEOUT)
            if missing:
                raise RuntimeError("existing simulation is not ready: " + ", ".join(missing))
        else:
            launch_proc, launch_cmd = _start_challenge_task(seed)
            print(f"[INFO] started challenge_task: {' '.join(launch_cmd)}")
            _wait_for_roscore(launch_proc, timeout=30.0)
            _init_ros_node("scene2_sorting_pipeline")

        topic_wait_timeout = TOPIC_TIMEOUT if args.use_existing_sim else LAUNCH_TIMEOUT
        missing_topics = _wait_for_topics(["/sensors_data_raw"], topic_wait_timeout)
        if missing_topics:
            raise RuntimeError("required topics missing: " + ", ".join(missing_topics))

        _publish_head_target(TOPIC_TIMEOUT)
        gripper_hold = _start_gripper_hold(TOPIC_TIMEOUT)
        arm_hold = _start_arm_traj_hold(TOPIC_TIMEOUT)
        bag_path = os.path.join(
            os.path.abspath(args.output_dir),
            f"{SCENE_NAME}_seed_{seed}_{_now_tag()}.bag",
        )
        recorder = RosbagRecorder(args.record_rosbag, bag_path, ROSBAG_TOPICS)

        _set_arm_mode(ARM_MODE_EXTERNAL_CONTROL, timeout=TOPIC_TIMEOUT)
        arm_mode_changed = True
        _run_sorting_pipeline(
            gripper_hold,
            arm_hold,
            recorder=recorder,
            first_pick=args.first_pick,
            insert_before=args.insert_before,
        )
        if args.record_rosbag:
            if not _check_scene2_sorted_success(timeout=TOPIC_TIMEOUT):
                raise RuntimeError("scene2 final bin check failed; discard rosbag")
            if not _check_rosbag_camera_frequency(recorder.bag_path):
                raise RuntimeError("scene2 rosbag camera frequency check failed; discard rosbag")
            recorder.mark_keep()
        _set_arm_mode(ARM_MODE_AUTO_SWING, timeout=TOPIC_TIMEOUT)
        arm_mode_changed = False
        status = "success"
    except KeyboardInterrupt:
        status = "interrupted"
        print("[WARN] scene2 sorting interrupted by Ctrl+C")
    except Exception as exc:
        status = "failed"
        print(f"[ERROR] scene2 sorting failed: {exc}")
    finally:
        if recorder is not None:
            if recorder.keep and status == "success":
                recorder.stop()
            else:
                recorder.discard()
        if gripper_hold is not None:
            gripper_hold.stop()
        if arm_hold is not None:
            arm_hold.stop()
        if arm_mode_changed:
            if status == "success":
                try:
                    _set_arm_mode(ARM_MODE_AUTO_SWING, timeout=TOPIC_TIMEOUT)
                except Exception as exc:
                    print(f"[WARN] failed to restore arm mode: {exc}")
            else:
                print(
                    "[WARN] task failed; keep arm in external-control mode to avoid auto-swing crash"
                )
        _terminate_process_group(launch_proc, signal.SIGINT, timeout=20)
        print(f"[INFO] scene2 sorting pipeline {status} (seed={seed})")
        _pkill_ros_processes()

    return 0 if status == "success" else 1


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Scene2 sorting pipeline for six parts.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Scene seed passed to challenge_task.py; random if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for optional rosbag output.",
    )
    parser.add_argument(
        "--record-rosbag",
        action="store_true",
        help="Record selected scene2 topics from first work pose until final work pose; discard on failure.",
    )
    parser.add_argument(
        "--first-pick",
        choices=SORTING_OBJECT_ORDER,
        default=None,
        help="Move this object to the front of the picking order.",
    )
    parser.add_argument(
        "--insert-before",
        nargs=2,
        choices=SORTING_OBJECT_ORDER,
        metavar=("MOVE_OBJECT", "BEFORE_OBJECT"),
        default=None,
        help="Move MOVE_OBJECT to immediately before BEFORE_OBJECT in the picking order.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Keep running until N valid runs are saved. With --record-rosbag, failed checks discard the bag and do not count.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically collect one valid bag for each seed in [--seed-start, --seed-end].",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="First seed for --auto mode, inclusive.",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=None,
        help="Last seed for --auto mode, inclusive.",
    )
    parser.add_argument(
        "--max-attempts-per-seed",
        type=int,
        default=5,
        help="Maximum attempts for each seed in --auto mode before skipping to the next seed.",
    )
    parser.add_argument("--headless", action="store_true", help="Set MUJOCO_HEADLESS=1 for this run.")
    parser.add_argument(
        "--use-existing-sim",
        action="store_true",
        help="Attach to an already running scene2 simulation.",
    )
    return parser


def _strip_repeat_args(argv):
    stripped = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--repeat":
            skip_next = True
            continue
        if arg.startswith("--repeat="):
            continue
        stripped.append(arg)
    return stripped


def _strip_auto_args(argv):
    stripped = []
    skip_next = False
    options_with_values = {
        "--seed",
        "--repeat",
        "--seed-start",
        "--seed-end",
        "--max-attempts-per-seed",
    }
    options_with_prefix = tuple(option + "=" for option in options_with_values)
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--auto":
            continue
        if arg in options_with_values:
            skip_next = True
            continue
        if arg.startswith(options_with_prefix):
            continue
        stripped.append(arg)
    return stripped


def _run_repeated(repeat_count):
    base_argv = _strip_repeat_args(sys.argv[1:])
    script_path = os.path.abspath(__file__)
    valid_count = 0
    attempt_count = 0
    while valid_count < repeat_count:
        attempt_count += 1
        print(
            f"[INFO] scene2 repeat attempt {attempt_count}, "
            f"valid {valid_count}/{repeat_count}"
        )
        proc = None
        try:
            proc = subprocess.Popen(
                [sys.executable, script_path] + base_argv,
                preexec_fn=os.setsid,
            )
            code = proc.wait()
        except KeyboardInterrupt:
            print(
                f"[WARN] scene2 repeat interrupted, "
                f"valid {valid_count}/{repeat_count}"
            )
            _terminate_process_group(proc, signal.SIGINT, timeout=20)
            _pkill_ros_processes()
            return 1
        if code == 0:
            valid_count += 1
            print(f"[INFO] scene2 repeat valid run saved: {valid_count}/{repeat_count}")
            continue
        if code != 0:
            print(
                f"[WARN] scene2 repeat attempt {attempt_count} failed with code {code}; "
                "retrying until enough valid runs are saved"
            )
    print(f"[INFO] scene2 repeat finished: valid {valid_count}/{repeat_count}")
    return 0


def _run_auto_seed_range(seed_start, seed_end, max_attempts_per_seed):
    base_argv = _strip_auto_args(sys.argv[1:])
    script_path = os.path.abspath(__file__)
    step = 1 if seed_start <= seed_end else -1
    seeds = range(seed_start, seed_end + step, step)
    success_count = 0
    skipped = []

    for seed in seeds:
        seed_saved = False
        print(
            f"[INFO] scene2 auto seed {seed}: "
            f"collect 1 valid bag, max attempts {max_attempts_per_seed}"
        )
        for attempt in range(1, max_attempts_per_seed + 1):
            print(f"[INFO] scene2 auto seed {seed} attempt {attempt}/{max_attempts_per_seed}")
            proc = None
            try:
                proc = subprocess.Popen(
                    [sys.executable, script_path] + base_argv + ["--seed", str(seed)],
                    preexec_fn=os.setsid,
                )
                code = proc.wait()
            except KeyboardInterrupt:
                print(
                    f"[WARN] scene2 auto interrupted at seed {seed}, "
                    f"saved {success_count} seed(s)"
                )
                _terminate_process_group(proc, signal.SIGINT, timeout=20)
                _pkill_ros_processes()
                return 1

            if code == 0:
                success_count += 1
                seed_saved = True
                print(f"[INFO] scene2 auto seed {seed} saved successfully")
                break

            print(
                f"[WARN] scene2 auto seed {seed} attempt {attempt} failed with code {code}"
            )

        if not seed_saved:
            skipped.append(seed)
            print(
                f"[WARN] scene2 auto seed {seed} skipped after "
                f"{max_attempts_per_seed} failed attempt(s)"
            )

    print(
        f"[INFO] scene2 auto finished: saved {success_count} seed(s), "
        f"skipped {len(skipped)} seed(s)"
    )
    if skipped:
        print("[WARN] scene2 auto skipped seeds: " + ", ".join(str(seed) for seed in skipped))
    return 0


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.auto:
        if args.seed_start is None or args.seed_end is None:
            parser.error("--auto requires --seed-start and --seed-end")
        if args.max_attempts_per_seed < 1:
            parser.error("--max-attempts-per-seed must be >= 1")
        return _run_auto_seed_range(args.seed_start, args.seed_end, args.max_attempts_per_seed)
    if args.repeat is not None:
        if args.repeat < 1:
            parser.error("--repeat must be >= 1")
        return _run_repeated(args.repeat)
    return _run_once(args)


if __name__ == "__main__":
    sys.exit(main())
