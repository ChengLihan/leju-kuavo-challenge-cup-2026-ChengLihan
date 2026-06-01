#!/usr/bin/env python3
"""Collect scene2 data with fixed absolute pick/place attempts for nylon fittings."""

import argparse
import datetime as _datetime
import math
import os
import signal
import subprocess
import sys
import threading
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SCENE_NAME = "scene2"
TARGET_OBJECT_ALIAS = "nylon_water_pipe_fitting"
TARGET_BIN = "sorting_bin_b"
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "bags", SCENE_NAME)

ARM_JOINT_NAMES = ["arm_joint_" + str(i) for i in range(1, 15)]
DEFAULT_TOPICS = [
    "/gripper/command",
    "/gripper/state",
    "/sensors_data_raw",
    "/kuavo_arm_traj",
    "/robot_head_motion_data",
    "/cam_h/color/image_raw/compressed",
    "/cam_l/color/image_raw/compressed",
    "/cam_r/color/image_raw/compressed",
    "/cam_h/depth/image_raw/compressedDepth",
    "/cam_r/depth/image_rect_raw/compressedDepth",
    "/cam_l/depth/image_rect_raw/compressedDepth",
]

PREGRASP_POINTS_DEG = [
    [0, 0, 0, -30, 0, 0, 0, 0, 0, 0, -30, 0, 0, 0],
    [0, 0, 0, -30, 0, 0, 0, 45, -15, -5, -105, 60, 0, 0],
    [0, 0, 0, -30, 0, 0, 0, 22, -6, 18, -125, 88, 8, 0],
]

FIXED_RIGHT_HAND_TARGET_XYZ = [0.285966, -0.083886, -0.093783]
SECOND_RIGHT_HAND_TARGET_XYZ = [0.285966, 0.016114, -0.093783]
BIN_B_PLACE_TARGETS_XYZ = [
    [0.565486, -0.013608, 0.174811],
    [0.565486, 0.026392, 0.174811],
]
RIGHT_HAND_QUAT_XYZW = [-0.081987, -0.152343, 0.857876, 0.483858]
LIFT_Z_OFFSET = 0.060
PLACE_DWELL = 0.8

HEAD_TARGET = [0.0, 20.0]
HEAD_SETTLE_TIME = 0.8
ARM_MODE_EXTERNAL_CONTROL = 2
ARM_MODE_AUTO_SWING = 1
ARM_MODE_SERVICE = "/arm_traj_change_mode"
ARM_TARGET_POSES_TOPIC = "/kuavo_arm_target_poses"
ARM_TRAJ_TOPIC = "/kuavo_arm_traj"
ARM_TRAJ_HZ = 100.0
ARM_MOVE_TIME = 2.0
ARM_SETTLE_TIME = 0.3
ORIENTATION_TOLERANCE_RAD = math.radians(20.0)
GRASP_POSITION_TOLERANCE = 0.012
IK_MODE_POS_HARD_ORI_SOFT = 0x02
IK_MODE_THREE_POINT_MIXED = 0x06
THREE_POINT_WEIGHT = 2.0
FAST_GRASP_SETTLE_HOLD = 2.0
GRIPPER_CLOSE_TIME = 1.0
RECORD_SETTLE_TIME = 2.0
POST_ARM_MODE_RECORD_TIME = 1.0
TOPIC_TIMEOUT = 20.0
LAUNCH_TIMEOUT = 120.0
RIGHT_GRIPPER_OPEN = 0.0
LEFT_GRIPPER_OPEN = 0.0
RIGHT_GRIPPER_CLOSE = 255.0
GRIPPER_COMMAND_HZ = 100.0

PICK_PLACE_JOBS = [
    {
        "object": "part_type_b_1",
        "alias": TARGET_OBJECT_ALIAS,
        "grasp": FIXED_RIGHT_HAND_TARGET_XYZ,
        "place": BIN_B_PLACE_TARGETS_XYZ[0],
        "bin": TARGET_BIN,
    },
    {
        "object": "part_type_b_2",
        "alias": TARGET_OBJECT_ALIAS,
        "grasp": SECOND_RIGHT_HAND_TARGET_XYZ,
        "place": BIN_B_PLACE_TARGETS_XYZ[1],
        "bin": TARGET_BIN,
    },
]


def _now_tag():
    return _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _start_scene_launch():
    cmd = [
        "roslaunch",
        "challenge_cup_simulator",
        "load_kuavo_mujoco_challenge.launch",
        f"scene_name:={SCENE_NAME}",
        "raw_image:=false",
        "mujoco_vsync:=true",
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
            raise RuntimeError(f"roslaunch exited early with code {proc.returncode}")
        try:
            master = xmlrpc.client.ServerProxy(master_uri)
            code, _message, _state = master.getSystemState("/scene2_dataset_collector_wait")
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
    rospy.loginfo("scene2 dataset: arm mode -> %s: %s", mode, response.message)


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


def _publish_gripper_open(gripper_hold):
    gripper_hold.set_open()


def _publish_right_gripper_close(gripper_hold):
    gripper_hold.set_right_closed()


def _publish_arm_target_poses(target_pub, degrees_list, move_time):
    from kuavo_msgs.msg import armTargetPoses

    msg = armTargetPoses()
    msg.times = [float(move_time)]
    msg.values = [float(v) for v in degrees_list]
    target_pub.publish(msg)


def _publish_arm_traj_point(traj_pub, degrees_list):
    import rospy
    from sensor_msgs.msg import JointState

    traj_msg = JointState()
    traj_msg.header.stamp = rospy.Time.now()
    traj_msg.name = ARM_JOINT_NAMES
    traj_msg.position = [float(v) for v in degrees_list]
    traj_pub.publish(traj_msg)


def _publish_arm_traj_interpolation(traj_pub, start_degrees, target_degrees, duration):
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
        _publish_arm_traj_point(traj_pub, point)
        if step < steps:
            rate.sleep()


def _execute_arm_motion(target_pub, traj_pub, start_degrees, target_degrees, move_time, settle):
    import rospy

    _publish_arm_target_poses(target_pub, target_degrees, move_time)
    _publish_arm_traj_interpolation(traj_pub, start_degrees, target_degrees, move_time)
    rospy.sleep(settle)


def _move_arm_to(target_pub, traj_pub, degrees_list, move_time=ARM_MOVE_TIME, settle=ARM_SETTLE_TIME):
    start = _rad_to_deg(_read_current_arm_joints(TOPIC_TIMEOUT))
    _execute_arm_motion(target_pub, traj_pub, start, degrees_list, move_time, settle)


def _measure_right_pose(timeout):
    q = _read_current_arm_joints(timeout)
    pose = _call_fk(q, timeout).right_pose
    return list(pose.pos_xyz), list(pose.quat_xyzw)


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


def _call_right_hand_ik(
    current_joint_values,
    right_pos,
    timeout,
    desired_quat=None,
    constraint_mode=IK_MODE_POS_HARD_ORI_SOFT,
    pos_cost_weight=0.0,
):
    import rospy
    from kuavo_msgs.msg import twoArmHandPoseCmd
    from kuavo_msgs.srv import twoArmHandPoseCmdSrv

    fk_poses = _call_fk(current_joint_values, timeout)
    request = twoArmHandPoseCmd()
    request.use_custom_ik_param = True
    request.joint_angles_as_q0 = True
    request.ik_param = _make_ik_param(
        constraint_mode=constraint_mode,
        pos_cost_weight=pos_cost_weight,
    )
    request.hand_poses.left_pose.joint_angles = list(current_joint_values[:7])
    request.hand_poses.right_pose.joint_angles = list(current_joint_values[7:])
    request.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.left_pose.pos_xyz = list(fk_poses.left_pose.pos_xyz)
    request.hand_poses.left_pose.quat_xyzw = list(fk_poses.left_pose.quat_xyzw)
    request.hand_poses.right_pose.pos_xyz = list(right_pos)
    request.hand_poses.right_pose.quat_xyzw = (
        list(desired_quat) if desired_quat is not None else list(fk_poses.right_pose.quat_xyzw)
    )

    rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=timeout)
    response = rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)(request)
    if not response.success:
        raise RuntimeError(
            "/ik/two_arm_hand_pose_cmd_srv failed: "
            + getattr(response, "error_reason", "")
            + f" target={list(right_pos)} constraint_mode={constraint_mode} desired_quat={desired_quat}"
        )

    right_result = list(response.hand_poses.right_pose.joint_angles)
    if len(right_result) == 7:
        return list(current_joint_values[:7]) + right_result
    if len(response.q_arm) >= 14:
        return list(response.q_arm[:14])
    raise RuntimeError("IK response did not contain arm joints")


def _move_right_hand_ik_once(
    target_pub,
    traj_pub,
    right_pos,
    right_quat,
    timeout,
    label,
    constraint_mode=IK_MODE_THREE_POINT_MIXED,
    pos_cost_weight=0.0,
    move_time=ARM_MOVE_TIME,
    settle_time=FAST_GRASP_SETTLE_HOLD,
):
    import rospy

    current = _read_current_arm_joints(timeout)
    ik_q = _call_right_hand_ik(
        current,
        right_pos,
        timeout,
        desired_quat=right_quat,
        constraint_mode=constraint_mode,
        pos_cost_weight=pos_cost_weight,
    )
    cmd14 = _rad_to_deg(ik_q)
    _execute_arm_motion(target_pub, traj_pub, _rad_to_deg(current), cmd14, move_time, settle_time)

    actual, actual_quat = _measure_right_pose(timeout)
    pos_err = _axis_error(actual, right_pos, (True, True, True))
    quat_err = _quat_angle_error(actual_quat, right_quat) if right_quat is not None else None
    rospy.loginfo(
        "scene2 dataset: %s one-shot IK actual=%s pos_err=%.4f m quat_err=%s",
        label,
        [round(v, 4) for v in actual],
        pos_err,
        "%.1fdeg" % math.degrees(quat_err) if quat_err is not None else "n/a",
    )
    return pos_err, quat_err, actual, actual_quat, cmd14


def _pick_part_absolute(arm_pub, traj_pub, gripper_hold, job, hold_time):
    import rospy

    grasp_target = list(job["grasp"])
    rospy.loginfo(
        "scene2 dataset: %s grasp target=%s",
        job["object"],
        [round(v, 4) for v in grasp_target],
    )
    pos_err, quat_err, actual, actual_quat, _cmd14 = _move_right_hand_ik_once(
        arm_pub,
        traj_pub,
        grasp_target,
        RIGHT_HAND_QUAT_XYZW,
        TOPIC_TIMEOUT,
        f"{job['object']}_grasp",
        constraint_mode=IK_MODE_THREE_POINT_MIXED,
        pos_cost_weight=THREE_POINT_WEIGHT,
        move_time=ARM_MOVE_TIME,
        settle_time=hold_time,
    )
    if pos_err > GRASP_POSITION_TOLERANCE or quat_err > ORIENTATION_TOLERANCE_RAD:
        rospy.logwarn(
            "scene2 dataset: %s 抓取姿态未达到精度 (xyz_err=%.4f m / %.4f m, quat_err=%.1fdeg / %.1fdeg)",
            job["object"],
            pos_err,
            GRASP_POSITION_TOLERANCE,
            math.degrees(quat_err),
            math.degrees(ORIENTATION_TOLERANCE_RAD),
        )
    else:
        rospy.loginfo(
            "scene2 dataset: %s 抓取姿态到位 xyz_err=%.4f m quat_err=%.1fdeg，闭合右夹爪",
            job["object"],
            pos_err,
            math.degrees(quat_err),
        )

    _publish_right_gripper_close(gripper_hold)
    rospy.sleep(GRIPPER_CLOSE_TIME)

    lift_target = [grasp_target[0], grasp_target[1], grasp_target[2] + LIFT_Z_OFFSET]
    lift_err, _lift_quat_err, _lift_actual, _lift_quat, _lift_cmd14 = _move_right_hand_ik_once(
        arm_pub,
        traj_pub,
        lift_target,
        actual_quat,
        TOPIC_TIMEOUT,
        f"{job['object']}_lift",
        constraint_mode=IK_MODE_POS_HARD_ORI_SOFT,
        pos_cost_weight=0.0,
        move_time=ARM_MOVE_TIME,
        settle_time=hold_time,
    )
    rospy.loginfo(
        "scene2 dataset: %s lift finished xyz_err=%.4f m",
        job["object"],
        lift_err,
    )


def _place_part_absolute(arm_pub, traj_pub, gripper_hold, job, hold_time):
    import rospy

    rospy.loginfo(
        "scene2 dataset: %s move through high transport pose before placing",
        job["object"],
    )
    _move_arm_to(
        arm_pub,
        traj_pub,
        PREGRASP_POINTS_DEG[2],
        move_time=ARM_MOVE_TIME,
        settle=FAST_GRASP_SETTLE_HOLD,
    )

    place_target = list(job["place"])
    place_err, quat_err, actual, _actual_quat, _cmd14 = _move_right_hand_ik_once(
        arm_pub,
        traj_pub,
        place_target,
        None,
        TOPIC_TIMEOUT,
        f"{job['object']}_place_{job['bin']}",
        constraint_mode=IK_MODE_POS_HARD_ORI_SOFT,
        pos_cost_weight=0.0,
        move_time=ARM_MOVE_TIME,
        settle_time=hold_time,
    )
    rospy.loginfo(
        "scene2 dataset: %s place release actual=%s xyz_err=%.4f m quat_err=%s，打开右夹爪",
        job["object"],
        [round(v, 4) for v in actual],
        place_err,
        "%.1fdeg" % math.degrees(quat_err) if quat_err is not None else "n/a",
    )
    _publish_gripper_open(gripper_hold)
    rospy.sleep(PLACE_DWELL)

    _move_arm_to(
        arm_pub,
        traj_pub,
        PREGRASP_POINTS_DEG[2],
        move_time=ARM_MOVE_TIME,
        settle=ARM_SETTLE_TIME,
    )


def _run_absolute_pick_place_jobs(arm_pub, traj_pub, gripper_hold, jobs, hold_time):
    import rospy

    for index, job in enumerate(jobs, start=1):
        rospy.loginfo(
            "scene2 dataset: pick/place job %d/%d %s -> %s grasp=%s place=%s",
            index,
            len(jobs),
            job["object"],
            job["bin"],
            [round(v, 4) for v in job["grasp"]],
            [round(v, 4) for v in job["place"]],
        )
        _publish_gripper_open(gripper_hold)
        _pick_part_absolute(arm_pub, traj_pub, gripper_hold, job, hold_time)
        _place_part_absolute(arm_pub, traj_pub, gripper_hold, job, hold_time)


def _motion_points():
    return PREGRASP_POINTS_DEG[1:], [PREGRASP_POINTS_DEG[1], PREGRASP_POINTS_DEG[0]]


def _run_fixed_grasp_motion(gripper_hold):
    import rospy
    from sensor_msgs.msg import JointState
    from kuavo_msgs.msg import armTargetPoses

    arm_pub = rospy.Publisher(ARM_TARGET_POSES_TOPIC, armTargetPoses, queue_size=10)
    traj_pub = rospy.Publisher(ARM_TRAJ_TOPIC, JointState, queue_size=10)
    _wait_for_connection(arm_pub, TOPIC_TIMEOUT)
    _wait_for_connection(traj_pub, TOPIC_TIMEOUT)

    _publish_gripper_open(gripper_hold)

    pregrasp_points, retract_points = _motion_points()

    for point in pregrasp_points:
        _move_arm_to(arm_pub, traj_pub, point)

    _run_absolute_pick_place_jobs(
        arm_pub,
        traj_pub,
        gripper_hold,
        PICK_PLACE_JOBS,
        FAST_GRASP_SETTLE_HOLD,
    )

    for point in retract_points:
        _move_arm_to(arm_pub, traj_pub, point)
    rospy.loginfo("scene2 dataset: retracted through pregrasp")


def _start_rosbag(bag_path, topics):
    cmd = ["rosbag", "record", "-O", bag_path] + topics
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, cmd


def _run_once(args, run_index=1, run_total=1):
    if args.headless:
        os.environ["MUJOCO_HEADLESS"] = "1"

    run_tag = f"{SCENE_NAME}_{_now_tag()}"
    if run_total > 1:
        run_tag += f"_run_{run_index:03d}"
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    bag_path = os.path.join(output_dir, f"{run_tag}.bag")
    topics = list(DEFAULT_TOPICS)

    launch_proc = None
    bag_proc = None
    gripper_hold = None
    arm_mode_changed = False
    status = "failed"
    try:
        if args.use_existing_sim:
            _init_ros_node("scene2_dataset_collector")
            missing = _wait_for_topics(["/sensors_data_raw"], LAUNCH_TIMEOUT)
            if missing:
                raise RuntimeError("existing simulation is not ready: " + ", ".join(missing))
        else:
            launch_proc, _launch_cmd = _start_scene_launch()
            _wait_for_roscore(launch_proc, timeout=30.0)
            _init_ros_node("scene2_dataset_collector")

        topic_wait_timeout = TOPIC_TIMEOUT if args.use_existing_sim else LAUNCH_TIMEOUT
        missing_topics = _wait_for_topics(["/sensors_data_raw"], topic_wait_timeout)
        if missing_topics:
            raise RuntimeError("required topics missing: " + ", ".join(missing_topics))

        _publish_head_target(TOPIC_TIMEOUT)
        gripper_hold = _start_gripper_hold(TOPIC_TIMEOUT)

        bag_proc, _bag_cmd = _start_rosbag(bag_path, topics)
        time.sleep(1.0 + RECORD_SETTLE_TIME)

        _set_arm_mode(ARM_MODE_EXTERNAL_CONTROL, timeout=TOPIC_TIMEOUT)
        arm_mode_changed = True
        _run_fixed_grasp_motion(gripper_hold)
        _set_arm_mode(ARM_MODE_AUTO_SWING, timeout=TOPIC_TIMEOUT)
        arm_mode_changed = False

        time.sleep(POST_ARM_MODE_RECORD_TIME)
        status = "success"
    finally:
        _terminate_process_group(bag_proc, signal.SIGINT, timeout=10)
        if gripper_hold is not None:
            gripper_hold.stop()
        if arm_mode_changed:
            try:
                _set_arm_mode(ARM_MODE_AUTO_SWING, timeout=TOPIC_TIMEOUT)
            except Exception as exc:
                print(f"[WARN] failed to restore arm mode: {exc}")
        _terminate_process_group(launch_proc, signal.SIGINT, timeout=20)
        print(f"[INFO] scene2 dataset run {status}: {bag_path}")

    return 0 if status == "success" else 1


def _child_args_for_run(args, run_index, run_total):
    child_args = [
        sys.executable,
        os.path.abspath(__file__),
        "--output-dir", args.output_dir,
        "--duration", str(args.duration),
        "--count", "1",
        "--run-index", str(run_index),
        "--run-total", str(run_total),
    ]
    if args.headless:
        child_args.append("--headless")
    return child_args


def _run_count(args):
    if args.count < 1:
        raise ValueError("--count must be >= 1")
    if args.count == 1:
        return _run_once(args, args.run_index, args.run_total)
    if args.use_existing_sim:
        raise ValueError("--count > 1 cannot be used with --use-existing-sim")

    for index in range(1, args.count + 1):
        print(f"[INFO] scene2 dataset batch {index}/{args.count}: start")
        result = subprocess.run(_child_args_for_run(args, index, args.count))
        if result.returncode != 0:
            print(
                f"[ERROR] scene2 dataset batch {index}/{args.count} failed with code {result.returncode}",
                file=sys.stderr,
            )
            return result.returncode
        print(f"[INFO] scene2 dataset batch {index}/{args.count}: complete")
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect scene2 fixed-grasp rosbag datasets.")
    parser.add_argument("--count", type=int, default=1, help="Number of simulation restart-and-record cycles to run.")
    parser.add_argument("--run-index", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--run-total", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for bag files.")
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Deprecated compatibility option; recording now stops 1s after arm mode is restored.",
    )
    parser.add_argument("--headless", action="store_true", help="Set MUJOCO_HEADLESS=1 for this run.")
    parser.add_argument("--use-existing-sim", action="store_true", help="Attach to an already running scene2 simulation.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return _run_count(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
