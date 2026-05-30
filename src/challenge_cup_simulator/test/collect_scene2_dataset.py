#!/usr/bin/env python3
"""Collect scene2 data with fixed absolute pick/place attempts for nylon fittings."""

import argparse
import datetime as _datetime
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
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
    "/kuavo_arm_target_poses",
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
BAG_PREFIX = "dataset"
RIGHT_GRIPPER_OPEN = 0.0
LEFT_GRIPPER_OPEN = 0.0
RIGHT_GRIPPER_CLOSE = 255.0

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


def _sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sim_pkg_path():
    try:
        import rospkg

        return rospkg.RosPack().get_path("challenge_cup_simulator")
    except Exception:
        return os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def _scene_file_path():
    return os.path.join(
        _sim_pkg_path(),
        "models",
        "biped_s52",
        "xml",
        f"{SCENE_NAME}.xml",
    )


def _start_scene_launch(log_path):
    log_file = open(log_path, "w")
    cmd = [
        "roslaunch",
        "challenge_cup_simulator",
        "load_kuavo_mujoco_challenge.launch",
        f"scene_name:={SCENE_NAME}",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, log_file, cmd


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


def _publish_gripper(pub, left_cmd, right_cmd, repeats=12, dt=0.05):
    import rospy
    from sensor_msgs.msg import JointState

    for _ in range(repeats):
        if rospy.is_shutdown():
            break
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["left_gripper_joint", "right_gripper_joint"]
        msg.position = [float(left_cmd), float(right_cmd)]
        pub.publish(msg)
        rospy.sleep(dt)


def _publish_gripper_open(pub):
    _publish_gripper(pub, LEFT_GRIPPER_OPEN, RIGHT_GRIPPER_OPEN)


def _publish_right_gripper_close(pub):
    _publish_gripper(pub, LEFT_GRIPPER_OPEN, RIGHT_GRIPPER_CLOSE, repeats=20)


def _publish_arm_target_poses(pub, degrees_list, move_time):
    from kuavo_msgs.msg import armTargetPoses

    msg = armTargetPoses()
    msg.times = [float(move_time)]
    msg.values = [float(v) for v in degrees_list]
    pub.publish(msg)


def _move_arm_to(pub, degrees_list, move_time=ARM_MOVE_TIME, settle=ARM_SETTLE_TIME):
    import rospy

    _publish_arm_target_poses(pub, degrees_list, move_time)
    rospy.sleep(move_time + settle)


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
    pub,
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
    _publish_arm_target_poses(pub, cmd14, move_time)
    rospy.sleep(move_time + settle_time)

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


def _pick_part_absolute(arm_pub, gripper_pub, job, hold_time):
    import rospy

    grasp_target = list(job["grasp"])
    rospy.loginfo(
        "scene2 dataset: %s grasp target=%s",
        job["object"],
        [round(v, 4) for v in grasp_target],
    )
    pos_err, quat_err, actual, actual_quat, _cmd14 = _move_right_hand_ik_once(
        arm_pub,
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

    _publish_right_gripper_close(gripper_pub)
    rospy.sleep(GRIPPER_CLOSE_TIME)

    lift_target = [grasp_target[0], grasp_target[1], grasp_target[2] + LIFT_Z_OFFSET]
    lift_err, _lift_quat_err, _lift_actual, _lift_quat, _lift_cmd14 = _move_right_hand_ik_once(
        arm_pub,
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


def _place_part_absolute(arm_pub, gripper_pub, job, hold_time):
    import rospy

    rospy.loginfo(
        "scene2 dataset: %s move through high transport pose before placing",
        job["object"],
    )
    _move_arm_to(
        arm_pub,
        PREGRASP_POINTS_DEG[2],
        move_time=ARM_MOVE_TIME,
        settle=FAST_GRASP_SETTLE_HOLD,
    )

    place_target = list(job["place"])
    place_err, quat_err, actual, _actual_quat, _cmd14 = _move_right_hand_ik_once(
        arm_pub,
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
    _publish_gripper_open(gripper_pub)
    rospy.sleep(PLACE_DWELL)

    _move_arm_to(
        arm_pub,
        PREGRASP_POINTS_DEG[2],
        move_time=ARM_MOVE_TIME,
        settle=ARM_SETTLE_TIME,
    )


def _run_absolute_pick_place_jobs(arm_pub, gripper_pub, jobs, hold_time):
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
        _publish_gripper_open(gripper_pub)
        _pick_part_absolute(arm_pub, gripper_pub, job, hold_time)
        _place_part_absolute(arm_pub, gripper_pub, job, hold_time)


def _motion_points():
    return PREGRASP_POINTS_DEG[1:], [PREGRASP_POINTS_DEG[1], PREGRASP_POINTS_DEG[0]]


def _run_fixed_grasp_motion():
    import rospy
    from sensor_msgs.msg import JointState
    from kuavo_msgs.msg import armTargetPoses

    arm_pub = rospy.Publisher(ARM_TARGET_POSES_TOPIC, armTargetPoses, queue_size=10)
    gripper_pub = rospy.Publisher("/gripper/command", JointState, queue_size=10)
    _wait_for_connection(arm_pub, TOPIC_TIMEOUT)
    _wait_for_connection(gripper_pub, TOPIC_TIMEOUT)

    _publish_gripper_open(gripper_pub)

    pregrasp_points, retract_points = _motion_points()

    for point in pregrasp_points:
        _move_arm_to(arm_pub, point)

    _run_absolute_pick_place_jobs(
        arm_pub,
        gripper_pub,
        PICK_PLACE_JOBS,
        FAST_GRASP_SETTLE_HOLD,
    )

    for point in retract_points:
        _move_arm_to(arm_pub, point)
    rospy.loginfo("scene2 dataset: retracted through pregrasp")


def _start_rosbag(bag_path, topics, log_path):
    log_file = open(log_path, "w")
    cmd = ["rosbag", "record", "-O", bag_path] + topics
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, log_file, cmd


def _write_metadata(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _run_once(args, run_index=1, run_total=1):
    if args.headless:
        os.environ["MUJOCO_HEADLESS"] = "1"

    run_tag = f"{SCENE_NAME}_{_now_tag()}"
    if run_total > 1:
        run_tag += f"_run_{run_index:03d}"
    run_dir = os.path.abspath(os.path.join(args.output_dir, run_tag))
    os.makedirs(run_dir, exist_ok=True)

    bag_path = os.path.join(run_dir, f"{BAG_PREFIX}_{SCENE_NAME}.bag")
    metadata_path = os.path.join(run_dir, "metadata.json")
    topics = list(DEFAULT_TOPICS)
    scene_file = _scene_file_path()

    metadata = {
        "scene": SCENE_NAME,
        "run_index": run_index,
        "run_total": run_total,
        "use_existing_sim": args.use_existing_sim,
        "target_object": PICK_PLACE_JOBS[0]["object"],
        "target_objects": [job["object"] for job in PICK_PLACE_JOBS],
        "target_object_alias": TARGET_OBJECT_ALIAS,
        "target_bin": TARGET_BIN,
        "pick_place_jobs": PICK_PLACE_JOBS,
        "fixed_right_hand_target_xyz": FIXED_RIGHT_HAND_TARGET_XYZ,
        "second_right_hand_target_xyz": SECOND_RIGHT_HAND_TARGET_XYZ,
        "bin_b_place_targets_xyz": BIN_B_PLACE_TARGETS_XYZ,
        "right_hand_quat_xyzw": RIGHT_HAND_QUAT_XYZW,
        "lift_z_offset": LIFT_Z_OFFSET,
        "place_dwell": PLACE_DWELL,
        "ik_constraint_mode_position": IK_MODE_POS_HARD_ORI_SOFT,
        "ik_constraint_mode_orientation": IK_MODE_THREE_POINT_MIXED,
        "ik_constraint_mode_lift": IK_MODE_POS_HARD_ORI_SOFT,
        "ik_three_point_weight": THREE_POINT_WEIGHT,
        "fast_grasp_settle_hold": FAST_GRASP_SETTLE_HOLD,
        "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
        "grasp_position_tolerance": GRASP_POSITION_TOLERANCE,
        "ik_strategy": "two_absolute_pick_place_jobs_one_shot_ik",
        "right_gripper_close": RIGHT_GRIPPER_CLOSE,
        "post_arm_mode_record_time": POST_ARM_MODE_RECORD_TIME,
        "topics": topics,
        "bag_path": bag_path,
        "roslaunch_command": None,
        "status": "starting",
        "started_at": _datetime.datetime.now().isoformat(),
    }
    _write_metadata(metadata_path, metadata)

    launch_proc = None
    launch_log = None
    bag_proc = None
    bag_log = None
    arm_mode_changed = False
    status = "failed"
    try:
        if args.use_existing_sim:
            _init_ros_node("scene2_dataset_collector")
            missing = _wait_for_topics(["/sensors_data_raw"], LAUNCH_TIMEOUT)
            if missing:
                raise RuntimeError("existing simulation is not ready: " + ", ".join(missing))
        else:
            launch_proc, launch_log, launch_cmd = _start_scene_launch(os.path.join(run_dir, "roslaunch.log"))
            metadata["roslaunch_command"] = launch_cmd
            _write_metadata(metadata_path, metadata)
            _wait_for_roscore(launch_proc, timeout=30.0)
            _init_ros_node("scene2_dataset_collector")

        metadata["scene_file"] = scene_file
        metadata["scene_file_sha256"] = _sha256_file(scene_file)
        metadata["status"] = "simulation_ready"
        _write_metadata(metadata_path, metadata)

        topic_wait_timeout = TOPIC_TIMEOUT if args.use_existing_sim else LAUNCH_TIMEOUT
        missing_topics = _wait_for_topics(["/sensors_data_raw"], topic_wait_timeout)
        if missing_topics:
            raise RuntimeError("required topics missing: " + ", ".join(missing_topics))

        bag_proc, bag_log, bag_cmd = _start_rosbag(
            bag_path,
            topics,
            os.path.join(run_dir, "rosbag_record.log"),
        )
        metadata["rosbag_command"] = bag_cmd
        metadata["status"] = "recording"
        _write_metadata(metadata_path, metadata)
        time.sleep(1.0 + RECORD_SETTLE_TIME)

        _publish_head_target(TOPIC_TIMEOUT)
        _set_arm_mode(ARM_MODE_EXTERNAL_CONTROL, timeout=TOPIC_TIMEOUT)
        arm_mode_changed = True
        _run_fixed_grasp_motion()
        _set_arm_mode(ARM_MODE_AUTO_SWING, timeout=TOPIC_TIMEOUT)
        arm_mode_changed = False

        time.sleep(POST_ARM_MODE_RECORD_TIME)
        status = "success"
    finally:
        _terminate_process_group(bag_proc, signal.SIGINT, timeout=10)
        if bag_log is not None:
            bag_log.close()
        if arm_mode_changed:
            try:
                _set_arm_mode(ARM_MODE_AUTO_SWING, timeout=TOPIC_TIMEOUT)
            except Exception as exc:
                print(f"[WARN] failed to restore arm mode: {exc}")
        _terminate_process_group(launch_proc, signal.SIGINT, timeout=20)
        if launch_log is not None:
            launch_log.close()

        metadata["finished_at"] = _datetime.datetime.now().isoformat()
        metadata["status"] = status
        if os.path.isfile(bag_path):
            metadata["bag_size_bytes"] = os.path.getsize(bag_path)
        _write_metadata(metadata_path, metadata)
        print(f"[INFO] scene2 dataset run {status}: {run_dir}")

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
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for run folders.")
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
