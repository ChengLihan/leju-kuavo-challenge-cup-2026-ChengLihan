#!/usr/bin/env python3
"""Collect scene2 data with one fixed IK grasp attempt for the nylon fitting."""

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
TARGET_OBJECT = "part_type_b_1"
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

# Fixed scene2 grasp pose read from live FK after manual adjustment.
FIXED_RIGHT_HAND_TARGET_XYZ = [0.301627, -0.041467, -0.084687]
RIGHT_HAND_QUAT_XYZW = [-0.173916, -0.389729, 0.887919, 0.171654]

HEAD_TARGET = [0.0, 20.0]
HEAD_SETTLE_TIME = 0.8
ARM_MODE_EXTERNAL_CONTROL = 2
ARM_MODE_AUTO_SWING = 1
ARM_MODE_SERVICE = "/arm_traj_change_mode"
# 与可用的 keyboard 脚本对齐：经 /kuavo_arm_target_poses -> humanoid_Arm_time_target_control
# -> /humanoid_mpc_target_arm 进入 OCS2 MPC，是本控制器真正驱动手臂的路径；
# 直接发 /kuavo_arm_traj 不走 MPC，执行不稳定。
ARM_TARGET_POSES_TOPIC = "/kuavo_arm_target_poses"
ARM_MOVE_TIME = 4.0       # 每个航点由 MPC 在该时间内平滑插值到位（秒）
ARM_SETTLE_TIME = 0.5     # 到位后额外稳定时间（秒）
REACH_TOLERANCE = 0.03    # FK 验证：右手末端与目标的允许误差（米）
RECORD_SETTLE_TIME = 2.0
RUN_COOLDOWN = 2.0
TOPIC_TIMEOUT = 20.0
LAUNCH_TIMEOUT = 120.0
BAG_PREFIX = "dataset"
RIGHT_GRIPPER_OPEN = 0.0
LEFT_GRIPPER_OPEN = 0.0


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


def _load_challenge_launcher():
    utils_dir = os.path.join(_sim_pkg_path(), "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
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


def _right_hand_target():
    import rospy

    rospy.loginfo(
        "scene2 dataset: %s fixed hand_target=%s",
        TARGET_OBJECT,
        ["%.4f" % v for v in FIXED_RIGHT_HAND_TARGET_XYZ],
    )
    return list(FIXED_RIGHT_HAND_TARGET_XYZ)


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


def _publish_gripper_open(pub):
    import rospy
    from sensor_msgs.msg import JointState

    for _ in range(12):
        if rospy.is_shutdown():
            break
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["left_gripper_joint", "right_gripper_joint"]
        msg.position = [LEFT_GRIPPER_OPEN, RIGHT_GRIPPER_OPEN]
        pub.publish(msg)
        rospy.sleep(0.05)


def _publish_arm_target_poses(pub, degrees_list, move_time):
    """发布 armTargetPoses（与 keyboard 脚本一致）：values 为 14 个关节角(度)，
    times=[move_time] 表示在 move_time 秒内由 MPC 平滑插值到位。"""
    from kuavo_msgs.msg import armTargetPoses

    msg = armTargetPoses()
    msg.times = [float(move_time)]
    msg.values = [float(v) for v in degrees_list]
    pub.publish(msg)


def _move_arm_to(pub, degrees_list, move_time=ARM_MOVE_TIME, settle=ARM_SETTLE_TIME):
    import rospy

    _publish_arm_target_poses(pub, degrees_list, move_time)
    rospy.sleep(move_time + settle)


def _verify_right_hand_reach(target_xyz, timeout):
    """发布后用当前关节做 FK，验证右手末端是否真的到达目标，返回 (误差m, 实际xyz)。"""
    import rospy

    current_q = _read_current_arm_joints(timeout)
    poses = _call_fk(current_q, timeout)
    actual = list(poses.right_pose.pos_xyz)
    err = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, target_xyz)))
    rospy.loginfo(
        "scene2 dataset: FK reach check target=%s actual=%s err=%.4f m",
        [round(v, 4) for v in target_xyz], [round(v, 4) for v in actual], err,
    )
    return err, actual


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


def _make_ik_param():
    from kuavo_msgs.msg import ikSolveParam

    param = ikSolveParam()
    param.major_optimality_tol = 1e-3
    param.major_feasibility_tol = 1e-3
    param.minor_feasibility_tol = 1e-3
    param.major_iterations_limit = 100
    param.oritation_constraint_tol = 1e-3
    param.pos_constraint_tol = 1e-3
    param.pos_cost_weight = 1.0
    param.constraint_mode = 0x06
    return param


def _call_fk(joint_angles, timeout):
    import rospy
    from kuavo_msgs.srv import fkSrv

    rospy.wait_for_service("/ik/fk_srv", timeout=timeout)
    response = rospy.ServiceProxy("/ik/fk_srv", fkSrv)(joint_angles)
    if not response.success:
        raise RuntimeError("/ik/fk_srv returned success=false")
    return response.hand_poses


def _call_right_hand_ik(current_joint_values, right_pos, timeout):
    import rospy
    from kuavo_msgs.msg import twoArmHandPoseCmd
    from kuavo_msgs.srv import twoArmHandPoseCmdSrv

    fk_poses = _call_fk(current_joint_values, timeout)
    request = twoArmHandPoseCmd()
    request.use_custom_ik_param = True
    request.joint_angles_as_q0 = True
    request.ik_param = _make_ik_param()
    request.hand_poses.left_pose.joint_angles = list(current_joint_values[:7])
    request.hand_poses.right_pose.joint_angles = list(current_joint_values[7:])
    request.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
    request.hand_poses.left_pose.pos_xyz = list(fk_poses.left_pose.pos_xyz)
    request.hand_poses.left_pose.quat_xyzw = list(fk_poses.left_pose.quat_xyzw)
    request.hand_poses.right_pose.pos_xyz = list(right_pos)
    request.hand_poses.right_pose.quat_xyzw = list(RIGHT_HAND_QUAT_XYZW)

    rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=timeout)
    response = rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)(request)
    if not response.success:
        raise RuntimeError(
            "/ik/two_arm_hand_pose_cmd_srv failed: "
            + getattr(response, "error_reason", "")
        )

    right_result = list(response.hand_poses.right_pose.joint_angles)
    if len(right_result) == 7:
        return list(current_joint_values[:7]) + right_result
    if len(response.q_arm) >= 14:
        return list(response.q_arm[:14])
    raise RuntimeError("IK response did not contain arm joints")


def _run_fixed_grasp_motion(scene_file):
    import rospy
    from sensor_msgs.msg import JointState
    from kuavo_msgs.msg import armTargetPoses

    arm_pub = rospy.Publisher(ARM_TARGET_POSES_TOPIC, armTargetPoses, queue_size=10)
    gripper_pub = rospy.Publisher("/gripper/command", JointState, queue_size=10)
    _wait_for_connection(arm_pub, TOPIC_TIMEOUT)
    _wait_for_connection(gripper_pub, TOPIC_TIMEOUT)

    _publish_gripper_open(gripper_pub)

    # 经过预抓取序列：每个航点交给 MPC 在 ARM_MOVE_TIME 内平滑插值到位
    for point in PREGRASP_POINTS_DEG:
        _move_arm_to(arm_pub, point)

    # 求解右手 IK 并移动到抓取位姿
    right_target = _right_hand_target()
    current_q = _read_current_arm_joints(TOPIC_TIMEOUT)
    ik_q = _call_right_hand_ik(current_q, right_target, TOPIC_TIMEOUT)
    ik_point_deg = _rad_to_deg(ik_q)
    _move_arm_to(arm_pub, ik_point_deg)

    # 到位后用 FK 验证末端是否真的到达目标（IK success 不代表实际到位）
    err, _actual = _verify_right_hand_reach(right_target, TOPIC_TIMEOUT)
    if err > REACH_TOLERANCE:
        rospy.logwarn(
            "scene2 dataset: right hand未到达目标 (err=%.4f m > %.4f m)",
            err, REACH_TOLERANCE,
        )

    # 缩回预抓取序列
    for point in reversed(PREGRASP_POINTS_DEG):
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


def _run_single(args):
    if args.headless:
        os.environ["MUJOCO_HEADLESS"] = "1"

    run_tag = f"{SCENE_NAME}_seed_{args.seed}_{_now_tag()}"
    run_dir = os.path.abspath(os.path.join(args.output_dir, run_tag))
    os.makedirs(run_dir, exist_ok=True)

    bag_path = os.path.join(run_dir, f"{BAG_PREFIX}_{SCENE_NAME}_seed_{args.seed}.bag")
    metadata_path = os.path.join(run_dir, "metadata.json")
    topics = list(DEFAULT_TOPICS)
    scene_file = _scene_file_path()

    metadata = {
        "scene": SCENE_NAME,
        "seed": args.seed,
        "target_object": TARGET_OBJECT,
        "target_object_alias": TARGET_OBJECT_ALIAS,
        "target_bin": TARGET_BIN,
        "fixed_right_hand_target_xyz": FIXED_RIGHT_HAND_TARGET_XYZ,
        "right_hand_quat_xyzw": RIGHT_HAND_QUAT_XYZW,
        "topics": topics,
        "bag_path": bag_path,
        "status": "starting",
        "started_at": _datetime.datetime.now().isoformat(),
    }
    _write_metadata(metadata_path, metadata)

    launcher = None
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
            ChallengeSimLauncher = _load_challenge_launcher()
            launcher = ChallengeSimLauncher(scene=SCENE_NAME, seed=args.seed)
            launcher.start(node_name="scene2_dataset_collector", timeout=LAUNCH_TIMEOUT)
            scene_file = getattr(launcher, "_scene_file", None) or scene_file

        metadata["scene_file"] = scene_file
        metadata["scene_file_sha256"] = _sha256_file(scene_file)
        metadata["status"] = "simulation_ready"
        _write_metadata(metadata_path, metadata)

        missing_topics = _wait_for_topics(["/sensors_data_raw"], TOPIC_TIMEOUT)
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

        record_started = time.time()
        _publish_head_target(TOPIC_TIMEOUT)
        _set_arm_mode(ARM_MODE_EXTERNAL_CONTROL, timeout=TOPIC_TIMEOUT)
        arm_mode_changed = True
        _run_fixed_grasp_motion(scene_file)
        _set_arm_mode(ARM_MODE_AUTO_SWING, timeout=TOPIC_TIMEOUT)
        arm_mode_changed = False

        while time.time() - record_started < args.duration:
            time.sleep(0.2)
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
        if launcher is not None:
            launcher.stop()

        metadata["finished_at"] = _datetime.datetime.now().isoformat()
        metadata["status"] = status
        if os.path.isfile(bag_path):
            metadata["bag_size_bytes"] = os.path.getsize(bag_path)
        _write_metadata(metadata_path, metadata)
        print(f"[INFO] scene2 dataset run {status}: {run_dir}")

    return 0 if status == "success" else 1


def _expand_seeds(args):
    if args.seeds:
        return args.seeds
    return list(range(args.seed_start, args.seed_start + args.count))


def _child_args(args, seed):
    child = [
        sys.executable,
        os.path.abspath(__file__),
        "--single-run",
        "--seed",
        str(seed),
        "--output-dir",
        os.path.abspath(args.output_dir),
        "--duration",
        str(args.duration),
    ]
    if args.headless:
        child.append("--headless")
    if args.use_existing_sim:
        child.append("--use-existing-sim")
    return child


def _run_batch(args):
    os.makedirs(args.output_dir, exist_ok=True)
    failures = []
    seeds = _expand_seeds(args)
    print(f"[INFO] scene2 dataset seeds: {seeds}")
    for index, seed in enumerate(seeds, start=1):
        print(f"[INFO] starting run {index}/{len(seeds)} seed={seed}")
        result = subprocess.run(_child_args(args, seed))
        if result.returncode != 0:
            failures.append(seed)
        if index < len(seeds):
            time.sleep(RUN_COOLDOWN)
    if failures:
        print(f"[ERROR] failed seeds: {failures}")
        return 1
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect scene2 fixed-grasp rosbag datasets.")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", type=int, nargs="+", help="Explicit seed list.")
    seed_group.add_argument("--seed-start", type=int, default=0, help="First seed when --seeds is not set.")
    parser.add_argument("--count", type=int, default=1, help="Number of sequential seeds to collect.")
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for run folders.")
    parser.add_argument("--duration", type=float, default=20.0, help="Recording duration after arm motion starts.")
    parser.add_argument("--headless", action="store_true", help="Set MUJOCO_HEADLESS=1 for this run.")
    parser.add_argument("--use-existing-sim", action="store_true", help="Attach to an already running scene2 simulation.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be >= 1")
    if args.duration <= 0:
        parser.error("--duration must be > 0")
    if args.single_run:
        return _run_single(args)
    return _run_batch(args)


if __name__ == "__main__":
    sys.exit(main())
