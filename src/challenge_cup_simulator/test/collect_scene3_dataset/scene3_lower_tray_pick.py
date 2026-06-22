#!/usr/bin/env python3
"""Scene3 lower tray pick: approach shelf, squat, grasp lower tray, stand, step back."""

import argparse
import os
import sys
import yaml

from scene3_body_control import bend_forward, squat, stand, stand_straight, step_back
from scene3_navigation_utils import Scene3ShelfNavigator
from scene3_rosbag_utils import (
    init_ros_node,
    start_scene3_challenge_task,
    terminate_process_group,
    wait_for_roscore,
    wait_for_topics,
)
from scene3_success_checker import Scene3SuccessChecker
from scene3_tray_grasp_expert import Scene3TrayGraspExpert, resolve_pose_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "configs", "scene3_lower_tray_pick.yaml")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Scene3 lower tray pick pipeline")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument("--seed", type=int, default=0, help="scene seed")
    parser.add_argument("--launch-sim", action="store_true", help="launch Scene3 sim")
    parser.add_argument("--headless", action="store_true", help="headless sim")
    parser.add_argument("--approach-distance", type=float, default=None, help="override approach shelf distance (m)")
    parser.add_argument("--bend-angle", type=float, default=None, help="override bend forward angle (degrees)")
    parser.add_argument("--step-back-distance", type=float, default=None, help="override step back distance (m)")
    parser.add_argument("--nav-open-loop", action="store_true", help="force open-loop shelf approach")
    parser.add_argument("--print-plan", action="store_true", help="print arm motion plan and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.approach_distance is not None:
        cfg.setdefault("navigation", {})["approach_shelf_distance"] = args.approach_distance
    if args.bend_angle is not None:
        cfg.setdefault("bend", {})["angle_deg"] = args.bend_angle
    if args.step_back_distance is not None:
        cfg.setdefault("bend", {})["step_back_distance"] = args.step_back_distance
    if args.nav_open_loop:
        cfg.setdefault("navigation", {})["force_open_loop"] = True

    pose_config_path = resolve_pose_config(SCRIPT_DIR, cfg)

    if args.print_plan:
        expert = Scene3TrayGraspExpert(cfg, pose_config_path, observer=None)
        expert.print_plan()
        return 0

    launch_sim = args.launch_sim or not cfg.get("scene", {}).get("use_existing_sim", True)
    sim_proc = None

    try:
        if launch_sim:
            sim_proc, cmd = start_scene3_challenge_task(args.seed, headless=args.headless)
            print("[INFO] launched scene3 sim: " + " ".join(cmd))
            wait_for_roscore(sim_proc, timeout=cfg.get("episode", {}).get("launch_timeout_sec", 120.0))

        init_ros_node("scene3_lower_tray_pick")

        topics_cfg = cfg.get("topics", {})
        missing = wait_for_topics(
            [
                topics_cfg.get("sensors", "/sensors_data_raw"),
                topics_cfg.get("rgb", "/cam_h/color/image_raw/compressed"),
                topics_cfg.get("depth", "/cam_h/depth/image_raw/compressedDepth"),
                topics_cfg.get("camera_info", "/cam_h/color/camera_info"),
            ],
            cfg.get("episode", {}).get("topic_wait_timeout_sec", 20.0),
        )
        if missing:
            raise RuntimeError("missing topics: " + ",".join(missing))

        navigator = Scene3ShelfNavigator(cfg)

        # Step 1: Approach shelf
        print("[STEP 1/5] Approaching shelf ...")
        navigator.approach_shelf()
        print("[STEP 1/5] Done.")

        # Step 2: Squat first, then bend forward
        import math
        bend_cfg = cfg.get("bend", {})
        height_delta = float(bend_cfg.get("height_delta", -0.25))
        angle_deg = float(bend_cfg.get("angle_deg", 20))
        angle_rad = math.radians(angle_deg)

        print(f"[STEP 2a] Squatting {height_delta:.2f}m ...")
        squat(height_delta=height_delta, duration=float(bend_cfg.get("squat_duration", 2.0)))

        print(f"[STEP 2b] Bending forward {angle_deg:.0f}deg ...")
        bend_forward(angle_rad=angle_rad, duration=float(bend_cfg.get("bend_duration", 2.0)))
        print("[STEP 2] Done.")

        # Step 3: Grasp lower tray
        print("[STEP 3] Grasping lower tray ...")
        expert = Scene3TrayGraspExpert(cfg, pose_config_path, observer=None)
        expert.setup_ros(timeout=cfg.get("episode", {}).get("topic_wait_timeout_sec", 20.0))
        checker = Scene3SuccessChecker(cfg)
        _run_grasp(expert, cfg, checker)
        print("[STEP 3] Done.")

        # Step 4: Unbend first, then rise (reverse of step 2)
        print("[STEP 4a] Standing straight (unbend) ...")
        stand_straight(duration=float(bend_cfg.get("stand_duration", 2.0)))

        print("[STEP 4b] Rising to normal height ...")
        stand(duration=float(bend_cfg.get("rise_duration", 2.0)))
        print("[STEP 4] Done.")

        # Step 5: Extract + stow at standing height, then step back
        print("[STEP 5] Extract + stow + step back ...")
        _run_extract_stow(expert, cfg, checker)
        step_back(
            distance=float(bend_cfg.get("step_back_distance", 0.30)),
            speed=float(bend_cfg.get("step_back_speed", 0.10)),
        )
        print("[STEP 5/5] Done.")

        print("[INFO] Lower tray pick pipeline completed successfully.")
        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if sim_proc is not None:
            terminate_process_group(sim_proc)


def _run_grasp(expert, cfg, checker):
    """Pregrasp -> approach -> close gripper -> retract to pregrasp (robot is squatted+bent)."""
    import rospy

    expert.set_arm_mode_external()
    expert.publish_head_target()
    expert.open_gripper()

    _move_and_log(expert, checker, "safe_home")
    if "scene3_ready_pose_lower" in expert.poses:
        _move_and_log(expert, checker, "scene3_ready_pose_lower")

    for name in expert._waypoints("pregrasp_waypoints", ["lower_tray_pregrasp"]):
        _move_and_log(expert, checker, name)
    expert.zero_hand_roll()
    rospy.sleep(cfg.get("episode", {}).get("settle_before_record_sec", 0.6))

    for name in expert._waypoints("approach_waypoints", ["lower_tray_edge_approach"]):
        _move_and_log(expert, checker, name)
    expert.zero_hand_roll()

    print("  gripper: close")
    _log_distance(checker, "close_gripper")
    if not expert.close_gripper():
        raise RuntimeError("CLAW_FAILED: close command returned false")

    # Retract arm back to pregrasp while body is still low (tray secured near body)
    print("  arm: retract to pregrasp (before standing)")
    retract_names = list(reversed(expert._waypoints("pregrasp_waypoints", ["lower_tray_pregrasp"])))
    for name in retract_names:
        _move_and_log(expert, checker, name)


def _run_extract_stow(expert, cfg, checker):
    """Extract tray + stow at waist (robot is standing)."""
    import rospy

    for name in expert._waypoints("extract_waypoints", ["upper_tray_extract_mid", "upper_tray_extract_out"]):
        _move_and_log(expert, checker, name)

    for name in expert._waypoints("stow_waypoints", ["upper_tray_shelf_clearance", "waist_stow_pose", "finish_hold_pose"]):
        _move_and_log(expert, checker, name)

    rospy.sleep(cfg.get("episode", {}).get("settle_after_stow_sec", 1.0))


def _move_and_log(expert, checker, pose_name):
    print(f"  arm: {pose_name}")
    expert.move_to_named_pose(pose_name)
    _log_distance(checker, pose_name)


def _log_distance(checker, label):
    try:
        info = checker.measure_target_gripper_distance()
    except Exception as exc:
        print(f"  [DEBUG] {label}: distance unavailable ({exc})")
        return
    d = info.get("target_gripper_distance_m")
    dist = "nan" if d is None else f"{float(d):.3f}m"
    target = _fmt_xyz(info.get("target_base_xyz"))
    gripper = _fmt_xyz(info.get("active_gripper_base_xyz"))
    delta = _fmt_xyz(info.get("target_minus_gripper_base_xyz"))
    print(f"  [DEBUG] {label}: dist={dist}  target(base)={target}  gripper(base)={gripper}  delta={delta}")


def _fmt_xyz(values, ndigits=3):
    if values is None:
        return "None"
    return f"[{', '.join(f'{float(v):.{ndigits}f}' for v in values)}]"


if __name__ == "__main__":
    raise SystemExit(main())
