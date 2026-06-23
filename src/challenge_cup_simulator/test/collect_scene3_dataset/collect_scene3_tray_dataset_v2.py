#!/usr/bin/env python3
"""
collect_scene3_tray_dataset_v2.py — Scene3 upper-tray grasp script.

Flow: launch sim → navigate 1.3m → pregrasp → approach → close gripper → done.
No recording, no extract, no stow.
"""

import argparse
import copy
import os
import sys
import traceback

from core.config_loader import (
    DEFAULT_CONFIG, DEFAULT_ROI_CONFIG,
    deep_update, load_yaml, resolve_pose_config,
)
from scene3_navigation_utils import Scene3ShelfNavigator
from scene3_rosbag_utils import (
    init_ros_node, start_scene3_challenge_task, terminate_process_group,
    wait_for_roscore, wait_for_topics,
)
from scene3_success_checker import Scene3SuccessChecker
from scene3_tray_grasp_expert_v2 import Scene3TrayGraspExpert, resolve_pose_config as _resolve_pose

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(args):
    cfg = copy.deepcopy(load_yaml(args.config))
    use_existing_sim = bool(args.use_existing_sim)
    overrides = {
        "scene": {
            "target_slot": args.slot,
            "use_existing_sim": use_existing_sim,
            "navigation_included": not args.no_navigation,
        },
        "episode": {
            "max_attempts_per_seed": 1,
            "wait_for_user_ready": False,
        },
        "navigation": {
            "enabled": not args.no_navigation,
            "approach_shelf_distance": 1.3,
            "max_forward_speed_shelf": 0.15,
        },
        "expert": {
            "truth_ik": {
                "enabled": True,
                "stage_offsets": {
                    "pregrasp": {"x": -0.30, "y": 0.0, "z": 0.10, "duration": 2.0},
                    "approach": {"x": -0.15, "y": 0.0, "z": 0.10, "duration": 1.2},
                },
            },
        },
        "success": {
            "active_gripper_frame": "zarm_r7_end_effector",
            "gripper_tf_timeout_sec": 3.0,
        },
    }
    if args.approach_shelf_distance is not None:
        overrides["navigation"]["approach_shelf_distance"] = args.approach_shelf_distance
    if args.nav_open_loop:
        overrides["navigation"]["force_open_loop"] = True
    if args.no_truth_ik or args.named_pose_mode:
        overrides["expert"]["truth_ik"]["enabled"] = False
    if args.assume_at_shelf:
        overrides["episode"]["wait_for_user_ready"] = False
    if args.no_navigation:
        overrides["episode"]["wait_for_user_ready"] = True

    cfg = deep_update(cfg, overrides)
    cfg["scene"]["target_tray_name"] = "smt_tray_4"
    cfg["success"]["target_tray_name"] = "smt_tray_4"
    return cfg


class Scene3TrayGraspRunner:
    """Runs one grasp cycle: sim → nav → pregrasp IK → approach IK → close. No recording."""

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.sim_proc = None

    def run(self, seed):
        self.ensure_sim(seed)
        init_ros_node("scene3_grasp_v2")
        self.wait_topics()
        self.navigate()
        self.wait_ready(seed)

        timeout = self.cfg.get("episode", {}).get("topic_wait_timeout_sec", 20.0)
        pose_cfg_path = _resolve_pose(SCRIPT_DIR, self.cfg)
        expert = Scene3TrayGraspExpert(self.cfg, pose_cfg_path, observer=None)
        expert.setup_ros(timeout=timeout)
        expert.prepare_robot()
        checker = Scene3SuccessChecker(self.cfg)
        self.log_distance(checker, "pregrasp")

        expert.approach_tray_edge()
        self.log_distance(checker, "approach")

        if not expert.close_gripper():
            print("[ERROR] gripper close failed")
            expert.safe_stop()
            expert.shutdown()
            self.cleanup_sim()
            return

        # Lift tray 15cm up
        expert.lift_tray()
        self.log_distance(checker, "lift")

        # Stow to chest
        expert.stow_tray()
        self._sleep(1.0)
        self.log_distance(checker, "stow")

        print("[INFO] grasp complete — tray held at chest")
        self._sleep(2.0)
        expert.safe_stop()
        expert.shutdown()
        self.cleanup_sim()

    # ── infra ────────────────────────────────────────────────────────

    def ensure_sim(self, seed):
        if self.cfg.get("scene", {}).get("use_existing_sim", True):
            return
        self.sim_proc, cmd = start_scene3_challenge_task(seed, headless=bool(self.args.headless))
        print("[INFO] launched scene3 sim: " + " ".join(cmd))
        wait_for_roscore(self.sim_proc, timeout=self.cfg.get("episode", {}).get("launch_timeout_sec", 120.0))

    def wait_topics(self):
        topic_wait = self.cfg.get("episode", {}).get("topic_wait_timeout_sec", 20.0)
        topics = self.cfg.get("topics", {})
        required = [
            topics.get("sensors", "/sensors_data_raw"),
            topics.get("claw_state", "/leju_claw_state"),
        ]
        missing = wait_for_topics(required, topic_wait)
        if missing:
            raise RuntimeError("MISSING_TOPICS: " + ",".join(missing))

    def navigate(self):
        if not self.cfg.get("navigation", {}).get("enabled", False):
            return
        dist = self.cfg.get("navigation", {}).get("approach_shelf_distance", 1.3)
        print(f"[INFO] navigating forward {dist}m ...")
        navigator = Scene3ShelfNavigator(self.cfg)
        navigator.approach_shelf()
        print("[INFO] navigation complete.")

    def wait_ready(self, seed):
        if not self.cfg.get("episode", {}).get("wait_for_user_ready", True):
            return
        if not sys.stdin.isatty():
            raise RuntimeError("interactive confirmation required; use --assume-at-shelf")
        input(f"\n[READY] seed={seed}. Press Enter to start grasp, Ctrl-C to abort.\n> ")

    def log_distance(self, checker, stage):
        try:
            info = checker.measure_target_gripper_distance()
        except Exception as exc:
            print(f"[WARN] {stage} distance unavailable: {exc}")
            return
        d = info.get("target_gripper_distance_m")
        print(f"[INFO] {stage} dist={d:.3f}m "
              f"target_base={[round(float(v),3) for v in (info.get('target_base_xyz') or [])]} "
              f"delta={[round(float(v),3) for v in (info.get('target_minus_gripper_base_xyz') or [])]}")

    def cleanup_sim(self):
        if self.sim_proc is not None and not self.cfg.get("scene", {}).get("use_existing_sim", True):
            terminate_process_group(self.sim_proc)
            self.sim_proc = None

    @staticmethod
    def _sleep(seconds):
        import rospy
        rospy.sleep(float(seconds))


# ── CLI ──────────────────────────────────────────────────────────────

def build_arg_parser():
    p = argparse.ArgumentParser(description="Scene3 upper-tray grasp (v2).\n"
        "Flow: launch sim → nav 1.3m → pregrasp → approach → close gripper → done.\n"
        "Examples:\n"
        "  python3 collect_scene3_tray_dataset_v2.py --seed 0\n"
        "  python3 collect_scene3_tray_dataset_v2.py --seed 0 --use-existing-sim\n"
        "  python3 collect_scene3_tray_dataset_v2.py --seed 0 --no-navigation --assume-at-shelf",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slot", choices=["upper"], default="upper")

    p.add_argument("--use-existing-sim", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-navigation", action="store_true")
    p.add_argument("--approach-shelf-distance", type=float, default=None)
    p.add_argument("--nav-open-loop", action="store_true")
    p.add_argument("--assume-at-shelf", action="store_true")
    p.add_argument("--print-plan", action="store_true")
    p.add_argument("--no-truth-ik", action="store_true")
    p.add_argument("--named-pose-mode", action="store_true")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args)

    if args.print_plan:
        pose_path = _resolve_pose(SCRIPT_DIR, cfg)
        expert = Scene3TrayGraspExpert(cfg, pose_path, observer=None)
        expert.print_plan()
        return 0

    runner = Scene3TrayGraspRunner(cfg, args)
    try:
        runner.run(args.seed)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
