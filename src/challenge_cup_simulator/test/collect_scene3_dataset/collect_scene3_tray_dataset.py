#!/usr/bin/env python3
"""Scene3 upper-tray local-skill data collector.

This script starts from the shelf-front work position. It does not navigate,
choose targets, complete the outbound task, or serve as an evaluation policy.
"""

import argparse
import copy
import os
import sys
import traceback

import yaml

from scene3_observation_recorder import Scene3ObservationRecorder
from scene3_navigation_utils import Scene3ShelfNavigator
from scene3_rosbag_utils import (
    RosbagRecorder,
    append_jsonl,
    build_record_topics,
    ensure_repo_relative,
    init_ros_node,
    now_tag,
    start_scene3_challenge_task,
    terminate_process_group,
    wait_for_roscore,
    wait_for_topics,
)
from scene3_success_checker import Scene3SuccessChecker
from scene3_tray_grasp_expert import Scene3TrayGraspExpert, resolve_pose_config


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "configs", "scene3_collect.yaml")
DEFAULT_ROI_CONFIG = os.path.join(SCRIPT_DIR, "configs", "scene3_roi.yaml")


def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def compact_yaml(data):
    if not data:
        return ""
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=True, width=4096).strip()


def load_config(args):
    cfg = copy.deepcopy(load_yaml(args.config))
    use_existing_sim = cfg.get("scene", {}).get("use_existing_sim", True)
    if args.launch_sim or (args.full_run and not args.use_existing_sim):
        use_existing_sim = False
    elif args.use_existing_sim:
        use_existing_sim = True

    overrides = {
        "scene": {
            "target_slot": args.slot,
            "use_existing_sim": bool(use_existing_sim),
        },
        "record": {
            "output_dir": args.output_dir or cfg.get("record", {}).get("output_dir", "bags/scene3_tray_stow"),
        },
        "episode": {},
        "navigation": {},
    }
    if args.full_run or args.navigate_to_shelf:
        overrides["scene"]["navigation_included"] = True
        overrides["navigation"]["enabled"] = True
        overrides["episode"]["wait_for_user_ready"] = bool(args.confirm_after_navigation)
    if args.approach_shelf_distance is not None:
        overrides["navigation"]["approach_shelf_distance"] = args.approach_shelf_distance
    if args.nav_timeout is not None:
        overrides["navigation"]["approach_timeout_sec"] = args.nav_timeout
    if args.nav_open_loop:
        overrides["navigation"]["force_open_loop"] = True
    if args.record_rosbag:
        overrides["record"]["record_rosbag"] = True
    if args.no_rosbag:
        overrides["record"]["record_rosbag"] = False
    if args.record_derived_topics:
        overrides["record"]["record_derived_topics"] = True
    if args.no_delete_failed_bag:
        overrides["record"]["delete_failed_bag"] = False
    if args.min_camera_fps is not None:
        overrides["record"]["min_rgb_fps"] = args.min_camera_fps
    if args.skip_pointcloud_precheck:
        overrides.setdefault("pointcloud", {})["precheck_required"] = False
    if args.min_roi_points is not None:
        overrides.setdefault("pointcloud", {})["min_roi_points"] = args.min_roi_points
    if args.no_truth_ik or args.named_pose_mode:
        overrides.setdefault("expert", {}).setdefault("truth_ik", {})["enabled"] = False
    if args.repeat is not None:
        overrides["episode"]["repeat"] = args.repeat
    if args.max_attempts_per_seed is not None:
        overrides["episode"]["max_attempts_per_seed"] = args.max_attempts_per_seed
    if args.hold_seconds is not None:
        overrides["episode"]["hold_seconds"] = args.hold_seconds
    if args.debug_run_once:
        overrides["episode"]["repeat"] = 1
        overrides["episode"]["max_attempts_per_seed"] = 1
    if args.assume_at_shelf:
        overrides["episode"]["wait_for_user_ready"] = False

    cfg = deep_update(cfg, overrides)
    cfg.setdefault("scene", {})["target_tray_name"] = cfg.get("success", {}).get(
        "target_tray_name",
        cfg.get("scene", {}).get("target_tray_name", "smt_tray_5"),
    )
    return cfg


class Scene3TrayDatasetCollector:
    def __init__(self, cfg, roi_cfg, args):
        self.cfg = cfg
        self.roi_cfg = roi_cfg
        self.args = args
        self.output_dir = ensure_repo_relative(cfg.get("record", {}).get("output_dir", "bags/scene3_tray_stow"))
        self.success_manifest = os.path.join(self.output_dir, "success_manifest.txt")
        self.failed_manifest = os.path.join(self.output_dir, "failed_seeds.txt")
        self.sim_proc = None

    def collect_seed(self, seed):
        repeat = int(self.cfg.get("episode", {}).get("repeat", 1))
        max_attempts = int(self.cfg.get("episode", {}).get("max_attempts_per_seed", 5))
        valid_count = 0
        attempts = 0
        while valid_count < repeat:
            attempts += 1
            if attempts > max_attempts:
                self.write_failed(seed, None, "MAX_ATTEMPTS", {"attempts": attempts - 1})
                break
            ok = self.collect_once(seed, attempts)
            if ok:
                valid_count += 1
            elif self.args.debug_run_once:
                break
        return valid_count == repeat

    def collect_once(self, seed, attempt):
        episode_id = f"scene3_seed_{seed}_upper_{now_tag()}_attempt_{attempt}"
        bag_path = os.path.join(self.output_dir, episode_id + ".bag")
        metadata_path = os.path.join(self.output_dir, episode_id + ".yaml")
        recorder = None
        expert = None
        reason = "UNKNOWN_EXCEPTION"
        details = {}
        try:
            self.ensure_sim(seed)
            init_ros_node("scene3_tray_dataset_collector")
            self.wait_ready()
            self.navigate_to_shelf_if_requested()
            self.wait_for_shelf_front_ready(seed, episode_id)

            observer = Scene3ObservationRecorder(self.cfg, self.roi_cfg)
            if not observer.wait_for_observation(timeout=self.cfg.get("episode", {}).get("topic_wait_timeout_sec", 20.0)):
                raise RuntimeError("CAMERA_MISSING: observation topics did not produce messages")

            pose_config_path = resolve_pose_config(SCRIPT_DIR, self.cfg)
            expert = Scene3TrayGraspExpert(self.cfg, pose_config_path, observer=observer)
            expert.setup_ros(timeout=self.cfg.get("episode", {}).get("topic_wait_timeout_sec", 20.0))
            expert.prepare_robot()
            checker = Scene3SuccessChecker(self.cfg)
            self.log_target_gripper_distance(checker, "pregrasp")
            if self.args.preset_only:
                hold_seconds = float(self.cfg.get("episode", {}).get("hold_seconds", 0.0))
                if hold_seconds > 0.0:
                    self.sleep(hold_seconds)
                metadata = self.make_metadata(seed, episode_id, bag_path, status="preset_only", success=True)
                dump_yaml(metadata_path, metadata)
                return True

            pc_cfg = self.cfg.get("pointcloud", {})
            if pc_cfg.get("precheck_required", True):
                if not observer.check_roi_points():
                    raise RuntimeError(
                        "POINTCLOUD_EMPTY: not enough points in pregrasp ROI. "
                        "Move the robot to the shelf-front work position, check head camera/TF, "
                        "or run once with --skip-pointcloud-precheck while tuning configs/scene3_roi.yaml."
                    )
            else:
                observer.check_roi_points()

            initial_check_snapshot = None
            if self.cfg.get("success", {}).get("use_privileged_check", True):
                initial_check_snapshot = checker.capture_privileged_snapshot()

            record_enabled = bool(self.cfg.get("record", {}).get("record_rosbag", True))
            recorder = RosbagRecorder(record_enabled, bag_path, build_record_topics(self.cfg))
            if record_enabled:
                recorder.start()

            observer.publish_episode_info(self.make_metadata(seed, episode_id, bag_path, status="running"))
            observer.publish_stage("pregrasp")
            self.sleep(self.cfg.get("episode", {}).get("settle_before_record_sec", 0.6))

            observer.publish_stage("approach")
            expert.approach_tray_edge()
            self.log_target_gripper_distance(checker, "approach")

            observer.publish_stage("close_gripper")
            if not expert.close_gripper():
                raise RuntimeError("CLAW_FAILED: close command returned false")

            observer.publish_stage("lift")
            expert.lift_tray()

            observer.publish_stage("extract")
            expert.extract_tray()

            observer.publish_stage("stow")
            expert.move_to_waist_stow()
            self.sleep(self.cfg.get("episode", {}).get("settle_after_stow_sec", 1.0))
            hold_seconds = float(self.cfg.get("episode", {}).get("hold_seconds", 0.0))
            if hold_seconds > 0.0:
                self.sleep(hold_seconds)

            if recorder is not None:
                recorder.stop()

            ok, reason, details = checker.check(
                bag_path=bag_path,
                rosbag_enabled=record_enabled,
                initial_snapshot=initial_check_snapshot,
            )
            if not ok:
                raise RuntimeError(self.format_failure(reason, details))

            if recorder is not None:
                recorder.mark_keep()
            metadata = self.make_metadata(seed, episode_id, bag_path, status="success", success=True, details=details)
            dump_yaml(metadata_path, metadata)
            append_jsonl(
                self.success_manifest,
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "slot": self.cfg.get("scene", {}).get("target_slot", "upper"),
                    "bag_path": bag_path,
                    "metadata_path": metadata_path,
                },
            )
            expert.restore_arm_mode_if_needed()
            print(f"[INFO] scene3 episode saved: {episode_id}")
            return True
        except Exception as exc:
            reason = str(exc)
            details["traceback"] = traceback.format_exc()
            print(f"[ERROR] scene3 collection failed: {reason}", file=sys.stderr)
            if expert is not None:
                expert.safe_stop()
            if recorder is not None:
                if self.cfg.get("record", {}).get("delete_failed_bag", True):
                    recorder.discard()
                else:
                    recorder.stop()
            metadata = self.make_metadata(seed, episode_id, bag_path, status="failed", success=False, reason=reason, details=details)
            dump_yaml(metadata_path, metadata)
            self.write_failed(seed, episode_id, reason, {"metadata_path": metadata_path})
            return False
        finally:
            if expert is not None:
                expert.shutdown()
            if self.sim_proc is not None and not self.cfg.get("scene", {}).get("use_existing_sim", True):
                terminate_process_group(self.sim_proc)
                self.sim_proc = None

    def ensure_sim(self, seed):
        if self.cfg.get("scene", {}).get("use_existing_sim", True):
            return
        self.sim_proc, cmd = start_scene3_challenge_task(seed, headless=bool(self.args.headless))
        print("[INFO] launched scene3 sim: " + " ".join(cmd))
        wait_for_roscore(self.sim_proc, timeout=self.cfg.get("episode", {}).get("launch_timeout_sec", 120.0))

    def wait_ready(self):
        topic_wait = self.cfg.get("episode", {}).get("topic_wait_timeout_sec", 20.0)
        topics = self.cfg.get("topics", {})
        required = [
            topics.get("sensors", "/sensors_data_raw"),
            topics.get("rgb", "/cam_h/color/image_raw/compressed"),
            topics.get("depth", "/cam_h/depth/image_raw/compressedDepth"),
            topics.get("camera_info", "/cam_h/color/camera_info"),
            topics.get("claw_state", "/leju_claw_state"),
        ]
        missing = wait_for_topics(required, topic_wait)
        if missing:
            raise RuntimeError("CAMERA_MISSING: missing topics " + ",".join(missing))

    def navigate_to_shelf_if_requested(self):
        if not self.cfg.get("navigation", {}).get("enabled", False):
            return
        print("[INFO] navigating to Scene3 shelf-front work position...")
        navigator = Scene3ShelfNavigator(self.cfg)
        navigator.approach_shelf()
        print("[INFO] shelf approach complete; starting local tray skill.")

    def wait_for_shelf_front_ready(self, seed, episode_id):
        if not self.cfg.get("episode", {}).get("wait_for_user_ready", True):
            print("[INFO] --assume-at-shelf enabled; starting local tray skill immediately.")
            return
        prompt = (
            "\n[WAIT] Scene3 tray collector only records the local shelf-front skill.\n"
            f"       seed={seed}, episode={episode_id}\n"
            "       Please navigate or move the robot to the shelf-front work position first.\n"
            "       Requirements: base stopped, robot facing shelf, head can see upper tray,\n"
            "       arms have enough clearance, and the target is reachable.\n"
            "       Press Enter to start pregrasp/grasp/extract/stow, or Ctrl-C to abort: "
        )
        if not sys.stdin.isatty():
            raise RuntimeError(
                "ROBOT_NOT_CONFIRMED_AT_SHELF: interactive confirmation is required before grasping. "
                "Run from a terminal and press Enter after navigation, or pass --assume-at-shelf "
                "only when the robot is already at the shelf-front work position."
            )
        input(prompt)

    def make_metadata(self, seed, episode_id, bag_path, status, success=None, reason=None, details=None):
        pc_cfg = self.cfg.get("pointcloud", {})
        topics = self.cfg.get("topics", {})
        data = {
            "episode_id": episode_id,
            "seed": int(seed),
            "scene": "scene3",
            "slot": self.cfg.get("scene", {}).get("target_slot", "upper"),
            "target_tray_name": self.cfg.get("scene", {}).get("target_tray_name"),
            "stage": "extract_and_stow",
            "status": status,
            "bag_path": bag_path,
            "start_condition": {
                "robot_at_shelf_front": True,
                "navigation_included": bool(self.cfg.get("scene", {}).get("navigation_included", False)),
                "approach_shelf_distance_m": self.cfg.get("navigation", {}).get("approach_shelf_distance"),
            },
            "observation": {
                "rgb_topic": topics.get("rgb", "/cam_h/color/image_raw/compressed"),
                "depth_topic": topics.get("depth", "/cam_h/depth/image_raw/compressedDepth"),
                "camera_info_topic": topics.get("camera_info", "/cam_h/color/camera_info"),
                "joint_topic": topics.get("sensors", "/sensors_data_raw"),
                "claw_topic": topics.get("claw_state", "/leju_claw_state"),
                "pointcloud_format": "xyzrgb",
                "pointcloud_frame": pc_cfg.get("frame", "base_link"),
                "pointcloud_num_points": pc_cfg.get("num_points", 1024),
            },
            "action": {
                "action_type": self.cfg.get("robot", {}).get("action_type", "joint_delta"),
                "arms": "both" if self.cfg.get("robot", {}).get("use_both_arms", False) else "single",
                "active_arm": self.cfg.get("robot", {}).get("active_arm", "right"),
                "gripper": "rq2f85",
            },
        }
        if success is not None:
            data["success"] = {
                "ok": bool(success),
                "tray_extracted": bool(success),
                "tray_stowed_at_waist": bool(success),
                "dropped": False if success else None,
                "rosbag_valid": bool(success),
            }
        if reason:
            data["failure_reason"] = reason
        if details:
            data["details"] = details
        return data

    def write_failed(self, seed, episode_id, reason, extra=None):
        row = {"seed": int(seed), "episode_id": episode_id, "reason": reason}
        if extra:
            row.update(extra)
        append_jsonl(self.failed_manifest, row)

    def log_target_gripper_distance(self, checker, stage):
        if not self.cfg.get("success", {}).get("log_target_gripper_distance", True):
            return
        try:
            info = checker.measure_target_gripper_distance()
        except Exception as exc:
            print(f"[WARN] {stage} target/gripper distance unavailable: {exc}")
            return
        distance = info.get("target_gripper_distance_m")
        distance_text = "nan" if distance is None else f"{float(distance):.3f}m"
        print(
            "[INFO] {stage} target/gripper distance={distance} "
            "target_base={target} gripper_base={gripper} delta_target_minus_gripper={delta}".format(
                stage=stage,
                distance=distance_text,
                target=self._round_list(info.get("target_base_xyz")),
                gripper=self._round_list(info.get("active_gripper_base_xyz")),
                delta=self._round_list(info.get("target_minus_gripper_base_xyz")),
            )
        )

    @staticmethod
    def format_failure(reason, details):
        summary = compact_yaml(details)
        if not summary:
            return str(reason)
        return f"{reason}: {summary}"

    @staticmethod
    def _round_list(values, ndigits=3):
        if values is None:
            return None
        return [round(float(v), int(ndigits)) for v in values]

    @staticmethod
    def sleep(seconds):
        import rospy

        rospy.sleep(float(seconds))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect Scene3 upper-tray grasp-and-stow dataset")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="main YAML config path")
    parser.add_argument("--roi-config", default=DEFAULT_ROI_CONFIG, help="ROI YAML config path")
    parser.add_argument("--seed", type=int, default=0, help="Scene3 seed")
    parser.add_argument("--auto", action="store_true", help="collect seed range")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed for --auto")
    parser.add_argument("--seed-end", type=int, default=100, help="last seed for --auto")
    parser.add_argument("--repeat", type=int, default=None, help="valid episodes per seed")
    parser.add_argument("--max-attempts-per-seed", type=int, default=None, help="max attempts per seed")
    parser.add_argument("--slot", choices=["upper"], default="upper", help="target slot; this collector only supports the upper shelf tray")
    parser.add_argument("--output-dir", default=None, help="rosbag and metadata output directory")

    sim_group = parser.add_mutually_exclusive_group()
    sim_group.add_argument("--use-existing-sim", action="store_true", default=False, help="attach to a running Scene3 sim")
    sim_group.add_argument("--launch-sim", action="store_true", help="launch Scene3 sim from this script")
    parser.add_argument("--headless", action="store_true", help="set CHALLENGE_HEADLESS=1 when launching sim")
    parser.add_argument("--full-run", action="store_true", help="launch Scene3 unless --use-existing-sim is set, navigate to shelf, then collect the local tray skill")
    parser.add_argument("--navigate-to-shelf", action="store_true", help="drive forward to the shelf-front work pose before collecting")
    parser.add_argument("--approach-shelf-distance", type=float, default=None, help="forward distance in meters for --navigate-to-shelf")
    parser.add_argument("--nav-timeout", type=float, default=None, help="timeout in seconds for shelf approach")
    parser.add_argument("--nav-open-loop", action="store_true", help="force timed open-loop shelf approach instead of position feedback")
    parser.add_argument("--confirm-after-navigation", action="store_true", help="pause for Enter after automatic shelf navigation")

    rosbag_group = parser.add_mutually_exclusive_group()
    rosbag_group.add_argument("--record-rosbag", action="store_true", help="enable rosbag recording")
    rosbag_group.add_argument("--no-rosbag", action="store_true", help="run expert without recording rosbag")
    parser.add_argument("--record-derived-topics", action="store_true", help="record stage/action/xyzrgb derived topics")
    parser.add_argument("--min-camera-fps", type=float, default=None, help="minimum RGB camera FPS")
    parser.add_argument("--skip-pointcloud-precheck", action="store_true", help="do not fail before recording when RGB-D ROI has too few points")
    parser.add_argument("--min-roi-points", type=int, default=None, help="minimum pregrasp ROI point count")

    parser.add_argument("--print-plan", action="store_true", help="print named-pose expert plan and exit")
    parser.add_argument("--debug-run-once", action="store_true", help="single attempt, no retry loop")
    parser.add_argument("--hold-seconds", type=float, default=None, help="hold at final stow pose for observation")
    parser.add_argument("--preset-only", action="store_true", help="move to pregrasp only, then stop")
    parser.add_argument("--no-delete-failed-bag", action="store_true", help="keep failed bag files")
    parser.add_argument("--assume-at-shelf", action="store_true", help="skip the interactive shelf-front readiness confirmation")
    parser.add_argument("--no-truth-ik", action="store_true", help="disable truth IK; use named poses for all stages")
    parser.add_argument("--named-pose-mode", action="store_true", help="alias for --no-truth-ik")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.slot != "upper":
        raise SystemExit("only --slot upper is implemented in the first version")
    cfg = load_config(args)
    roi_cfg = load_yaml(args.roi_config)

    pose_config_path = resolve_pose_config(SCRIPT_DIR, cfg)
    if args.print_plan:
        expert = Scene3TrayGraspExpert(cfg, pose_config_path, observer=None)
        expert.print_plan()
        return 0

    collector = Scene3TrayDatasetCollector(cfg, roi_cfg, args)
    if args.auto:
        ok_all = True
        for seed in range(int(args.seed_start), int(args.seed_end) + 1):
            ok = collector.collect_seed(seed)
            ok_all = ok_all and ok
        return 0 if ok_all else 1
    return 0 if collector.collect_seed(args.seed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
