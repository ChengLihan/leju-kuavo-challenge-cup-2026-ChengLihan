#!/usr/bin/env python3
"""Convert Scene3 tray-stow rosbags into compact IL episode files."""

import argparse
import bisect
import json
import os
import sys

import numpy as np
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECT_DIR = os.path.dirname(SCRIPT_DIR)
if COLLECT_DIR not in sys.path:
    sys.path.insert(0, COLLECT_DIR)

from scene3_observation_recorder import STAGE_IDS
from scene3_pointcloud_utils import (
    build_xyzrgb_sample,
    decode_compressed_depth,
    decode_compressed_rgb,
    matrix_from_transform_stamped,
    roi_for_stage,
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def nearest(items, stamp):
    if not items:
        return None
    times = [item[0] for item in items]
    idx = bisect.bisect_left(times, stamp)
    if idx <= 0:
        return items[0][1]
    if idx >= len(items):
        return items[-1][1]
    before = items[idx - 1]
    after = items[idx]
    return before[1] if abs(before[0] - stamp) <= abs(after[0] - stamp) else after[1]


def read_manifest_bags(manifest_path):
    bags = []
    if not manifest_path or not os.path.exists(manifest_path):
        return bags
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("bag_path"):
                bags.append(data["bag_path"])
    return bags


def discover_bags(rosbag_dir, manifest_path=None):
    bags = read_manifest_bags(manifest_path)
    if bags:
        return bags
    if not os.path.isdir(rosbag_dir):
        return []
    out = []
    for name in sorted(os.listdir(rosbag_dir)):
        if name.endswith(".bag"):
            out.append(os.path.join(rosbag_dir, name))
    return out


def arm_q_from_sensors(msg):
    joint_q = list(msg.joint_data.joint_q)
    if len(joint_q) >= 27:
        return np.asarray(joint_q[13:27], dtype=np.float32)
    if len(joint_q) >= 26:
        return np.asarray(joint_q[12:26], dtype=np.float32)
    return np.zeros(14, dtype=np.float32)


def gripper_from_claw(msg, arm="right"):
    try:
        idx = list(msg.data.name).index(f"{arm}_claw")
        return np.asarray([float(msg.data.position[idx])], dtype=np.float32)
    except Exception:
        return np.zeros(1, dtype=np.float32)


def stage_from_msg(msg):
    return str(getattr(msg, "data", "") or "pregrasp")


def action_from_expert_msg(msg):
    values = np.asarray(list(msg.data), dtype=np.float32)
    if values.size >= 16:
        return values[2:16], values[1:2]
    return None, None


def read_bag_messages(bag_path, cfg):
    import rosbag

    topics = cfg.get("topics", {})
    wanted = {
        topics.get("rgb", "/cam_h/color/image_raw/compressed"): "rgb",
        topics.get("depth", "/cam_h/depth/image_raw/compressedDepth"): "depth",
        topics.get("camera_info", "/cam_h/color/camera_info"): "camera_info",
        topics.get("sensors", "/sensors_data_raw"): "sensors",
        topics.get("claw_state", "/leju_claw_state"): "claw",
        topics.get("stage", "/scene3_collect/stage"): "stage",
        topics.get("expert_action", "/scene3_collect/expert_action"): "expert_action",
        topics.get("tf", "/tf"): "tf",
        topics.get("tf_static", "/tf_static"): "tf_static",
    }
    streams = {key: [] for key in wanted.values()}
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, stamp in bag.read_messages(topics=list(wanted.keys())):
            streams[wanted[topic]].append((stamp.to_sec(), msg))
    for key in streams:
        streams[key].sort(key=lambda item: item[0])
    return streams


def build_tf_buffer(streams):
    try:
        import tf2_ros
    except Exception:
        return None
    buffer = tf2_ros.Buffer()
    for _stamp, msg in streams.get("tf_static", []):
        for transform in msg.transforms:
            try:
                buffer.set_transform_static(transform, "scene3_bag_converter")
            except Exception:
                pass
    for _stamp, msg in streams.get("tf", []):
        for transform in msg.transforms:
            try:
                buffer.set_transform(transform, "scene3_bag_converter")
            except Exception:
                pass
    return buffer


def lookup_transform_matrix(tf_buffer, target_frame, source_frame, stamp):
    if tf_buffer is None or not target_frame or not source_frame or target_frame == source_frame:
        return None
    try:
        import rospy

        transform = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time.from_sec(float(stamp)),
            rospy.Duration(0.05),
        )
        return matrix_from_transform_stamped(transform)
    except Exception:
        return None


def sample_times(rgb_stream, train_hz):
    if not rgb_stream:
        return []
    start = rgb_stream[0][0]
    end = rgb_stream[-1][0]
    step = 1.0 / float(train_hz)
    times = []
    t = start
    while t <= end:
        times.append(t)
        t += step
    return times


def convert_one_bag(bag_path, episode_index, cfg, roi_cfg, args):
    streams = read_bag_messages(bag_path, cfg)
    tf_buffer = build_tf_buffer(streams)
    times = sample_times(streams["rgb"], args.train_hz)
    pc_cfg = cfg.get("pointcloud", {})
    active_arm = cfg.get("robot", {}).get("active_arm", "right")

    images = []
    points = []
    q_arm = []
    q_gripper = []
    stage_ids = []
    timestamps = []
    expert_q_targets = []
    expert_gripper_cmds = []

    camera_info = streams["camera_info"][0][1] if streams["camera_info"] else None
    if camera_info is None:
        raise RuntimeError(f"{bag_path}: missing camera_info")

    for stamp in times:
        rgb_msg = nearest(streams["rgb"], stamp)
        depth_msg = nearest(streams["depth"], stamp)
        sensors_msg = nearest(streams["sensors"], stamp)
        claw_msg = nearest(streams["claw"], stamp)
        stage_msg = nearest(streams["stage"], stamp)
        expert_msg = nearest(streams["expert_action"], stamp)
        if rgb_msg is None or depth_msg is None or sensors_msg is None:
            continue

        stage = stage_from_msg(stage_msg) if stage_msg is not None else "pregrasp"
        rgb = decode_compressed_rgb(rgb_msg)
        depth = decode_compressed_depth(depth_msg)
        roi = roi_for_stage(roi_cfg, stage)
        source_frame = (
            getattr(getattr(depth_msg, "header", None), "frame_id", None)
            or getattr(getattr(rgb_msg, "header", None), "frame_id", None)
            or getattr(getattr(camera_info, "header", None), "frame_id", None)
            or pc_cfg.get("camera_frame", "cam_h_color_optical_frame")
        )
        transform = lookup_transform_matrix(tf_buffer, pc_cfg.get("frame", "base_link"), source_frame, stamp)
        xyzrgb, _valid_count = build_xyzrgb_sample(
            rgb,
            depth,
            camera_info,
            roi,
            num_points=pc_cfg.get("num_points", args.point_num),
            normalize=pc_cfg.get("normalize_xyz", True),
            max_depth_m=pc_cfg.get("max_depth_m", 4.0),
            stride=pc_cfg.get("stride", 2),
            transform_4x4=transform,
        )
        q = arm_q_from_sensors(sensors_msg)
        g = gripper_from_claw(claw_msg, arm=active_arm) if claw_msg is not None else np.zeros(1, dtype=np.float32)
        q_target, g_cmd = action_from_expert_msg(expert_msg) if expert_msg is not None else (None, None)

        images.append(rgb)
        points.append(xyzrgb)
        q_arm.append(q)
        q_gripper.append(g)
        stage_ids.append(STAGE_IDS.get(stage, 0))
        timestamps.append(stamp)
        expert_q_targets.append(q_target if q_target is not None else np.full(14, np.nan, dtype=np.float32))
        expert_gripper_cmds.append(g_cmd if g_cmd is not None else np.full(1, np.nan, dtype=np.float32))

    if len(q_arm) < 2:
        raise RuntimeError(f"{bag_path}: not enough synchronized frames")

    q_arm = np.asarray(q_arm, dtype=np.float32)
    observed_dq = np.zeros_like(q_arm)
    observed_dq[:-1] = q_arm[1:] - q_arm[:-1]
    observed_dq[-1] = observed_dq[-2]

    return {
        "images_head_rgb": np.asarray(images, dtype=np.uint8),
        "points_head_xyzrgb": np.asarray(points, dtype=np.float32),
        "states_q_arm": q_arm,
        "states_q_gripper": np.asarray(q_gripper, dtype=np.float32),
        "states_stage_id": np.asarray(stage_ids, dtype=np.int64),
        "actions_observed_dq_arm": observed_dq,
        "actions_expert_q_target": np.asarray(expert_q_targets, dtype=np.float32),
        "actions_expert_gripper_cmd": np.asarray(expert_gripper_cmds, dtype=np.float32),
        "timestamp": np.asarray(timestamps, dtype=np.float64),
        "episode_index": np.asarray([episode_index], dtype=np.int64),
        "source_bag": np.asarray([bag_path]),
    }


def write_episode_npz(path, episode):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **episode)


def write_episode_hdf5(path, episode):
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(f"h5py is required for --format hdf5: {exc}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in episode.items():
            if value.dtype.kind in ("U", "O"):
                f.create_dataset(key, data=value.astype("S"))
            else:
                f.create_dataset(key, data=value, compression="gzip")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert Scene3 tray rosbags to IL dataset episodes")
    parser.add_argument("--rosbag-dir", default="bags/scene3_tray_stow")
    parser.add_argument("--output-dataset-dir", default="dataset/scene3_tray_stow")
    parser.add_argument("--config", default=os.path.join(COLLECT_DIR, "configs", "scene3_collect.yaml"))
    parser.add_argument("--roi-config", default=os.path.join(COLLECT_DIR, "configs", "scene3_roi.yaml"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--format", choices=["npz", "hdf5"], default="npz")
    parser.add_argument("--train-hz", type=float, default=10.0)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--point-num", type=int, default=1024)
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    roi_cfg = load_yaml(args.roi_config)
    manifest = args.manifest or os.path.join(args.rosbag_dir, "success_manifest.txt")
    bags = discover_bags(args.rosbag_dir, manifest)
    episode_dir = os.path.join(args.output_dataset_dir, "episodes")
    meta_dir = os.path.join(args.output_dataset_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    stats = {"episodes": 0, "frames": 0, "bags": []}
    for idx, bag_path in enumerate(bags):
        episode = convert_one_bag(bag_path, idx, cfg, roi_cfg, args)
        suffix = "npz" if args.format == "npz" else "hdf5"
        out_path = os.path.join(episode_dir, f"episode_{idx:06d}.{suffix}")
        if args.format == "npz":
            write_episode_npz(out_path, episode)
        else:
            write_episode_hdf5(out_path, episode)
        stats["episodes"] += 1
        stats["frames"] += int(episode["states_q_arm"].shape[0])
        stats["bags"].append({"bag_path": bag_path, "episode_path": out_path})
        print(f"[INFO] wrote {out_path}")

    with open(os.path.join(meta_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "fps": args.train_hz,
                "chunk_size": args.chunk_size,
                "point_num": args.point_num,
                "action_type": cfg.get("robot", {}).get("action_type", "joint_delta"),
                "stage_ids": STAGE_IDS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(meta_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[INFO] conversion complete: {stats['episodes']} episodes, {stats['frames']} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
