#!/usr/bin/env python3
"""ROS launch, topic waiting, rosbag, and manifest helpers for Scene3 collection."""

import datetime as _datetime
import json
import os
import signal
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
CHALLENGE_TASK_SCRIPT = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "collect_scene2_dataset",
    "challenge_task.py",
)


def now_tag():
    return _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_repo_relative(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def init_ros_node(node_name):
    import rospy

    if not rospy.core.is_initialized():
        rospy.init_node(node_name, anonymous=True)


def terminate_process_group(proc, sig=signal.SIGINT, timeout=10.0):
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
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            print(f"[WARN] process group {proc.pid} did not exit after SIGKILL", file=sys.stderr)
    except ProcessLookupError:
        return


def wait_for_roscore(proc=None, timeout=120.0):
    import xmlrpc.client

    master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"challenge_task exited early with code {proc.returncode}")
        try:
            master = xmlrpc.client.ServerProxy(master_uri)
            code, _message, _state = master.getSystemState("/scene3_collect_wait")
            if code == 1:
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"roscore did not become ready within {timeout:.1f}s")


def wait_for_topics(topics, timeout):
    import rospy

    required = list(dict.fromkeys([topic for topic in topics if topic]))
    start = time.time()
    while time.time() - start < float(timeout) and not rospy.is_shutdown():
        published = {name for name, _msg_type in rospy.get_published_topics()}
        missing = [topic for topic in required if topic not in published]
        if not missing:
            return []
        time.sleep(0.5)
    return missing


def wait_for_connection(pub, timeout):
    import rospy

    start = time.time()
    while pub.get_num_connections() == 0 and time.time() - start < float(timeout) and not rospy.is_shutdown():
        rospy.sleep(0.2)
    if pub.get_num_connections() == 0:
        raise RuntimeError(f"topic {pub.name} has no subscriber")


def start_scene3_challenge_task(seed, headless=False):
    cmd = [
        sys.executable,
        CHALLENGE_TASK_SCRIPT,
        "--scene",
        "scene3",
        "--seed",
        str(int(seed)),
        "--no-timer-gui",
    ]
    env = os.environ.copy()
    if headless:
        env["CHALLENGE_HEADLESS"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env,
    )
    return proc, cmd


def build_record_topics(cfg):
    topics_cfg = cfg.get("topics", {})
    topics = [
        topics_cfg.get("rgb", "/cam_h/color/image_raw/compressed"),
        topics_cfg.get("depth", "/cam_h/depth/image_raw/compressedDepth"),
        topics_cfg.get("camera_info", "/cam_h/color/camera_info"),
        topics_cfg.get("sensors", "/sensors_data_raw"),
        topics_cfg.get("claw_state", "/leju_claw_state"),
        topics_cfg.get("tf", "/tf"),
        topics_cfg.get("tf_static", "/tf_static"),
        topics_cfg.get("arm_command", "/kuavo_arm_traj"),
        topics_cfg.get("cmd_vel", "/cmd_vel"),
        topics_cfg.get("head_command", "/robot_head_motion_data"),
        topics_cfg.get("gripper_command_joint", "/gripper/command"),
        topics_cfg.get("gripper_command_leju", "/leju_claw_command"),
    ]
    if cfg.get("record", {}).get("record_derived_topics", True):
        topics.extend(
            [
                topics_cfg.get("stage", "/scene3_collect/stage"),
                topics_cfg.get("expert_action", "/scene3_collect/expert_action"),
                topics_cfg.get("head_xyzrgb_roi", "/scene3_collect/head_xyzrgb_roi"),
                topics_cfg.get("episode_info", "/scene3_collect/episode_info"),
            ]
        )
    return list(dict.fromkeys([topic for topic in topics if topic]))


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
        terminate_process_group(self.proc, signal.SIGINT, timeout=10.0)
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


def check_rosbag_topics_and_frequency(
    bag_path,
    required_topics,
    min_hz_by_topic=None,
    min_size_mb=1.0,
    min_duration_sec=3.0,
    max_duration_sec=25.0,
):
    if not bag_path:
        return False, "ROSBAG_DISABLED", {}
    if not os.path.exists(bag_path):
        return False, "ROSBAG_MISSING", {}
    size_mb = os.path.getsize(bag_path) / (1024.0 * 1024.0)
    if size_mb < float(min_size_mb):
        return False, f"ROSBAG_TOO_SMALL:{size_mb:.2f}MB", {"size_mb": size_mb}

    try:
        import rosbag
    except Exception as exc:
        return False, f"ROSBAG_IMPORT_FAILED:{exc}", {"size_mb": size_mb}

    min_hz_by_topic = min_hz_by_topic or {}
    try:
        with rosbag.Bag(bag_path, "r") as bag:
            topic_info = bag.get_type_and_topic_info()[1]
            missing = [topic for topic in required_topics if topic not in topic_info]
            if missing:
                return False, "ROSBAG_MISSING_TOPICS:" + ",".join(missing), {"size_mb": size_mb}

            stats = {
                topic: {"count": 0, "first": None, "last": None}
                for topic in set(required_topics) | set(min_hz_by_topic)
            }
            for topic, _msg, timestamp in bag.read_messages(topics=list(stats.keys())):
                stamp = timestamp.to_sec()
                item = stats[topic]
                item["count"] += 1
                if item["first"] is None:
                    item["first"] = stamp
                item["last"] = stamp
    except Exception as exc:
        return False, f"ROSBAG_READ_FAILED:{exc}", {"size_mb": size_mb}

    first_stamps = [item["first"] for item in stats.values() if item["first"] is not None]
    last_stamps = [item["last"] for item in stats.values() if item["last"] is not None]
    duration = (max(last_stamps) - min(first_stamps)) if first_stamps and last_stamps else 0.0
    if duration < float(min_duration_sec):
        return False, f"ROSBAG_TOO_SHORT:{duration:.2f}s", {"size_mb": size_mb, "duration_sec": duration}
    if duration > float(max_duration_sec):
        return False, f"ROSBAG_TOO_LONG:{duration:.2f}s", {"size_mb": size_mb, "duration_sec": duration}

    hz_summary = {}
    failed_hz = []
    for topic, min_hz in min_hz_by_topic.items():
        item = stats.get(topic, {})
        count = item.get("count", 0)
        first = item.get("first")
        last = item.get("last")
        if count < 2 or first is None or last is None or last <= first:
            failed_hz.append(f"{topic}=insufficient({count})")
            continue
        hz = float(count - 1) / float(last - first)
        hz_summary[topic] = hz
        if hz < float(min_hz):
            failed_hz.append(f"{topic}={hz:.2f}Hz")
    if failed_hz:
        return False, "ROSBAG_LOW_FPS:" + ",".join(failed_hz), {
            "size_mb": size_mb,
            "duration_sec": duration,
            "hz": hz_summary,
        }
    return True, "OK", {"size_mb": size_mb, "duration_sec": duration, "hz": hz_summary}


def append_jsonl(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    row = dict(data)
    row.setdefault("time", _datetime.datetime.now().isoformat(timespec="seconds"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

