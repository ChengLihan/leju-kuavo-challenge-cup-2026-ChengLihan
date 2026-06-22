#!/usr/bin/env python3
"""Success and quality checks for Scene3 upper-tray data collection."""

import math
import os
import xml.etree.ElementTree as ET

from scene3_rosbag_utils import check_rosbag_topics_and_frequency, ensure_repo_relative


FAIL_REASONS = {
    "IK_FAILED",
    "ARM_TIMEOUT",
    "CLAW_FAILED",
    "CAMERA_MISSING",
    "DEPTH_INVALID",
    "POINTCLOUD_EMPTY",
    "TRAY_NOT_EXTRACTED",
    "TRAY_DROPPED",
    "TRAY_NOT_STOWED",
    "SCENE_DISTURBED",
    "ROSBAG_INVALID",
    "UNKNOWN_EXCEPTION",
}


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


def _walk_body_for_qpos(body, qpos_addr, qpos_map, body_pos_map, body_freejoint_map):
    body_name = body.get("name")
    if body_name and body.get("pos"):
        body_pos_map[body_name] = [float(v) for v in body.get("pos").split()[:3]]
    for child in list(body):
        size = _joint_qpos_size(child)
        if size:
            name = child.get("name")
            if name:
                qpos_map[name] = qpos_addr
            if child.tag == "freejoint" and body_name:
                body_freejoint_map[body_name] = qpos_addr
            qpos_addr += size
    for child in list(body):
        if child.tag == "body":
            qpos_addr = _walk_body_for_qpos(child, qpos_addr, qpos_map, body_pos_map, body_freejoint_map)
    return qpos_addr


def build_qpos_and_body_maps(xml_path):
    qpos_addr = 0
    qpos_map = {}
    body_pos_map = {}
    body_freejoint_map = {}
    for child, _base_dir in _iter_xml_children_with_includes(xml_path):
        if child.tag != "worldbody":
            continue
        for body in list(child):
            if body.tag == "body":
                qpos_addr = _walk_body_for_qpos(body, qpos_addr, qpos_map, body_pos_map, body_freejoint_map)
    return qpos_map, body_pos_map, body_freejoint_map


def pose_from_qpos(qpos, qpos_map, name):
    if name not in qpos_map:
        raise KeyError(f"freejoint '{name}' not found in XML qpos map")
    addr = qpos_map[name]
    if addr + 2 >= len(qpos):
        raise IndexError(f"qpos for '{name}' address {addr} outside qpos length {len(qpos)}")
    return [float(qpos[addr]), float(qpos[addr + 1]), float(qpos[addr + 2])]


def pose_from_body_freejoint(qpos, body_freejoint_map, name):
    if name not in body_freejoint_map:
        raise KeyError(f"freejoint body '{name}' not found in XML qpos map")
    addr = body_freejoint_map[name]
    if addr + 6 >= len(qpos):
        raise IndexError(f"qpos for body '{name}' address {addr} outside qpos length {len(qpos)}")
    xyz = [float(qpos[addr]), float(qpos[addr + 1]), float(qpos[addr + 2])]
    quat_wxyz = [float(qpos[addr + 3]), float(qpos[addr + 4]), float(qpos[addr + 5]), float(qpos[addr + 6])]
    return xyz, quat_wxyz


def in_bbox(xyz, bbox):
    return (
        float(bbox["x_min"]) <= xyz[0] <= float(bbox["x_max"])
        and float(bbox["y_min"]) <= xyz[1] <= float(bbox["y_max"])
        and float(bbox["z_min"]) <= xyz[2] <= float(bbox["z_max"])
    )


def distance_xy(a, b):
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


def distance_xyz(a, b):
    return (
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    ) ** 0.5


def yaw_from_quat_wxyz(quat):
    w, x, y, z = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def world_to_base_xyz(point_world, base_world, base_yaw):
    dx = float(point_world[0]) - float(base_world[0])
    dy = float(point_world[1]) - float(base_world[1])
    c = math.cos(-float(base_yaw))
    s = math.sin(-float(base_yaw))
    return [
        c * dx - s * dy,
        s * dx + c * dy,
        float(point_world[2]) - float(base_world[2]),
    ]


class Scene3SuccessChecker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.topics = cfg.get("topics", {})
        self.record_cfg = cfg.get("record", {})
        self.success_cfg = cfg.get("success", {})

    def check(self, bag_path=None, rosbag_enabled=True, initial_snapshot=None):
        details = {}
        if rosbag_enabled:
            ok, reason, bag_details = self._check_rosbag(bag_path)
            details["rosbag"] = bag_details
            if not ok:
                return False, reason, details

        if self.success_cfg.get("use_privileged_check", True):
            ok, reason, privileged_details = self._check_privileged_tray_pose(initial_snapshot=initial_snapshot)
            details["privileged"] = privileged_details
            if not ok:
                return False, reason, details
        return True, "OK", details

    def capture_privileged_snapshot(self, timeout=5.0):
        import rospy
        from std_msgs.msg import Float64MultiArray

        xml_path = ensure_repo_relative(self.success_cfg.get("active_scene_xml"))
        if not xml_path or not os.path.exists(xml_path):
            raise RuntimeError(f"TRAY_NOT_EXTRACTED:ACTIVE_XML_MISSING: {xml_path}")
        qpos_map, _body_pos_map, body_freejoint_map = build_qpos_and_body_maps(xml_path)
        msg = rospy.wait_for_message("/mujoco/qpos", Float64MultiArray, timeout=float(timeout))
        qpos = list(msg.data)
        base_xyz, base_quat = pose_from_body_freejoint(qpos, body_freejoint_map, "base_link")
        base_yaw = yaw_from_quat_wxyz(base_quat)
        tray_xyz = self._tray_positions(qpos, qpos_map)
        return {
            "source": "live_qpos_at_record_start",
            "base_link_xyz": base_xyz,
            "base_yaw_rad": base_yaw,
            "tray_xyz": tray_xyz,
        }

    def measure_target_gripper_distance(self, timeout=5.0):
        import rospy
        from std_msgs.msg import Float64MultiArray

        xml_path = ensure_repo_relative(self.success_cfg.get("active_scene_xml"))
        target = self.success_cfg.get("target_tray_name") or self.cfg.get("scene", {}).get("target_tray_name")
        if not xml_path or not os.path.exists(xml_path):
            raise RuntimeError(f"TRAY_NOT_EXTRACTED:ACTIVE_XML_MISSING: {xml_path}")
        qpos_map, _body_pos_map, body_freejoint_map = build_qpos_and_body_maps(xml_path)
        msg = rospy.wait_for_message("/mujoco/qpos", Float64MultiArray, timeout=float(timeout))
        qpos = list(msg.data)
        target_world_xyz = pose_from_qpos(qpos, qpos_map, target)
        base_xyz, base_quat = pose_from_body_freejoint(qpos, body_freejoint_map, "base_link")
        base_yaw = yaw_from_quat_wxyz(base_quat)
        target_base_xyz = world_to_base_xyz(target_world_xyz, base_xyz, base_yaw)
        gripper_frame, gripper_base_xyz = self._lookup_active_gripper_base_xyz()
        distance = distance_xyz(target_base_xyz, gripper_base_xyz) if gripper_base_xyz is not None else None
        delta = None
        if gripper_base_xyz is not None:
            delta = [target_base_xyz[i] - gripper_base_xyz[i] for i in range(3)]
        return {
            "target": target,
            "target_world_xyz": target_world_xyz,
            "base_link_xyz": base_xyz,
            "target_base_xyz": target_base_xyz,
            "active_gripper_frame": gripper_frame,
            "active_gripper_base_xyz": gripper_base_xyz,
            "target_minus_gripper_base_xyz": delta,
            "target_gripper_distance_m": distance,
        }

    def _check_rosbag(self, bag_path):
        required = [
            self.topics.get("rgb", "/cam_h/color/image_raw/compressed"),
            self.topics.get("depth", "/cam_h/depth/image_raw/compressedDepth"),
            self.topics.get("camera_info", "/cam_h/color/camera_info"),
            self.topics.get("sensors", "/sensors_data_raw"),
            self.topics.get("claw_state", "/leju_claw_state"),
            self.topics.get("tf", "/tf"),
            self.topics.get("tf_static", "/tf_static"),
            self.topics.get("arm_command", "/kuavo_arm_traj"),
            self.topics.get("stage", "/scene3_collect/stage"),
            self.topics.get("expert_action", "/scene3_collect/expert_action"),
        ]
        min_hz_by_topic = {
            self.topics.get("rgb", "/cam_h/color/image_raw/compressed"): self.record_cfg.get("min_rgb_fps", 25.0),
            self.topics.get("depth", "/cam_h/depth/image_raw/compressedDepth"): self.record_cfg.get("min_depth_fps", 20.0),
        }
        ok, reason, details = check_rosbag_topics_and_frequency(
            bag_path,
            required,
            min_hz_by_topic=min_hz_by_topic,
            min_size_mb=self.record_cfg.get("min_bag_size_mb", 1.0),
            min_duration_sec=self.record_cfg.get("min_duration_sec", 3.0),
            max_duration_sec=self.record_cfg.get("max_duration_sec", 25.0),
        )
        return ok, "ROSBAG_INVALID:" + reason if not ok else "OK", details

    def _check_privileged_tray_pose(self, initial_snapshot=None):
        import rospy
        from std_msgs.msg import Float64MultiArray

        xml_path = ensure_repo_relative(self.success_cfg.get("active_scene_xml"))
        target = self.success_cfg.get("target_tray_name") or self.cfg.get("scene", {}).get("target_tray_name")
        if not xml_path or not os.path.exists(xml_path):
            return False, "TRAY_NOT_EXTRACTED:ACTIVE_XML_MISSING", {"xml_path": xml_path}
        try:
            qpos_map, body_pos_map, body_freejoint_map = build_qpos_and_body_maps(xml_path)
            initial_positions = self._initial_tray_positions(initial_snapshot, body_pos_map)
            initial = initial_positions.get(target)
            msg = rospy.wait_for_message("/mujoco/qpos", Float64MultiArray, timeout=5.0)
            qpos = list(msg.data)
            final = pose_from_qpos(qpos, qpos_map, target)
            base_xyz, base_quat = pose_from_body_freejoint(qpos, body_freejoint_map, "base_link")
            base_yaw = yaw_from_quat_wxyz(base_quat)
            target_base_xyz = world_to_base_xyz(final, base_xyz, base_yaw)
        except Exception as exc:
            return False, f"TRAY_NOT_EXTRACTED:QPOS_FAILED:{exc}", {"target": target}

        tray_motion = self._tray_motion_summary(qpos, qpos_map, initial_positions)
        min_extract = float(self.success_cfg.get("min_extract_distance_m", 0.10))
        extract_tolerance = float(self.success_cfg.get("extract_distance_tolerance_m", 0.005))
        max_non_target_motion = float(self.success_cfg.get("max_non_target_tray_motion_m", 0.03))
        disturbed_trays = {
            name: info
            for name, info in tray_motion.items()
            if name != target and float(info.get("dxy", 0.0)) > max_non_target_motion
        }
        gripper_frame, gripper_base_xyz = self._lookup_active_gripper_base_xyz()
        gripper_distance = distance_xyz(target_base_xyz, gripper_base_xyz) if gripper_base_xyz is not None else None
        max_gripper_distance = float(self.success_cfg.get("max_gripper_tray_distance_m", 0.30))
        details = {
            "target": target,
            "initial_source": (initial_snapshot or {}).get("source", "xml_body_pos"),
            "initial_xyz": initial,
            "final_xyz": final,
            "record_start_snapshot": self._public_snapshot(initial_snapshot),
            "base_link_xyz": base_xyz,
            "target_base_xyz": target_base_xyz,
            "min_extract_distance_m": min_extract,
            "active_gripper_frame": gripper_frame,
            "active_gripper_base_xyz": gripper_base_xyz,
            "target_gripper_distance_m": gripper_distance,
            "max_gripper_tray_distance_m": max_gripper_distance,
            "extract_distance_tolerance_m": extract_tolerance,
            "max_non_target_tray_motion_m": max_non_target_motion,
            "disturbed_non_target_trays": disturbed_trays,
            "tray_motion_xy_m": tray_motion,
        }
        if self.success_cfg.get("fail_on_non_target_tray_motion", True) and disturbed_trays:
            return False, "SCENE_DISTURBED:NON_TARGET_TRAY_MOVED", details

        if initial is not None:
            dxy = distance_xy(final, initial)
            details["extract_distance_xy_m"] = dxy
            if dxy + extract_tolerance < min_extract:
                return False, "TRAY_NOT_EXTRACTED", details
        if final[2] < float(self.success_cfg.get("floor_z_threshold_m", 0.20)):
            return False, "TRAY_DROPPED", details

        if self.success_cfg.get("require_gripper_near_tray", True):
            if gripper_distance is None or not math.isfinite(gripper_distance):
                return False, "TRAY_NOT_STOWED:GRIPPER_POSE_MISSING", details
            if gripper_distance > max_gripper_distance:
                return False, "TRAY_NOT_STOWED:NOT_HELD_BY_GRIPPER", details

        bbox = self.success_cfg.get("waist_stow_bbox")
        if self.success_cfg.get("require_waist_bbox", False) and bbox and not in_bbox(final, bbox):
            details["waist_stow_bbox"] = bbox
            return False, "TRAY_NOT_STOWED", details
        return True, "OK", details

    def _lookup_active_gripper_base_xyz(self):
        import rospy
        import tf2_ros

        active_arm = self.cfg.get("robot", {}).get("active_arm", "right")
        configured = self.success_cfg.get("active_gripper_frames")
        if configured:
            frames = [str(name) for name in configured]
        else:
            frames = [
                str(self.success_cfg.get("active_gripper_frame", f"{active_arm}_gripper_base")),
                f"{active_arm}_base",
                f"{active_arm}_pinch",
            ]

        parent = str(self.success_cfg.get("gripper_reference_frame", "base_link"))
        timeout = float(self.success_cfg.get("gripper_tf_timeout_sec", 1.0))
        buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(buffer)
        rospy.sleep(0.2)

        last_error = None
        for frame in frames:
            try:
                transform = buffer.lookup_transform(parent, frame, rospy.Time(0), rospy.Duration(timeout))
                translation = transform.transform.translation
                return frame, [float(translation.x), float(translation.y), float(translation.z)]
            except Exception as exc:
                last_error = exc
        return f"TF_LOOKUP_FAILED:{last_error}", None

    @staticmethod
    def _tray_positions(qpos, qpos_map):
        positions = {}
        for name in sorted(qpos_map):
            if not name.startswith("smt_tray_"):
                continue
            try:
                positions[name] = pose_from_qpos(qpos, qpos_map, name)
            except Exception:
                continue
        return positions

    @staticmethod
    def _initial_tray_positions(initial_snapshot, body_pos_map):
        if initial_snapshot and initial_snapshot.get("tray_xyz"):
            return dict(initial_snapshot.get("tray_xyz"))
        return {
            name: xyz
            for name, xyz in body_pos_map.items()
            if str(name).startswith("smt_tray_")
        }

    @staticmethod
    def _public_snapshot(initial_snapshot):
        if not initial_snapshot:
            return None
        return {
            "source": initial_snapshot.get("source"),
            "base_link_xyz": initial_snapshot.get("base_link_xyz"),
            "tray_xyz": initial_snapshot.get("tray_xyz"),
        }

    @staticmethod
    def _tray_motion_summary(qpos, qpos_map, initial_positions):
        summary = {}
        for name in sorted(qpos_map):
            if not name.startswith("smt_tray_"):
                continue
            initial = initial_positions.get(name)
            if initial is None:
                continue
            try:
                final = pose_from_qpos(qpos, qpos_map, name)
            except Exception:
                continue
            summary[name] = {
                "initial_xyz": initial,
                "final_xyz": final,
                "dxy": distance_xy(final, initial),
            }
        return summary
