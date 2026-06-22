#!/usr/bin/env python3
"""Observation and derived-topic publisher for Scene3 tray collection."""

import json
import threading
import time

from scene3_pointcloud_utils import (
    build_xyzrgb_sample,
    decode_compressed_depth,
    decode_compressed_rgb,
    matrix_from_transform_stamped,
    make_xyzrgb_pointcloud2,
    roi_for_stage,
)


STAGE_IDS = {
    "pregrasp": 0,
    "approach": 1,
    "close_gripper": 2,
    "extract": 3,
    "lift": 4,
    "stow": 5,
}


class Scene3ObservationRecorder:
    def __init__(self, cfg, roi_cfg):
        import rospy
        from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2
        from std_msgs.msg import Float64MultiArray, String
        from kuavo_msgs.msg import lejuClawState, sensorsData

        self.cfg = cfg
        self.roi_cfg = roi_cfg
        topics = cfg.get("topics", {})
        self.rgb_topic = topics.get("rgb", "/cam_h/color/image_raw/compressed")
        self.depth_topic = topics.get("depth", "/cam_h/depth/image_raw/compressedDepth")
        self.camera_info_topic = topics.get("camera_info", "/cam_h/color/camera_info")
        self.sensors_topic = topics.get("sensors", "/sensors_data_raw")
        self.claw_topic = topics.get("claw_state", "/leju_claw_state")
        self.stage_topic = topics.get("stage", "/scene3_collect/stage")
        self.expert_action_topic = topics.get("expert_action", "/scene3_collect/expert_action")
        self.pointcloud_topic = topics.get("head_xyzrgb_roi", "/scene3_collect/head_xyzrgb_roi")
        self.episode_info_topic = topics.get("episode_info", "/scene3_collect/episode_info")

        self._lock = threading.Lock()
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_camera_info = None
        self._latest_sensors = None
        self._latest_claw = None
        self._stage = "pregrasp"

        rospy.Subscriber(self.rgb_topic, CompressedImage, self._cb_rgb, queue_size=1)
        rospy.Subscriber(self.depth_topic, CompressedImage, self._cb_depth, queue_size=1)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._cb_camera_info, queue_size=1)
        rospy.Subscriber(self.sensors_topic, sensorsData, self._cb_sensors, queue_size=1)
        rospy.Subscriber(self.claw_topic, lejuClawState, self._cb_claw, queue_size=1)

        self.stage_pub = rospy.Publisher(self.stage_topic, String, queue_size=20, latch=True)
        self.expert_action_pub = rospy.Publisher(self.expert_action_topic, Float64MultiArray, queue_size=100)
        self.episode_info_pub = rospy.Publisher(self.episode_info_topic, String, queue_size=10, latch=True)
        self.pointcloud_pub = rospy.Publisher(self.pointcloud_topic, PointCloud2, queue_size=5)

        self.tf_buffer = None
        self.tf_listener = None
        target_frame = cfg.get("pointcloud", {}).get("frame", "base_link")
        camera_frame = cfg.get("pointcloud", {}).get("camera_frame", "cam_h_color_optical_frame")
        if target_frame and target_frame != camera_frame:
            try:
                import tf2_ros

                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            except Exception as exc:
                rospy.logwarn("Scene3ObservationRecorder: TF listener unavailable: %s", exc)

    def _cb_rgb(self, msg):
        with self._lock:
            self._latest_rgb = msg

    def _cb_depth(self, msg):
        with self._lock:
            self._latest_depth = msg

    def _cb_camera_info(self, msg):
        with self._lock:
            self._latest_camera_info = msg

    def _cb_sensors(self, msg):
        with self._lock:
            self._latest_sensors = msg

    def _cb_claw(self, msg):
        with self._lock:
            self._latest_claw = msg

    def latest_snapshot(self):
        with self._lock:
            return {
                "rgb": self._latest_rgb,
                "depth": self._latest_depth,
                "camera_info": self._latest_camera_info,
                "sensors": self._latest_sensors,
                "claw": self._latest_claw,
                "stage": self._stage,
            }

    def wait_for_observation(self, timeout=10.0):
        import rospy

        deadline = time.time() + float(timeout)
        rate = rospy.Rate(20)
        while time.time() < deadline and not rospy.is_shutdown():
            snap = self.latest_snapshot()
            if (
                snap["rgb"] is not None
                and snap["depth"] is not None
                and snap["camera_info"] is not None
                and snap["sensors"] is not None
            ):
                return True
            rate.sleep()
        return False

    def publish_stage(self, stage):
        from std_msgs.msg import String

        stage = str(stage)
        with self._lock:
            self._stage = stage
        msg = String()
        msg.data = stage
        self.stage_pub.publish(msg)

    def publish_episode_info(self, data):
        from std_msgs.msg import String

        msg = String()
        msg.data = json.dumps(dict(data), ensure_ascii=False, sort_keys=True)
        self.episode_info_pub.publish(msg)

    def publish_expert_action(self, stage, q_target_deg, gripper_cmd):
        from std_msgs.msg import Float64MultiArray

        msg = Float64MultiArray()
        msg.data = [float(STAGE_IDS.get(stage, -1)), float(gripper_cmd)] + [float(v) for v in q_target_deg]
        self.expert_action_pub.publish(msg)

    def publish_head_xyzrgb_roi(self):
        import rospy

        snap = self.latest_snapshot()
        if snap["rgb"] is None or snap["depth"] is None or snap["camera_info"] is None:
            return 0
        pc_cfg = self.cfg.get("pointcloud", {})
        rgb = decode_compressed_rgb(snap["rgb"])
        depth = decode_compressed_depth(snap["depth"])
        roi = roi_for_stage(self.roi_cfg, snap["stage"])
        transform = self._lookup_camera_to_target_transform(snap)
        points, valid_count = build_xyzrgb_sample(
            rgb,
            depth,
            snap["camera_info"],
            roi,
            num_points=pc_cfg.get("num_points", 1024),
            normalize=pc_cfg.get("normalize_xyz", True),
            max_depth_m=pc_cfg.get("max_depth_m", 4.0),
            stride=pc_cfg.get("stride", 2),
            transform_4x4=transform,
        )
        frame_id = pc_cfg.get("frame", "base_link")
        msg = make_xyzrgb_pointcloud2(points, frame_id=frame_id, stamp=rospy.Time.now())
        self.pointcloud_pub.publish(msg)
        return valid_count

    def _lookup_camera_to_target_transform(self, snap):
        import rospy

        pc_cfg = self.cfg.get("pointcloud", {})
        target_frame = pc_cfg.get("frame", "base_link")
        camera_frame = (
            getattr(getattr(snap["depth"], "header", None), "frame_id", None)
            or getattr(getattr(snap["rgb"], "header", None), "frame_id", None)
            or getattr(getattr(snap["camera_info"], "header", None), "frame_id", None)
            or pc_cfg.get("camera_frame", "cam_h_color_optical_frame")
        )
        if not target_frame or not camera_frame or target_frame == camera_frame or self.tf_buffer is None:
            return None
        try:
            stamp = getattr(getattr(snap["depth"], "header", None), "stamp", None) or rospy.Time(0)
            transform = self.tf_buffer.lookup_transform(target_frame, camera_frame, stamp, rospy.Duration(0.05))
            return matrix_from_transform_stamped(transform)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Scene3ObservationRecorder: TF %s <- %s unavailable: %s", target_frame, camera_frame, exc)
            return None

    def check_roi_points(self, min_points=None):
        pc_cfg = self.cfg.get("pointcloud", {})
        min_points = int(min_points if min_points is not None else pc_cfg.get("min_roi_points", 300))
        try:
            return self.publish_head_xyzrgb_roi() >= min_points
        except Exception as exc:
            import rospy

            rospy.logwarn("Scene3ObservationRecorder: pointcloud check failed: %s", exc)
            return False

    def get_arm_joint_radians(self):
        snap = self.latest_snapshot()
        sensors = snap["sensors"]
        if sensors is None:
            return []
        joint_q = list(sensors.joint_data.joint_q)
        if len(joint_q) >= 27:
            return [float(v) for v in joint_q[13:27]]
        if len(joint_q) >= 26:
            return [float(v) for v in joint_q[12:26]]
        return []

    def is_claw_closed(self, arm="right"):
        snap = self.latest_snapshot()
        claw = snap["claw"]
        if claw is None:
            return False
        try:
            idx = list(claw.data.name).index(f"{arm}_claw")
            return float(claw.data.position[idx]) > 50.0
        except Exception:
            pass
        try:
            idx = 1 if arm == "right" else 0
            return int(claw.state[idx]) >= int(claw.kReached)
        except Exception:
            return False
