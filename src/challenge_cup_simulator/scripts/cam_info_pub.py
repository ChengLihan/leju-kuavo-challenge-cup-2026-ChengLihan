#!/usr/bin/env python3
"""Publish CameraInfo for all cameras (head, left wrist, right wrist) in simulation."""
import math
import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo

WIDTH = 1280
HEIGHT = 720

CAMERAS = [
    # (image_topic, info_topic, fovy_deg)
    # 仅发布 color/camera_info：MuJoCo 中 color 与 depth 来自同一虚拟相机，内参完全一致；
    # 使用深度时下游请显式 remap，例如：camera_info:=/cam_h/color/camera_info
    ("/cam_h/color/image_raw/compressed", "/cam_h/color/camera_info", 85.0),
    ("/cam_l/color/image_raw/compressed", "/cam_l/color/camera_info", 60.0),
    ("/cam_r/color/image_raw/compressed", "/cam_r/color/camera_info", 60.0),
]


def compute_focal(fovy_deg):
    return (HEIGHT / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)


def build_camera_info(stamp, frame_id, fx, fy):
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = WIDTH
    msg.height = HEIGHT
    msg.distortion_model = "plumb_bob"
    cx = WIDTH / 2.0
    cy = HEIGHT / 2.0
    msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.K = [fx, 0, cx,  0, fy, cy,  0, 0, 1]
    msg.R = [1, 0, 0,  0, 1, 0,  0, 0, 1]
    msg.P = [fx, 0, cx, 0,  0, fy, cy, 0,  0, 0, 1, 0]
    return msg


def make_callback(pub, fx, fy):
    def cb(msg):
        ci = build_camera_info(msg.header.stamp, msg.header.frame_id, fx, fy)
        pub.publish(ci)
    return cb


if __name__ == "__main__":
    rospy.init_node("sim_camera_info_publisher")

    for img_topic, info_topic, fovy in CAMERAS:
        f = compute_focal(fovy)
        pub = rospy.Publisher(info_topic, CameraInfo, queue_size=1)
        rospy.Subscriber(img_topic, CompressedImage, make_callback(pub, f, f))
        rospy.loginfo("camera_info: %s (fovy=%.0f, f=%.1f)", info_topic, fovy, f)

    rospy.spin()
