#!/usr/bin/env python3
"""RGB-D to xyzrgb point cloud utilities for Scene3 imitation-learning data."""

import math

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _require_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"OpenCV cv2 is required to decode compressed images: {exc}")
    return cv2


def decode_compressed_rgb(msg):
    cv2 = _require_cv2()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("failed to decode compressed RGB image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_compressed_depth(msg, depth_scale=0.001):
    cv2 = _require_cv2()
    raw = bytes(msg.data)
    start = raw.find(PNG_SIGNATURE)
    if start < 0:
        start = 0
    encoded = np.frombuffer(raw[start:], dtype=np.uint8)
    depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("failed to decode compressedDepth image")
    depth = np.asarray(depth)
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) * float(depth_scale)
    return depth.astype(np.float32)


def camera_intrinsics(camera_info_msg):
    k = list(camera_info_msg.K)
    if len(k) < 9:
        raise ValueError("CameraInfo.K must contain 9 values")
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


def rgbd_to_xyzrgb(rgb, depth_m, camera_info_msg, max_depth_m=4.0, stride=1):
    if rgb is None or depth_m is None:
        return np.zeros((0, 6), dtype=np.float32)
    if rgb.shape[:2] != depth_m.shape[:2]:
        h = min(rgb.shape[0], depth_m.shape[0])
        w = min(rgb.shape[1], depth_m.shape[1])
        rgb = rgb[:h, :w]
        depth_m = depth_m[:h, :w]

    fx, fy, cx, cy = camera_intrinsics(camera_info_msg)
    step = max(1, int(stride))
    depth = depth_m[::step, ::step]
    rgb_sample = rgb[::step, ::step]

    valid = np.isfinite(depth) & (depth > 0.0)
    if max_depth_m is not None:
        valid &= depth < float(max_depth_m)
    if not np.any(valid):
        return np.zeros((0, 6), dtype=np.float32)

    vv, uu = np.indices(depth.shape)
    uu = uu.astype(np.float32) * step
    vv = vv.astype(np.float32) * step
    z = depth[valid].astype(np.float32)
    x = (uu[valid] - float(cx)) * z / float(fx)
    y = (vv[valid] - float(cy)) * z / float(fy)
    colors = rgb_sample[valid].astype(np.float32) / 255.0
    return np.column_stack([x, y, z, colors]).astype(np.float32)


def transform_xyzrgb(points, transform_4x4):
    if points.size == 0 or transform_4x4 is None:
        return points
    matrix = np.asarray(transform_4x4, dtype=np.float32).reshape(4, 4)
    xyz = points[:, :3]
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    transformed = (matrix @ np.concatenate([xyz, ones], axis=1).T).T[:, :3]
    out = points.copy()
    out[:, :3] = transformed
    return out


def matrix_from_transform_stamped(transform_stamped):
    transform = transform_stamped.transform
    quat = [
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    ]
    xyz = [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ]
    return quaternion_to_matrix_xyzw(quat, xyz=xyz)


def crop_roi(points, roi):
    if points.size == 0 or not roi:
        return points
    mask = (
        (points[:, 0] >= float(roi["x_min"]))
        & (points[:, 0] <= float(roi["x_max"]))
        & (points[:, 1] >= float(roi["y_min"]))
        & (points[:, 1] <= float(roi["y_max"]))
        & (points[:, 2] >= float(roi["z_min"]))
        & (points[:, 2] <= float(roi["z_max"]))
    )
    return points[mask]


def fixed_size_sample(points, num_points, rng=None):
    num_points = int(num_points)
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if points.size == 0:
        return np.zeros((num_points, 6), dtype=np.float32)
    rng = rng or np.random.default_rng()
    count = points.shape[0]
    replace = count < num_points
    indices = rng.choice(count, size=num_points, replace=replace)
    return points[indices].astype(np.float32)


def roi_center_and_scale(roi, default_scale=0.5):
    if not roi:
        return np.zeros(3, dtype=np.float32), float(default_scale)
    center = np.array(
        [
            0.5 * (float(roi["x_min"]) + float(roi["x_max"])),
            0.5 * (float(roi["y_min"]) + float(roi["y_max"])),
            0.5 * (float(roi["z_min"]) + float(roi["z_max"])),
        ],
        dtype=np.float32,
    )
    spans = [
        abs(float(roi["x_max"]) - float(roi["x_min"])),
        abs(float(roi["y_max"]) - float(roi["y_min"])),
        abs(float(roi["z_max"]) - float(roi["z_min"])),
    ]
    scale = max(max(spans) * 0.5, float(default_scale), 1e-6)
    return center, scale


def normalize_xyzrgb(points, roi=None, scale=None):
    if points.size == 0:
        return points.astype(np.float32)
    center, inferred_scale = roi_center_and_scale(roi)
    scale = float(scale) if scale is not None else inferred_scale
    out = points.copy().astype(np.float32)
    out[:, :3] = (out[:, :3] - center.reshape(1, 3)) / scale
    out[:, 3:6] = np.clip(out[:, 3:6], 0.0, 1.0)
    return out


def roi_for_stage(roi_cfg, stage):
    stage = str(stage or "pregrasp")
    stage_to_roi = roi_cfg.get("stage_to_roi", {})
    roi_name = stage_to_roi.get(stage, "extract_upper")
    return roi_cfg.get("roi", {}).get(roi_name, {})


def build_xyzrgb_sample(
    rgb,
    depth_m,
    camera_info_msg,
    roi,
    num_points=1024,
    normalize=True,
    max_depth_m=4.0,
    stride=2,
    transform_4x4=None,
    rng=None,
):
    raw = rgbd_to_xyzrgb(rgb, depth_m, camera_info_msg, max_depth_m=max_depth_m, stride=stride)
    if transform_4x4 is not None:
        raw = transform_xyzrgb(raw, transform_4x4)
    cropped = crop_roi(raw, roi)
    sampled = fixed_size_sample(cropped, num_points, rng=rng)
    if normalize:
        sampled = normalize_xyzrgb(sampled, roi=roi)
    return sampled, int(cropped.shape[0])


def make_xyzrgb_pointcloud2(points, frame_id, stamp=None):
    import rospy
    from sensor_msgs import point_cloud2
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("r", 12, PointField.FLOAT32, 1),
        PointField("g", 16, PointField.FLOAT32, 1),
        PointField("b", 20, PointField.FLOAT32, 1),
    ]
    header = Header()
    header.stamp = stamp if stamp is not None else rospy.Time.now()
    header.frame_id = frame_id
    cloud = point_cloud2.create_cloud(header, fields, points.astype(np.float32).tolist())
    assert isinstance(cloud, PointCloud2)
    return cloud


def finite_point_ratio(points):
    if points.size == 0:
        return 0.0
    return float(np.isfinite(points).all(axis=1).sum()) / float(points.shape[0])


def quaternion_to_matrix_xyzw(quat_xyzw, xyz=None):
    x, y, z, w = [float(v) for v in quat_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    if xyz is not None:
        matrix[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return matrix
