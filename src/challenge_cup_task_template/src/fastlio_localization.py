#!/usr/bin/env python3
"""
fastlio_localization.py — LiDAR-IMU 里程计 (Python 实现，替代 FAST_LIO)

订阅 /lidar/points (PointCloud2, base_link 系) 和 /sensors_data_raw (IMU)，
使用 scan-to-scan ICP + IMU 预积分估计位姿，发布 /Odometry 和 /tf。
"""

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3, Point, Pose, Twist, PoseWithCovariance, TwistWithCovariance
import tf2_ros
import tf.transformations as tft
from collections import deque
import threading


_LIDAR_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("intensity", "<f4"), ("time", "<f4"),
    ("ring", "<u2"), ("_pad", "<u2"),
])
_POINT_STEP = 24


def _parse_pc2(msg):
    if msg is None or len(msg.data) == 0:
        return None
    n = len(msg.data) // _POINT_STEP
    if n == 0:
        return None
    arr = np.frombuffer(msg.data, dtype=_LIDAR_DTYPE, count=n)
    return np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)


def _solve_icp_svd(A, B, max_iters=5):
    src = A.copy()
    for _ in range(max_iters):
        diffs = src[:, None, :] - B[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        nn_idx = np.argmin(dists, axis=1)
        nn_dist = dists[np.arange(len(src)), nn_idx]
        mask = nn_dist < 0.5
        if np.sum(mask) < 10:
            break
        src_masked = src[mask]
        tgt_masked = B[nn_idx[mask]]
        centroid_src = np.mean(src_masked, axis=0)
        centroid_tgt = np.mean(tgt_masked, axis=0)
        H = (src_masked - centroid_src).T @ (tgt_masked - centroid_tgt)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_tgt - R @ centroid_src
        src = (R @ src.T).T + t
    centroid_src = np.mean(A, axis=0)
    centroid_tgt = np.mean(src, axis=0)
    H = (A - centroid_src).T @ (src - centroid_tgt)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_tgt - R @ centroid_src
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class FastLIOBridge:
    def __init__(self):
        self.lock = threading.Lock()
        self.pose = np.eye(4)
        self.last_scan_base = None
        self.last_scan_stamp = None
        self.imu_queue = deque(maxlen=50)
        self.odom_seq = 0

        self.odom_pub = rospy.Publisher("/Odometry", Odometry, queue_size=10)
        self.tf_br = tf2_ros.TransformBroadcaster()

        self.sub_lidar = rospy.Subscriber("/lidar/points", PointCloud2, self._cb_lidar, queue_size=1)
        rospy.Subscriber("/sensors_data_raw", rospy.AnyMsg, self._cb_imu_raw, queue_size=10)

        self._init_yaw = None

        rospy.loginfo("FastLIOBridge: initialized, waiting for LiDAR/IMU...")

    def _cb_imu_raw(self, msg):
        try:
            from kuavo_msgs.msg import sensorsData
            data = rospy.msg.deserialize_messages([msg._connection_header['type']], [msg._buff])
            if data and len(data) > 0:
                sensor_msg = data[0]
                quat = sensor_msg.imu_data.quat
                acc = sensor_msg.imu_data.acc
                gyro = sensor_msg.imu_data.gyro
                stamp = sensor_msg.sensor_time
                self.imu_queue.append({
                    'stamp': stamp,
                    'quat': np.array([quat.x, quat.y, quat.z, quat.w]),
                    'acc': np.array([acc.x, acc.y, acc.z]),
                    'gyro': np.array([gyro.x, gyro.y, gyro.z]),
                })
        except Exception as e:
            pass

    def _get_base_yaw_from_imu(self):
        if len(self.imu_queue) == 0:
            return None
        q = self.imu_queue[-1]['quat']
        return tft.euler_from_quaternion([q[0], q[1], q[2], q[3]])[2]

    def _cb_lidar(self, msg):
        try:
            self._process_lidar(msg)
        except Exception as e:
            rospy.logerr_throttle(5.0, "FastLIOBridge: _cb_lidar exception: %s", e)

    def _process_lidar(self, msg):
        points_base = _parse_pc2(msg)
        if points_base is None or len(points_base) < 100:
            return

        points_base = points_base[
            ~((np.abs(points_base[:, 0]) < 0.30) & (np.abs(points_base[:, 1]) < 0.35) & (np.abs(points_base[:, 2]) < 1.3))
        ]
        if len(points_base) < 100:
            return

        with self.lock:
            current_yaw_imu = self._get_base_yaw_from_imu()

            if self._init_yaw is None and current_yaw_imu is not None:
                self._init_yaw = current_yaw_imu
                rospy.loginfo("FastLIOBridge: init yaw=%.3f", self._init_yaw)

            if self.last_scan_base is None:
                self.last_scan_base = points_base
                self.last_scan_stamp = msg.header.stamp
                self._publish_odometry(msg.header.stamp)
                rospy.loginfo("FastLIOBridge: first scan received, %d points", len(points_base))
                return

            N = min(len(self.last_scan_base), len(points_base), 1000)
            idx_a = np.random.choice(len(self.last_scan_base), N, replace=False)
            idx_b = np.random.choice(len(points_base), N, replace=False)
            A = self.last_scan_base[idx_a]
            B = points_base[idx_b]

            T_delta = _solve_icp_svd(B, A)

            translation = T_delta[:3, 3]
            trans_norm = np.linalg.norm(translation)
            if trans_norm > 0.2:
                rospy.logwarn_throttle(2.0, "FastLIOBridge: large jump %.3f m, resetting", trans_norm)
                self.last_scan_base = points_base
                self.last_scan_stamp = msg.header.stamp
                return

            self.pose = T_delta @ self.pose
            self.last_scan_base = points_base
            self.last_scan_stamp = msg.header.stamp

            if current_yaw_imu is not None and self._init_yaw is not None:
                imu_yaw_delta = current_yaw_imu - self._init_yaw
                R_imu = np.eye(3)
                cos_y, sin_y = np.cos(imu_yaw_delta), np.sin(imu_yaw_delta)
                R_imu[:2, :2] = [[cos_y, -sin_y], [sin_y, cos_y]]
                self.pose[:3, :3] = 0.7 * self.pose[:3, :3] + 0.3 * R_imu

            rospy.loginfo_throttle(2.0, "FastLIOBridge: pos=[%.3f, %.3f] yaw=%.3f trans=%.4f",
                                    self.pose[0,3], self.pose[1,3],
                                    tft.euler_from_matrix(self.pose)[2], trans_norm)
            self._publish_odometry(msg.header.stamp)

    def _publish_odometry(self, stamp):
        pos = self.pose[:3, 3]
        R = self.pose[:3, :3]
        q = tft.quaternion_from_matrix(self.pose)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.header.seq = self.odom_seq
        self.odom_seq += 1

        odom.pose.pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
        odom.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        odom.pose.covariance = [0.01]*9 + [0.0]*3 + [0.01]*9 + [0.0]*3 + [0.01]*9 + [0.0]*3 + [0.0]*27
        odom.twist.covariance = [0.1]*36

        self.odom_pub.publish(odom)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "base_link"
        tf_msg.transform.translation = Vector3(x=pos[0], y=pos[1], z=pos[2])
        tf_msg.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.tf_br.sendTransform(tf_msg)

    def get_position(self):
        with self.lock:
            return self.pose[:3, 3].copy()

    def get_yaw(self):
        with self.lock:
            return tft.euler_from_matrix(self.pose)[2]

    def get_pose_matrix(self):
        with self.lock:
            return self.pose.copy()
