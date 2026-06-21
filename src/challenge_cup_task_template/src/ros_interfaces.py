#!/usr/bin/env python3
"""
ROS 接口管理器：封装所有官方允许的 ROS 话题/服务通信。
"""

import rospy
import numpy as np
from collections import deque

from sensor_msgs.msg import JointState, CompressedImage, CameraInfo, PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest
from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
from kuavo_msgs.msg import sensorsData, lejuClawState, endEffectorData


class InterfaceManager:
    def __init__(self):
        self._init_publishers()
        self._init_subscribers()
        self._init_services()

    def _init_publishers(self):
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)

    def _init_subscribers(self):
        self._latest_sensors = None
        self._latest_head_rgb = None
        self._latest_head_depth = None
        self._latest_camera_info = None
        self._latest_claw_state = None
        self._latest_lidar = None
        self._latest_base_pos_xyz = None
        self._latest_base_angular_zyx = None

        rospy.Subscriber("/sensors_data_raw", sensorsData, self._cb_sensors, queue_size=1)
        rospy.Subscriber("/cam_h/color/image_raw/compressed", CompressedImage, self._cb_rgb, queue_size=1)
        rospy.Subscriber("/cam_h/depth/image_raw/compressedDepth", CompressedImage, self._cb_depth, queue_size=1)
        rospy.Subscriber("/cam_h/color/camera_info", CameraInfo, self._cb_camera_info, queue_size=1)
        rospy.Subscriber("/leju_claw_state", lejuClawState, self._cb_claw_state, queue_size=1)

        try:
            rospy.Subscriber("/lidar/points", PointCloud2, self._cb_lidar, queue_size=1)
        except Exception:
            pass

        try:
            rospy.Subscriber("/state_estimate/base/pos_xyz", Float64MultiArray,
                             self._cb_pos_xyz, queue_size=1)
        except Exception:
            pass

        try:
            rospy.Subscriber("/state_estimate/base/angular_zyx", Float64MultiArray,
                             self._cb_angular_zyx, queue_size=1)
        except Exception:
            pass

    def _init_services(self):
        rospy.loginfo("InterfaceManager: waiting for /control_robot_leju_claw service...")
        try:
            rospy.wait_for_service("/control_robot_leju_claw", timeout=10.0)
            self._claw_srv = rospy.ServiceProxy("/control_robot_leju_claw", controlLejuClaw)
            rospy.loginfo("InterfaceManager: /control_robot_leju_claw ready")
        except rospy.ROSException:
            rospy.logerr("InterfaceManager: /control_robot_leju_claw not available")
            self._claw_srv = None

        self._arm_mode_srv = None
        self._arm_mode_initialized = False

    def switch_arm_control_mode(self, mode=2, timeout=10.0):
        if self._arm_mode_srv is None:
            try:
                rospy.wait_for_service("/arm_traj_change_mode", timeout=timeout)
                self._arm_mode_srv = rospy.ServiceProxy(
                    "/arm_traj_change_mode", changeArmCtrlMode
                )
            except rospy.ROSException:
                rospy.logerr("InterfaceManager: /arm_traj_change_mode not available")
                return False

        try:
            req = changeArmCtrlModeRequest()
            req.control_mode = mode
            resp = self._arm_mode_srv(req)
            if resp.result:
                rospy.loginfo("InterfaceManager: arm control mode set to %d (%s)",
                              mode, resp.message)
                self._arm_mode_initialized = True
                return True
            else:
                rospy.logerr("InterfaceManager: arm mode switch failed: %s", resp.message)
                return False
        except rospy.ServiceException as e:
            rospy.logerr("InterfaceManager: arm mode service call failed: %s", e)
            return False

    def _cb_sensors(self, msg):
        self._latest_sensors = msg

    def _cb_rgb(self, msg):
        self._latest_head_rgb = msg

    def _cb_depth(self, msg):
        self._latest_head_depth = msg

    def _cb_camera_info(self, msg):
        self._latest_camera_info = msg

    def _cb_claw_state(self, msg):
        self._latest_claw_state = msg

    def _cb_lidar(self, msg):
        self._latest_lidar = msg

    def _cb_pos_xyz(self, msg):
        self._latest_base_pos_xyz = msg.data

    def _cb_angular_zyx(self, msg):
        self._latest_base_angular_zyx = msg.data

    @property
    def sensors(self):
        return self._latest_sensors

    @property
    def head_rgb(self):
        return self._latest_head_rgb

    @property
    def head_depth(self):
        return self._latest_head_depth

    @property
    def camera_info(self):
        return self._latest_camera_info

    @property
    def claw_state(self):
        return self._latest_claw_state

    @property
    def lidar(self):
        return self._latest_lidar

    @property
    def base_pos_xyz(self):
        return self._latest_base_pos_xyz

    @property
    def base_angular_zyx(self):
        return self._latest_base_angular_zyx

    @property
    def lidar_available(self):
        return self._latest_lidar is not None

    @property
    def base_yaw(self):
        ang = self._latest_base_angular_zyx
        if ang is not None and len(ang) >= 3:
            return float(ang[2])
        quat = self.get_imu_quaternion()
        if quat is not None:
            return self._quat_to_yaw(quat)
        return 0.0

    def get_base_position(self):
        pos = self._latest_base_pos_xyz
        if pos is not None and len(pos) >= 3:
            return np.array([pos[0], pos[1], pos[2]], dtype=np.float64)
        return None

    @staticmethod
    def _quat_to_yaw(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def all_topics_ready(self):
        return (
            self._latest_sensors is not None
            and self._latest_head_rgb is not None
            and self._latest_head_depth is not None
            and self._latest_camera_info is not None
            and self._latest_lidar is not None
        )

    def wait_for_all_topics(self, timeout=10.0):
        start = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.all_topics_ready():
                rospy.loginfo("InterfaceManager: all key topics ready")
                return True
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logerr("InterfaceManager: timeout waiting for topics")
                return False
            rate.sleep()
        return False

    def publish_cmd_vel(self, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

    def stop_base(self):
        self.publish_cmd_vel(0.0, 0.0, 0.0)

    def publish_arm_joints(self, joint_names, joint_positions):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(joint_names)
        msg.position = [np.deg2rad(p) for p in joint_positions]
        self.arm_traj_pub.publish(msg)

    def publish_arm_joints_deg(self, joint_names, joint_positions_deg):
        self.publish_arm_joints(joint_names, joint_positions_deg)

    def open_claw(self, which="both"):
        return self._control_claw(0.0, which)

    def close_claw(self, which="both"):
        return self._control_claw(100.0, which)

    def _control_claw(self, position, which="both"):
        if self._claw_srv is None:
            rospy.logwarn("InterfaceManager: claw service not available")
            return False
        try:
            names = []
            if which in ("both", "right"):
                names.append("right_claw")
            if which in ("both", "left"):
                names.append("left_claw")

            req = controlLejuClawRequest()
            req.data = endEffectorData()
            req.data.name = names
            req.data.position = [position] * len(names)
            req.data.velocity = [50.0] * len(names)
            req.data.effort = [1.0] * len(names)

            resp = self._claw_srv(req)
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr("InterfaceManager: claw service call failed: %s", e)
            return False

    def get_arm_joint_positions(self):
        if self._latest_sensors is None:
            return {}
        joint_q = self._latest_sensors.joint_data.joint_q
        arm_joint_names = [
            "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch",
            "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
            "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch",
            "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
        ]
        positions = {}
        arm_start = 12
        for i, name in enumerate(arm_joint_names):
            idx = arm_start + i
            if idx < len(joint_q):
                positions[name] = np.rad2deg(joint_q[idx])
        return positions

    def get_imu_quaternion(self):
        if self._latest_sensors is None:
            return None
        return self._latest_sensors.imu_data.quat

    def is_claw_closed(self, which="right"):
        if self._latest_claw_state is None:
            return False
        try:
            data = self._latest_claw_state.data
            idx = data.name.index("%s_claw" % which)
            return data.position[idx] > 50.0
        except (ValueError, IndexError, AttributeError):
            return False

    def is_claw_open(self, which="right"):
        if self._latest_claw_state is None:
            return True
        try:
            data = self._latest_claw_state.data
            idx = data.name.index("%s_claw" % which)
            return data.position[idx] < 10.0
        except (ValueError, IndexError, AttributeError):
            return True
