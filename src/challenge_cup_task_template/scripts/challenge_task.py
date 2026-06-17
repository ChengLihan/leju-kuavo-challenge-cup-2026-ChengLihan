#!/usr/bin/env python3
"""
挑战杯三场景统一任务入口。

推荐运行方式：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene2 --seed 3
  rosrun challenge_cup_task_template challenge_task.py --scene scene3 --seed 3
"""

import argparse
import os
import sys
import numpy as np

SCENE_CONFIGS = {
    "scene1": {
        "node_name": "challenge_task_scene1",
        "title": "场景一：包裹称重与摆放",
    },
    "scene2": {
        "node_name": "challenge_task_scene2",
        "title": "场景二：分拣归档",
    },
    "scene3": {
        "node_name": "challenge_task_scene3",
        "title": "场景三：SMT 料盘出库",
    },
}


def _load_launcher():
    try:
        import rospkg
        sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
    except Exception:
        sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "challenge_cup_simulator", "utils")
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher
    return ChallengeSimLauncher


# ═══════════════════════════════════════════════════════════
# Hough 线段 → 矩形匹配（模块级工具函数）
# ═══════════════════════════════════════════════════════════

def _lines_to_infos(lines):
    """将 HoughLinesP 结果转为 [(x1,y1,x2,y2, angle_0_pi, length, mx,my), ...]。
    angle 标准化到 [0, π)。"""
    infos = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        a = np.arctan2(dy, dx)
        if a < 0:
            a += np.pi
        length = np.hypot(dx, dy)
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        infos.append((float(x1), float(y1), float(x2), float(y2),
                      a, length, mx, my))
    return infos


def _angle_diff(a, b):
    """两条线夹角（归一化到 [0, π/2] 内的锐角）。"""
    d = abs(a - b)
    if d > np.pi / 2:
        d = np.pi - d
    return d


def _line_intersection(l1, l2):
    """求两条无限直线交点。每线给 (x1,y1,x2,y2)。"""
    x1, y1, x2, y2 = l1[:4]
    x3, y3, x4, y4 = l2[:4]
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-8:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def _match_rectangles(infos):
    """从 ≤60 条线段 infos 中匹配矩形。角度分8个桶，
    桶内找平行对，桶间交叉找垂直对，复杂度 O(n²/8)≈450。"""
    n = len(infos)
    if n < 4:
        return []

    # 角度分 8 个桶（每个 ~22.5°），桶索引 = 0..7
    buckets = [[] for _ in range(8)]
    for i, info in enumerate(infos):
        bi = min(int(info[4] / (np.pi / 8)), 7)
        buckets[bi].append(i)

    # 桶内找平行对（角度差 < 15°）
    par_pairs = []  # [(i,j), ...]
    for bi in range(8):
        idxs = buckets[bi]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if _angle_diff(infos[idxs[a]][4], infos[idxs[b]][4]) < np.deg2rad(15):
                    par_pairs.append((idxs[a], idxs[b]))

    # 硬限制：最多 200 对
    if len(par_pairs) > 200:
        par_pairs = par_pairs[:200]

    # 配对两对平行线形成矩形（两组方向大致垂直）
    rects = []
    for pa in range(len(par_pairs)):
        i1, i2 = par_pairs[pa]
        a1 = (infos[i1][4] + infos[i2][4]) / 2.0
        for pb in range(pa + 1, len(par_pairs)):
            j1, j2 = par_pairs[pb]
            if len({i1, i2, j1, j2}) != 4:
                continue
            a2 = (infos[j1][4] + infos[j2][4]) / 2.0
            if _angle_diff(a1, a2) < np.deg2rad(40):
                continue

            l1, l2 = infos[i1], infos[i2]
            l3, l4 = infos[j1], infos[j2]

            corners = []
            for (a, b) in [(l1, l3), (l1, l4), (l2, l3), (l2, l4)]:
                pt = _line_intersection(a, b)
                if pt is None:
                    break
                corners.append(pt)
            if len(corners) != 4:
                continue

            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            if w < 20 or h < 20 or max(w, h) / min(w, h) > 8:
                continue

            cx, cy = np.mean(xs), np.mean(ys)
            rects.append((l1, l2, l3, l4, cx, cy))
            if len(rects) >= 4:
                break
        if len(rects) >= 4:
            break

    # 去重：中心距 < 15px 的合并
    filtered = []
    for r in rects:
        if all(np.hypot(r[4] - f[4], r[5] - f[5]) >= 15 for f in filtered):
            filtered.append(r)
    return filtered[:4]


class RobotBase:
    """封装机器人移动、手臂、夹爪、头部等底层 ROS 接口。"""

    def __init__(self):
        import rospy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import JointState
        from kuavo_msgs.msg import robotHeadMotionData

        self._rospy = rospy
        self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
        self._head_pub = rospy.Publisher("/robot_head_motion_data", robotHeadMotionData, queue_size=10)
        self._last_twist = Twist()

        rospy.sleep(0.5)

    def look_at(self, pitch=0.0, yaw=0.0):
        """控制头部俯仰(pitch)和偏航(yaw)，单位：度。"""
        from kuavo_msgs.msg import robotHeadMotionData
        msg = robotHeadMotionData()
        msg.joint_data = [float(yaw), float(pitch)]
        self._head_pub.publish(msg)
        self._rospy.sleep(0.5)

    def move_velocity(self, linear_x=0.0, linear_y=0.0, angular_z=0.0, duration=1.0):
        twist = self._Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        self._cmd_vel_pub.publish(twist)
        self._last_twist = twist
        self._rospy.sleep(duration)
        self._stop()

    def _stop(self):
        twist = self._Twist()
        self._cmd_vel_pub.publish(twist)

    def switch_arm_control_mode(self, mode=2):
        """切换手臂控制模式: 0=保持姿势  1=自动摆手  2=外部控制"""
        from kuavo_msgs.srv import changeArmCtrlMode
        self._rospy.wait_for_service("/humanoid_change_arm_ctrl_mode", timeout=5)
        try:
            srv = self._rospy.ServiceProxy("/humanoid_change_arm_ctrl_mode", changeArmCtrlMode)
            srv(mode)
            self._rospy.sleep(0.3)
        except Exception as e:
            self._rospy.logwarn("切换手臂模式失败: %s", e)

    def send_arm_trajectory(self, left_joints=None, right_joints=None):
        """发送双臂关节角度，各 7 维向量，单位：度。"""
        LEFT_NAMES = ["l_arm_pitch", "l_arm_roll", "l_arm_yaw",
                      "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll"]
        RIGHT_NAMES = ["r_arm_pitch", "r_arm_roll", "r_arm_yaw",
                       "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll"]
        msg = self._JointState()
        if left_joints is not None:
            msg.name += LEFT_NAMES
            msg.position += list(left_joints)
        if right_joints is not None:
            msg.name += RIGHT_NAMES
            msg.position += list(right_joints)
        self._arm_traj_pub.publish(msg)

    @property
    def _Twist(self):
        from geometry_msgs.msg import Twist
        return Twist

    @property
    def _JointState(self):
        from sensor_msgs.msg import JointState
        return JointState

    def control_gripper(self, position_percent, arm="right"):
        """控制夹爪，0 张开，100 闭合。"""
        from kuavo_msgs.srv import controlLejuClaw
        from kuavo_msgs.msg import endEffectorData
        self._rospy.wait_for_service("/control_robot_leju_claw", timeout=5)
        try:
            srv = self._rospy.ServiceProxy("/control_robot_leju_claw", controlLejuClaw)
            data = endEffectorData()
            data.name = [f"{arm}_claw"]
            data.position = [float(position_percent)]
            srv(data)
        except Exception as e:
            self._rospy.logwarn("夹爪控制失败: %s", e)


class Perception:
    """视觉感知模块：RGB/D相机物体检测与3D定位。"""

    def __init__(self):
        import rospy
        from cv_bridge import CvBridge
        import tf2_ros
        from sensor_msgs.msg import CompressedImage, CameraInfo, Image
        from kuavo_msgs.msg import sensorsData

        self._rospy = rospy
        self._bridge = CvBridge()
        self._latest_rgb = {}
        self._latest_depth = None
        self._cam_info = None
        self._cam_frame = None
        self._head_yaw_rad = None
        self._head_pitch_rad = None

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._sub_h = rospy.Subscriber(
            "/cam_h/color/image_raw/compressed", CompressedImage,
            self._cb_rgb("head"), queue_size=1)
        self._sub_l = rospy.Subscriber(
            "/cam_l/color/image_raw/compressed", CompressedImage,
            self._cb_rgb("left_wrist"), queue_size=1)
        self._sub_r = rospy.Subscriber(
            "/cam_r/color/image_raw/compressed", CompressedImage,
            self._cb_rgb("right_wrist"), queue_size=1)
        self._sub_depth = rospy.Subscriber(
            "/cam_h/depth/image_raw/compressedDepth", CompressedImage,
            self._cb_depth, queue_size=1)
        self._sub_info = rospy.Subscriber(
            "/cam_h/color/camera_info", CameraInfo,
            self._cb_cam_info, queue_size=1)
        self._sub_sensors = rospy.Subscriber(
            "/sensors_data_raw", sensorsData,
            self._cb_sensors, queue_size=1)

        self._pub_edge = rospy.Publisher(
            "/challenge/vision/edges", Image, queue_size=1)
        self._pub_contour = rospy.Publisher(
            "/challenge/vision/contours", Image, queue_size=1)

        self._last_viz_edge = None
        self._last_viz_contour = None

        rospy.sleep(1.0)

    # ── callbacks ──────────────────────────────────────

    def _cb_rgb(self, cam_name):
        def cb(msg):
            self._latest_rgb[cam_name] = msg
        return cb

    def _cb_depth(self, msg):
        self._latest_depth = msg

    def _cb_cam_info(self, msg):
        self._cam_info = msg
        self._cam_frame = msg.header.frame_id

    def _cb_sensors(self, msg):
        q = msg.joint_data.joint_q
        # 打印一次数组长度，便于确认索引
        if not hasattr(self, '_sensor_len_logged'):
            self._rospy.loginfo("sensors joint_q 长度=%d  末6位: %s",
                                len(q), str([round(v, 3) for v in q[-6:]]))
            self._sensor_len_logged = True
        # 尝试找 head pitch：文档说末2位 [yaw, pitch]
        if len(q) >= 2:
            self._head_yaw_rad = q[-2]
            self._head_pitch_rad = q[-1]

    # ── 等待数据就绪 ──────────────────────────────────

    def wait_for_data(self, timeout=10.0):
        deadline = self._rospy.Time.now().to_sec() + timeout
        while self._rospy.Time.now().to_sec() < deadline:
            if ("head" in self._latest_rgb and self._latest_depth is not None
                    and self._cam_info is not None):
                return True
            self._rospy.sleep(0.1)
        self._rospy.logwarn("等待相机数据超时")
        return False

    def wait_for_head_pitch(self, target_deg=20.0, tolerance=3.0):
        """等待头部俯仰角到达目标角度（度），超时 3s 返回 False。"""
        target_rad = target_deg * np.pi / 180.0
        tol_rad = tolerance * np.pi / 180.0
        deadline = self._rospy.Time.now().to_sec() + 3.0
        while self._rospy.Time.now().to_sec() < deadline:
            if self._head_pitch_rad is not None:
                error = abs(self._head_pitch_rad - target_rad)
                cur_deg = self._head_pitch_rad * 180.0 / np.pi
                self._rospy.loginfo_throttle(1.0, "head pitch: %.1f°  error=%.1f°",
                                              cur_deg, error * 180.0 / np.pi)
                if error < tol_rad:
                    self._rospy.loginfo("头部已到位: %.1f°", cur_deg)
                    return True
            self._rospy.sleep(0.2)
        self._rospy.logwarn("等待头部到位超时")
        return False

    # ── 图像转 cv2 ────────────────────────────────────

    def _get_cv_image(self, cam_name="head"):
        msg = self._latest_rgb.get(cam_name)
        if msg is None:
            return None
        return self._bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

    def _get_depth_cv(self):
        if self._latest_depth is None:
            return None
        return self._bridge.compressed_imgmsg_to_cv2(self._latest_depth, "passthrough")

    # ── 方块检测（头部 RGB） ────────────────────────────

    def detect_blocks(self):
        """用 Hough 线段拼矩形。只取 15<长度<100 的短边（盒子顶面轮廓），
        剔除外框长边和内部突起杂线。发布边缘图+轮廓标注到可视化 topic。"""
        import cv2
        cv_img = self._get_cv_image("head")
        if cv_img is None:
            return []

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        edges = cv2.dilate(edges, None, iterations=1)

        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self._pub_edge.publish(self._bridge.cv2_to_imgmsg(edge_bgr, "bgr8"))

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=15, maxLineGap=20)

        draw_img = cv_img.copy()
        blocks = []
        n_raw = 0 if lines is None else len(lines)

        if lines is not None:
            # ① 只保留 15 < 长度 < 100 的短边（盒顶轮廓），剔除长线（桌面边）和极短线（噪声）
            lines_short = []
            lines_long = []
            for l in lines:
                x1, y1, x2, y2 = l[0]
                length = np.hypot(x2 - x1, y2 - y1)
                if 15 < length < 100:
                    lines_short.append(l)
                else:
                    lines_long.append(l)

            # ② 画图：长线灰色虚线（被过滤），短线白色实线（参与匹配）
            for l in lines_long:
                x1, y1, x2, y2 = l[0]
                cv2.line(draw_img, (x1, y1), (x2, y2), (80, 80, 80), 1)
            for l in lines_short:
                x1, y1, x2, y2 = l[0]
                cv2.line(draw_img, (x1, y1), (x2, y2), (200, 200, 200), 1)

            # ③ 短线按长度排序取前 60
            short_arr = np.array(lines_short) if lines_short else np.empty((0, 4))
            if len(short_arr) > 60:
                lengths = [np.hypot(l[0][2] - l[0][0], l[0][3] - l[0][1])
                           for l in short_arr]
                short_arr = short_arr[np.argsort(lengths)[::-1][:60]]

            if len(short_arr) >= 4:
                infos = _lines_to_infos(short_arr)
                rects = _match_rectangles(infos)

                COLORS = [(0, 255, 0), (255, 200, 0), (0, 200, 255), (200, 0, 255)]
                for ri, (l1, l2, l3, l4, cx, cy) in enumerate(rects):
                    c = COLORS[ri % len(COLORS)]
                    for l in (l1, l2, l3, l4):
                        cv2.line(draw_img, (int(l[0]), int(l[1])),
                                 (int(l[2]), int(l[3])), c, 2)
                    cv2.circle(draw_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                    blocks.append((int(cx), int(cy), 1000))

        self._last_viz_edge = edge_bgr
        self._last_viz_contour = draw_img
        self._pub_contour.publish(self._bridge.cv2_to_imgmsg(draw_img, "bgr8"))

        n_short = len(lines_short) if lines is not None else 0
        self._rospy.loginfo("detect: img=%dx%d  raw=%d  short_15_100=%d  top60=%d  rects=%d",
                            cv_img.shape[1], cv_img.shape[0],
                            n_raw, n_short, min(n_short, 60), len(blocks))

        blocks.sort(key=lambda b: b[2], reverse=True)
        return blocks

    def republish_viz(self):
        """持久重发最后一帧可视化图，供事后检查。"""
        if hasattr(self, '_last_viz_edge') and self._last_viz_edge is not None:
            self._pub_edge.publish(self._bridge.cv2_to_imgmsg(self._last_viz_edge, "bgr8"))
        if hasattr(self, '_last_viz_contour') and self._last_viz_contour is not None:
            self._pub_contour.publish(self._bridge.cv2_to_imgmsg(self._last_viz_contour, "bgr8"))

    # ── 像素 → 3D 坐标 ───────────────────────────────

    def _project_to_camera_3d(self, u, v, depth_m):
        K = self._cam_info.K
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        return (x, y, depth_m)

    def _get_depth_at(self, u, v):
        depth_cv = self._get_depth_cv()
        if depth_cv is None:
            return None
        h, w = depth_cv.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None

        patch = depth_cv[max(0, v-2):v+3, max(0, u-2):u+3]
        valid = patch[patch > 0]
        if len(valid) == 0:
            return None
        return float(np.median(valid))

    # ── TF 坐标变换 ──────────────────────────────────

    def _transform_to_base(self, x_cam, y_cam, z_cam):
        from geometry_msgs.msg import PointStamped
        p = PointStamped()
        p.header.frame_id = self._cam_frame
        p.header.stamp = self._rospy.Time(0)
        p.point.x = x_cam
        p.point.y = y_cam
        p.point.z = z_cam
        try:
            p_base = self._tf_buffer.transform(p, "base_link",
                                               timeout=self._rospy.Duration(1.0))
            return (p_base.point.x, p_base.point.y, p_base.point.z)
        except Exception as e:
            self._rospy.logwarn("TF 变换失败: %s", e)
            return None

    # ── 对外接口 ──────────────────────────────────────

    def get_blocks_3d(self):
        """检测所有方块并返回其在 base_link 下的 3D 位置列表。"""
        blocks_px = self.detect_blocks()
        if not blocks_px:
            return []

        results = []
        for cx, cy, area in blocks_px:
            d = self._get_depth_at(cx, cy)
            if d is None or d <= 0:
                continue
            xc, yc, zc = self._project_to_camera_3d(cx, cy, d)
            pose = self._transform_to_base(xc, yc, zc)
            if pose is not None:
                results.append((pose, area))
        return results

    def get_nearest_block(self):
        """返回最近方块的 (x, y, z) 在 base_link 下，若无返回 None。"""
        blocks = self.get_blocks_3d()
        if not blocks:
            return None
        blocks.sort(key=lambda b: np.hypot(b[0][0], b[0][1]))
        return blocks[0][0]

    def get_wrist_cv(self, arm="right"):
        """返回腕部相机的 cv2 图像。"""
        key = "right_wrist" if arm == "right" else "left_wrist"
        return self._get_cv_image(key)

    def get_block_center_in_wrist(self, arm="right"):
        """在腕部图像中检测方块中心（边缘检测），返回 (cx, cy) 或 None。"""
        import cv2
        cv_img = self.get_wrist_cv(arm)
        if cv_img is None:
            return None
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = max(contours, key=cv2.contourArea, default=None)
        if best is None or cv2.contourArea(best) < 200:
            return None
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


class Navigation:
    """导航模块：负责将机器人移动到目标位置。"""

    def __init__(self, robot):
        self._robot = robot

    def move_to(self, target_x, target_y, tolerance=0.1):
        """移动到目标点 (target_x, target_y)。"""
        dx = target_x
        dy = target_y
        dist = np.hypot(dx, dy)
        while dist > tolerance:
            vx = np.clip(dx, -0.3, 0.3)
            vy = np.clip(dy, -0.3, 0.3)
            self._robot.move_velocity(linear_x=vx, linear_y=vy, duration=0.5)
            dx -= vx * 0.5
            dy -= vy * 0.5
            dist = np.hypot(dx, dy)

    def rotate_to(self, target_yaw, tolerance=0.05):
        """原地旋转到目标朝向。"""
        import rospy
        yaw_error = target_yaw
        while abs(yaw_error) > tolerance:
            az = np.clip(yaw_error, -0.5, 0.5)
            self._robot.move_velocity(angular_z=az, duration=0.3)
            yaw_error = target_yaw
            rospy.sleep(0.1)


class Manipulation:
    """手臂与夹爪操作模块。"""

    def __init__(self, robot):
        self._robot = robot
        self._arm_mode_set = False

    def _ensure_arm_mode(self):
        """首次调用时切换到外部控制模式。"""
        if not self._arm_mode_set:
            self._robot.switch_arm_control_mode(2)
            self._arm_mode_set = True
            self._robot._rospy.sleep(0.3)

    def move_arm_to_pose(self, arm, target_joints, duration=2.0):
        """将指定手臂移动到目标关节角度（度）。"""
        self._ensure_arm_mode()
        if arm == "left":
            self._robot.send_arm_trajectory(left_joints=target_joints)
        else:
            self._robot.send_arm_trajectory(right_joints=target_joints)
        self._robot._rospy.sleep(duration)

    def pick(self, arm="right"):
        """执行抓取动作。"""
        if arm == "right":
            pregrasp = [0.0, -45.0, 0.0, -60.0, 0.0, 0.0, 0.0]
            grasp = [10.0, -60.0, 0.0, -90.0, 0.0, 0.0, 0.0]
        else:
            pregrasp = [0.0, 45.0, 0.0, -60.0, 0.0, 0.0, 0.0]
            grasp = [-10.0, 60.0, 0.0, -90.0, 0.0, 0.0, 0.0]
        self.move_arm_to_pose(arm, pregrasp)
        self.move_arm_to_pose(arm, grasp)
        self._robot.control_gripper(80, arm)
        self._robot._rospy.sleep(0.5)

    def place(self, arm="right"):
        """执行放置动作。"""
        lift = [0.0, -30.0, 0.0, -45.0, 0.0, 0.0, 0.0]
        self.move_arm_to_pose(arm, lift)
        self._robot.control_gripper(0, arm)
        self._robot._rospy.sleep(0.3)

    def home(self, arm="right"):
        """回到初始位姿。"""
        home_pos = [0.0, 0.0, 0.0, -45.0, 0.0, 0.0, 0.0]
        self.move_arm_to_pose(arm, home_pos)


class Scene1Controller:
    """场景一：包裹称重与摆放（含数据集采集模式）。"""

    def __init__(self, robot, perception, navigation, manipulation, seed=0):
        self._robot = robot
        self._perception = perception
        self._nav = navigation
        self._manip = manipulation
        self._seed = seed
        self._exit_after_run = False

    def run(self):
        import rospy
        import cv2
        rospy.loginfo("=== 场景一：包裹称重与摆放（seed=%d） ===", self._seed)

        self._robot.look_at(pitch=+20.0, yaw=0.0)

        rospy.loginfo("等待头部到位 (+20°) ...")
        self._perception.wait_for_head_pitch(+20.0)

        rospy.loginfo("等待相机数据就绪...")
        if not self._perception.wait_for_data():
            rospy.logerr("相机数据未就绪，退出")
            return

        # 确保 images 目录存在
        images_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "images")
        os.makedirs(images_dir, exist_ok=True)

        crop_size = 640

        # 每个 seed 截取 3 张，间隔 0.8s 以获取不同帧
        for i in range(3):
            head_img = self._perception._get_cv_image("head")
            if head_img is None:
                rospy.logerr("[seed=%d 第%d张] 未能获取头部相机图像", self._seed, i + 1)
                continue

            h, w = head_img.shape[:2]
            x_start = (w - crop_size) // 2
            y_start = h - crop_size
            cropped = head_img[y_start:y_start + crop_size, x_start:x_start + crop_size]

            save_path = os.path.join(images_dir,
                                     "scene1_seed{}_c{}.jpg".format(self._seed, i + 1))
            cv2.imwrite(save_path, cropped)
            rospy.loginfo("已保存: %s (%dx%d)", save_path, crop_size, crop_size)

            if i < 2:
                rospy.sleep(0.8)

        self._exit_after_run = True
        rospy.loginfo("场景一（seed=%d）数据集采集完成，共 3 张", self._seed)

    def _approach_block(self, bx, by):
        """导航到方块前方约 0.4m 处。"""
        import rospy
        approach_x = max(bx - 0.4, 0.1)
        approach_y = by
        rospy.loginfo("导航到接近点: x=%.2f, y=%.2f", approach_x, approach_y)
        self._nav.move_to(approach_x, approach_y, tolerance=0.08)
        self._nav.rotate_to(0.0)
        self._robot.look_at(pitch=-15.0, yaw=0.0)
        rospy.sleep(0.5)

    def _fine_align_and_grasp(self):
        """使用腕部相机精调后抓取。"""
        import rospy

        self._manip.move_arm_to_pose("right", [10.0, -30.0, 0.0, -60.0, 0.0, 0.0, 0.0])
        rospy.sleep(0.5)

        for _ in range(5):
            center = self._perception.get_block_center_in_wrist("right")
            if center is None:
                rospy.loginfo("腕部相机未检测到方块")
                break
            cx, cy = center
            h, w = 480, 640
            err_x = cx - w // 2
            rospy.loginfo("腕部相机: 方块中心 (%d, %d), 偏移 %d", cx, cy, err_x)
            if abs(err_x) < 20:
                rospy.loginfo("精调完成")
                break
            adj = 0.0
            if err_x > 0:
                adj = -3.0
            elif err_x < 0:
                adj = 3.0
            self._robot.send_arm_trajectory(
                right_joints=[10.0 + adj, -30.0, 0.0, -60.0, 0.0, 0.0, 0.0])
            rospy.sleep(0.5)

        self._manip.pick("right")
        rospy.sleep(0.3)


class Scene2Controller:
    """场景二：零件分拣归档（含数据集采集模式）。"""

    def __init__(self, robot, perception, navigation, manipulation, seed=0):
        self._robot = robot
        self._perception = perception
        self._nav = navigation
        self._manip = manipulation
        self._seed = seed
        self._exit_after_run = False

    def run(self):
        import rospy
        import cv2
        rospy.loginfo("=== 场景二：零件分拣归档（seed=%d） ===", self._seed)
        self._robot.look_at(pitch=+20.0, yaw=0.0)

        rospy.loginfo("等待头部到位 (+20°) ...")
        self._perception.wait_for_head_pitch(+20.0)

        rospy.loginfo("等待相机数据就绪...")
        if not self._perception.wait_for_data():
            rospy.logerr("相机数据未就绪，退出")
            return

        # 确保 images 目录存在
        images_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "images")
        os.makedirs(images_dir, exist_ok=True)

        crop_size = 640

        # 每个 seed 截取 3 张，间隔 0.8s 以获取不同帧
        for i in range(3):
            # 获取头部相机图像
            head_img = self._perception._get_cv_image("head")
            if head_img is None:
                rospy.logerr("[seed=%d 第%d张] 未能获取头部相机图像", self._seed, i + 1)
                continue

            h, w = head_img.shape[:2]
            # 截取中下方 640x640 区域
            x_start = (w - crop_size) // 2
            y_start = h - crop_size
            cropped = head_img[y_start:y_start + crop_size, x_start:x_start + crop_size]

            save_path = os.path.join(images_dir,
                                     "scene2_seed{}_c{}.jpg".format(self._seed, i + 1))
            cv2.imwrite(save_path, cropped)
            rospy.loginfo("已保存: %s (%dx%d)", save_path, crop_size, crop_size)

            if i < 2:
                rospy.sleep(0.8)

        self._exit_after_run = True
        rospy.loginfo("场景二（seed=%d）数据集采集完成，共 3 张", self._seed)

    def _detect_and_pick(self):
        # TODO: 识别零件类别
        self._nav.move_to(0.5, 0.0)
        self._manip.pick("right")

    def _sort_and_place(self):
        # TODO: 根据类别导航到对应区域
        self._nav.move_to(0.0, 0.3)
        self._manip.place("right")


class Scene3Controller:
    """场景三：SMT 料盘出库。"""

    def __init__(self, robot, perception, navigation, manipulation):
        self._robot = robot
        self._perception = perception
        self._nav = navigation
        self._manip = manipulation

    def run(self):
        import rospy
        rospy.loginfo("=== 场景三：SMT 料盘出库 ===")
        for i in range(5):
            rospy.loginfo("处理第 %d 个料盘", i + 1)
            # 1. 导航到货架
            # 2. 识别目标料盘
            # 3. 从货架取出
            # 4. 导航到出库区
            # 5. 放置
            self._retrieve_tray()
            self._deliver_tray()
        rospy.loginfo("场景三完成")

    def _retrieve_tray(self):
        self._nav.move_to(0.6, 0.0)
        self._manip.pick("right")

    def _deliver_tray(self):
        self._nav.move_to(0.0, -0.4)
        self._manip.place("right")


def run_scene(scene, seed, node_name=None, timeout=120,
              time_limit=None, timer_gui=True):
    if scene not in SCENE_CONFIGS:
        raise ValueError("unknown scene: {}".format(scene))

    config = SCENE_CONFIGS[scene]
    ChallengeSimLauncher = _load_launcher()

    launcher = ChallengeSimLauncher(
        scene=scene,
        seed=seed,
        match_time_limit=time_limit,
        timer_gui=timer_gui,
    )
    launcher.start(node_name=node_name or config["node_name"], timeout=timeout)

    import rospy
    rospy.loginfo("=== %s任务启动 ===", config["title"])

    robot = RobotBase()
    perception = Perception()
    navigation = Navigation(robot)
    manipulation = Manipulation(robot)

    rospy.sleep(1.0)
    rospy.loginfo("场景实例已初始化。")

    if scene == "scene1":
        ctrl = Scene1Controller(robot, perception, navigation, manipulation, seed=seed)
    elif scene == "scene2":
        ctrl = Scene2Controller(robot, perception, navigation, manipulation, seed=seed)
    elif scene == "scene3":
        ctrl = Scene3Controller(robot, perception, navigation, manipulation)
    else:
        rospy.logerr("未知场景: %s", scene)
        rospy.spin()
        return


    ctrl.run()

    if hasattr(ctrl, '_exit_after_run') and ctrl._exit_after_run:
        rospy.loginfo("数据集采集模式，自动退出。")
        rospy.signal_shutdown("dataset_collection_done")
        return

    rospy.loginfo("任务结束，持续发布可视化结果，Ctrl-C 退出...")
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        perception.republish_viz()
        rate.sleep()


def main():
    parser = argparse.ArgumentParser(description="挑战杯三场景统一任务入口")
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="scene1",
                        help="要启动的比赛场景")
    parser.add_argument("--seed", type=int, default=0,
                        help="场景种子；正式评测 seed 由组委会指定")
    parser.add_argument("--node-name", default=None,
                        help="ROS 节点名；默认按 scene 自动设置")
    parser.add_argument("--timeout", type=int, default=120,
                        help="等待仿真就绪的超时时间，单位秒")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="比赛时长，单位秒；默认读取 CHALLENGE_TIME_LIMIT，未设置则不限时")
    parser.add_argument("--no-timer-gui", action="store_true",
                        help="不弹出计时器窗口，仅保留后台计时日志")
    args = parser.parse_args()

    run_scene(
        scene=args.scene,
        seed=args.seed,
        node_name=args.node_name,
        timeout=args.timeout,
        time_limit=args.time_limit,
        timer_gui=not args.no_timer_gui,
    )


if __name__ == "__main__":
    main()
