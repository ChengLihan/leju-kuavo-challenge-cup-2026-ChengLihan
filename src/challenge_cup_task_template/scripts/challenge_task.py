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
    # 公共启动器位于受保护包 challenge_cup_simulator/utils/（选手不可改动），
    # 从那里导入，确保完整性校验无法被绕过。
    try:
        import rospkg
        sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
    except Exception:
        sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "challenge_cup_simulator", "utils")
    sys.path.insert(0, sim_utils)
    from challenge_sim_launcher import ChallengeSimLauncher
    return ChallengeSimLauncher

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
        """发送双臂关节角度（始终14维），单位：度。"""
        LEFT_NAMES = ["l_arm_pitch", "l_arm_roll", "l_arm_yaw",
                      "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll"]
        RIGHT_NAMES = ["r_arm_pitch", "r_arm_roll", "r_arm_yaw",
                       "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll"]
        # 始终发14个关节：未指定的手臂填充零位
        lj = list(left_joints) if left_joints is not None else [0.0] * 7
        rj = list(right_joints) if right_joints is not None else [0.0] * 7
        msg = self._JointState()
        msg.name = LEFT_NAMES + RIGHT_NAMES
        msg.position = lj + rj
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

        # ROS1 tf2: 注册 PointStamped/PoseStamped 等几何消息类型转换
        try:
            import tf2_geometry_msgs  # noqa: F401
            rospy.loginfo("tf2_geometry_msgs 已加载，PointStamped TF 支持已注册")
        except Exception as e:
            rospy.logwarn("tf2_geometry_msgs 导入失败: %s", e)

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

        # ── YOLO 分割检测器 (Scene2) ──
        self._yolo_detector = None
        try:
            # 确保 scripts 目录在 Python 路径中
            _scripts_dir = os.path.dirname(os.path.abspath(__file__))
            if _scripts_dir not in sys.path:
                sys.path.insert(0, _scripts_dir)

            # 注入 yolo_gpu site-packages（仅用于导入 ultralytics）
            import glob as _glob
            _yolo_sp_dirs = []
            for _yolo_base in ["/root/yolo_gpu", os.path.expanduser("~/yolo_gpu")]:
                _sp = os.path.join(_yolo_base, "lib")
                if not os.path.isdir(_sp):
                    continue
                _site_pkgs = sorted(_glob.glob(
                    os.path.join(_sp, "python*", "site-packages")))
                _yolo_sp_dirs = _site_pkgs
                if _site_pkgs:
                    break

            # 临时注入 yolo_gpu site-packages（导入完成后移除，避免污染 ROS 消息类型）
            _injected = []
            for _sp_dir in reversed(_yolo_sp_dirs):
                if _sp_dir not in sys.path:
                    sys.path.insert(0, _sp_dir)
                    _injected.append(_sp_dir)
                    rospy.loginfo("临时注入 yolo_gpu site-packages: %s", _sp_dir)

            # 清除可能被 ROS 预加载的旧 ultralytics 缓存
            for _mod in list(sys.modules):
                if _mod == "ultralytics" or _mod.startswith("ultralytics."):
                    del sys.modules[_mod]

            import traceback as _tb
            rospy.loginfo("尝试导入 scene2_yolo_detector (scripts_dir=%s)...",
                          _scripts_dir)

            from scene2_yolo_detector import Scene2YOLODetector
            rospy.loginfo("scene2_yolo_detector 导入成功")

            # 移除临时注入的路径，恢复 ROS 消息类型优先级
            for _sp_dir in _injected:
                if _sp_dir in sys.path:
                    sys.path.remove(_sp_dir)
            if _injected:
                rospy.loginfo("已移除临时 yolo_gpu site-packages，恢复 ROS 路径")

            # 搜索模型文件（多路径回退，尝试多个文件名）
            _model_names = ["yolov8n_seg_scene2_demo.pt", "best.pt"]
            _candidates = []

            for _model_name in _model_names:
                # 候选1: rospkg 路径
                try:
                    import rospkg
                    _pkg = rospkg.RosPack().get_path("challenge_cup_task_template")
                    _candidates.append(os.path.normpath(
                        os.path.join(_pkg, "..", "..", "models", "yolo", _model_name)))
                except Exception:
                    pass

                # 候选2: __file__ 相对路径
                _candidates.append(os.path.normpath(
                    os.path.join(_scripts_dir, "..", "..", "..", "models", "yolo", _model_name)))

                # 候选3: 工作目录相对路径
                _candidates.append(os.path.normpath(
                    os.path.join(os.getcwd(), "models", "yolo", _model_name)))

            # 候选4: 搜索 runs/ 目录下的 best.pt
            try:
                _ws_root = os.path.normpath(
                    os.path.join(_scripts_dir, "..", "..", ".."))
                import glob as _glob
                _bests = sorted(_glob.glob(
                    os.path.join(_ws_root, "runs", "**", "weights", "best.pt"),
                    recursive=True))
                if _bests:
                    _candidates.append(_bests[-1])  # 最新的 best.pt
            except Exception:
                pass

            # 去重后逐个检查
            model_path = None
            for _c in _candidates:
                rospy.loginfo("  候选路径: %s  exists=%s", _c, os.path.exists(_c))
                if os.path.exists(_c):
                    model_path = _c
                    break

            if model_path is None:
                rospy.logerr("YOLO 模型未找到! 搜索的候选路径:\n  %s\n"
                             "请执行:\n"
                             "  cp $(find runs -path '*/weights/best.pt' "
                             "| sort | tail -1) models/yolo/best.pt",
                             "\n  ".join(_candidates))
            else:
                rospy.loginfo("正在加载 YOLO 模型: %s", model_path)
                self._yolo_detector = Scene2YOLODetector(
                    model_path=model_path, conf=0.15, imgsz=640, device=0)
                rospy.loginfo("YOLO Scene2 检测器已加载 (类别: %s)",
                              self._yolo_detector.class_names)
        except Exception as e:
            import traceback as _tb
            rospy.logerr("YOLO 检测器加载失败:\n%s",
                         _tb.format_exc())

        # 可视化发布器 (YOLO debug)
        self._pub_yolo_debug = rospy.Publisher(
            "/challenge/vision/yolo_debug", Image, queue_size=1)
        self._last_viz_yolo = None

        # RViz MarkerArray (3D 目标位置)
        from visualization_msgs.msg import MarkerArray
        self._pub_markers = rospy.Publisher(
            "/challenge/vision/objects_3d", MarkerArray, queue_size=1)

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
        if not hasattr(self, '_cam_frame_logged'):
            self._rospy.loginfo("camera frame: %s", self._cam_frame)
            self._cam_frame_logged = True

    def _cb_sensors(self, msg):
        q = msg.joint_data.joint_q
        if not hasattr(self, '_sensor_len_logged'):
            self._rospy.loginfo("sensors joint_q 长度=%d  全部: %s",
                                len(q),
                                str([round(v, 3) for v in q]))
            self._sensor_len_logged = True
            # 同时打印关节名列表（如果消息包含）
            if hasattr(msg.joint_data, 'joint_name'):
                names = msg.joint_data.joint_name
                for i, n in enumerate(names):
                    self._rospy.loginfo("  joint[%d] %s = %.3f",
                                        i, n, q[i] if i < len(q) else 0)
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

        msg = self._latest_depth
        depth_raw = None

        # 方法1: cv2.imdecode 直接解压 PNG
        # data 可能带有前导零字节（已观察到 12 字节 padding），
        # 需要定位 PNG 魔数 \x89PNG
        try:
            import cv2 as _cv2
            raw_bytes = msg.data
            # 查找 PNG 魔数
            png_magic = b'\x89PNG\r\n\x1a\n'
            png_start = raw_bytes.find(png_magic)
            if png_start < 0:
                png_start = 0  # 回退：假设无前导
            if png_start > 0 and not hasattr(self, '_depth_padding_logged'):
                self._rospy.loginfo("深度 PNG 前有 %d 字节 padding，跳过", png_start)
                self._depth_padding_logged = True
            depth_raw = _cv2.imdecode(
                np.frombuffer(raw_bytes[png_start:], np.uint8),
                _cv2.IMREAD_UNCHANGED)
            if depth_raw is not None and depth_raw.size > 0:
                if not hasattr(self, '_depth_decode_method_logged'):
                    self._rospy.loginfo("深度图通过 cv2.imdecode 解码成功 "
                                        "(shape=%s dtype=%s)",
                                        depth_raw.shape, depth_raw.dtype)
                    self._depth_decode_method_logged = True
        except Exception as e:
            if not hasattr(self, '_depth_cv2_err'):
                self._rospy.logwarn("cv2.imdecode 失败: %s", e)
                self._depth_cv2_err = True

        # 方法2: cv_bridge compressed_imgmsg_to_cv2 (尝试多种编码)
        if depth_raw is None:
            for encoding in ("passthrough", "32FC1", "16UC1",
                             "16UC1; compressedDepth png",
                             "mono16", "mono8"):
                try:
                    depth_raw = self._bridge.compressed_imgmsg_to_cv2(
                        msg, encoding)
                    if depth_raw is not None and depth_raw.size > 0:
                        if not hasattr(self, '_depth_decode_method_logged'):
                            self._rospy.loginfo("深度图通过 cv_bridge 解码成功 "
                                                "(encoding=%s shape=%s dtype=%s)",
                                                encoding, depth_raw.shape,
                                                depth_raw.dtype)
                            self._depth_decode_method_logged = True
                        break
                except Exception:
                    continue

        if depth_raw is None:
            if not hasattr(self, '_depth_decode_warned'):
                self._rospy.logwarn("compressedDepth 解码失败 "
                                    "(format=%s, data_len=%d, "
                                    "data[:20]=%s)",
                                    getattr(msg, 'format', '?'),
                                    len(msg.data),
                                    msg.data[:20].hex())
                self._depth_decode_warned = True
            return None

        # 转为 float32 米单位
        if depth_raw.dtype == np.uint16:
            depth_m = depth_raw.astype(np.float32) / 1000.0
        elif depth_raw.dtype in (np.float32, np.float64):
            depth_m = depth_raw.astype(np.float32)
        else:
            depth_m = depth_raw.astype(np.float32)

        # 打印一次格式信息
        if not hasattr(self, '_depth_fmt_logged'):
            self._rospy.loginfo("深度图: shape=%s dtype=%s→float32 "
                                "min=%.3f max=%.3f m",
                                depth_raw.shape, depth_raw.dtype,
                                float(np.min(depth_m[depth_m > 0]))
                                if np.any(depth_m > 0) else -1,
                                float(np.max(depth_m)))
            self._depth_fmt_logged = True
        return depth_m

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
        if hasattr(self, '_last_viz_yolo') and self._last_viz_yolo is not None:
            self._pub_yolo_debug.publish(self._bridge.cv2_to_imgmsg(self._last_viz_yolo, "bgr8"))

    # ── YOLO 分割检测 (Scene2) ──────────────────────

    def detect_objects_yolo(self):
        """用 YOLOv8-seg 检测头部相机画面中的 Scene2 零件。
        返回 list[dict]，每个实例含 class_name/confidence/bbox/mask/center_uv。
        """
        if self._yolo_detector is None:
            self._rospy.logwarn("YOLO 检测器未加载，无法检测")
            return []

        cv_img = self._get_cv_image("head")
        if cv_img is None:
            self._rospy.logwarn("未获取到头部相机图像")
            return []

        instances = self._yolo_detector.detect(cv_img)
        self._rospy.loginfo("YOLO 检测到 %d 个目标", len(instances))

        # 发布 YOLO debug 可视化
        if instances:
            vis = self._yolo_detector.draw_results(cv_img, instances)
            self._last_viz_yolo = vis
            self._pub_yolo_debug.publish(self._bridge.cv2_to_imgmsg(vis, "bgr8"))

        return instances

    # ── YOLO + Depth + TF → 3D (Scene2 第六步) ─────

    def get_objects_3d_yolo(self):
        """YOLO分割 + 深度图 + TF → base_link 坐标系下的3D目标列表。

        流程:
          1. YOLO 检测 → mask / center_uv  (始终执行+发布debug图)
          2. mask 区域取稳定深度中值
          3. 像素 + 深度 → 相机3D
          4. TF 变换 → base_link

        返回 list[dict]:
          { class_name, confidence, center_uv, position_camera, position_base,
            mask_area, target_bin }
        """
        if self._yolo_detector is None:
            if not hasattr(self, '_yolo_warned'):
                self._rospy.logwarn("YOLO 检测器未加载，跳过检测")
                self._yolo_warned = True
            return []

        cv_img = self._get_cv_image("head")
        if cv_img is None:
            self._rospy.logwarn_throttle(5, "未获取到头部相机RGB图像")
            return []

        # ── 步骤0: YOLO 检测 (始终执行，发布debug图) ──
        t0 = self._rospy.Time.now()
        instances = self._yolo_detector.detect(cv_img)
        dt = (self._rospy.Time.now() - t0).to_sec() * 1000
        self._rospy.loginfo_throttle(2, "YOLO: %d 目标 %.0fms",
                                      len(instances), dt)

        # 始终发布 YOLO debug 图（稍后追加重投影点）
        vis = self._yolo_detector.draw_results(cv_img, instances)

        if not instances:
            self._last_viz_yolo = vis
            self._pub_yolo_debug.publish(self._bridge.cv2_to_imgmsg(vis, "bgr8"))
            return []

        # ── 步骤1-4: depth + TF → 3D (失败不影响 YOLO 检测) ──
        depth_cv = self._get_depth_cv()
        cam_ok = self._cam_info is not None

        # 状态仅首次打印（或出错时）
        if not hasattr(self, '_status_ok'):
            self._rospy.loginfo("感知链路就绪: depth cam_info tf_buffer 全部可用")
            self._status_ok = True

        if depth_cv is None:
            return instances

        if not cam_ok:
            return instances

        # 获取相机内参矩阵
        K = self._cam_info.K
        camera_matrix = np.array([[K[0], 0, K[2]],
                                  [0, K[4], K[5]],
                                  [0, 0, 1]], dtype=np.float64)

        results = []
        tf_ok = 0
        for inst in instances:
            mask = inst["mask"]
            u, v = inst["center_uv"]

            # 步骤1: mask → 稳定深度 (depth_cv 已经是米单位 float32)
            Z = self._yolo_detector.get_mask_median_depth(depth_cv, mask)
            if Z is None or Z <= 0:
                continue

            # 步骤2: 像素 → 相机3D
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
            X_cam = (u - cx) * Z / fx
            Y_cam = (v - cy) * Z / fy
            Z_cam = Z

            # 步骤3: TF 变换 → base_link
            pose_base = self._transform_to_base(X_cam, Y_cam, Z_cam)
            if pose_base is None:
                continue
            tf_ok += 1

            # 步骤4: 重投影验证 (base → camera → uv)
            cam_reproj = self._transform_from_base(*pose_base)
            reproj_uv = None
            if cam_reproj is not None:
                fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
                cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
                u_reproj = cam_reproj[0] * fx / cam_reproj[2] + cx
                v_reproj = cam_reproj[1] * fy / cam_reproj[2] + cy
                reproj_uv = [round(u_reproj, 1), round(v_reproj, 1)]

            results.append({
                "class_name":       inst["class_name"],
                "class_id":         inst["class_id"],
                "confidence":       inst["confidence"],
                "center_uv":        inst["center_uv"],
                "reproj_uv":        reproj_uv,
                "position_camera":  [round(X_cam, 4), round(Y_cam, 4), round(Z_cam, 4)],
                "position_base":    [round(x, 4) for x in pose_base],
                "mask_area":        inst["mask_area"],
                "target_bin":       inst["target_bin"],
            })

        # ── 追加重投影验证点到 debug 图 ──
        import cv2 as _cv2
        for r in results:
            if r.get("reproj_uv"):
                ur, vr = int(r["reproj_uv"][0]), int(r["reproj_uv"][1])
                _cv2.drawMarker(vis, (ur, vr), (0, 255, 0),
                                _cv2.MARKER_CROSS, 20, 2)
                uc, vc = int(r["center_uv"][0]), int(r["center_uv"][1])
                _cv2.line(vis, (uc, vc), (ur, vr), (0, 255, 255), 1)

        self._last_viz_yolo = vis
        self._pub_yolo_debug.publish(self._bridge.cv2_to_imgmsg(vis, "bgr8"))

        # ── 发布 RViz MarkerArray ──
        self._publish_3d_markers(results)

        self._rospy.loginfo_throttle(2, "YOLO 3D: %d/%d TF=%d",
                                      len(results), len(instances), tf_ok)

        # 如果 YOLO 有结果但 3D 全失败，返回原始 instances，
        # 让上层知道是 depth/TF 问题，而不是 YOLO 没检测到。
        if not results and instances:
            return instances

        return results

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
        """camera frame 下 3D 点转换到 base_link。"""
        from geometry_msgs.msg import PointStamped

        # ROS1 tf2 必须导入这个，才能注册 PointStamped 的 do_transform
        import tf2_geometry_msgs  # noqa: F401

        if self._cam_frame is None:
            self._rospy.logwarn_throttle(2.0, "cam_frame 为空，无法 TF")
            return None

        p = PointStamped()
        p.header.frame_id = self._cam_frame
        p.header.stamp = self._rospy.Time(0)
        p.point.x = float(x_cam)
        p.point.y = float(y_cam)
        p.point.z = float(z_cam)

        try:
            p_base = self._tf_buffer.transform(
                p, "base_link",
                timeout=self._rospy.Duration(1.0))
            return (p_base.point.x, p_base.point.y, p_base.point.z)
        except Exception as e:
            self._rospy.logwarn_throttle(
                2.0,
                "TF 变换失败: %s -> base_link, err=%s",
                self._cam_frame, e)
            return None

    def _publish_3d_markers(self, results):
        """发布 RViz MarkerArray：每个目标在 base_link 下的彩色球体 + 文字。"""
        from visualization_msgs.msg import Marker, MarkerArray

        BIN_COLORS = {
            "blue_bin":   (0.0, 0.0, 1.0),
            "orange_bin": (1.0, 0.5, 0.0),
            "purple_bin": (1.0, 0.0, 1.0),
        }

        markers = MarkerArray()
        for i, r in enumerate(results):
            x, y, z = r["position_base"]
            color = BIN_COLORS.get(r.get("target_bin", ""), (0.0, 1.0, 0.0))

            # 球体
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self._rospy.Time.now()
            m.ns = "objects_3d"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.03
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 0.8
            markers.markers.append(m)

            # 文字标签
            mt = Marker()
            mt.header.frame_id = "base_link"
            mt.header.stamp = self._rospy.Time.now()
            mt.ns = "objects_3d_labels"
            mt.id = i + 1000
            mt.type = Marker.TEXT_VIEW_FACING
            mt.action = Marker.ADD
            mt.pose.position.x = x
            mt.pose.position.y = y
            mt.pose.position.z = z + 0.04
            mt.scale.z = 0.03
            mt.color.r, mt.color.g, mt.color.b = color
            mt.color.a = 1.0
            mt.text = "{} {:.2f}".format(r["class_name"], r["confidence"])
            markers.markers.append(mt)

        # 删除旧 marker（如果数量减少）
        for j in range(len(results), 20):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self._rospy.Time.now()
            m.ns = "objects_3d"
            m.id = j
            m.action = Marker.DELETE
            markers.markers.append(m)

        self._pub_markers.publish(markers)

    def _transform_from_base(self, x_base, y_base, z_base):
        """base_link → camera frame 逆向变换，用于重投影验证。"""
        from geometry_msgs.msg import PointStamped
        import tf2_geometry_msgs  # noqa: F401

        if self._cam_frame is None:
            return None

        p = PointStamped()
        p.header.frame_id = "base_link"
        p.header.stamp = self._rospy.Time(0)
        p.point.x = float(x_base)
        p.point.y = float(y_base)
        p.point.z = float(z_base)

        try:
            p_cam = self._tf_buffer.transform(
                p, self._cam_frame,
                timeout=self._rospy.Duration(1.0))
            return (p_cam.point.x, p_cam.point.y, p_cam.point.z)
        except Exception as e:
            self._rospy.logwarn_throttle(
                2.0, "逆向 TF 失败 base_link->%s: %s", self._cam_frame, e)
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
            yaw_error -= az * 0.3
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
    """场景二：零件分拣归档。

    管线:
      头部相机 YOLO + 深度 + TF → 3D 目标列表
      → 按类别排序选择目标
      → 导航 + 预抓取 → 腕部相机二次校准
      → 抓取 → 移动到对应颜色箱 → 放置
    """

    # ── 类别 → 目标箱 ──
    BIN_MAP = {
        "screwdriver": "purple_bin",
        "pipe_clamp":  "blue_bin",
        "pipe_fitting": "orange_bin",
    }

    # ── 目标箱 → base_link 下的大致放置位置 ──
    BIN_POSITIONS = {
        "purple_bin":  (0.60,  0.25),
        "orange_bin":  (0.60,  0.00),
        "blue_bin":    (0.60, -0.25),
    }

    # 从上往下抓取：末端局部 -Z 轴接近 base_link 的 -Z 方向。
    # 右手姿态来自 scene2 固定抓取数据；左手为其左右镜像。
    TOP_DOWN_QUAT = {
        "right": [-0.081987, -0.152343, 0.857876, 0.483858],
        "left":  [-0.081987,  0.152343, 0.857876, -0.483858],
    }
    # 放置时夹爪水平：末端-Z指向base_link-X方向（水平前伸放入箱内）
    HORIZONTAL_PLACE_QUAT = {
        "right": [0.0, 0.707, 0.0, 0.707],
        "left":  [0.0, 0.707, 0.0, 0.707],
    }
    GRASP_APPROACH_CLEARANCE = 0.25
    GRASP_DESCEND_CLEARANCE = -0.1
    TABLE_GUARD_CLEARANCE = 0.20
    TABLE_EE_MIN_CLEARANCE = 0.08
    IK_MODE_POS_HARD_ORI_SOFT = 0x02
    IK_MODE_THREE_POINT_MIXED = 0x06
    IK_TIMEOUT = 0.8
    FK_POSITION_WARN = 0.04
    DOWN_AXIS_WARN_DEG = 25.0
    GRASP_TF_ORIENTATION_TOL_DEG = 10.0
    GRASP_TF_MAX_CHECKS = 5
    GRASP_BACK_OFFSET_PER_DEG = 0.001
    GRASP_BACK_OFFSET_MAX = 0.04
    GRASP_SETTLE_AFTER_COMPENSATION = 1.0
    PLACE_GUARD_Z = 0.26
    PLACE_RELEASE_Z = 0.16

    def __init__(self, robot, perception, navigation, manipulation, seed=0):
        self._robot = robot
        self._perception = perception
        self._nav = navigation
        self._manip = manipulation
        self._seed = seed
        self._exit_after_run = False
        # 双臂状态追踪：始终同步发送双手，避免单手动作导致另一手回零
        self._last_left  = [0.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._last_right = [0.0, -60.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _send_arms(self, left=None, right=None):
        """发送双臂命令，保持未指定手臂的 last_known 状态。"""
        if left is not None:
            self._last_left = list(left)
        if right is not None:
            self._last_right = list(right)
        self._robot.send_arm_trajectory(
            left_joints=self._last_left, right_joints=self._last_right)

    def _hold_arms(self):
        """重发当前双臂位置，防止外部控制器超时释放。"""
        self._robot.send_arm_trajectory(
            left_joints=self._last_left, right_joints=self._last_right)

    def _sleep_hold(self, duration):
        """带手臂位置保持的延时, 每秒刷新一次防掉落。"""
        import rospy
        end = rospy.Time.now().to_sec() + duration
        while rospy.Time.now().to_sec() < end:
            self._hold_arms()
            rospy.sleep(min(0.8, max(0.1, end - rospy.Time.now().to_sec())))

    def _move_arm_slow(self, arm, target_joints, duration=2.0, steps=12):
        """对单臂做线性插值，减少抓取后抬起和放箱时的冲击。"""
        import rospy
        start = list(self._last_left if arm == "left" else self._last_right)
        target = list(target_joints)
        steps = max(1, int(steps))
        dt = float(duration) / float(steps)
        for i in range(1, steps + 1):
            alpha = float(i) / float(steps)
            point = [
                start[j] + (target[j] - start[j]) * alpha
                for j in range(7)
            ]
            self._send_arms(**{arm: point})
            rospy.sleep(dt)

    # 关节限位
    JOINT_LIMITS_R = {
        # 来源: biped_v3_arm.urdf zarm_r1~r7 关节限位 (度)
        "r_arm_pitch":  (-137, 34),
        "r_arm_roll":   (-84,  20),
        "r_arm_yaw":    (-26,  90),
        "r_forearm":    (-150, 0),
        "r_hand_yaw":   (-90,  90),
        "r_hand_pitch": (-40,  40),
        "r_hand_roll":  (-40,  75),
    }
    JOINT_LIMITS_L = {
        # 来源: biped_v3_arm.urdf zarm_l1~l7 关节限位 (度)
        "l_arm_pitch":  (-137, 34),
        "l_arm_roll":   (-20,  84),
        "l_arm_yaw":    (-90,  26),
        "l_forearm":    (-150, 0),
        "l_hand_yaw":   (-90,  90),
        "l_hand_pitch": (-40,  40),
        "l_hand_roll":  (-75,  40),
    }
    LIMIT_KEYS = ["arm_pitch", "arm_roll", "arm_yaw",
                  "forearm", "hand_yaw", "hand_pitch", "hand_roll"]

    def _clamp_joints(self, joints, arm):
        limits = self.JOINT_LIMITS_R if arm == "right" else self.JOINT_LIMITS_L
        prefix = "r_" if arm == "right" else "l_"
        clamped = []
        for i, key in enumerate(self.LIMIT_KEYS):
            lo, hi = limits[prefix + key]
            clamped.append(max(lo, min(hi, joints[i])))
        return clamped

    def _compute_arm_to_target(self, bx, by, bz, arm="right", z_offset=0.15):
        """根据 base_link 目标计算手臂关节角。
        z_offset: 手部参考点高于目标的高度(m), 默认0.15=15cm。
        肩关节原点来自 URDF (biped_v3_arm):
          zarm_r1: xyz="-0.003 -0.255 0.283" / zarm_l1: xyz="-0.003 +0.255 0.283"
          waist_yaw_joint offset: z=0.1114  → 肩≈(0, ±0.255, 0.395)"""
        import math
        shoulder_z = 0.395
        shoulder_y = -0.255 if arm == "right" else 0.255
        target_z = bz + z_offset

        dx = bx
        dy = by - shoulder_y
        dz = target_z - shoulder_z  # 负值 = 目标在肩下方
        dist_xy = math.hypot(dx, dy)
        dist_3d = math.hypot(dist_xy, dz)

        # 肩部 roll：左右方向
        if arm == "right":
            roll = -math.degrees(math.atan2(abs(dy), max(dx, 0.01)))
        else:
            roll = +math.degrees(math.atan2(abs(dy), max(dx, 0.01)))

        # IK: 2 连杆 (L1=上臂, L2=前臂) 在 roll 对齐平面内
        L1, L2 = 0.32, 0.35
        d = min(dist_3d, L1 + L2 - 0.02)
        cos_elbow = (d**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        elbow = math.degrees(math.acos(cos_elbow))
        # shoulder pitch: 直接用 dist_xy (roll 后的平面距离)
        beta = math.atan2(L2 * math.sin(math.radians(elbow)),
                          L1 + L2 * math.cos(math.radians(elbow)))
        # 两种解：pitch1 (elbow-down) 和 pitch2 (elbow-up)
        pitch1 = math.degrees(math.atan2(dist_xy, -dz) - beta)
        pitch2 = math.degrees(math.atan2(dist_xy, -dz) + beta)
        # 选 pitch 在 [-137,34] 内且使肘更前伸的解
        pitch = pitch1 if (-137 <= pitch1 <= 34) else pitch2
        if not (-137 <= pitch <= 34):
            pitch = pitch1  # 回退

        # r_arm_pitch: 负=前伸, 正=后摆，故取反
        # r_forearm_pitch: URDF [-150,0], 负=肘弯曲，故取反
        joints = [round(-pitch, 1), round(roll, 1), 0.0,
                  round(-elbow, 1), 0.0, 0.0, 0.0]
        rospy = __import__('rospy')
        rospy.loginfo("IK: dx=%.2f dy=%.2f dz=%.2f d=%.2f -> pitch=%.1f→%.1f roll=%.1f elbow=%.1f",
                       dist_xy, dy, dz, d, pitch, -pitch, roll, -elbow)
        return joints  # 原始值, 由调用方 clamp 并检测可达性

    def _target_quat_for_arm(self, arm):
        quat = list(self.TOP_DOWN_QUAT[arm])
        norm = float(np.linalg.norm(quat))
        if norm > 1e-8:
            quat = [v / norm for v in quat]
        return quat

    def _current_arm_joints_rad(self, timeout=0.2):
        """读取当前双臂 14 维关节(rad)，失败则使用本控制器记录的 last joints。"""
        import math
        import rospy
        try:
            from kuavo_msgs.msg import sensorsData
            msg = rospy.wait_for_message("/sensors_data_raw",
                                         sensorsData,
                                         timeout=timeout)
            q = list(msg.joint_data.joint_q)
            if len(q) >= 27:
                return q[13:27]
            if len(q) >= 26:
                return q[12:26]
        except Exception as e:
            rospy.logwarn_throttle(5.0,
                                   "读取当前手臂关节失败，使用last joints: %s", e)
        return [math.radians(v) for v in (self._last_left + self._last_right)]

    def _make_ik_param(self, constraint_mode, pos_cost_weight):
        from kuavo_msgs.msg import ikSolveParam
        param = ikSolveParam()
        param.major_optimality_tol = 1e-3
        param.major_feasibility_tol = 1e-3
        param.minor_feasibility_tol = 1e-3
        param.major_iterations_limit = 500
        param.oritation_constraint_tol = 1e-3
        param.pos_constraint_tol = 1e-3
        param.pos_cost_weight = float(pos_cost_weight)
        param.constraint_mode = int(constraint_mode)
        return param

    def _call_fk(self, q14_rad, timeout=None):
        import rospy
        try:
            from kuavo_msgs.srv import fkSrv
            rospy.wait_for_service("/ik/fk_srv",
                                   timeout=timeout or self.IK_TIMEOUT)
            res = rospy.ServiceProxy("/ik/fk_srv", fkSrv)(list(q14_rad))
            if not res.success:
                rospy.logwarn("FK服务返回失败")
                return None
            return res.hand_poses
        except Exception as e:
            rospy.logwarn_throttle(5.0, "FK服务不可用: %s", e)
            return None

    def _quat_angle_error(self, q1, q2):
        import math
        if q1 is None or q2 is None:
            return None
        dot = sum(float(q1[i]) * float(q2[i]) for i in range(4))
        dot = max(-1.0, min(1.0, abs(dot)))
        return 2.0 * math.acos(dot)

    def _table_safe_z(self, object_z, clearance):
        """用物体高度近似桌面参考，非抓取阶段保持末端离桌面足够高。"""
        return float(object_z + max(clearance, self.TABLE_EE_MIN_CLEARANCE))

    def _quat_to_matrix(self, quat_xyzw):
        q = np.array(quat_xyzw, dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-8:
            return np.eye(3)
        x, y, z, w = q / n
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
             2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
             2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w),
             1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    def _down_axis_error(self, quat_xyzw):
        """末端局部 -Z 与 base_link -Z 的夹角(rad)。"""
        import math
        rot = self._quat_to_matrix(quat_xyzw)
        local_minus_z = -rot[:, 2]
        target_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        denom = np.linalg.norm(local_minus_z)
        if denom < 1e-8:
            return None
        dot = float(np.dot(local_minus_z / denom, target_down))
        dot = max(-1.0, min(1.0, dot))
        return math.acos(dot)

    def _solve_top_down_ik(self, arm, target_xyz, target_quat,
                           seed_q14=None,
                           constraint_mode=None, pos_cost_weight=2.0):
        """调用 URDF IK 服务，把末端位置和朝下姿态一起纳入求解。"""
        import math
        import rospy
        try:
            from kuavo_msgs.msg import twoArmHandPoseCmd
            from kuavo_msgs.srv import twoArmHandPoseCmdSrv

            current = (list(seed_q14) if seed_q14 is not None
                       else self._current_arm_joints_rad(timeout=0.2))
            fk_poses = self._call_fk(current, timeout=self.IK_TIMEOUT)
            if fk_poses is None:
                return None

            req = twoArmHandPoseCmd()
            req.use_custom_ik_param = True
            req.joint_angles_as_q0 = True
            req.ik_param = self._make_ik_param(
                constraint_mode if constraint_mode is not None
                else self.IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight,
            )
            req.hand_poses.left_pose.joint_angles = list(current[:7])
            req.hand_poses.right_pose.joint_angles = list(current[7:])
            req.hand_poses.left_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]
            req.hand_poses.right_pose.elbow_pos_xyz = [0.0, 0.0, 0.0]

            if arm == "left":
                req.hand_poses.left_pose.pos_xyz = list(target_xyz)
                req.hand_poses.left_pose.quat_xyzw = list(target_quat)
                req.hand_poses.right_pose.pos_xyz = list(fk_poses.right_pose.pos_xyz)
                req.hand_poses.right_pose.quat_xyzw = list(fk_poses.right_pose.quat_xyzw)
            else:
                req.hand_poses.left_pose.pos_xyz = list(fk_poses.left_pose.pos_xyz)
                req.hand_poses.left_pose.quat_xyzw = list(fk_poses.left_pose.quat_xyzw)
                req.hand_poses.right_pose.pos_xyz = list(target_xyz)
                req.hand_poses.right_pose.quat_xyzw = list(target_quat)

            rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv",
                                   timeout=self.IK_TIMEOUT)
            res = rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv",
                                     twoArmHandPoseCmdSrv)(req)
            if not res.success:
                rospy.logwarn("Top-down IK失败: %s",
                              getattr(res, "error_reason", "unknown"))
                return None

            if len(res.q_arm) >= 14:
                q14 = list(res.q_arm[:14])
            else:
                left_q = list(res.hand_poses.left_pose.joint_angles)
                right_q = list(res.hand_poses.right_pose.joint_angles)
                if len(left_q) != 7 or len(right_q) != 7:
                    rospy.logwarn("IK返回关节维度异常: left=%d right=%d",
                                  len(left_q), len(right_q))
                    return None
                q14 = left_q + right_q

            joints_rad = q14[:7] if arm == "left" else q14[7:14]
            joints_deg = [math.degrees(v) for v in joints_rad]
            joints_deg = self._clamp_joints(joints_deg, arm)
            return joints_deg, q14
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Top-down IK服务不可用: %s", e)
            return None

    def _fallback_top_down_joints(self, target_xyz, arm):
        """IK服务不可用时的几何IK回退：位置优先，腕部pitch补偿朝下。"""
        bx, by, bz = target_xyz
        raw = self._compute_arm_to_target(bx, by, bz, arm, z_offset=0.0)
        joints = self._clamp_joints(raw, arm)
        # 近似让 shoulder pitch + elbow + wrist pitch 的总pitch回到竖直。
        joints[5] = -joints[0] - joints[3]
        joints = self._clamp_joints(joints, arm)
        return joints

    def _plan_top_down_grasp(self, bx, by, bz, arm):
        target_quat = self._target_quat_for_arm(arm)
        guard_xyz = [
            float(bx),
            float(by),
            self._table_safe_z(bz, self.TABLE_GUARD_CLEARANCE),
        ]
        # 桌面避障：approach 不低于 bz（桌面），grasp 最多低于 bz 5cm 夹取。
        approach_z = max(float(bz + self.GRASP_APPROACH_CLEARANCE), float(bz))
        grasp_z = max(float(bz + self.GRASP_DESCEND_CLEARANCE), float(bz) - 0.05)
        approach_xyz = [
            float(bx),
            float(by),
            approach_z,
        ]
        grasp_xyz = [
            float(bx),
            float(by),
            grasp_z,
        ]

        guard = self._solve_top_down_ik(
            arm, guard_xyz, target_quat,
            constraint_mode=self.IK_MODE_THREE_POINT_MIXED,
            pos_cost_weight=2.0,
        )
        approach = None
        if guard is not None:
            approach = self._solve_top_down_ik(
                arm, approach_xyz, target_quat,
                seed_q14=guard[1],
                constraint_mode=self.IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight=2.0,
            )
        grasp = None
        if approach is not None:
            grasp = self._solve_top_down_ik(
                arm, grasp_xyz, target_quat,
                seed_q14=approach[1],
                constraint_mode=self.IK_MODE_THREE_POINT_MIXED,
                pos_cost_weight=2.0,
            )

        if guard is not None and approach is not None and grasp is not None:
            guard_joints, guard_q14 = guard
            approach_joints, approach_q14 = approach
            grasp_joints, grasp_q14 = grasp
            self._verify_fk_plan(arm, guard_q14, guard_xyz,
                                 target_quat, "guard", table_z_ref=bz)
            self._verify_fk_plan(arm, approach_q14, approach_xyz,
                                 target_quat, "approach", table_z_ref=bz)
            self._verify_fk_plan(arm, grasp_q14, grasp_xyz,
                                 target_quat, "grasp")
            return {
                "arm": arm,
                "guard_xyz": guard_xyz,
                "approach_xyz": approach_xyz,
                "grasp_xyz": grasp_xyz,
                "target_quat": target_quat,
                "guard_joints": guard_joints,
                "approach_joints": approach_joints,
                "grasp_joints": grasp_joints,
                "used_ik": True,
            }

        rospy = __import__('rospy')
        rospy.logwarn("Top-down IK不可用，使用几何IK回退规划")
        return {
            "arm": arm,
            "guard_xyz": guard_xyz,
            "approach_xyz": approach_xyz,
            "grasp_xyz": grasp_xyz,
            "target_quat": target_quat,
            "guard_joints": self._fallback_top_down_joints(guard_xyz, arm),
            "approach_joints": self._fallback_top_down_joints(approach_xyz, arm),
            "grasp_joints": self._fallback_top_down_joints(grasp_xyz, arm),
            "used_ik": False,
        }

    def _verify_fk_plan(self, arm, q14_rad, target_xyz, target_quat, label,
                        table_z_ref=None):
        import math
        import rospy
        fk = self._call_fk(q14_rad, timeout=self.IK_TIMEOUT)
        if fk is None:
            return
        pose = fk.left_pose if arm == "left" else fk.right_pose
        actual = np.array(list(pose.pos_xyz), dtype=np.float64)
        desired = np.array(target_xyz, dtype=np.float64)
        pos_err = float(np.linalg.norm(actual - desired))
        quat_err = self._quat_angle_error(list(pose.quat_xyzw), target_quat)
        down_err = self._down_axis_error(list(pose.quat_xyzw))
        rospy.loginfo(
            "[FK校验:%s/%s] pos_err=%.3fm quat_err=%s down_err=%s actual=(%.3f,%.3f,%.3f)",
            arm, label, pos_err,
            "%.1f°" % math.degrees(quat_err) if quat_err is not None else "n/a",
            "%.1f°" % math.degrees(down_err) if down_err is not None else "n/a",
            actual[0], actual[1], actual[2])
        if pos_err > self.FK_POSITION_WARN:
            rospy.logwarn("[FK校验:%s/%s] 末端位置误差偏大: %.3fm",
                          arm, label, pos_err)
        if table_z_ref is not None:
            margin = float(actual[2] - table_z_ref)
            rospy.loginfo("[桌面安全:%s/%s] ee_z_margin=%.3fm",
                          arm, label, margin)
            if margin < self.TABLE_EE_MIN_CLEARANCE:
                rospy.logwarn("[桌面安全:%s/%s] 末端离桌面参考过近: %.3fm",
                              arm, label, margin)

    def _verify_tf_pose(self, arm, target_xyz, target_quat, label,
                        table_z_ref=None):
        """用当前 TF 树检查实际末端 frame，验证 URDF/TF 下的落点和朝向。"""
        import math
        import rospy
        frame = "zarm_l7_end_effector" if arm == "left" else "zarm_r7_end_effector"
        try:
            tf = self._perception._tf_buffer.lookup_transform(
                "base_link", frame, rospy.Time(0), rospy.Duration(0.5))
            trans = tf.transform.translation
            rot = tf.transform.rotation
            actual_xyz = np.array([trans.x, trans.y, trans.z], dtype=np.float64)
            desired_xyz = np.array(target_xyz, dtype=np.float64)
            actual_quat = [rot.x, rot.y, rot.z, rot.w]
            pos_err = float(np.linalg.norm(actual_xyz - desired_xyz))
            quat_err = self._quat_angle_error(actual_quat, target_quat)
            down_err = self._down_axis_error(actual_quat)
            rospy.loginfo(
                "[TF校验:%s/%s] frame=%s pos_err=%.3fm quat_err=%s down_err=%s actual=(%.3f,%.3f,%.3f)",
                arm, label, frame, pos_err,
                "%.1f°" % math.degrees(quat_err) if quat_err is not None else "n/a",
                "%.1f°" % math.degrees(down_err) if down_err is not None else "n/a",
                actual_xyz[0], actual_xyz[1], actual_xyz[2])
            quat_err_deg = math.degrees(quat_err) if quat_err is not None else None
            down_err_deg = math.degrees(down_err) if down_err is not None else None
            if down_err_deg is not None and down_err_deg > self.DOWN_AXIS_WARN_DEG:
                rospy.logwarn("[TF校验:%s/%s] 夹爪朝下角度偏差 %.1f°",
                              arm, label, down_err_deg)
            if table_z_ref is not None:
                margin = float(actual_xyz[2] - table_z_ref)
                rospy.loginfo("[桌面安全TF:%s/%s] ee_z_margin=%.3fm",
                              arm, label, margin)
                if margin < self.TABLE_EE_MIN_CLEARANCE:
                    rospy.logwarn("[桌面安全TF:%s/%s] 末端离桌面参考过近: %.3fm",
                                  arm, label, margin)
            return {
                "pos_err": pos_err,
                "quat_err_deg": quat_err_deg,
                "down_err_deg": down_err_deg,
                "actual_xyz": actual_xyz.tolist(),
                "actual_quat": actual_quat,
            }
        except Exception as e:
            rospy.logwarn_throttle(5.0, "TF校验失败 %s -> base_link: %s",
                                   frame, e)
            return None

    def _wait_for_grasp_tf_ready(self, arm, plan):
        """闭爪前最多5轮TF检查；超限后按角度误差后移补偿并继续抓取。"""
        import rospy
        tol = self.GRASP_TF_ORIENTATION_TOL_DEG
        last_err = None
        for check_i in range(1, self.GRASP_TF_MAX_CHECKS + 1):
            if rospy.is_shutdown():
                return False
            self._send_arms(**{arm: plan["grasp_joints"]})
            result = self._verify_tf_pose(
                arm, plan["grasp_xyz"], plan["target_quat"],
                "grasp_wait_%d" % check_i)
            if result is not None and result.get("quat_err_deg") is not None:
                err = result["quat_err_deg"]
                last_err = err
                if err <= tol:
                    rospy.loginfo("[GraspReady] %s臂 TF姿态误差 %.1f° <= %.1f°，开始抓取",
                                  arm, err, tol)
                    return True
                rospy.loginfo("[GraspWait %d/%d] %s臂 TF姿态误差 %.1f° > %.1f°",
                              check_i, self.GRASP_TF_MAX_CHECKS,
                              arm, err, tol)
            self._sleep_hold(0.4)

        if last_err is not None:
            self._compensate_grasp_backward(arm, plan, last_err)
        else:
            rospy.logwarn("[GraspWait] %s臂 5轮内没有有效TF角度，保持当前抓取位姿", arm)

        self._send_arms(**{arm: plan["grasp_joints"]})
        self._sleep_hold(self.GRASP_SETTLE_AFTER_COMPENSATION)
        self._verify_tf_pose(arm, plan["grasp_xyz"],
                             plan["target_quat"], "grasp_compensated")
        rospy.loginfo("[GraspReady] %s臂 5轮TF检查结束，等待稳定后继续抓取", arm)
        return True

    def _compensate_grasp_backward(self, arm, plan, quat_err_deg):
        """姿态误差越大，抓取点沿base_link -X方向轻微后移。"""
        import rospy
        excess = max(0.0, float(quat_err_deg) - self.GRASP_TF_ORIENTATION_TOL_DEG)
        offset = min(self.GRASP_BACK_OFFSET_MAX,
                     excess * self.GRASP_BACK_OFFSET_PER_DEG)
        if offset <= 1e-4:
            rospy.loginfo("[GraspComp] %s臂 姿态误差%.1f°，无需后移补偿",
                          arm, quat_err_deg)
            return

        new_xyz = list(plan["grasp_xyz"])
        new_xyz[0] -= offset
        solved = self._solve_top_down_ik(
            arm, new_xyz, plan["target_quat"],
            constraint_mode=self.IK_MODE_THREE_POINT_MIXED,
            pos_cost_weight=2.0,
        )
        if solved is not None:
            new_joints, _q14 = solved
        else:
            new_joints = self._fallback_top_down_joints(new_xyz, arm)

        plan["grasp_xyz"] = new_xyz
        plan["grasp_joints"] = new_joints
        rospy.loginfo("[GraspComp] %s臂 姿态误差%.1f°，抓取点向后补偿 %.3fm -> (%.3f,%.3f,%.3f)",
                      arm, quat_err_deg, offset,
                      new_xyz[0], new_xyz[1], new_xyz[2])

    def _plan_bin_place(self, target_class, arm, bz=None):
        """为已抓取物体规划箱子上方和箱内释放点。
        bz: 物体/桌面在 base_link 下的 Z 坐标，用于计算桌面相对安全高度。
        """
        import rospy
        bin_name = self.BIN_MAP.get(target_class)
        if bin_name is None or bin_name not in self.BIN_POSITIONS:
            rospy.logwarn("[放箱] 未知类别/箱子: %s", target_class)
            return None

        bin_x, bin_y = self.BIN_POSITIONS[bin_name]
        target_quat = list(self.HORIZONTAL_PLACE_QUAT[arm])
        # 放箱避障：有 bz 时使用桌面相对安全高度；否则用旧绝对常量回退。
        if bz is not None:
            place_guard_z = self._table_safe_z(bz, self.TABLE_GUARD_CLEARANCE)
            place_release_z = float(bz) + 0.04  # 箱内释放点：桌面以上4cm，低于前壁(6cm)
        else:
            place_guard_z = self.PLACE_GUARD_Z
            place_release_z = self.PLACE_RELEASE_Z
        guard_xyz = [float(bin_x), float(bin_y), place_guard_z]
        release_xyz = [float(bin_x), float(bin_y), place_release_z]

        guard = self._solve_top_down_ik(
            arm, guard_xyz, target_quat,
            constraint_mode=self.IK_MODE_POS_HARD_ORI_SOFT,
            pos_cost_weight=0.0,
        )
        release = None
        if guard is not None:
            release = self._solve_top_down_ik(
                arm, release_xyz, target_quat,
                seed_q14=guard[1],
                constraint_mode=self.IK_MODE_POS_HARD_ORI_SOFT,
                pos_cost_weight=0.0,
            )

        if guard is not None and release is not None:
            guard_joints, guard_q14 = guard
            release_joints, release_q14 = release
            self._verify_fk_plan(arm, guard_q14, guard_xyz,
                                 target_quat, "place_guard", table_z_ref=bz)
            self._verify_fk_plan(arm, release_q14, release_xyz,
                                 target_quat, "place_release", table_z_ref=bz)
            return {
                "bin_name": bin_name,
                "guard_xyz": guard_xyz,
                "release_xyz": release_xyz,
                "target_quat": target_quat,
                "guard_joints": guard_joints,
                "release_joints": release_joints,
                "used_ik": True,
            }

        rospy.logwarn("[放箱] IK不可用，使用几何IK回退放置")
        return {
            "bin_name": bin_name,
            "guard_xyz": guard_xyz,
            "release_xyz": release_xyz,
            "target_quat": target_quat,
            "guard_joints": self._fallback_top_down_joints(guard_xyz, arm),
            "release_joints": self._fallback_top_down_joints(release_xyz, arm),
            "used_ik": False,
        }

    def _place_object_in_bin(self, target_class, arm, bz=None):
        import rospy
        place = self._plan_bin_place(target_class, arm, bz=bz)
        if place is None:
            return False

        rospy.loginfo("[放箱] %s -> %s guard=(%.3f,%.3f,%.3f) release=(%.3f,%.3f,%.3f) ik=%s",
                      target_class, place["bin_name"],
                      place["guard_xyz"][0], place["guard_xyz"][1], place["guard_xyz"][2],
                      place["release_xyz"][0], place["release_xyz"][1], place["release_xyz"][2],
                      place["used_ik"])

        rospy.loginfo("[PlaceGuard] %s臂 慢速移动到箱子上方", arm)
        self._move_arm_slow(arm, place["guard_joints"], duration=2.5, steps=15)
        self._sleep_hold(0.4)
        self._verify_tf_pose(arm, place["guard_xyz"],
                             place["target_quat"], "place_guard", table_z_ref=bz)

        rospy.loginfo("[PlaceRelease] %s臂 慢速下放到箱内释放点", arm)
        self._move_arm_slow(arm, place["release_joints"], duration=2.0, steps=12)
        self._sleep_hold(0.5)
        self._verify_tf_pose(arm, place["release_xyz"],
                             place["target_quat"], "place_release", table_z_ref=bz)

        self._robot.control_gripper(0, arm)
        rospy.sleep(0.8)
        rospy.loginfo("[PlaceRelease] %s臂 已打开夹爪释放", arm)

        rospy.loginfo("[PlaceRetreat] %s臂 慢速退回箱子上方", arm)
        self._move_arm_slow(arm, place["guard_joints"], duration=1.5, steps=10)
        self._sleep_hold(0.4)
        return True

    # ── 腕部相机颜色检测 + 二次校准 ─────────────────

    def _detect_white_in_wrist(self, arm="right"):
        """手腕相机检测白色物体(螺丝刀)。HSV: 低饱和度+高亮度。
        返回 (cx, cy) 像素坐标，未检测到返回 None。"""
        import cv2
        cv_img = self._perception.get_wrist_cv(arm)
        if cv_img is None:
            return None
        h, w = cv_img.shape[:2]
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        lower = (0, 0, 160)
        upper = (180, 45, 255)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 80]
        if not valid:
            return None
        best = max(valid, key=cv2.contourArea)
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        import rospy
        rospy.loginfo("[腕部] 白色目标 @ (%d,%d) area=%.0f img=%dx%d",
                      cx, cy, cv2.contourArea(best), w, h)
        return (cx, cy)

    def _detect_black_in_wrist(self, arm="right"):
        """手腕相机检测黑色物体(pipe_clamp)。HSV: 低亮度。
        返回 (cx, cy) 像素坐标，未检测到返回 None。"""
        import cv2
        cv_img = self._perception.get_wrist_cv(arm)
        if cv_img is None:
            return None
        h, w = cv_img.shape[:2]
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        lower = (0, 0, 0)
        upper = (180, 255, 60)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 80]
        if not valid:
            return None
        best = max(valid, key=cv2.contourArea)
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        import rospy
        rospy.loginfo("[腕部] 黑色目标 @ (%d,%d) area=%.0f img=%dx%d",
                      cx, cy, cv2.contourArea(best), w, h)
        return (cx, cy)

    def _fine_align_wrist(self, base_joints, arm="right", max_iter=8, detector=None):
        """根据腕部相机颜色检测微调手臂关节角。
        detector: 颜色检测函数, 默认白色检测。
        返回调整后的 joints list[7]（度）。"""
        import rospy
        if detector is None:
            detector = self._detect_white_in_wrist
        joints = list(base_joints)
        for it in range(max_iter):
            rospy.sleep(0.6)
            center = detector(arm)
            if center is None:
                rospy.loginfo("[校准] 第%d次未检测到目标", it + 1)
                continue

            cx, cy = center
            w = 1280
            h = 720
            err_x = cx - w // 2
            err_y = cy - h // 2
            rospy.loginfo("[校准 %d/%d] err_x=%d err_y=%d",
                          it + 1, max_iter, err_x, err_y)

            if abs(err_x) < 25 and abs(err_y) < 25:
                rospy.loginfo("[校准] 已对齐")
                break

            # 水平偏移 → 调 arm_roll
            step_x = 1.5 if abs(err_x) < 60 else 3.0
            if err_x > 0:
                joints[1] -= step_x   # 目标偏右 → 手臂更外摆
            else:
                joints[1] += step_x

            # 垂直偏移 → 调 arm_pitch
            step_y = 1.5 if abs(err_y) < 60 else 3.0
            if err_y > 0:
                joints[0] -= step_y   # 目标偏下 → 手臂前伸
            else:
                joints[0] += step_y

            joints = self._clamp_joints(joints, arm)
            self._send_arms(**{arm: joints})
            rospy.sleep(0.8)

        return joints

    def _execute_pick(self, current_joints, arm="right"):
        """从当前关节位置执行抓取: 下降 → 夹紧 → 抬起。
        forearm URDF [-150,0]: -150=最弯(手近), 0=全伸展(手远)。
        下降→forearm趋向0(伸展), 抬起→forearm趋向-150(弯曲)。"""
        import rospy
        # 下降: forearm伸展(手向下/外), arm_pitch微前伸保持手掌水平
        lower = [
            current_joints[0] - 5,        # arm_pitch 略前伸
            current_joints[1],            # arm_roll 保持
            current_joints[2],            # arm_yaw 保持
            min(current_joints[3] + 25, 0),  # forearm 伸展(趋向0=手下降)
            current_joints[4],            # hand_yaw 保持
            current_joints[5],            # hand_pitch 保持(掌心向下)
            current_joints[6],            # hand_roll 保持
        ]
        clamped_lower = self._clamp_joints(lower, arm)
        rospy.loginfo("[抓取] 下降: %s", clamped_lower)
        self._send_arms(**{arm: clamped_lower})
        self._sleep_hold(2.0)

        # 夹紧
        self._robot.control_gripper(70, arm)
        rospy.sleep(0.6)
        rospy.loginfo("[抓取] 夹爪闭合")

        # 抬起: forearm弯曲(手靠近身体), arm后收内收
        lift = [
            current_joints[0] + 15,       # arm_pitch 后收
            current_joints[1] - 15,       # arm_roll 内收
            current_joints[2],
            current_joints[3] - 25,       # forearm 弯曲(趋向-150=手抬起)
            current_joints[4],            # hand_yaw 保持
            current_joints[5],            # hand_pitch 保持
            current_joints[6],            # hand_roll 保持
        ]
        clamped_lift = self._clamp_joints(lift, arm)
        rospy.loginfo("[抓取] 抬起: %s", clamped_lift)
        self._send_arms(**{arm: clamped_lift})
        self._sleep_hold(2.0)

    # ═══════════════════════════════════════════════════════
    # 各物体类型专用抓取流程
    # ═══════════════════════════════════════════════════════

    def _pick_place_pipe_fitting(self, bx, by, bz, arm=None):
        """pipe_fitting 专用：base_link坐标 → 规划 → 抓取 → 放箱。
        Args:
            bx, by, bz: 物体在 base_link 下的坐标
            arm: 指定手臂("left"/"right")，None则按Y自动选臂
        Returns:
            bool: True=成功, False=失败需跳过
        """
        import rospy
        import math

        if arm is None:
            arm = "left" if by > 0 else "right"
        rospy.loginfo("[pipe_fitting] 目标 Y=%.3f → 选%s臂", by, arm)

        plan = self._plan_top_down_grasp(bx, by, bz, arm)
        if plan is None:
            fallback = "right" if arm == "left" else "left"
            rospy.loginfo("[pipe_fitting] %s臂规划失败, 切换%s臂", arm, fallback)
            arm = fallback
            plan = self._plan_top_down_grasp(bx, by, bz, arm)
            if plan is None:
                rospy.logerr("[pipe_fitting] 双臂均无法生成top-down抓取规划!")
                return False

        rospy.loginfo("[TopDown] %s臂 guard=(%.3f,%.3f,%.3f) "
                      "approach=(%.3f,%.3f,%.3f) grasp=(%.3f,%.3f,%.3f) ik=%s",
                      arm,
                      plan["guard_xyz"][0], plan["guard_xyz"][1], plan["guard_xyz"][2],
                      plan["approach_xyz"][0], plan["approach_xyz"][1], plan["approach_xyz"][2],
                      plan["grasp_xyz"][0], plan["grasp_xyz"][1], plan["grasp_xyz"][2],
                      plan["used_ik"])

        self._robot.control_gripper(0, arm)
        rospy.sleep(0.2)

        rospy.loginfo("[Guard] %s臂 桌面避障高位: %s",
                      arm, [round(v, 1) for v in plan["guard_joints"]])
        self._move_arm_slow(arm, plan["guard_joints"], duration=2.0, steps=12)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["guard_xyz"],
                             plan["target_quat"], "guard", table_z_ref=bz)

        rospy.loginfo("[Approach] %s臂 夹爪正上方: %s",
                      arm, [round(v, 1) for v in plan["approach_joints"]])
        self._move_arm_slow(arm, plan["approach_joints"], duration=2.0, steps=12)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["approach_xyz"],
                             plan["target_quat"], "approach", table_z_ref=bz)

        rospy.loginfo("[Descend] %s臂 垂直下探抓取点: %s",
                      arm, [round(v, 1) for v in plan["grasp_joints"]])
        self._move_arm_slow(arm, plan["grasp_joints"], duration=1.5, steps=10)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["grasp_xyz"],
                             plan["target_quat"], "grasp")

        rospy.loginfo("[DescendRepeat] %s臂 重复确认夹爪到物体方位", arm)
        self._move_arm_slow(arm, plan["grasp_joints"], duration=1.5, steps=10)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["grasp_xyz"],
                             plan["target_quat"], "grasp_repeat")

        if not self._wait_for_grasp_tf_ready(arm, plan):
            rospy.logwarn("[Grasp] %s臂 TF姿态未满足10°要求，跳过闭爪", arm)
            return False

        self._robot.control_gripper(70, arm)
        rospy.sleep(0.6)
        rospy.loginfo("[Grasp] %s臂 夹爪闭合 pipe_fitting", arm)

        rospy.loginfo("[LiftSlow] %s臂 缓慢抬回物体正上方", arm)
        self._move_arm_slow(arm, plan["approach_joints"],
                            duration=2.0, steps=12)
        self._sleep_hold(0.4)

        rospy.loginfo("[RetreatSlow] %s臂 缓慢退回桌面避障高位", arm)
        self._move_arm_slow(arm, plan["guard_joints"],
                            duration=1.5, steps=10)
        self._sleep_hold(0.4)

        self._place_object_in_bin("pipe_fitting", arm, bz=bz)
        return True

    def _pick_place_pipe_clamp(self, bx, by, bz, arm=None):
        """pipe_clamp 专用：base_link坐标 → 规划 → 抓取 → 放箱(蓝色)。"""
        import rospy
        import math
        target_class = "pipe_clamp"
        if arm is None:
            arm = "left" if by > 0 else "right"
        rospy.loginfo("[%s] 目标 Y=%.3f → 选%s臂", target_class, by, arm)

        plan = self._plan_top_down_grasp(bx, by, bz, arm)
        if plan is None:
            fallback = "right" if arm == "left" else "left"
            rospy.loginfo("[%s] %s臂规划失败, 切换%s臂", target_class, arm, fallback)
            arm = fallback
            plan = self._plan_top_down_grasp(bx, by, bz, arm)
            if plan is None:
                rospy.logerr("[%s] 双臂均无法生成top-down抓取规划!", target_class)
                return False

        rospy.loginfo("[TopDown] %s臂 guard=(%.3f,%.3f,%.3f) "
                      "approach=(%.3f,%.3f,%.3f) grasp=(%.3f,%.3f,%.3f) ik=%s",
                      arm,
                      plan["guard_xyz"][0], plan["guard_xyz"][1], plan["guard_xyz"][2],
                      plan["approach_xyz"][0], plan["approach_xyz"][1], plan["approach_xyz"][2],
                      plan["grasp_xyz"][0], plan["grasp_xyz"][1], plan["grasp_xyz"][2],
                      plan["used_ik"])

        self._robot.control_gripper(0, arm)
        rospy.sleep(0.2)

        rospy.loginfo("[Guard] %s臂 桌面避障高位: %s",
                      arm, [round(v, 1) for v in plan["guard_joints"]])
        self._move_arm_slow(arm, plan["guard_joints"], duration=2.0, steps=12)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["guard_xyz"],
                             plan["target_quat"], "guard", table_z_ref=bz)

        rospy.loginfo("[Approach] %s臂 夹爪正上方: %s",
                      arm, [round(v, 1) for v in plan["approach_joints"]])
        self._move_arm_slow(arm, plan["approach_joints"], duration=2.0, steps=12)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["approach_xyz"],
                             plan["target_quat"], "approach", table_z_ref=bz)

        rospy.loginfo("[Descend] %s臂 垂直下探抓取点: %s",
                      arm, [round(v, 1) for v in plan["grasp_joints"]])
        self._move_arm_slow(arm, plan["grasp_joints"], duration=1.5, steps=10)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["grasp_xyz"],
                             plan["target_quat"], "grasp")

        rospy.loginfo("[DescendRepeat] %s臂 重复确认夹爪到物体方位", arm)
        self._move_arm_slow(arm, plan["grasp_joints"], duration=1.5, steps=10)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["grasp_xyz"],
                             plan["target_quat"], "grasp_repeat")

        if not self._wait_for_grasp_tf_ready(arm, plan):
            rospy.logwarn("[Grasp] %s臂 TF姿态未满足10°要求，跳过闭爪", arm)
            return False

        self._robot.control_gripper(70, arm)
        rospy.sleep(0.6)
        rospy.loginfo("[Grasp] %s臂 夹爪闭合 %s", arm, target_class)

        rospy.loginfo("[LiftSlow] %s臂 缓慢抬回物体正上方", arm)
        self._move_arm_slow(arm, plan["approach_joints"],
                            duration=2.0, steps=12)
        self._sleep_hold(0.4)

        rospy.loginfo("[RetreatSlow] %s臂 缓慢退回桌面避障高位", arm)
        self._move_arm_slow(arm, plan["guard_joints"],
                            duration=1.5, steps=10)
        self._sleep_hold(0.4)

        self._place_object_in_bin(target_class, arm, bz=bz)
        return True

    def _pick_place_screwdriver(self, bx, by, bz, arm=None):
        """screwdriver 专用：base_link坐标 → 规划 → 抓取 → 放箱(紫色)。"""
        import rospy
        import math
        target_class = "screwdriver"
        if arm is None:
            arm = "left" if by > 0 else "right"
        rospy.loginfo("[%s] 目标 Y=%.3f → 选%s臂", target_class, by, arm)

        plan = self._plan_top_down_grasp(bx, by, bz, arm)
        if plan is None:
            fallback = "right" if arm == "left" else "left"
            rospy.loginfo("[%s] %s臂规划失败, 切换%s臂", target_class, arm, fallback)
            arm = fallback
            plan = self._plan_top_down_grasp(bx, by, bz, arm)
            if plan is None:
                rospy.logerr("[%s] 双臂均无法生成top-down抓取规划!", target_class)
                return False

        rospy.loginfo("[TopDown] %s臂 guard=(%.3f,%.3f,%.3f) "
                      "approach=(%.3f,%.3f,%.3f) grasp=(%.3f,%.3f,%.3f) ik=%s",
                      arm,
                      plan["guard_xyz"][0], plan["guard_xyz"][1], plan["guard_xyz"][2],
                      plan["approach_xyz"][0], plan["approach_xyz"][1], plan["approach_xyz"][2],
                      plan["grasp_xyz"][0], plan["grasp_xyz"][1], plan["grasp_xyz"][2],
                      plan["used_ik"])

        self._robot.control_gripper(0, arm)
        rospy.sleep(0.2)

        rospy.loginfo("[Guard] %s臂 桌面避障高位: %s",
                      arm, [round(v, 1) for v in plan["guard_joints"]])
        self._move_arm_slow(arm, plan["guard_joints"], duration=2.0, steps=12)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["guard_xyz"],
                             plan["target_quat"], "guard", table_z_ref=bz)

        rospy.loginfo("[Approach] %s臂 夹爪正上方: %s",
                      arm, [round(v, 1) for v in plan["approach_joints"]])
        self._move_arm_slow(arm, plan["approach_joints"], duration=2.0, steps=12)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["approach_xyz"],
                             plan["target_quat"], "approach", table_z_ref=bz)

        rospy.loginfo("[Descend] %s臂 垂直下探抓取点: %s",
                      arm, [round(v, 1) for v in plan["grasp_joints"]])
        self._move_arm_slow(arm, plan["grasp_joints"], duration=1.5, steps=10)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["grasp_xyz"],
                             plan["target_quat"], "grasp")

        rospy.loginfo("[DescendRepeat] %s臂 重复确认夹爪到物体方位", arm)
        self._move_arm_slow(arm, plan["grasp_joints"], duration=1.5, steps=10)
        self._sleep_hold(0.3)
        self._verify_tf_pose(arm, plan["grasp_xyz"],
                             plan["target_quat"], "grasp_repeat")

        if not self._wait_for_grasp_tf_ready(arm, plan):
            rospy.logwarn("[Grasp] %s臂 TF姿态未满足10°要求，跳过闭爪", arm)
            return False

        self._robot.control_gripper(70, arm)
        rospy.sleep(0.6)
        rospy.loginfo("[Grasp] %s臂 夹爪闭合 %s", arm, target_class)

        rospy.loginfo("[LiftSlow] %s臂 缓慢抬回物体正上方", arm)
        self._move_arm_slow(arm, plan["approach_joints"],
                            duration=2.0, steps=12)
        self._sleep_hold(0.4)

        rospy.loginfo("[RetreatSlow] %s臂 缓慢退回桌面避障高位", arm)
        self._move_arm_slow(arm, plan["guard_joints"],
                            duration=1.5, steps=10)
        self._sleep_hold(0.4)

        self._place_object_in_bin(target_class, arm, bz=bz)
        return True

    # ═══════════════════════════════════════════════════════
    # 主流程: 检测 → 校准 → 抓取 → 放箱
    # ═══════════════════════════════════════════════════════

    def run(self):
        import rospy
        import math
        rospy.loginfo("=== 场景二：零件分拣归档（seed=%d） ===", self._seed)

        # ═══ Phase 1: 初始化 — 低头+双臂60°侧举 ═══
        self._robot.look_at(pitch=+20.0, yaw=0.0)
        self._robot.switch_arm_control_mode(2)
        rospy.sleep(0.5)

        pregrasp_left  = [0.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pregrasp_right = [0.0, -60.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._send_arms(left=pregrasp_left, right=pregrasp_right)
        self._sleep_hold(1.0)
        rospy.loginfo("[初始化] 双臂60°侧举")

        rospy.loginfo("等待头部到位...")
        self._perception.wait_for_head_pitch(+20.0)

        rospy.loginfo("等待相机数据就绪...")
        if not self._perception.wait_for_data():
            rospy.logerr("相机数据未就绪，退出")
            return

        self._robot.switch_arm_control_mode(2)
        rospy.sleep(0.3)
        self._send_arms(left=pregrasp_left, right=pregrasp_right)
        rospy.sleep(0.5)
        rospy.loginfo("[初始化] 手臂模式重锁")

        # ═══ Phase 2: 首次检测 — 收集所有类别物体位置 ═══
        TARGET_CLASSES = ["pipe_fitting", "pipe_clamp", "screwdriver"]
        PICK_FUNCTIONS = {
            "pipe_fitting": self._pick_place_pipe_fitting,
            "pipe_clamp":  self._pick_place_pipe_clamp,
            "screwdriver": self._pick_place_screwdriver,
        }
        self._sleep_hold(0.5)
        self._perception.republish_viz()

        objects = self._perception.get_objects_3d_yolo()
        objects_3d = [o for o in (objects or [])
                      if "position_base" in o and o["confidence"] >= 0.5
                      and o["class_name"] in TARGET_CLASSES
                      and abs(o["position_base"][0]) < 2.0]

        if not objects_3d:
            rospy.logerr("未检测到任何物体")
        else:
            rospy.loginfo("检测到 %d 个物体", len(objects_3d))

            # 按类别分组，每组内按左→右排序 (Y从大到小)
            class_positions = {}
            for o in objects_3d:
                cls = o["class_name"]
                class_positions.setdefault(cls, []).append(
                    (o["position_base"][0], o["position_base"][1], o["position_base"][2]))

            for cls in TARGET_CLASSES:
                positions = class_positions.get(cls, [])
                if not positions:
                    rospy.loginfo("[%s] 未检测到，跳过", cls)
                    continue
                positions.sort(key=lambda p: p[1], reverse=True)
                rospy.loginfo("检测到 %d 个 %s: %s",
                              len(positions), cls,
                              ["(%.3f,%.3f,%.3f)" % p for p in positions])

            # ═══ Phase 3: 按类别逐个抓取放置 ═══
            first_pick_done = False
            for cls in TARGET_CLASSES:
                positions = class_positions.get(cls, [])
                if not positions:
                    continue
                pick_fn = PICK_FUNCTIONS[cls]
                for pick_i, (bx, by, bz) in enumerate(positions):
                    rospy.loginfo("[%s] 抓取 %d/%d 初始坐标=(%.3f,%.3f,%.3f)",
                                  cls, pick_i + 1, len(positions), bx, by, bz)

                    if not first_pick_done and pick_i == 0:
                        # 首次抓取：收敛采集精确定位
                        first_pick_done = True
                        self._sleep_hold(1.0)
                        rospy.loginfo("收敛采集: 等待连续2帧稳定...")
                        prev_x = prev_y = prev_z = None
                        prev_count = 0
                        max_frames = 30
                        for frame_i in range(max_frames):
                            self._hold_arms()
                            rospy.sleep(0.4)
                            self._perception.republish_viz()
                            objs = self._perception.get_objects_3d_yolo()
                            class_objs = [o for o in (objs or [])
                                         if o.get("class_name") == cls
                                         and "position_base" in o
                                         and o["confidence"] >= 0.5]
                            total_objs = len([o for o in (objs or [])
                                              if "position_base" in o and o["confidence"] >= 0.5])
                            if not class_objs:
                                continue
                            class_objs.sort(key=lambda o: o["position_base"][1], reverse=True)
                            cur = class_objs[0]["position_base"]
                            cur_x, cur_y, cur_z = cur[0], cur[1], cur[2]
                            dist = 999 if prev_x is None else math.hypot(cur_x - prev_x, cur_y - prev_y)

                            rospy.loginfo("[收敛 %d/%d] 总数=%d %s=(%.3f,%.3f,%.3f) "
                                          "Δpos=%.3f prev_cnt=%d",
                                          frame_i + 1, max_frames,
                                          total_objs, cls, cur_x, cur_y, cur_z,
                                          dist, prev_count)

                            if (prev_x is not None and total_objs == 6 and prev_count == 6
                                    and dist < 0.03):
                                bx = (cur_x + prev_x) / 2.0
                                by = (cur_y + prev_y) / 2.0
                                bz = (cur_z + prev_z) / 2.0
                                rospy.loginfo("→ 收敛! 2帧平均: (%.3f, %.3f, %.3f)", bx, by, bz)
                                positions[pick_i] = (bx, by, bz)
                                break

                            prev_x, prev_y, prev_z = cur_x, cur_y, cur_z
                            prev_count = total_objs
                        else:
                            if prev_x is not None:
                                bx, by, bz = prev_x, prev_y, prev_z
                                rospy.logwarn("未收敛, 用最后帧: (%.3f, %.3f, %.3f)", bx, by, bz)
                                positions[pick_i] = (bx, by, bz)
                            else:
                                rospy.logerr("无有效 %s 坐标", cls)
                                continue

                    rospy.loginfo("[%s] 抓取 %d/%d 坐标=(%.3f,%.3f,%.3f)",
                                  cls, pick_i + 1, len(positions), bx, by, bz)

                    if pick_fn(bx, by, bz):
                        rospy.loginfo("[%s] %d/%d 完成", cls,
                                      pick_i + 1, len(positions))
                    rospy.sleep(0.5)

        # ═══ 任务完成, 进入可视化保持 ═══
        rospy.loginfo("[完成] 任务完成, 进入可视化保持, Ctrl-C退出")
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            self._perception.republish_viz()
            rate.sleep()

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
