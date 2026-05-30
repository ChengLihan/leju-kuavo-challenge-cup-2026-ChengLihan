#!/usr/bin/env python3
"""
LiDAR mid360 仿真节点：订阅 /mujoco/qpos，与当前仿真状态同步后做射线追踪，发布点云。
需先启动 MuJoCo 仿真（如 load_kuavo_mujoco_challenge.launch），并确保场景 XML 中含 lidar_site。

设计要点：
  1. 点云变换到 base_link 坐标系后发布，使 FAST_LIO 外参始终为 identity，
     彻底规避 waist_yaw_joint / zhead_1_joint 运动导致固定外参失效的漂移问题。
  2. 订阅 /sensors_data_raw，发布未低通滤波 IMU 到 /lidar_imu（stamp=sensor_time），
     避免 /imu_data 低通滤波导致的相位滞后。
  3. 点云时间戳优先对齐 sensor_time（与 /lidar_imu 同轴），
     缺失时回退到 rospy.Time.now()。
  4. 支持两种输出：
     - PointCloud2（/lidar/points，供 FAST_LIO lidar_type=2 使用）
     - Livox CustomMsg（/livox/lidar，供 FAST_LIO lidar_type=1 使用）
     通过参数 ~output_type 在 pointcloud2 / livox / both 间切换。
"""
from __future__ import division

import sys
import os
import traceback

_script_dir = os.path.dirname(os.path.abspath(__file__))
_ws_lib = os.path.dirname(_script_dir)
_devel = os.path.dirname(_ws_lib)
_ws_root = os.path.dirname(_devel)


def _prepend_if_dir(path, added):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
        added.append(path)


_msg_paths_added = []

# 当前工作空间（优先）
_prepend_if_dir(os.path.join(_ws_root, "devel", "lib", "python3", "dist-packages"), _msg_paths_added)
_prepend_if_dir(
    os.path.join(_ws_root, "devel", ".private", "livox_ros_driver2", "lib", "python3", "dist-packages"),
    _msg_paths_added,
)
_prepend_if_dir(
    os.path.join(_ws_root, "devel", ".private", "kuavo_msgs", "lib", "python3", "dist-packages"),
    _msg_paths_added,
)

# 可选：通过环境变量追加消息路径（冒号分隔），避免在代码里硬编码机器路径
# 示例：
#   export LIVOX_MSG_PY_PATHS=/path/to/ws1/devel/lib/python3/dist-packages:/path/to/ws2/devel/.private/livox_ros_driver2/lib/python3/dist-packages
for _p in os.environ.get("LIVOX_MSG_PY_PATHS", "").split(":"):
    _p = _p.strip()
    if _p:
        _prepend_if_dir(_p, _msg_paths_added)

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu, PointCloud2, PointField

_LIVOX_MSG_OK = True
try:
    from livox_ros_driver2.msg import CustomMsg as LivoxCustomMsg
    from livox_ros_driver2.msg import CustomPoint as LivoxCustomPoint
except ImportError:
    _LIVOX_MSG_OK = False
    LivoxCustomMsg = None
    LivoxCustomPoint = None

try:
    from kuavo_msgs.msg import sensorsData as KuavoSensorsData
except ImportError:
    print(
        "lidar_mid360: 缺少 kuavo_msgs Python 消息定义。"
        "发布 /lidar_imu 需要 kuavo_msgs/sensorsData。"
        "\n已尝试追加的消息路径:\n  - " + "\n  - ".join(_msg_paths_added if _msg_paths_added else ["<none>"]),
        file=sys.stderr,
    )
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

try:
    import mujoco
    from mujoco_lidar import scan_gen
    try:
        from mujoco_lidar import MjLidarWrapper
    except ImportError:
        MjLidarWrapper = None
except ImportError:
    print(
        "lidar_mid360: 缺少 mujoco/mujoco_lidar。"
        "请先安装 Python 依赖，例如："
        "pip3 install --no-deps -r src/challenge_cup_simulator/requirements.txt",
        file=sys.stderr,
    )
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


class _CpuMjLidarWrapper(object):
    """CPU LiDAR wrapper compatible with mujoco_lidar.MjLidarWrapper.

    PyPI mujoco-lidar 0.2.x keeps scan_gen, but its wheel/sdist can miss
    mujoco_lidar.core_cpu. The CPU path is small and uses MuJoCo mj_multiRay
    directly, matching CRAIC's backend behavior while keeping the dependency
    installed from pip.
    """

    def __init__(self, mj_model, site_name, cutoff_dist=100.0, args=None):
        args = args or {}
        self.mj_model = mj_model
        self.site_name = site_name
        self.cutoff_dist = cutoff_dist
        self.geomgroup = args.get("geomgroup", None)
        self.bodyexclude = args.get("bodyexclude", -1)
        self._sensor_pose = np.eye(4, dtype=np.float32)
        self._hit_points = None
        self._dist = None

    @property
    def sensor_position(self):
        return self._sensor_pose[:3, 3].copy()

    @property
    def sensor_rotation(self):
        return self._sensor_pose[:3, :3].copy()

    def _update_sensor_pose(self, mj_data):
        site = mj_data.site(self.site_name)
        self._sensor_pose[:3, :3] = site.xmat.reshape(3, 3)
        self._sensor_pose[:3, 3] = site.xpos

    def trace_rays(self, mj_data, ray_theta, ray_phi):
        if ray_phi.shape[0] != ray_theta.shape[0]:
            raise ValueError("ray_phi and ray_theta must have the same shape")

        self._update_sensor_pose(mj_data)
        nray = ray_phi.shape[0]
        self._dist = np.full(nray, self.cutoff_dist, dtype=np.float64)
        geomid = np.full(nray, 0, dtype=np.int32)

        site_pos = self._sensor_pose[:3, 3]
        site_mat = self._sensor_pose[:3, :3]
        pnt = np.array([site_pos], dtype=np.float64).T

        x = np.cos(ray_phi) * np.cos(ray_theta)
        y = np.cos(ray_phi) * np.sin(ray_theta)
        z = np.sin(ray_phi)
        local_vecs = np.stack((x, y, z), axis=-1).astype(np.float64, copy=False)
        world_vecs = local_vecs @ site_mat.T
        world_vecs /= np.linalg.norm(world_vecs, axis=1, keepdims=True)

        mujoco.mj_multiRay(
            m=self.mj_model,
            d=mj_data,
            pnt=pnt,
            vec=np.ascontiguousarray(world_vecs).ravel(),
            geomgroup=self.geomgroup,
            flg_static=1,
            bodyexclude=self.bodyexclude,
            geomid=geomid,
            dist=self._dist,
            nray=nray,
            cutoff=self.cutoff_dist,
        )

        self._dist[geomid == -1] = 0.0
        self._hit_points = (local_vecs * self._dist[:, np.newaxis]).astype(np.float32, copy=False)
        return self._dist

    def get_hit_points(self):
        return self._hit_points


def _publish_livox(pub, points, frame_id, stamp):
    """发布 livox_ros_driver2/CustomMsg。

    时间策略：
    - 所有点 offset_time=0（瞬时快照，不触发帧内去畸变）
    - lidar_mean_scantime 收敛为 0，FAST_LIO 视为无扫描窗口，
      EKF IMU 积分区间 = 相邻帧间隔（约100ms），正确。

    为什么不用"尾部非零"技巧：
    - tail hack 使 lidar_mean_scantime=100ms → lidar_end_time = beg+100ms。
    - FAST_LIO 对所有 offset_time=0 的点做 100ms 的 IMU 去畸变，
      但快照中所有点都在同一瞬间采集，去畸变完全错误。
    - 机器人以 10°/s 转动时每帧引入 ~1° 误差，累积漂移明显。
    """
    if not _LIVOX_MSG_OK:
        return

    n = points.shape[0]
    pts = points.astype(np.float32)
    msg = LivoxCustomMsg()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.timebase        = stamp.to_nsec()
    msg.point_num       = n
    msg.lidar_id        = 1
    msg.points = [
        LivoxCustomPoint(offset_time=0,
                         x=float(pts[i, 0]), y=float(pts[i, 1]),
                         z=float(pts[i, 2]), reflectivity=100, tag=0, line=0)
        for i in range(n)
    ]
    pub.publish(msg)


def _publish_pointcloud2(pub, points, frame_id, stamp, time_mode="zero", scan_duration_us=100000.0):
    """发布 sensor_msgs/PointCloud2，包含 FAST_LIO type=2 需要的 time/ring 字段。"""
    n = points.shape[0]
    if n == 0:
        return

    pts = points.astype(np.float32, copy=False)
    dtype = np.dtype([
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
        ("time", np.float32),   # 单位由 FAST_LIO 的 timestamp_unit 解释
        ("ring", np.uint16),
        ("_pad", np.uint16),    # 补齐到 24 bytes/point
    ])
    buf = np.zeros(n, dtype=dtype)
    buf["x"] = pts[:, 0]
    buf["y"] = pts[:, 1]
    buf["z"] = pts[:, 2]
    buf["intensity"] = 100.0
    buf["ring"] = 0

    tm = (time_mode or "zero").lower()
    if tm == "linear" and n > 1:
        buf["time"] = np.linspace(0.0, float(scan_duration_us), n, endpoint=False, dtype=np.float32)
    elif tm == "last_nonzero":
        buf["time"] = 0.0
        buf["time"][-1] = np.float32(1.0)  # 1 us，几乎不引入额外去畸变
    else:
        buf["time"] = 0.0

    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="time", offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="ring", offset=20, datatype=PointField.UINT16, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * n
    msg.is_dense = True
    msg.data = buf.tobytes()
    pub.publish(msg)


def _build_raw_imu_msg(sensor_msg):
    """由 kuavo_msgs/sensorsData 构造未滤波 IMU 消息。"""
    imu_out = Imu()
    imu_out.header.stamp = sensor_msg.sensor_time
    # 与现有 /imu_data 保持同一 frame_id，减少后端配置变更
    imu_out.header.frame_id = "dummy_link"
    imu_out.orientation = sensor_msg.imu_data.quat
    imu_out.angular_velocity = sensor_msg.imu_data.gyro
    imu_out.linear_acceleration = sensor_msg.imu_data.acc
    return imu_out


def main():
    rospy.init_node("lidar_mid360_node", anonymous=False)

    scene_path = rospy.get_param(
        "legged_robot_scene_param",
        rospy.get_param("scene_path", None),
    )
    if not scene_path:
        pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scene_path = os.path.normpath(os.path.join(
            pkg_path, "models", "biped_s52", "xml", "scene1.xml"
        ))
        if not os.path.isfile(scene_path):
            rospy.logerr("lidar_mid360: 未找到场景 XML，请设置 legged_robot_scene_param。")
            sys.exit(1)

    rospy.loginfo("lidar_mid360: 加载场景 %s", scene_path)
    try:
        mj_model = mujoco.MjModel.from_xml_path(scene_path)
        mj_data  = mujoco.MjData(mj_model)
    except Exception as e:
        rospy.logerr("lidar_mid360: 加载场景失败: %s", e)
        sys.exit(1)

    site_name = "lidar_site"
    try:
        _ = mj_data.site(site_name)
    except Exception:
        rospy.logerr("lidar_mid360: 场景中未找到 site '%s'", site_name)
        sys.exit(1)

    backend     = rospy.get_param("~backend", "cpu").strip().lower()
    cutoff_dist = rospy.get_param("~cutoff_dist", 50.0)

    use_geomgroup = rospy.get_param("~use_geomgroup", True)
    if use_geomgroup:
        geomgroup  = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)
        lidar_args = {"geomgroup": geomgroup}
        rospy.loginfo("lidar_mid360: geomgroup 过滤，检测 group=0/2/3")
    else:
        try:
            base_body_id = mj_model.body("base_link").id
        except Exception:
            base_body_id = -1
        lidar_args = {"bodyexclude": base_body_id}
        rospy.loginfo("lidar_mid360: bodyexclude=%d", base_body_id)

    if backend == "cpu":
        lidar = _CpuMjLidarWrapper(
            mj_model,
            site_name=site_name,
            cutoff_dist=cutoff_dist,
            args=lidar_args,
        )
    else:
        if MjLidarWrapper is None:
            rospy.logerr("lidar_mid360: 当前环境缺少 mujoco_lidar.MjLidarWrapper，无法使用 backend=%s", backend)
            sys.exit(1)
        lidar = MjLidarWrapper(
            mj_model,
            site_name=site_name,
            backend=backend,
            cutoff_dist=cutoff_dist,
            args=lidar_args,
        )
    livox = scan_gen.LivoxGenerator("mid360")

    output_type = rospy.get_param("~output_type", "pointcloud2").strip().lower()
    if output_type not in ("pointcloud2", "livox", "both"):
        rospy.logwarn("lidar_mid360: 未知 output_type=%s，回退到 pointcloud2", output_type)
        output_type = "pointcloud2"

    livox_topic = rospy.get_param("~livox_topic", "/livox/lidar")
    pc2_topic = rospy.get_param("~pc2_topic", "/lidar/points")
    pc2_time_mode = rospy.get_param("~pc2_time_mode", "last_nonzero").strip().lower()

    livox_pub = None
    pc2_pub = None
    if output_type in ("livox", "both"):
        if not _LIVOX_MSG_OK:
            rospy.logerr(
                "lidar_mid360: output_type=%s 需要 livox_ros_driver2 Python 消息定义，但当前环境不可用。"
                "请安装并 source 对应工作空间，或将 output_type 改为 pointcloud2。",
                output_type,
            )
            sys.exit(1)
        livox_pub = rospy.Publisher(livox_topic, LivoxCustomMsg, queue_size=1)
    if output_type in ("pointcloud2", "both"):
        pc2_pub = rospy.Publisher(pc2_topic, PointCloud2, queue_size=1)

    raw_imu_topic = "/lidar_imu"
    raw_imu_pub = rospy.Publisher(raw_imu_topic, Imu, queue_size=100)

    # 点云发布在 base_link 坐标系，与 /imu_data 的 frame 一致，外参 = identity
    lidar_frame = rospy.get_param("~lidar_frame", "base_link")
    rate_hz     = rospy.get_param("~rate", 10.0)
    nq          = mj_model.nq

    scan_duration_us = 1e6 / max(rate_hz, 1e-6)
    rospy.loginfo(
        "lidar_mid360: mid360，site=%s，%g Hz，output=%s，pc2_topic=%s，livox_topic=%s，frame=%s，raw_imu_topic=%s，pc2_time_mode=%s",
        site_name, rate_hz, output_type, pc2_topic, livox_topic, lidar_frame, raw_imu_topic, pc2_time_mode
    )

    # _last_qpos_holder: (stamp, qpos_array)
    # stamp 优先取 /sensors_data_raw.sensor_time（与 /lidar_imu 同轴）；
    # 无可用数据时回退 now()。
    _last_qpos_holder = [None]
    _latest_sensor_stamp_holder = [None]

    def on_sensor(msg):
        _latest_sensor_stamp_holder[0] = msg.sensor_time
        raw_imu_pub.publish(_build_raw_imu_msg(msg))

    def on_qpos(msg):
        if len(msg.data) >= nq:
            t = _latest_sensor_stamp_holder[0]
            if t is None:
                t = rospy.Time.now()
            _last_qpos_holder[0] = (t, np.array(msg.data[:nq], dtype=np.float64))

    rospy.Subscriber("/sensors_data_raw", KuavoSensorsData, on_sensor, queue_size=100)
    rospy.Subscriber("/mujoco/qpos", Float64MultiArray, on_qpos, queue_size=1)

    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        entry = _last_qpos_holder[0]
        if entry is not None:
            qpos_stamp, qpos = entry
            mj_data.qpos[:] = qpos
            mujoco.mj_forward(mj_model, mj_data)

            rays_theta, rays_phi = livox.sample_ray_angles()
            rays_theta = np.ascontiguousarray(rays_theta.astype(np.float32))
            rays_phi   = np.ascontiguousarray(rays_phi.astype(np.float32))
            lidar.trace_rays(mj_data, rays_theta, rays_phi)
            points_local = lidar.get_hit_points()   # (N,3) lidar_site 局部系

            if points_local is not None and points_local.size > 0:
                # 将点云从 lidar_site 局部系变换到 base_link 系：
                #   local → world → base_link
                # 原因：雷达经 waist_yaw_joint + zhead_1_joint 连接到 base_link，
                # 这两关节会动，固定外参会随关节角漂移，所以在这里用实时运动学消除外参误差。
                R_sensor   = lidar.sensor_rotation                          # world←lidar_site (3,3)
                pos_sensor = lidar.sensor_position                          # lidar_site 世界坐标 (3,)
                base_xpos  = np.array(mj_data.body('base_link').xpos)      # base_link 世界坐标 (3,)
                base_xmat  = np.array(mj_data.body('base_link').xmat).reshape(3, 3)  # world←base_link
                points_world = points_local @ R_sensor.T + pos_sensor
                points_base  = (points_world - base_xpos) @ base_xmat

                if pc2_pub is not None:
                    _publish_pointcloud2(
                        pc2_pub, points_base, lidar_frame, qpos_stamp,
                        time_mode=pc2_time_mode, scan_duration_us=scan_duration_us
                    )
                if livox_pub is not None:
                    _publish_livox(livox_pub, points_base, lidar_frame, qpos_stamp)

        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception:
        traceback.print_exc()
        sys.exit(1)
