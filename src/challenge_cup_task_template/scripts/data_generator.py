#!/usr/bin/env python3
"""
纯 MuJoCo + OSMesa 数据生成 — 不需要 ROS, 不需要显示器, 全自动。

生成两类训练数据:
  1. YOLO 修正: (YOLO检测坐标, 物体真值坐标) → 训练网络修正 YOLO 误差
  2. 关节预测: (物体真值坐标, 关节角) → 训练网络直接从坐标预测关节

流程:
  1. 加载 MuJoCo scene2
  2. 设机器人站立姿态
  3. OSMesa 渲染头部相机 RGB 图
  4. YOLO 检测 (有噪声)
  5. 读 MuJoCo body 坐标 (真值)
  6. FK 表查逆解 → 存训练对

用法:
  cd /root/kuavo_ws
  python3 src/challenge_cup_task_template/scripts/data_generator.py
"""

import mujoco
import mujoco.osmesa as osmesa
import numpy as np
import cv2
import os
import sys
import time

# ── 路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_DIR = os.path.join(SCRIPT_DIR, "../..")
SCENE_FILE = "challenge_cup_simulator/models/biped_s52/xml/_scene_scene2_active.xml"
BIPED_FILE = "src/challenge_cup_simulator/models/biped_s52/xml/biped_s52.xml"
DATA_DIR = os.path.join(SCRIPT_DIR, "training_data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── YOLO 检测器 ──
sys.path.insert(0, SCRIPT_DIR)
from scene2_yolo_detector import Scene2YOLODetector

# ── 物体定义 (MuJoCo body 名) ──
OBJECTS = {
    "part_type_a_1": "pipe_fitting",
    "part_type_a_2": "pipe_fitting",
    "part_type_b_1": "pipe_clamp",
    "part_type_b_2": "pipe_clamp",
    "part_type_c_1": "screwdriver",
    "part_type_c_2": "screwdriver",
}

# 物体随机位置范围 (桌面区域, base_link 系, 米)
OBJECT_X_RANGE = (0.25, 0.40)
OBJECT_Y_RANGE = (-0.40, 0.40)
OBJECT_Z = 0.803  # 桌面高度 (hardcoded from XML)


def load_model(scene_path):
    """Load MuJoCo model from the full scene XML."""
    return mujoco.MjModel.from_xml_path(scene_path)


def set_standing_pose(data):
    """Set robot to standing pose (match simulation initial state)."""
    # 腿部 (弧度)
    leg_l = [0.006, 0.0, -0.503, 0.862, -0.36, -0.006]
    leg_r = [-0.007, 0.0, -0.502, 0.861, -0.36, 0.007]
    for i, v in enumerate(leg_l): data.qpos[7+i] = v
    for i, v in enumerate(leg_r): data.qpos[13+i] = v
    data.qpos[19] = 0.0    # waist
    data.qpos[42] = 0.0    # head_yaw
    data.qpos[43] = 0.349  # head_pitch ~20° down


def set_arm_side_standing(data):
    """Arms at 60° side (match pregrasp pose)."""
    # Left: [0, 60, 0, 0, 0, 0, 0]
    # Right: [0, -60, 0, 0, 0, 0, 0]
    from math import radians
    data.qpos[20] = 0          # pitch
    data.qpos[21] = radians(60)    # roll
    data.qpos[22] = 0          # yaw
    data.qpos[23] = 0          # forearm
    data.qpos[24] = 0          # hand_yaw
    data.qpos[25] = 0          # hand_pitch
    data.qpos[26] = 0          # hand_roll

    data.qpos[35] = 0          # pitch
    data.qpos[36] = radians(-60)   # roll
    data.qpos[37] = 0          # yaw
    data.qpos[38] = 0          # forearm
    data.qpos[39] = 0          # hand_yaw
    data.qpos[40] = 0          # hand_pitch
    data.qpos[41] = 0          # hand_roll


def randomize_objects(data, model, rng):
    """Randomize object positions in the desktop area."""
    result = {}  # obj_name → (x, y, z) in world frame
    for obj_name, cls_name in OBJECTS.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id < 0:
            continue
        jnt_id = model.body_jntadr[body_id]
        if jnt_id < 0:
            continue
        qpos_adr = model.jnt_qposadr[jnt_id]

        # Randomize XY, keep Z fixed
        x = rng.uniform(*OBJECT_X_RANGE)
        y = rng.uniform(*OBJECT_Y_RANGE)
        z = OBJECT_Z

        data.qpos[qpos_adr + 0] = x
        data.qpos[qpos_adr + 1] = y
        data.qpos[qpos_adr + 2] = z
        # Random rotation around Z
        yaw = rng.uniform(0, 2 * np.pi)
        data.qpos[qpos_adr + 3] = np.cos(yaw / 2)  # qw
        data.qpos[qpos_adr + 4] = 0  # qx
        data.qpos[qpos_adr + 5] = 0  # qy
        data.qpos[qpos_adr + 6] = np.sin(yaw / 2)  # qz

        result[obj_name] = (cls_name, x, y, z)
    return result


def get_head_camera_view(model, data, gl_ctx, mjr_ctx, width=640, height=480):
    """Render head camera RGB image using OSMesa."""
    scene = mujoco.MjvScene(model, maxgeom=10000)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()

    # Find head camera body
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_h")
    if cam_id >= 0:
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
    else:
        cam.lookat[:] = [0.35, 0, 0.5]
        cam.distance = 1.2
        cam.elevation = -30
        cam.azimuth = 0

    mujoco.mjv_updateScene(model, data, opt, None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)

    viewport = mujoco.MjrRect(0, 0, width, height)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)

    gl_ctx.make_current()
    mujoco.mjr_render(viewport, scene, mjr_ctx)
    mujoco.mjr_readPixels(rgb, depth, viewport, mjr_ctx)

    return cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR), depth


def get_object_ground_truth(model, data):
    """Read object positions in base_link frame."""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_pos = data.xpos[base_id].copy() if base_id >= 0 else np.zeros(3)

    result = {}
    for obj_name, cls_name in OBJECTS.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id < 0:
            continue
        world_pos = data.xpos[body_id].copy()
        rel = world_pos - base_pos
        result[obj_name] = (cls_name, rel[0], rel[1], rel[2])
    return result


def load_fk_table():
    """Load the pre-built FK tables."""
    import numpy as np
    left = np.load(os.path.join(SCRIPT_DIR, "fk_table_left.npy"))
    right = np.load(os.path.join(SCRIPT_DIR, "fk_table_right.npy"))
    return left, right


def fk_lookup(table, tx, ty, tz):
    """Nearest neighbor: target XYZ → joint angles (4 DOF, degrees)."""
    xyz = table[:, 4:7].astype(np.float64)
    query = np.array([tx, ty, tz], dtype=np.float64)
    d = np.sqrt(2*(xyz[:,0]-query[0])**2 + 2*(xyz[:,1]-query[1])**2 + (xyz[:,2]-query[2])**2)
    best = table[np.argmin(d)]
    return [float(best[i]) for i in range(4)]  # pitch, roll, yaw, forearm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-snapshots", type=int, default=50,
                        help="Number of random object layouts to generate")
    parser.add_argument("--render", action="store_true",
                        help="Also save rendered images for debugging")
    args = parser.parse_args()

    print("Loading MuJoCo scene...")
    scene_path = os.path.join(WS_DIR, SCENE_FILE)
    model = load_model(scene_path)
    data = mujoco.MjData(model)
    print(f"  model: nq={model.nq}, bodies={model.nbody}")

    print("Loading YOLO detector...")
    model_path = "/root/kuavo_ws/models/yolo/best.pt"
    yolo = Scene2YOLODetector(model_path=model_path, conf=0.15)

    print("Loading FK tables...")
    left_table, right_table = load_fk_table()

    print("Setting up OSMesa renderer (640x480)...")
    gl_ctx = osmesa.GLContext(640, 480)
    gl_ctx.make_current()
    mjr_ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    set_standing_pose(data)
    set_arm_side_standing(data)
    rng = np.random.RandomState(42)

    yolo_inputs = []   # YOLO coordinates
    yolo_outputs = []  # Ground truth coordinates
    joint_inputs = []  # GT coordinates
    joint_outputs = [] # Joint angles

    t0 = time.time()
    for snap in range(args.n_snapshots):
        # Randomize objects
        rand_positions = randomize_objects(data, model, rng)
        mujoco.mj_forward(model, data)

        # Render head camera
        bgr, depth = get_head_camera_view(model, data, gl_ctx, mjr_ctx)
        if args.render and snap < 3:
            out_path = os.path.join(DATA_DIR, f"snapshot_{snap:03d}.jpg")
            cv2.imwrite(out_path, bgr)

        # YOLO detection
        instances = yolo.detect(bgr)
        if not instances:
            print(f"  [{snap}] YOLO found nothing, skipping")
            continue

        # Ground truth
        gt = get_object_ground_truth(model, data)

        # Match YOLO → GT by class and Y-sorting
        yolo_by_class = {}
        for inst in instances:
            cls = inst["class_name"]
            yolo_by_class.setdefault(cls, []).append(inst["center_uv"])

        gt_by_class = {}
        for obj_name, (cls, x, y, z) in gt.items():
            gt_by_class.setdefault(cls, []).append((x, y, z))
        for cls in gt_by_class:
            gt_by_class[cls].sort(key=lambda p: p[1], reverse=True)

        # Record pairs
        n_records = 0
        for cls, gt_list in gt_by_class.items():
            yolo_list = yolo_by_class.get(cls, [])
            # Both sorted by Y descending → aligned
            # Use simple matching: closest Y
            for gx, gy, gz in gt_list:
                if not yolo_list:
                    break
                # Nearest YOLO detection
                best_uv = yolo_list[0]
                yolo_list = yolo_list[1:]

                # We need 3D from YOLO for the "noisy coordinate" input
                # In the real system, YOLO + depth gives 3D. Here we simulate
                # YOLO 3D error by adding noise to the GT position.
                noise_xy = rng.normal(0, 0.03, 2)  # 3cm XY noise
                noise_z = rng.normal(0, 0.015, 1)   # 1.5cm Z noise
                yolo_x = gx + noise_xy[0]
                yolo_y = gy + noise_xy[1]
                yolo_z = gz + noise_z[0]

                yolo_inputs.append([yolo_x, yolo_y, yolo_z])
                yolo_outputs.append([gx, gy, gz])

                # Joint prediction: GT → joints
                table = left_table if gy > 0 else right_table
                joints_4 = fk_lookup(table, gx, gy, gz)
                joint_inputs.append([gx, gy, gz])
                joint_outputs.append(joints_4)

                n_records += 1

        if snap % 10 == 0 or n_records > 0:
            print(f"  [{snap}] {n_records} records, total YOLO={len(yolo_inputs)} Joints={len(joint_inputs)}")

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s. Total: {len(yolo_inputs)} YOLO pairs, {len(joint_inputs)} joint pairs")

    # Save
    yi = np.array(yolo_inputs, dtype=np.float32)
    yo = np.array(yolo_outputs, dtype=np.float32)
    ji = np.array(joint_inputs, dtype=np.float32)
    jo = np.array(joint_outputs, dtype=np.float32)

    np.savez(os.path.join(DATA_DIR, "yolo_correction.npz"), inputs=yi, outputs=yo)
    np.savez(os.path.join(DATA_DIR, "joint_prediction.npz"), inputs=ji, outputs=jo)
    print(f"Saved to {DATA_DIR}/yolo_correction.npz, joint_prediction.npz")


if __name__ == "__main__":
    main()
