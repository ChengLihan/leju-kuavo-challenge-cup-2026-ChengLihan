#!/usr/bin/env python3
"""
Scene2 YOLOv8-Seg 检测模块 (第五步 + 第六步)
=============================================
可脱离 ROS 独立运行，也可被 Perception 类调用。

功能:
  1. 加载 YOLOv8-seg 模型进行分割推理
  2. 提取每个实例: class_id, class_name, confidence, bbox_xyxy, center_uv, mask
  3. 结合深度图: mask → 稳定深度中值 → 相机坐标系 3D 点
  4. 可视化: mask叠加 + bbox + 中心点

独立测试:
  python scene2_yolo_detector.py --image test.jpg --depth depth.png
"""

import argparse
import json
import os
import sys
from typing import Optional

import cv2
import numpy as np
# 注意: torch/ultralytics 需要调用方在 import 本模块前
# 将 yolo_gpu site-packages 临时注入 sys.path。
# 本模块本身不做延迟导入，确保模块加载时依赖全部就绪。
import torch
from ultralytics import YOLO

# ── 场景二类别映射 ──────────────────────────────────
CLASS_NAMES = {
    0: "pipe_clamp",
    1: "pipe_fitting",
    2: "screwdriver",
}

TARGET_BIN = {
    "pipe_clamp":  "blue_bin",
    "pipe_fitting": "orange_bin",
    "screwdriver": "purple_bin",
}

BIN_COLORS_BGR = {
    "blue_bin":   (255, 0, 0),
    "orange_bin": (0, 165, 255),
    "purple_bin": (255, 0, 255),
}


class Scene2YOLODetector:
    """Scene2 YOLO分割检测器 — 独立于ROS，纯numpy/cv2输入输出"""

    def __init__(self, model_path: str = "models/yolo/yolov8n_seg_scene2_demo.pt",
                 conf: float = 0.15, imgsz: int = 640, device: int = 0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.class_names = self.model.names  # {0: "pipe_clamp", ...}

    # ── YOLO 推理 ──────────────────────────────────

    def detect(self, bgr_image: np.ndarray) -> list:
        """对单张BGR图像推理，返回实例列表。"""
        results = self.model(bgr_image, imgsz=self.imgsz,
                             conf=self.conf, device=self.device,
                             retina_masks=True, verbose=False)
        result = results[0]

        if result.masks is None:
            return []

        img_h, img_w = result.orig_shape[:2]
        instances = []

        for i in range(len(result.boxes)):
            conf_val = float(result.boxes.conf[i])
            if conf_val < self.conf:
                continue

            cls_id = int(result.boxes.cls[i])
            cls_name = self.class_names.get(cls_id, f"unknown_{cls_id}")

            # bbox
            bbox = result.boxes.xyxy[i].cpu().numpy().tolist()

            # mask (二值, 原图尺寸)
            mask_raw = result.masks.data[i].cpu().numpy()
            mask = (mask_raw > 0.5).astype(np.uint8)
            if mask.shape != (img_h, img_w):
                mask = cv2.resize(mask, (img_w, img_h),
                                  interpolation=cv2.INTER_NEAREST)

            # mask 中心点 (像素均值)
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                u, v = float(np.mean(xs)), float(np.mean(ys))
            else:
                u, v = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

            instances.append({
                "class_id":   cls_id,
                "class_name": cls_name,
                "confidence": round(conf_val, 4),
                "bbox_xyxy":  [round(x, 1) for x in bbox],
                "center_uv":  [round(u, 1), round(v, 1)],
                "mask":       mask,
                "mask_area":  int(np.sum(mask > 0)),
                "target_bin": TARGET_BIN.get(cls_name, "unknown_bin"),
            })

        return instances

    # ── 深度 → 3D ───────────────────────────────────

    @staticmethod
    def get_mask_median_depth(depth_img: np.ndarray,
                              mask: np.ndarray) -> Optional[float]:
        """从mask区域提取稳定深度值（中值 + 10%/90%分位离群剔除）。"""
        valid = depth_img[mask > 0]
        valid = valid[np.isfinite(valid)]
        valid = valid[valid > 0]
        if len(valid) < 30:
            return None
        low = np.percentile(valid, 10)
        high = np.percentile(valid, 90)
        valid = valid[(valid >= low) & (valid <= high)]
        return float(np.median(valid))

    def detect_with_depth(self, bgr_image: np.ndarray,
                          depth_image: np.ndarray,
                          camera_matrix: Optional[np.ndarray] = None
                          ) -> list:
        """YOLO检测 + 深度 → 相机坐标系3D点。

        camera_matrix: 3x3 内参矩阵 [[fx,0,cx],[0,fy,cy],[0,0,1]]
        若为 None 则只返回像素坐标。
        """
        instances = self.detect(bgr_image)

        for inst in instances:
            mask = inst["mask"]
            Z = self.get_mask_median_depth(depth_image, mask)
            inst["depth_m"] = round(Z, 4) if Z is not None else None

            if Z is not None and camera_matrix is not None:
                u, v = inst["center_uv"]
                fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
                cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                inst["position_camera"] = [round(X, 4), round(Y, 4), round(Z, 4)]
            elif Z is not None:
                inst["position_camera"] = None  # 无内参，无法反投影

        return instances

    # ── 可视化 ──────────────────────────────────────

    def draw_results(self, bgr_image: np.ndarray,
                     instances: list) -> np.ndarray:
        """在图像上绘制mask、bbox、中心点和标签。"""
        vis = bgr_image.copy()
        alpha = 0.35

        for inst in instances:
            color = BIN_COLORS_BGR.get(inst.get("target_bin", ""), (0, 255, 0))
            label = f"{inst['class_name']} {inst['confidence']:.2f}"

            mask = inst["mask"]
            overlay = np.zeros_like(vis)
            overlay[mask > 0] = color
            vis = cv2.addWeighted(vis, 1.0, overlay, alpha, 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)

            bbox = [int(x) for x in inst["bbox_xyxy"]]
            cv2.rectangle(vis, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), color, 2)

            cu, cv_val = int(inst["center_uv"][0]), int(inst["center_uv"][1])
            cv2.circle(vis, (cu, cv_val), 6, (0, 0, 255), -1)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(bbox[1] - 8, th + 4)
            cv2.rectangle(vis, (bbox[0], ly - th - 4),
                          (bbox[0] + tw + 4, ly), color, -1)
            cv2.putText(vis, label, (bbox[0] + 2, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis


# ══════════════════════════════════════════════════════
# 独立运行入口（用于脱离ROS测试）
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Scene2 YOLO分割检测")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--depth", default=None, help="深度图路径 (可选)")
    parser.add_argument("--model", default="models/yolo/yolov8n_seg_scene2_demo.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--output", default="outputs/scene2_yolo_debug",
                        help="输出目录")
    args = parser.parse_args()

    detector = Scene2YOLODetector(model_path=args.model, conf=args.conf)
    print(f"[INFO] 类别: {detector.class_names}")

    bgr = cv2.imread(args.image)
    if bgr is None:
        sys.exit(f"无法读取: {args.image}")

    if args.depth:
        depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
        if depth is None:
            sys.exit(f"无法读取深度图: {args.depth}")
        # 使用默认内参（仿真头部相机）
        K = np.array([[554.25, 0, 320], [0, 554.25, 240], [0, 0, 1]],
                     dtype=np.float64)
        instances = detector.detect_with_depth(bgr, depth, K)
    else:
        instances = detector.detect(bgr)

    print(f"\n检测到 {len(instances)} 个目标:")
    for inst in instances:
        extra = ""
        if inst.get("position_camera"):
            x, y, z = inst["position_camera"]
            extra = f"  3D_cam=({x:.3f}, {y:.3f}, {z:.3f})"
        elif inst.get("depth_m") is not None:
            extra = f"  depth={inst['depth_m']:.3f}m"
        print(f"  {inst['class_name']:>15s}  "
              f"conf={inst['confidence']:.4f}  "
              f"center=({inst['center_uv'][0]:6.1f},{inst['center_uv'][1]:6.1f})  "
              f"-> {inst['target_bin']}{extra}")

    vis = detector.draw_results(bgr, instances)
    os.makedirs(args.output, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(args.output, f"{basename}_debug.jpg")
    cv2.imwrite(out_path, vis)
    print(f"\n可视化: {out_path}")


if __name__ == "__main__":
    main()