#!/usr/bin/env python3
"""
训练两个小网络:
  网络A (YOLO误差修正): 3→128→128→3, YOLO坐标 → 真值坐标
  网络B (关节角预测):   3→128→128→7, 真值坐标 → 7关节角

用法:
  python3 train_networks.py

输出:
  training_data/yolo_corrector.pt
  training_data/joint_predictor.pt
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")


class YoloCorrector(nn.Module):
    """YOLO有噪声坐标 → 真值坐标"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.net(x)


class JointPredictor(nn.Module):
    """真值物体坐标 → 4关节角(前4个, 手腕固定) (度)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.net(x)


def train():
    yolo_path = os.path.join(DATA_DIR, "yolo_correction.npz")
    joint_path = os.path.join(DATA_DIR, "joint_prediction.npz")

    if not os.path.exists(yolo_path) or not os.path.exists(joint_path):
        print("数据文件不存在! 请先运行 collect_training_data.py")
        return

    yolo_data = np.load(yolo_path)
    joint_data = np.load(joint_path)

    yi = yolo_data["inputs"]   # YOLO 坐标 (N,3)
    yo = yolo_data["outputs"]  # 真值坐标 (N,3)
    ji = joint_data["inputs"]  # 真值坐标 (M,3)
    jo = joint_data["outputs"] # 关节角 (M,7)

    print(f"YOLO 修正数据: {len(yi)} 对")
    print(f"关节预测数据: {len(ji)} 对")

    # ── 训练网络A: YOLO修正 ──
    model_a = YoloCorrector()
    opt = optim.Adam(model_a.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X = torch.tensor(yi, dtype=torch.float32)
    Y = torch.tensor(yo, dtype=torch.float32)

    print("\n训练 YOLO 修正网络...")
    for epoch in range(2000):
        opt.zero_grad()
        pred = model_a(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
        if epoch % 500 == 0:
            err = loss.item() ** 0.5
            print(f"  epoch {epoch:4d}: RMSE={err:.4f}m")

    # 评估
    with torch.no_grad():
        pred = model_a(X)
        errs = torch.sqrt(((pred - Y) ** 2).sum(dim=1))
        print(f"YOLO修正 平均误差: {errs.mean():.4f}m 最大: {errs.max():.4f}m")

    torch.save(model_a.state_dict(), os.path.join(DATA_DIR, "yolo_corrector.pt"))
    print("保存: yolo_corrector.pt")

    # ── 训练网络B: 关节预测 ──
    model_b = JointPredictor()
    opt = optim.Adam(model_b.parameters(), lr=1e-3)

    Xj = torch.tensor(ji, dtype=torch.float32)
    Yj = torch.tensor(jo, dtype=torch.float32)

    print("\n训练关节角预测网络...")
    for epoch in range(2000):
        opt.zero_grad()
        pred = model_b(Xj)
        loss = loss_fn(pred, Yj)
        loss.backward()
        opt.step()
        if epoch % 500 == 0:
            err = loss.item() ** 0.5
            print(f"  epoch {epoch:4d}: RMSE={err:.2f}°")

    with torch.no_grad():
        pred = model_b(Xj)
        errs = torch.sqrt(((pred - Yj) ** 2).sum(dim=1))
        print(f"关节预测 平均误差: {errs.mean():.2f}° 最大: {errs.max():.2f}°")

    torch.save(model_b.state_dict(), os.path.join(DATA_DIR, "joint_predictor.pt"))
    print("保存: joint_predictor.pt")

    print("\n✅ 训练完成!")


if __name__ == "__main__":
    train()
