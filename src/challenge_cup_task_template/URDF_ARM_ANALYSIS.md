# Biped_S52 双臂 URDF 运动学分析

**来源文件**: `src/challenge_cup_simulator/models/biped_s52/urdf/drake/biped_v3_arm.urdf`

## 坐标系约定

ROS 标准坐标系 (base_link):
- **X+**: 机器人前方
- **Y+**: 机器人左侧
- **Z+**: 机器人上方

所有关节角以右手定则确定正方向。

---

## 肩关节原点

| 臂 | waist_yaw_link 偏移 | zarm 偏移 | base_link 坐标 |
|----|---------------------|-----------|----------------|
| 左肩 | `(0, 0, 0.111)` | `(-0.003, +0.2547, 0.2831)` | `(-0.003, +0.2547, 0.3945)` |
| 右肩 | `(0, 0, 0.111)` | `(-0.003, -0.2547, 0.2831)` | `(-0.003, -0.2547, 0.3945)` |

---

## 左臂关节链

```
waist_yaw_link
 │
 └─[zarm_l1] l_arm_pitch      轴: Y=[0,1,0]    [-137.5°,  +34.4°]
     │  负=手臂前伸, 正=手臂后摆
     │
     └─[zarm_l2] l_arm_roll   轴: X=[1,0,0]    [ -20.0°,  +83.7°]
         │  正=手臂外摆
         │
         └─[zarm_l3] l_arm_yaw   轴: Z=[0,0,1]    [ -90.0°,  +26.4°]
             │
             └─[zarm_l4] l_forearm_pitch  轴: Y=[0,1,0]    [-150.0°,    0.0°]
                 │  负=肘弯曲(手靠近身体), 0=全伸展(手远离身体)
                 │
                 └─[zarm_l5] l_hand_yaw  轴: Z=[0,0,1]    [ -90.0°,  +90.0°]
                     │
                     └─[zarm_l6] l_hand_roll  轴: X=[1,0,0]    [ -75.0°,  +40.0°]
                         │
                         └─[zarm_l7] l_hand_pitch  轴: Y=[0,1,0]    [ -40.0°,  +40.0°]
                             │
                            [end_effector]  xyz="0, -0.03, -0.17"
```

## 右臂关节链

```
waist_yaw_link
 │
 └─[zarm_r1] r_arm_pitch      轴: Y=[0,1,0]    [-137.5°,  +34.4°]
     │  负=手臂前伸, 正=手臂后摆
     │
     └─[zarm_r2] r_arm_roll   轴: X=[1,0,0]    [ -83.7°,  +20.0°]
         │  负=手臂外摆
         │
         └─[zarm_r3] r_arm_yaw   轴: Z=[0,0,1]    [ -26.4°,  +90.0°]
             │
             └─[zarm_r4] r_forearm_pitch  轴: Y=[0,1,0]    [-150.0°,    0.0°]
                 │  负=肘弯曲(手靠近身体), 0=全伸展(手远离身体)
                 │
                 └─[zarm_r5] r_hand_yaw  轴: Z=[0,0,1]    [ -90.0°,  +90.0°]
                     │
                     └─[zarm_r6] r_hand_roll  轴: X=[1,0,0]    [ -40.0°,  +75.0°]
                         │
                         └─[zarm_r7] r_hand_pitch  轴: Y=[0,1,0]    [ -40.0°,  +40.0°]
                             │
                            [end_effector]  xyz="0, +0.03, -0.17"
```

---

## 代码数组 → URDF 关节映射

`send_arm_trajectory` 发送的 `JointState` 带 name 列表，控制器按名称路由，索引顺序不影响。

| index | 左臂 name | 右臂 name | 对应 URDF 关节 | 关节类型 |
|-------|-----------|-----------|----------------|----------|
| `[0]` | `l_arm_pitch` | `r_arm_pitch` | zarm_l1 / zarm_r1 | 肩 pitch |
| `[1]` | `l_arm_roll` | `r_arm_roll` | zarm_l2 / zarm_r2 | 肩 roll |
| `[2]` | `l_arm_yaw` | `r_arm_yaw` | zarm_l3 / zarm_r3 | 肩 yaw |
| `[3]` | `l_forearm_pitch` | `r_forearm_pitch` | zarm_l4 / zarm_r4 | 肘 pitch |
| `[4]` | `l_hand_yaw` | `r_hand_yaw` | zarm_l5 / zarm_r5 | 腕 yaw |
| `[5]` | `l_hand_pitch` | `r_hand_pitch` | zarm_l7 / zarm_r7 | 腕 pitch |
| `[6]` | `l_hand_roll` | `r_hand_roll` | zarm_l6 / zarm_r6 | 腕 roll |

> ⚠️ URDF 运动学顺序为 `...→J5(yaw)→J6(roll)→J7(pitch)`，代码 name 列表顺序为 `J5(yaw)→J6(pitch)→J7(roll)`。由于 `JointState` 按名称匹配，索引顺序不影响最终路由。

---

## 限位汇总表 (度)

| 关节 | 左臂 min | 左臂 max | 右臂 min | 右臂 max | 方向含义 |
|------|----------|----------|----------|----------|----------|
| arm_pitch | -137.5 | +34.4 | -137.5 | +34.4 | 负=前伸, 正=后摆 |
| arm_roll | -20.0 | +83.7 | -83.7 | +20.0 | 左正=外摆, 右负=外摆 |
| arm_yaw | -90.0 | +26.4 | -26.4 | +90.0 | |
| forearm_pitch | **-150.0** | **0.0** | **-150.0** | **0.0** | 负=肘弯曲, 0=全伸展 |
| hand_yaw | -90.0 | +90.0 | -90.0 | +90.0 | |
| hand_pitch | -40.0 | +40.0 | -40.0 | +40.0 | 独立弯腕 |
| hand_roll | -75.0 | +40.0 | -40.0 | +75.0 | |

---

## 手臂姿态示意

```
                Z↑ (上)
                │
        肩 ●───┼─── arm_pitch=0° (臂垂直下垂)
                │ \
                │  \  arm_pitch<0 (前伸)
                │   \
                │    ○ 肘 (forearm_pitch)
                │   / \
                │  /   \ forearm<0 (弯曲→手近身)
                │ /     \
                │/       ● 手 (end_effector ≈ 手心下17cm)
                └────────────→ X+ (前)


  forearm 角度含义:
    -150° ───────────────── 0°
    最弯(手贴肩)             全伸展(手离肩最远)


  掌心方向由  arm_pitch + forearm_pitch + hand_pitch  叠加决定:
    0°     → 掌心朝前
    -90°   → 掌心朝下 (竖直向下)
    -130°  → 掌心朝后上方 (需要 forearm=-90 + hand_pitch=-40)
```

---

## 当前代码中的 IK 假设

| 参数 | 值 | 来源 |
|------|-----|------|
| 上臂长度 L1 | 0.32 m | 估测 |
| 前臂长度 L2 | 0.35 m | 估测 |
| 肩高 shoulder_z | 0.395 m | URDF: 0.1114 + 0.2831 |
| 肩横偏 shoulder_y | ±0.255 m | URDF origin Y |
| 目标上方偏移 | 0.15 m | 可调 |
| IK 类型 | 2连杆 (pitch+forearm) | arm_roll 单独计算 |
