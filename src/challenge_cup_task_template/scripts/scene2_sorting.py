#!/usr/bin/env python3
"""
场景二：分拣归档任务脚本模板

运行方式：
  rosrun challenge_cup_task_template scene2_sorting.py            # 使用默认种子 0
  rosrun challenge_cup_task_template scene2_sorting.py --seed 3   # 指定随机种子

注意：--seed 在本场景有效——scene2 配置了 shuffleable_parts（6 个零件），
      不同 seed 会打乱零件的摆放位置（构建期随机化）。

可用接口（挑战杯 challenge_cup_simulator）：
  - /cmd_vel (geometry_msgs/Twist)              速度指令: linear.x=前进, linear.y=侧移, angular.z=转向
  - /kuavo_arm_traj (sensor_msgs/JointState)    手臂轨迹控制
  - /lidar/points (sensor_msgs/PointCloud2)     雷达点云
  - /sensors_data_raw (kuavo_msgs/sensorsData)  传感器原始数据（IMU、关节等）
  夹爪（注意：用挑战杯自己的 Leju 夹爪接口，不是 CRAIC 的 GripperController）：
  - service /control_robot_leju_claw (kuavo_msgs/controlLejuClaw)  夹爪控制服务
  - topic   /leju_claw_command (kuavo_msgs/lejuClawCommand)        夹爪命令
  - topic   /leju_claw_state   (kuavo_msgs/lejuClawState)          夹爪状态
  参考实现：challenge_cup_simulator/scripts/sim_leju_claw_interface.py、leju_claw_keyboard.py
"""

import argparse
import os
import sys

# 公共启动器位于受保护包 challenge_cup_simulator/utils/（选手不可改动），
# 从那里导入，确保完整性校验无法被绕过。
try:
    import rospkg
    _sim_utils = os.path.join(rospkg.RosPack().get_path("challenge_cup_simulator"), "utils")
except Exception:
    _sim_utils = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "..", "challenge_cup_simulator", "utils")
sys.path.insert(0, _sim_utils)
from challenge_sim_launcher import ChallengeSimLauncher


def main():
    parser = argparse.ArgumentParser(description="场景二：分拣归档")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（scene2 有效：打乱 6 个零件的摆放位置）")
    args = parser.parse_args()

    # ---- 启动仿真（生成场景XML + roslaunch + 初始化ROS节点 + 等待就绪） ----
    launcher = ChallengeSimLauncher(scene="scene2", seed=args.seed)
    launcher.start(node_name="scene2_sorting")

    # 以下代码在 ROS 节点初始化完成后执行
    import rospy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import JointState

    rospy.loginfo("=== 场景二：分拣归档任务启动 ===")

    # ---- 发布器 ----
    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    arm_traj_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)

    rospy.sleep(1.0)  # 等待节点初始化

    # ========================================
    # TODO: 在此实现你的分拣归档逻辑
    # ========================================

    # 示例：通过 Leju 夹爪服务抓取/释放
    # from kuavo_msgs.srv import controlLejuClaw
    # rospy.wait_for_service("/control_robot_leju_claw")
    # claw = rospy.ServiceProxy("/control_robot_leju_claw", controlLejuClaw)
    # ... 按 sim_leju_claw_interface.py 的请求格式填充 ...

    rospy.spin()


if __name__ == "__main__":
    main()
