# RL Grasping Module for Kuavo Challenge Cup Scene 2
# Residual RL (Route B) + Gripper Strategy RL (Route A)
#
# Training (no ROS needed):
#   python3 rl/train.py --total-timesteps 5000000 --n-envs 8 --curriculum
#
# Deployment (ROS container):
#   ONNX models auto-loaded by challenge_task.py Scene2Controller
