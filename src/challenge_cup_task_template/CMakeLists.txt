cmake_minimum_required(VERSION 3.0.2)
project(challenge_cup_task_template)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/challenge_task.py
  src/scene3_controller.py
  src/ros_interfaces.py
  src/perception_helper.py
  src/lidar_perception.py
  src/fastlio_localization.py
  src/base_navigator.py
  src/arm_pose_manager.py
  src/skill_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml")
