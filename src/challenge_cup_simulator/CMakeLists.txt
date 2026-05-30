cmake_minimum_required(VERSION 3.0.2)
project(challenge_cup_simulator)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/arm_keyboard_control.py
  scripts/cam_info_pub.py
  scripts/gripper_tf_publisher.py
  scripts/leju_claw_keyboard.py
  scripts/sim_leju_claw_interface.py
  utils/lidar_mid360_node.py
  utils/scene_builder.py
  utils/challenge_sim_launcher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch models utils lib
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
