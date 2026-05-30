cmake_minimum_required(VERSION 3.0.2)
project(challenge_cup_task_template)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/scene1_express.py
  scripts/scene2_sorting.py
  scripts/scene3_tray.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
