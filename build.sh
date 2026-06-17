cd /root/kuavo_ws
catkin config -DCMAKE_ASM_COMPILER=/usr/bin/as -DCMAKE_BUILD_TYPE=Release
source installed/setup.zsh
catkin build kuavo_msgs challenge_cup_simulator challenge_cup_task_template humanoid_controllers
source devel/setup.zsh
