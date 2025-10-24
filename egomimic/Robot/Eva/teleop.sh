export LD_LIBRARY_PATH=/root/.local/share/mamba/envs/arx-py310/lib:$LD_LIBRARY_PATH

cd /home/robot/robot_ws/egomimic/Robot/Eva/eva_ws/
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch eva eva_bringup.launch.py arm:=both
