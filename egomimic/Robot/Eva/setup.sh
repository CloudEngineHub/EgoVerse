source /opt/ros/humble/setup.bash
eval "$(micromamba shell hook --shell bash)"
micromamba create -y -f /home/robot/robot_ws/egomimic/Robot/Eva/Stanford-Repo/conda_environments/py310_environment.yaml -n arx-py310 
micromamba activate arx-py310

cd /home/robot/robot_ws/egomimic/Robot/Eva/Stanford-Repo 
mkdir build && cd build
# cmake .. -DCMAKE_PREFIX_PATH=/opt/ros/humble
cmake .. -DCMAKE_PREFIX_PATH=/opt/ros/humble -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++
make -j

cd /home/robot/robot_ws/egomimic/Robot/Eva/Stanford-Repo/python 
mkdir /opt/ros/humble/lib/python3.10/site-packages/arx5
cp arx5_interface.cpython-310-x86_64-linux-gnu.so /opt/ros/humble/lib/python3.10/site-packages/arx5/arx5_interface.cpython-310-x86_64-linux-gnu.so