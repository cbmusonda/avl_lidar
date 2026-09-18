#!/bin/bash
# Voxel pipeline on the live /velodyne_points (no drivers, no second TF publisher).
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source /home/dinosaur/IGVC/install/setup.bash
source /home/dinosaur/avl_sensorFusion/sensor_fusion_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml
exec ros2 launch sensor_fusion_bringup voxel_mapping.launch.py
