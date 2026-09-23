#!/bin/bash
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source /home/dinosaur/IGVC/install/setup.bash
source /home/dinosaur/avl_sensorFusion/sensor_fusion_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml
export DISPLAY=:1001 LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa GALLIUM_DRIVER=llvmpipe QT_X11_NO_MITSHM=1
exec rviz2 -d /home/dinosaur/avl_sensorFusion/sensor_fusion_ws/src/voxel_mapper/rviz/sensor_fusion.rviz
