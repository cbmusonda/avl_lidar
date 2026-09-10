"""
odom -> base_link and map -> odom, via robot_localization dual EKF + GPS.

Matches IGVC_ROS2's src/avros_bringup/launch/localization.launch.py pattern
(same robot_localization/navsat_transform setup, same input topics):

  EKF #1 (odom): /imu/data + /wheel_odom (vx)                  -> odom -> base_link
  EKF #2 (map):  /imu/data + /wheel_odom (vx) + /odometry/gps  -> map -> odom
  navsat_transform_node: /gnss (NavSatFix) -> /odometry/gps for EKF #2

/wheel_odom and /gnss are expected to already be published by nodes outside
this workspace (an actuator/motor-controller node and a GPS driver,
respectively) -- this launch file only consumes them, it does not launch
drivers for them.

enable_odometry gates EKF #1 (the one the voxel mapper's TF chain actually
needs). enable_gps additionally gates EKF #2 + navsat_transform, since GPS
may be unavailable (indoor testing) even when local odometry is wanted.
Each EKF owns exactly one TF edge (odom->base_link / map->odom) -- no other
node in this workspace should ever publish either.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory('sensor_fusion_bringup')
    ekf_config = os.path.join(bringup_share, 'config', 'ekf.yaml')
    navsat_config = os.path.join(bringup_share, 'config', 'navsat.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_odometry = LaunchConfiguration('enable_odometry')
    enable_gps = LaunchConfiguration('enable_gps')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('enable_odometry', default_value='true'),
        DeclareLaunchArgument('enable_gps', default_value='true'),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_odom',
            parameters=[ekf_config, {'use_sim_time': use_sim_time}],
            remappings=[('odometry/filtered', '/odometry/filtered')],
            condition=IfCondition(enable_odometry),
        ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_map',
            parameters=[ekf_config, {'use_sim_time': use_sim_time}],
            remappings=[('odometry/filtered', '/odometry/global')],
            condition=IfCondition(enable_gps),
        ),

        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform',
            parameters=[navsat_config, {'use_sim_time': use_sim_time}],
            remappings=[
                ('imu/data', '/imu/data'),
                ('gps/fix', '/gnss'),
                ('gps/filtered', '/gps/filtered'),
                ('odometry/gps', '/odometry/gps'),
                ('odometry/filtered', '/odometry/global'),
            ],
            condition=IfCondition(enable_gps),
        ),
    ])
