"""
Sensor drivers + TF backbone: robot_state_publisher, Velodyne, Xsens.

robot_state_publisher always starts (TF backbone is needed even with
sensors disabled, e.g. for bench testing). Velodyne and Xsens are each
independently toggleable so they can be developed/tested in isolation.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory('sensor_fusion_bringup')
    urdf_path = os.path.join(bringup_share, 'urdf', 'vehicle.urdf.xacro')
    velodyne_config = os.path.join(bringup_share, 'config', 'velodyne.yaml')
    xsens_config = os.path.join(bringup_share, 'config', 'xsens.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_lidar = LaunchConfiguration('enable_lidar')
    enable_xsens = LaunchConfiguration('enable_xsens')
    velodyne_ip = LaunchConfiguration('velodyne_ip')
    xsens_port = LaunchConfiguration('xsens_port')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('enable_lidar', default_value='true'),
        DeclareLaunchArgument('enable_xsens', default_value='true'),
        DeclareLaunchArgument('velodyne_ip', default_value='192.168.13.11'),
        DeclareLaunchArgument('xsens_port', default_value='/dev/ttyUSB0'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': Command(['xacro ', urdf_path]),
                'use_sim_time': use_sim_time,
            }],
        ),

        Node(
            package='velodyne_driver',
            executable='velodyne_driver_node',
            name='velodyne_driver_node',
            parameters=[velodyne_config, {
                'device_ip': velodyne_ip,
                'use_sim_time': use_sim_time,
            }],
            condition=IfCondition(enable_lidar),
        ),
        Node(
            package='velodyne_pointcloud',
            executable='velodyne_transform_node',
            name='velodyne_transform_node',
            parameters=[velodyne_config, {'use_sim_time': use_sim_time}],
            condition=IfCondition(enable_lidar),
        ),

        # xsens_mti_ros2_driver's actual default IMU output topic is
        # unverified without the source repo present. The 'remappings'
        # entry below is a placeholder -- update its *first* element to
        # match the driver's real default topic once confirmed on the
        # installed driver (e.g. `ros2 topic list` while it runs), so the
        # fused IMU output lands on /imu/data (avl_slam's convention,
        # required by odometry.launch.py) regardless of the driver default.
        Node(
            package='xsens_mti_ros2_driver',
            executable='xsens_mti_node',
            name='xsens_mti_node',
            parameters=[xsens_config, {
                'port': xsens_port,
                'use_sim_time': use_sim_time,
            }],
            remappings=[('/imu/data', '/imu/data')],  # TODO: verify source topic
            condition=IfCondition(enable_xsens),
        ),
    ])
