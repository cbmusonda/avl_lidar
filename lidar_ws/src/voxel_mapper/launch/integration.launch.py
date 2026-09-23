"""
Full bring-up: sensors + odometry + voxel mapping (+ optional RViz).

Sets RMW/DDS configuration here at the top level, not in nested launch
files, per the brief's guidance -- avoids duplicate/conflicting DDS config
if any of the nested launch files are ever included standalone elsewhere.

Uses CycloneDDS (not the ROS 2 default FastDDS), matching IGVC_ROS2's own
config/cyclonedds.xml -- their known-issues notes document FastDDS
corrupting action goal payloads when directly-instantiated nodes (the
robot_localization EKFs, navsat_transform) came up on a different RMW
default than the sensor driver stack.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    voxel_mapper_share = get_package_share_directory('voxel_mapper')
    rviz_config = os.path.join(voxel_mapper_share, 'rviz', 'lidar.rviz')
    cyclonedds_config = os.path.join(voxel_mapper_share, 'config', 'cyclonedds.xml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    launch_args = [
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('enable_lidar', default_value='true'),
        DeclareLaunchArgument('enable_xsens', default_value='true'),
        DeclareLaunchArgument('enable_odometry', default_value='true'),
        DeclareLaunchArgument('enable_gps', default_value='true'),
        DeclareLaunchArgument('enable_raycasting', default_value='true'),
        DeclareLaunchArgument('publish_rviz', default_value='false'),
        DeclareLaunchArgument('velodyne_ip', default_value='192.168.13.11'),
        DeclareLaunchArgument('xsens_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('voxel_resolution', default_value='0.3'),
    ]

    # RMW/DDS configuration belongs here (top-level), not duplicated in
    # each nested launch file.
    env = [
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
        SetEnvironmentVariable('CYCLONEDDS_URI', 'file://' + cyclonedds_config),
    ]

    voxel_mapper_share_sub = FindPackageShare('voxel_mapper')

    def voxel_mapper_launch_file(name):
        return PathJoinSubstitution([voxel_mapper_share_sub, 'launch', name])

    sensors = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(voxel_mapper_launch_file('sensors.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'enable_lidar': LaunchConfiguration('enable_lidar'),
            'enable_xsens': LaunchConfiguration('enable_xsens'),
            'velodyne_ip': LaunchConfiguration('velodyne_ip'),
            'xsens_port': LaunchConfiguration('xsens_port'),
        }.items(),
    )

    odometry = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(voxel_mapper_launch_file('odometry.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'enable_odometry': LaunchConfiguration('enable_odometry'),
            'enable_gps': LaunchConfiguration('enable_gps'),
        }.items(),
    )

    voxel_mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(voxel_mapper_launch_file('voxel_mapping.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'enable_raycasting': LaunchConfiguration('enable_raycasting'),
            'voxel_resolution': LaunchConfiguration('voxel_resolution'),
        }.items(),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('publish_rviz')),
    )

    return LaunchDescription(launch_args + env + [sensors, odometry, voxel_mapping, rviz])
