"""
Standalone test of the 3D voxel pipeline: synthetic LiDAR -> voxel_mapper
-> RViz2. No real sensors, no lidar_preprocessor, no TF tree required
(everything is published directly in base_link).

    ros2 launch sensor_fusion_bringup voxel_test.launch.py

Pass use_rviz:=false to run just the data pipeline (e.g. when RViz will be
launched separately, on a machine that doesn't have this workspace).
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
    voxel_config = os.path.join(bringup_share, 'config', 'voxel_mapper.yaml')
    rviz_config = os.path.join(bringup_share, 'rviz', 'sensor_fusion.rviz')

    enable_raycasting = LaunchConfiguration('enable_raycasting')
    voxel_resolution = LaunchConfiguration('voxel_resolution')
    use_rviz = LaunchConfiguration('use_rviz')

    return LaunchDescription([
        DeclareLaunchArgument('enable_raycasting', default_value='true'),
        DeclareLaunchArgument('voxel_resolution', default_value='0.3'),
        DeclareLaunchArgument('use_rviz', default_value='true'),

        Node(
            package='voxel_mapper',
            executable='fake_lidar_publisher',
            name='fake_lidar_publisher',
            parameters=[{
                'rate_hz': 10.0,
                'frame_id': 'base_link',
                'room_half_extent': 8.0,
                'orbit_radius': 4.0,
            }],
        ),
        Node(
            package='voxel_mapper',
            executable='voxel_mapper_node',
            name='voxel_mapper',
            parameters=[voxel_config, {
                'enable_raycasting': enable_raycasting,
                'resolution': voxel_resolution,
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            condition=IfCondition(use_rviz),
        ),
    ])
