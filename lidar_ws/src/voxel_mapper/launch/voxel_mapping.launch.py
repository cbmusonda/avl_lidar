"""lidar_preprocessor + voxel_mapper: the local 3D voxel obstacle pipeline."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    voxel_mapper_share = get_package_share_directory('voxel_mapper')
    voxel_config = os.path.join(voxel_mapper_share, 'config', 'voxel_mapper.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_raycasting = LaunchConfiguration('enable_raycasting')
    voxel_resolution = LaunchConfiguration('voxel_resolution')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('enable_raycasting', default_value='true'),
        DeclareLaunchArgument('voxel_resolution', default_value='0.3'),

        Node(
            package='voxel_mapper',
            executable='lidar_preprocessor',
            name='lidar_preprocessor',
            parameters=[voxel_config, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='voxel_mapper',
            executable='voxel_mapper_node',
            name='voxel_mapper',
            parameters=[voxel_config, {
                'use_sim_time': use_sim_time,
                'enable_raycasting': enable_raycasting,
                'resolution': voxel_resolution,
            }],
        ),
    ])
