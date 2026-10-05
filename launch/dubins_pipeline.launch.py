from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory('planner'))
    parameters_file = package_share / 'config' / 'dubins_pipeline.yaml'

    return LaunchDescription(
        [
            Node(
                package='planner',
                executable='gps_imu_tf_broadcaster',
                name='gps_imu_tf_broadcaster',
                output='screen',
            ),
            Node(
                package='planner',
                executable='waypoint_array_dubins',
                name='waypoint_array_dubins',
                parameters=[str(parameters_file)],
                output='screen',
            ),
            Node(
                package='planner',
                executable='path_frame_transformer',
                name='waypoint_array_transformer',
                parameters=[str(parameters_file)],
                output='screen',
            ),
        ]
    )
