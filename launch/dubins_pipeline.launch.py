
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    package_share = Path(
        get_package_share_directory('planner')
    )

    show_live_tracker = LaunchConfiguration(
        'show_live_tracker'
    )

    parameters_file = (
        package_share
        / 'config'
        / 'dubins_pipeline.yaml'
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'show_live_tracker',
                default_value='true',
                description=(
                    'Launch standalone coverage '
                    'tracking viewer'
                ),
            ),
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
                executable='dubins_reference_path_publisher',
                name='dubins_reference_path_publisher',
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

            Node(
                package='planner',
                executable='coverage_path_live_viewer',
                name='coverage_path_live_viewer',
                parameters=[
                    {
                        'path_csv': str(
                            package_share
                            / 'output'
                            / 'sydney_coverage_dubins_path.csv'
                        )
                    }
                ],
                output='screen',
                condition=IfCondition(
                    show_live_tracker
                ),
            ),
        ]
    )
