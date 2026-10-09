import csv
import math
import os

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)

from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)

from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)

from launch.substitutions import (
    LaunchConfiguration,
)

from launch_ros.actions import Node

import vrx_gz.launch
from vrx_gz.model import Model


# =============================================================
# VEHICLE GEOMETRY
# =============================================================

PREDECESSOR_REAR_EXTENT = 2.822
FOLLOWER_FRONT_EXTENT = 2.549
FORMATION_DISTANCE = 5.0

CENTER_SPACING = (
    PREDECESSOR_REAR_EXTENT
    + FORMATION_DISTANCE
    + FOLLOWER_FRONT_EXTENT
)


def launch_system(context):

    # =========================================================
    # Launch arguments
    # =========================================================

    headless = (
        LaunchConfiguration(
            'headless'
        ).perform(context).lower()
        == 'true'
    )

    simulation_profile = (
        LaunchConfiguration(
            'simulation_profile'
        ).perform(context).lower()
    )

    run_name = LaunchConfiguration(
        'run_name'
    ).perform(context)

    results_dir = os.path.expanduser(
        LaunchConfiguration(
            'results_dir'
        ).perform(context)
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    leader_csv = os.path.join(
        results_dir,
        f'{run_name}_r1.csv',
    )

    r2_csv = os.path.join(
        results_dir,
        f'{run_name}_r2.csv',
    )

    r3_csv = os.path.join(
        results_dir,
        f'{run_name}_r3.csv',
    )

    for output_file in (
        leader_csv,
        r2_csv,
        r3_csv,
    ):
        if os.path.exists(output_file):
            raise RuntimeError(
                'Result file already exists: '
                + output_file
            )

    if simulation_profile not in (
        'full',
        'light',
    ):
        raise RuntimeError(
            'simulation_profile must be '
            "'full' or 'light'"
        )

    # =========================================================
    # Package locations
    # =========================================================

    planner_share = (
        get_package_share_directory(
            'planner'
        )
    )

    bringup_share = (
        get_package_share_directory(
            'platoon_bringup'
        )
    )

    control_share = (
        get_package_share_directory(
            'platoon_control'
        )
    )

    platoon_planner_share = (
        get_package_share_directory(
            'platoon_planner'
        )
    )

    # =========================================================
    # Read first Dubins waypoint
    #
    # CSV coordinates are Gazebo ENU:
    #   x = East
    #   y = North
    #   yaw = Gazebo ENU yaw
    # =========================================================

    trajectory_csv = os.path.join(
        planner_share,
        'output',
        'sydney_coverage_dubins_path.csv',
    )

    if not os.path.isfile(
        trajectory_csv
    ):
        raise RuntimeError(
            'Dubins trajectory does not exist:\n'
            f'{trajectory_csv}\n\n'
            'Generate and accept a path in the GUI first.'
        )

    with open(
        trajectory_csv,
        'r',
        newline='',
        encoding='utf-8',
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        required_columns = {
            'x',
            'y',
            'yaw',
        }

        if reader.fieldnames is None:
            raise RuntimeError(
                'Trajectory CSV has no header.'
            )

        missing = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing:
            raise RuntimeError(
                'Trajectory CSV missing columns: '
                + ', '.join(
                    sorted(missing)
                )
            )

        first_waypoint = next(
            reader,
            None,
        )

    if first_waypoint is None:
        raise RuntimeError(
            'Trajectory CSV contains no waypoints.'
        )

    try:

        r1_x = float(
            first_waypoint['x']
        )

        r1_y = float(
            first_waypoint['y']
        )

        start_yaw = float(
            first_waypoint['yaw']
        )

    except (
        TypeError,
        ValueError,
    ) as error:

        raise RuntimeError(
            'Invalid first waypoint.'
        ) from error

    if not all(
        math.isfinite(value)
        for value in (
            r1_x,
            r1_y,
            start_yaw,
        )
    ):
        raise RuntimeError(
            'First waypoint contains '
            'non-finite values.'
        )

    # =========================================================
    # Place R2 and R3 behind R1 along initial heading
    # =========================================================

    dx = (
        CENTER_SPACING
        * math.cos(start_yaw)
    )

    dy = (
        CENTER_SPACING
        * math.sin(start_yaw)
    )

    r2_x = r1_x - dx
    r2_y = r1_y - dy

    r3_x = r1_x - 2.0 * dx
    r3_y = r1_y - 2.0 * dy

    # =========================================================
    # URDF selection
    # =========================================================

    r1_urdf = ''
    r2_urdf = ''
    r3_urdf = ''

    if simulation_profile == 'light':

        r1_urdf = os.path.join(
            bringup_share,
            'urdf',
            'wamv_light.urdf.xacro',
        )

        r2_urdf = os.path.join(
            bringup_share,
            'urdf',
            'wamv_light_r2.urdf.xacro',
        )

        r3_urdf = os.path.join(
            bringup_share,
            'urdf',
            'wamv_light_r3.urdf.xacro',
        )

    # =========================================================
    # Models
    # =========================================================

    r1 = Model(
        'wamv',
        'wam-v',
        [
            r1_x,
            r1_y,
            0.0,
            0.0,
            0.0,
            start_yaw,
        ],
    )

    r2 = Model(
        'wamv2',
        'wam-v',
        [
            r2_x,
            r2_y,
            0.0,
            0.0,
            0.0,
            start_yaw,
        ],
    )

    r3 = Model(
        'wamv3',
        'wam-v',
        [
            r3_x,
            r3_y,
            0.0,
            0.0,
            0.0,
            start_yaw,
        ],
    )

    if r1_urdf:
        r1.set_urdf(
            r1_urdf
        )
        r2.set_urdf(
            r2_urdf
        )
        r3.set_urdf(
            r3_urdf
        )

    actions = []

    # =========================================================
    # Gazebo
    # =========================================================

    actions.extend(
        vrx_gz.launch.simulation(
            'sydney_regatta',
            headless=headless,
            paused=False,
            extra_gz_args='',
        )
    )

    actions.extend(
        vrx_gz.launch.spawn(
            'full',
            'sydney_regatta',
            [
                r1,
                r2,
                r3,
            ],
        )
    )

    actions.extend(
        vrx_gz.launch.competition_bridges(
            'sydney_regatta',
            False,
        )
    )

    # =========================================================
    # Vehicle-state nodes
    # =========================================================

    # R1 keeps the exact state architecture used by the
    # working single-leader Dubins launch.
    state_config = os.path.join(
        get_package_share_directory(
            'platoon_state'
        ),
        'config',
        'origin.yaml',
    )

    actions.append(
        Node(
            package='platoon_state',
            executable='vehicle_state',
            name='vehicle_state_node',
            output='screen',
            parameters=[
                state_config,
                {
                    'use_sim_time':
                        True,
                },
            ],
        )
    )

    # R1 also publishes the generic platoon state required
    # by the follower planner.
    actions.append(
        Node(
            package='platoon_state',
            executable='multi_vehicle_state',
            namespace='r1',
            name='multi_vehicle_state',
            output='screen',

            parameters=[{
                'vehicle_id':
                    'r1',

                'gps_topic':
                    '/wamv/sensors/gps/gps/fix',

                'imu_topic':
                    '/wamv/sensors/imu/imu/data',

                'state_topic':
                    '/r1/vehicle_state',

                'use_sim_time':
                    True,
            }],
        )
    )

    # R2 and R3 use the generic multi-vehicle state nodes.
    for vehicle_id, model in (
        ('r2', 'wamv2'),
        ('r3', 'wamv3'),
    ):

        actions.append(
            Node(
                package='platoon_state',
                executable='multi_vehicle_state',
                namespace=vehicle_id,
                name='multi_vehicle_state',
                output='screen',

                parameters=[{
                    'vehicle_id':
                        vehicle_id,

                    'gps_topic':
                        f'/{model}/sensors/gps/gps/fix',

                    'imu_topic':
                        f'/{model}/sensors/imu/imu/data',

                    'state_topic':
                        f'/{vehicle_id}/vehicle_state',

                    'use_sim_time':
                        True,
                }],
            )
        )

    # =========================================================
    # R1 Dubins coverage planner pipeline
    # =========================================================

    dubins_pipeline = os.path.join(
        planner_share,
        'launch',
        'dubins_pipeline.launch.py',
    )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                dubins_pipeline
            ),
            launch_arguments={
                'show_live_tracker':
                    'false',
            }.items(),
        )
    )

    # =========================================================
    # R1 controller
    # =========================================================

    actions.append(
        Node(
            package='platoon_control',
            executable='leader_pid_controller',
            name='leader_pid_controller',
            output='screen',

            parameters=[
                os.path.join(
                    control_share,
                    'config',
                    'leader_pid.yaml',
                ),
                {
                    'use_sim_time':
                        True,
                },
            ],
        )
    )

    # =========================================================
    # R2 <- R1 planner
    # =========================================================

    actions.append(
        Node(
            package='platoon_planner',
            executable='proactive_follower_planner_v22',
            namespace='r2',
            name='follower_planner',
            output='screen',

            parameters=[
                {
                    'follower_id':
                        'r2',

                    'predecessor_id':
                        'r1',

                    'predecessor_success_topic':
                        '/r1/success',

                    'reference_path_topic':
                        '/planner/reference_path',

                    'terminal_handoff_mode':
                        'reference_path',

                    'terminal_behavior':
                        'hold',

                    'mission_state_topic':
                        '/r2/mission_state',

                    'mission_success_topic':
                        '/r2/pair_success',

                    'formation_distance':
                        FORMATION_DISTANCE,

                    'use_sim_time':
                        True,
                },

                os.path.join(
                    platoon_planner_share,
                    'config',
                    'r2_follower_v22_tuned.yaml',
                ),
            ],
        )
    )

    # =========================================================
    # R2 controller
    # =========================================================

    actions.append(
        Node(
            package='platoon_control',
            executable='proactive_follower_controller_v22',
            namespace='r2',
            name='follower_controller',
            output='screen',

            parameters=[
                {
                    'follower_id':
                        'r2',

                    'predecessor_id':
                        'r1',

                    'actuator_prefix':
                        '/wamv2',

                    'mission_state_topic':
                        '/r2/mission_state',

                    'formation_distance':
                        FORMATION_DISTANCE,

                    'use_sim_time':
                        True,
                },

                os.path.join(
                    control_share,
                    'config',
                    'r2_follower_v22_tuned.yaml',
                ),
            ],
        )
    )

    # =========================================================
    # R3 <- R2 planner
    # =========================================================

    actions.append(
        Node(
            package='platoon_planner',
            executable='proactive_follower_planner_v22',
            namespace='r3',
            name='follower_planner',
            output='screen',

            parameters=[
                {
                    'follower_id':
                        'r3',

                    'predecessor_id':
                        'r2',

                    'predecessor_success_topic':
                        '/r2/pair_success',

                    'terminal_handoff_mode':
                        'predecessor_mission',

                    'predecessor_mission_state_topic':
                        '/r2/mission_state',

                    'release_mode':
                        'predecessor_release_bootstrap',

                    'predecessor_release_topic':
                        '/planner/r2/released',

                    'terminal_behavior':
                        'hold',

                    'mission_state_topic':
                        '/r3/mission_state',

                    'mission_success_topic':
                        '/r3/pair_success',

                    'formation_distance':
                        FORMATION_DISTANCE,

                    'use_sim_time':
                        True,
                },

                os.path.join(
                    platoon_planner_share,
                    'config',
                    'r3_follower_v22_seed.yaml',
                ),
            ],
        )
    )

    # =========================================================
    # R3 controller
    # =========================================================

    actions.append(
        Node(
            package='platoon_control',
            executable='proactive_follower_controller_v22',
            namespace='r3',
            name='follower_controller',
            output='screen',

            parameters=[
                {
                    'follower_id':
                        'r3',

                    'predecessor_id':
                        'r2',

                    'actuator_prefix':
                        '/wamv3',

                    'mission_state_topic':
                        '/r3/mission_state',

                    'formation_distance':
                        FORMATION_DISTANCE,

                    'use_sim_time':
                        True,
                },

                os.path.join(
                    control_share,
                    'config',
                    'r3_follower_v22_seed.yaml',
                ),
            ],
        )
    )

    # =========================================================
    # Experiment logging
    # =========================================================

    # R1 trajectory / reference / thrust logger.
    actions.append(
        Node(
            package='planner',
            executable='trajectory_logger',
            name='trajectory_logger',
            output='screen',
            parameters=[{
                'output_file':
                    leader_csv,

                'sample_rate':
                    10.0,

                'use_sim_time':
                    True,
            }],
        )
    )

    # R2 <- R1 V2.2 formation logger.
    actions.append(
        Node(
            package='platoon_monitor',
            executable='follower_logger_v22',
            namespace='r2',
            name='follower_logger',
            output='screen',
            parameters=[{
                'follower_id':
                    'r2',

                'predecessor_id':
                    'r1',

                'predecessor_actuator_prefix':
                    '/wamv',

                'follower_actuator_prefix':
                    '/wamv2',

                'mission_state_topic':
                    '/r2/mission_state',

                # leader_pid_controller publishes success here.
                'predecessor_success_topic':
                    '/experiment/success',

                'pair_success_topic':
                    '/r2/pair_success',

                'formation_distance':
                    FORMATION_DISTANCE,

                'output_file':
                    r2_csv,

                'use_sim_time':
                    True,
            }],
        )
    )

    # R3 <- R2 V2.2 formation logger.
    actions.append(
        Node(
            package='platoon_monitor',
            executable='follower_logger_v22',
            namespace='r3',
            name='follower_logger',
            output='screen',
            parameters=[{
                'follower_id':
                    'r3',

                'predecessor_id':
                    'r2',

                'predecessor_actuator_prefix':
                    '/wamv2',

                'follower_actuator_prefix':
                    '/wamv3',

                'mission_state_topic':
                    '/r3/mission_state',

                'predecessor_success_topic':
                    '/r2/pair_success',

                'pair_success_topic':
                    '/r3/pair_success',

                'formation_distance':
                    FORMATION_DISTANCE,

                'output_file':
                    r3_csv,

                'use_sim_time':
                    True,
            }],
        )
    )

    # =========================================================
    # Summary
    # =========================================================

    print(
        '\n'
        '============================================\n'
        ' DUBINS THREE-ROBOT COVERAGE PLATOON\n'
        '============================================\n'
        f' R1 spawn : ({r1_x:.3f}, {r1_y:.3f})\n'
        f' R2 spawn : ({r2_x:.3f}, {r2_y:.3f})\n'
        f' R3 spawn : ({r3_x:.3f}, {r3_y:.3f})\n'
        f' Heading  : {math.degrees(start_yaw):.2f} deg\n'
        '\n'
        f' Center spacing : {CENTER_SPACING:.3f} m\n'
        f' Hull gap       : {FORMATION_DISTANCE:.3f} m\n'
        '\n'
        ' R1 : Dubins coverage trajectory\n'
        ' R2 : follows R1 breadcrumbs\n'
        ' R3 : follows R2 breadcrumbs\n'
        '\n'
        f' Run name : {run_name}\n'
        f' R1 CSV   : {leader_csv}\n'
        f' R2 CSV   : {r2_csv}\n'
        f' R3 CSV   : {r3_csv}\n'
        '============================================\n'
    )

    return actions


def generate_launch_description():

    # =========================================================
    # Gazebo resources
    # =========================================================

    resource_paths = [
        os.path.join(
            get_package_prefix(
                'wamv_description'
            ),
            'share',
        ),

        os.path.join(
            get_package_prefix(
                'wamv_gazebo'
            ),
            'share',
        ),

        os.path.join(
            get_package_prefix(
                'vrx_gz'
            ),
            'share',
        ),

        os.path.join(
            get_package_prefix(
                'platoon_bringup'
            ),
            'share',
        ),
    ]

    existing = os.environ.get(
        'GZ_SIM_RESOURCE_PATH',
        '',
    )

    if existing:
        resource_paths.append(
            existing
        )

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name='GZ_SIM_RESOURCE_PATH',
                value=':'.join(
                    resource_paths
                ),
            ),

            DeclareLaunchArgument(
                'headless',
                default_value='False',
            ),

            DeclareLaunchArgument(
                'simulation_profile',
                default_value='light',
                description=(
                    'Simulation profile: '
                    'light or full'
                ),
            ),

            DeclareLaunchArgument(
                'run_name',
                default_value='dubins_three_robot_run',
                description=(
                    'Experiment filename prefix'
                ),
            ),

            DeclareLaunchArgument(
                'results_dir',
                default_value=os.path.expanduser(
                    '~/vrx_ws/src/planner/results'
                ),
                description=(
                    'Directory for experiment CSV files'
                ),
            ),

            OpaqueFunction(
                function=launch_system,
            ),
        ]
    )
