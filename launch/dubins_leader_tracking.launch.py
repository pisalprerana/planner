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
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)

from launch.conditions import IfCondition

from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)

from launch.substitutions import (
    LaunchConfiguration,
)

from launch_ros.actions import Node

import vrx_gz.launch
from vrx_gz.model import Model


def launch_experiment(context):

    # =========================================================
    # Read launch arguments
    # =========================================================

    trajectory_type = LaunchConfiguration(
        'trajectory_type'
    ).perform(context)

    run_name = LaunchConfiguration(
        'run_name'
    ).perform(context)

    simulation_profile = LaunchConfiguration(
        'simulation_profile'
    ).perform(context).lower()

    show_live_map = LaunchConfiguration(
        'show_live_map'
    ).perform(context)

    results_dir = os.path.expanduser(
        LaunchConfiguration(
            'results_dir'
        ).perform(context)
    )

    # =========================================================
    # Validate simulation profile
    # =========================================================

    if simulation_profile not in (
        'full',
        'light',
    ):
        raise RuntimeError(
            '\n'
            'Invalid simulation_profile.\n'
            'Use:\n'
            '  full\n'
            '  light\n'
        )

    # =========================================================
    # Results directory
    # =========================================================

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        results_dir,
        f'{run_name}.csv'
    )

    if os.path.exists(output_file):
        raise RuntimeError(
            '\n'
            'RESULT FILE ALREADY EXISTS:\n'
            f'{output_file}\n\n'
            'Choose a different run_name.'
        )

    # =========================================================
    # Configuration files
    # =========================================================

    state_config = os.path.join(
        get_package_share_directory(
            'platoon_state'
        ),
        'config',
        'origin.yaml'
    )
    controller_config = os.path.join(
        get_package_share_directory(
            'platoon_control'
        ),
        'config',
        'leader_pid.yaml'
    )

    # =========================================================
    # Select WAM-V model
    # =========================================================

    if simulation_profile == 'light':

        robot_urdf = os.path.join(
            get_package_share_directory(
                'platoon_bringup'
            ),
            'urdf',
            'wamv_light.urdf.xacro'
        )

    else:

        # Empty string makes VRX use the normal WAM-V model.
        robot_urdf = ''

    # Spawn R1 at the first waypoint in the planner's Gazebo ENU CSV.
    trajectory_csv = os.path.join(
        get_package_share_directory('planner'),
        'output',
        'sydney_coverage_dubins_path.csv',
    )

    with open(
        trajectory_csv,
        'r',
        newline='',
        encoding='utf-8',
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {'x', 'y', 'yaw'}
        if reader.fieldnames is None:
            raise RuntimeError(
                f'Trajectory CSV has no header: {trajectory_csv}'
            )

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise RuntimeError(
                f'Trajectory CSV is missing columns: '
                f'{", ".join(sorted(missing_columns))}'
            )

        first_waypoint = next(reader, None)

    if first_waypoint is None:
        raise RuntimeError(
            f'Trajectory CSV contains no waypoints: {trajectory_csv}'
        )

    try:
        spawn_x, spawn_y, spawn_yaw = (
            float(first_waypoint[column])
            for column in ('x', 'y', 'yaw')
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f'First trajectory waypoint has invalid ENU pose in '
            f'{trajectory_csv}: {error}'
        ) from error

    if not all(
        math.isfinite(value)
        for value in (spawn_x, spawn_y, spawn_yaw)
    ):
        raise RuntimeError(
            f'First trajectory waypoint contains a non-finite ENU pose: '
            f'x={spawn_x}, y={spawn_y}, yaw={spawn_yaw}'
        )

    r1 = Model(
        'wamv',
        'wam-v',
        [
            spawn_x,
            spawn_y,
            0.0,
            0.0,
            0.0,
            spawn_yaw,
        ],
    )
    if robot_urdf:
        r1.set_urdf(robot_urdf)

    # =========================================================
    # Gazebo / VRX
    # =========================================================

    gazebo = []
    gazebo.extend(
        vrx_gz.launch.simulation(
            'sydney_regatta',
            headless=False,
            paused=False,
            extra_gz_args='',
        )
    )
    gazebo.extend(
        vrx_gz.launch.spawn(
            'full',
            'sydney_regatta',
            [r1],
        )
    )
    gazebo.extend(
        vrx_gz.launch.competition_bridges(
            'sydney_regatta',
            False,
        )
    )

    camera_move_to = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'gz',
                    'service',
                    '-s',
                    '/gui/move_to',
                    '--reqtype',
                    'gz.msgs.StringMsg',
                    '--reptype',
                    'gz.msgs.Boolean',
                    '--timeout',
                    '3000',
                    '--req',
                    'data: "wamv"',
                ],
                output='screen',
            ),
        ],
    )

    camera_follow = TimerAction(
        period=10.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'gz',
                    'service',
                    '-s',
                    '/gui/follow',
                    '--reqtype',
                    'gz.msgs.StringMsg',
                    '--reptype',
                    'gz.msgs.Boolean',
                    '--timeout',
                    '3000',
                    '--req',
                    'data: "wamv"',
                ],
                output='screen',
            ),
        ],
    )

    # =========================================================
    # Vehicle state
    # =========================================================

    vehicle_state = Node(
        package='platoon_state',
        executable='vehicle_state',
        name='vehicle_state_node',
        output='screen',
        parameters=[
            state_config,
            {
                'use_sim_time': True
            }
        ]
    )

    # =========================================================
    # Planner
    # =========================================================

    # =========================================================
    # External planning-team Dubins pipeline
    #
    # This replaces the control repository's trajectory planner.
    # The planning package is the only publisher of:
    #     /planner/reference_path
    # =========================================================

    planner_launch_file = os.path.join(
        get_package_share_directory('planner'),
        'launch',
        'dubins_pipeline.launch.py',
    )

    planner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            planner_launch_file
        )
    )

    # =========================================================
    # Logger
    # =========================================================

    logger = Node(
        package='platoon_monitor',
        executable='trajectory_logger',
        name='trajectory_logger',
        output='screen',

        parameters=[
            {
                'output_file':
                    output_file,

                'lookahead_distance':
                    3.0,

                'target_speed':
                    1.0,

                'controller_state_topic':
                    '/experiment/controller_state',

                'success_topic':
                    '/experiment/success',

                'use_sim_time':
                    True,
            }
        ]
    )
    # =========================================================
    # Optional live map
    # =========================================================

    live_map = Node(
        package='platoon_monitor',
        executable='live_map',
        name='live_map',
        output='screen',

        condition=IfCondition(
            LaunchConfiguration(
                'show_live_map'
            )
        ),

        parameters=[
            {
                'refresh_period':
                    5.0,

                'path_topic':
                    '/planner/reference_path',

                'robot_topics':
                    [
                        '/wamv/state/vehicle'
                    ],

                'use_sim_time':
                    True,
            }
        ]
    )

    # =========================================================
    # PID controller
    #
    # Start immediately. The controller itself waits until
    # VehicleState and a valid reference path are available.
    # =========================================================

    controller = Node(
        package='platoon_control',
        executable='leader_pid_controller',
        name='leader_pid_controller',
        output='screen',
        parameters=[
            controller_config,
            {
                'use_sim_time': True
            }
        ]
    )

    # =========================================================
    # Experiment summary
    # =========================================================

    print(
        '\n'
        '============================================\n'
        ' VRX LEADER TRACKING EXPERIMENT\n'
        '============================================\n'
        f' Trajectory         : {trajectory_type}\n'
        f' Simulation profile : {simulation_profile}\n'
        f' Live map           : {show_live_map}\n'
        f' Run name           : {run_name}\n'
        f' CSV output         : {output_file}\n'
        ' Use sim time       : True\n'
        '============================================\n'
    )

    return [
        *gazebo,
        vehicle_state,
        planner,
        logger,
        live_map,
        controller,
        camera_move_to,
        camera_follow,
    ]


def generate_launch_description():

    # =========================================================
    # Gazebo resource paths
    # =========================================================

    wamv_description_resource = os.path.join(
        get_package_prefix(
            'wamv_description'
        ),
        'share'
    )

    wamv_gazebo_resource = os.path.join(
        get_package_prefix(
            'wamv_gazebo'
        ),
        'share'
    )

    vrx_gz_resource = os.path.join(
        get_package_prefix(
            'vrx_gz'
        ),
        'share'
    )

    platoon_bringup_resource = os.path.join(
        get_package_prefix(
            'platoon_bringup'
        ),
        'share'
    )

    existing_resource_path = os.environ.get(
        'GZ_SIM_RESOURCE_PATH',
        ''
    )

    resource_paths = [
        wamv_description_resource,
        wamv_gazebo_resource,
        vrx_gz_resource,
        platoon_bringup_resource,
    ]

    if existing_resource_path:
        resource_paths.append(
            existing_resource_path
        )

    resource_path = ':'.join(
        resource_paths
    )

    set_gz_resources = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=resource_path
    )

    # =========================================================
    # Launch arguments
    # =========================================================

    trajectory_argument = DeclareLaunchArgument(
        'trajectory_type',
        default_value='straight',
        description=(
            'Reference trajectory: '
            'straight or curved'
        )
    )

    run_name_argument = DeclareLaunchArgument(
        'run_name',
        default_value='leader_tracking_run',
        description=(
            'CSV filename without .csv'
        )
    )

    simulation_profile_argument = DeclareLaunchArgument(
        'simulation_profile',
        default_value='light',
        description=(
            'Simulation profile: '
            'full or light'
        )
    )

    show_live_map_argument = DeclareLaunchArgument(
        'show_live_map',
        default_value='false',
        description=(
            'Display the live Sydney map'
        )
    )

    results_dir_argument = DeclareLaunchArgument(
        'results_dir',

        default_value=os.path.expanduser(
    '~/vrx_ws/src/'
    'planner/results'
    ),

        description=(
            'Directory for experiment CSV files'
        )
    )

    # =========================================================
    # Complete launch description
    # =========================================================

    return LaunchDescription([
        trajectory_argument,
        run_name_argument,
        simulation_profile_argument,
        show_live_map_argument,
        results_dir_argument,

        set_gz_resources,

        OpaqueFunction(
            function=launch_experiment
        ),
    ])
