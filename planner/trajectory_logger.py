#!/usr/bin/env python3

import csv
import os
from datetime import datetime

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from nav_msgs.msg import Path
from platoon_interfaces.msg import VehicleState


class TrajectoryLogger(Node):

    def __init__(self):
        super().__init__('trajectory_logger')

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------

        self.declare_parameter('output_file', 'results/trajectory.csv')
        self.declare_parameter('sample_rate', 10.0)

        output_file = self.get_parameter(
            'output_file'
        ).get_parameter_value().string_value

        sample_rate = self.get_parameter(
            'sample_rate'
        ).get_parameter_value().double_value

        # ---------------------------------------------------------
        # Create output directory
        # ---------------------------------------------------------

        output_directory = os.path.dirname(output_file)

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        # ---------------------------------------------------------
        # Data storage
        # ---------------------------------------------------------

        self.vehicle_state = None
        self.reference_path = None
        self.left_thrust = 0.0
        self.right_thrust = 0.0

        self.start_time = self.get_clock().now()

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------

        self.create_subscription(
            VehicleState,
            '/wamv/state/vehicle',
            self.vehicle_state_callback,
            10
        )

        self.create_subscription(
            Path,
            '/planner/reference_path',
            self.reference_path_callback,
            10
        )

        self.create_subscription(
            Float64,
            '/wamv/thrusters/left/thrust',
            self.left_thrust_callback,
            10
        )

        self.create_subscription(
            Float64,
            '/wamv/thrusters/right/thrust',
            self.right_thrust_callback,
            10
        )

        # ---------------------------------------------------------
        # CSV file
        # ---------------------------------------------------------

        self.csv_file = open(
            output_file,
            'w',
            newline=''
        )

        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            'time_s',
            'x_m',
            'y_m',
            'vx_mps',
            'vy_mps',
            'speed_mps',
            'course_angle_rad',
            'body_yaw_rad',
            'reference_x_m',
            'reference_y_m',
            'left_thrust',
            'right_thrust'
        ])

        self.csv_file.flush()

        # ---------------------------------------------------------
        # Timer
        # ---------------------------------------------------------

        timer_period = 1.0 / sample_rate

        self.timer = self.create_timer(
            timer_period,
            self.log_data
        )

        self.get_logger().info(
            f'Trajectory logger started.'
        )

        self.get_logger().info(
            f'Logging data to: {output_file}'
        )

    # =============================================================
    # Callbacks
    # =============================================================

    def vehicle_state_callback(self, msg):
        self.vehicle_state = msg

    def reference_path_callback(self, msg):
        self.reference_path = msg

    def left_thrust_callback(self, msg):
        self.left_thrust = msg.data

    def right_thrust_callback(self, msg):
        self.right_thrust = msg.data

    # =============================================================
    # Find closest reference point
    # =============================================================

    def get_reference_position(self):

        if self.vehicle_state is None:
            return None, None

        if self.reference_path is None:
            return None, None

        if len(self.reference_path.poses) == 0:
            return None, None

        current_x = self.vehicle_state.x
        current_y = self.vehicle_state.y

        closest_pose = None
        closest_distance = float('inf')

        for pose in self.reference_path.poses:

            ref_x = pose.pose.position.x
            ref_y = pose.pose.position.y

            dx = ref_x - current_x
            dy = ref_y - current_y

            distance_squared = dx * dx + dy * dy

            if distance_squared < closest_distance:
                closest_distance = distance_squared
                closest_pose = pose

        if closest_pose is None:
            return None, None

        return (
            closest_pose.pose.position.x,
            closest_pose.pose.position.y
        )

    # =============================================================
    # Log one sample
    # =============================================================

    def log_data(self):

        if self.vehicle_state is None:
            return

        reference_x, reference_y = self.get_reference_position()

        if reference_x is None:
            return

        now = self.get_clock().now()

        time_s = (
            now - self.start_time
        ).nanoseconds / 1e9

        self.writer.writerow([
            f'{time_s:.3f}',

            f'{self.vehicle_state.x:.6f}',
            f'{self.vehicle_state.y:.6f}',

            f'{self.vehicle_state.vx:.6f}',
            f'{self.vehicle_state.vy:.6f}',

            f'{self.vehicle_state.speed:.6f}',

            f'{self.vehicle_state.course_angle:.6f}',
            f'{self.vehicle_state.body_yaw:.6f}',

            f'{reference_x:.6f}',
            f'{reference_y:.6f}',

            f'{self.left_thrust:.6f}',
            f'{self.right_thrust:.6f}'
        ])

        self.csv_file.flush()

    # =============================================================
    # Shutdown
    # =============================================================

    def destroy_node(self):

        if hasattr(self, 'csv_file'):
            self.csv_file.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = TrajectoryLogger()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()