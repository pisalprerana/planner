#!/usr/bin/env python3

import csv
import math
import os

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

        sample_rate = float(self.get_parameter(
            'sample_rate'
        ).value)
        if not math.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError('sample_rate must be a finite value greater than zero.')

        # ---------------------------------------------------------
        # Create output directory
        # ---------------------------------------------------------

        output_directory = os.path.dirname(output_file)

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        # ---------------------------------------------------------
        # Data storage
        # ---------------------------------------------------------

        self.r1_state = None
        self.r2_state = None
        self.r3_state = None
        self.reference_path = None
        self.r2_path_gap = None
        self.r3_path_gap = None
        self.start_time = self.get_clock().now()
        self.waiting_for_states_logged = False
        self.waiting_for_path_logged = False

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------

        self.create_subscription(
            VehicleState,
            '/r1/vehicle_state',
            self.r1_state_callback,
            10
        )

        self.create_subscription(
            VehicleState,
            '/r2/vehicle_state',
            self.r2_state_callback,
            10
        )

        self.create_subscription(
            VehicleState,
            '/r3/vehicle_state',
            self.r3_state_callback,
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
            '/planner/r2/path_gap',
            self.r2_path_gap_callback,
            10
        )

        self.create_subscription(
            Float64,
            '/planner/r3/path_gap',
            self.r3_path_gap_callback,
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
            'r1_x_m',
            'r1_y_m',
            'r1_speed_mps',
            'r1_yaw_rad',
            'r2_x_m',
            'r2_y_m',
            'r2_speed_mps',
            'r2_yaw_rad',
            'r3_x_m',
            'r3_y_m',
            'r3_speed_mps',
            'r3_yaw_rad',
            'reference_x_m',
            'reference_y_m',
            'leader_tracking_error_m',
            'd12_euclidean_m',
            'd23_euclidean_m',
            'e12_euclidean_m',
            'e23_euclidean_m',
            'r2_path_gap_m',
            'r3_path_gap_m',
            'r2_path_gap_error_m',
            'r3_path_gap_error_m',
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
            'Three-boat trajectory logger started.'
        )

        self.get_logger().info(
            f'CSV output: {output_file}'
        )

        self.get_logger().info(
            f'Sample rate: {sample_rate:g} Hz'
        )

        self.get_logger().info(
            'Waiting for R1, R2, and R3 vehicle states.'
        )
        self.waiting_for_states_logged = True

        self.get_logger().info(
            'Waiting for a non-empty /planner/reference_path.'
        )
        self.waiting_for_path_logged = True

    # =============================================================
    # Callbacks
    # =============================================================

    def r1_state_callback(self, msg):
        self.r1_state = msg

    def r2_state_callback(self, msg):
        self.r2_state = msg

    def r3_state_callback(self, msg):
        self.r3_state = msg

    def reference_path_callback(self, msg):
        self.reference_path = msg

    def r2_path_gap_callback(self, msg):
        self.r2_path_gap = float(msg.data)

    def r3_path_gap_callback(self, msg):
        self.r3_path_gap = float(msg.data)

    # =============================================================
    # Find closest reference point
    # =============================================================

    def get_reference_position(self):

        if self.r1_state is None:
            return None, None

        if self.reference_path is None:
            return None, None

        if len(self.reference_path.poses) == 0:
            return None, None

        current_x = self.r1_state.x
        current_y = self.r1_state.y

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

        if any(
            state is None
            for state in (
                self.r1_state,
                self.r2_state,
                self.r3_state,
            )
        ):
            if not self.waiting_for_states_logged:
                self.get_logger().info(
                    'Waiting for R1, R2, and R3 vehicle states.'
                )
                self.waiting_for_states_logged = True
            return

        self.waiting_for_states_logged = False

        if (
            self.reference_path is None
            or not self.reference_path.poses
        ):
            if not self.waiting_for_path_logged:
                self.get_logger().info(
                    'Waiting for a non-empty /planner/reference_path.'
                )
                self.waiting_for_path_logged = True
            return

        self.waiting_for_path_logged = False

        reference_x, reference_y = self.get_reference_position()

        if reference_x is None:
            return

        r1_x = float(self.r1_state.x)
        r1_y = float(self.r1_state.y)
        r2_x = float(self.r2_state.x)
        r2_y = float(self.r2_state.y)
        r3_x = float(self.r3_state.x)
        r3_y = float(self.r3_state.y)

        leader_tracking_error = math.hypot(
            r1_x - reference_x,
            r1_y - reference_y,
        )
        d12 = math.hypot(r2_x - r1_x, r2_y - r1_y)
        d23 = math.hypot(r3_x - r2_x, r3_y - r2_y)
        desired_distance = 5.0
        e12 = d12 - desired_distance
        e23 = d23 - desired_distance

        r2_path_gap = (
            float(self.r2_path_gap)
            if self.r2_path_gap is not None
            else float('nan')
        )
        r3_path_gap = (
            float(self.r3_path_gap)
            if self.r3_path_gap is not None
            else float('nan')
        )
        r2_path_gap_error = (
            r2_path_gap - desired_distance
            if math.isfinite(r2_path_gap)
            else float('nan')
        )
        r3_path_gap_error = (
            r3_path_gap - desired_distance
            if math.isfinite(r3_path_gap)
            else float('nan')
        )

        now = self.get_clock().now()

        time_s = (
            now - self.start_time
        ).nanoseconds / 1e9

        self.writer.writerow([
            f'{time_s:.3f}',
            f'{r1_x:.6f}',
            f'{r1_y:.6f}',
            f'{self.r1_state.speed:.6f}',
            f'{self.r1_state.body_yaw:.6f}',
            f'{r2_x:.6f}',
            f'{r2_y:.6f}',
            f'{self.r2_state.speed:.6f}',
            f'{self.r2_state.body_yaw:.6f}',
            f'{r3_x:.6f}',
            f'{r3_y:.6f}',
            f'{self.r3_state.speed:.6f}',
            f'{self.r3_state.body_yaw:.6f}',
            f'{reference_x:.6f}',
            f'{reference_y:.6f}',
            f'{leader_tracking_error:.6f}',
            f'{d12:.6f}',
            f'{d23:.6f}',
            f'{e12:.6f}',
            f'{e23:.6f}',
            f'{r2_path_gap:.6f}',
            f'{r3_path_gap:.6f}',
            f'{r2_path_gap_error:.6f}',
            f'{r3_path_gap_error:.6f}',
        ])

        self.csv_file.flush()

    # =============================================================
    # Shutdown
    # =============================================================

    def destroy_node(self):

        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.flush()
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