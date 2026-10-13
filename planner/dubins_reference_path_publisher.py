"""Publish a Dubins ENU trajectory as a NED nav_msgs/Path."""

import csv
import math
from pathlib import Path as FilePath

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from platoon_interfaces.msg import VehicleState
from std_msgs.msg import Float64
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


def wrap_angle(angle):
    """Wrap an angle to the interval [-pi, pi]."""

    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def enu_to_ned(x_enu, y_enu, yaw_enu):
    """Convert a 2D Gazebo ENU pose into the controller's NED frame."""

    x_ned = y_enu
    y_ned = x_enu
    yaw_ned = wrap_angle(
        (math.pi / 2.0) - yaw_enu
    )

    return x_ned, y_ned, yaw_ned


class DubinsReferencePathPublisher(Node):
    """Load the Dubins CSV and publish a controller-compatible path."""

    def __init__(self):
        super().__init__(
            'dubins_reference_path_publisher'
        )

        self.declare_parameter(
            'csv_file',
            'sydney_coverage_dubins_path.csv',
        )
        self.declare_parameter(
            'output_topic',
            '/planner/reference_path',
        )
        self.declare_parameter(
            'output_frame',
            'world_ned',
        )
        self.declare_parameter(
            'publish_rate',
            1.0,
        )
        self.declare_parameter(
            'vehicle_state_topic',
            '/r1/vehicle_state',
        )
        self.declare_parameter(
            'preview_distance',
            20.0,
        )
        self.declare_parameter(
            'curvature_sample_distance',
            2.0,
        )
        self.declare_parameter(
            'preview_max_speed',
            3.0,
        )
        self.declare_parameter(
            'preview_decel',
            0.2646349942459316,
        )
        self.declare_parameter(
            'preview_max_lateral_accel',
            0.30,
        )

        csv_file = str(
            self.get_parameter('csv_file').value
        )
        output_topic = str(
            self.get_parameter('output_topic').value
        )
        self.output_frame = str(
            self.get_parameter('output_frame').value
        )
        publish_rate = float(
            self.get_parameter('publish_rate').value
        )
        vehicle_state_topic = str(
            self.get_parameter(
                'vehicle_state_topic'
            ).value
        )
        self.preview_distance = float(
            self.get_parameter(
                'preview_distance'
            ).value
        )
        self.curvature_sample_distance = float(
            self.get_parameter(
                'curvature_sample_distance'
            ).value
        )
        self.preview_max_speed = float(
            self.get_parameter(
                'preview_max_speed'
            ).value
        )
        self.preview_decel = float(
            self.get_parameter(
                'preview_decel'
            ).value
        )
        self.preview_max_lateral_accel = float(
            self.get_parameter(
                'preview_max_lateral_accel'
            ).value
        )

        if publish_rate <= 0.0:
            raise ValueError(
                'publish_rate must be greater than zero.'
            )

        if self.preview_distance <= 0.0:
            raise ValueError(
                'preview_distance must be greater than zero.'
            )

        if self.curvature_sample_distance <= 0.0:
            raise ValueError(
                'curvature_sample_distance must be greater '
                'than zero.'
            )

        if self.preview_max_speed <= 0.0:
            raise ValueError(
                'preview_max_speed must be greater than zero.'
            )

        if self.preview_decel <= 0.0:
            raise ValueError(
                'preview_decel must be greater than zero.'
            )

        if self.preview_max_lateral_accel <= 0.0:
            raise ValueError(
                'preview_max_lateral_accel must be greater '
                'than zero.'
            )

        self.csv_path = FilePath(csv_file)

        if not self.csv_path.is_absolute():
            package_share = FilePath(
                get_package_share_directory('planner')
            )
            self.csv_path = (
                package_share
                / 'output'
                / self.csv_path
            )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.publisher = self.create_publisher(
            Path,
            output_topic,
            qos,
        )

        self.preview_curvature_publisher = (
            self.create_publisher(
                Float64,
                '/planner/preview_curvature',
                10,
            )
        )

        self.preview_curvature_distance_publisher = (
            self.create_publisher(
                Float64,
                '/planner/preview_curvature_distance',
                10,
            )
        )

        self.preview_curve_speed_publisher = (
            self.create_publisher(
                Float64,
                '/planner/preview_curve_speed',
                10,
            )
        )

        self.recommended_speed_publisher = (
            self.create_publisher(
                Float64,
                '/planner/recommended_speed',
                10,
            )
        )

        self.path_message = self.load_path()

        # ----------------------------------------------------
        # Live R1 progress on the known Dubins reference path
        # ----------------------------------------------------
        self.current_vehicle_state = None
        self.current_path_index = 0
        self.preview_path_index = 0
        self.have_path_progress = False

        self.path_points = [
            (
                pose.pose.position.x,
                pose.pose.position.y,
            )
            for pose in self.path_message.poses
        ]

        self.path_distances = [0.0]

        for index in range(
            1,
            len(self.path_points),
        ):
            previous = self.path_points[index - 1]
            current = self.path_points[index]

            segment_length = math.hypot(
                current[0] - previous[0],
                current[1] - previous[1],
            )

            self.path_distances.append(
                self.path_distances[-1]
                + segment_length
            )

        self.vehicle_state_subscription = (
            self.create_subscription(
                VehicleState,
                vehicle_state_topic,
                self.vehicle_state_callback,
                10,
            )
        )

        self.timer = self.create_timer(
            1.0 / publish_rate,
            self.publish_path,
        )

        self.get_logger().info(
            f'Loaded {len(self.path_message.poses)} '
            f'Dubins waypoints from {self.csv_path}'
        )
        self.get_logger().info(
            f'Publishing NED reference path on {output_topic}'
        )
        self.get_logger().info(
            f'Output frame: {self.output_frame}'
        )
        self.get_logger().info(
            f'Tracking R1 progress from '
            f'{vehicle_state_topic}'
        )
        self.get_logger().info(
            f'Preview distance: '
            f'{self.preview_distance:.1f} m'
        )
        self.get_logger().info(
            f'Curvature sample distance: '
            f'{self.curvature_sample_distance:.1f} m'
        )
        self.get_logger().info(
            f'Predictive speed limit: '
            f'{self.preview_max_speed:.2f} m/s'
        )
        self.get_logger().info(
            f'Predictive deceleration: '
            f'{self.preview_decel:.3f} m/s^2'
        )
        self.get_logger().info(
            f'Maximum lateral acceleration: '
            f'{self.preview_max_lateral_accel:.2f} m/s^2'
        )

    def vehicle_state_callback(self, msg):
        """Update R1 position and locate it on the Dubins path."""

        self.current_vehicle_state = msg

        if not self.path_points:
            return

        # On the first state message, search the complete route.
        if not self.have_path_progress:
            start_index = 0
            end_index = len(self.path_points)

        else:
            # Afterwards search mainly forward along the route.
            # A small backtrack allowance avoids index jitter while
            # preventing jumps to nearby parallel coverage passes.
            start_index = max(
                0,
                self.current_path_index - 40,
            )
            end_index = min(
                len(self.path_points),
                self.current_path_index + 500,
            )

        best_index = start_index
        best_distance_sq = float('inf')

        for index in range(
            start_index,
            end_index,
        ):
            px, py = self.path_points[index]

            dx = msg.x - px
            dy = msg.y - py

            distance_sq = (
                dx * dx
                + dy * dy
            )

            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_index = index

        self.current_path_index = best_index

        self.preview_path_index = (
            self.index_at_distance_ahead(
                self.current_path_index,
                self.preview_distance,
            )
        )

        if not self.have_path_progress:
            self.have_path_progress = True

            self.get_logger().info(
                'R1 acquired on Dubins path at '
                f'waypoint {best_index}, '
                f'cross-track distance '
                f'{math.sqrt(best_distance_sq):.2f} m'
            )

    def index_at_distance_ahead(
        self,
        start_index,
        distance_ahead,
    ):
        """Return the path index approximately distance_ahead ahead."""

        if not self.path_distances:
            return start_index

        start_index = max(
            0,
            min(
                start_index,
                len(self.path_distances) - 1,
            ),
        )

        target_distance = (
            self.path_distances[start_index]
            + max(0.0, distance_ahead)
        )

        for index in range(
            start_index,
            len(self.path_distances),
        ):
            if (
                self.path_distances[index]
                >= target_distance
            ):
                return index

        return len(self.path_distances) - 1

    @staticmethod
    def three_point_curvature(p0, p1, p2):
        """Return signed planar curvature through three points."""

        ax = p1[0] - p0[0]
        ay = p1[1] - p0[1]

        bx = p2[0] - p1[0]
        by = p2[1] - p1[1]

        cx = p2[0] - p0[0]
        cy = p2[1] - p0[1]

        a = math.hypot(ax, ay)
        b = math.hypot(bx, by)
        c = math.hypot(cx, cy)

        denominator = a * b * c

        if denominator <= 1e-9:
            return 0.0

        cross = (
            ax * cy
            - ay * cx
        )

        return (
            2.0 * cross
            / denominator
        )

    def preview_curvature(self):
        """Find the strongest curvature in the active preview window."""

        if (
            not self.have_path_progress
            or len(self.path_points) < 3
        ):
            return None

        start = self.current_path_index
        end = self.preview_path_index

        if end <= start:
            return None

        strongest_curvature = 0.0
        strongest_distance = 0.0

        index = start

        while index <= end:

            before_target = max(
                0.0,
                self.path_distances[index]
                - self.curvature_sample_distance,
            )

            after_target = min(
                self.path_distances[-1],
                self.path_distances[index]
                + self.curvature_sample_distance,
            )

            before = index

            while (
                before > 0
                and self.path_distances[before]
                > before_target
            ):
                before -= 1

            after = index

            while (
                after < len(self.path_distances) - 1
                and self.path_distances[after]
                < after_target
            ):
                after += 1

            if (
                before < index
                and index < after
            ):
                curvature = self.three_point_curvature(
                    self.path_points[before],
                    self.path_points[index],
                    self.path_points[after],
                )

                if (
                    abs(curvature)
                    > abs(strongest_curvature)
                ):
                    strongest_curvature = curvature
                    strongest_distance = (
                        self.path_distances[index]
                        - self.path_distances[start]
                    )

            index += 1

        return (
            strongest_curvature,
            strongest_distance,
        )

    def recommended_speed(
        self,
        curvature,
        distance_ahead,
    ):
        """Return curve speed and present recommended speed."""

        curvature_magnitude = abs(curvature)

        if curvature_magnitude <= 1e-6:
            curve_speed = self.preview_max_speed

        else:
            curve_speed = math.sqrt(
                self.preview_max_lateral_accel
                / curvature_magnitude
            )

            curve_speed = min(
                curve_speed,
                self.preview_max_speed,
            )

        # Determine the highest speed that still leaves enough
        # distance to decelerate to curve_speed before the
        # strongest upcoming curvature.
        speed_now = math.sqrt(
            max(
                0.0,
                curve_speed * curve_speed
                + 2.0
                * self.preview_decel
                * max(0.0, distance_ahead),
            )
        )

        speed_now = min(
            speed_now,
            self.preview_max_speed,
        )

        return (
            curve_speed,
            speed_now,
        )

    def load_path(self):
        """Read, validate and convert the CSV trajectory."""

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f'CSV file not found: {self.csv_path}'
            )

        path_message = Path()
        path_message.header.frame_id = self.output_frame

        required_columns = {
            'waypoint_id',
            'x',
            'y',
            'yaw',
        }

        expected_id = 0

        with self.csv_path.open(
            'r',
            newline='',
            encoding='utf-8',
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames is None:
                raise ValueError(
                    'The CSV file has no header.'
                )

            missing_columns = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing_columns:
                raise ValueError(
                    'CSV is missing columns: '
                    + ', '.join(sorted(missing_columns))
                )

            for row in reader:
                waypoint_id = int(row['waypoint_id'])
                x_enu = float(row['x'])
                y_enu = float(row['y'])
                yaw_enu = float(row['yaw'])

                values = (
                    x_enu,
                    y_enu,
                    yaw_enu,
                )

                if not all(
                    math.isfinite(value)
                    for value in values
                ):
                    raise ValueError(
                        f'Waypoint {waypoint_id} contains '
                        'a non-finite value.'
                    )

                if waypoint_id != expected_id:
                    raise ValueError(
                        f'Expected waypoint ID {expected_id}, '
                        f'but found {waypoint_id}.'
                    )

                expected_id += 1

                x_ned, y_ned, yaw_ned = enu_to_ned(
                    x_enu,
                    y_enu,
                    yaw_enu,
                )

                pose = PoseStamped()
                pose.header.frame_id = self.output_frame

                pose.pose.position.x = x_ned
                pose.pose.position.y = y_ned
                pose.pose.position.z = 0.0

                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = math.sin(
                    yaw_ned / 2.0
                )
                pose.pose.orientation.w = math.cos(
                    yaw_ned / 2.0
                )

                path_message.poses.append(pose)

        if not path_message.poses:
            raise ValueError(
                f'No waypoints were loaded from {self.csv_path}'
            )

        return path_message

    def publish_path(self):
        """Publish the complete NED reference trajectory."""

        now = self.get_clock().now().to_msg()

        self.path_message.header.stamp = now

        for pose in self.path_message.poses:
            pose.header.stamp = now

        self.publisher.publish(
            self.path_message
        )

        curvature_preview = self.preview_curvature()

        if curvature_preview is not None:
            curvature, distance_ahead = (
                curvature_preview
            )

            msg = Float64()
            msg.data = float(curvature)
            self.preview_curvature_publisher.publish(
                msg
            )

            msg = Float64()
            msg.data = float(distance_ahead)
            self.preview_curvature_distance_publisher.publish(
                msg
            )

            curve_speed, recommended_speed = (
                self.recommended_speed(
                    curvature,
                    distance_ahead,
                )
            )

            msg = Float64()
            msg.data = float(curve_speed)
            self.preview_curve_speed_publisher.publish(
                msg
            )

            msg = Float64()
            msg.data = float(recommended_speed)
            self.recommended_speed_publisher.publish(
                msg
            )


def main(args=None):
    rclpy.init(args=args)

    node = DubinsReferencePathPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
