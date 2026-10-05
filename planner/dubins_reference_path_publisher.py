"""Publish a Dubins ENU trajectory as a NED nav_msgs/Path."""

import csv
import math
from pathlib import Path as FilePath

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
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

        if publish_rate <= 0.0:
            raise ValueError(
                'publish_rate must be greater than zero.'
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

        self.path_message = self.load_path()

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
