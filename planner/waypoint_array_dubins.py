import csv
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class WaypointArrayDubins(Node):

    def __init__(self):
        super().__init__('waypoint_array_dubins')

        output_dir = Path(__file__).resolve().parent / 'output'
        self.declare_parameter(
            'csv_file',
            'sydney_coverage_dubins_path.csv',
        )
        self.declare_parameter(
            'output_topic',
            '/planner/waypoints_dubins',
        )
        self.declare_parameter('publish_rate', 1.0)

        csv_file = Path(
            self.get_parameter('csv_file').value
        ).expanduser()
        if csv_file.is_absolute():
            self.csv_path = csv_file
        else:
            package_share = Path(
                get_package_share_directory('planner')
            )
            self.csv_path = package_share / 'output' / csv_file

        output_topic = self.get_parameter('output_topic').value
        publish_rate = float(self.get_parameter('publish_rate').value)

        if publish_rate <= 0.0:
            raise ValueError('publish_rate must be greater than zero')

        self.publisher = self.create_publisher(
            Float64MultiArray,
            output_topic,
            10
        )

        self.path_array = self.load_csv()

        self.get_logger().info(
            f'Loaded {len(self.path_array)} Dubins waypoints '
            f'from {self.csv_path}'
        )

        self.timer = self.create_timer(
            1.0 / publish_rate,
            self.publish_path
        )

    def load_csv(self):
        path_array = []

        try:
            with self.csv_path.open('r', newline='') as csv_file:
                reader = csv.DictReader(csv_file)

                for row in reader:
                    waypoint_id = int(row['waypoint_id'])
                    x = float(row['x'])
                    y = float(row['y'])
                    yaw = float(row['yaw'])

                    path_array.append(
                        [waypoint_id, x, y, yaw]
                    )

        except FileNotFoundError:
            self.get_logger().error(
                f'CSV file not found: {self.csv_path}'
            )

        except (KeyError, ValueError) as error:
            self.get_logger().error(
                f'Invalid CSV data: {error}'
            )

        return path_array

    def publish_path(self):
        if not self.path_array:
            return

        msg = Float64MultiArray()

        for waypoint in self.path_array:
            msg.data.extend(waypoint)

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = WaypointArrayDubins()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()