import csv

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


CSV_PATH = (
    '/home/bot/vrx_ws/src/planner/'
    'planner/ky_py_pkg/output/'
    'sydney_coverage_dubins_path.csv'
)


class WaypointArrayDubins(Node):

    def __init__(self):
        super().__init__('waypoint_array_dubins')

        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/planner/waypoints_dubins',
            10
        )

        self.path_array = self.load_csv()

        self.get_logger().info(
            f'Loaded {len(self.path_array)} Dubins waypoints'
        )

        self.timer = self.create_timer(
            1.0,
            self.publish_path
        )

    def load_csv(self):
        path_array = []

        try:
            with open(CSV_PATH, 'r', newline='') as csv_file:
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
                f'CSV file not found: {CSV_PATH}'
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