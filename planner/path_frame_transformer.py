"""Transform world waypoint arrays into the live boat frame."""

import math

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener


class WaypointArrayTransformer(Node):
    """Continuously transform waypoint coordinates into the boat frame."""

    def __init__(self):
        super().__init__("waypoint_array_transformer")

        self.declare_parameter(
            "input_topic",
            "/planner/waypoints_dubins",
        )
        self.declare_parameter(
            "output_topic",
            "/planner/waypoints_dubins_local",
        )
        self.declare_parameter("source_frame", "world")
        self.declare_parameter(
            "target_frame",
            "wamv/wamv/base_link",
        )
        self.declare_parameter("publish_rate", 2.0)

        input_topic = self.get_parameter(
            "input_topic"
        ).value
        output_topic = self.get_parameter(
            "output_topic"
        ).value
        self.source_frame = self.get_parameter(
            "source_frame"
        ).value
        self.target_frame = self.get_parameter(
            "target_frame"
        ).value
        publish_rate = float(
            self.get_parameter("publish_rate").value
        )

        if publish_rate <= 0.0:
            raise ValueError(
                "publish_rate must be greater than zero"
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.subscription = self.create_subscription(
            Float64MultiArray,
            input_topic,
            self.waypoint_callback,
            10,
        )

        self.publisher = self.create_publisher(
            Float64MultiArray,
            output_topic,
            10,
        )

        self.world_waypoints = None
        self.waiting_for_transform_logged = False
        self.first_local_path_published = False

        self.timer = self.create_timer(
            1.0 / publish_rate,
            self.transform_and_publish,
        )

        self.get_logger().info(
            f"Waypoint input: {input_topic}"
        )
        self.get_logger().info(
            f"Source frame: {self.source_frame}"
        )
        self.get_logger().info(
            f"Target frame: {self.target_frame}"
        )
        self.get_logger().info(
            f"Local waypoint output: {output_topic}"
        )

    def waypoint_callback(self, message):
        """Store the newest flattened world-waypoint array."""

        if len(message.data) == 0:
            self.get_logger().warning(
                "Received an empty waypoint array"
            )
            return

        if len(message.data) % 4 != 0:
            self.get_logger().error(
                "Waypoint array length must be divisible by 4. "
                "Expected [id, x, y, yaw, id, x, y, yaw, ...]"
            )
            return

        self.world_waypoints = list(message.data)

        self.get_logger().info(
            f"Stored {len(self.world_waypoints) // 4} "
            "world waypoints"
        )

    def transform_and_publish(self):
        """Transform all stored points using the latest boat TF."""

        if self.world_waypoints is None:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )

        except TransformException as error:
            if not self.waiting_for_transform_logged:
                self.get_logger().warning(
                    f"Waiting for TF from "
                    f"'{self.source_frame}' to "
                    f"'{self.target_frame}': {error}"
                )
                self.waiting_for_transform_logged = True
            return

        if self.waiting_for_transform_logged:
            self.get_logger().info(
                f"TF available: {self.source_frame} "
                f"to {self.target_frame}"
            )
            self.waiting_for_transform_logged = False

        local_message = Float64MultiArray()

        for index in range(
            0,
            len(self.world_waypoints),
            4,
        ):
            waypoint_id = self.world_waypoints[index]
            world_x = self.world_waypoints[index + 1]
            world_y = self.world_waypoints[index + 2]
            world_yaw = self.world_waypoints[index + 3]

            world_point = PointStamped()
            world_point.header.frame_id = self.source_frame
            world_point.header.stamp = (
                self.get_clock().now().to_msg()
            )
            world_point.point.x = world_x
            world_point.point.y = world_y
            world_point.point.z = 0.0

            local_point = do_transform_point(
                world_point,
                transform,
            )

            heading_point = PointStamped()
            heading_point.header.frame_id = self.source_frame
            heading_point.header.stamp = world_point.header.stamp
            heading_point.point.x = world_x + math.cos(world_yaw)
            heading_point.point.y = world_y + math.sin(world_yaw)
            heading_point.point.z = 0.0

            local_heading_point = do_transform_point(
                heading_point,
                transform,
            )
            local_yaw = math.atan2(
                local_heading_point.point.y - local_point.point.y,
                local_heading_point.point.x - local_point.point.x,
            )

            local_message.data.extend(
                [
                    waypoint_id,
                    local_point.point.x,
                    local_point.point.y,
                    local_yaw,
                ]
            )

        self.publisher.publish(local_message)

        if not self.first_local_path_published:
            self.get_logger().info(
                f"Published "
                f"{len(local_message.data) // 4} "
                "waypoints in the boat frame"
            )
            self.first_local_path_published = True


def main(args=None):
    rclpy.init(args=args)

    node = WaypointArrayTransformer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
