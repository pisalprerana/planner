"""Broadcast world-to-base TF using Sydney GPS and WAM-V IMU."""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix
from tf2_ros import TransformBroadcaster


# WGS84 ellipsoid
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(latitude, longitude, altitude):
    """Convert WGS84 latitude/longitude/altitude to ECEF."""

    latitude = math.radians(latitude)
    longitude = math.radians(longitude)

    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    sin_longitude = math.sin(longitude)
    cos_longitude = math.cos(longitude)

    radius = WGS84_A / math.sqrt(
        1.0 - WGS84_E2 * sin_latitude**2
    )

    x = (
        radius + altitude
    ) * cos_latitude * cos_longitude

    y = (
        radius + altitude
    ) * cos_latitude * sin_longitude

    z = (
        radius * (1.0 - WGS84_E2) + altitude
    ) * sin_latitude

    return x, y, z


def rotate_vector(vector, quaternion):
    """Rotate an XYZ vector using an XYZW quaternion."""

    vx, vy, vz = vector
    qx, qy, qz, qw = quaternion

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)

    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)

    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    return (
        r00 * vx + r01 * vy + r02 * vz,
        r10 * vx + r11 * vy + r12 * vz,
        r20 * vx + r21 * vy + r22 * vz,
    )


class GpsImuTfBroadcaster(Node):
    """Create the missing live world-to-boat transform."""

    def __init__(self):
        super().__init__("gps_imu_tf_broadcaster")

        self.declare_parameter(
            "gps_topic",
            "/wamv/sensors/gps/gps/fix",
        )
        self.declare_parameter(
            "imu_topic",
            "/wamv/sensors/imu/imu/data",
        )
        self.declare_parameter("world_frame", "world")
        self.declare_parameter(
            "base_frame",
            "wamv/wamv/base_link",
        )

        self.declare_parameter(
            "origin_latitude",
            -33.724223,
        )
        self.declare_parameter(
            "origin_longitude",
            150.679736,
        )
        self.declare_parameter("origin_altitude", 0.0)

        self.declare_parameter("gps_offset_x", -0.85)
        self.declare_parameter("gps_offset_y", 0.0)
        self.declare_parameter("gps_offset_z", 1.30)

        gps_topic = self.get_parameter("gps_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        self.world_frame = self.get_parameter(
            "world_frame"
        ).value
        self.base_frame = self.get_parameter(
            "base_frame"
        ).value

        self.origin_latitude = float(
            self.get_parameter("origin_latitude").value
        )
        self.origin_longitude = float(
            self.get_parameter("origin_longitude").value
        )
        self.origin_altitude = float(
            self.get_parameter("origin_altitude").value
        )

        self.gps_offset = (
            float(self.get_parameter("gps_offset_x").value),
            float(self.get_parameter("gps_offset_y").value),
            float(self.get_parameter("gps_offset_z").value),
        )

        self.origin_ecef = geodetic_to_ecef(
            self.origin_latitude,
            self.origin_longitude,
            self.origin_altitude,
        )

        self.latest_gps_enu = None
        self.latest_orientation = None
        self.first_transform_logged = False

        self.tf_broadcaster = TransformBroadcaster(self)

        self.gps_subscription = self.create_subscription(
            NavSatFix,
            gps_topic,
            self.gps_callback,
            qos_profile_sensor_data,
        )

        self.imu_subscription = self.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"GPS input: {gps_topic}"
        )
        self.get_logger().info(
            f"IMU input: {imu_topic}"
        )
        self.get_logger().info(
            f"Broadcasting {self.world_frame} "
            f"to {self.base_frame}"
        )

    def gps_callback(self, message):
        """Convert GPS coordinates to Sydney Gazebo ENU."""

        if message.status.status < 0:
            self.get_logger().warning("GPS has no valid fix")
            return

        gps_ecef = geodetic_to_ecef(
            message.latitude,
            message.longitude,
            message.altitude,
        )

        dx = gps_ecef[0] - self.origin_ecef[0]
        dy = gps_ecef[1] - self.origin_ecef[1]
        dz = gps_ecef[2] - self.origin_ecef[2]

        latitude = math.radians(self.origin_latitude)
        longitude = math.radians(self.origin_longitude)

        east = (
            -math.sin(longitude) * dx
            + math.cos(longitude) * dy
        )

        north = (
            -math.sin(latitude) * math.cos(longitude) * dx
            - math.sin(latitude) * math.sin(longitude) * dy
            + math.cos(latitude) * dz
        )

        up = (
            math.cos(latitude) * math.cos(longitude) * dx
            + math.cos(latitude) * math.sin(longitude) * dy
            + math.sin(latitude) * dz
        )

        self.latest_gps_enu = (east, north, up)
        self.publish_transform(message.header.stamp)

    def imu_callback(self, message):
        """Store the latest boat orientation."""

        quaternion = (
            message.orientation.x,
            message.orientation.y,
            message.orientation.z,
            message.orientation.w,
        )

        norm = math.sqrt(
            sum(component**2 for component in quaternion)
        )

        if norm == 0.0:
            self.get_logger().warning(
                "Ignoring invalid zero IMU quaternion"
            )
            return

        self.latest_orientation = tuple(
            component / norm
            for component in quaternion
        )

        self.publish_transform(message.header.stamp)

    def publish_transform(self, stamp):
        """Publish world-to-base when GPS and IMU are available."""

        if (
            self.latest_gps_enu is None
            or self.latest_orientation is None
        ):
            return

        offset_world = rotate_vector(
            self.gps_offset,
            self.latest_orientation,
        )

        base_x = self.latest_gps_enu[0] - offset_world[0]
        base_y = self.latest_gps_enu[1] - offset_world[1]
        base_z = self.latest_gps_enu[2] - offset_world[2]

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.world_frame
        transform.child_frame_id = self.base_frame

        transform.transform.translation.x = base_x
        transform.transform.translation.y = base_y
        transform.transform.translation.z = base_z

        (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ) = self.latest_orientation

        self.tf_broadcaster.sendTransform(transform)

        if not self.first_transform_logged:
            self.get_logger().info(
                f"First boat position: "
                f"x={base_x:.3f}, "
                f"y={base_y:.3f}, "
                f"z={base_z:.3f}"
            )
            self.first_transform_logged = True


def main(args=None):
    rclpy.init(args=args)

    node = GpsImuTfBroadcaster()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
