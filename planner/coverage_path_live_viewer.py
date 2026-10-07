
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import rclpy

from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener


class CoveragePathLiveViewer(Node):

    def __init__(self):

        super().__init__("coverage_path_live_viewer")

        self.declare_parameter("path_csv", "")

        path_csv = self.get_parameter("path_csv").value

        if not path_csv:
            raise ValueError(
                "path_csv parameter was not provided"
            )

        self.path_csv = Path(path_csv)

        if not self.path_csv.exists():
            raise FileNotFoundError(
                f"Coverage path CSV not found: {self.path_csv}"
            )

        # Planned coverage path.
        self.path_x = []
        self.path_y = []

        self.load_path()

        # TF for live WAM-V position.
        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # Actual path followed by the WAM-V.
        self.trail_x = []
        self.trail_y = []

        # Previous position/time used to calculate speed.
        self.previous_x = None
        self.previous_y = None
        self.previous_time = None

        self.current_speed = 0.0

        self.figure, self.ax = plt.subplots(
            figsize=(14, 9)
        )

        # Planned coverage path.
        self.ax.plot(
            self.path_x,
            self.path_y,
            linewidth=2.0,
            label="Coverage path",
        )

        # Actual path followed by the WAM-V.
        self.boat_trail, = self.ax.plot(
            [],
            [],
            linewidth=2.0,
            label="Actual path",
        )

        # Current WAM-V position.
        self.boat_dot, = self.ax.plot(
            [],
            [],
            marker="o",
            markersize=9,
            linestyle="None",
            label="WAM-V",
        )

        # Live position and speed text.
        self.boat_text = self.ax.text(
            0.02,
            0.98,
            "Boat position: waiting for TF...",
            transform=self.ax.transAxes,
            verticalalignment="top",
        )

        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")

        self.ax.set_title(
            "Sydney Regatta — Live Coverage Path Tracking"
        )

        self.ax.grid(True, alpha=0.25)

        self.ax.legend(loc="best")

        self.ax.set_aspect(
            "equal",
            adjustable="box",
        )

        self.figure.show()

        self.timer = self.create_timer(
            0.1,
            self.update_boat_position,
        )

        self.get_logger().info(
            f"Loaded {len(self.path_x)} coverage waypoints"
        )

    def load_path(self):

        with self.path_csv.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                self.path_x.append(
                    float(row["x"])
                )

                self.path_y.append(
                    float(row["y"])
                )

        if not self.path_x:
            raise ValueError(
                "Coverage path CSV contains no waypoints"
            )

    def update_boat_position(self):

        try:

            transform = self.tf_buffer.lookup_transform(
                "world",
                "wamv/wamv/base_link",
                rclpy.time.Time(),
            )

        except TransformException:

            return

        x = transform.transform.translation.x
        y = transform.transform.translation.y

        current_time = time.monotonic()

        # Calculate speed from consecutive TF positions.
        if (
            self.previous_x is not None
            and self.previous_y is not None
            and self.previous_time is not None
        ):

            delta_x = x - self.previous_x
            delta_y = y - self.previous_y
            delta_time = current_time - self.previous_time

            if delta_time > 0.0:

                distance = math.sqrt(
                    delta_x ** 2
                    + delta_y ** 2
                )

                self.current_speed = (
                    distance / delta_time
                )

        # Save the current position/time for the next update.
        self.previous_x = x
        self.previous_y = y
        self.previous_time = current_time

        # Add current position to the actual travelled path.
        self.trail_x.append(x)
        self.trail_y.append(y)

        # Update current WAM-V position.
        self.boat_dot.set_data(
            [x],
            [y],
        )

        # Update actual travelled path.
        self.boat_trail.set_data(
            self.trail_x,
            self.trail_y,
        )

        # Update live information.
        self.boat_text.set_text(
            f"Boat position:\n"
            f"X: {x:.2f} m\n"
            f"Y: {y:.2f} m\n"
            f"Speed: {self.current_speed:.2f} m/s"
        )

        self.figure.canvas.draw_idle()

        self.figure.canvas.flush_events()


def main(args=None):

    rclpy.init(args=args)

    node = CoveragePathLiveViewer()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()