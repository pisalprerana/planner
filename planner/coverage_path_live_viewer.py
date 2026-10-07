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

        # ---------------------------------------------------------
        # Planned coverage path
        # ---------------------------------------------------------

        self.path_x = []
        self.path_y = []

        self.load_path()

        # ---------------------------------------------------------
        # TF for live WAM-V position
        # ---------------------------------------------------------

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # ---------------------------------------------------------
        # Actual path followed by the WAM-V
        # ---------------------------------------------------------

        self.trail_x = []
        self.trail_y = []

        # Previous position/time for speed calculation
        self.previous_x = None
        self.previous_y = None
        self.previous_time = None

        self.current_speed = 0.0

        # ---------------------------------------------------------
        # Matplotlib figure
        # ---------------------------------------------------------

        self.figure, self.ax = plt.subplots(
            figsize=(14, 9)
        )

        # Planned coverage path
        self.coverage_line, = self.ax.plot(
            self.path_x,
            self.path_y,
            linewidth=2.0,
            label="Coverage path",
        )

        # Actual path followed by WAM-V
        self.boat_trail, = self.ax.plot(
            [],
            [],
            linewidth=2.5,
            label="Actual path",
        )

        # Current WAM-V position
        self.boat_dot, = self.ax.plot(
            [],
            [],
            marker="o",
            markersize=10,
            linestyle="None",
            label="WAM-V",
            zorder=10,
        )

        # ---------------------------------------------------------
        # Live information box
        # ---------------------------------------------------------

        self.boat_text = self.ax.text(
            0.98,
            0.98,
            "Boat position:\nWaiting for TF...",
            transform=self.ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=10,
            zorder=20,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="gray",
                alpha=0.85,
            ),
        )

        # ---------------------------------------------------------
        # Plot formatting
        # ---------------------------------------------------------

        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")

        self.ax.set_title(
            "Sydney Regatta — Live Coverage Path Tracking"
        )

        self.ax.grid(
            True,
            alpha=0.25,
        )

        self.ax.legend(
            loc="lower right"
        )

        self.ax.set_aspect(
            "equal",
            adjustable="box",
        )

        self.set_initial_limits()

        # Show Matplotlib window
        self.figure.show()

        # ---------------------------------------------------------
        # ROS timer
        # ---------------------------------------------------------

        self.timer = self.create_timer(
            0.1,
            self.update_boat_position,
        )

        self.get_logger().info(
            f"Loaded {len(self.path_x)} coverage waypoints"
        )

    # =============================================================
    # Load coverage CSV
    # =============================================================

    def load_path(self):

        with self.path_csv.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                # Keep the coverage CSV coordinates unchanged.
                #
                # Do NOT swap x and y here.
                # The viewer compares the coverage path with
                # the WAM-V position in the world frame.

                x = float(row["x"])
                y = float(row["y"])

                self.path_x.append(x)
                self.path_y.append(y)

        if not self.path_x:
            raise ValueError(
                "Coverage path CSV contains no waypoints"
            )

    # =============================================================
    # Initial plot limits
    # =============================================================

    def set_initial_limits(self):

        if not self.path_x or not self.path_y:
            return

        min_x = min(self.path_x)
        max_x = max(self.path_x)

        min_y = min(self.path_y)
        max_y = max(self.path_y)

        range_x = max_x - min_x
        range_y = max_y - min_y

        if range_x < 1.0:
            range_x = 1.0

        if range_y < 1.0:
            range_y = 1.0

        padding_x = max(
            10.0,
            range_x * 0.05,
        )

        padding_y = max(
            10.0,
            range_y * 0.05,
        )

        self.ax.set_xlim(
            min_x - padding_x,
            max_x + padding_x,
        )

        self.ax.set_ylim(
            min_y - padding_y,
            max_y + padding_y,
        )

    # =============================================================
    # Update live WAM-V position
    # =============================================================

    def update_boat_position(self):

        try:

            transform = self.tf_buffer.lookup_transform(
                "world",
                "wamv/wamv/base_link",
                rclpy.time.Time(),
            )

        except TransformException:

            return

        # ---------------------------------------------------------
        # Current WAM-V world position
        # ---------------------------------------------------------

        x = transform.transform.translation.x
        y = transform.transform.translation.y

        current_time = time.monotonic()

        # ---------------------------------------------------------
        # Calculate speed
        # ---------------------------------------------------------

        if (
            self.previous_x is not None
            and self.previous_y is not None
            and self.previous_time is not None
        ):

            delta_x = x - self.previous_x
            delta_y = y - self.previous_y

            delta_time = (
                current_time - self.previous_time
            )

            if delta_time > 0.0:

                distance = math.sqrt(
                    delta_x ** 2
                    + delta_y ** 2
                )

                self.current_speed = (
                    distance / delta_time
                )

        # Save current position/time
        self.previous_x = x
        self.previous_y = y
        self.previous_time = current_time

        # ---------------------------------------------------------
        # Add current position to actual trail
        # ---------------------------------------------------------

        self.trail_x.append(x)
        self.trail_y.append(y)

        # ---------------------------------------------------------
        # Update WAM-V dot
        # ---------------------------------------------------------

        self.boat_dot.set_data(
            [x],
            [y],
        )

        # ---------------------------------------------------------
        # Update actual trail
        # ---------------------------------------------------------

        self.boat_trail.set_data(
            self.trail_x,
            self.trail_y,
        )

        # ---------------------------------------------------------
        # Update live information
        # ---------------------------------------------------------

        self.boat_text.set_text(
            f"Boat position:\n"
            f"X: {x:.2f} m\n"
            f"Y: {y:.2f} m\n"
            f"Speed: {self.current_speed:.2f} m/s"
        )

        # ---------------------------------------------------------
        # Keep boat and trail visible
        # ---------------------------------------------------------

        all_x = self.path_x + self.trail_x + [x]
        all_y = self.path_y + self.trail_y + [y]

        min_x = min(all_x)
        max_x = max(all_x)

        min_y = min(all_y)
        max_y = max(all_y)

        range_x = max_x - min_x
        range_y = max_y - min_y

        if range_x < 1.0:
            range_x = 1.0

        if range_y < 1.0:
            range_y = 1.0

        padding_x = max(
            10.0,
            range_x * 0.05,
        )

        padding_y = max(
            10.0,
            range_y * 0.05,
        )

        self.ax.set_xlim(
            min_x - padding_x,
            max_x + padding_x,
        )

        self.ax.set_ylim(
            min_y - padding_y,
            max_y + padding_y,
        )

        # ---------------------------------------------------------
        # Force Matplotlib redraw
        # ---------------------------------------------------------

        self.figure.canvas.draw_idle()


def main(args=None):

    rclpy.init(args=args)

    node = CoveragePathLiveViewer()

    try:

        while rclpy.ok():

            rclpy.spin_once(
                node,
                timeout_sec=0.05,
            )

            # Allow Matplotlib to process GUI events
            plt.pause(0.001)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.try_shutdown()

        plt.close("all")


if __name__ == "__main__":
    main()