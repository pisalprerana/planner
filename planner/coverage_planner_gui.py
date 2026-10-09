import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy

from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener

from PyQt5.QtCore import (
    QObject,
    QProcess,
    QThread,
    QTimer,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from planner.sydney_coverage_dubins import (
    generate_coverage_plan,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

OCCUPANCY_PATH = DATA_DIR / "sydney_local_occupancy.npy"
COORDINATES_PATH = DATA_DIR / "sydney_world_coordinates.json"


class PlannerWorker(QObject):

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        bounds,
        path_width,
        turning_radius,
    ):
        super().__init__()

        self.bounds = bounds
        self.path_width = path_width
        self.turning_radius = turning_radius

    @pyqtSlot()
    def run(self):

        try:
            (
                x_min,
                x_max,
                y_min,
                y_max,
            ) = self.bounds

            result = generate_coverage_plan(
                x_min,
                x_max,
                y_min,
                y_max,
                self.path_width,
                self.turning_radius,
                show_plot=False,
                create_plot=False,
            )

            self.finished.emit(result)

        except Exception as error:
            self.failed.emit(str(error))


class CoveragePlannerGui(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("VRX Coverage Planner")

        # The current Qt layout has an effective minimum width
        # of about 840 px. Locking it prevents GNOME/XWayland
        # from unexpectedly enlarging the window.
        self.gui_target_width = 840
        self.setFixedWidth(
            self.gui_target_width
        )

        self.coverage_points = []

        self.planner_thread = None
        self.planner_worker = None
        self.latest_plan = None
        self.mission_process = None
        self.mission_pgid = None

        # Gazebo native-window positioning / lifecycle.
        self.gazebo_layout_attempts = 0
        self.gazebo_window_seen = False
        self.gazebo_missing_checks = 0

        self.gazebo_layout_timer = QTimer(self)
        self.gazebo_layout_timer.timeout.connect(
            self.position_gazebo_window
        )

        # Once Gazebo has appeared, watch for it closing.
        # Closing Gazebo should shut down the complete ROS
        # mission rather than leave controllers behind.
        self.gazebo_watch_timer = QTimer(self)
        self.gazebo_watch_timer.timeout.connect(
            self.watch_gazebo_window
        )

        # -----------------------------------------------------
        # Live Gazebo coverage tracking
        #
        # Same TF source used by coverage_path_live_viewer.py:
        #
        #   world -> wamv/wamv/base_link
        #
        # Coordinates remain in Gazebo ENU.
        # -----------------------------------------------------

        self.trail_x = []
        self.trail_y = []

        self.previous_x = None
        self.previous_y = None
        self.previous_time = None

        self.current_speed = 0.0

        self.boat_dot = None
        self.boat_trail = None
        self.boat_text = None

        if not rclpy.ok():
            rclpy.init(args=None)

        self.ros_node = Node(
            "coverage_planner_gui_tracker"
        )

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self.ros_node,
            spin_thread=True,
        )

        self.ros_timer = QTimer(self)

        self.ros_timer.timeout.connect(
            self.update_live_tracking
        )

        # Same update rate as coverage_path_live_viewer.py.
        self.ros_timer.start(100)

        self.load_map_data()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        self.status_label = QLabel(
            "Right-click two opposite corners of the coverage area."
        )
        self.status_label.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(self.status_label)

        # -----------------------------------------------------
        # Top planning panel
        # -----------------------------------------------------

        self.top_figure = Figure()
        self.top_canvas = FigureCanvas(self.top_figure)
        self.top_ax = self.top_figure.add_subplot(111)

        self.draw_top_map()

        self.top_canvas.mpl_connect(
            "button_press_event",
            self.on_top_click,
        )

        main_layout.addWidget(
            self.top_canvas,
            stretch=1,
        )

        # -----------------------------------------------------
        # Planning parameters
        # -----------------------------------------------------

        parameter_layout = QHBoxLayout()

        width_label = QLabel("Path width [m]:")

        self.path_width_input = QDoubleSpinBox()
        self.path_width_input.setRange(1.0, 200.0)
        self.path_width_input.setDecimals(1)
        self.path_width_input.setSingleStep(1.0)
        self.path_width_input.setValue(10.0)

        radius_label = QLabel(
            "Turning radius [m]:"
        )

        self.turning_radius_input = QDoubleSpinBox()
        self.turning_radius_input.setRange(1.0, 200.0)
        self.turning_radius_input.setDecimals(1)
        self.turning_radius_input.setSingleStep(1.0)
        self.turning_radius_input.setValue(10.0)

        self.path_width_input.valueChanged.connect(
            self.on_parameter_changed
        )

        self.turning_radius_input.valueChanged.connect(
            self.on_parameter_changed
        )

        parameter_layout.addWidget(width_label)
        parameter_layout.addWidget(
            self.path_width_input
        )
        parameter_layout.addWidget(radius_label)
        parameter_layout.addWidget(
            self.turning_radius_input
        )

        main_layout.addLayout(
            parameter_layout
        )

        # -----------------------------------------------------
        # Buttons
        # -----------------------------------------------------

        controls_layout = QHBoxLayout()

        self.reset_button = QPushButton(
            "Reset Selection"
        )
        self.reset_button.clicked.connect(
            self.reset_selection
        )

        self.generate_button = QPushButton(
            "Generate Path"
        )
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(
            self.generate_real_path
        )

        self.launch_button = QPushButton(
            "Accept and Launch"
        )
        self.launch_button.setEnabled(False)
        self.launch_button.clicked.connect(
            self.accept_and_launch
        )

        controls_layout.addWidget(
            self.reset_button
        )
        controls_layout.addWidget(
            self.generate_button
        )
        controls_layout.addWidget(
            self.launch_button
        )

        main_layout.addLayout(
            controls_layout
        )

        # -----------------------------------------------------
        # Bottom section:
        # live coverage tracking + GUI instructions
        # -----------------------------------------------------

        bottom_layout = QHBoxLayout()

        # Keep the lower section close to the window edges.
        bottom_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        bottom_layout.setSpacing(4)

        # -----------------------------------------------------
        # Bottom-left: live coverage tracking
        # -----------------------------------------------------

        self.bottom_figure = Figure()
        self.bottom_canvas = FigureCanvas(
            self.bottom_figure
        )
        self.bottom_ax = (
            self.bottom_figure.add_subplot(111)
        )

        self.draw_empty_bottom()

        bottom_layout.addWidget(
            self.bottom_canvas,
            stretch=2,
        )

        # -----------------------------------------------------
        # Bottom-right: numbered operating instructions
        # -----------------------------------------------------

        self.instructions_label = QLabel(
            "<b>How to use the planner</b><br><br>"
            "<b>1.</b> Select the boat's start and end points "
            "by right-clicking on the Sydney Regatta map.<br><br>"
            "<b>2.</b> Set the required path width and "
            "minimum turning radius.<br><br>"
            "<b>3.</b> Press <b>Generate Path</b>.<br><br>"
            "<b>4.</b> Review the generated coverage path "
            "in the tracking panel.<br><br>"
            "<b>5.</b> Press <b>Accept and Launch</b> "
            "to start Gazebo and the mission.<br><br>"
            "<b>6.</b> Monitor the WAM-V position, "
            "travelled path and speed live."
        )

        self.instructions_label.setWordWrap(True)
        self.instructions_label.setAlignment(
            Qt.AlignTop | Qt.AlignLeft
        )

        self.instructions_label.setStyleSheet(
            """
            QLabel {
                padding: 6px;
                border: 1px solid #b0b0b0;
                border-radius: 4px;
                background: white;
            }
            """
        )

        bottom_layout.addWidget(
            self.instructions_label,
            stretch=1,
        )

        main_layout.addLayout(
            bottom_layout,
            stretch=1,
        )

        self.position_on_right_third()

    def load_map_data(self):

        self.occupancy = np.load(
            OCCUPANCY_PATH
        )

        with COORDINATES_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.coordinates = json.load(file)

        self.map_x_min = self.coordinates["x_min"]
        self.map_x_max = self.coordinates["x_max"]
        self.map_y_min = self.coordinates["y_min"]
        self.map_y_max = self.coordinates["y_max"]


    def draw_map_background(self, ax):

        ax.imshow(
            self.occupancy,
            origin="lower",
            extent=[
                self.map_x_min,
                self.map_x_max,
                self.map_y_min,
                self.map_y_max,
            ],
            cmap="binary",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            aspect="equal",
        )

    def draw_top_map(self):

        self.top_ax.clear()

        self.draw_map_background(
            self.top_ax
        )

        self.top_ax.set_xlim(
            self.map_x_min,
            self.map_x_max,
        )
        self.top_ax.set_ylim(
            self.map_y_min,
            self.map_y_max,
        )

        self.top_ax.set_title(
            "Planning Setup — Sydney Regatta"
        )
        self.top_ax.set_xlabel("East [m]")
        self.top_ax.set_ylabel("North [m]")
        self.top_ax.grid(True, alpha=0.20)

        for x, y in self.coverage_points:
            self.top_ax.scatter(
                x,
                y,
                s=50,
                marker="o",
                edgecolors="black",
                linewidths=1.0,
                zorder=10,
            )

        if len(self.coverage_points) == 2:

            x1, y1 = self.coverage_points[0]
            x2, y2 = self.coverage_points[1]

            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)

            self.top_ax.plot(
                [
                    x_min,
                    x_max,
                    x_max,
                    x_min,
                    x_min,
                ],
                [
                    y_min,
                    y_min,
                    y_max,
                    y_max,
                    y_min,
                ],
                linestyle="--",
                linewidth=2.0,
                label="Coverage area",
            )

            path_width = (
                self.path_width_input.value()
                if hasattr(
                    self,
                    "path_width_input",
                )
                else 10.0
            )

            preview_y = y_min
            first_lane = True

            while preview_y <= y_max:

                self.top_ax.plot(
                    [x_min, x_max],
                    [preview_y, preview_y],
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                    label=(
                        f"Lane spacing = "
                        f"{path_width:.1f} m"
                        if first_lane
                        else None
                    ),
                )

                first_lane = False
                preview_y += path_width

            turning_radius = (
                self.turning_radius_input.value()
                if hasattr(
                    self,
                    "turning_radius_input",
                )
                else 10.0
            )

            circle_center_x = (
                x_min + turning_radius
            )
            circle_center_y = (
                y_min + turning_radius
            )

            radius_circle = Circle(
                (
                    circle_center_x,
                    circle_center_y,
                ),
                turning_radius,
                fill=False,
                linewidth=2.5,
                linestyle="-.",
                label=(
                    f"Turning radius = "
                    f"{turning_radius:.1f} m"
                ),
            )

            self.top_ax.add_patch(
                radius_circle
            )

            self.top_ax.plot(
                [
                    circle_center_x,
                    circle_center_x
                    + turning_radius,
                ],
                [
                    circle_center_y,
                    circle_center_y,
                ],
                linewidth=2.0,
            )

            self.top_ax.text(
                0.02,
                0.98,
                (
                    f"Path width: "
                    f"{path_width:.1f} m\n"
                    f"Minimum turning radius: "
                    f"{turning_radius:.1f} m"
                ),
                transform=self.top_ax.transAxes,
                verticalalignment="top",
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.85,
                },
            )

            self.top_ax.legend(
                loc="best",
                fontsize=7,
            )

        self.top_canvas.draw_idle()

    def draw_empty_bottom(self):

        self.bottom_ax.clear()

        self.bottom_ax.set_title(
            "Final / Live Coverage Path"
        )
        self.bottom_ax.set_xlabel("East [m]")
        self.bottom_ax.set_ylabel("North [m]")
        self.bottom_ax.grid(
            True,
            alpha=0.25,
        )

        self.bottom_canvas.draw_idle()

    def on_top_click(self, event):

        if event.button != 3:
            return

        if event.inaxes != self.top_ax:
            return

        if (
            event.xdata is None
            or event.ydata is None
        ):
            return

        if len(self.coverage_points) >= 2:
            return

        self.coverage_points.append(
            (
                float(event.xdata),
                float(event.ydata),
            )
        )

        if len(self.coverage_points) == 1:

            self.status_label.setText(
                "First corner selected. "
                "Right-click the opposite corner."
            )

        else:

            self.status_label.setText(
                "Coverage rectangle selected. "
                "Adjust parameters or generate the path."
            )

            self.generate_button.setEnabled(
                True
            )

        self.draw_top_map()

    def on_parameter_changed(self):

        if len(self.coverage_points) == 2:

            self.draw_top_map()

            self.status_label.setText(
                "Parameters updated. "
                "Press Generate Path when ready."
            )

    def get_bounds(self):

        x1, y1 = self.coverage_points[0]
        x2, y2 = self.coverage_points[1]

        return (
            min(x1, x2),
            max(x1, x2),
            min(y1, y2),
            max(y1, y2),
        )

    def generate_real_path(self):

        if len(self.coverage_points) != 2:
            return

        self.generate_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.path_width_input.setEnabled(False)
        self.turning_radius_input.setEnabled(False)

        self.status_label.setText(
            "Generating safe Dubins coverage path..."
        )

        self.planner_thread = QThread()

        self.planner_worker = PlannerWorker(
            self.get_bounds(),
            self.path_width_input.value(),
            self.turning_radius_input.value(),
        )

        self.planner_worker.moveToThread(
            self.planner_thread
        )

        self.planner_thread.started.connect(
            self.planner_worker.run
        )

        self.planner_worker.finished.connect(
            self.on_plan_finished
        )

        self.planner_worker.failed.connect(
            self.on_plan_failed
        )

        self.planner_worker.finished.connect(
            self.planner_thread.quit
        )

        self.planner_worker.failed.connect(
            self.planner_thread.quit
        )

        self.planner_worker.finished.connect(
            self.planner_worker.deleteLater
        )

        self.planner_worker.failed.connect(
            self.planner_worker.deleteLater
        )

        self.planner_thread.finished.connect(
            self.planner_thread.deleteLater
        )

        self.planner_thread.finished.connect(
            self.on_planner_thread_finished
        )

        self.planner_thread.start()

    def on_planner_thread_finished(self):
        """Clear references after the planner worker thread finishes."""

        self.planner_thread = None
        self.planner_worker = None

    @pyqtSlot(object)
    def on_plan_finished(self, result):

        self.latest_plan = result

        self.draw_plan_result(
            result
        )

        self.launch_button.setEnabled(True)

        self.status_label.setText(
            "Path generated successfully. "
            f"{len(result['world_path'])} waypoints, "
            f"{result['total_distance']:.1f} m."
        )

        self.restore_controls()

    @pyqtSlot(str)
    def on_plan_failed(self, message):

        self.status_label.setText(
            f"Planner error: {message}"
        )

        self.restore_controls()

    def restore_controls(self):

        self.generate_button.setEnabled(
            len(self.coverage_points) == 2
        )

        self.reset_button.setEnabled(True)
        self.path_width_input.setEnabled(True)
        self.turning_radius_input.setEnabled(True)

    def draw_plan_result(self, result):
        """Display the same ENU path used by the Gazebo live tracker."""

        self.bottom_ax.clear()

        world_path = result["world_path"]

        path_x = world_path[:, 0]
        path_y = world_path[:, 1]

        # -----------------------------------------------------
        # Planned coverage path
        #
        # Identical coordinate convention to
        # coverage_path_live_viewer.py:
        #
        # X = Gazebo world X / East
        # Y = Gazebo world Y / North
        # -----------------------------------------------------

        self.bottom_ax.plot(
            path_x,
            path_y,
            linewidth=2.0,
            label="Coverage path",
        )

        # -----------------------------------------------------
        # Actual path travelled by WAM-V
        # -----------------------------------------------------

        self.boat_trail, = self.bottom_ax.plot(
            self.trail_x,
            self.trail_y,
            linewidth=2.0,
            label="Actual path",
        )

        # -----------------------------------------------------
        # Current WAM-V
        # -----------------------------------------------------

        self.boat_dot, = self.bottom_ax.plot(
            [],
            [],
            marker="o",
            markersize=9,
            linestyle="None",
            label="WAM-V",
        )

        # -----------------------------------------------------
        # Live telemetry
        # -----------------------------------------------------

        self.boat_text = self.bottom_ax.text(
            0.01,
            1.55,
            "Boat position: waiting for TF...",
            transform=self.bottom_ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            clip_on=False,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
            },
        )

        self.bottom_ax.set_xlabel(
            "X [m]"
        )

        self.bottom_ax.set_ylabel(
            "Y [m]"
        )

        self.bottom_ax.set_title(
            "Sydney Regatta — Live Coverage Path Tracking"
        )

        self.bottom_ax.grid(
            True,
            alpha=0.25,
        )

        self.bottom_ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.99, 1.57),
            borderaxespad=0.0,
            fontsize=8,
        )

        self.bottom_ax.set_aspect(
            "equal",
            adjustable="box",
        )

        # Keep the whole generated mission visible.
        padding = 25.0

        self.bottom_ax.set_xlim(
            float(path_x.min()) - padding,
            float(path_x.max()) + padding,
        )

        self.bottom_ax.set_ylim(
            float(path_y.min()) - padding,
            float(path_y.max()) + padding,
        )

        # Leave room above the axes for telemetry + legend.
        self.bottom_figure.subplots_adjust(
            top=0.52
        )

        self.bottom_canvas.draw_idle()

    def update_live_tracking(self):
        """Use the exact TF tracking source used by Gazebo viewer."""

        if self.ros_node is None:
            return

        if (
            self.latest_plan is None
            or self.boat_dot is None
            or self.boat_trail is None
            or self.boat_text is None
        ):
            return

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

        # Same speed calculation as coverage_path_live_viewer.py.
        if (
            self.previous_x is not None
            and self.previous_y is not None
            and self.previous_time is not None
        ):
            delta_x = x - self.previous_x
            delta_y = y - self.previous_y
            delta_time = (
                current_time
                - self.previous_time
            )

            if delta_time > 0.0:
                distance = math.sqrt(
                    delta_x ** 2
                    + delta_y ** 2
                )

                self.current_speed = (
                    distance / delta_time
                )

        self.previous_x = x
        self.previous_y = y
        self.previous_time = current_time

        # Same behavior as standalone live viewer:
        # append every TF update.
        self.trail_x.append(x)
        self.trail_y.append(y)

        self.boat_dot.set_data(
            [x],
            [y],
        )

        self.boat_trail.set_data(
            self.trail_x,
            self.trail_y,
        )

        self.boat_text.set_text(
            f"Boat position:\n"
            f"X: {x:.2f} m\n"
            f"Y: {y:.2f} m\n"
            f"Speed: {self.current_speed:.2f} m/s"
        )

        self.bottom_canvas.draw_idle()

    def reset_selection(self):

        if (
            self.planner_thread is not None
            and self.planner_thread.isRunning()
        ):
            return

        self.coverage_points.clear()
        self.latest_plan = None

        self.trail_x.clear()
        self.trail_y.clear()

        self.previous_x = None
        self.previous_y = None
        self.previous_time = None
        self.current_speed = 0.0

        self.generate_button.setEnabled(
            False
        )
        self.launch_button.setEnabled(
            False
        )

        self.status_label.setText(
            "Right-click two opposite corners "
            "of the coverage area."
        )

        self.draw_top_map()
        self.draw_empty_bottom()

    def accept_and_launch(self):
        """Launch the existing VRX mission using the generated path."""

        if self.latest_plan is None:
            return

        if (
            self.mission_process is not None
            and self.mission_process.state()
            != QProcess.NotRunning
        ):
            return

        script_path = (
            BASE_DIR.parent
            / "scripts"
            / "run_dubins_mission.sh"
        )

        if not script_path.exists():
            self.status_label.setText(
                f"Mission script not found: {script_path}"
            )
            return

        self.launch_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.reset_button.setEnabled(False)

        self.status_label.setText(
            "Building planner package and launching mission..."
        )

        self.mission_process = QProcess(self)

        self.mission_process.setWorkingDirectory(
            str(BASE_DIR.parent)
        )

        self.mission_process.setProcessChannelMode(
            QProcess.MergedChannels
        )

        self.mission_process.readyReadStandardOutput.connect(
            self.read_mission_output
        )

        self.mission_process.finished.connect(
            self.on_mission_finished
        )

        self.mission_process.started.connect(
            self.on_mission_started
        )

        # Start the entire mission in a dedicated Unix session.
        # This lets us terminate Gazebo + ROS + controller +
        # logger together instead of leaving orphan processes.
        self.mission_process.start(
            "/usr/bin/setsid",
            [
                "/usr/bin/bash",
                str(script_path),
                "--skip-planner",
            ],
        )

        # -----------------------------------------------------
        # Wait for the native Gazebo window to appear, then
        # place it on the left 2/3 of the available screen.
        # -----------------------------------------------------

        self.gazebo_layout_attempts = 0
        self.gazebo_window_seen = False
        self.gazebo_missing_checks = 0

        self.gazebo_layout_timer.start(500)

    def on_mission_started(self):
        """Remember the mission process-group leader PID."""

        if self.mission_process is None:
            return

        pid = int(
            self.mission_process.processId()
        )

        if pid > 0:
            self.mission_pgid = pid

            print(
                "Mission process group started: "
                f"{self.mission_pgid}"
            )

    def terminate_mission(self):
        """Terminate the complete mission process group."""

        self.gazebo_layout_timer.stop()
        self.gazebo_watch_timer.stop()

        pgid = self.mission_pgid

        if pgid is None:
            return

        try:
            os.killpg(
                pgid,
                signal.SIGTERM,
            )

            print(
                "Stopping complete mission process group: "
                f"{pgid}"
            )

        except ProcessLookupError:
            pass

        except PermissionError as error:
            print(
                "Could not stop mission process group: "
                f"{error}"
            )

        self.mission_pgid = None

    def gazebo_window_exists(self):
        """Return True while the native Gazebo window exists."""

        try:
            output = subprocess.check_output(
                ["wmctrl", "-lx"],
                text=True,
                stderr=subprocess.DEVNULL,
            )

        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
        ):
            return False

        for line in output.splitlines():
            lower = line.lower()

            if (
                "gz-sim-gui.gazebo gui" in lower
                and "gazebo sim" in lower
            ):
                return True

        return False

    def watch_gazebo_window(self):
        """Stop the ROS mission when its Gazebo window closes."""

        if not self.gazebo_window_seen:
            return

        if self.gazebo_window_exists():
            self.gazebo_missing_checks = 0
            return

        self.gazebo_missing_checks += 1

        # Require several consecutive misses in case the window
        # manager briefly fails to report the window.
        if self.gazebo_missing_checks >= 3:
            print(
                "Gazebo closed. Stopping the complete mission."
            )

            self.terminate_mission()

    def find_x11_window(
        self,
        title_text,
        class_text=None,
    ):
        """Return wmctrl window id for a visible XWayland/X11 window."""

        try:
            output = subprocess.check_output(
                ["wmctrl", "-lx"],
                text=True,
                stderr=subprocess.DEVNULL,
            )

        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
        ):
            return None

        for line in output.splitlines():

            lower = line.lower()

            if title_text.lower() not in lower:
                continue

            if (
                class_text is not None
                and class_text.lower() not in lower
            ):
                continue

            parts = line.split()

            if parts:
                return parts[0]

        return None

    def set_x11_window_geometry(
        self,
        window_id,
        x,
        y,
        width,
        height,
    ):
        """Move and resize an XWayland/X11 window."""

        if window_id is None:
            return False

        # Remove maximized state first.
        subprocess.run(
            [
                "wmctrl",
                "-ir",
                window_id,
                "-b",
                "remove,maximized_vert,maximized_horz",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Move + resize.
        subprocess.run(
            [
                "wmctrl",
                "-ir",
                window_id,
                "-e",
                (
                    f"0,"
                    f"{int(x)},"
                    f"{int(y)},"
                    f"{int(width)},"
                    f"{int(height)}"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return True

    def position_gazebo_window(self):
        """Tile Gazebo left 2/3 and GUI right 1/3."""

        self.gazebo_layout_attempts += 1

        gazebo_window_id = self.find_x11_window(
            "Gazebo Sim",
            "gz-sim-gui",
        )

        gui_window_id = self.find_x11_window(
            "VRX Coverage Planner",
        )

        if (
            gazebo_window_id is None
            or gui_window_id is None
        ):
            # Give Gazebo up to ~30 seconds to appear.
            if self.gazebo_layout_attempts >= 60:
                self.gazebo_layout_timer.stop()

            return

        screen = QApplication.primaryScreen()

        if screen is None:
            return

        geometry = screen.availableGeometry()

        screen_x = geometry.x()
        screen_y = geometry.y()
        screen_width = geometry.width()
        screen_height = geometry.height()

        # Leave enough room for the GUI's real minimum width.
        #
        # Screen width: 1920 px
        # GUI width:     840 px
        # Safety gap:     30 px
        # Gazebo:       ~1050 px

        safety_gap = 0

        gui_width = self.gui_target_width

        gazebo_width = (
            screen_width
            - gui_width
            - safety_gap
        )

        # -----------------------------------------------------
        # Gazebo only: left side with a small safety gap.
        #
        # Do NOT resize the GUI repeatedly while Gazebo starts.
        # -----------------------------------------------------

        self.set_x11_window_geometry(
            gazebo_window_id,
            screen_x,
            screen_y,
            gazebo_width,
            screen_height,
        )

        self.gazebo_window_seen = True
        self.gazebo_missing_checks = 0

        # Start lifecycle watcher as soon as Gazebo is found.
        if not self.gazebo_watch_timer.isActive():
            self.gazebo_watch_timer.start(1000)

        # Gazebo can resize itself during startup.
        #
        # Reapply layout for several seconds instead of
        # stopping immediately after the first successful move.
        if self.gazebo_layout_attempts >= 20:
            self.gazebo_layout_timer.stop()

            print(
                "Final window layout applied: "
                f"Gazebo={gazebo_width}x{screen_height}, "
                "GUI kept at right-third size"
            )

    def read_mission_output(self):
        """Read mission output without blocking the GUI."""

        if self.mission_process is None:
            return

        output = bytes(
            self.mission_process.readAllStandardOutput()
        ).decode(
            "utf-8",
            errors="replace",
        )

        if not output:
            return

        print(
            output,
            end="",
            flush=True,
        )

        lines = [
            line.strip()
            for line in output.splitlines()
            if line.strip()
        ]

        if lines:
            self.status_label.setText(
                lines[-1][:160]
            )

    def on_mission_finished(
        self,
        exit_code,
        exit_status,
    ):
        """Handle mission process termination."""

        self.gazebo_layout_timer.stop()
        self.gazebo_watch_timer.stop()

        self.gazebo_window_seen = False
        self.gazebo_missing_checks = 0
        self.mission_pgid = None

        if exit_code == 0:
            self.status_label.setText(
                "Mission process finished."
            )
        else:
            self.status_label.setText(
                f"Mission exited with code {exit_code}."
            )

        self.reset_button.setEnabled(True)

        if self.latest_plan is not None:
            self.launch_button.setEnabled(True)

    def closeEvent(self, event):
        """Cleanly stop the GUI ROS node."""

        if hasattr(self, "ros_timer"):
            self.ros_timer.stop()

        if hasattr(self, "gazebo_layout_timer"):
            self.gazebo_layout_timer.stop()

        if hasattr(self, "gazebo_watch_timer"):
            self.gazebo_watch_timer.stop()

        self.terminate_mission()

        if getattr(self, "ros_node", None) is not None:
            self.ros_node.destroy_node()
            self.ros_node = None

        if rclpy.ok():
            rclpy.shutdown()

        event.accept()

    def position_on_right_third(self):
        """Keep the GUI at a stable fixed width on the right."""

        screen = QApplication.primaryScreen()

        if screen is None:
            return

        geometry = screen.availableGeometry()

        gui_width = self.gui_target_width

        gui_x = (
            geometry.x()
            + geometry.width()
            - gui_width
        )

        gui_window_id = self.find_x11_window(
            "VRX Coverage Planner",
        )

        if gui_window_id is not None:

            self.set_x11_window_geometry(
                gui_window_id,
                gui_x,
                geometry.y(),
                gui_width,
                geometry.height(),
            )

        else:
            self.setGeometry(
                gui_x,
                geometry.y(),
                gui_width,
                geometry.height(),
            )


def main(args=None):

    # Both this GUI and Gazebo must be controllable through
    # the same XWayland/X11 window-management path.
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    app = QApplication(sys.argv)

    window = CoveragePlannerGui()
    window.show()

    # The native X11 window only exists after show().
    QTimer.singleShot(
        300,
        window.position_on_right_third,
    )

    QTimer.singleShot(
        1000,
        window.position_on_right_third,
    )

    sys.exit(
        app.exec_()
    )


if __name__ == "__main__":
    main()
