import json
import sys
from pathlib import Path

import numpy as np

from PyQt5.QtCore import (
    QObject,
    QThread,
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

        self.coverage_points = []

        self.planner_thread = None
        self.planner_worker = None
        self.latest_plan = None

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

        controls_layout.addWidget(
            self.reset_button
        )
        controls_layout.addWidget(
            self.generate_button
        )

        main_layout.addLayout(
            controls_layout
        )

        # -----------------------------------------------------
        # Bottom panel
        # -----------------------------------------------------

        self.bottom_figure = Figure()
        self.bottom_canvas = FigureCanvas(
            self.bottom_figure
        )
        self.bottom_ax = (
            self.bottom_figure.add_subplot(111)
        )

        self.draw_empty_bottom()

        main_layout.addWidget(
            self.bottom_canvas,
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

        self.planner_thread.start()

    @pyqtSlot(object)
    def on_plan_finished(self, result):

        self.latest_plan = result

        self.draw_plan_result(
            result
        )

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

        self.bottom_ax.clear()

        self.draw_map_background(
            self.bottom_ax
        )

        (
            x_min,
            x_max,
            y_min,
            y_max,
        ) = result["coverage_bounds"]

        self.bottom_ax.plot(
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

        component_mask = result[
            "component_mask"
        ]

        self.bottom_ax.contour(
            component_mask.astype(float),
            levels=[0.5],
            origin="lower",
            extent=[
                self.map_x_min,
                self.map_x_max,
                self.map_y_min,
                self.map_y_max,
            ],
            linewidths=2.0,
        )

        for index, line in enumerate(
            result["coverage_lines"]
        ):

            self.bottom_ax.plot(
                line[:, 0],
                line[:, 1],
                linewidth=0.8,
                alpha=0.7,
                label=(
                    "Coverage lines"
                    if index == 0
                    else None
                ),
            )

        for index, transition in enumerate(
            result["dubins_transitions"]
        ):

            self.bottom_ax.plot(
                transition[:, 0],
                transition[:, 1],
                linewidth=2.0,
                label=(
                    "Dubins transitions"
                    if index == 0
                    else None
                ),
            )

        for index, transition in enumerate(
            result["astar_transitions"]
        ):

            self.bottom_ax.plot(
                transition[:, 0],
                transition[:, 1],
                linewidth=1.5,
                label=(
                    "A* fallback"
                    if index == 0
                    else None
                ),
            )

        world_path = result[
            "world_path"
        ]

        self.bottom_ax.plot(
            world_path[:, 0],
            world_path[:, 1],
            linewidth=1.0,
            label="Final trajectory",
        )

        start = world_path[0]
        end = world_path[-1]

        self.bottom_ax.scatter(
            start[0],
            start[1],
            marker="o",
            s=70,
            zorder=20,
            label="Start",
        )

        self.bottom_ax.scatter(
            end[0],
            end[1],
            marker="X",
            s=90,
            zorder=20,
            label="End",
        )

        padding = 25.0

        self.bottom_ax.set_xlim(
            max(
                self.map_x_min,
                x_min - padding,
            ),
            min(
                self.map_x_max,
                x_max + padding,
            ),
        )

        self.bottom_ax.set_ylim(
            max(
                self.map_y_min,
                y_min - padding,
            ),
            min(
                self.map_y_max,
                y_max + padding,
            ),
        )

        self.bottom_ax.set_title(
            "Generated Dubins Coverage Path"
        )

        self.bottom_ax.set_xlabel(
            "East [m]"
        )

        self.bottom_ax.set_ylabel(
            "North [m]"
        )

        self.bottom_ax.grid(
            True,
            alpha=0.25,
        )

        self.bottom_ax.legend(
            loc="best",
            fontsize=7,
        )

        self.bottom_ax.text(
            0.02,
            0.98,
            (
                f"Path width: "
                f"{result['path_width']:.1f} m\n"
                f"Turning radius: "
                f"{result['turning_radius']:.1f} m\n"
                f"Distance: "
                f"{result['total_distance']:.1f} m\n"
                f"Waypoints: "
                f"{len(world_path)}"
            ),
            transform=self.bottom_ax.transAxes,
            verticalalignment="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
            },
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

        self.generate_button.setEnabled(
            False
        )

        self.status_label.setText(
            "Right-click two opposite corners "
            "of the coverage area."
        )

        self.draw_top_map()
        self.draw_empty_bottom()

    def position_on_right_third(self):

        screen = QApplication.primaryScreen()

        if screen is None:
            return

        geometry = screen.availableGeometry()

        screen_width = geometry.width()
        screen_height = geometry.height()

        gui_width = screen_width // 3

        x = (
            geometry.x()
            + screen_width
            - gui_width
        )

        self.setGeometry(
            x,
            geometry.y(),
            gui_width,
            screen_height,
        )


def main(args=None):

    app = QApplication(sys.argv)

    window = CoveragePlannerGui()
    window.show()

    sys.exit(
        app.exec_()
    )


if __name__ == "__main__":
    main()
