import json
import math
import sys
from pathlib import Path

import numpy as np

from PyQt5.QtCore import Qt
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


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

OCCUPANCY_PATH = DATA_DIR / "sydney_local_occupancy.npy"
COORDINATES_PATH = DATA_DIR / "sydney_world_coordinates.json"


class CoveragePlannerGui(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("VRX Coverage Planner")

        self.coverage_points = []

        self.load_map_data()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # -----------------------------------------------------
        # Status
        # -----------------------------------------------------

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
        self.path_width_input.setRange(
            1.0,
            200.0,
        )
        self.path_width_input.setDecimals(1)
        self.path_width_input.setSingleStep(1.0)
        self.path_width_input.setValue(10.0)

        radius_label = QLabel(
            "Turning radius [m]:"
        )

        self.turning_radius_input = QDoubleSpinBox()
        self.turning_radius_input.setRange(
            1.0,
            200.0,
        )
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
            self.generate_test_path
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

        self.bottom_ax.set_title(
            "Final / Live Coverage Path"
        )

        self.bottom_ax.set_xlabel("East [m]")
        self.bottom_ax.set_ylabel("North [m]")

        self.bottom_ax.grid(
            True,
            alpha=0.25,
        )

        main_layout.addWidget(
            self.bottom_canvas,
            stretch=1,
        )

        self.position_on_right_third()

    def load_map_data(self):
        """Load Sydney occupancy map and coordinates."""

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

    def draw_top_map(self):
        """Draw map, rectangle and parameter preview."""

        self.top_ax.clear()

        self.top_ax.imshow(
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

        self.top_ax.grid(
            True,
            alpha=0.20,
        )

        # Selected corners.
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

            # ---------------------------------------------
            # Selected rectangle
            # ---------------------------------------------

            rectangle_x = [
                x_min,
                x_max,
                x_max,
                x_min,
                x_min,
            ]

            rectangle_y = [
                y_min,
                y_min,
                y_max,
                y_max,
                y_min,
            ]

            self.top_ax.plot(
                rectangle_x,
                rectangle_y,
                linestyle="--",
                linewidth=2.0,
                label="Coverage area",
            )

            # ---------------------------------------------
            # Path-width preview
            #
            # Current planner creates horizontal coverage
            # rows separated vertically by path_width.
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Turning-radius preview
            # ---------------------------------------------

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

            if (
                circle_center_x + turning_radius
                <= x_max
                and circle_center_y + turning_radius
                <= y_max
            ):

                radius_circle = Circle(
                    (
                        circle_center_x,
                        circle_center_y,
                    ),
                    turning_radius,
                    fill=False,
                    linewidth=2.0,
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
                    linewidth=1.5,
                )

            # ---------------------------------------------
            # Parameter text
            # ---------------------------------------------

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

    def on_top_click(self, event):
        """Handle right-click coverage selection."""

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

        elif len(self.coverage_points) == 2:

            self.status_label.setText(
                "Coverage rectangle selected. "
                "Adjust path width and turning radius."
            )

            self.generate_button.setEnabled(
                True
            )

        self.draw_top_map()

    def on_parameter_changed(self):
        """Refresh parameter preview."""

        if len(self.coverage_points) == 2:

            self.draw_top_map()

            self.status_label.setText(
                "Parameters updated. "
                "Press Generate Path when ready."
            )

    def reset_selection(self):
        """Clear selected coverage area."""

        self.coverage_points.clear()

        self.generate_button.setEnabled(
            False
        )

        self.status_label.setText(
            "Right-click two opposite corners "
            "of the coverage area."
        )

        self.draw_top_map()

        self.bottom_ax.clear()

        self.bottom_ax.set_title(
            "Final / Live Coverage Path"
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

        self.bottom_canvas.draw_idle()

    def position_on_right_third(self):
        """Place GUI on right third of screen."""

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

        y = geometry.y()

        self.setGeometry(
            x,
            y,
            gui_width,
            screen_height,
        )

    def generate_test_path(self):
        """Temporary generation test."""

        if len(self.coverage_points) != 2:
            return

        x1, y1 = self.coverage_points[0]
        x2, y2 = self.coverage_points[1]

        x_min = min(x1, x2)
        x_max = max(x1, x2)

        y_min = min(y1, y2)
        y_max = max(y1, y2)

        path_width = (
            self.path_width_input.value()
        )

        self.bottom_ax.clear()

        direction_right = True
        current_y = y_min

        test_x = []
        test_y = []

        while current_y <= y_max:

            if direction_right:
                test_x.extend(
                    [x_min, x_max]
                )
            else:
                test_x.extend(
                    [x_max, x_min]
                )

            test_y.extend(
                [current_y, current_y]
            )

            current_y += path_width
            direction_right = (
                not direction_right
            )

        self.bottom_ax.plot(
            test_x,
            test_y,
            linewidth=1.5,
        )

        self.bottom_ax.set_xlim(
            x_min - 20,
            x_max + 20,
        )

        self.bottom_ax.set_ylim(
            y_min - 20,
            y_max + 20,
        )

        self.bottom_ax.set_title(
            "Coverage Preview — Test Only"
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

        self.bottom_canvas.draw_idle()

        self.status_label.setText(
            "Test coverage generated. "
            "Real Dubins planner not connected yet."
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
