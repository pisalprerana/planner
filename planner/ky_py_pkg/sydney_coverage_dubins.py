import numpy as np
import matplotlib.pyplot as plt
import heapq
import math
import csv

from pathlib import Path
from planner.ky_py_pkg import sydney_coordinates
from planner.ky_py_pkg.dubins_path import generate_dubins_path


# ============================================================
# PROJECT PATHS
# ============================================================

# Directory containing this Python file
BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR
if not (PROJECT_DIR / "data").is_dir() and (BASE_DIR.parent / "data").is_dir():
    PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

# Automatically create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input occupancy grid
occupancy_path = DATA_DIR / "sydney_local_occupancy.npy"

coordinates_path = DATA_DIR / "sydney_world_coordinates.json"

# Output files
path_npy = OUTPUT_DIR / "sydney_coverage_dubins_path.npy"
path_csv = OUTPUT_DIR / "sydney_coverage_dubins_path.csv"
path_png = OUTPUT_DIR / "sydney_coverage_dubins_path.png"

STRAIGHT_STEP = 2.0
CURVE_STEP = 0.5


# ============================================================
# CONFIG
# ============================================================

# Load the coordinate system generated with the occupancy grid
coordinates = sydney_coordinates.load_coordinates(
    coordinates_path
)

map_x_min = coordinates["x_min"]
map_x_max = coordinates["x_max"]
map_y_min = coordinates["y_min"]
map_y_max = coordinates["y_max"]
resolution = coordinates["resolution"]


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not occupancy_path.exists():

    raise FileNotFoundError(
        "\nOccupancy map was not found:\n"
        f"{occupancy_path}\n\n"
        "Expected project structure:\n"
        "ky_py_pkg/\n"
        "├── sydney_coverage.py\n"
        "└── data/\n"
        "    └── sydney_local_occupancy.npy\n"
    )


# ============================================================
# LOAD OCCUPANCY GRID
# ============================================================

occupancy = np.load(occupancy_path)

grid_height, grid_width = occupancy.shape

expected_shape = sydney_coordinates.grid_shape(
    coordinates
)

if occupancy.shape != expected_shape:
    raise ValueError(
        "Occupancy grid shape does not match "
        f"coordinate metadata: grid={occupancy.shape}, "
        f"metadata={expected_shape}"
    )

print()
print("========================================")
print("MAP")
print("========================================")

print("Occupancy file:", occupancy_path)
print("Grid shape:", occupancy.shape)
print("Resolution:", resolution, "m")
print("X:", map_x_min, "to", map_x_max)
print("Y:", map_y_min, "to", map_y_max)


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def world_to_grid(x, y):
    return sydney_coordinates.world_to_grid(
        x,
        y,
        coordinates,
    )


def grid_to_world(row, col):
    return sydney_coordinates.grid_to_world(
        row,
        col,
        coordinates,
    )


def inside_grid(row, col):

    return (
        0 <= row < grid_height
        and
        0 <= col < grid_width
    )


def is_free(row, col):

    if not inside_grid(row, col):
        return False

    return occupancy[row, col] == 0


# ============================================================
# A*
# ============================================================

def heuristic(a, b):

    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def astar(start, goal):

    if not is_free(*start):

        print(
            "WARNING: A* start occupied:",
            start
        )

        return None


    if not is_free(*goal):

        print(
            "WARNING: A* goal occupied:",
            goal
        )

        return None


    neighbors = [

        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),

        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2))
    ]


    open_heap = []

    heapq.heappush(
        open_heap,
        (
            0.0,
            start
        )
    )


    came_from = {}

    g_score = {
        start: 0.0
    }

    closed = set()


    while open_heap:

        _, current = heapq.heappop(
            open_heap
        )

        if current in closed:
            continue


        if current == goal:

            path = [current]

            while current in came_from:

                current = came_from[
                    current
                ]

                path.append(current)

            path.reverse()

            return path


        closed.add(current)

        cr, cc = current


        for dr, dc, move_cost in neighbors:

            nr = cr + dr
            nc = cc + dc

            neighbor = (
                nr,
                nc
            )


            if not is_free(
                nr,
                nc
            ):
                continue


            # Prevent diagonal corner cutting

            if dr != 0 and dc != 0:

                if not (
                    is_free(
                        cr + dr,
                        cc
                    )
                    and
                    is_free(
                        cr,
                        cc + dc
                    )
                ):
                    continue


            tentative_g = (
                g_score[current]
                + move_cost
            )


            if tentative_g < g_score.get(
                neighbor,
                float("inf")
            ):

                came_from[
                    neighbor
                ] = current

                g_score[
                    neighbor
                ] = tentative_g


                f_score = (
                    tentative_g
                    + heuristic(
                        neighbor,
                        goal
                    )
                )


                heapq.heappush(
                    open_heap,
                    (
                        f_score,
                        neighbor
                    )
                )


    return None


# ============================================================
# INTERACTIVE RECTANGLE SELECTION
# ============================================================

def select_coverage_rectangle():

    print()
    print("========================================")
    print("MAP NAVIGATION")
    print("========================================")
    print()
    print("Controls:")
    print("  Left drag   : Move map")
    print("  Mouse wheel : Zoom in / out")
    print("  Right click : Select TWO opposite corners")
    print()

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    # ===============================
    # DISPLAY MAP
    # ===============================

    ax.imshow(
        occupancy,
        origin="lower",
        extent=[
            map_x_min,
            map_x_max,
            map_y_min,
            map_y_max
        ],
        cmap="binary",
        interpolation="nearest",
        vmin=0,
        vmax=1,
        aspect="equal"
    )

    ax.set_xlim(
        map_x_min,
        map_x_max
    )

    ax.set_ylim(
        map_y_min,
        map_y_max
    )

    ax.set_title(
        "Navigation mode - right-click two opposite corners"
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    points = []
    selecting = True
    drag_state = None

    # ===============================
    # Mouse interaction callbacks
    # ===============================

    def onbuttonpress(event):

        nonlocal selecting, drag_state

        if not selecting:
            return

        if event.button == 1 and event.inaxes == ax:

            x_min_view, x_max_view = ax.get_xlim()
            y_min_view, y_max_view = ax.get_ylim()

            drag_state = (
                event.x,
                event.y,
                x_min_view,
                x_max_view,
                y_min_view,
                y_max_view,
                (x_max_view - x_min_view) / ax.bbox.width,
                (y_max_view - y_min_view) / ax.bbox.height
            )

            return

        if event.button != 3 or event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        points.append(
            (event.xdata, event.ydata)
        )

        print(
            "Selected point:",
            event.xdata,
            event.ydata
        )

        ax.scatter(
            event.xdata,
            event.ydata,
            s=20
        )

        fig.canvas.draw()

        if len(points) == 2:

            selecting = False

            x1, y1 = points[0]
            x2, y2 = points[1]

            ax.plot(
                [x1, x2, x2, x1, x1],
                [y1, y1, y2, y2, y1],
                linestyle="--",
                linewidth=2
            )

            fig.canvas.draw()

            print(
                "Two points selected and saved."
            )

            plt.close(fig)

    def onmotion(event):

        if drag_state is None:
            return

        if event.x is None or event.y is None:
            return

        (
            start_x,
            start_y,
            start_x_min,
            start_x_max,
            start_y_min,
            start_y_max,
            x_per_pixel,
            y_per_pixel
        ) = drag_state

        dx = (event.x - start_x) * x_per_pixel
        dy = (event.y - start_y) * y_per_pixel

        ax.set_xlim(
            start_x_min - dx,
            start_x_max - dx
        )

        ax.set_ylim(
            start_y_min - dy,
            start_y_max - dy
        )

        fig.canvas.draw_idle()

    def onbuttonrelease(event):

        nonlocal drag_state

        if event.button == 1:
            drag_state = None

    def onscroll(event):

        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        x_min_view, x_max_view = ax.get_xlim()
        y_min_view, y_max_view = ax.get_ylim()

        x_range = x_max_view - x_min_view
        y_range = y_max_view - y_min_view

        zoom_factor = 0.85 if event.button == "up" else 1.15

        x_rel = (event.xdata - x_min_view) / x_range
        y_rel = (event.ydata - y_min_view) / y_range

        new_x_range = x_range * zoom_factor
        new_y_range = y_range * zoom_factor

        new_x_min = event.xdata - x_rel * new_x_range
        new_x_max = event.xdata + (1.0 - x_rel) * new_x_range

        new_y_min = event.ydata - y_rel * new_y_range
        new_y_max = event.ydata + (1.0 - y_rel) * new_y_range

        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect(
        "button_press_event",
        onbuttonpress
    )

    fig.canvas.mpl_connect(
        "motion_notify_event",
        onmotion
    )

    fig.canvas.mpl_connect(
        "button_release_event",
        onbuttonrelease
    )

    fig.canvas.mpl_connect(
        "scroll_event",
        onscroll
    )

    plt.show()

    if len(points) != 2:

        raise RuntimeError(
            "Coverage area not selected"
        )

    x1, y1 = points[0]
    x2, y2 = points[1]

    rect_x_min = max(
        min(x1, x2),
        map_x_min
    )

    rect_x_max = min(
        max(x1, x2),
        map_x_max
    )

    rect_y_min = max(
        min(y1, y2),
        map_y_min
    )

    rect_y_max = min(
        max(y1, y2),
        map_y_max
    )

    print()
    print("Selected rectangle:")
    print(
        "X:",
        rect_x_min,
        "to",
        rect_x_max
    )

    print(
        "Y:",
        rect_y_min,
        "to",
        rect_y_max
    )

    return (
        rect_x_min,
        rect_x_max,
        rect_y_min,
        rect_y_max
    )


# ============================================================
# SELECT RECTANGLE
# ============================================================

(
    coverage_x_min,
    coverage_x_max,
    coverage_y_min,
    coverage_y_max
) = select_coverage_rectangle()


# ============================================================
# PATH WIDTH
# ============================================================

print()
print("========================================")
print("PATH WIDTH")
print("========================================")


try:

    path_width = float(
        input(
            "Enter coverage path width [m] "
            "(example 10): "
        )
    )

except ValueError:

    raise RuntimeError(
        "Path width must be a number."
    )


if path_width <= 0:

    raise RuntimeError(
        "Path width must be > 0."
    )


try:

    turning_radius = float(
        input(
            "Enter minimum turning radius [m]: "
        )
    )

except ValueError:

    raise RuntimeError(
        "Minimum turning radius must be a number."
    )


if not math.isfinite(turning_radius) or turning_radius <= 0:

    raise RuntimeError(
        "Minimum turning radius must be a positive finite value."
    )


row_spacing = max(
    1,
    int(
        round(
            path_width / resolution
        )
    )
)


print()
print(
    "Path width:",
    path_width,
    "m"
)

print(
    "Grid row spacing:",
    row_spacing
)


# ============================================================
# RECTANGLE -> GRID
# ============================================================

row_min, col_min = world_to_grid(
    coverage_x_min,
    coverage_y_min
)

row_max, col_max = world_to_grid(
    coverage_x_max,
    coverage_y_max
)


row_min = max(
    0,
    min(
        row_min,
        grid_height - 1
    )
)

row_max = max(
    0,
    min(
        row_max,
        grid_height - 1
    )
)

col_min = max(
    0,
    min(
        col_min,
        grid_width - 1
    )
)

col_max = max(
    0,
    min(
        col_max,
        grid_width - 1
    )
)


if row_min > row_max:

    row_min, row_max = (
        row_max,
        row_min
    )


if col_min > col_max:

    col_min, col_max = (
        col_max,
        col_min
    )


# ============================================================
# FIND FREE SEGMENTS
# ============================================================

def find_free_segments(
    row,
    col_start,
    col_end
):

    segments = []

    segment_start = None


    for col in range(
        col_start,
        col_end + 1
    ):

        free = is_free(
            row,
            col
        )


        if free:

            if segment_start is None:

                segment_start = col


        else:

            if segment_start is not None:

                segments.append(
                    (
                        segment_start,
                        col - 1
                    )
                )

                segment_start = None


    if segment_start is not None:

        segments.append(
            (
                segment_start,
                col_end
            )
        )


    return segments


# ============================================================
# BUILD COVERAGE TARGETS
# ============================================================

coverage_targets = []
coverage_lines_grid = []

direction_left_to_right = True


rows = list(
    range(
        row_min,
        row_max + 1,
        row_spacing
    )
)


for row in rows:

    segments = find_free_segments(
        row,
        col_min,
        col_max
    )


    if len(segments) == 0:

        continue


    if direction_left_to_right:

        ordered_segments = segments

    else:

        ordered_segments = list(
            reversed(
                segments
            )
        )


    for seg_start, seg_end in ordered_segments:

        if direction_left_to_right:

            start_point = (
                row,
                seg_start
            )

            end_point = (
                row,
                seg_end
            )

        else:

            start_point = (
                row,
                seg_end
            )

            end_point = (
                row,
                seg_start
            )

        fallback_yaw = (
            0.0
            if direction_left_to_right
            else math.pi
        )

        coverage_lines_grid.append(
            (
                start_point,
                end_point,
                fallback_yaw
            )
        )


        coverage_targets.append(
            start_point
        )


        if end_point != start_point:

            coverage_targets.append(
                end_point
            )


    direction_left_to_right = (
        not direction_left_to_right
    )


if len(coverage_targets) == 0:

    raise RuntimeError(
        "No free coverage path "
        "inside selected rectangle."
    )


print()
print("========================================")
print("COVERAGE")
print("========================================")

print("Number of coverage targets:", len(coverage_targets))
print("Number of coverage lines:", len(coverage_lines_grid))


# ============================================================
# COVERAGE LINES + DUBINS TRANSITIONS
# ============================================================

def sample_straight_segment(start_xy, end_xy, yaw, step_size):

    distance = math.hypot(
        end_xy[0] - start_xy[0],
        end_xy[1] - start_xy[1]
    )

    if distance <= 1e-12:

        return np.asarray(
            [[start_xy[0], start_xy[1], yaw]],
            dtype=np.float64
        )

    sample_count = max(
        1,
        int(
            math.ceil(
                distance / step_size
            )
        )
    )

    fractions = np.linspace(
        0.0,
        1.0,
        sample_count + 1
    )

    x_values = (
        start_xy[0]
        + fractions * (end_xy[0] - start_xy[0])
    )

    y_values = (
        start_xy[1]
        + fractions * (end_xy[1] - start_xy[1])
    )

    yaw_values = np.full(
        sample_count + 1,
        yaw
    )

    return np.column_stack(
        (x_values, y_values, yaw_values)
    )


def align_yaw(yaw, reference_yaw):

    return yaw + 2.0 * math.pi * round(
        (reference_yaw - yaw) / (2.0 * math.pi)
    )


world_path_parts = []
baseline_path_parts = []
coverage_lines_world = []
dubins_transitions = []
previous_line_end = None


for start_grid, end_grid, fallback_yaw in coverage_lines_grid:

    start_xy = np.asarray(
        grid_to_world(*start_grid),
        dtype=np.float64
    )

    end_xy = np.asarray(
        grid_to_world(*end_grid),
        dtype=np.float64
    )

    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    line_length = math.hypot(dx, dy)
    line_yaw = (
        math.atan2(dy, dx)
        if line_length > 1e-12
        else fallback_yaw
    )

    coverage_lines_world.append(
        np.vstack((start_xy, end_xy))
    )

    if previous_line_end is not None:

        baseline_transition = generate_dubins_path(
            previous_line_end,
            [start_xy[0], start_xy[1], line_yaw],
            turning_radius,
            CURVE_STEP,
            CURVE_STEP
        )

        baseline_path_parts.append(
            baseline_transition[1:]
        )

        transition = generate_dubins_path(
            previous_line_end,
            [start_xy[0], start_xy[1], line_yaw],
            turning_radius,
            STRAIGHT_STEP,
            CURVE_STEP
        )

        dubins_transitions.append(transition)
        world_path_parts.append(transition[1:])
        line_yaw = align_yaw(
            line_yaw,
            transition[-1, 2]
        )

    baseline_straight_segment = sample_straight_segment(
        start_xy,
        end_xy,
        line_yaw,
        CURVE_STEP
    )

    straight_segment = sample_straight_segment(
        start_xy,
        end_xy,
        line_yaw,
        STRAIGHT_STEP
    )

    if previous_line_end is None:

        baseline_path_parts.append(
            baseline_straight_segment
        )

        world_path_parts.append(straight_segment)

    else:

        baseline_path_parts.append(
            baseline_straight_segment[1:]
        )

        world_path_parts.append(straight_segment[1:])

    previous_line_end = straight_segment[-1]


world_path = np.vstack(
    world_path_parts
)

world_path_before_adaptive_sampling = np.vstack(
    baseline_path_parts
)

world_path = np.asarray(
    world_path,
    dtype=np.float64
)


if len(world_path) == 0:

    raise RuntimeError(
        "No valid path was generated."
    )


# ============================================================
# SAVE NPY
# ============================================================

np.save(
    path_npy,
    world_path
)


# ============================================================
# SAVE CSV
# ============================================================

with open(
    path_csv,
    "w",
    newline=""
) as f:

    writer = csv.writer(
        f
    )


    writer.writerow([
        "waypoint_id",
        "x",
        "y",
        "yaw"
    ])


    for i, point in enumerate(
        world_path
    ):

        writer.writerow([
            i,
            point[0],
            point[1],
            point[2]
        ])


print()
print("========================================")
print("PATH OUTPUT")
print("========================================")

print(
    "NPY:",
    path_npy
)

print(
    "CSV:",
    path_csv
)

print(
    "Number of waypoints before adaptive sampling:",
    len(world_path_before_adaptive_sampling)
)

print(
    "Number of waypoints after adaptive sampling:",
    len(world_path)
)

print(
    "First waypoint:",
    world_path[0]
)

print(
    "Last waypoint:",
    world_path[-1]
)


# ============================================================
# CALCULATE PATH LENGTH
# ============================================================

if len(world_path) > 1:

    total_distance = np.sum(
        np.linalg.norm(
            np.diff(world_path[:, :2], axis=0),
            axis=1
        )
    )

else:

    total_distance = 0.0


print()
print(
    "Total path distance:",
    round(
        total_distance,
        2
    ),
    "m"
)


# ============================================================
# PLOT RESULT
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 8)
)


ax.imshow(
    occupancy,
    origin="lower",
    extent=[
        map_x_min,
        map_x_max,
        map_y_min,
        map_y_max
    ],
    cmap="binary",
    interpolation="nearest",
    vmin=0,
    vmax=1,
    aspect="equal"
)


# ============================================================
# COVERAGE RECTANGLE
# ============================================================

rectangle_x = [
    coverage_x_min,
    coverage_x_max,
    coverage_x_max,
    coverage_x_min,
    coverage_x_min
]

rectangle_y = [
    coverage_y_min,
    coverage_y_min,
    coverage_y_max,
    coverage_y_max,
    coverage_y_min
]


ax.plot(
    rectangle_x,
    rectangle_y,
    linestyle="--",
    linewidth=2,
    label="Coverage area"
)


# ============================================================
# ORIGINAL COVERAGE LINES
# ============================================================

for index, line in enumerate(coverage_lines_world):

    ax.plot(
        line[:, 0],
        line[:, 1],
        color="tab:blue",
        linewidth=0.55,
        alpha=0.7,
        label="Straight coverage lines" if index == 0 else None
    )


# ============================================================
# DUBINS TRANSITIONS
# ============================================================

for index, transition in enumerate(dubins_transitions):

    ax.plot(
        transition[:, 0],
        transition[:, 1],
        color="tab:orange",
        linewidth=0.7,
        label="Dubins transitions" if index == 0 else None
    )

ax.scatter(
    world_path[0, 0],
    world_path[0, 1],
    marker="o",
    s=12,
    color="tab:green",
    label="Trajectory start"
)

ax.plot(
    world_path[:, 0],
    world_path[:, 1],
    color="black",
    linewidth=0.55,
    alpha=0.85,
    label="Final trajectory"
)


# ============================================================
# PLOT SETTINGS
# ============================================================

ax.set_xlabel(
    "X [m]"
)

ax.set_ylabel(
    "Y [m]"
)

ax.set_title(
    "Sydney Regatta Dubins Coverage Path"
)

ax.set_xlim(
    map_x_min,
    map_x_max
)

ax.set_ylim(
    map_y_min,
    map_y_max
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

plt.tight_layout()


plt.savefig(
    path_png,
    dpi=200
)


print(
    "PNG:",
    path_png
)


plt.show()
