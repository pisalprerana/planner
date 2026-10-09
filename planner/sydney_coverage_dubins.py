import csv
import heapq
import math
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from planner import sydney_coordinates
from planner.dubins_path import generate_dubins_path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR
if not (PROJECT_DIR / "data").is_dir() and (BASE_DIR.parent / "data").is_dir():
    PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

occupancy_path = DATA_DIR / "sydney_local_occupancy.npy"
coordinates_path = DATA_DIR / "sydney_world_coordinates.json"

path_npy = OUTPUT_DIR / "sydney_coverage_dubins_path.npy"
path_csv = OUTPUT_DIR / "sydney_coverage_dubins_path.csv"
path_png = OUTPUT_DIR / "sydney_coverage_dubins_path.png"


# ============================================================
# PLANNER SETTINGS
# ============================================================

STRAIGHT_STEP = 2.0
CURVE_STEP = 0.5

# Land is treated as forbidden up to this distance.
SAFETY_CLEARANCE_METERS = 10.0

# A Dubins turn is always attempted first.
#
# The endpoints of coverage lines are moved inward by several
# candidate distances until a collision-free Dubins turn is found.
TURN_SEARCH_STEP_METERS = 2.0
TURN_SEARCH_MAX_FACTOR = 3.0

# Do not shorten a coverage line below this fraction of its
# original length.
MIN_COVERAGE_FRACTION = 0.55

# A* is only a last-resort connector when no safe Dubins
# transition can be found after endpoint retreat.
USE_ASTAR_FALLBACK = True

# A* searches only a local window around the transition.
ASTAR_INITIAL_MARGIN_METERS = 40.0
ASTAR_MARGIN_STEP_METERS = 40.0
ASTAR_MAX_MARGIN_METERS = 240.0

# Plotting.
SAFE_BOUNDARY_LINE_WIDTH = 4.0
TRAJECTORY_LINE_WIDTH = 0.9
DUBINS_LINE_WIDTH = 2.2
ASTAR_LINE_WIDTH = 1.6
PLOT_PADDING_METERS = 25.0


# ============================================================
# LOAD MAP
# ============================================================

coordinates = sydney_coordinates.load_coordinates(coordinates_path)

map_x_min = coordinates["x_min"]
map_x_max = coordinates["x_max"]
map_y_min = coordinates["y_min"]
map_y_max = coordinates["y_max"]
resolution = coordinates["resolution"]

if not occupancy_path.exists():
    raise FileNotFoundError(
        f"Occupancy map not found:\n{occupancy_path}"
    )

occupancy = np.load(occupancy_path)
grid_height, grid_width = occupancy.shape

expected_shape = sydney_coordinates.grid_shape(coordinates)

if occupancy.shape != expected_shape:
    raise ValueError(
        "Occupancy grid shape does not match coordinate metadata: "
        f"grid={occupancy.shape}, metadata={expected_shape}"
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
print("Land safety clearance:", SAFETY_CLEARANCE_METERS, "m")


# ============================================================
# COORDINATE HELPERS
# ============================================================

def world_to_grid(x, y):
    return sydney_coordinates.world_to_grid(
        x, y, coordinates
    )


def grid_to_world(row, col):
    return sydney_coordinates.grid_to_world(
        row, col, coordinates
    )


def inside_grid(row, col):
    return (
        0 <= row < grid_height
        and 0 <= col < grid_width
    )


def is_free(row, col):
    if not inside_grid(row, col):
        return False
    return occupancy[row, col] == 0


# ============================================================
# SAFETY MASK
# ============================================================

def build_safe_water_mask(clearance_meters):
    """
    Build a conservative safe-water mask.

    A cell is safe only if it is water and there is no land
    inside a square neighborhood corresponding to the requested
    clearance. The square is intentionally conservative.
    """

    clearance_cells = int(
        math.ceil(clearance_meters / resolution)
    )

    if clearance_cells <= 0:
        return occupancy == 0

    free = occupancy == 0

    # Prefix sum makes the local occupied-cell test fast.
    land = (~free).astype(np.int32)

    padded = np.pad(
        land,
        (
            (clearance_cells, clearance_cells),
            (clearance_cells, clearance_cells),
        ),
        mode="constant",
        constant_values=1,
    )

    # Add a zero border to the integral image. This is important:
    # without it, the sliding-window result is one row and one
    # column smaller than the original occupancy grid.
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(
        integral,
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    )

    k = clearance_cells
    window = (
        integral[2 * k + 1:, 2 * k + 1:]
        - integral[:-2 * k - 1, 2 * k + 1:]
        - integral[2 * k + 1:, :-2 * k - 1]
        + integral[:-2 * k - 1, :-2 * k - 1]
    )

    # window is now exactly the same shape as occupancy.
    if window.shape != occupancy.shape:
        raise RuntimeError(
            "Internal safety-mask shape mismatch: "
            f"window={window.shape}, occupancy={occupancy.shape}"
        )

    return free & (window == 0)


safe_water = build_safe_water_mask(
    SAFETY_CLEARANCE_METERS
)

clearance_cells = int(
    math.ceil(SAFETY_CLEARANCE_METERS / resolution)
)

print("Safety clearance cells:", clearance_cells)


def is_safe(row, col):
    if not inside_grid(row, col):
        return False
    return bool(safe_water[row, col])


def path_is_safe(path):
    """
    Check every sampled point against the safe-water mask.
    """

    if path is None or len(path) == 0:
        return False

    for point in path:
        x = float(point[0])
        y = float(point[1])

        if not math.isfinite(x) or not math.isfinite(y):
            return False

        row, col = world_to_grid(x, y)

        if not is_safe(row, col):
            return False

    return True


# ============================================================
# CONNECTED COMPONENT
# ============================================================

def largest_connected_component(mask, row_min, row_max, col_min, col_max):
    """
    Find the largest 8-connected safe-water region inside the
    selected search window.

    This prevents the planner from pretending that two separate
    bodies of water are connected.
    """

    visited = np.zeros(mask.shape, dtype=np.uint8)

    best_component = set()

    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):

            if not mask[row, col]:
                continue

            if visited[row, col]:
                continue

            queue = deque([(row, col)])
            visited[row, col] = 1
            component = []

            while queue:
                cr, cc = queue.popleft()
                component.append((cr, cc))

                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):

                        if dr == 0 and dc == 0:
                            continue

                        nr = cr + dr
                        nc = cc + dc

                        if (
                            nr < row_min
                            or nr > row_max
                            or nc < col_min
                            or nc > col_max
                        ):
                            continue

                        if visited[nr, nc]:
                            continue

                        if not mask[nr, nc]:
                            continue

                        visited[nr, nc] = 1
                        queue.append((nr, nc))

            if len(component) > len(best_component):
                best_component = set(component)

    return best_component


# ============================================================
# INTERACTIVE SELECTION
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

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(
        occupancy,
        origin="lower",
        extent=[
            map_x_min,
            map_x_max,
            map_y_min,
            map_y_max,
        ],
        cmap="binary",
        interpolation="nearest",
        vmin=0,
        vmax=1,
        aspect="equal",
    )

    ax.set_xlim(map_x_min, map_x_max)
    ax.set_ylim(map_y_min, map_y_max)
    ax.set_title(
        "Navigation mode - right-click two opposite corners"
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    points = []
    selecting = True
    drag_state = None

    def onbuttonpress(event):
        nonlocal selecting, drag_state

        if not selecting:
            return

        if event.button == 1 and event.inaxes == ax:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()

            drag_state = (
                event.x,
                event.y,
                x0,
                x1,
                y0,
                y1,
                (x1 - x0) / ax.bbox.width,
                (y1 - y0) / ax.bbox.height,
            )
            return

        if event.button != 3 or event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        points.append((event.xdata, event.ydata))

        print(
            "Selected point:",
            event.xdata,
            event.ydata,
        )

        ax.scatter(
            event.xdata,
            event.ydata,
            s=60,
            marker="o",
            edgecolors="black",
            linewidths=1.0,
        )

        fig.canvas.draw_idle()

        if len(points) == 2:
            selecting = False

            x1, y1 = points[0]
            x2, y2 = points[1]

            ax.plot(
                [x1, x2, x2, x1, x1],
                [y1, y1, y2, y2, y1],
                linestyle="--",
                linewidth=2,
            )

            fig.canvas.draw_idle()
            print("Two points selected and saved.")
            plt.close(fig)

    def onmotion(event):
        if drag_state is None:
            return

        if event.x is None or event.y is None:
            return

        (
            start_x,
            start_y,
            x0,
            x1,
            y0,
            y1,
            x_per_pixel,
            y_per_pixel,
        ) = drag_state

        dx = (event.x - start_x) * x_per_pixel
        dy = (event.y - start_y) * y_per_pixel

        ax.set_xlim(x0 - dx, x1 - dx)
        ax.set_ylim(y0 - dy, y1 - dy)
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

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        xr = x1 - x0
        yr = y1 - y0

        zoom_factor = 0.85 if event.button == "up" else 1.15

        x_rel = (event.xdata - x0) / xr
        y_rel = (event.ydata - y0) / yr

        new_xr = xr * zoom_factor
        new_yr = yr * zoom_factor

        ax.set_xlim(
            event.xdata - x_rel * new_xr,
            event.xdata + (1.0 - x_rel) * new_xr,
        )

        ax.set_ylim(
            event.ydata - y_rel * new_yr,
            event.ydata + (1.0 - y_rel) * new_yr,
        )

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect(
        "button_press_event",
        onbuttonpress,
    )
    fig.canvas.mpl_connect(
        "motion_notify_event",
        onmotion,
    )
    fig.canvas.mpl_connect(
        "button_release_event",
        onbuttonrelease,
    )
    fig.canvas.mpl_connect(
        "scroll_event",
        onscroll,
    )

    plt.show()

    if len(points) != 2:
        raise RuntimeError("Coverage area not selected.")

    x1, y1 = points[0]
    x2, y2 = points[1]

    return (
        max(min(x1, x2), map_x_min),
        min(max(x1, x2), map_x_max),
        max(min(y1, y2), map_y_min),
        min(max(y1, y2), map_y_max),
    )


def generate_coverage_plan(
    coverage_x_min,
    coverage_x_max,
    coverage_y_min,
    coverage_y_max,
    path_width,
    turning_radius,
    show_plot=False,
    create_plot=True,
    scan_direction="auto",
    _trial=False,
):
    """Generate a safe Dubins coverage path using horizontal or vertical sweeps."""

    coverage_x_min = float(coverage_x_min)
    coverage_x_max = float(coverage_x_max)
    coverage_y_min = float(coverage_y_min)
    coverage_y_max = float(coverage_y_max)

    path_width = float(path_width)
    turning_radius = float(turning_radius)

    if coverage_x_min > coverage_x_max:
        coverage_x_min, coverage_x_max = (
            coverage_x_max,
            coverage_x_min,
        )

    if coverage_y_min > coverage_y_max:
        coverage_y_min, coverage_y_max = (
            coverage_y_max,
            coverage_y_min,
        )

    coverage_x_min = max(
        coverage_x_min,
        map_x_min,
    )
    coverage_x_max = min(
        coverage_x_max,
        map_x_max,
    )
    coverage_y_min = max(
        coverage_y_min,
        map_y_min,
    )
    coverage_y_max = min(
        coverage_y_max,
        map_y_max,
    )

    if coverage_x_min >= coverage_x_max:
        raise RuntimeError(
            "Coverage area has invalid X bounds."
        )

    if coverage_y_min >= coverage_y_max:
        raise RuntimeError(
            "Coverage area has invalid Y bounds."
        )

    if (
        not math.isfinite(path_width)
        or path_width <= 0.0
    ):
        raise RuntimeError(
            "Path width must be positive."
        )

    if (
        not math.isfinite(turning_radius)
        or turning_radius <= 0.0
    ):
        raise RuntimeError(
            "Minimum turning radius must be positive."
        )

    if scan_direction not in ("auto", "horizontal", "vertical"):
        raise ValueError("scan_direction must be auto, horizontal, or vertical")

    if scan_direction == "auto":
        candidates = []
        for direction in ("horizontal", "vertical"):
            print(f"\n========== TRY {direction.upper()} SWEEPS ==========")
            try:
                candidate = generate_coverage_plan(
                    coverage_x_min, coverage_x_max,
                    coverage_y_min, coverage_y_max,
                    path_width, turning_radius,
                    show_plot=False, create_plot=False,
                    scan_direction=direction, _trial=True,
                )
                candidates.append((direction, candidate))
                print(
                    f"{direction}: distance={candidate['total_distance']:.1f} m, "
                    f"lines={len(candidate['coverage_lines'])}, "
                    f"fallbacks={len(candidate['astar_transitions'])}, "
                    f"coverage length={candidate['coverage_length']:.1f} m"
                )
            except (RuntimeError, ValueError) as error:
                print(f"{direction}: not feasible ({error})")

        if not candidates:
            raise RuntimeError("Neither horizontal nor vertical coverage is feasible")

        # Avoid selecting a significantly less complete sweep just because it is short.
        # Scan-line length is a proxy, not an exact swept-area coverage measurement.
        max_coverage = max(c['coverage_length'] for _, c in candidates)
        eligible = [
            (direction, c) for direction, c in candidates
            if c['coverage_length'] >= 0.95 * max_coverage
        ]
        selected_direction, selected = min(
            eligible,
            key=lambda item: (
                len(item[1]['astar_transitions']) > 0,
                item[1]['total_distance'],
                len(item[1]['coverage_lines']),
            ),
        )
        print(f"\nSELECTED SCAN DIRECTION: {selected_direction.upper()}")
        print("(Coverage-length filter: at least 95% of the best candidate)")
        if _trial:
            return selected
        return generate_coverage_plan(
            coverage_x_min, coverage_x_max,
            coverage_y_min, coverage_y_max,
            path_width, turning_radius,
            show_plot=show_plot, create_plot=create_plot,
            scan_direction=selected_direction, _trial=False,
        )

    row_spacing = max(
        1,
        int(round(path_width / resolution)),
    )

    print()
    print("========================================")
    print("PLANNER INPUT")
    print("========================================")
    print(
        "Selected X:",
        coverage_x_min,
        "to",
        coverage_x_max,
    )
    print(
        "Selected Y:",
        coverage_y_min,
        "to",
        coverage_y_max,
    )
    print("Path width:", path_width, "m")
    print(
        "Turning radius:",
        turning_radius,
        "m",
    )
    print(
        "Grid row spacing:",
        row_spacing,
    )

    # ============================================================
    # SELECTED RECTANGLE -> GRID
    # ============================================================

    row_min, col_min = world_to_grid(
        coverage_x_min,
        coverage_y_min,
    )

    row_max, col_max = world_to_grid(
        coverage_x_max,
        coverage_y_max,
    )

    row_min = max(0, min(row_min, grid_height - 1))
    row_max = max(0, min(row_max, grid_height - 1))
    col_min = max(0, min(col_min, grid_width - 1))
    col_max = max(0, min(col_max, grid_width - 1))

    if row_min > row_max:
        row_min, row_max = row_max, row_min

    if col_min > col_max:
        col_min, col_max = col_max, col_min


    # ============================================================
    # LARGEST SAFE WATER REGION
    # ============================================================

    print()
    print("========================================")
    print("SAFE WATER REGION")
    print("========================================")

    largest_component = largest_connected_component(
        safe_water,
        row_min,
        row_max,
        col_min,
        col_max,
    )

    if not largest_component:
        raise RuntimeError(
            "No connected safe-water region exists inside "
            "the selected area."
        )

    component_mask = np.zeros_like(safe_water, dtype=bool)

    for cell in largest_component:
        component_mask[cell] = True

    print(
        "Safe-water cells in selected region:",
        len(largest_component),
    )
    print(
        "Only the largest connected safe-water region "
        "will be used for coverage."
    )


    def is_component_safe(row, col):
        if not inside_grid(row, col):
            return False
        return bool(component_mask[row, col])


    def path_is_component_safe(path):
        if path is None or len(path) == 0:
            return False

        for point in path:
            x = float(point[0])
            y = float(point[1])

            if not math.isfinite(x) or not math.isfinite(y):
                return False

            row, col = world_to_grid(x, y)

            if not is_component_safe(row, col):
                return False

        return True


    # ============================================================
    # BUILD HORIZONTAL OR VERTICAL COVERAGE LINES
    # ============================================================

    def find_component_segments(fixed_index):
        """Contiguous safe cells along the scan line."""
        segments = []
        start = None
        if scan_direction == "horizontal":
            indices = range(col_min, col_max + 1)
            to_cell = lambda variable: (fixed_index, variable)
        else:
            indices = range(row_min, row_max + 1)
            to_cell = lambda variable: (variable, fixed_index)

        for variable in indices:
            if is_component_safe(*to_cell(variable)):
                if start is None:
                    start = variable
            elif start is not None:
                segments.append((start, variable - 1))
                start = None
        if start is not None:
            segments.append((start, indices.stop - 1))
        return segments

    coverage_lines_grid = []
    forward = True
    fixed_indices = (
        range(row_min, row_max + 1, row_spacing)
        if scan_direction == "horizontal"
        else range(col_min, col_max + 1, row_spacing)
    )

    for fixed in fixed_indices:
        useful = [
            seg for seg in find_component_segments(fixed)
            if seg[1] - seg[0] >= 2
        ]
        if not useful:
            continue
        for lo, hi in (useful if forward else reversed(useful)):
            start_value, end_value = (lo, hi) if forward else (hi, lo)
            if scan_direction == "horizontal":
                start_grid = (fixed, start_value)
                end_grid = (fixed, end_value)
            else:
                start_grid = (start_value, fixed)
                end_grid = (end_value, fixed)
            coverage_lines_grid.append((start_grid, end_grid, forward))
        forward = not forward

    if not coverage_lines_grid:
        raise RuntimeError("No usable coverage lines in the safe-water region")

    print("\n========================================")
    print("COVERAGE")
    print("========================================")
    print("Scan direction:", scan_direction)
    print("Number of coverage lines:", len(coverage_lines_grid))


    # ============================================================
    # BASIC GEOMETRY HELPERS
    # ============================================================

    def sample_straight_segment(
        start_xy,
        end_xy,
        yaw,
        step_size,
    ):
        distance = math.hypot(
            end_xy[0] - start_xy[0],
            end_xy[1] - start_xy[1],
        )

        if distance <= 1e-12:
            return np.asarray(
                [[start_xy[0], start_xy[1], yaw]],
                dtype=np.float64,
            )

        count = max(
            1,
            int(math.ceil(distance / step_size)),
        )

        fractions = np.linspace(
            0.0,
            1.0,
            count + 1,
        )

        x = (
            start_xy[0]
            + fractions * (end_xy[0] - start_xy[0])
        )

        y = (
            start_xy[1]
            + fractions * (end_xy[1] - start_xy[1])
        )

        yaw_values = np.full(
            count + 1,
            yaw,
        )

        return np.column_stack(
            (x, y, yaw_values)
        )


    def align_yaw(yaw, reference_yaw):
        return yaw + 2.0 * math.pi * round(
            (reference_yaw - yaw)
            / (2.0 * math.pi)
        )


    def wrapped_yaw_difference(yaw_a, yaw_b):
        return math.atan2(
            math.sin(yaw_b - yaw_a),
            math.cos(yaw_b - yaw_a),
        )


    def position_distance(point_a, point_b):
        return math.hypot(
            point_b[0] - point_a[0],
            point_b[1] - point_a[1],
        )


    def retreat_point(point, direction, distance):
        """
        Move a point inward along the coverage line.

        direction is a unit vector pointing in the travel
        direction of the line.
        """
        return np.asarray(
            [
                point[0] - direction[0] * distance,
                point[1] - direction[1] * distance,
            ],
            dtype=np.float64,
        )


    def generate_dubins(
        start_xy,
        start_yaw,
        goal_xy,
        goal_yaw,
        turning_radius,
    ):
        return generate_dubins_path(
            [
                float(start_xy[0]),
                float(start_xy[1]),
                float(start_yaw),
            ],
            [
                float(goal_xy[0]),
                float(goal_xy[1]),
                float(goal_yaw),
            ],
            turning_radius,
            STRAIGHT_STEP,
            CURVE_STEP,
        )


    # ============================================================
    # A* LAST-RESORT CONNECTOR
    # ============================================================

    def astar_local(start_xy, goal_xy):
        """
        Local A* in the already safe-water component.

        This is deliberately a last resort. It does not replace
        Dubins as the normal transition method.
        """

        sr, sc = world_to_grid(*start_xy)
        gr, gc = world_to_grid(*goal_xy)

        if not is_component_safe(sr, sc):
            return None

        if not is_component_safe(gr, gc):
            return None

        center_r = int(round((sr + gr) / 2.0))
        center_c = int(round((sc + gc) / 2.0))

        base_distance = max(
            abs(sr - gr),
            abs(sc - gc),
        )

        initial = int(
            math.ceil(
                ASTAR_INITIAL_MARGIN_METERS
                / resolution
            )
        )

        step = int(
            math.ceil(
                ASTAR_MARGIN_STEP_METERS
                / resolution
            )
        )

        maximum = int(
            math.ceil(
                ASTAR_MAX_MARGIN_METERS
                / resolution
            )
        )

        for margin in range(
            initial,
            maximum + step,
            step,
        ):

            radius = max(
                margin,
                base_distance // 2 + 10,
            )

            local_r_min = max(
                row_min,
                center_r - radius,
            )
            local_r_max = min(
                row_max,
                center_r + radius,
            )
            local_c_min = max(
                col_min,
                center_c - radius,
            )
            local_c_max = min(
                col_max,
                center_c + radius,
            )

            neighbors = (
                (-1, 0, 1.0),
                (1, 0, 1.0),
                (0, -1, 1.0),
                (0, 1, 1.0),
                (-1, -1, math.sqrt(2)),
                (-1, 1, math.sqrt(2)),
                (1, -1, math.sqrt(2)),
                (1, 1, math.sqrt(2)),
            )

            open_heap = [(0.0, (sr, sc))]
            came_from = {}
            g_score = {(sr, sc): 0.0}
            closed = set()

            while open_heap:

                _, current = heapq.heappop(open_heap)

                if current in closed:
                    continue

                if current == (gr, gc):

                    cells = [current]

                    while current in came_from:
                        current = came_from[current]
                        cells.append(current)

                    cells.reverse()

                    points = []

                    for row, col in cells:
                        points.append(
                            [
                                *grid_to_world(row, col),
                                0.0,
                            ]
                        )

                    return np.asarray(
                        points,
                        dtype=np.float64,
                    )

                closed.add(current)

                cr, cc = current

                for dr, dc, cost in neighbors:

                    nr = cr + dr
                    nc = cc + dc

                    if (
                        nr < local_r_min
                        or nr > local_r_max
                        or nc < local_c_min
                        or nc > local_c_max
                    ):
                        continue

                    if not is_component_safe(nr, nc):
                        continue

                    if dr != 0 and dc != 0:

                        if not (
                            is_component_safe(cr + dr, cc)
                            and is_component_safe(cr, cc + dc)
                        ):
                            continue

                    neighbor = (nr, nc)

                    tentative = (
                        g_score[current] + cost
                    )

                    if tentative < g_score.get(
                        neighbor,
                        float("inf"),
                    ):

                        came_from[neighbor] = current
                        g_score[neighbor] = tentative

                        f_score = (
                            tentative
                            + math.hypot(
                                nr - gr,
                                nc - gc,
                            )
                        )

                        heapq.heappush(
                            open_heap,
                            (f_score, neighbor),
                        )

        return None


    def simplify_polyline(path):
        """
        Remove unnecessary A* grid points by keeping only points
        where the travel direction changes.
        """

        if path is None or len(path) <= 2:
            return path

        simplified = [path[0]]

        previous_direction = None

        for i in range(1, len(path)):

            dx = (
                path[i, 0]
                - path[i - 1, 0]
            )
            dy = (
                path[i, 1]
                - path[i - 1, 1]
            )

            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue

            direction = (
                round(math.atan2(dy, dx), 3)
            )

            if (
                previous_direction is not None
                and direction != previous_direction
            ):
                simplified.append(path[i - 1])

            previous_direction = direction

        simplified.append(path[-1])

        return np.asarray(
            simplified,
            dtype=np.float64,
        )


    def smooth_polyline(path):
        """
        Replace sharp A* corners with short circular-looking
        quadratic interpolation only when the smoothed path
        remains inside safe water.

        This is only for the last-resort A* path.
        """

        if path is None or len(path) < 3:
            return path

        result = [path[0]]

        for i in range(1, len(path) - 1):

            p0 = path[i - 1, :2]
            p1 = path[i, :2]
            p2 = path[i + 1, :2]

            v0 = p0 - p1
            v1 = p2 - p1

            n0 = np.linalg.norm(v0)
            n1 = np.linalg.norm(v1)

            if n0 < 1e-9 or n1 < 1e-9:
                continue

            trim = min(
                2.0,
                0.25 * n0,
                0.25 * n1,
            )

            a = p1 + v0 / n0 * trim
            b = p1 + v1 / n1 * trim

            samples = []

            for t in np.linspace(0.0, 1.0, 5):
                u = 1.0 - t

                x = (
                    u * u * a[0]
                    + 2.0 * u * t * p1[0]
                    + t * t * b[0]
                )

                y = (
                    u * u * a[1]
                    + 2.0 * u * t * p1[1]
                    + t * t * b[1]
                )

                samples.append([x, y, 0.0])

            candidate = np.asarray(
                samples,
                dtype=np.float64,
            )

            if path_is_component_safe(candidate):

                result.append(
                    [
                        a[0],
                        a[1],
                        0.0,
                    ]
                )

                result.extend(candidate[1:-1])

                result.append(
                    [
                        b[0],
                        b[1],
                        0.0,
                    ]
                )

            else:
                result.append(path[i])

        result.append(path[-1])

        result = np.asarray(
            result,
            dtype=np.float64,
        )

        for i in range(len(result) - 1):

            dx = (
                result[i + 1, 0]
                - result[i, 0]
            )
            dy = (
                result[i + 1, 1]
                - result[i, 1]
            )

            if math.hypot(dx, dy) > 1e-9:
                result[i, 2] = math.atan2(dy, dx)

        if len(result) > 1:
            result[-1, 2] = result[-2, 2]

        return result


    # ============================================================
    # CONVERT COVERAGE LINES TO WORLD COORDINATES
    # ============================================================

    coverage_lines = []

    for start_grid, end_grid, left_to_right in coverage_lines_grid:

        start_xy = np.asarray(
            grid_to_world(*start_grid),
            dtype=np.float64,
        )

        end_xy = np.asarray(
            grid_to_world(*end_grid),
            dtype=np.float64,
        )

        delta = end_xy - start_xy
        length = float(np.linalg.norm(delta))

        if length < 2.0 * resolution:
            continue

        yaw = math.atan2(
            delta[1],
            delta[0],
        )

        direction = delta / length

        coverage_lines.append(
            {
                "start": start_xy,
                "end": end_xy,
                "yaw": yaw,
                "direction": direction,
                "length": length,
            }
        )

    if len(coverage_lines) < 2:
        raise RuntimeError(
            "Not enough usable coverage lines in the selected "
            "safe-water region."
        )


    # ============================================================
    # DUBINS TRANSITION SEARCH
    # ============================================================

    def find_safe_dubins_transition(
        previous_line,
        current_line,
    ):
        """
        Try the normal line endpoints first.

        If the turn is unsafe, progressively retreat both endpoints
        into the water. The first collision-free Dubins path wins.

        This makes Dubins the normal solution instead of A*.
        """

        previous_length = previous_line["length"]
        current_length = current_line["length"]

        max_previous_retreat = max(
            0.0,
            previous_length
            * (1.0 - MIN_COVERAGE_FRACTION),
        )

        max_current_retreat = max(
            0.0,
            current_length
            * (1.0 - MIN_COVERAGE_FRACTION),
        )

        requested_max = max(
            TURN_SEARCH_MAX_FACTOR * turning_radius,
            2.0 * turning_radius,
        )

        max_previous_retreat = min(
            max_previous_retreat,
            requested_max,
        )

        max_current_retreat = min(
            max_current_retreat,
            requested_max,
        )

        previous_steps = int(
            math.floor(
                max_previous_retreat
                / TURN_SEARCH_STEP_METERS
            )
        )

        current_steps = int(
            math.floor(
                max_current_retreat
                / TURN_SEARCH_STEP_METERS
            )
        )

        previous_distances = [
            i * TURN_SEARCH_STEP_METERS
            for i in range(previous_steps + 1)
        ]

        current_distances = [
            i * TURN_SEARCH_STEP_METERS
            for i in range(current_steps + 1)
        ]

        best = None

        for d_prev in previous_distances:

            previous_end = retreat_point(
                previous_line["end"],
                previous_line["direction"],
                d_prev,
            )

            if not path_is_component_safe(
                np.asarray(
                    [
                        [
                            previous_end[0],
                            previous_end[1],
                            previous_line["yaw"],
                        ]
                    ]
                )
            ):
                continue

            for d_curr in current_distances:

                current_start = retreat_point(
                    current_line["start"],
                    -current_line["direction"],
                    d_curr,
                )

                if not path_is_component_safe(
                    np.asarray(
                        [
                            [
                                current_start[0],
                                current_start[1],
                                current_line["yaw"],
                            ]
                        ]
                    )
                ):
                    continue

                transition = generate_dubins(
                    previous_end,
                    previous_line["yaw"],
                    current_start,
                    current_line["yaw"],
                    turning_radius,
                )

                if not path_is_component_safe(transition):
                    continue

                retreat_cost = d_prev + d_curr

                if best is None or retreat_cost < best["cost"]:
                    best = {
                        "transition": transition,
                        "previous_end": previous_end,
                        "current_start": current_start,
                        "previous_retreat": d_prev,
                        "current_retreat": d_curr,
                        "cost": retreat_cost,
                    }

        return best


    # ============================================================
    # GENERATE FINAL PATH
    # ============================================================

    print()
    print("========================================")
    print("PATH GENERATION")
    print("========================================")

    world_path_parts = []
    coverage_lines_world = []
    dubins_transitions = []
    astar_transitions = []

    first_line = coverage_lines[0]

    first_segment = sample_straight_segment(
        first_line["start"],
        first_line["end"],
        first_line["yaw"],
        STRAIGHT_STEP,
    )

    world_path_parts.append(first_segment)

    coverage_lines_world.append(
        np.vstack(
            (
                first_line["start"],
                first_line["end"],
            )
        )
    )

    previous_line = first_line

    for index in range(1, len(coverage_lines)):

        current_line = coverage_lines[index]

        print()
        print(f"Transition {index}")

        result = find_safe_dubins_transition(
            previous_line,
            current_line,
        )

        if result is not None:

            transition = result["transition"]

            print(
                "  Method: DUBINS"
            )

            print(
                "  Previous endpoint retreat:",
                round(result["previous_retreat"], 2),
                "m",
            )

            print(
                "  Current endpoint retreat:",
                round(result["current_retreat"], 2),
                "m",
            )

            previous_start = previous_line["start"]
            previous_end = result["previous_end"]
            current_start = result["current_start"]

            # The previous coverage line was initially sampled all the way to
            # its nominal end. A collision-free turn may retreat that endpoint,
            # so replace the line tail and join the Dubins path at its true start.
            trimmed_previous_segment = sample_straight_segment(
                previous_start,
                previous_end,
                previous_line["yaw"],
                STRAIGHT_STEP,
            )
            if index > 1:
                trimmed_previous_segment = trimmed_previous_segment[1:]

            world_path_parts[-1] = trimmed_previous_segment

            if not path_is_component_safe(world_path_parts[-1]):
                raise RuntimeError(
                    f"Transition {index}: trimmed coverage line "
                    "failed safe-water validation."
                )

            start_pose = np.asarray(
                [
                    previous_end[0],
                    previous_end[1],
                    previous_line["yaw"],
                ]
            )

            print(
                "  Interface diagnostic:",
                "start_pose=", start_pose,
                "transition[0]=", transition[0],
                "transition[1]=", transition[1],
                "transition[-1]=", transition[-1],
                "next_line_start=", current_start,
            )
            print(
                "  Interface distances:",
                "line_to_transition_start=",
                position_distance(
                    world_path_parts[-1][-1],
                    transition[0],
                ),
                "transition_start_to_next_sample=",
                position_distance(transition[0], transition[1]),
                "transition_end_to_line_start=",
                position_distance(transition[-1], current_start),
            )
            print(
                "  Interface wrapped yaw differences:",
                "line_to_transition_start=",
                wrapped_yaw_difference(
                    world_path_parts[-1][-1, 2],
                    transition[0, 2],
                ),
                "transition_start_to_next_sample=",
                wrapped_yaw_difference(
                    transition[0, 2],
                    transition[1, 2],
                ),
                "transition_end_to_next_line=",
                wrapped_yaw_difference(
                    transition[-1, 2],
                    current_line["yaw"],
                ),
            )

            if (
                position_distance(
                    world_path_parts[-1][-1],
                    transition[0],
                ) > 1e-8
                or position_distance(
                    transition[-1],
                    current_start,
                ) > 1e-8
            ):
                raise RuntimeError(
                    f"Transition {index}: Dubins path position "
                    "does not meet the coverage-line endpoints."
                )

            if (
                abs(
                    wrapped_yaw_difference(
                        previous_line["yaw"],
                        transition[0, 2],
                    )
                ) > 1e-8
                or abs(
                    wrapped_yaw_difference(
                        transition[-1, 2],
                        current_line["yaw"],
                    )
                ) > 1e-8
            ):
                raise RuntimeError(
                    f"Transition {index}: Dubins path heading "
                    "does not match the coverage-line headings."
                )

            dubins_transitions.append(
                transition
            )

            world_path_parts.append(
                transition[1:]
            )

            # Continue the current coverage line from the
            # endpoint where the Dubins turn actually arrives.
            straight_segment = sample_straight_segment(
                current_start,
                current_line["end"],
                current_line["yaw"],
                STRAIGHT_STEP,
            )

            if not path_is_component_safe(
                straight_segment
            ):
                raise RuntimeError(
                    f"Transition {index}: current coverage line "
                    "became unsafe after Dubins endpoint retreat."
                )

            world_path_parts.append(
                straight_segment[1:]
            )

            coverage_lines_world.append(
                np.vstack(
                    (
                        current_start,
                        current_line["end"],
                    )
                )
            )

        else:

            print(
                "  No collision-free Dubins turn found "
                "with endpoint retreat."
            )

            if not USE_ASTAR_FALLBACK:
                raise RuntimeError(
                    f"Transition {index} is impossible for the "
                    f"requested turning radius of {turning_radius} m."
                )

            previous_end = previous_line["end"]
            current_start = current_line["start"]

            connector = astar_local(
                previous_end,
                current_start,
            )

            if connector is None:
                raise RuntimeError(
                    f"Transition {index} is impossible: "
                    "no collision-free Dubins transition and "
                    "no local A* connector."
                )

            simplified = simplify_polyline(connector)
            smoothed = smooth_polyline(simplified)

            if not path_is_component_safe(smoothed):
                smoothed = simplified

            print(
                "  Method: A* LAST RESORT"
            )
            print(
                "  Raw A* points:",
                len(connector),
            )
            print(
                "  Simplified points:",
                len(simplified),
            )
            print(
                "  Final A* points:",
                len(smoothed),
            )

            print(
                "  Interface diagnostic:",
                "start_pose=",
                [
                    previous_line["end"][0],
                    previous_line["end"][1],
                    previous_line["yaw"],
                ],
                "connector[0]=", smoothed[0],
                "connector[1]=", smoothed[1],
                "connector[-1]=", smoothed[-1],
                "next_line_start=", current_start,
            )
            print(
                "  Interface distances:",
                "line_to_connector_start=",
                position_distance(
                    world_path_parts[-1][-1],
                    smoothed[0],
                ),
                "connector_start_to_next_sample=",
                position_distance(smoothed[0], smoothed[1]),
                "connector_end_to_line_start=",
                position_distance(smoothed[-1], current_start),
            )
            print(
                "  Interface wrapped yaw differences:",
                "line_to_connector_start=",
                wrapped_yaw_difference(
                    previous_line["yaw"],
                    smoothed[0, 2],
                ),
                "connector_start_to_next_sample=",
                wrapped_yaw_difference(
                    smoothed[0, 2],
                    smoothed[1, 2],
                ),
                "connector_end_to_next_line=",
                wrapped_yaw_difference(
                    smoothed[-1, 2],
                    current_line["yaw"],
                ),
            )

            astar_transitions.append(
                smoothed
            )

            world_path_parts.append(
                smoothed[1:]
            )

            straight_segment = sample_straight_segment(
                current_line["start"],
                current_line["end"],
                current_line["yaw"],
                STRAIGHT_STEP,
            )

            if not path_is_component_safe(
                straight_segment
            ):
                raise RuntimeError(
                    f"Coverage line {index} is not safe."
                )

            world_path_parts.append(
                straight_segment[1:]
            )

            coverage_lines_world.append(
                np.vstack(
                    (
                        current_line["start"],
                        current_line["end"],
                    )
                )
            )

        previous_line = {
            **current_line,
            "start": current_start,
        }


    # ============================================================
    # FINAL PATH
    # ============================================================

    world_path = np.vstack(
        world_path_parts
    ).astype(np.float64)


    # Remove consecutive duplicate points.
    if len(world_path) > 1:

        deltas = np.linalg.norm(
            np.diff(
                world_path[:, :2],
                axis=0,
            ),
            axis=1,
        )

        keep = np.concatenate(
            (
                [True],
                deltas > 1e-8,
            )
        )

        world_path = world_path[keep]


    # Recalculate yaw from the final trajectory.
    for i in range(len(world_path) - 1):

        dx = (
            world_path[i + 1, 0]
            - world_path[i, 0]
        )

        dy = (
            world_path[i + 1, 1]
            - world_path[i, 1]
        )

        distance = math.hypot(dx, dy)

        if distance > 1e-9:
            world_path[i, 2] = math.atan2(dy, dx)

    if len(world_path) > 1:
        world_path[-1, 2] = world_path[-2, 2]


    # ============================================================
    # FINAL VALIDATION
    # ============================================================

    print()
    print("========================================")
    print("FINAL VALIDATION")
    print("========================================")

    if not path_is_component_safe(world_path):
        raise RuntimeError(
            "FINAL PATH FAILED SAFE-WATER VALIDATION."
        )

    if not np.all(np.isfinite(world_path)):
        raise RuntimeError(
            "FINAL PATH CONTAINS NON-FINITE VALUES."
        )

    print("Safety clearance check: PASS")
    print("Finite-value check: PASS")
    print("Final waypoint count:", len(world_path))
    print("Dubins transitions:", len(dubins_transitions))
    print("A* last-resort transitions:", len(astar_transitions))


    if not _trial:
        # ============================================================
        # SAVE NPY
        # ============================================================

        np.save(
            path_npy,
            world_path,
        )


        # ============================================================
        # SAVE CSV
        # ============================================================

        with path_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "waypoint_id",
                    "x",
                    "y",
                    "yaw",
                ]
            )

            for i, point in enumerate(world_path):

                writer.writerow(
                    [
                        i,
                        point[0],
                        point[1],
                        point[2],
                    ]
                )


    # ============================================================
    # PATH STATISTICS
    # ============================================================

    if len(world_path) > 1:

        total_distance = float(
            np.sum(
                np.linalg.norm(
                    np.diff(
                        world_path[:, :2],
                        axis=0,
                    ),
                    axis=1,
                )
            )
        )

    else:
        total_distance = 0.0

    print()
    print("========================================")
    print("PATH OUTPUT")
    print("========================================")
    print("NPY:", path_npy)
    print("CSV:", path_csv)
    print("Total path distance:", round(total_distance, 2), "m")
    print("First waypoint:", world_path[0])
    print("Last waypoint:", world_path[-1])


    if create_plot and not _trial:
        # ============================================================
        # PLOT
        # ============================================================

        fig, ax = plt.subplots(
            figsize=(14, 9)
        )

        ax.imshow(
            occupancy,
            origin="lower",
            extent=[
                map_x_min,
                map_x_max,
                map_y_min,
                map_y_max,
            ],
            cmap="binary",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            aspect="equal",
        )


        # ------------------------------------------------------------
        # Selected search window
        # ------------------------------------------------------------

        rectangle_x = [
            coverage_x_min,
            coverage_x_max,
            coverage_x_max,
            coverage_x_min,
            coverage_x_min,
        ]

        rectangle_y = [
            coverage_y_min,
            coverage_y_min,
            coverage_y_max,
            coverage_y_max,
            coverage_y_min,
        ]

        ax.plot(
            rectangle_x,
            rectangle_y,
            linestyle="--",
            linewidth=2,
            label="Coverage search window",
        )


        # ------------------------------------------------------------
        # Safe-water boundary
        # ------------------------------------------------------------

        safe_rows, safe_cols = np.where(
            component_mask
        )

        if len(safe_rows) > 0:

            safe_x, safe_y = grid_to_world(
                safe_rows[0],
                safe_cols[0],
            )

            # Draw the safe-water boundary with a contour.
            # Contour is based on the actual safe-water mask.
            ax.contour(
                component_mask.astype(float),
                levels=[0.5],
                origin="lower",
                extent=[
                    map_x_min,
                    map_x_max,
                    map_y_min,
                    map_y_max,
                ],
                colors=["tab:green"],
                linewidths=SAFE_BOUNDARY_LINE_WIDTH,
                alpha=0.95,
            )


        # ------------------------------------------------------------
        # Coverage lines
        # ------------------------------------------------------------

        for i, line in enumerate(coverage_lines_world):

            ax.plot(
                line[:, 0],
                line[:, 1],
                color="tab:blue",
                linewidth=0.65,
                alpha=0.75,
                label=(
                    "Water coverage lines"
                    if i == 0
                    else None
                ),
            )


        # ------------------------------------------------------------
        # Dubins transitions
        # ------------------------------------------------------------

        for i, transition in enumerate(dubins_transitions):

            ax.plot(
                transition[:, 0],
                transition[:, 1],
                color="tab:orange",
                linewidth=DUBINS_LINE_WIDTH,
                alpha=0.95,
                label=(
                    "Dubins transitions"
                    if i == 0
                    else None
                ),
            )


        # ------------------------------------------------------------
        # A* last-resort transitions
        # ------------------------------------------------------------

        for i, transition in enumerate(astar_transitions):

            ax.plot(
                transition[:, 0],
                transition[:, 1],
                color="tab:red",
                linewidth=ASTAR_LINE_WIDTH,
                alpha=0.9,
                label=(
                    "A* last-resort connectors"
                    if i == 0
                    else None
                ),
            )


        # ------------------------------------------------------------
        # Final trajectory
        # ------------------------------------------------------------

        ax.plot(
            world_path[:, 0],
            world_path[:, 1],
            color="black",
            linewidth=TRAJECTORY_LINE_WIDTH,
            alpha=0.85,
            label="Final trajectory",
        )


        # ------------------------------------------------------------
        # Start / end markers
        # ------------------------------------------------------------

        start = world_path[0]
        goal = world_path[-1]

        ax.scatter(
            start[0],
            start[1],
            s=150,
            marker="o",
            color="tab:green",
            edgecolors="black",
            linewidths=1.5,
            zorder=20,
            label="Trajectory start",
        )

        ax.scatter(
            goal[0],
            goal[1],
            s=180,
            marker="X",
            color="tab:red",
            edgecolors="black",
            linewidths=1.5,
            zorder=20,
            label="Trajectory end",
        )

        ax.annotate(
            "START",
            xy=(start[0], start[1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="tab:green",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="tab:green",
                alpha=0.9,
            ),
            zorder=21,
        )

        ax.annotate(
            "END",
            xy=(goal[0], goal[1]),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="tab:red",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="tab:red",
                alpha=0.9,
            ),
            zorder=21,
        )


        # ------------------------------------------------------------
        # Plot limits with padding
        # ------------------------------------------------------------

        plot_x_min = max(
            map_x_min,
            coverage_x_min - PLOT_PADDING_METERS,
        )

        plot_x_max = min(
            map_x_max,
            coverage_x_max + PLOT_PADDING_METERS,
        )

        plot_y_min = max(
            map_y_min,
            coverage_y_min - PLOT_PADDING_METERS,
        )

        plot_y_max = min(
            map_y_max,
            coverage_y_max + PLOT_PADDING_METERS,
        )

        # Include trajectory if it extends outside the selected rectangle.
        plot_x_min = max(
            map_x_min,
            min(
                plot_x_min,
                float(np.min(world_path[:, 0])) - 10.0,
            ),
        )

        plot_x_max = min(
            map_x_max,
            max(
                plot_x_max,
                float(np.max(world_path[:, 0])) + 10.0,
            ),
        )

        plot_y_min = max(
            map_y_min,
            min(
                plot_y_min,
                float(np.min(world_path[:, 1])) - 10.0,
            ),
        )

        plot_y_max = min(
            map_y_max,
            max(
                plot_y_max,
                float(np.max(world_path[:, 1])) + 10.0,
            ),
        )

        ax.set_xlim(
            plot_x_min,
            plot_x_max,
        )

        ax.set_ylim(
            plot_y_min,
            plot_y_max,
        )

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

        ax.set_title(
            "Sydney Regatta — Safe-Water Dubins Coverage"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend(
            loc="best",
        )

        plt.tight_layout()

        plt.savefig(
            path_png,
            dpi=200,
            bbox_inches="tight",
        )

        print("PNG:", path_png)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return {
        "world_path": world_path,
        "coverage_lines": coverage_lines_world,
        "dubins_transitions": dubins_transitions,
        "astar_transitions": astar_transitions,
        "component_mask": component_mask,
        "total_distance": total_distance,
        "coverage_length": sum(float(np.linalg.norm(line[-1] - line[0])) for line in coverage_lines_world),
        "scan_direction": scan_direction,
        "coverage_bounds": (
            coverage_x_min,
            coverage_x_max,
            coverage_y_min,
            coverage_y_max,
        ),
        "path_width": path_width,
        "turning_radius": turning_radius,
        "path_csv": path_csv,
        "path_npy": path_npy,
        "path_png": path_png,
    }



def main():
    """Run the original interactive planner workflow."""

    (
        coverage_x_min,
        coverage_x_max,
        coverage_y_min,
        coverage_y_max,
    ) = select_coverage_rectangle()

    print()
    print("Selected rectangle:")
    print(
        "X:",
        coverage_x_min,
        "to",
        coverage_x_max,
    )
    print(
        "Y:",
        coverage_y_min,
        "to",
        coverage_y_max,
    )

    try:
        path_width = float(
            input(
                "Enter coverage path width [m] "
                "(example 10): "
            )
        )
    except ValueError as error:
        raise RuntimeError(
            "Path width must be a number."
        ) from error

    try:
        turning_radius = float(
            input(
                "Enter minimum turning radius [m]: "
            )
        )
    except ValueError as error:
        raise RuntimeError(
            "Minimum turning radius must be a number."
        ) from error

    result = generate_coverage_plan(
        coverage_x_min,
        coverage_x_max,
        coverage_y_min,
        coverage_y_max,
        path_width,
        turning_radius,
        show_plot=True,
    )

    print()
    print("========================================")
    print("PATH CONFIRMATION")
    print("========================================")

    while True:
        confirmation = input(
            "Use this generated path? [y/n]: "
        ).strip().lower()

        if confirmation in ("y", "yes"):
            print(
                "Path accepted. "
                "The mission can now be launched."
            )
            return result

        if confirmation in ("n", "no"):
            print(
                "Path rejected. "
                "Returning to path selection."
            )
            raise SystemExit(2)

        print(
            "Invalid answer. Enter y or n."
        )


if __name__ == "__main__":
    main()