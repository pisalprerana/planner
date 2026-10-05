"""Coordinate utilities for the full Sydney Regatta world."""

import json
import math
from pathlib import Path

import numpy as np


# ROS/Gazebo coordinate-frame name
WORLD_FRAME = 'world'

# Occupancy-grid resolution in metres per cell
DEFAULT_RESOLUTION = 2.0

# Extra space around the terrain limits
DEFAULT_MARGIN = 10.0

# Initial WAM-V position in absolute Gazebo world coordinates
BOAT_START_X = -533.7402476486775
BOAT_START_Y = 161.7656784075805


def calculate_world_bounds(
    vertices,
    resolution=DEFAULT_RESOLUTION,
    margin=DEFAULT_MARGIN,
):
    """Calculate full Sydney bounds from terrain mesh vertices.

    The limits are expanded by ``margin`` and aligned with the
    occupancy-grid resolution.
    """

    vertices = np.asarray(vertices, dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] < 2:
        raise ValueError(
            'Vertices must be a two-dimensional array containing x and y.'
        )

    raw_x_min = float(np.min(vertices[:, 0]))
    raw_x_max = float(np.max(vertices[:, 0]))
    raw_y_min = float(np.min(vertices[:, 1]))
    raw_y_max = float(np.max(vertices[:, 1]))

    x_min = (
        math.floor((raw_x_min - margin) / resolution)
        * resolution
    )

    x_max = (
        math.ceil((raw_x_max + margin) / resolution)
        * resolution
    )

    y_min = (
        math.floor((raw_y_min - margin) / resolution)
        * resolution
    )

    y_max = (
        math.ceil((raw_y_max + margin) / resolution)
        * resolution
    )

    return {
        'frame_id': WORLD_FRAME,
        'resolution': float(resolution),
        'margin': float(margin),
        'x_min': float(x_min),
        'x_max': float(x_max),
        'y_min': float(y_min),
        'y_max': float(y_max),
    }


def grid_shape(coordinates):
    """Calculate occupancy-grid height and width."""

    resolution = coordinates['resolution']

    grid_width = int(
        math.ceil(
            (coordinates['x_max'] - coordinates['x_min'])
            / resolution
        )
    )

    grid_height = int(
        math.ceil(
            (coordinates['y_max'] - coordinates['y_min'])
            / resolution
        )
    )

    return grid_height, grid_width


def world_to_grid(x, y, coordinates):
    """Convert absolute Gazebo world coordinates to grid indices."""

    resolution = coordinates['resolution']

    column = math.floor(
        (x - coordinates['x_min']) / resolution
    )

    row = math.floor(
        (y - coordinates['y_min']) / resolution
    )

    return row, column


def grid_to_world(row, column, coordinates):
    """Convert a grid cell to its absolute world-coordinate centre."""

    resolution = coordinates['resolution']

    x = (
        coordinates['x_min']
        + (column + 0.5) * resolution
    )

    y = (
        coordinates['y_min']
        + (row + 0.5) * resolution
    )

    return x, y


def inside_world(x, y, coordinates):
    """Return True when a world position is inside the full map."""

    return (
        coordinates['x_min']
        <= x
        < coordinates['x_max']
        and coordinates['y_min']
        <= y
        < coordinates['y_max']
    )


def inside_grid(row, column, coordinates):
    """Return True when a cell is inside the occupancy grid."""

    grid_height, grid_width = grid_shape(coordinates)

    return (
        0 <= row < grid_height
        and 0 <= column < grid_width
    )


def save_coordinates(coordinates, output_file):
    """Save the coordinate-frame information as JSON."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(
            coordinates,
            file,
            indent=4,
        )


def load_coordinates(input_file):
    """Load previously generated coordinate-frame information."""

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(
            f'Coordinate metadata not found: {input_path}'
        )

    with input_path.open(
        'r',
        encoding='utf-8',
    ) as file:
        return json.load(file)
