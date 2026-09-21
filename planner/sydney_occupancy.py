import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# ============================================================
# CONFIG
# ============================================================

dae_path = "/home/boat/.gz/fuel/fuel.gazebosim.org/openrobotics/models/sydney_regatta/3/meshes/sydney_regatta_shore.dae"

# Focused area around the boat
x_min = -700.0
x_max = -300.0

y_min = 100.0
y_max = 400.0

resolution = 2.0

boat_x = -533.7402476486775
boat_y = 161.7656784075805

png_path = "/home/boat/sydney_local_occupancy.png"
npy_path = "/home/boat/sydney_local_occupancy.npy"

# ============================================================
# READ DAE
# ============================================================

tree = ET.parse(dae_path)
root = tree.getroot()

ns = {
    "c": "http://www.collada.org/2005/11/COLLADASchema"
}

# ------------------------------------------------------------
# Load terrain vertices
# ------------------------------------------------------------

arr = root.find(
    ".//c:float_array[@id='SydneyTerrainCollider-POSITION-array']",
    ns
)

if arr is None:
    raise RuntimeError(
        "Could not find SydneyTerrainCollider POSITION array"
    )

values = np.array(
    list(map(float, arr.text.split())),
    dtype=np.float64
)

vertices = values.reshape(-1, 3)

# DAE node scale = 0.001
vertices *= 0.001

print("Vertices:", len(vertices))
print(
    "X:",
    vertices[:, 0].min(),
    "to",
    vertices[:, 0].max()
)
print(
    "Y:",
    vertices[:, 1].min(),
    "to",
    vertices[:, 1].max()
)
print(
    "Z:",
    vertices[:, 2].min(),
    "to",
    vertices[:, 2].max()
)

# ------------------------------------------------------------
# Load triangle indices
# ------------------------------------------------------------

triangles_element = root.find(
    ".//c:triangles",
    ns
)

if triangles_element is None:
    raise RuntimeError(
        "Could not find triangle element"
    )

triangle_count_expected = int(
    triangles_element.attrib["count"]
)

p = triangles_element.find(
    "c:p",
    ns
)

if p is None:
    raise RuntimeError(
        "Could not find triangle index list"
    )

indices_raw = np.array(
    list(map(int, p.text.split())),
    dtype=np.int64
)

# Inputs:
# VERTEX offset = 0
# NORMAL offset = 1
# TEXCOORD offset = 2
#
# Therefore:
# [vertex_index, normal_index, texcoord_index]

stride = 3

vertex_indices = indices_raw[0::stride]

if len(vertex_indices) % 3 != 0:
    raise RuntimeError(
        "Triangle vertex index count is invalid"
    )

triangles = vertex_indices.reshape(-1, 3)

print("Triangles:", len(triangles))
print(
    "Expected triangles:",
    triangle_count_expected
)

# ============================================================
# CREATE GRID
# ============================================================

grid_width = int(
    np.ceil(
        (x_max - x_min) / resolution
    )
)

grid_height = int(
    np.ceil(
        (y_max - y_min) / resolution
    )
)

grid = np.zeros(
    (grid_height, grid_width),
    dtype=np.uint8
)

print(
    "Grid:",
    grid_width,
    "x",
    grid_height
)

# Cell center coordinates

x_centers = (
    x_min
    + (np.arange(grid_width) + 0.5)
    * resolution
)

y_centers = (
    y_min
    + (np.arange(grid_height) + 0.5)
    * resolution
)

# ============================================================
# RASTERIZE TRIANGLES INTO OCCUPANCY GRID
# ============================================================

triangles_in_roi = 0

for tri_indices in triangles:

    tri_xyz = vertices[
        tri_indices
    ]

    # Only XY projection is needed
    tri_xy = tri_xyz[:, :2]

    tx_min = tri_xy[:, 0].min()
    tx_max = tri_xy[:, 0].max()

    ty_min = tri_xy[:, 1].min()
    ty_max = tri_xy[:, 1].max()

    # Skip triangle if completely outside ROI

    if tx_max < x_min:
        continue

    if tx_min > x_max:
        continue

    if ty_max < y_min:
        continue

    if ty_min > y_max:
        continue

    triangles_in_roi += 1

    # Determine affected grid cells

    ix0 = max(
        0,
        int(
            np.floor(
                (tx_min - x_min)
                / resolution
            )
        )
    )

    ix1 = min(
        grid_width - 1,
        int(
            np.floor(
                (tx_max - x_min)
                / resolution
            )
        )
    )

    iy0 = max(
        0,
        int(
            np.floor(
                (ty_min - y_min)
                / resolution
            )
        )
    )

    iy1 = min(
        grid_height - 1,
        int(
            np.floor(
                (ty_max - y_min)
                / resolution
            )
        )
    )

    if ix1 < ix0 or iy1 < iy0:
        continue

    # Grid-cell centers inside triangle bounding box

    xs = x_centers[
        ix0:ix1 + 1
    ]

    ys = y_centers[
        iy0:iy1 + 1
    ]

    xx, yy = np.meshgrid(
        xs,
        ys
    )

    pts = np.column_stack(
        [
            xx.ravel(),
            yy.ravel()
        ]
    )

    triangle_path = Path(
        tri_xy
    )

    inside = (
        triangle_path
        .contains_points(
            pts,
            radius=1e-9
        )
    )

    inside = inside.reshape(
        len(ys),
        len(xs)
    )

    subgrid = grid[
        iy0:iy1 + 1,
        ix0:ix1 + 1
    ]

    subgrid[inside] = 1

# ============================================================
# BOAT GRID POSITION
# ============================================================

boat_ix = int(
    np.floor(
        (boat_x - x_min)
        / resolution
    )
)

boat_iy = int(
    np.floor(
        (boat_y - y_min)
        / resolution
    )
)

print()
print(
    "Triangles intersecting ROI:",
    triangles_in_roi
)

occupied = int(
    np.sum(grid)
)

free = int(
    grid.size - occupied
)

print(
    "Occupied cells:",
    occupied
)

print(
    "Free cells:",
    free
)

print()
print("Boat:")
print(
    "x =",
    boat_x
)
print(
    "y =",
    boat_y
)
print(
    "grid ix =",
    boat_ix
)
print(
    "grid iy =",
    boat_iy
)

if (
    0 <= boat_ix < grid_width
    and
    0 <= boat_iy < grid_height
):

    print(
        "Boat cell occupancy =",
        int(
            grid[
                boat_iy,
                boat_ix
            ]
        )
    )

else:

    print(
        "WARNING: Boat is outside grid"
    )

# ============================================================
# SAVE GRID
# ============================================================

np.save(
    npy_path,
    grid
)

# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(12, 8)
)

plt.imshow(
    grid,
    origin="lower",
    extent=[
        x_min,
        x_max,
        y_min,
        y_max
    ],
    cmap="binary",
    interpolation="nearest",
    vmin=0,
    vmax=1,
    aspect="equal"
)

plt.scatter(
    boat_x,
    boat_y,
    marker="x",
    s=120,
    linewidths=2.5,
    label="Boat"
)

plt.xlabel(
    "X [m]"
)

plt.ylabel(
    "Y [m]"
)

plt.title(
    "Sydney Regatta Local Occupancy Grid"
)

plt.xlim(
    x_min,
    x_max
)

plt.ylim(
    y_min,
    y_max
)

plt.grid(
    True,
    alpha=0.35
)

plt.legend()

plt.tight_layout()

plt.savefig(
    png_path,
    dpi=200
)

print()
print("Saved:")
print(
    png_path
)
print(
    npy_path
)

plt.show()