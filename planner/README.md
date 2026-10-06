# Sydney Regatta planner

The `planner` ROS 2 Python package creates a coverage trajectory for the
Sydney Regatta world and makes it available to ROS nodes. Planning operates
on Gazebo world coordinates. The generated CSV stores each waypoint as
`waypoint_id,x,y,yaw`, where `x` and `y` are metres and `yaw` is in radians.

## Requirements

- A sourced ROS 2 installation and a colcon workspace
- Python packages `numpy` and `matplotlib` for map and trajectory generation
- The Sydney Regatta Gazebo Fuel model, including its terrain mesh at:

  ```text
  ~/.gz/fuel/fuel.gazebosim.org/openrobotics/models/sydney_regatta/
  ```

## Generate the map and trajectory

Run these commands from the workspace root (`~/vrx_ws`). Build and source
the package first so its ROS executables and launch files are available:

```bash
cd ~/vrx_ws
source /opt/ros/<ros-distro>/setup.bash
colcon build --packages-select planner --symlink-install
source install/setup.bash
```

Generate the occupancy grid and world-coordinate metadata from the installed
Sydney Regatta terrain mesh:

```bash
PYTHONPATH=src/planner python3 -m planner.sydney_occupancy
```

This writes the occupancy grid and coordinate metadata to:

```text
src/planner/planner/data/sydney_local_occupancy.npy
src/planner/planner/data/sydney_world_coordinates.json
```

Generate the coverage trajectory using those files:

```bash
PYTHONPATH=src/planner python3 -m planner.sydney_coverage_dubins
```

When prompted, select the two opposite corners of the desired coverage
rectangle in the map window. The planner creates straight coverage sweeps
with Dubins transitions and saves:

```text
src/planner/planner/output/sydney_coverage_dubins_path.csv
src/planner/planner/output/sydney_coverage_dubins_path.npy
src/planner/planner/output/sydney_coverage_dubins_path.png
```

The CSV has the header `waypoint_id,x,y,yaw`. Rebuild after generating or
changing the CSV: `setup.py` installs CSV files from `planner/output/` into
the package share directory, where the ROS publishers load them.

```bash
cd ~/vrx_ws
colcon build --packages-select planner --symlink-install
source install/setup.bash
```

## Run the ROS pipeline

Launch all configured nodes with:

```bash
ros2 launch planner dubins_pipeline.launch.py
```

The launch file starts four nodes:

- `gps_imu_tf_broadcaster` publishes the boat transform used by the local
  waypoint transformer.
- `waypoint_array_dubins` reads the CSV and publishes flattened waypoint
  arrays on `/planner/waypoints_dubins`.
- `path_frame_transformer` transforms the waypoint positions and headings
  from the `world` frame to `wamv/wamv/base_link`, publishing the result on
  `/planner/waypoints_dubins_local`.
- `dubins_reference_path_publisher` reads the CSV and publishes a
  `nav_msgs/Path` on `/planner/reference_path`.

The array topics use the repeated four-value format:

```text
[id, x, y, yaw, id, x, y, yaw, ...]
```

The reference-path publisher converts the CSV's Gazebo ENU coordinates to
NED coordinates and sets the path frame to `world_ned`. The array publisher
and frame transformer are a separate output path; they do not use the
reference publisher's NED conversion.

## Configuration

The launch file loads `config/dubins_pipeline.yaml` from the installed
package. It configures the CSV filename, topics, publish rates, and frames.
The defaults are:

| Node | Setting | Default |
|---|---|---|
| `waypoint_array_dubins` | Output topic | `/planner/waypoints_dubins` |
| `waypoint_array_dubins` | Publish rate | `1.0` Hz |
| `dubins_reference_path_publisher` | Output topic | `/planner/reference_path` |
| `dubins_reference_path_publisher` | Output frame | `world_ned` |
| `dubins_reference_path_publisher` | Publish rate | `1.0` Hz |
| `path_frame_transformer` | Input topic | `/planner/waypoints_dubins` |
| `path_frame_transformer` | Output topic | `/planner/waypoints_dubins_local` |
| `path_frame_transformer` | Source / target frames | `world` / `wamv/wamv/base_link` |
| `path_frame_transformer` | Publish rate | `2.0` Hz |

After editing package configuration, rebuild and source the workspace before
launching the pipeline.
