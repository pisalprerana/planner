# ROS 2 planner

This `ament_python` package provides a Dubins coverage-planning pipeline for
the Sydney Regatta environment.

## Dubins coverage workflow

Run commands from the workspace root after sourcing ROS 2 and the workspace:

```bash
cd ~/vrx_ws
source /opt/ros/<ros-distro>/setup.bash
colcon build --packages-select planner --symlink-install
source install/setup.bash
```

Generate or refresh the occupancy grid and its coordinate metadata:

```bash
PYTHONPATH=src/planner python3 -m planner.sydney_occupancy
```

The occupancy generator requires the Sydney Regatta Gazebo Fuel mesh:

```text
~/.gz/fuel/fuel.gazebosim.org/openrobotics/models/sydney_regatta/
```

Generate a Dubins coverage trajectory:

```bash
PYTHONPATH=src/planner python3 -m planner.sydney_coverage_dubins
```

The planner uses:

```text
planner/data/sydney_local_occupancy.npy
planner/data/sydney_world_coordinates.json
```

It saves the trajectory in `planner/output/`:

```text
sydney_coverage_dubins_path.csv
sydney_coverage_dubins_path.npy
sydney_coverage_dubins_path.png
```

Build or rebuild the package after generating the CSV. The package setup
installs CSV files from `planner/output/` into the package share directory,
where the publisher can find them after ROS installation:

```bash
cd ~/vrx_ws
colcon build --packages-select planner --symlink-install
source install/setup.bash
```

The CSV header is `waypoint_id,x,y,yaw`. Each waypoint is published in the
flattened `Float64MultiArray` format:

```text
[id, x, y, yaw, id, x, y, yaw, ...]
```

## Launch the Dubins pipeline

```bash
ros2 launch planner dubins_pipeline.launch.py
```

This single command starts `gps_imu_tf_broadcaster`,
`waypoint_array_dubins`, and `path_frame_transformer`. The installed
configuration is `config/dubins_pipeline.yaml`; it sets:

- CSV: `sydney_coverage_dubins_path.csv`
- waypoint topic: `/planner/waypoints_dubins`
- publisher rate: `1.0` Hz
- transformer input: `/planner/waypoints_dubins`
- local output topic: `/planner/waypoints_dubins_local`
- source and target frames: `world` and `wamv/wamv/base_link`

The frame transformer consumes the flattened waypoints, transforms both
position and heading, and republishes the same four-value format:

```text
/planner/waypoints_dubins
    -> /planner/waypoints_dubins_local
```

Defaults are source frame `world` and target frame `wamv/wamv/base_link`. The
target matches the default child frame published by
`gps_imu_tf_broadcaster`. The YAML file can be edited to change CSV, topics,
frames, or publish rates; rebuild and source the workspace after changing
package files.

The Python modules are installed under the `planner` namespace. Local generated
data and output paths are resolved relative to the module directory.
