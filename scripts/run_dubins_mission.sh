#!/usr/bin/env bash

workspace="${VRX_WS:-$HOME/vrx_ws}"
planner_repo="$workspace/src/planner"

if [ ! -d "$planner_repo/planner" ]; then
    echo "Planner repository not found: $planner_repo"
    exit 1
fi

source /opt/ros/jazzy/setup.bash

if [ -f "$workspace/install/setup.bash" ]; then
    source "$workspace/install/setup.bash"
fi


# ------------------------------------------------------------
# Interactive path selection and confirmation
# ------------------------------------------------------------

while true; do
    cd "$planner_repo" || exit 1

    PYTHONPATH=. python3 -m planner.sydney_coverage_dubins
    planner_status=$?

    if [ "$planner_status" -eq 0 ]; then
        echo "Selected path accepted."
        break
    fi

    if [ "$planner_status" -eq 2 ]; then
        echo "Restarting interactive path selection..."
        continue
    fi

    echo "Path generation failed with code $planner_status."
    exit "$planner_status"
done

# ------------------------------------------------------------
# Install the newly generated CSV
# ------------------------------------------------------------

cd "$workspace" || exit 1

colcon build \
    --merge-install \
    --symlink-install \
    --packages-select \
    planner

build_status=$?

if [ "$build_status" -ne 0 ]; then
    echo "Build failed. Mission will not be launched."
    exit "$build_status"
fi

source "$workspace/install/setup.bash"

# ------------------------------------------------------------
# Mission name
# ------------------------------------------------------------

run_name="dubins_$(date +%Y%m%d_%H%M%S)"

echo "Mission name: $run_name"
echo "Launching simulation and controller..."

# ------------------------------------------------------------
# Simulation + planning + state + control + logger
# ------------------------------------------------------------

exec ros2 launch planner \
    dubins_leader_tracking.launch.py \
    run_name:="$run_name" \
    simulation_profile:=light \
    show_live_map:=false
