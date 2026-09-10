# sensor_fusion_ws

ROS 2 Humble workspace implementing the first deliverable from
`../SENSOR_FUSION_BRIEF.md`: Velodyne VLP-16 + Xsens MTi-680G + TF/URDF ->
filtered obstacle points -> raycasted 3D voxel grid, in `base_link`.

Depth cameras, path planning, and full SLAM are explicitly out of scope for
this package; see the brief for the longer-term plan.

## Packages

- `voxel_mapper` — the two nodes (`lidar_preprocessor`, `voxel_mapper_node`)
  and the pure `voxel_grid.py`/`lidar_preprocessor.py` logic they wrap.
- `sensor_fusion_bringup` — launch files, config YAMLs, URDF, RViz config.
  No nodes of its own.

## Odometry

`odom -> base_link` and `map -> odom` come from a dual `robot_localization`
EKF + `navsat_transform_node`, matching IGVC_ROS2's
`src/avros_bringup/launch/localization.launch.py` / `config/ekf.yaml`
pattern exactly (not RTAB-Map ICP):

```text
EKF #1 (odom): /imu/data + /wheel_odom (vx)                  -> odom -> base_link
EKF #2 (map):  /imu/data + /wheel_odom (vx) + /odometry/gps  -> map -> odom
navsat_transform_node: /gnss (NavSatFix) -> /odometry/gps for EKF #2
```

`/wheel_odom` and `/gnss` must already be published by nodes outside this
workspace (an actuator/motor-controller node and a GPS driver) — nothing
here launches drivers for them. `config/navsat.yaml`'s `datum` is a
placeholder `[0.0, 0.0, 0.0]` and **must** be set to a real `[lat, lon, 0.0]`
near your test site before GPS fusion does anything useful (see the
comments in that file). `enable_gps:=false` (on `integration.launch.py` or
`odometry.launch.py` directly) skips EKF #2 and `navsat_transform` entirely
and runs local-only odometry, useful for indoor testing without GPS.

The voxel mapper itself never depends on any of this — it only needs the
static `base_link -> velodyne` transform from the URDF/`robot_state_publisher`,
so `voxel_mapping.launch.py` works standalone with `enable_odometry:=false`.

## Build prerequisites

Everything below except the Xsens driver is rosdep-resolvable:

```bash
sudo apt install ros-humble-velodyne ros-humble-robot-localization \
    ros-humble-xacro python3-scipy python3-numpy
```

**`xsens_mti_ros2_driver` is not an apt/rosdep package.** It must be cloned
as source into `src/` before building:

```bash
cd src
git clone <xsens_mti_ros2_driver repo URL>
```

Its own build dependencies (a vendored `xspublic` C library, possibly
`libudev-dev`) can't be enumerated here without the repo present — check
its own README/package.xml once cloned.

`config/xsens.yaml` and `sensors.launch.py`'s Xsens node block use
placeholder parameter/topic names (`xsens_mti_node`, `port`) that need
verifying against the driver actually installed — `sensors.launch.py`
remaps the driver's IMU output onto `/imu/data` so the rest of the stack
(the EKFs, `navsat_transform_node`) doesn't depend on the driver's default
topic name being correct, but the remap's *source* topic name still needs
confirming.

## Build and test

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
colcon test
colcon test-result --verbose
```

`voxel_mapper`'s pure-Python logic (`voxel_grid.py`, most of
`lidar_preprocessor.py`) can also be unit-tested directly with `pytest`,
without a colcon build, once the ROS 2 environment is sourced (for the
`sensor_msgs`/`geometry_msgs` Python bindings):

```bash
source /opt/ros/humble/setup.bash
cd src/voxel_mapper
PYTHONPATH="$(pwd):$PYTHONPATH" python3 -m pytest test/ -v
```

## Running

```bash
source install/setup.bash
ros2 launch sensor_fusion_bringup integration.launch.py \
    enable_lidar:=true enable_xsens:=true enable_odometry:=true \
    publish_rviz:=true
```

Individual stages (`sensors.launch.py`, `odometry.launch.py`,
`voxel_mapping.launch.py`) can also be launched standalone for isolated
testing — see the brief's "Validation Plan" section for the exact
`ros2 topic hz/echo` and `tf2_echo` commands to run against real hardware.

`voxel_mapping.launch.py` needs no external hardware/driver packages and
was verified end-to-end in this environment (synthetic `/velodyne_points`
+ a static TF in, `/perception/lidar_obstacles` and `/perception/voxels`
out, correctly separating a synthetic ground plane from a synthetic
obstacle cluster). `sensors.launch.py` and `odometry.launch.py` require
the driver packages above (plus `/wheel_odom` and `/gnss` publishers) and
real (or simulated) hardware to exercise, and were only verified to parse
correctly (`ros2 launch sensor_fusion_bringup <file> --show-args`) in this
environment — `robot_localization` isn't installed in this sandbox either.

## Known verification gaps (need the real robot)

- Xsens driver's actual topic/param names (see above).
- Physical accuracy of `urdf/vehicle.urdf.xacro`'s sensor offsets — the
  brief notes these are approximate and must be measured on the real
  chassis before trusting mapping output.
- `config/navsat.yaml`'s `datum` placeholder needs a real site coordinate.
- RANSAC ground-removal parameters tuned against real terrain/chassis
  pitch and roll (defaults are from the brief's Pegasus reference values).
- Whether your `/wheel_odom` publisher reports the same
  `nav_msgs/Odometry` layout (`twist.twist.linear.x` for vx) IGVC_ROS2's
  `actuator_node` does — `ekf.yaml`'s `odom0`/`odom1` assume it.
