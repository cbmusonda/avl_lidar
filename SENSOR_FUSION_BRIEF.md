# ROS 2 Sensor Fusion and 3D Voxel Mapping Brief

## Goal

Build a ROS 2 perception package that initially uses:

- Velodyne VLP-16 LiDAR
- Xsens MTi-680G IMU
- Existing vehicle TF/URDF model

The first deliverable is a local 3D voxel obstacle representation. Depth cameras are a future extension and must not be required by the initial implementation.

The voxel output will eventually be consumed by another perception/path-planning project.

## Source Repositories

- [avl_slam](https://github.com/changwemusonda/avl_slam): real sensor drivers, LiDAR/Xsens conventions, RTAB-Map ICP integration.
- [IGVC_ROS2](https://github.com/Paarseus/IGVC_ROS2): vehicle URDF, sensor placement, launch organization, localization and Nav2 conventions.
- [Ros-workspace](https://github.com/Pegasus-Disaster-Response/Ros-workspace): custom 3D voxel costmap, LiDAR preprocessing, raycasting, decay, inflation, and planner output.

## Findings From `avl_slam`

Repository conventions:

- ROS 2 Humble
- Velodyne VLP-16
- Xsens MTi-680G
- Jetson Orin AGX / Ubuntu aarch64
- LiDAR topic: `/velodyne_points`
- LiDAR frame: `velodyne`
- IMU topic: `/imu/data`
- IMU frame: `imu_link`
- LiDAR IP: `192.168.13.11`
- LiDAR UDP port: `2368`
- VLP-16 speed: `600 rpm`, approximately 10 Hz
- Point-cloud range in the launch file: approximately 0.4 to 100 m

The Velodyne pipeline is:

```text
velodyne_driver_node
    -> VelodyneScan packets
velodyne_transform_node
    -> /velodyne_points
```

The Xsens driver publishes a fused `sensor_msgs/Imu` message containing orientation, angular velocity, and linear acceleration. The driver publishes this on `/imu/data`; `/filter/imu/data` should not be used with this driver.

The existing SLAM setup uses:

```text
/velodyne_points + /imu/data
    -> RTAB-Map ICP odometry
    -> /odom
```

Important lessons:

- Do not add an external Madgwick filter unless the selected Xsens driver requires it.
- Do not allow multiple nodes to publish the same TF edge.
- Static sensor transforms must come from one authoritative URDF or static-TF source.
- `wait_imu_to_init: true` is important for RTAB-Map ICP when using IMU gravity alignment.
- Validate timestamps and TF availability before tuning mapping algorithms.
- The repository contains several hardware-specific camera modes; initially reuse only the LiDAR, IMU, and TF portions.

Relevant paths include:

```text
src/avl_slam/config/vlp16.yaml
src/avl_slam/config/xsens.yaml
src/avl_slam/config/rtabmap.yaml
src/avl_slam/launch/slam.launch.py
src/avl_slam/SLAM.md
```

## Findings From `IGVC_ROS2`

The relevant package is:

```text
src/avros_bringup/
├── launch/
├── config/
└── urdf/avros.urdf.xacro
```

The preferred TF structure is:

```text
map
└── odom
    └── base_link
        ├── imu_link
        ├── velodyne
        └── camera frames
```

Documented approximate sensor locations are:

```text
imu_link:  (0.000,  0.000, 0.500)
velodyne:  (0.089,  0.000, 0.659)
zed_front: (0.397,  0.000, 0.557)
zed_left:  (0.098, +0.286, 0.557), yaw +pi/2
```

The URDF should be the source of truth for the vehicle model, but the repository notes that physical sensor mounts still require verification. A wrong LiDAR transform moves every obstacle voxel, so physical measurement is more important than early algorithm tuning.

Best practices from this repository:

- Use `robot_state_publisher`.
- Use vendor-provided camera URDF macros when cameras are added later.
- Disable camera-driver TF publication when the vehicle URDF owns those transforms.
- Do not hand-roll camera optical-frame chains.
- Keep sensor launch arguments independently switchable.
- Store ROS parameters in YAML files.
- Set DDS/RMW configuration in the top-level launch file, not only in a nested sensor launch.

Important warning: the repository's own review reports that its Humble Nav2 configuration contains a VoxelLayer block but does not activate it in the local costmap plugin list. Reuse its URDF and launch organization, but do not copy its costmap configuration blindly.

Relevant paths include:

```text
src/avros_bringup/urdf/avros.urdf.xacro
src/avros_bringup/launch/sensors.launch.py
src/avros_bringup/launch/localization.launch.py
src/avros_bringup/config/velodyne.yaml
src/avros_bringup/config/xsens.yaml
src/avros_bringup/config/nav2_params_humble.yaml
```

## Findings From `Ros-workspace`

Most relevant files:

```text
src/pegasus_ros/pegasus_autonomy/lidar_costmap_layer_node.py
src/pegasus_ros/pegasus_autonomy/local_costmap_node.py
src/pegasus_ros/config/local_costmap.yaml
src/pegasus_ros/launch/local_costmap.launch.py
```

The architecture is:

```text
/velodyne_points
    -> lidar_costmap_layer_node
    -> /pegasus/lidar_obstacles
    -> local_costmap_node
    -> 3D voxel grid
```

### LiDAR preprocessing

The LiDAR preprocessing node:

1. Subscribes to `/velodyne_points`.
2. Uses TF2 to transform points from `velodyne` into `base_link`.
3. Filters invalid and out-of-range points.
4. Downsamples points into approximately 0.2 m voxels.
5. Removes ground using RANSAC plane fitting.
6. Applies a height band.
7. Publishes obstacle points in `base_link`.
8. Publishes the LiDAR origin in `base_link`.
9. Publishes sensor health.

Default preprocessing values include:

```text
range minimum:           0.5 m
range maximum:           50 m
preprocessing voxel:     0.2 m
ground removal:          enabled
RANSAC iterations:       50
ground candidate band:   0.5 m
minimum obstacle points: 3
```

RANSAC is preferable to a fixed Z threshold when the vehicle pitches or rolls, but its parameters must be tuned against the real chassis and terrain.

### Voxel grid

The default Pegasus grid is:

```text
size:                 40 m x 40 m x 20 m
resolution:           0.3 m
frame:                base_link
obstacle persistence: 2 seconds
decay check rate:     2 Hz
raycasting:           enabled
inflation radius:     2.5 m
publish rate:         10 Hz
3D grid rate:         5 Hz
```

Each voxel stores:

```text
cost:
  -1   unknown
   0   observed free
 100   occupied

timestamp:
  last update time

source:
  0 none
  1 LiDAR
  2 depth camera
  3 both
```

For every LiDAR scan:

1. Raycast from the LiDAR origin to each returned obstacle point.
2. Mark intermediate voxels as free.
3. Mark endpoint voxels as occupied.
4. Record the update timestamp.
5. Decay stale occupied voxels.
6. Compute inflation using a distance transform.
7. Publish raw and inflated representations.

The raycasting uses a 3D Bresenham-style traversal. This prevents unknown space from being confused with free space.

### Published outputs

The Pegasus implementation publishes:

```text
/pegasus/local_costmap
  PointCloud2, raw occupied voxels

/pegasus/local_costmap_inflated
  PointCloud2, inflated voxels with cost/intensity

/pegasus/local_costmap_3d_grid
  Int8MultiArray, complete 3D grid

/pegasus/local_costmap_markers
  MarkerArray, RViz cube visualization

/pegasus/local_costmap_2d
  OccupancyGrid, 2D projection

/pegasus/costmap_metadata
  JSON String containing grid metadata

/pegasus/sensor_status
  String such as nominal or lidar_only
```

### QoS

The Pegasus sensor subscribers use BEST_EFFORT QoS with KEEP_LAST history. This is important because Velodyne and other sensor streams commonly publish with BEST_EFFORT QoS. A RELIABLE subscriber can silently fail to receive them.

## Recommended Initial Architecture

Create a dedicated workspace rather than modifying the SLAM package directly:

```text
sensor_fusion_ws/
└── src/
    ├── sensor_fusion_bringup/
    │   ├── launch/
    │   │   ├── sensors.launch.py
    │   │   ├── odometry.launch.py
    │   │   ├── voxel_mapping.launch.py
    │   │   └── integration.launch.py
    │   ├── config/
    │   │   ├── velodyne.yaml
    │   │   ├── xsens.yaml
    │   │   ├── voxel_mapper.yaml
    │   │   └── robot_localization.yaml
    │   └── urdf/
    │       └── vehicle.urdf.xacro
    │
    └── voxel_mapper/
        ├── voxel_mapper/
        │   ├── lidar_preprocessor.py
        │   ├── voxel_grid.py
        │   └── voxel_mapper_node.py
        ├── test/
        ├── setup.py
        └── package.xml
```

Use two nodes initially:

```text
lidar_preprocessor
    /velodyne_points
    -> /perception/lidar_obstacles
    -> /perception/lidar_origin

voxel_mapper
    /perception/lidar_obstacles
    /perception/lidar_origin
    -> voxel outputs
```

This preserves the Pegasus separation between sensor-specific preprocessing and generic voxel fusion. Future depth-camera nodes can publish the same obstacle-cloud, origin, and health interfaces.

## Proposed Topic Contract

### Inputs

```text
/velodyne_points
  sensor_msgs/PointCloud2
  frame: velodyne

/imu/data
  sensor_msgs/Imu
  frame: imu_link
```

### Intermediate topics

```text
/perception/lidar_obstacles
  sensor_msgs/PointCloud2
  frame: base_link

/perception/lidar_origin
  geometry_msgs/PointStamped
  frame: base_link

/perception/lidar_health
  std_msgs/Bool
```

### Outputs

```text
/perception/voxels
  sensor_msgs/PointCloud2
  occupied voxel centers

/perception/voxels_inflated
  sensor_msgs/PointCloud2
  voxel centers with cost/intensity

/perception/voxel_grid
  std_msgs/Int8MultiArray
  flat 3D grid for planners

/perception/voxel_grid_metadata
  custom message preferred; JSON String acceptable for prototype

/perception/obstacle_costmap_2d
  nav_msgs/OccupancyGrid

/perception/voxel_markers
  visualization_msgs/MarkerArray

/perception/sensor_status
  diagnostic/status message
```

The long-term version should replace JSON metadata with a custom message containing:

```text
frame_id
size_x
size_y
size_z
resolution
grid_nx
grid_ny
grid_nz
origin
inflation_radius
timestamp
sensor_mode
```

## Grid Representation

Use a centered grid in `base_link` initially:

```text
x: -size_x/2 to +size_x/2
y: -size_y/2 to +size_y/2
z: -size_z/2 to +size_z/2
```

Index mapping:

```text
grid[ix, iy, iz]
data[ix * ny * nz + iy * nz + iz]
```

Voxel center conversion:

```text
x = (ix + 0.5) * resolution - size_x / 2
y = (iy + 0.5) * resolution - size_y / 2
z = (iz + 0.5) * resolution - size_z / 2
```

For a ground vehicle, an asymmetric vertical range may be more efficient than the Pegasus UAV defaults, for example:

```text
x: -20 to +20 m
y: -20 to +20 m
z: -1 to +3 m
```

The Pegasus grid is symmetric around `z=0`, which may waste memory for a ground vehicle.

## IMU and Odometry Role

The IMU does not directly create obstacle voxels. It supports:

- LiDAR ICP gravity alignment
- Vehicle orientation
- Odometry stability
- Correct scan transformation while the vehicle moves

The initial system needs one authoritative source for:

```text
odom -> base_link
```

Possible choices:

1. RTAB-Map `icp_odometry` using LiDAR plus Xsens IMU.
2. `robot_localization` using Xsens and wheel odometry.
3. An existing odometry source from the consuming project.

The voxel mapper may operate entirely in `base_link`, but it still needs a valid `base_link -> velodyne` transform at each point-cloud timestamp.

## Launch Structure

Recommended separation:

```text
sensors.launch.py
  - robot_state_publisher
  - Velodyne driver
  - Velodyne point-cloud conversion
  - Xsens driver

odometry.launch.py
  - LiDAR ICP odometry
  - robot_localization, if selected

voxel_mapping.launch.py
  - lidar_preprocessor
  - voxel_mapper

integration.launch.py
  - includes sensors
  - includes odometry
  - includes voxel mapping
  - optional RViz
```

Provide launch arguments such as:

```text
use_sim_time
enable_lidar
enable_xsens
enable_odometry
enable_raycasting
publish_rviz
velodyne_ip
xsens_port
voxel_resolution
```

Avoid launching duplicate Velodyne or Xsens nodes. The AVL documentation reports duplicate Velodyne processes causing abnormal publication rates.

## Validation Plan

### Sensor validation

```bash
ros2 topic hz /velodyne_points
ros2 topic echo /velodyne_points --once
ros2 topic hz /imu/data
ros2 topic echo /imu/data --once
```

Verify:

```text
PointCloud2 frame_id: velodyne
IMU frame_id: imu_link
IMU orientation becomes non-zero after initialization
```

### TF validation

```bash
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo base_link velodyne
ros2 run tf2_ros tf2_echo base_link imu_link
```

Required chain:

```text
odom -> base_link -> velodyne
                  -> imu_link
```

### Voxel validation

Use a static wall or box and confirm:

- Occupied endpoint voxels appear.
- Intermediate ray voxels become free.
- Unknown space remains `-1`.
- Removing the obstacle causes stale voxels to decay.
- Vehicle motion does not produce large ghost trails.
- RViz displays the cloud in the correct physical location.
- Output timestamps follow the documented input/publication policy.

### Automated tests

At minimum, test:

- PointCloud2 parsing with NaN values.
- Range filtering.
- Voxel coordinate conversion.
- Out-of-bounds handling.
- 3D raycasting.
- Occupied endpoint preservation.
- Inflation radius behavior.
- Unknown/free/occupied encoding.
- Flat-array index mapping.
- Sensor timeout and health state.
- TF lookup failure behavior.

## Risks and Warnings

1. **Conflicting TF trees**: the three repositories use different sensor offsets and slightly different frame conventions. Select one URDF and measure the real hardware.
2. **Different Xsens drivers**: `avl_slam` references `ros2_xsens_mti_driver`; `IGVC_ROS2` references `xsens_mti_ros2_driver`. Confirm which driver is installed. Preserve `/imu/data` if possible.
3. **Approximate sensor mounts**: a wrong LiDAR transform moves every obstacle voxel.
4. **Ground removal**: RANSAC must be tuned for the real terrain and chassis.
5. **QoS mismatch**: use BEST_EFFORT for sensor subscriptions where required.
6. **Moving-frame persistence**: a `base_link`-centered grid requires explicit testing while the vehicle moves.
7. **Large grids**: a 40 x 40 x 20 m grid at 0.3 m resolution contains approximately 118,000 voxels. Finer resolution increases memory and inflation cost quickly.
8. **Nav2 configuration drift**: do not assume a declared VoxelLayer is active. Check the plugin list and live topic graph.
9. **TF ownership**: only one node should publish each transform, especially `odom -> base_link` and `base_link -> sensor`.

## Questions To Resolve Before Implementation

1. Is the target platform the same ground vehicle represented in `IGVC_ROS2`?
2. Which ROS 2 distribution will the new stack use?
3. Is the LiDAR definitely a Velodyne VLP-16, and is its current IP still `192.168.13.11`?
4. Which Xsens ROS 2 driver is installed and currently working?
5. Should the first version use RTAB-Map ICP odometry, `robot_localization`, wheel odometry, or an existing odometry topic?
6. Should the voxel grid be centered in `base_link`, `odom`, or a fixed `map` frame?
7. What obstacle height range matters for path planning?
8. What voxel resolution, local width/length, and vertical range are required?
9. Should unknown space be treated as blocked, free, or explicitly unknown by the downstream planner?
10. Does the receiving project require a Nav2 costmap, custom `Int8MultiArray`, `PointCloud2`, or all three?
11. Should future camera integration use the Pegasus-style obstacle-cloud, origin, and health interfaces?
12. Is the intended output only local obstacle perception, or should this package also provide SLAM and localization?

## Recommended Starting Point

Reuse the AVL LiDAR/Xsens topic conventions, use the IGVC URDF structure after verifying physical measurements, and adapt the Pegasus LiDAR preprocessing plus 3D voxel fusion into a standalone package.

Do not begin with depth cameras, path planning, or full sensor fusion. First prove:

```text
LiDAR + IMU + correct TF
    -> filtered obstacle points
    -> raycasted 3D voxel grid
    -> validated RViz and planner outputs
```
