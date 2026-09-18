"""
Fuses LiDAR obstacle points into a base_link-centered voxel grid.

compute_sensor_status is module-level and pure so it's testable without a
running node.
"""

import json
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Header, Int8MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

from voxel_mapper.voxel_grid import VoxelGrid


def compute_sensor_status(lidar_healthy: bool) -> str:
    """
    Map current sensor health flags to a status string.

    v1 has only one sensor source (LiDAR), so "lidar_only" (the brief's
    example value for a degraded multi-sensor system) doesn't apply here;
    unhealthy LiDAR means no usable obstacle data at all.
    """
    return "nominal" if lidar_healthy else "lidar_unavailable"


class VoxelMapperNode(Node):
    def __init__(self):
        super().__init__("voxel_mapper")

        self.declare_parameter("resolution", 0.3)
        self.declare_parameter("x_min", -20.0)
        self.declare_parameter("x_max", 20.0)
        self.declare_parameter("y_min", -20.0)
        self.declare_parameter("y_max", 20.0)
        self.declare_parameter("z_min", -1.0)
        self.declare_parameter("z_max", 3.0)
        self.declare_parameter("persistence_sec", 2.0)
        self.declare_parameter("decay_rate_hz", 2.0)
        self.declare_parameter("inflation_radius", 1.2)
        self.declare_parameter("enable_raycasting", True)
        self.declare_parameter("hit_logodds", 0.85)
        self.declare_parameter("miss_logodds", -0.4)
        self.declare_parameter("logodds_min", -0.8)
        self.declare_parameter("logodds_max", 2.0)

        self.inflation_radius = self.get_parameter("inflation_radius").value
        self.enable_raycasting = self.get_parameter("enable_raycasting").value

        self.grid = VoxelGrid(
            x_range=(
                self.get_parameter("x_min").value,
                self.get_parameter("x_max").value,
            ),
            y_range=(
                self.get_parameter("y_min").value,
                self.get_parameter("y_max").value,
            ),
            z_range=(
                self.get_parameter("z_min").value,
                self.get_parameter("z_max").value,
            ),
            resolution=self.get_parameter("resolution").value,
            persistence_sec=self.get_parameter("persistence_sec").value,
            hit_logodds=self.get_parameter("hit_logodds").value,
            miss_logodds=self.get_parameter("miss_logodds").value,
            logodds_min=self.get_parameter("logodds_min").value,
            logodds_max=self.get_parameter("logodds_max").value,
        )

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._lidar_origin: Optional[np.ndarray] = None
        self._lidar_healthy = False

        self.create_subscription(
            PointCloud2, "/perception/lidar_obstacles", self._obstacles_callback, sensor_qos
        )
        self.create_subscription(
            PointStamped, "/perception/lidar_origin", self._origin_callback, sensor_qos
        )
        self.create_subscription(
            Bool, "/perception/lidar_health", self._health_callback, sensor_qos
        )

        self.voxels_pub = self.create_publisher(PointCloud2, "/perception/voxels", 10)
        self.voxels_inflated_pub = self.create_publisher(
            PointCloud2, "/perception/voxels_inflated", 10
        )
        self.voxel_grid_pub = self.create_publisher(
            Int8MultiArray, "/perception/voxel_grid", 10
        )
        self.voxel_grid_metadata_pub = self.create_publisher(
            String, "/perception/voxel_grid_metadata", 10
        )
        self.costmap_2d_pub = self.create_publisher(
            OccupancyGrid, "/perception/obstacle_costmap_2d", 10
        )
        self.markers_pub = self.create_publisher(
            MarkerArray, "/perception/voxel_markers", 10
        )
        self.sensor_status_pub = self.create_publisher(
            String, "/perception/sensor_status", 10
        )

        decay_period = 1.0 / self.get_parameter("decay_rate_hz").value
        self.create_timer(decay_period, self._decay_timer_callback)

    def _health_callback(self, msg: Bool):
        self._lidar_healthy = msg.data
        self.sensor_status_pub.publish(String(data=compute_sensor_status(self._lidar_healthy)))

    def _origin_callback(self, msg: PointStamped):
        self._lidar_origin = np.array([msg.point.x, msg.point.y, msg.point.z])

    def _obstacles_callback(self, msg: PointCloud2):
        if self._lidar_origin is None:
            return

        points = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        points_xyz = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
        if points_xyz.shape[0] == 0:
            return

        now = self.get_clock().now().nanoseconds / 1e9

        if self.enable_raycasting:
            self.grid.raycast_update(self._lidar_origin, points_xyz, timestamp=now)

        self._publish_all(msg.header.stamp, now)

    def _decay_timer_callback(self):
        now = self.get_clock().now().nanoseconds / 1e9
        self.grid.decay(now)

    def _publish_all(self, stamp, now: float):
        header = Header()
        header.stamp = stamp
        header.frame_id = "base_link"

        occupied = self.grid.occupied_points()
        self.voxels_pub.publish(
            point_cloud2.create_cloud_xyz32(header, occupied.tolist())
        )

        inflated = self.grid.inflate(self.inflation_radius)
        inflated_points = self.grid.inflated_points_with_cost(inflated)
        fields = [
            point_cloud2.PointField(
                name="x", offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1
            ),
            point_cloud2.PointField(
                name="y", offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1
            ),
            point_cloud2.PointField(
                name="z", offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1
            ),
            point_cloud2.PointField(
                name="intensity",
                offset=12,
                datatype=point_cloud2.PointField.FLOAT32,
                count=1,
            ),
        ]
        self.voxels_inflated_pub.publish(
            point_cloud2.create_cloud(header, fields, inflated_points.tolist())
        )

        flat = self.grid.to_flat_int8()
        multi_array = Int8MultiArray()
        multi_array.data = flat.tolist()
        self.voxel_grid_pub.publish(multi_array)

        metadata = self.grid.metadata_dict(
            inflation_radius=self.inflation_radius,
            timestamp=now,
            sensor_mode=compute_sensor_status(self._lidar_healthy),
        )
        self.voxel_grid_metadata_pub.publish(String(data=json.dumps(metadata)))

        self.costmap_2d_pub.publish(self._build_occupancy_grid(header))
        self.markers_pub.publish(self._build_markers(header, occupied))

    def _build_occupancy_grid(self, header: Header) -> OccupancyGrid:
        projection = self.grid.max_projection_2d()  # (nx, ny)
        msg = OccupancyGrid()
        msg.header = header
        msg.info.resolution = self.grid.resolution
        msg.info.width = self.grid.nx
        msg.info.height = self.grid.ny
        msg.info.origin.position.x = self.grid.origin[0]
        msg.info.origin.position.y = self.grid.origin[1]
        # OccupancyGrid is row-major over (height=ny, width=nx); our
        # projection is indexed (ix, iy), so transpose before flattening.
        data = projection.T.astype(np.int8)
        # unknown=-1 maps directly; occupied 100 maps directly; free 0 maps directly
        msg.data = data.flatten(order="C").tolist()
        return msg

    def _build_markers(self, header: Header, occupied_points: np.ndarray) -> MarkerArray:
        marker = Marker()
        marker.header = header
        marker.ns = "voxels"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self.grid.resolution
        marker.scale.y = self.grid.resolution
        marker.scale.z = self.grid.resolution
        marker.color.r = 1.0
        marker.color.a = 0.8
        marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in occupied_points]
        return MarkerArray(markers=[marker])


def main(args=None):
    rclpy.init(args=args)
    node = VoxelMapperNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
