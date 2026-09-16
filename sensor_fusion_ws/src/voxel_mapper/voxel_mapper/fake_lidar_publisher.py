"""
Synthetic /perception/lidar_obstacles publisher for testing voxel_mapper
without real LiDAR hardware or lidar_preprocessor in the loop.

Publishes a static room boundary plus one obstacle that orbits the origin,
so raycasting/decay/inflation are all visibly exercised in RViz.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Header


def _room_boundary_points(half_extent: float, spacing: float) -> np.ndarray:
    """Perimeter wall points at two heights, forming a hollow square room."""
    edge = np.arange(-half_extent, half_extent + spacing, spacing)
    heights = (0.0, 1.2)
    points = []
    for z in heights:
        for x in edge:
            points.append((x, -half_extent, z))
            points.append((x, half_extent, z))
        for y in edge:
            points.append((-half_extent, y, z))
            points.append((half_extent, y, z))
    return np.array(points, dtype=np.float32)


def _cube_points(center: np.ndarray, half_size: float, spacing: float) -> np.ndarray:
    edge = np.arange(-half_size, half_size + spacing, spacing)
    xs, ys, zs = np.meshgrid(edge, edge, edge, indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    return (pts + center).astype(np.float32)


class FakeLidarPublisher(Node):
    def __init__(self):
        super().__init__("fake_lidar_publisher")

        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("room_half_extent", 8.0)
        self.declare_parameter("orbit_radius", 4.0)

        self.frame_id = self.get_parameter("frame_id").value
        self.orbit_radius = self.get_parameter("orbit_radius").value
        rate_hz = self.get_parameter("rate_hz").value

        self._room = _room_boundary_points(
            self.get_parameter("room_half_extent").value, spacing=0.5
        )
        self._t0 = self.get_clock().now().nanoseconds / 1e9

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.obstacles_pub = self.create_publisher(
            PointCloud2, "/perception/lidar_obstacles", sensor_qos
        )
        self.origin_pub = self.create_publisher(
            PointStamped, "/perception/lidar_origin", sensor_qos
        )
        self.health_pub = self.create_publisher(Bool, "/perception/lidar_health", sensor_qos)

        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"Publishing synthetic obstacles on /perception/lidar_obstacles "
            f"(frame_id={self.frame_id})"
        )

    def _tick(self):
        now = self.get_clock().now()
        stamp = now.to_msg()
        elapsed = now.nanoseconds / 1e9 - self._t0

        orbit_center = np.array(
            [
                self.orbit_radius * math.cos(elapsed * 0.4),
                self.orbit_radius * math.sin(elapsed * 0.4),
                0.5,
            ]
        )
        obstacle = _cube_points(orbit_center, half_size=0.4, spacing=0.15)

        all_points = np.vstack([self._room, obstacle])

        header = Header(stamp=stamp, frame_id=self.frame_id)
        self.obstacles_pub.publish(
            point_cloud2.create_cloud_xyz32(header, all_points.tolist())
        )

        origin = PointStamped()
        origin.header = header
        origin.point.x = 0.0
        origin.point.y = 0.0
        origin.point.z = 0.0
        self.origin_pub.publish(origin)

        self.health_pub.publish(Bool(data=True))


def main(args=None):
    rclpy.init(args=args)
    node = FakeLidarPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
