"""
LiDAR point-cloud preprocessing: filter, downsample, transform, remove ground.

Pure functions are module-level and take/return numpy arrays or plain
values so they're testable without a live ROS node or TF listener.
LidarPreprocessorNode wires them to subscriptions/publishers/TF.
"""

from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Header
from tf2_ros import (
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    Buffer,
    TransformListener,
)


# ---- pure functions ---------------------------------------------------------


def pointcloud2_to_xyz_array(msg: PointCloud2, skip_nans: bool = True) -> np.ndarray:
    """Extract an (N, 3) xyz array from a PointCloud2, dropping NaN rows."""
    points = point_cloud2.read_points(
        msg, field_names=("x", "y", "z"), skip_nans=False
    )
    arr = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if skip_nans:
        arr = arr[~np.isnan(arr).any(axis=1)]
    return arr


def range_filter(
    points_xyz: np.ndarray, min_range: float, max_range: float
) -> np.ndarray:
    """
    Keep points whose distance from the sensor origin (0,0,0) is in range.

    Must be called on points still in the sensor's own frame (e.g. `velodyne`)
    -- filtering after transforming into base_link would measure range from
    the wrong origin, since the sensor is offset from base_link.
    """
    if points_xyz.shape[0] == 0:
        return points_xyz
    dist = np.linalg.norm(points_xyz, axis=1)
    mask = (dist >= min_range) & (dist <= max_range)
    return points_xyz[mask]


def downsample_voxel(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Reduce to one representative point per voxel_size cube."""
    if points_xyz.shape[0] == 0:
        return points_xyz
    keys = np.floor(points_xyz / voxel_size).astype(np.int64)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    return points_xyz[np.sort(first_indices)]


def fit_ground_plane_ransac(
    points_xyz: np.ndarray,
    iterations: int = 50,
    distance_threshold: float = 0.1,
    candidate_band: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Hand-rolled RANSAC plane fit for ground removal.

    candidate_band restricts which points are eligible to seed/vote for the
    plane (a coarse z-band near the lowest points), separate from the
    per-point inlier distance_threshold used to score a candidate plane.

    Returns (plane_coeffs, inlier_mask) where plane_coeffs is [a, b, c, d]
    for ax+by+cz+d=0, normalized so (a,b,c) is unit length. Returns
    (None, all-False mask) if fitting isn't possible (too few points).
    """
    n = points_xyz.shape[0]
    inlier_mask = np.zeros(n, dtype=bool)
    if n < 3:
        return None, inlier_mask

    rng = np.random.default_rng(seed)

    z_min = np.min(points_xyz[:, 2])
    candidate_mask = points_xyz[:, 2] <= (z_min + candidate_band)
    candidate_indices = np.where(candidate_mask)[0]
    if candidate_indices.shape[0] < 3:
        candidate_indices = np.arange(n)

    best_coeffs = None
    best_count = -1

    for _ in range(iterations):
        sample_idx = rng.choice(candidate_indices, size=3, replace=False)
        p1, p2, p3 = points_xyz[sample_idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -np.dot(normal, p1)
        coeffs = np.array([normal[0], normal[1], normal[2], d])

        dist = np.abs(points_xyz @ coeffs[:3] + coeffs[3])
        count = int(np.sum(dist <= distance_threshold))

        if count > best_count:
            best_count = count
            best_coeffs = coeffs

    if best_coeffs is None:
        return None, inlier_mask

    dist = np.abs(points_xyz @ best_coeffs[:3] + best_coeffs[3])
    inlier_mask = dist <= distance_threshold
    return best_coeffs, inlier_mask


def remove_ground(points_xyz: np.ndarray, inlier_mask: np.ndarray) -> np.ndarray:
    """Return non-ground (obstacle) points, i.e. those NOT in inlier_mask."""
    if points_xyz.shape[0] == 0:
        return points_xyz
    return points_xyz[~inlier_mask]


def align_plane_sign(coeffs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Flip coeffs' sign if needed so its normal points the same way as reference's.

    RANSAC's cross-product normal direction depends on sample order and is
    arbitrary per fit -- two fits of the same physical plane can come back
    as [a,b,c,d] or [-a,-b,-c,-d]. Averaging those directly would cancel
    instead of reinforcing, so callers doing temporal blending must align
    signs first.
    """
    if np.dot(coeffs[:3], reference[:3]) < 0:
        return -coeffs
    return coeffs


def smooth_plane(
    previous: Optional[np.ndarray], candidate: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Exponentially blend a new RANSAC plane fit onto a running estimate.

    A ground vehicle's ground plane changes gradually (ramps, curbs, terrain
    undulation) even though an unseeded per-frame RANSAC fit on essentially
    the same points can jitter between fits from run to run. Blending
    trades a bit of lag for a stable plane, which is what should be used to
    classify ground vs. obstacle points, rather than each frame's raw fit.

    alpha is the weight given to the new candidate fit (0 < alpha <= 1);
    previous=None (first fit) returns candidate unchanged.
    """
    if previous is None:
        return candidate
    aligned = align_plane_sign(candidate, previous)
    blended = alpha * aligned + (1.0 - alpha) * previous
    norm = np.linalg.norm(blended[:3])
    if norm < 1e-9:
        return previous
    return blended / norm


def classify_ground_inliers(
    points_xyz: np.ndarray, plane_coeffs: np.ndarray, distance_threshold: float
) -> np.ndarray:
    """Boolean mask of points within distance_threshold of plane_coeffs."""
    if points_xyz.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    dist = np.abs(points_xyz @ plane_coeffs[:3] + plane_coeffs[3])
    return dist <= distance_threshold


def apply_height_band(
    points_xyz: np.ndarray, z_min: float, z_max: float
) -> np.ndarray:
    if points_xyz.shape[0] == 0:
        return points_xyz
    mask = (points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] <= z_max)
    return points_xyz[mask]


def remove_self_footprint(
    points_xyz: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
) -> np.ndarray:
    """
    Drop points (already in base_link) inside the vehicle's own bounding box.

    Without this, the vehicle's own chassis -- the closest, most persistent,
    most geometrically detailed thing any beam can see -- is eligible to be
    detected as an obstacle sitting next to itself, and is a likely
    contributor to voxel flicker right around the sensor (see
    ground_plane_ransac_seed / the grid log-odds: this is a different,
    upstream cause those don't address). The box defaults come from
    IGVC_ROS2's measured chassis dimensions, not this repo's own placeholder
    URDF -- see node docstring for the derivation.
    """
    if points_xyz.shape[0] == 0:
        return points_xyz
    inside = (
        (points_xyz[:, 0] >= x_range[0]) & (points_xyz[:, 0] <= x_range[1])
        & (points_xyz[:, 1] >= y_range[0]) & (points_xyz[:, 1] <= y_range[1])
        & (points_xyz[:, 2] >= z_range[0]) & (points_xyz[:, 2] <= z_range[1])
    )
    return points_xyz[~inside]


def transform_matrix_from_stamped(tf: TransformStamped) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform matrix from a TransformStamped.

    Hand-rolled quaternion->rotation-matrix instead of adding
    tf_transformations/transforms3d as a dependency.
    """
    t = tf.transform.translation
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w

    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm > 0:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def apply_transform(points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if points_xyz.shape[0] == 0:
        return points_xyz
    homogeneous = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
    transformed = homogeneous @ matrix.T
    return transformed[:, :3]


def is_sensor_healthy(
    last_msg_time: Optional[float], now: float, timeout_sec: float
) -> bool:
    if last_msg_time is None:
        return False
    return (now - last_msg_time) <= timeout_sec


# ---- ROS node -------------------------------------------------------------


class LidarPreprocessorNode(Node):
    def __init__(self):
        super().__init__("lidar_preprocessor")

        self.declare_parameter("input_topic", "/velodyne_points")
        self.declare_parameter("sensor_frame", "velodyne")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("range_min", 0.5)
        self.declare_parameter("range_max", 50.0)
        self.declare_parameter("downsample_voxel_size", 0.2)
        self.declare_parameter("ransac_iterations", 50)
        self.declare_parameter("ransac_candidate_band", 0.5)
        self.declare_parameter("ransac_distance_threshold", 0.1)
        self.declare_parameter("ground_plane_ransac_seed", 42)
        self.declare_parameter("ground_plane_smoothing_alpha", 0.25)
        self.declare_parameter("height_band_min", -0.8)
        self.declare_parameter("height_band_max", 3.0)
        self.declare_parameter("sensor_timeout_sec", 0.5)

        # Self-footprint exclusion box, in base_link. Defaults are the union
        # bounding box of IGVC_ROS2's measured chassis geometry (main body +
        # top sensor platform + both tread enclosures, from avros.urdf.xacro,
        # resurveyed 2026-05-06) plus a 0.05m margin, NOT this repo's own
        # placeholder vehicle.urdf.xacro (which has no collision geometry).
        # Re-derive these if the real chassis is measured directly, or if
        # this vehicle's dimensions differ from IGVC_ROS2's.
        self.declare_parameter("footprint_x_min", -0.23)
        self.declare_parameter("footprint_x_max", 0.78)
        self.declare_parameter("footprint_y_min", -0.46)
        self.declare_parameter("footprint_y_max", 0.46)
        self.declare_parameter("footprint_z_min", -0.05)
        self.declare_parameter("footprint_z_max", 0.72)

        self.sensor_frame = self.get_parameter("sensor_frame").value
        self.target_frame = self.get_parameter("target_frame").value
        self.sensor_timeout_sec = self.get_parameter("sensor_timeout_sec").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            self.get_parameter("input_topic").value,
            self._cloud_callback,
            sensor_qos,
        )

        self.obstacles_pub = self.create_publisher(
            PointCloud2, "/perception/lidar_obstacles", 10
        )
        self.origin_pub = self.create_publisher(
            PointStamped, "/perception/lidar_origin", 10
        )
        self.health_pub = self.create_publisher(Bool, "/perception/lidar_health", 10)

        self._last_msg_time: Optional[float] = None
        self._smoothed_ground_plane: Optional[np.ndarray] = None
        self._timeout_timer = self.create_timer(0.2, self._check_timeout)

    def _check_timeout(self):
        now = self.get_clock().now().nanoseconds / 1e9
        healthy = is_sensor_healthy(self._last_msg_time, now, self.sensor_timeout_sec)
        if not healthy:
            self.health_pub.publish(Bool(data=False))

    def _cloud_callback(self, msg: PointCloud2):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self._last_msg_time = now_sec

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, self.sensor_frame, msg.header.stamp
            )
        except (LookupException, ExtrapolationException, ConnectivityException) as exc:
            self.get_logger().warn(
                f"TF lookup {self.target_frame}<-{self.sensor_frame} failed: {exc}",
                throttle_duration_sec=2.0,
            )
            self.health_pub.publish(Bool(data=False))
            return

        points = pointcloud2_to_xyz_array(msg)
        points = range_filter(
            points,
            self.get_parameter("range_min").value,
            self.get_parameter("range_max").value,
        )
        points = downsample_voxel(
            points, self.get_parameter("downsample_voxel_size").value
        )

        matrix = transform_matrix_from_stamped(tf)
        points_base = apply_transform(points, matrix)
        points_base = remove_self_footprint(
            points_base,
            x_range=(
                self.get_parameter("footprint_x_min").value,
                self.get_parameter("footprint_x_max").value,
            ),
            y_range=(
                self.get_parameter("footprint_y_min").value,
                self.get_parameter("footprint_y_max").value,
            ),
            z_range=(
                self.get_parameter("footprint_z_min").value,
                self.get_parameter("footprint_z_max").value,
            ),
        )

        distance_threshold = self.get_parameter("ransac_distance_threshold").value
        candidate_coeffs, _ = fit_ground_plane_ransac(
            points_base,
            iterations=self.get_parameter("ransac_iterations").value,
            distance_threshold=distance_threshold,
            candidate_band=self.get_parameter("ransac_candidate_band").value,
            seed=self.get_parameter("ground_plane_ransac_seed").value,
        )
        if candidate_coeffs is not None:
            self._smoothed_ground_plane = smooth_plane(
                self._smoothed_ground_plane,
                candidate_coeffs,
                self.get_parameter("ground_plane_smoothing_alpha").value,
            )

        if self._smoothed_ground_plane is not None:
            inlier_mask = classify_ground_inliers(
                points_base, self._smoothed_ground_plane, distance_threshold
            )
        else:
            inlier_mask = np.zeros(points_base.shape[0], dtype=bool)

        obstacles = remove_ground(points_base, inlier_mask)
        obstacles = apply_height_band(
            obstacles,
            self.get_parameter("height_band_min").value,
            self.get_parameter("height_band_max").value,
        )

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.target_frame

        cloud_msg = point_cloud2.create_cloud_xyz32(header, obstacles.tolist())
        self.obstacles_pub.publish(cloud_msg)

        origin_point = PointStamped()
        origin_point.header = header
        origin_point.point.x = matrix[0, 3]
        origin_point.point.y = matrix[1, 3]
        origin_point.point.z = matrix[2, 3]
        self.origin_pub.publish(origin_point)

        self.health_pub.publish(Bool(data=True))


def main(args=None):
    rclpy.init(args=args)
    node = LidarPreprocessorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
