import json

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PointStamped
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from voxel_mapper.voxel_mapper_node import VoxelMapperNode, compute_sensor_status


# ---- compute_sensor_status (pure) --------------------------------------


def test_compute_sensor_status_nominal_when_healthy():
    assert compute_sensor_status(True) == "nominal"


def test_compute_sensor_status_unavailable_when_unhealthy():
    assert compute_sensor_status(False) == "lidar_unavailable"


# ---- node-level behavior, using a real rclpy node (no hardware needed) ----


@pytest.fixture(scope="module", autouse=True)
def rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = VoxelMapperNode()
    yield n
    n.destroy_node()


def make_obstacle_cloud(points, stamp=None):
    header = Header()
    header.frame_id = "base_link"
    if stamp is not None:
        header.stamp = stamp
    return point_cloud2.create_cloud_xyz32(header, points)


def test_obstacles_callback_without_origin_does_not_crash(node):
    # No /perception/lidar_origin received yet -- must be a safe no-op,
    # not a crash (mirrors the "TF lookup failure must not crash" requirement
    # one hop upstream: this node must tolerate missing upstream state too).
    msg = make_obstacle_cloud([(1.0, 1.0, 0.5)])
    node._obstacles_callback(msg)  # should not raise
    assert node.grid.occupied_points().shape[0] == 0


def test_obstacles_callback_with_origin_updates_grid(node):
    node._lidar_origin = np.array([0.0, 0.0, 0.5])
    msg = make_obstacle_cloud([(2.0, 0.0, 0.5)])
    node._obstacles_callback(msg)
    assert node.grid.occupied_points().shape[0] >= 1


def test_obstacles_callback_empty_cloud_is_safe(node):
    node._lidar_origin = np.array([0.0, 0.0, 0.5])
    msg = make_obstacle_cloud([])
    node._obstacles_callback(msg)  # should not raise
    assert node.grid.occupied_points().shape[0] == 0


def test_health_callback_updates_status_flag(node):
    from std_msgs.msg import Bool

    node._health_callback(Bool(data=True))
    assert node._lidar_healthy is True
    node._health_callback(Bool(data=False))
    assert node._lidar_healthy is False


def test_origin_callback_stores_point(node):
    msg = PointStamped()
    msg.point.x, msg.point.y, msg.point.z = 0.089, 0.0, 0.659
    node._origin_callback(msg)
    assert np.allclose(node._lidar_origin, [0.089, 0.0, 0.659])


def test_metadata_dict_json_shape_via_grid(node):
    metadata = node.grid.metadata_dict(
        inflation_radius=node.inflation_radius, timestamp=1.0, sensor_mode="nominal"
    )
    serialized = json.dumps(metadata)
    parsed = json.loads(serialized)
    for key in (
        "frame_id",
        "size_x",
        "size_y",
        "size_z",
        "resolution",
        "grid_nx",
        "grid_ny",
        "grid_nz",
        "origin",
        "inflation_radius",
        "timestamp",
        "sensor_mode",
    ):
        assert key in parsed


def test_build_occupancy_grid_dimensions(node):
    header = Header()
    header.frame_id = "base_link"
    occ = node._build_occupancy_grid(header)
    assert occ.info.width == node.grid.nx
    assert occ.info.height == node.grid.ny
    assert len(occ.data) == node.grid.nx * node.grid.ny


def test_build_markers_matches_occupied_count(node):
    header = Header()
    header.frame_id = "base_link"
    occupied = np.array([[1.0, 1.0, 0.5], [2.0, 2.0, 0.5]])
    markers = node._build_markers(header, occupied)
    assert len(markers.markers) == 1
    assert len(markers.markers[0].points) == 2
