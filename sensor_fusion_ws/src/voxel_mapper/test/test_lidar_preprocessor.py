import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from voxel_mapper.lidar_preprocessor import (
    align_plane_sign,
    apply_height_band,
    apply_transform,
    classify_ground_inliers,
    downsample_voxel,
    fit_ground_plane_ransac,
    is_sensor_healthy,
    pointcloud2_to_xyz_array,
    range_filter,
    remove_ground,
    smooth_plane,
    transform_matrix_from_stamped,
)


def make_cloud(points):
    header = Header()
    header.frame_id = "velodyne"
    return point_cloud2.create_cloud_xyz32(header, points)


# ---- pointcloud2_to_xyz_array ------------------------------------------


def test_pointcloud2_to_xyz_array_drops_nans():
    points = [(1.0, 2.0, 3.0), (float("nan"), 1.0, 1.0), (4.0, 5.0, 6.0)]
    msg = make_cloud(points)
    arr = pointcloud2_to_xyz_array(msg, skip_nans=True)
    assert arr.shape == (2, 3)
    assert np.allclose(arr, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_pointcloud2_to_xyz_array_empty_cloud():
    msg = make_cloud([])
    arr = pointcloud2_to_xyz_array(msg)
    assert arr.shape == (0, 3)


def test_pointcloud2_to_xyz_array_keeps_nans_when_not_skipping():
    points = [(1.0, 2.0, 3.0), (float("nan"), 1.0, 1.0)]
    msg = make_cloud(points)
    arr = pointcloud2_to_xyz_array(msg, skip_nans=False)
    assert arr.shape == (2, 3)
    assert np.isnan(arr[1, 0])


# ---- range_filter -----------------------------------------------------


def test_range_filter_keeps_points_in_range():
    points = np.array([[1, 0, 0], [10, 0, 0], [60, 0, 0], [0.1, 0, 0]])
    filtered = range_filter(points, min_range=0.5, max_range=50.0)
    assert filtered.shape[0] == 2
    dists = np.linalg.norm(filtered, axis=1)
    assert np.all(dists >= 0.5) and np.all(dists <= 50.0)


def test_range_filter_empty_input():
    points = np.empty((0, 3))
    filtered = range_filter(points, 0.5, 50.0)
    assert filtered.shape == (0, 3)


# ---- downsample_voxel ----------------------------------------------------


def test_downsample_voxel_reduces_dense_cluster():
    rng = np.random.default_rng(0)
    cluster = rng.uniform(0, 0.1, size=(100, 3))
    downsampled = downsample_voxel(cluster, voxel_size=0.2)
    assert downsampled.shape[0] == 1


def test_downsample_voxel_keeps_separated_points():
    points = np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20]])
    downsampled = downsample_voxel(points, voxel_size=0.2)
    assert downsampled.shape[0] == 3


# ---- RANSAC ground removal -------------------------------------------------


def test_fit_ground_plane_ransac_flat_ground():
    rng = np.random.default_rng(42)
    ground = np.column_stack(
        [rng.uniform(-5, 5, 200), rng.uniform(-5, 5, 200), np.zeros(200)]
    )
    obstacle = np.array([[0, 0, 2.0], [1, 1, 1.5]])
    points = np.vstack([ground, obstacle])

    coeffs, inlier_mask = fit_ground_plane_ransac(
        points, iterations=50, distance_threshold=0.05, candidate_band=0.5, seed=1
    )
    assert coeffs is not None
    assert inlier_mask[:200].sum() >= 190  # most ground points are inliers
    assert not inlier_mask[200]  # obstacle points are not inliers
    assert not inlier_mask[201]


def test_fit_ground_plane_ransac_too_few_points():
    points = np.array([[0, 0, 0], [1, 1, 1]])
    coeffs, inlier_mask = fit_ground_plane_ransac(points)
    assert coeffs is None
    assert not inlier_mask.any()


def test_remove_ground_returns_non_inliers():
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    mask = np.array([True, False, True])
    obstacles = remove_ground(points, mask)
    assert obstacles.shape[0] == 1
    assert np.allclose(obstacles[0], [1, 1, 1])


# ---- ground plane smoothing --------------------------------------------


def test_align_plane_sign_flips_opposing_normal():
    reference = np.array([0.0, 0.0, 1.0, 0.0])
    flipped = np.array([0.0, 0.0, -1.0, 0.0])
    aligned = align_plane_sign(flipped, reference)
    assert np.allclose(aligned, [0.0, 0.0, 1.0, 0.0])


def test_align_plane_sign_leaves_matching_normal():
    reference = np.array([0.0, 0.0, 1.0, 0.0])
    same = np.array([0.0, 0.0, 1.0, -0.1])
    aligned = align_plane_sign(same, reference)
    assert np.allclose(aligned, same)


def test_smooth_plane_first_fit_returns_candidate_unchanged():
    candidate = np.array([0.0, 0.0, 1.0, -0.2])
    smoothed = smooth_plane(None, candidate, alpha=0.25)
    assert np.allclose(smoothed, candidate)


def test_smooth_plane_blends_toward_candidate():
    previous = np.array([0.0, 0.0, 1.0, 0.0])
    candidate = np.array([0.0, 0.0, 1.0, -1.0])
    smoothed = smooth_plane(previous, candidate, alpha=0.25)
    # Blended d should move partway from 0 toward -1, not jump all the way.
    assert -1.0 < smoothed[3] < 0.0
    assert abs(smoothed[3] - (-0.25)) < 0.05


def test_smooth_plane_aligns_sign_flipped_candidate_before_blending():
    previous = np.array([0.0, 0.0, 1.0, 0.0])
    flipped_candidate = np.array([0.0, 0.0, -1.0, 1.0])  # same plane, opposite sign
    smoothed = smooth_plane(previous, flipped_candidate, alpha=0.5)
    # Should blend as if candidate were [0,0,1,-1], not cancel toward zero.
    assert smoothed[2] > 0.9
    assert smoothed[3] < 0.0


def test_smooth_plane_stays_unit_normal():
    previous = np.array([0.0, 0.0, 1.0, 0.0])
    candidate = np.array([0.1, 0.05, 0.99, -0.3]) / np.linalg.norm([0.1, 0.05, 0.99])
    smoothed = smooth_plane(previous, candidate, alpha=0.3)
    assert np.isclose(np.linalg.norm(smoothed[:3]), 1.0)


def test_classify_ground_inliers_matches_threshold():
    plane = np.array([0.0, 0.0, 1.0, 0.0])  # z = 0 plane
    points = np.array([[0, 0, 0.02], [0, 0, 0.2], [5, 5, -0.01]])
    mask = classify_ground_inliers(points, plane, distance_threshold=0.05)
    assert mask.tolist() == [True, False, True]


def test_classify_ground_inliers_empty_points():
    plane = np.array([0.0, 0.0, 1.0, 0.0])
    points = np.empty((0, 3))
    mask = classify_ground_inliers(points, plane, distance_threshold=0.05)
    assert mask.shape == (0,)


# ---- height band ------------------------------------------------------


def test_apply_height_band_filters_correctly():
    points = np.array([[0, 0, -2.0], [0, 0, 0.5], [0, 0, 5.0]])
    filtered = apply_height_band(points, z_min=-1.0, z_max=3.0)
    assert filtered.shape[0] == 1
    assert filtered[0, 2] == 0.5


# ---- transform --------------------------------------------------------


def test_transform_matrix_from_stamped_identity():
    tf = TransformStamped()
    tf.transform.rotation.w = 1.0
    matrix = transform_matrix_from_stamped(tf)
    assert np.allclose(matrix, np.eye(4))


def test_transform_matrix_from_stamped_translation_only():
    tf = TransformStamped()
    tf.transform.translation.x = 0.089
    tf.transform.translation.z = 0.659
    tf.transform.rotation.w = 1.0
    matrix = transform_matrix_from_stamped(tf)
    points = np.array([[0.0, 0.0, 0.0]])
    transformed = apply_transform(points, matrix)
    assert np.allclose(transformed[0], [0.089, 0.0, 0.659])


def test_transform_matrix_from_stamped_90deg_yaw():
    # 90 degree rotation about Z: (x,y,z)=(0,0,sin(45)),w=cos(45)
    tf = TransformStamped()
    tf.transform.rotation.z = 0.70710678
    tf.transform.rotation.w = 0.70710678
    matrix = transform_matrix_from_stamped(tf)
    points = np.array([[1.0, 0.0, 0.0]])
    transformed = apply_transform(points, matrix)
    assert np.allclose(transformed[0], [0.0, 1.0, 0.0], atol=1e-6)


def test_apply_transform_empty_points():
    matrix = np.eye(4)
    points = np.empty((0, 3))
    transformed = apply_transform(points, matrix)
    assert transformed.shape == (0, 3)


# ---- sensor health ------------------------------------------------------


def test_is_sensor_healthy_within_timeout():
    assert is_sensor_healthy(last_msg_time=10.0, now=10.3, timeout_sec=0.5)


def test_is_sensor_healthy_exceeds_timeout():
    assert not is_sensor_healthy(last_msg_time=10.0, now=11.0, timeout_sec=0.5)


def test_is_sensor_healthy_never_received():
    assert not is_sensor_healthy(last_msg_time=None, now=10.0, timeout_sec=0.5)
