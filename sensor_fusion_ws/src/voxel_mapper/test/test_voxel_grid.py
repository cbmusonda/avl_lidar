import numpy as np
import pytest

from voxel_mapper.voxel_grid import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    VoxelGrid,
    bresenham3d,
)


def make_grid(resolution=0.3, persistence_sec=2.0):
    return VoxelGrid(
        x_range=(-20.0, 20.0),
        y_range=(-20.0, 20.0),
        z_range=(-1.0, 3.0),
        resolution=resolution,
        persistence_sec=persistence_sec,
    )


# ---- geometry / index mapping ----------------------------------------------


def test_grid_dimensions_use_actual_not_nominal_extent():
    grid = make_grid()
    assert grid.nx == round(40 / 0.3)
    assert grid.ny == round(40 / 0.3)
    assert grid.nz == round(4 / 0.3)
    # actual extent differs from nominal 40/4 because 0.3 doesn't divide evenly
    assert abs(grid.nx * grid.resolution - 40.0) > 1e-9


def test_world_to_index_round_trip():
    grid = make_grid(resolution=0.5)
    idx = grid.world_to_index(1.2, -3.7, 0.6)
    assert idx is not None
    world = grid.index_to_world(*idx)
    # round trip should land back within half a voxel
    for a, b in zip(world, (1.2, -3.7, 0.6)):
        assert abs(a - b) <= grid.resolution


def test_world_to_index_out_of_bounds_returns_none():
    grid = make_grid()
    assert grid.world_to_index(1000, 0, 0) is None
    assert grid.world_to_index(0, -1000, 0) is None
    assert grid.world_to_index(0, 0, 100) is None


def test_in_bounds():
    grid = make_grid()
    assert grid.in_bounds(0, 0, 0)
    assert grid.in_bounds(grid.nx - 1, grid.ny - 1, grid.nz - 1)
    assert not grid.in_bounds(-1, 0, 0)
    assert not grid.in_bounds(grid.nx, 0, 0)


def test_flat_index_matches_c_order_flatten():
    grid = make_grid(resolution=1.0)
    grid.x_range = (-2, 2)
    # populate with distinct values and confirm flatten(order='C') agrees
    # with the documented flat index formula ix*ny*nz + iy*nz + iz
    arange = np.arange(grid.nx * grid.ny * grid.nz).reshape(
        (grid.nx, grid.ny, grid.nz)
    )
    flat = arange.flatten(order="C")
    for ix in range(grid.nx):
        for iy in range(grid.ny):
            for iz in range(grid.nz):
                expected = grid.flat_index(ix, iy, iz)
                assert arange[ix, iy, iz] == flat[expected]


# ---- bresenham3d -------------------------------------------------------


def test_bresenham3d_straight_line_x_axis():
    cells = bresenham3d((0, 0, 0), (5, 0, 0))
    assert cells == [(i, 0, 0) for i in range(6)]


def test_bresenham3d_includes_both_endpoints():
    cells = bresenham3d((0, 0, 0), (3, 2, 1))
    assert cells[0] == (0, 0, 0)
    assert cells[-1] == (3, 2, 1)


def test_bresenham3d_diagonal_symmetric():
    cells = bresenham3d((0, 0, 0), (2, 2, 2))
    assert cells == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]


def test_bresenham3d_reverse_direction():
    forward = bresenham3d((0, 0, 0), (4, -3, 2))
    backward = bresenham3d((4, -3, 2), (0, 0, 0))
    assert forward[0] == (0, 0, 0) and forward[-1] == (4, -3, 2)
    assert backward[0] == (4, -3, 2) and backward[-1] == (0, 0, 0)


# ---- raycasting ---------------------------------------------------------


def test_raycast_marks_endpoint_occupied_and_intermediate_free():
    grid = make_grid(resolution=1.0)
    origin = np.array([0.5, 0.5, 0.5])  # cell (20,20,1) roughly, inside grid
    # pick a point 5 voxels away along +x
    origin_idx = grid.world_to_index(*origin)
    end_world = grid.index_to_world(origin_idx[0] + 5, origin_idx[1], origin_idx[2])
    points = np.array([end_world])

    grid.raycast_update(origin, points, timestamp=1.0)

    end_idx = grid.world_to_index(*end_world)
    assert grid.cost[end_idx] == OCCUPIED

    # an intermediate cell should be free
    mid_idx = (origin_idx[0] + 2, origin_idx[1], origin_idx[2])
    assert grid.cost[mid_idx] == FREE


def test_raycast_unknown_elsewhere():
    grid = make_grid(resolution=1.0)
    origin = np.array([0.5, 0.5, 0.5])
    origin_idx = grid.world_to_index(*origin)
    end_world = grid.index_to_world(origin_idx[0] + 3, origin_idx[1], origin_idx[2])
    grid.raycast_update(origin, np.array([end_world]), timestamp=1.0)

    far_idx = (0, 0, 0)
    if far_idx not in (origin_idx, grid.world_to_index(*end_world)):
        assert grid.cost[far_idx] == UNKNOWN


def test_raycast_occupied_endpoint_survives_another_rays_intermediate():
    grid = make_grid(resolution=1.0)
    origin = np.array([0.5, 0.5, 0.5])
    origin_idx = grid.world_to_index(*origin)

    near_end = grid.index_to_world(origin_idx[0] + 2, origin_idx[1], origin_idx[2])
    far_end = grid.index_to_world(origin_idx[0] + 5, origin_idx[1], origin_idx[2])

    # near_end is an endpoint of ray 1 and an intermediate cell of ray 2
    # (batched together) -- endpoint must win.
    grid.raycast_update(origin, np.array([near_end, far_end]), timestamp=1.0)

    near_idx = grid.world_to_index(*near_end)
    assert grid.cost[near_idx] == OCCUPIED


def test_raycast_ignores_out_of_bounds_origin():
    grid = make_grid(resolution=1.0)
    grid.raycast_update(
        np.array([1000.0, 0.0, 0.0]), np.array([[0.0, 0.0, 0.0]]), timestamp=1.0
    )
    assert np.all(grid.cost == UNKNOWN)


# ---- decay ----------------------------------------------------------------


def test_decay_reverts_stale_occupied_to_unknown_not_free():
    grid = make_grid(resolution=1.0, persistence_sec=2.0)
    idx = (5, 5, 1)
    grid.cost[idx] = OCCUPIED
    grid.timestamp[idx] = 0.0

    grid.decay(now=1.0)  # not yet stale
    assert grid.cost[idx] == OCCUPIED

    grid.decay(now=3.0)  # now stale (> 2s persistence)
    assert grid.cost[idx] == UNKNOWN


def test_decay_does_not_affect_free_or_fresh_occupied():
    grid = make_grid(resolution=1.0, persistence_sec=2.0)
    free_idx = (1, 1, 1)
    fresh_idx = (2, 2, 1)
    grid.cost[free_idx] = FREE
    grid.cost[fresh_idx] = OCCUPIED
    grid.timestamp[fresh_idx] = 10.0

    grid.decay(now=10.5)

    assert grid.cost[free_idx] == FREE
    assert grid.cost[fresh_idx] == OCCUPIED


# ---- inflation --------------------------------------------------------------


def test_inflate_single_voxel_known_offsets():
    grid = make_grid(resolution=1.0, persistence_sec=2.0)
    center = (10, 10, 1)
    grid.cost[center] = OCCUPIED

    inflated = grid.inflate(inflation_radius=1.5)

    assert inflated[center] == OCCUPIED
    neighbor = (11, 10, 1)  # 1.0m away, within radius
    assert inflated[neighbor] > FREE
    assert inflated[neighbor] < OCCUPIED

    far = (10, 15, 1)  # 5m away, outside radius
    assert inflated[far] == UNKNOWN  # untouched, still original value


def test_inflate_does_not_mutate_original_cost():
    grid = make_grid(resolution=1.0)
    grid.cost[(5, 5, 1)] = OCCUPIED
    original = grid.cost.copy()
    grid.inflate(inflation_radius=2.0)
    assert np.array_equal(grid.cost, original)


def test_inflate_zero_radius_is_noop_copy():
    grid = make_grid(resolution=1.0)
    grid.cost[(5, 5, 1)] = OCCUPIED
    inflated = grid.inflate(inflation_radius=0.0)
    assert np.array_equal(inflated, grid.cost)


def test_inflate_no_occupied_voxels_returns_copy():
    grid = make_grid(resolution=1.0)
    inflated = grid.inflate(inflation_radius=1.0)
    assert np.all(inflated == UNKNOWN)


# ---- 2D projection ----------------------------------------------------------


def test_max_projection_2d_priority_occupied_over_free_over_unknown():
    grid = make_grid(resolution=1.0)
    grid.cost[(3, 3, 0)] = FREE
    grid.cost[(3, 3, 1)] = OCCUPIED
    grid.cost[(3, 3, 2)] = UNKNOWN
    projection = grid.max_projection_2d()
    assert projection[3, 3] == OCCUPIED

    grid2 = make_grid(resolution=1.0)
    grid2.cost[(4, 4, 0)] = FREE
    projection2 = grid2.max_projection_2d()
    assert projection2[4, 4] == FREE


# ---- serialization ------------------------------------------------------


def test_occupied_points_matches_occupied_cells():
    grid = make_grid(resolution=1.0)
    grid.cost[(5, 5, 1)] = OCCUPIED
    grid.cost[(6, 6, 1)] = OCCUPIED
    points = grid.occupied_points()
    assert points.shape == (2, 3)


def test_occupied_points_empty_grid():
    grid = make_grid()
    points = grid.occupied_points()
    assert points.shape == (0, 3)


def test_inflated_points_with_cost_shape():
    grid = make_grid(resolution=1.0)
    grid.cost[(5, 5, 1)] = OCCUPIED
    inflated = grid.inflate(inflation_radius=1.5)
    points = grid.inflated_points_with_cost(inflated)
    assert points.shape[1] == 4
    assert points.shape[0] >= 1


def test_to_flat_int8_matches_flatten_c_order():
    grid = make_grid(resolution=1.0)
    grid.cost[(1, 2, 3)] = OCCUPIED
    flat = grid.to_flat_int8()
    assert flat.dtype == np.int8
    assert flat[grid.flat_index(1, 2, 3)] == OCCUPIED


def test_metadata_dict_uses_actual_computed_extents():
    grid = make_grid(resolution=0.3)
    meta = grid.metadata_dict(inflation_radius=1.2, timestamp=5.0, sensor_mode="nominal")
    assert meta["size_x"] == pytest.approx(grid.nx * 0.3)
    assert meta["size_x"] != 40.0  # nominal value must not leak through
    assert meta["grid_nx"] == grid.nx
    assert meta["grid_ny"] == grid.ny
    assert meta["grid_nz"] == grid.nz
    assert meta["frame_id"] == "base_link"
