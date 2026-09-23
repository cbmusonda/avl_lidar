"""
Dense 3D voxel occupancy grid with raycasting, decay, and inflation.

Pure numpy/scipy — no ROS message types imported here so this module is
testable with plain pytest, independent of a colcon build or ROS runtime.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

UNKNOWN = -1
FREE = 0
OCCUPIED = 100

# Occupied means log-odds above this; the epsilon keeps float32 rounding from
# leaving a cell that should sit exactly at 0 counted as occupied.
OCCUPIED_LOGODDS_EPS = 1e-4

SOURCE_NONE = 0
SOURCE_LIDAR = 1
SOURCE_DEPTH = 2
SOURCE_BOTH = 3


def bresenham3d(
    start: Tuple[int, int, int], end: Tuple[int, int, int]
) -> List[Tuple[int, int, int]]:
    """
    3D Bresenham line traversal, including both endpoints.

    Returns the list of integer grid cells visited from start to end.
    """
    x1, y1, z1 = start
    x2, y2, z2 = end

    points = [(x1, y1, z1)]

    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    x_inc = 1 if dx >= 0 else -1
    y_inc = 1 if dy >= 0 else -1
    z_inc = 1 if dz >= 0 else -1
    dx, dy, dz = abs(dx), abs(dy), abs(dz)

    x, y, z = x1, y1, z1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        for _ in range(dx):
            x += x_inc
            if p1 >= 0:
                y += y_inc
                p1 -= 2 * dx
            if p2 >= 0:
                z += z_inc
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            points.append((x, y, z))
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        for _ in range(dy):
            y += y_inc
            if p1 >= 0:
                x += x_inc
                p1 -= 2 * dy
            if p2 >= 0:
                z += z_inc
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            points.append((x, y, z))
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        for _ in range(dz):
            z += z_inc
            if p1 >= 0:
                y += y_inc
                p1 -= 2 * dz
            if p2 >= 0:
                x += x_inc
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            points.append((x, y, z))

    return points


class VoxelGrid:
    """
    Dense base_link-centered voxel occupancy grid.

    Grid extents need not divide evenly by resolution; the *actual* size
    (n * resolution) is what should be treated as authoritative downstream,
    not the nominal x_range/y_range/z_range passed in.
    """

    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        resolution: float,
        persistence_sec: float = 2.0,
        hit_logodds: float = 0.85,
        miss_logodds: float = -0.4,
        logodds_min: float = -0.8,
        logodds_max: float = 2.0,
    ):
        self.resolution = resolution
        self.persistence_sec = persistence_sec
        # Per-cell evidence in log-odds (OctoMap). hit +0.85 / miss -0.4 are
        # octomap_server's p=0.7 / p=0.4 defaults. The floor is -0.8 (OctoMap
        # uses -1.99) so one hit still makes a cell occupied: a new obstacle
        # shows up in one frame. The ceiling of 2.0 means a cell hit many
        # times needs 5 misses to clear.
        self.hit_logodds = hit_logodds
        self.miss_logodds = miss_logodds
        self.logodds_min = logodds_min
        self.logodds_max = logodds_max

        self.origin = (x_range[0], y_range[0], z_range[0])
        self.nx = max(1, round((x_range[1] - x_range[0]) / resolution))
        self.ny = max(1, round((y_range[1] - y_range[0]) / resolution))
        self.nz = max(1, round((z_range[1] - z_range[0]) / resolution))

        shape = (self.nx, self.ny, self.nz)
        self.cost = np.full(shape, UNKNOWN, dtype=np.int8)
        self.timestamp = np.zeros(shape, dtype=np.float64)
        self.source = np.full(shape, SOURCE_NONE, dtype=np.uint8)
        self.logodds = np.zeros(shape, dtype=np.float32)

    # ---- geometry -----------------------------------------------------

    def world_to_index(
        self, x: float, y: float, z: float
    ) -> Optional[Tuple[int, int, int]]:
        ix = int(np.floor((x - self.origin[0]) / self.resolution))
        iy = int(np.floor((y - self.origin[1]) / self.resolution))
        iz = int(np.floor((z - self.origin[2]) / self.resolution))
        if not self.in_bounds(ix, iy, iz):
            return None
        return ix, iy, iz

    def index_to_world(self, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
        x = self.origin[0] + (ix + 0.5) * self.resolution
        y = self.origin[1] + (iy + 0.5) * self.resolution
        z = self.origin[2] + (iz + 0.5) * self.resolution
        return x, y, z

    def in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz

    def flat_index(self, ix: int, iy: int, iz: int) -> int:
        return ix * self.ny * self.nz + iy * self.nz + iz

    # ---- mutation -------------------------------------------------------

    def _clip_segment_to_grid(
        self, p0: np.ndarray, p1: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Clip world-space segment p0->p1 to this grid's AABB (slab method).

        Returns the clipped (start, end) points, nudged slightly inward so
        they land in a valid cell rather than exactly on a boundary face,
        or None if the segment never enters the grid. Needed because the
        sensor origin is not guaranteed to sit inside a grid clamped to a
        thin z-band -- the sensor can be (and for a ground-hugging band,
        typically is) mounted above the slice it's scanning into. Clipping
        the ray to the grid, rather than requiring the raw origin to
        already be inside it, still lets the portion of each ray that
        passes through the grid mark free/occupied cells.
        """
        bounds_min = np.array(self.origin, dtype=np.float64)
        bounds_max = bounds_min + np.array([self.nx, self.ny, self.nz]) * self.resolution
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        direction = p1 - p0

        t_min, t_max = 0.0, 1.0
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if p0[axis] < bounds_min[axis] or p0[axis] > bounds_max[axis]:
                    return None
                continue
            t1 = (bounds_min[axis] - p0[axis]) / direction[axis]
            t2 = (bounds_max[axis] - p0[axis]) / direction[axis]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None

        clipped_start = p0 + t_min * direction
        clipped_end = p0 + t_max * direction
        seg_len = np.linalg.norm(clipped_end - clipped_start)
        if seg_len > 1e-9:
            # Nudge both ends inward by a fixed, resolution-negligible
            # distance so floor() lands in a valid cell instead of
            # exactly on the boundary face it's clipped to.
            unit_dir = (clipped_end - clipped_start) / seg_len
            nudge = min(1e-4, seg_len / 2)
            clipped_start = clipped_start + unit_dir * nudge
            clipped_end = clipped_end - unit_dir * nudge
        return clipped_start, clipped_end

    def raycast_update(
        self,
        origin_xyz: np.ndarray,
        points_xyz: np.ndarray,
        timestamp: float,
        source: int = SOURCE_LIDAR,
    ) -> None:
        """
        Raycast from origin_xyz to each row of points_xyz.

        Endpoint cells count as hits and cells along the ray as misses; an
        endpoint always wins over another ray's pass-through in the same batch.

        Each cell gets at most one update per call: one hit if any point ends
        in it, otherwise one miss if any ray passes through it. A cell is
        OCCUPIED while its log-odds is above 0 (plus a tiny epsilon), so a single stray or grazing
        ray cannot erase a cell that has been confirmed several times.
        Hits refresh the cell's timestamp; misses do not, so decay() can
        still expire cells that are no longer being hit.
        """
        origin_xyz = np.asarray(origin_xyz, dtype=np.float64)

        endpoint_cells = set()
        intermediate_cells = set()
        for point in points_xyz:
            # The true endpoint must itself be a valid cell -- an
            # out-of-grid point (e.g. beyond x/y range) is skipped
            # entirely, same as before this method clipped rays.
            end_idx = self.world_to_index(*point)
            if end_idx is None:
                continue
            # Only the ray's start gets clipped to the grid: the origin
            # (sensor position) need not itself be inside the grid, e.g.
            # a sensor mounted above a thin ground-hugging z-band.
            clipped = self._clip_segment_to_grid(origin_xyz, point)
            if clipped is None:
                continue
            clipped_origin, _ = clipped
            origin_idx = self.world_to_index(*clipped_origin)
            if origin_idx is None:
                continue
            cells = bresenham3d(origin_idx, end_idx)
            for cell in cells[:-1]:
                if self.in_bounds(*cell):
                    intermediate_cells.add(cell)
            endpoint_cells.add(end_idx)

        # A cell that's an intermediate for one ray but another ray's direct
        # endpoint this same frame is a hit, not a miss -- endpoints win.
        for cell in intermediate_cells - endpoint_cells:
            self._update_cell(cell, self.miss_logodds, source)

        for cell in endpoint_cells:
            if self.in_bounds(*cell):
                self._update_cell(cell, self.hit_logodds, source)
                self.timestamp[cell] = timestamp

    def _update_cell(
        self, cell: Tuple[int, int, int], delta: float, source: int
    ) -> None:
        value = min(max(self.logodds[cell] + delta, self.logodds_min), self.logodds_max)
        self.logodds[cell] = value
        self.cost[cell] = OCCUPIED if value > OCCUPIED_LOGODDS_EPS else FREE
        self.source[cell] = source

    def decay(self, now: float) -> None:
        """
        Revert stale occupied voxels to unknown (not free).

        Decay means "no current evidence", which is not the same claim as
        "confirmed clear" — reverting to unknown is the conservative choice
        for a planner consuming this grid.
        """
        stale = (self.cost == OCCUPIED) & (
            (now - self.timestamp) > self.persistence_sec
        )
        self.cost[stale] = UNKNOWN
        self.source[stale] = SOURCE_NONE
        self.logodds[stale] = 0.0

    # ---- derived views (non-mutating) -----------------------------------

    def inflate(self, inflation_radius: float) -> np.ndarray:
        """Return an inflated cost array; does not mutate self.cost."""
        if inflation_radius <= 0:
            return self.cost.copy()

        occupied_mask = self.cost == OCCUPIED
        if not occupied_mask.any():
            return self.cost.copy()

        dist = distance_transform_edt(
            ~occupied_mask, sampling=(self.resolution,) * 3
        )

        inflated = self.cost.copy()
        within_radius = (dist > 0) & (dist <= inflation_radius)
        falloff = np.clip(100.0 * (1.0 - dist / inflation_radius), 0, 100).astype(
            np.int8
        )
        # Only raise cost where inflation exceeds what's already there
        # (never lower an already-occupied or higher-cost cell).
        apply_mask = within_radius & (falloff > inflated)
        inflated[apply_mask] = falloff[apply_mask]
        return inflated

    def max_projection_2d(self) -> np.ndarray:
        """2D (nx, ny) projection: occupied(100) > free(0) > unknown(-1)."""
        return np.max(self.cost, axis=2)

    # ---- serialization ----------------------------------------------------

    def occupied_points(self) -> np.ndarray:
        ix, iy, iz = np.where(self.cost == OCCUPIED)
        if len(ix) == 0:
            return np.empty((0, 3), dtype=np.float64)
        xs = self.origin[0] + (ix + 0.5) * self.resolution
        ys = self.origin[1] + (iy + 0.5) * self.resolution
        zs = self.origin[2] + (iz + 0.5) * self.resolution
        return np.stack([xs, ys, zs], axis=1)

    def inflated_points_with_cost(self, inflated: np.ndarray) -> np.ndarray:
        ix, iy, iz = np.where(inflated > FREE)
        if len(ix) == 0:
            return np.empty((0, 4), dtype=np.float64)
        xs = self.origin[0] + (ix + 0.5) * self.resolution
        ys = self.origin[1] + (iy + 0.5) * self.resolution
        zs = self.origin[2] + (iz + 0.5) * self.resolution
        costs = inflated[ix, iy, iz].astype(np.float64)
        return np.stack([xs, ys, zs, costs], axis=1)

    def to_flat_int8(self) -> np.ndarray:
        return self.cost.flatten(order="C")

    def metadata_dict(
        self, inflation_radius: float, timestamp: float, sensor_mode: str
    ) -> dict:
        return {
            "frame_id": "base_link",
            "size_x": self.nx * self.resolution,
            "size_y": self.ny * self.resolution,
            "size_z": self.nz * self.resolution,
            "resolution": self.resolution,
            "grid_nx": self.nx,
            "grid_ny": self.ny,
            "grid_nz": self.nz,
            "origin": list(self.origin),
            "inflation_radius": inflation_radius,
            "timestamp": timestamp,
            "sensor_mode": sensor_mode,
        }
