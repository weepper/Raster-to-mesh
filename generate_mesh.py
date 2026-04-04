import os
import glob
import rasterio
from rasterio.enums import Resampling
import argparse
import numpy as np
from typing import Optional, BinaryIO
from numba import njit
import struct
import warnings
import logging
import tempfile
import gc
from dataclasses import dataclass
from collections import deque
from tqdm import tqdm

warnings.filterwarnings(
    "ignore", category=UserWarning, module="multiprocessing.resource_tracker"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Safety limits for extreme precision settings
MAX_TRIANGLES_HARD_LIMIT = 100_000_000  # 100M triangles - prevents memory exhaustion
MAX_TRIANGLES_STL_LIMIT = 4_294_967_295  # uint32 max for STL format
MIN_ERROR_THRESHOLD_M = (
    0.00006  # 0.06mm minimum - matches typical 0.06mm layer height printer capability
)
MAX_DEPTH_HARD_LIMIT = 50  # Prevents runaway refinement while allowing deep recursion
HIGH_RES_THRESHOLD = 2.0  # Resolution <= 2m uses nearest neighbor for sharp edges

STL_DTYPE = np.dtype(
    [
        ("normals", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("v3", "<f4", (3,)),
        ("attr", "<u2"),
    ]
)

STL_TRIANGLE_SIZE = STL_DTYPE.itemsize  # 50 bytes per triangle


# =============================================================================
# STL Streaming Writer
# =============================================================================


class StreamingSTLWriter:
    """Write STL triangles directly to disk without holding in memory."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.temp_path = filepath + ".tmp"
        self.triangle_count = 0
        self.file: Optional[BinaryIO] = None
        # Open file immediately
        self.file = open(self.temp_path, "wb")
        self.file.write(b"Binary STL - Streaming Writer".ljust(80, b"\x00"))
        self.file.write(struct.pack("<I", 0))  # Placeholder count

    def __enter__(self):
        self.file = open(self.temp_path, "wb")
        # Write placeholder header (will update later)
        self.file.write(b"Binary STL - Streaming Writer".ljust(80, b"\x00"))
        self.file.write(struct.pack("<I", 0))  # Placeholder count
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

        if exc_type is None:
            # Update triangle count in header
            with open(self.temp_path, "r+b") as f:
                f.seek(80)
                f.write(struct.pack("<I", self.triangle_count))

            # Phase 7: Final check and Normals Recomputation
            try:
                self.recompute_normals()
            except Exception as e:
                logger.warning(f"Failed to recompute normals: {e}")

            # Rename to final path
            os.replace(self.temp_path, self.filepath)
            logger.info(f"Wrote {self.triangle_count:,} triangles to {self.filepath}")
        else:
            # Cleanup on error
            try:
                os.unlink(self.temp_path)
            except Exception:
                pass

    def recompute_normals(self):
        """Recompute all face normals in the binary STL for correctness.

        Uses memory-mapped chunked processing for speed.
        """
        if self.triangle_count == 0:
            return

        logger.info("Recomputing mesh normals...")
        chunk_size = 100_000

        data = np.memmap(
            self.temp_path,
            dtype=STL_DTYPE,
            mode="r+",
            offset=84,
            shape=(self.triangle_count,),
        )

        for start in range(0, self.triangle_count, chunk_size):
            end = min(start + chunk_size, self.triangle_count)
            chunk = data[start:end]

            edge1 = chunk["v2"] - chunk["v1"]
            edge2 = chunk["v3"] - chunk["v1"]
            normals = np.cross(edge1, edge2)
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            lengths[lengths == 0] = 1.0
            chunk["normals"] = normals / lengths

        data.flush()
        del data

    def write_triangles(self, triangles: np.ndarray):
        """Write triangles to file."""
        if len(triangles) == 0:
            return

        # Check STL format limit (uint32 max = 4,294,967,295)
        if self.triangle_count + len(triangles) > MAX_TRIANGLES_STL_LIMIT:
            raise ValueError(
                f"Cannot write {len(triangles)} triangles - would exceed STL format limit "
                f"of {MAX_TRIANGLES_STL_LIMIT:,} triangles. Current count: {self.triangle_count:,}"
            )

        if self.file is not None:
            self.file.write(triangles.tobytes())
        self.triangle_count += len(triangles)

    def get_count(self) -> int:
        return self.triangle_count

    def close(self):
        """Close the file and update the header."""
        if self.file:
            self.file.close()
            self.file = None
        if os.path.exists(self.temp_path):
            with open(self.temp_path, "r+b") as f:
                f.seek(80)
                f.write(struct.pack("<I", self.triangle_count))


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def log_memory(phase: str):
    """Log current memory usage."""
    mem = get_memory_usage_mb()
    if mem > 0:
        logger.info(f"[{phase}] Memory usage: {mem:.1f} MB")


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class MeshConfig:
    width_cm: float
    length_cm: float
    height_cm: float
    base_cm: float
    target_faces: int
    resolution: float
    planar_tol: float = 0.05  # in meters
    smoothing: int = 0
    z_range_input: float = 1.0  # Scaling context
    uniform_grid: bool = True  # Use uniform grid instead of RTIN for terrain

    @property
    def width_mm(self) -> float:
        return self.width_cm * 10.0

    @property
    def length_mm(self) -> float:
        return self.length_cm * 10.0

    @property
    def height_mm(self) -> float:
        return self.height_cm * 10.0

    @property
    def base_mm(self) -> float:
        return self.base_cm * 10.0


@dataclass
class GeoBounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center_x(self) -> float:
        return self.min_x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.max_y - self.height / 2


# =============================================================================
# Tile Processing (Low Memory)
# =============================================================================


def scan_tile_bounds(tif_files: list) -> GeoBounds:
    """Scan tiles for bounds without loading data."""
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    for fp in tqdm(tif_files, desc="Scanning bounds", unit="file"):
        try:
            with rasterio.open(fp) as src:
                b = src.bounds
                min_x, min_y = min(min_x, b.left), min(min_y, b.bottom)
                max_x, max_y = max(max_x, b.right), max(max_y, b.top)
        except Exception as e:
            logger.warning(f"Could not read {fp}: {e}")

    if min_x == float("inf"):
        raise ValueError("No valid tiles found")

    return GeoBounds(min_x, min_y, max_x, max_y)


def stitch_tiles_lowmem(
    tif_files: list,
    bounds: GeoBounds,
    resolution: float,
    error_resolution=None,
) -> tuple:
    """
    Stitch tiles using memory-mapped array.
    Returns (Z_memmap, width, height, temp_path, Z_error_memmap, error_temp_path)

    Args:
        tif_files: List of GeoTIFF file paths
        bounds: Geographic bounds
        resolution: Target resolution for base mesh (may be downsampled)
        error_resolution: Resolution for error computation (defaults to source resolution if None)
    """
    global_width = int(np.ceil(bounds.width / resolution)) + 2
    global_height = int(np.ceil(bounds.height / resolution)) + 2

    logger.info(
        f"Canvas: {global_width} x {global_height} ({global_width * global_height * 4 / 1e6:.1f} MB on disk)"
    )

    # Create memory-mapped file for base mesh (downsampled)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".elevation.mmap")
    temp_path = temp_file.name
    temp_file.close()

    Z_global = np.memmap(
        temp_path, dtype=np.float32, mode="w+", shape=(global_height, global_width)
    )
    Z_global[:] = np.nan

    # Create memory-mapped file for error computation (high-res)
    # Use source resolution or 1m default for error computation
    if error_resolution is None:
        # Try to get native resolution from first file
        try:
            with rasterio.open(tif_files[0]) as src:
                error_resolution = src.res[0]  # Native resolution
        except:
            error_resolution = 1.0  # Default to 1m

    error_width = int(np.ceil(bounds.width / error_resolution)) + 2
    error_height = int(np.ceil(bounds.height / error_resolution)) + 2

    logger.info(
        f"Error canvas: {error_width} x {error_height} ({error_width * error_height * 4 / 1e6:.1f} MB on disk) @ {error_resolution}m"
    )

    error_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".error.mmap")
    error_temp_path = error_temp_file.name
    error_temp_file.close()

    Z_error = np.memmap(
        error_temp_file.name,
        dtype=np.float32,
        mode="w+",
        shape=(error_height, error_width),
    )
    Z_error[:] = np.nan

    # Process tiles one at a time
    for fp in tqdm(tif_files, desc="Stitching tiles", unit="file"):
        try:
            with rasterio.open(fp) as src:
                # Read at target resolution for base mesh
                tile_w = int(np.ceil((src.bounds.right - src.bounds.left) / resolution))
                tile_h = int(np.ceil((src.bounds.top - src.bounds.bottom) / resolution))

                # Read at error resolution for error computation
                error_tile_w = int(
                    np.ceil((src.bounds.right - src.bounds.left) / error_resolution)
                )
                error_tile_h = int(
                    np.ceil((src.bounds.top - src.bounds.bottom) / error_resolution)
                )

                if tile_w <= 0 or tile_h <= 0:
                    continue

                # Read tile data at target resolution (downsampled for base mesh)
                # Use bilinear for smooth terrain, but nearest for sharp features
                if (
                    resolution <= HIGH_RES_THRESHOLD
                ):  # High resolution - preserve sharp features
                    data = src.read(
                        1, out_shape=(tile_h, tile_w), resampling=Resampling.nearest
                    )
                else:
                    data = src.read(
                        1, out_shape=(tile_h, tile_w), resampling=Resampling.bilinear
                    )

                # Read tile data at error resolution (high-res for error computation)
                # Always use nearest neighbor to preserve sharp edges (buildings, trees)
                data_error = src.read(
                    1,
                    out_shape=(error_tile_h, error_tile_w),
                    resampling=Resampling.nearest,
                )

                # Place in base mesh canvas
                col_start = (
                    int(np.round((src.bounds.left - bounds.min_x) / resolution)) + 1
                )
                row_start = (
                    int(np.round((bounds.max_y - src.bounds.top) / resolution)) + 1
                )

                row_end = min(row_start + tile_h, global_height)
                col_end = min(col_start + tile_w, global_width)

                actual_h = row_end - row_start
                actual_w = col_end - col_start

                if actual_h > 0 and actual_w > 0:
                    data_slice = data[:actual_h, :actual_w]
                    valid_mask = data_slice > -100
                    Z_global[row_start:row_end, col_start:col_end][valid_mask] = (
                        data_slice[valid_mask]
                    )
                    Z_global.flush()

                # Place in error canvas
                error_col_start = (
                    int(np.round((src.bounds.left - bounds.min_x) / error_resolution))
                    + 1
                )
                error_row_start = (
                    int(np.round((bounds.max_y - src.bounds.top) / error_resolution))
                    + 1
                )

                error_row_end = min(error_row_start + error_tile_h, error_height)
                error_col_end = min(error_col_start + error_tile_w, error_width)

                error_actual_h = error_row_end - error_row_start
                error_actual_w = error_col_end - error_col_start

                if error_actual_h > 0 and error_actual_w > 0:
                    error_data_slice = data_error[:error_actual_h, :error_actual_w]
                    error_valid_mask = error_data_slice > -100
                    Z_error[
                        error_row_start:error_row_end, error_col_start:error_col_end
                    ][error_valid_mask] = error_data_slice[error_valid_mask]
                    Z_error.flush()

                del data, data_error

        except Exception as e:
            logger.warning(f"Error processing {fp}: {e}")

    gc.collect()

    return (
        Z_global,
        global_width,
        global_height,
        temp_path,
        Z_error,
        error_width,
        error_height,
        error_temp_path,
        error_resolution,
    )


def _compute_elevation_range(Z: np.ndarray) -> tuple:
    """Compute min/max elevation range from raster."""
    z_min, z_max = np.inf, -np.inf
    chunk_rows = 1000

    for r in range(0, Z.shape[0], chunk_rows):
        chunk = Z[r : r + chunk_rows]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 0:
            z_min = min(z_min, valid.min())
            z_max = max(z_max, valid.max())

    return z_min, z_max


def _apply_normalization(
    Z: np.ndarray, z_min: float, z_range: float, config: MeshConfig
):
    """Apply normalization to raster using pre-computed range."""
    chunk_rows = 1000

    for r in range(0, Z.shape[0], chunk_rows):
        chunk = Z[r : r + chunk_rows]
        valid = ~np.isnan(chunk)
        chunk[valid] = ((chunk[valid] - z_min) / z_range) * (
            config.height_mm - config.base_mm
        ) + config.base_mm


def normalize_elevation_inplace(Z: np.ndarray, config: MeshConfig):
    """Normalize elevation in-place to minimize memory."""
    # Find min/max in chunks to avoid loading all into RAM
    z_min, z_max = _compute_elevation_range(Z)
    z_range = z_max - z_min if (z_max - z_min) > 0 else 1.0
    logger.info(f"Elevation range: {z_min:.2f} to {z_max:.2f}")

    config.z_range_input = z_range

    # Apply normalization
    _apply_normalization(Z, z_min, z_range, config)


def repair_elevation_inplace(Z: np.ndarray, smoothing_it: int = 0):
    """
    Repair elevation grid in-place:
    1. Fill single-pixel NaN holes.
    2. Apply simple box-blur smoothing if smoothing_it > 0.
    Uses chunked processing to keep memory usage low.
    """
    h, w = Z.shape
    if h < 3 or w < 3:
        return

    logger.info(f"Phase 4.5: Repairing elevation (smoothing_it={smoothing_it})...")

    # Phase 1: Hole filling (single-pixel holes)
    # Process in chunks to avoid memory spikes
    chunk_rows = 1000
    for r_start in range(1, h - 1, chunk_rows):
        r_end = min(r_start + chunk_rows, h - 1)
        # We need a small buffer of rows around the chunk
        for r in range(r_start, r_end):
            row = Z[r]
            nan_indices = np.where(np.isnan(row[1:-1]))[0] + 1
            if len(nan_indices) == 0:
                continue

            prev_row = Z[r - 1]
            next_row = Z[r + 1]

            for c in nan_indices:
                # Check 4-connectivity
                neighbors = [row[c - 1], row[c + 1], prev_row[c], next_row[c]]
                valid_neighbors = [v for v in neighbors if not np.isnan(v)]
                if len(valid_neighbors) >= 3:
                    row[c] = sum(valid_neighbors) / len(valid_neighbors)

    # Phase 2: Smoothing (vectorized 3×3 box blur)
    for _ in range(smoothing_it):
        Z_old = np.array(Z, dtype=np.float32)

        # Compute 3×3 neighborhood sum and count for interior pixels
        total = np.zeros((h - 2, w - 2), dtype=np.float64)
        count = np.zeros((h - 2, w - 2), dtype=np.int32)

        for dr in range(3):
            for dc in range(3):
                block = Z_old[dr : dr + h - 2, dc : dc + w - 2]
                valid_block = ~np.isnan(block)
                total += np.where(valid_block, block, 0.0)
                count += valid_block.astype(np.int32)

        # Only update pixels that were originally valid with ≥1 valid neighbor
        valid_interior = ~np.isnan(Z_old[1:-1, 1:-1])
        count_safe = np.maximum(count, 1)
        smoothed = (total / count_safe).astype(np.float32)
        update = valid_interior & (count > 0)
        Z[1:-1, 1:-1][update] = smoothed[update]
        del Z_old, total, count, smoothed

    if hasattr(Z, "flush"):
        Z.flush()  # type: ignore[attr-defined]


def add_skirt_inplace(Z: np.ndarray):
    """Add skirt in-place at all outer boundaries of valid data.

    Unlike the previous implementation that relied on NaN neighbors,
    this adds skirt at the outer edge of valid data in all directions.
    """
    h, w = Z.shape

    # Find bounding box of valid data
    valid = ~np.isnan(Z)
    valid_any_row = np.any(valid, axis=1)
    valid_any_col = np.any(valid, axis=0)

    if not np.any(valid_any_row) or not np.any(valid_any_col):
        return

    # Find first and last valid row/column
    r_min = np.argmax(valid_any_row)
    r_max = h - 1 - np.argmax(valid_any_row[::-1])
    c_min = np.argmax(valid_any_col)
    c_max = w - 1 - np.argmax(valid_any_col[::-1])

    # Add skirt to top edge (row r_min)
    if r_min > 0:
        Z[r_min - 1, c_min : c_max + 1] = 0.0

    # Add skirt to bottom edge (row r_max)
    if r_max < h - 1:
        Z[r_max + 1, c_min : c_max + 1] = 0.0

    # Add skirt to left edge (column c_min)
    if c_min > 0:
        Z[r_min : r_max + 1, c_min - 1] = 0.0

    # Add skirt to right edge (column c_max)
    if c_max < w - 1:
        Z[r_min : r_max + 1, c_max + 1] = 0.0

    # Also fill corners if they're NaN
    corners = [
        (r_min - 1, c_min - 1),
        (r_min - 1, c_max + 1),
        (r_max + 1, c_min - 1),
        (r_max + 1, c_max + 1),
    ]
    for r, c in corners:
        if 0 <= r < h and 0 <= c < w and np.isnan(Z[r, c]):
            Z[r, c] = 0.0

    if hasattr(Z, "flush"):
        Z.flush()  # type: ignore[attr-defined]


# =============================================================================
# Triangle Generation (Streaming)
# =============================================================================

# =============================================================================
# RTIN Terrain Meshing (Martini algorithm, Numba-accelerated)
# =============================================================================
#
# Implements Right-Triangulated Irregular Networks for adaptive heightmap
# meshing.  Based on the Martini algorithm by Vladimir Agafonkin (Mapbox).
#
# The algorithm operates in three phases:
#   1. Precompute triangle coordinates for the 2^k+1 binary tree (once).
#   2. Compute approximation errors bottom-up (O(n), Numba JIT).
#   3. Extract an adaptive mesh top-down at a given error threshold.
#
# Chunk boundaries are locked (errors set to infinity) to guarantee
# matching triangulations between adjacent chunks.
# =============================================================================


def _martini_precompute_coords(grid_size: int) -> np.ndarray:
    """Precompute triangle coordinates for the RTIN binary tree.

    Returns a (num_triangles, 4) int32 array of (ax, ay, bx, by) for every
    triangle in the implicit binary tree, ordered so that leaves come last
    (i.e. iterating in reverse is bottom-up).
    """
    tile_size = grid_size - 1
    num_triangles = tile_size * tile_size * 2 - 2
    coords = np.zeros(num_triangles * 4, dtype=np.int32)

    for i in range(num_triangles):
        tri_id = i + 2
        ax, ay, bx, by, cx, cy = 0, 0, 0, 0, 0, 0

        if tri_id & 1:
            bx = by = cx = tile_size
        else:
            ax = ay = cy = tile_size

        while True:
            tri_id >>= 1
            if tri_id <= 1:
                break
            mx = (ax + bx) >> 1
            my = (ay + by) >> 1
            if tri_id & 1:
                bx, by = ax, ay
                ax, ay = cx, cy
            else:
                ax, ay = bx, by
                bx, by = cx, cy
            cx, cy = mx, my

        k = i * 4
        coords[k] = ax
        coords[k + 1] = ay
        coords[k + 2] = bx
        coords[k + 3] = by

    return coords


# Cache precomputed coords per grid_size to avoid recomputation.
_MARTINI_CACHE: dict[int, tuple[np.ndarray, int, int]] = {}


def _get_martini(grid_size: int) -> tuple[np.ndarray, int, int]:
    """Return (coords, num_triangles, num_parent_triangles) for grid_size."""
    if grid_size not in _MARTINI_CACHE:
        tile_size = grid_size - 1
        num_tri = tile_size * tile_size * 2 - 2
        num_parent = num_tri - tile_size * tile_size
        coords = _martini_precompute_coords(grid_size)
        _MARTINI_CACHE[grid_size] = (coords, num_tri, num_parent)
    return _MARTINI_CACHE[grid_size]


@njit(cache=True, boundscheck=True)
def _rtin_compute_errors(
    coords, num_triangles, num_parent_triangles, terrain_flat, grid_size, errors, h, w
):
    """Compute RTIN approximation errors (Numba JIT, bottom-up)."""
    # Force full resolution on the chunk boundaries and padding
    for r in range(grid_size):
        for c in range(grid_size):
            if r >= h or c >= w or r == 0 or r == h - 1 or c == 0 or c == w - 1:
                errors[r * grid_size + c] = 1e30  # infinity representation

    for i in range(num_triangles - 1, -1, -1):
        k = i * 4
        ax, ay, bx, by = coords[k], coords[k + 1], coords[k + 2], coords[k + 3]
        mx, my = (ax + bx) >> 1, (ay + by) >> 1
        cx, cy = mx + my - ay, my + ax - mx

        mid_idx = my * grid_size + mx

        # Interpolation error at the hypotenuse midpoint
        v_a = terrain_flat[ay * grid_size + ax]
        v_b = terrain_flat[by * grid_size + bx]
        v_m = terrain_flat[mid_idx]

        mid_error = abs((v_a + v_b) * 0.5 - v_m)
        if mid_error > errors[mid_idx]:
            errors[mid_idx] = mid_error

        if i < num_parent_triangles:
            # Propagate child errors upward
            left_idx = ((ay + cy) >> 1) * grid_size + ((ax + cx) >> 1)
            right_idx = ((by + cy) >> 1) * grid_size + ((bx + cx) >> 1)
            e_l = errors[left_idx]
            e_r = errors[right_idx]
            if e_l > errors[mid_idx]:
                errors[mid_idx] = e_l
            if e_r > errors[mid_idx]:
                errors[mid_idx] = e_r

    return errors


@njit(cache=True, boundscheck=True)
def _rtin_count(ax, ay, bx, by, cx, cy, errors, grid_size, max_error, indices, counts):
    """Pass 1: Count vertices/triangles and index them."""
    mx = (ax + bx) >> 1
    my = (ay + by) >> 1

    if abs(ax - cx) + abs(ay - cy) > 1 and errors[my * grid_size + mx] > max_error:
        _rtin_count(
            cx, cy, ax, ay, mx, my, errors, grid_size, max_error, indices, counts
        )
        _rtin_count(
            bx, by, cx, cy, mx, my, errors, grid_size, max_error, indices, counts
        )
    else:
        # Add tracking for uniqueness
        if indices[ay * grid_size + ax] == 0:
            counts[0] += 1
            indices[ay * grid_size + ax] = counts[0]
        if indices[by * grid_size + bx] == 0:
            counts[0] += 1
            indices[by * grid_size + bx] = counts[0]
        if indices[cy * grid_size + cx] == 0:
            counts[0] += 1
            indices[cy * grid_size + cx] = counts[0]

        counts[1] += 1


@njit(cache=True, boundscheck=True)
def _rtin_extract(
    ax,
    ay,
    bx,
    by,
    cx,
    cy,
    errors,
    grid_size,
    max_error,
    indices,
    vertices,
    triangles,
    counts,
):
    """Pass 2: Extract distinct vertices and triangles."""
    mx = (ax + bx) >> 1
    my = (ay + by) >> 1

    if abs(ax - cx) + abs(ay - cy) > 1 and errors[my * grid_size + mx] > max_error:
        _rtin_extract(
            cx,
            cy,
            ax,
            ay,
            mx,
            my,
            errors,
            grid_size,
            max_error,
            indices,
            vertices,
            triangles,
            counts,
        )
        _rtin_extract(
            bx,
            by,
            cx,
            cy,
            mx,
            my,
            errors,
            grid_size,
            max_error,
            indices,
            vertices,
            triangles,
            counts,
        )
    else:
        id_a = indices[ay * grid_size + ax] - 1
        id_b = indices[by * grid_size + bx] - 1
        id_c = indices[cy * grid_size + cx] - 1

        # Writes to same location are idempotent
        vertices[id_a, 0] = ax
        vertices[id_a, 1] = ay
        vertices[id_b, 0] = bx
        vertices[id_b, 1] = by
        vertices[id_c, 0] = cx
        vertices[id_c, 1] = cy

        t_idx = counts[1]
        triangles[t_idx, 0] = id_a
        triangles[t_idx, 1] = id_b
        triangles[t_idx, 2] = id_c
        counts[1] += 1


def _next_power_of_two_plus_one(n: int) -> int:
    """Return the smallest 2^k + 1 >= n."""
    if n <= 2:
        return 3
    p = 1
    while p + 1 < n:
        p <<= 1
    return p + 1


def _generate_flat_mesh(bounds, scale_xy, resolution, triangle_size_pixels):
    """Generate flat grid mesh covering the bounding box."""
    width_pixels = int(np.ceil((bounds.width / resolution)))
    height_pixels = int(np.ceil((bounds.height / resolution)))

    n_cols = max(1, width_pixels // triangle_size_pixels)
    n_rows = max(1, height_pixels // triangle_size_pixels)

    x_vals = np.linspace(bounds.min_x, bounds.max_x, n_cols + 1)
    y_vals = np.linspace(bounds.min_y, bounds.max_y, n_rows + 1)

    x_vals = (x_vals - bounds.center_x) * scale_xy
    y_vals = (y_vals - bounds.center_y) * scale_xy

    xx, yy = np.meshgrid(x_vals, y_vals)
    vertices = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros(len(xx.ravel()), dtype=np.float32)]
    )

    triangles = []
    for i in range(n_rows):
        for j in range(n_cols):
            v0 = i * (n_cols + 1) + j
            v1 = v0 + 1
            v2 = v0 + (n_cols + 1)
            v3 = v2 + 1

            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    triangles = np.array(triangles, dtype=np.int32)

    logger.info(f"Flat mesh: {len(vertices)} vertices, {len(triangles)} triangles")

    return vertices, triangles


def _sliding_windows(shape, window_size, overlap_ratio):
    """Generate sliding window positions for streaming RTIN processing."""
    h, w = shape
    overlap = int(window_size * overlap_ratio)
    stride = window_size - overlap

    row = 0
    while row < h:
        row_end = min(row + window_size, h)
        col = 0
        while col < w:
            col_end = min(col + window_size, w)

            core_row_start = row + overlap // 2 if row > 0 else row
            core_row_end = row_end - overlap // 2 if row_end < h else row_end
            core_col_start = col + overlap // 2 if col > 0 else col
            core_col_end = col_end - overlap // 2 if col_end < w else col_end

            yield (
                row,
                row_end,
                col,
                col_end,
                core_row_start,
                core_row_end,
                core_col_start,
                core_col_end,
            )

            col += stride
        row += stride


@njit(cache=True)
def _bilinear_sample_njit(Z, row, col):
    """Numba-accelerated bilinear interpolation of raster at (row, col)."""
    h, w = Z.shape
    r0 = int(np.floor(row))
    c0 = int(np.floor(col))
    r1 = min(r0 + 1, h - 1)
    c1 = min(c0 + 1, w - 1)

    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return 0.0

    tx = col - c0
    ty = row - r0

    # Handle NaN values - use 0.0 as fallback
    z00 = Z[r0, c0]
    if np.isnan(z00):
        z00 = 0.0

    z10 = Z[r0, c1]
    if np.isnan(z10):
        z10 = 0.0

    z01 = Z[r1, c0]
    if np.isnan(z01):
        z01 = 0.0

    z11 = Z[r1, c1]
    if np.isnan(z11):
        z11 = 0.0

    return (
        (1.0 - tx) * (1.0 - ty) * z00
        + tx * (1.0 - ty) * z10
        + (1.0 - tx) * ty * z01
        + tx * ty * z11
    )


def _bilinear_sample(Z, row, col):
    """Bilinear interpolation of raster at (row, col).

    Wrapper that calls the Numba-accelerated implementation.
    """
    return _bilinear_sample_njit(Z, row, col)


@njit(cache=True)
def _nearest_sample_njit(Z, row, col):
    """Numba-accelerated nearest neighbor sampling of raster at (row, col).

    Preserves sharp edges by returning the value of the closest pixel without interpolation.
    """
    h, w = Z.shape

    # Round to nearest pixel
    r = int(np.round(row))
    c = int(np.round(col))

    # Bounds check
    if r < 0 or c < 0 or r >= h or c >= w:
        return 0.0

    z = Z[r, c]
    if np.isnan(z):
        return 0.0

    return z


def _nearest_sample(Z, row, col):
    """Nearest neighbor sampling of raster at (row, col).

    Preserves sharp features by returning the closest pixel value without interpolation.
    Wrapper that calls the Numba-accelerated implementation.
    """
    return _nearest_sample_njit(Z, row, col)


@njit(cache=True)
def _compute_triangle_error_njit(
    v0x,
    v0y,
    v0z,
    v1x,
    v1y,
    v1z,
    v2x,
    v2y,
    v2z,
    Z_raster,
    bounds_min_x,
    bounds_max_y,
    bounds_center_x,
    bounds_center_y,
    scale_xy,
    resolution,
):
    """Numba-accelerated triangle error computation.

    Computes sum of squared errors between triangle plane and raster pixels inside triangle.
    Uses manual loops instead of meshgrid to avoid memory allocation overhead.
    """
    h_raster, w_raster = Z_raster.shape

    # Convert model coordinates to raster coordinates
    x_geo0 = v0x / scale_xy + bounds_center_x
    y_geo0 = v0y / scale_xy + bounds_center_y
    c0 = (x_geo0 - bounds_min_x) / resolution + 1.0
    r0 = (bounds_max_y - y_geo0) / resolution + 1.0

    x_geo1 = v1x / scale_xy + bounds_center_x
    y_geo1 = v1y / scale_xy + bounds_center_y
    c1 = (x_geo1 - bounds_min_x) / resolution + 1.0
    r1 = (bounds_max_y - y_geo1) / resolution + 1.0

    x_geo2 = v2x / scale_xy + bounds_center_x
    y_geo2 = v2y / scale_xy + bounds_center_y
    c2 = (x_geo2 - bounds_min_x) / resolution + 1.0
    r2 = (bounds_max_y - y_geo2) / resolution + 1.0

    # Compute bounding box in raster space
    min_r_float = min(r0, min(r1, r2))
    max_r_float = max(r0, max(r1, r2))
    min_c_float = min(c0, min(c1, c2))
    max_c_float = max(c0, max(c1, c2))

    min_r = max(0, int(min_r_float))
    max_r = min(h_raster - 1, int(max_r_float) + 1)
    min_c = max(0, int(min_c_float))
    max_c = min(w_raster - 1, int(max_c_float) + 1)

    if max_r <= min_r or max_c <= min_c:
        return 0.0

    # Skip triangles too close to raster boundaries to avoid edge artifacts
    # This prevents over-subdivision at edges where error computation is unreliable
    boundary_margin = 2
    if (
        min_r < boundary_margin
        or max_r >= h_raster - boundary_margin
        or min_c < boundary_margin
        or max_c >= w_raster - boundary_margin
    ):
        return 0.0  # Return zero error for boundary triangles to prevent subdivision

    # Precompute triangle plane parameters
    ax, ay, az = v0x, v0y, v0z
    bx, by, bz = v1x, v1y, v1z
    cx, cy, cz = v2x, v2y, v2z

    det = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(det) < 1e-10:
        return 0.0

    inv_det = 1.0 / det

    sse = 0.0

    # Iterate over bounding box pixels
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            # Convert raster coord to model coord
            x_geo = bounds_min_x + (c - 1) * resolution
            y_geo = bounds_max_y - (r - 1) * resolution
            mx = (x_geo - bounds_center_x) * scale_xy
            my = (y_geo - bounds_center_y) * scale_xy

            # Barycentric coordinates
            u = ((mx - ax) * (cy - ay) - (my - ay) * (cx - ax)) * inv_det
            v = ((bx - ax) * (my - ay) - (by - ay) * (mx - ax)) * inv_det
            w = 1.0 - u - v

            # Check if point is inside triangle
            if u >= 0.0 and v >= 0.0 and w >= 0.0:
                # Interpolate Z from triangle plane
                z_tri = w * az + u * bz + v * cz
                # Get raster Z value
                z_rast = Z_raster[r, c]
                # Only count valid (non-NaN) raster values
                if not np.isnan(z_rast):
                    diff = z_tri - z_rast
                    sse += diff * diff

    return sse


def _compute_triangle_error(v0, v1, v2, Z_raster, bounds, scale_xy, resolution):
    """Compute sum of squared errors between triangle plane and raster pixels inside triangle.

    Wrapper function that calls the Numba-accelerated implementation.
    """
    return _compute_triangle_error_njit(
        v0[0],
        v0[1],
        v0[2],
        v1[0],
        v1[1],
        v1[2],
        v2[0],
        v2[1],
        v2[2],
        Z_raster,
        bounds.min_x,
        bounds.max_y,
        bounds.center_x,
        bounds.center_y,
        scale_xy,
        resolution,
    )


def _refine_triangles_batch(
    triangles_list,
    vertices,
    Z_global,
    bounds,
    scale_xy,
    resolution,
    error_threshold,
    max_depth,
    edge_to_tris,
    progress_counter=None,
    Z_error=None,
    error_resolution=None,
):
    """Process all triangles in batch using single-threaded sequential processing.

    Note: Parallel processing has been disabled due to race conditions in shared
    memory. Single-threaded processing with synchronized edge splitting provides
    reliable, fast results.
    """
    # Always use sequential processing - parallel mode caused mesh corruption
    return _refine_triangles_batch_sequential(
        triangles_list,
        vertices,
        Z_global,
        bounds,
        scale_xy,
        resolution,
        error_threshold,
        max_depth,
        edge_to_tris,
        progress_counter=progress_counter,
        Z_error=Z_error,
        error_resolution=error_resolution,
    )


def _split_edge_atomic(
    split_edge,
    vertices,
    triangles_list,
    edge_to_tris,
    Z_global,
    bounds,
    scale_xy,
    resolution,
    processed,
    use_nearest=False,
):
    """
    Split an edge atomically across ALL triangles sharing it.
    Returns the new vertex index and list of (old_tri_idx, new_tri_1, new_tri_2) tuples.

    Args:
        use_nearest: If True, use nearest neighbor sampling for midpoint vertices to preserve sharp edges.
    """
    v_a, v_b = split_edge
    v_a_data = vertices[v_a]
    v_b_data = vertices[v_b]

    # Create single midpoint vertex
    mid_x = (v_a_data[0] + v_b_data[0]) / 2
    mid_y = (v_a_data[1] + v_b_data[1]) / 2
    mid_z = _sample_z_from_raster(
        mid_x, mid_y, Z_global, bounds, scale_xy, resolution, use_nearest=use_nearest
    )
    if np.isnan(mid_z):
        mid_z = 0.0

    mid_idx = len(vertices)
    vertices.append((mid_x, mid_y, mid_z))

    # Find all triangles sharing this edge
    if split_edge not in edge_to_tris:
        return mid_idx, []

    affected_tris = edge_to_tris[split_edge].copy()
    new_triangles_info = []

    # Remove the edge from adjacency (it will be replaced by two new edges)
    del edge_to_tris[split_edge]

    for tri_idx in affected_tris:
        if tri_idx >= len(triangles_list) or triangles_list[tri_idx] is None:
            continue

        tri = triangles_list[tri_idx]
        i0, i1, i2 = tri[0], tri[1], tri[2]

        # Find which edge matches our split edge
        edges = [
            (
                tuple(sorted([i0, i1])),
                i2,
                [i0, i1, i2],
            ),  # edge, opposite_vertex, full_tri
            (tuple(sorted([i1, i2])), i0, [i1, i2, i0]),
            (tuple(sorted([i2, i0])), i1, [i2, i0, i1]),
        ]

        for edge_key, opp_vert, tri_order in edges:
            if edge_key == split_edge:
                # Split this triangle along the edge
                # Create two new triangles: (v_a, mid, opp) and (mid, v_b, opp)
                v_a_local, v_b_local = tri_order[0], tri_order[1]
                opp_local = tri_order[2]

                new_tri_1 = [v_a_local, mid_idx, opp_local]
                new_tri_2 = [mid_idx, v_b_local, opp_local]

                # Mark old triangle as processed
                processed.add(tri_idx)
                triangles_list[tri_idx] = None  # Mark for deletion

                new_triangles_info.append((tri_idx, new_tri_1, new_tri_2))
                break

    return mid_idx, new_triangles_info


def _update_adjacency_after_split(
    edge_to_tris,
    split_edge,
    new_triangles_info,
    next_tri_idx_start,
):
    """
    Update edge-to-triangles mapping after splitting edges.
    """
    v_a, v_b = split_edge
    mid_idx = None

    # First pass: collect all new edges and their triangles
    for i, (old_idx, tri_1, tri_2) in enumerate(new_triangles_info):
        tri_1_idx = next_tri_idx_start + 2 * i
        tri_2_idx = next_tri_idx_start + 2 * i + 1

        # Extract the midpoint from the triangles
        if mid_idx is None:
            # Find midpoint: it's the vertex in both tri_1 and tri_2 but not in {v_a, v_b}
            verts_1 = set(tri_1)
            verts_2 = set(tri_2)
            common = verts_1 & verts_2
            for v in common:
                if v != v_a and v != v_b:
                    mid_idx = v
                    break

        # Add new edges for tri_1
        for edge in [(tri_1[0], tri_1[1]), (tri_1[1], tri_1[2]), (tri_1[2], tri_1[0])]:
            edge_key = tuple(sorted(edge))
            if edge_key not in edge_to_tris:
                edge_to_tris[edge_key] = []
            edge_to_tris[edge_key].append(tri_1_idx)

        # Add new edges for tri_2
        for edge in [(tri_2[0], tri_2[1]), (tri_2[1], tri_2[2]), (tri_2[2], tri_2[0])]:
            edge_key = tuple(sorted(edge))
            if edge_key not in edge_to_tris:
                edge_to_tris[edge_key] = []
            edge_to_tris[edge_key].append(tri_2_idx)


def _refine_triangles_batch_sequential(
    triangles_list,
    vertices,
    Z_global,
    bounds,
    scale_xy,
    resolution,
    error_threshold,
    max_depth,
    edge_to_tris,
    progress_counter=None,
    Z_error=None,
    error_resolution=None,
):
    """
    Sequential implementation of batch refinement with synchronized edge splitting.
    When an edge is split, ALL triangles sharing that edge are split atomically.

    Args:
        progress_counter: Optional list with single element to track triangle count for progress bar
        Z_error: Optional high-resolution raster for error computation (uses Z_global if None)
        error_resolution: Resolution of error raster (uses resolution if None)
    """
    # Use high-res raster for error computation if available
    Z_for_error = Z_error if Z_error is not None else Z_global
    resolution_for_error = (
        error_resolution if error_resolution is not None else resolution
    )

    # Calculate safe upper bound - prevent integer overflow
    # Cap max_depth for estimation purposes
    safe_depth = min(max_depth, 30)  # 2**30 is ~1 billion, manageable
    estimated_max = len(triangles_list) * (2**safe_depth)

    # Hard limit to prevent memory exhaustion
    max_triangles = min(estimated_max, MAX_TRIANGLES_HARD_LIMIT)

    if progress_counter is None:
        logger.info(
            f"  Refinement: {len(triangles_list)} initial triangles, "
            f"max depth {max_depth}, limit {max_triangles:,} triangles"
        )

    # Use Python set for dynamic growth - indices can exceed initial estimate
    processed = set()

    # Use deque for O(1) popleft() instead of list O(n) pop(0)
    queue = deque([(i, 0) for i in range(len(triangles_list))])

    triangle_count = len(triangles_list)

    while queue:
        list_idx, depth = queue.popleft()

        if list_idx in processed:
            continue
        processed.add(list_idx)

        if list_idx >= len(triangles_list):
            continue

        tri = triangles_list[list_idx]
        if tri is None:
            continue

        i0, i1, i2 = tri[0], tri[1], tri[2]

        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]

        # Compute error using high-resolution raster
        error = _compute_triangle_error(
            v0, v1, v2, Z_for_error, bounds, scale_xy, resolution_for_error
        )

        if error <= error_threshold or depth >= max_depth:
            continue

        # Determine which edge to split (longest for now)
        d01 = (v0[0] - v1[0]) ** 2 + (v0[1] - v1[1]) ** 2
        d12 = (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2
        d20 = (v2[0] - v0[0]) ** 2 + (v2[1] - v0[1]) ** 2

        if d01 >= d12 and d01 >= d20:
            split_edge = tuple(sorted([i0, i1]))
        elif d12 >= d01 and d12 >= d20:
            split_edge = tuple(sorted([i1, i2]))
        else:
            split_edge = tuple(sorted([i2, i0]))

        # Check if edge is already marked for splitting by another triangle
        if split_edge not in edge_to_tris:
            # Edge was already split by another triangle, skip this one
            continue

        # Atomic edge split - splits ALL triangles sharing this edge
        # Use nearest neighbor for high-detail meshes to preserve sharp edges
        mid_idx, new_tri_info = _split_edge_atomic(
            split_edge,
            vertices,
            triangles_list,
            edge_to_tris,
            Z_global,
            bounds,
            scale_xy,
            resolution,
            processed,
            use_nearest=(
                resolution <= HIGH_RES_THRESHOLD
            ),  # Preserve sharp edges for high-res meshes
        )

        if not new_tri_info:
            continue

        # Check triangle count limit before adding more
        new_triangle_count = triangle_count + len(new_tri_info) * 2
        if new_triangle_count > MAX_TRIANGLES_HARD_LIMIT:
            logger.warning(
                f"Triangle count limit ({MAX_TRIANGLES_HARD_LIMIT:,}) reached, "
                f"stopping refinement at {triangle_count:,} triangles"
            )
            break

        if new_triangle_count > MAX_TRIANGLES_STL_LIMIT:
            logger.warning(
                f"Approaching STL format limit ({MAX_TRIANGLES_STL_LIMIT:,}), "
                f"stopping refinement"
            )
            break

        new_depth = depth + 1
        next_tri_idx = len(triangles_list)

        # Add new triangles to list
        for old_idx, t1, t2 in new_tri_info:
            triangles_list.append(t1)
            t1_idx = len(triangles_list) - 1
            triangles_list.append(t2)
            t2_idx = len(triangles_list) - 1

            # Add to queue for further refinement
            queue.append((t1_idx, new_depth))
            queue.append((t2_idx, new_depth))

            # Update triangle count for progress tracking
            triangle_count += 2
            if progress_counter is not None:
                progress_counter[0] = triangle_count

        # Update adjacency with new edges
        _update_adjacency_after_split(
            edge_to_tris, split_edge, new_tri_info, next_tri_idx
        )

    # Filter out None entries and return clean triangle list
    return [[tri[0], tri[1], tri[2]] for tri in triangles_list if tri is not None]


def _sample_z_from_raster(
    mx, my, Z_global, bounds, scale_xy, resolution, use_nearest=False
):
    """Sample z value from raster at model coordinates.

    Args:
        use_nearest: If True, use nearest neighbor sampling to preserve sharp edges.
                    If False, use bilinear interpolation for smooth surfaces.
    """
    h, w = Z_global.shape
    x_geo = mx / scale_xy + bounds.center_x
    y_geo = my / scale_xy + bounds.center_y
    col = (x_geo - bounds.min_x) / resolution + 1
    row = (bounds.max_y - y_geo) / resolution + 1
    col = max(0, min(col, w - 1))
    row = max(0, min(row, h - 1))

    if use_nearest:
        return _nearest_sample(Z_global, row, col)
    else:
        return _bilinear_sample(Z_global, row, col)


def _generate_side_triangles_indexed(
    Z,
    valid,
    global_offset,
    global_shape,
    z_bottom,
    is_global_boundary=(True, True, True, True),
):
    """Grid-indexed side generator - only creates walls at global boundaries."""
    h, w = valid.shape
    row_start, col_start = global_offset
    total_h, total_w = global_shape

    # Unpack which sides are global boundaries
    is_top_global, is_bottom_global, is_left_global, is_right_global = (
        is_global_boundary
    )

    vg_list = []
    t_list = []
    z_list = []
    v_offset = 0

    # helper for indexed wall
    def append_wall(r1, c1, r2, c2):
        nonlocal v_offset
        z1_val = Z[r1, c1]
        z2_val = Z[r2, c2]
        # 4 vertices per strip
        vg_list.append([[c1, r1], [c1, r1], [c2, r2], [c2, r2]])
        z_list.append([z1_val, z_bottom, z2_val, z_bottom])
        # Wall is a rectangle (x1,y1,z1), (x1,y1,zb), (x2,y2,z2), (x2,y2,zb)
        # Tri 1: 0 -> 1 -> 2
        # Tri 2: 2 -> 1 -> 3
        t_list.append(
            [
                [v_offset + 0, v_offset + 1, v_offset + 2],
                [v_offset + 2, v_offset + 1, v_offset + 3],
            ]
        )
        v_offset += 4

    # Top/Bottom walls - only at global boundaries
    for r in range(h):
        for direction in ("top", "bottom"):
            # Only process if this is a global boundary
            is_this_global_boundary = (direction == "top" and is_top_global) or (
                direction == "bottom" and is_bottom_global
            )
            if not is_this_global_boundary:
                continue

            is_boundary = (
                (row_start + r == 0)
                if direction == "top"
                else (row_start + r == total_h - 1)
            )
            check_row = r - 1 if direction == "top" else r + 1
            c = 0
            while c < w:
                if not valid[r, c]:
                    c += 1
                    continue
                # Only create wall at global boundary (edge of valid data)
                on_bound = is_boundary
                if not on_bound or c + 1 >= w or not valid[r, c + 1]:
                    c += 1
                    continue
                # check run
                c_start = c
                c += 1
                while c < w and valid[r, c]:
                    if not is_boundary:
                        break
                    c += 1
                c_end = min(c, w - 1)
                if c_start == c_end:
                    continue
                if direction == "top":
                    append_wall(r, c_start, r, c_end)
                else:
                    append_wall(r, c_end, r, c_start)

    # Left/Right walls - only at global boundaries
    for c in range(w):
        for direction in ("left", "right"):
            # Only process if this is a global boundary
            is_this_global_boundary = (direction == "left" and is_left_global) or (
                direction == "right" and is_right_global
            )
            if not is_this_global_boundary:
                continue

            is_boundary = (
                (col_start + c == 0)
                if direction == "left"
                else (col_start + c == total_w - 1)
            )
            check_col = c - 1 if direction == "left" else c + 1
            r = 0
            while r < h:
                if not valid[r, c]:
                    r += 1
                    continue
                # Only create wall at global boundary
                on_bound = is_boundary
                if not on_bound or r + 1 >= h or not valid[r + 1, c]:
                    r += 1
                    continue
                r_start = r
                r += 1
                while r < h and valid[r, c]:
                    if not is_boundary:
                        break
                    r += 1
                r_end = min(r, h - 1)
                if r_start == r_end:
                    continue
                if direction == "left":
                    append_wall(r_end, c, r_start, c)
                else:
                    append_wall(r_start, c, r_end, c)

    if not vg_list:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    return (
        np.concatenate(vg_list),
        np.concatenate(t_list),
        np.concatenate(z_list).astype(np.float32),
    )


def estimate_triangle_count(
    Z: np.ndarray, chunk_size: int = 1000, multiplier: float = 2.5
) -> int:
    """
    Estimate total output triangles for ratio calculation.
    Multiplier accounts for top + bottom surfaces + side walls.
    Reduced from 3.0 to 2.5 since RTIN with tolerance merges flat regions.
    """
    h, w = Z.shape
    total_valid = 0

    for r in range(0, h, chunk_size):
        chunk = Z[r : min(r + chunk_size, h)]
        total_valid += np.sum(~np.isnan(chunk))

    return int(total_valid * multiplier)


def create_native_stl(
    input_folder: str,
    output_file: str,
    resolution: float,
    w_cm: float,
    l_cm: float,
    h_cm: float,
    b_cm: float,
    error_threshold_m: float = 0.05,
    max_depth: int = 5,
    window_size: int = 512,
    overlap: float = 0.5,
):
    """Main entry point: streaming terrain-to-STL pipeline with low memory usage."""
    # Validate max_depth parameter - warn only, no enforcement
    if max_depth > 15:
        logger.warning(
            f"High max_depth ({max_depth}) may generate millions of triangles "
            f"and require significant memory. Consider using -d 10 or lower "
            f"for faster processing."
        )

    log_memory("Start")

    # Find input files
    tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    if not tif_files:
        tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tiff")))

    if not tif_files:
        logger.error(f"No .tif files found in '{input_folder}'")
        return

    config = MeshConfig(w_cm, l_cm, h_cm, b_cm, 0, resolution)

    logger.info(f"Found {len(tif_files)} tiles")
    logger.info(
        f"Output: {config.width_mm:.1f} x {config.length_mm:.1f} x {config.height_mm:.1f} mm"
    )

    temp_files = []  # Track temp files for cleanup

    try:
        # Phase 1: Scan bounds
        logger.info("Phase 1: Scanning bounds...")
        bounds = scan_tile_bounds(tif_files)
        logger.info(f"Extent: {bounds.width:.2f} x {bounds.height:.2f}")
        log_memory("After bounds scan")

        # Phase 2: Stitch tiles (memory-mapped)
        logger.info("Phase 2: Stitching tiles...")
        # Create both downsampled (for base mesh) and high-res (for error computation) rasters
        (
            Z_global,
            global_width,
            global_height,
            z_temp_path,
            Z_error,
            error_width,
            error_height,
            error_temp_path,
            error_resolution,
        ) = stitch_tiles_lowmem(tif_files, bounds, resolution)
        temp_files.append(z_temp_path)
        temp_files.append(error_temp_path)
        log_memory("After stitching")

        # Check valid data
        valid_count = 0
        for r in range(0, global_height, 1000):
            chunk = Z_global[r : min(r + 1000, global_height)]
            valid_count += np.sum(~np.isnan(chunk))

        if valid_count == 0:
            logger.error("No valid elevation data")
            return
        logger.info(f"Valid pixels: {valid_count:,}")

        # Phase 3: Normalize (in-place)
        logger.info("Phase 3: Normalizing elevation...")
        # Compute normalization parameters from main raster
        z_min, z_max = _compute_elevation_range(Z_global)
        z_range = z_max - z_min if (z_max - z_min) > 0 else 1.0
        logger.info(f"Elevation range: {z_min:.2f} to {z_max:.2f}")
        config.z_range_input = z_range

        # Apply same normalization to both rasters
        _apply_normalization(Z_global, z_min, z_range, config)
        _apply_normalization(Z_error, z_min, z_range, config)
        log_memory("After normalization")

        # Keep error threshold in meters for comparison
        # The raster is normalized to model space (0 to height_mm), so we convert threshold to mm
        error_threshold_mm = error_threshold_m * 1000  # Convert to mm

        # Validate error threshold - warn only, no enforcement
        if error_threshold_m < 0.01:  # Less than 1cm
            logger.warning(
                f"Very small error threshold ({error_threshold_m:.6f}m) may generate "
                f"millions of triangles and take significant time/memory. "
                f"Consider using -e 0.01 or larger for faster processing."
            )

        logger.info(
            f"Error threshold: {error_threshold_m:.6f}m ({error_threshold_mm:.4f} mm)"
        )

        # Phase 4: Add skirt (in-place) - DISABLED to ensure mesh boundary matches terrain
        # The skirt was causing walls to extend beyond the actual terrain due to RTIN simplification
        # logger.info("Phase 4: Adding skirt...")
        # add_skirt_inplace(Z_global)
        # log_memory("After skirt")

        # Phase 4.5: Repair (always fills small holes)
        repair_elevation_inplace(Z_global)

        # Calculate scale factor for XY coordinates
        scale_xy = min(config.width_mm / bounds.width, config.length_mm / bounds.height)

        # Phase 5: Streaming RTIN - Process in attention windows
        logger.info("Phase 5: Streaming RTIN mesh generation...")
        logger.info(
            f"  Error threshold: {error_threshold_m}m ({error_threshold_mm:.1f} mm)"
        )

        # Generate flat mesh with coarser base (refinement will add detail)
        base_triangle_pixels = max(64, int(error_threshold_m * 200))
        logger.info(f"  Base triangle size: {base_triangle_pixels} pixels")

        flat_vertices, flat_triangles = _generate_flat_mesh(
            bounds, scale_xy, resolution, base_triangle_pixels
        )

        from decimation import indexed_to_stl

        h_raster, w_raster = Z_global.shape

        def sample_raster(vx, vy):
            x_geo = vx / scale_xy + bounds.center_x
            y_geo = vy / scale_xy + bounds.center_y

            col = (x_geo - bounds.min_x) / resolution + 1
            row = (bounds.max_y - y_geo) / resolution + 1

            col = max(0, min(col, w_raster - 1))
            row = max(0, min(row, h_raster - 1))

            return _bilinear_sample(Z_global, row, col)

        def sample_raster_batch(vertices_xy, use_nearest=False):
            """Vectorized batch sampling of elevation for multiple vertices."""
            # Convert model coords to geographic
            x_geo = vertices_xy[:, 0] / scale_xy + bounds.center_x
            y_geo = vertices_xy[:, 1] / scale_xy + bounds.center_y

            # Convert to raster coords
            cols = (x_geo - bounds.min_x) / resolution + 1
            rows = (bounds.max_y - y_geo) / resolution + 1

            # Clamp to valid raster bounds
            cols = np.clip(cols, 0, w_raster - 1)
            rows = np.clip(rows, 0, h_raster - 1)

            # Sample using Numba-accelerated interpolation
            # Use nearest neighbor for high-res meshes to preserve sharp edges
            n = len(vertices_xy)
            z_values = np.empty(n, dtype=np.float32)

            if use_nearest:
                for i in range(n):
                    z_values[i] = _nearest_sample_njit(Z_global, rows[i], cols[i])
            else:
                for i in range(n):
                    z_values[i] = _bilinear_sample_njit(Z_global, rows[i], cols[i])

            # Handle NaN values
            z_values = np.where(np.isnan(z_values), 0.0, z_values)

            return z_values

        # Sample flat vertices using vectorized batch sampling
        # Use nearest neighbor for high-res meshes to preserve sharp edges
        use_nearest_sampling = resolution <= HIGH_RES_THRESHOLD
        if use_nearest_sampling:
            logger.info(
                "  Sampling initial elevation (nearest neighbor for sharp features)..."
            )
        else:
            logger.info("  Sampling initial elevation (bilinear for smooth terrain)...")

        flat_vertices_xy = flat_vertices[:, :2].astype(np.float32)
        z_values = sample_raster_batch(
            flat_vertices_xy, use_nearest=use_nearest_sampling
        )

        # Combine XY with Z to create vertex list
        vertices_array = np.column_stack([flat_vertices_xy, z_values])
        vertices_list = [tuple(v) for v in vertices_array]

        # Build triangle adjacency map: edge -> list of triangle indices
        # Each edge can be shared by up to 2 triangles in a watertight mesh
        logger.info("  Building adjacency map...")
        edge_to_tris = {}
        for tri_idx, tri in enumerate(flat_triangles):
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]])),
            ]
            for edge in edges:
                if edge not in edge_to_tris:
                    edge_to_tris[edge] = []
                edge_to_tris[edge].append(tri_idx)

        # Create adjacency: tri_idx -> {edge: neighbor_tri_idx}
        # For each triangle, store which edge connects to which neighbor
        n_tris = len(flat_triangles)
        tri_adjacency = [{} for _ in range(n_tris)]
        for edge, tri_list in edge_to_tris.items():
            if len(tri_list) == 2:
                t1, t2 = tri_list
                # Store edge->neighbor for both triangles
                # Edge is sorted pair, need to reconstruct original edge
                tri_adjacency[t1][edge] = t2
                tri_adjacency[t2][edge] = t1

        logger.info(f"  Adjacency map built: {len(edge_to_tris)} edges")

        # Generate windows
        windows = list(_sliding_windows((h_raster, w_raster), window_size, overlap))
        logger.info(
            f"  Processing {len(windows)} windows ({window_size}x{window_size}, overlap={overlap})"
        )
        logger.info(
            f"  Error threshold: {error_threshold_m}m ({error_threshold_mm:.1f} mm)"
        )

        # Pre-compute triangle centroids for window assignment
        triangles_np = flat_triangles.copy()

        # Convert vertices list to array for vectorized operations
        vertices_array = np.array(vertices_list, dtype=np.float32)

        # Vectorized centroid calculation
        v0s = vertices_array[triangles_np[:, 0]]
        v1s = vertices_array[triangles_np[:, 1]]
        v2s = vertices_array[triangles_np[:, 2]]

        # Compute centroids in model space
        centroids_xy = (v0s[:, :2] + v1s[:, :2] + v2s[:, :2]) / 3.0

        # Convert to geographic coordinates
        x_geo = centroids_xy[:, 0] / scale_xy + bounds.center_x
        y_geo = centroids_xy[:, 1] / scale_xy + bounds.center_y

        # Convert to raster coordinates
        tri_centroids = np.empty((len(triangles_np), 2), dtype=np.float32)
        tri_centroids[:, 1] = (x_geo - bounds.min_x) / resolution + 1.0
        tri_centroids[:, 0] = (bounds.max_y - y_geo) / resolution + 1.0

        # Buffer size in pixels for edge triangles
        buffer_pixels = int(window_size * 0.25)  # 25% buffer

        # Track which window owns each triangle for output
        tri_window_idx = np.full(len(triangles_np), -1, dtype=np.int32)
        # Track which triangles are in buffer zone (for refinement but not output)
        tri_in_buffer = np.zeros(len(triangles_np), dtype=np.bool_)

        for w_idx, window in enumerate(windows):
            row_start, row_end, col_start, col_end = window[:4]
            core_row_start, core_row_end, core_col_start, core_col_end = window[4:]

            # Buffer extends beyond core region
            buffer_row_start = max(0, core_row_start - buffer_pixels)
            buffer_row_end = min(h_raster, core_row_end + buffer_pixels)
            buffer_col_start = max(0, core_col_start - buffer_pixels)
            buffer_col_end = min(w_raster, core_col_end + buffer_pixels)

            in_buffer = (
                (tri_centroids[:, 0] >= buffer_row_start)
                & (tri_centroids[:, 0] < buffer_row_end)
                & (tri_centroids[:, 1] >= buffer_col_start)
                & (tri_centroids[:, 1] < buffer_col_end)
            )

            in_core = (
                (tri_centroids[:, 0] >= core_row_start)
                & (tri_centroids[:, 0] < core_row_end)
                & (tri_centroids[:, 1] >= core_col_start)
                & (tri_centroids[:, 1] < core_col_end)
            )

            # Mark buffer triangles
            tri_in_buffer |= in_buffer

            # Only assign output ownership to core triangles
            tri_window_idx[in_core] = w_idx

        # Two-pass approach: accumulate terrain, then add walls and bottom
        all_terrain_stl = []
        window_count = 0

        for w_idx, window in enumerate(
            tqdm(
                windows,
                desc="  Processing windows",
                unit="window",
                leave=False,
                position=0,
            )
        ):
            row_start, row_end, col_start, col_end = window[:4]
            core_row_start, core_row_end, core_col_start, core_col_end = window[4:]

            buffer_row_start = max(0, core_row_start - buffer_pixels)
            buffer_row_end = min(h_raster, core_row_end + buffer_pixels)
            buffer_col_start = max(0, core_col_start - buffer_pixels)
            buffer_col_end = min(w_raster, core_col_end + buffer_pixels)

            in_buffer = (
                (tri_centroids[:, 0] >= buffer_row_start)
                & (tri_centroids[:, 0] < buffer_row_end)
                & (tri_centroids[:, 1] >= buffer_col_start)
                & (tri_centroids[:, 1] < buffer_col_end)
            )

            in_core = (
                (tri_centroids[:, 0] >= core_row_start)
                & (tri_centroids[:, 0] < core_row_end)
                & (tri_centroids[:, 1] >= core_col_start)
                & (tri_centroids[:, 1] < core_col_end)
            )

            window_tri_indices = triangles_np[in_buffer].tolist()

            n_original_vertices = len(vertices_list)
            refined_vertices = vertices_list

            buffer_indices = np.where(in_buffer)[0]
            tri_id_map = {buffer_indices[i]: i for i in range(len(buffer_indices))}
            initial_triangles = [
                [t[0], t[1], t[2], tri_id_map.get(i, i)]
                for i, t in enumerate(window_tri_indices)
            ]

            local_edge_to_tris = {}
            for local_idx, tri in enumerate(initial_triangles):
                edges = [
                    tuple(sorted([tri[0], tri[1]])),
                    tuple(sorted([tri[1], tri[2]])),
                    tuple(sorted([tri[2], tri[0]])),
                ]
                for edge in edges:
                    if edge not in local_edge_to_tris:
                        local_edge_to_tris[edge] = []
                    local_edge_to_tris[edge].append(local_idx)

            # Create progress counter for triangle refinement
            tri_progress = [len(initial_triangles)]

            refined_triangles = _refine_triangles_batch(
                initial_triangles,
                refined_vertices,
                Z_global,
                bounds,
                scale_xy,
                resolution,
                error_threshold_mm,
                max_depth,
                local_edge_to_tris,
                progress_counter=tri_progress,
                Z_error=Z_error,
                error_resolution=error_resolution
                if "error_resolution" in locals()
                else None,
            )

            if not refined_triangles:
                continue

            refined_v = np.array(refined_vertices, dtype=np.float32)
            # Use int64 for triangle indices to support >2B triangles (though STL format limits to 4.2B)
            refined_t = np.array(refined_triangles, dtype=np.int64)

            # Vectorized centroid calculation - much faster than Python loop
            v0s = refined_v[refined_t[:, 0]]
            v1s = refined_v[refined_t[:, 1]]
            v2s = refined_v[refined_t[:, 2]]

            # Compute centroids in model space
            centroids_xy = (v0s[:, :2] + v1s[:, :2] + v2s[:, :2]) / 3.0

            # Convert to geographic coordinates
            x_geo = centroids_xy[:, 0] / scale_xy + bounds.center_x
            y_geo = centroids_xy[:, 1] / scale_xy + bounds.center_y

            # Convert to raster coordinates
            refined_centroids = np.empty((len(refined_t), 2), dtype=np.float32)
            refined_centroids[:, 1] = (x_geo - bounds.min_x) / resolution + 1.0
            refined_centroids[:, 0] = (bounds.max_y - y_geo) / resolution + 1.0

            in_core_refined = (
                (refined_centroids[:, 0] >= core_row_start)
                & (refined_centroids[:, 0] < core_row_end)
                & (refined_centroids[:, 1] >= core_col_start)
                & (refined_centroids[:, 1] < core_col_end)
            )

            if not np.any(in_core_refined):
                continue

            output_t = refined_t[in_core_refined]

            # Round vertices to prevent non-manifold edges from floating point errors
            refined_v = np.round(refined_v * 1000.0) / 1000.0

            terrain_stl = indexed_to_stl(refined_v, output_t)
            all_terrain_stl.append(terrain_stl)
            window_count += 1

            del terrain_stl, refined_v, refined_t, refined_centroids
            gc.collect()

        logger.info(f"  Generated terrain from {window_count} windows")

        # Phase 6: Combine terrain, add walls and base from actual boundary
        logger.info("Phase 6: Generating walls and base from terrain boundary...")
        z_bottom = -config.base_mm if config.base_mm > 0 else 0.0

        if all_terrain_stl:
            combined_terrain = np.concatenate(all_terrain_stl)
        else:
            combined_terrain = np.zeros(0, dtype=STL_DTYPE)

        del all_terrain_stl
        gc.collect()

        if len(combined_terrain) > 0:
            from decimation import add_skirt_to_mesh, stl_to_mesh

            logger.info(
                f"  Converting {len(combined_terrain):,} terrain triangles to indexed mesh..."
            )
            V_coords, triangles = stl_to_mesh(combined_terrain)
            del combined_terrain
            gc.collect()

            # Robust deduplication of vertices to prevent non-manifold edges
            # Round to 0.001mm precision
            V_coords = np.round(V_coords * 1000.0) / 1000.0
            unique_v, inv_idx = np.unique(V_coords, axis=0, return_inverse=True)
            triangles = inv_idx[triangles].astype(np.int32)

            # Remove degenerate triangles from deduplication
            t0, t1, t2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
            non_degen = (t0 != t1) & (t1 != t2) & (t0 != t2)
            triangles = triangles[non_degen]

            # Remove duplicate triangles
            sorted_tris = np.sort(triangles, axis=1)
            tri_view = sorted_tris.view(np.dtype([("v", "i4", (3,))]))
            _, unique_idx = np.unique(tri_view, return_index=True)
            triangles = triangles[unique_idx]

            V_coords = unique_v

            logger.info(
                f"  Indexed mesh: {len(V_coords):,} vertices, {len(triangles):,} triangles"
            )

            # === FIX 1: Coordinate snapping pass to eliminate T-junctions ===
            # Use tighter tolerance to merge nearly-identical vertices that
            # would otherwise cause internal edges to be misidentified as boundaries
            SNAP_TOLERANCE = 0.0001  # 0.0001mm = 0.1 micron
            V_coords_snapped = np.round(V_coords / SNAP_TOLERANCE) * SNAP_TOLERANCE
            unique_v, inv_idx = np.unique(V_coords_snapped, axis=0, return_inverse=True)
            triangles = inv_idx[triangles].astype(np.int32)
            V_coords = unique_v

            # Remove any degenerate triangles that may have resulted from snapping
            t0, t1, t2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
            non_degen = (t0 != t1) & (t1 != t2) & (t0 != t2)
            triangles = triangles[non_degen]

            logger.info(
                f"  After snapping: {len(V_coords):,} vertices, {len(triangles):,} triangles"
            )

            logger.info("  Adding walls and base...")

            # Step 1: Remove sloped boundary triangles from terrain
            # These are RTIN triangles at boundaries that slope from terrain to Z≈0
            # Identify triangles with any vertex near Z=0 (before elevation)
            z_min_vertex = V_coords[:, 2].min()
            z_threshold_low = z_min_vertex + 0.1  # Within 0.1mm of minimum Z

            # Calculate per-vertex mask (is vertex near bottom?)
            vertex_near_bottom = V_coords[:, 2] < z_threshold_low

            # Identify triangles with vertices near bottom
            tris_with_bottom_vertex = (
                vertex_near_bottom[triangles[:, 0]]
                | vertex_near_bottom[triangles[:, 1]]
                | vertex_near_bottom[triangles[:, 2]]
            )

            # Also check if triangle has large Z range (sloped)
            tri_z_min = np.minimum(
                np.minimum(V_coords[triangles[:, 0], 2], V_coords[triangles[:, 1], 2]),
                V_coords[triangles[:, 2], 2],
            )
            tri_z_max = np.maximum(
                np.maximum(V_coords[triangles[:, 0], 2], V_coords[triangles[:, 1], 2]),
                V_coords[triangles[:, 2], 2],
            )
            tri_z_range = tri_z_max - tri_z_min

            # Mark sloped boundary triangles for removal
            # These have: (1) vertex near Z=0 AND (2) large Z range
            is_sloped_boundary = tris_with_bottom_vertex & (tri_z_range > 1.0)

            logger.info(
                f"    Removing {np.sum(is_sloped_boundary)} sloped boundary triangles"
            )

            # Keep only non-sloped triangles
            keep_mask = ~is_sloped_boundary
            triangles = triangles[keep_mask]

            # Remove unreferenced vertices
            referenced = np.unique(triangles)
            old_to_new = np.full(len(V_coords), -1, dtype=np.int32)
            old_to_new[referenced] = np.arange(len(referenced))
            V_coords = V_coords[referenced]
            triangles = old_to_new[triangles]

            logger.info(
                f"    After filtering: {len(V_coords)} vertices, {len(triangles)} triangles"
            )

            # Step 2: Extract boundary edges from filtered terrain
            from collections import defaultdict

            edge_to_tris = defaultdict(list)

            for tri_idx, (v0, v1, v2) in enumerate(triangles):
                edges = [
                    (min(v0, v1), max(v0, v1)),
                    (min(v1, v2), max(v1, v2)),
                    (min(v2, v0), max(v2, v0)),
                ]
                for edge in edges:
                    edge_to_tris[edge].append(tri_idx)

            boundary_edges = [
                edge for edge, tris in edge_to_tris.items() if len(tris) == 1
            ]

            logger.info(f"    Found {len(boundary_edges)} boundary edges")

            num_verts = len(V_coords)

            # Elevate terrain vertices by base_mm
            V_coords_elevated = V_coords.copy()
            V_coords_elevated[:, 2] += config.base_mm

            # Create walls by extruding boundary edges downward
            wall_vertices = []
            wall_triangles = []

            for v0_idx, v1_idx in boundary_edges:
                # Get the two vertices of this boundary edge
                v0_top = V_coords_elevated[v0_idx]
                v1_top = V_coords_elevated[v1_idx]

                # Create bottom vertices (same X,Y, at z_bottom)
                v0_bottom = np.array([v0_top[0], v0_top[1], z_bottom], dtype=np.float32)
                v1_bottom = np.array([v1_top[0], v1_top[1], z_bottom], dtype=np.float32)

                # Add vertices
                base_idx = len(wall_vertices)
                wall_vertices.extend([v0_top, v1_top, v0_bottom, v1_bottom])

                # Create two triangles for this wall segment
                # Triangle 1: v0_top, v1_top, v0_bottom
                # Triangle 2: v1_top, v1_bottom, v0_bottom
                wall_triangles.append([base_idx + 0, base_idx + 1, base_idx + 2])
                wall_triangles.append([base_idx + 1, base_idx + 3, base_idx + 2])

            wall_vertices = np.array(wall_vertices, dtype=np.float32)
            wall_triangles = np.array(wall_triangles, dtype=np.int64)

            # Offset wall triangle indices to account for terrain vertices
            wall_triangles += num_verts

            logger.info(f"    Created {len(wall_triangles)} wall triangles")

            # Create base (flat rectangle at z_bottom) matching wall footprint
            # Use the inner bounds (minimum coverage) to ensure base doesn't extend beyond walls
            if len(wall_vertices) > 0:
                wall_bottom_v0 = wall_vertices[2::4]  # v0_bottom vertices
                wall_bottom_v1 = wall_vertices[3::4]  # v1_bottom vertices
                all_bottom = np.vstack([wall_bottom_v0, wall_bottom_v1])

                # Find vertices on each edge with tolerance
                tol = 0.1
                x_min_global, x_max_global = (
                    all_bottom[:, 0].min(),
                    all_bottom[:, 0].max(),
                )
                y_min_global, y_max_global = (
                    all_bottom[:, 1].min(),
                    all_bottom[:, 1].max(),
                )

                left = all_bottom[np.abs(all_bottom[:, 0] - x_min_global) < tol]
                right = all_bottom[np.abs(all_bottom[:, 0] - x_max_global) < tol]
                bottom = all_bottom[np.abs(all_bottom[:, 1] - y_min_global) < tol]
                top = all_bottom[np.abs(all_bottom[:, 1] - y_max_global) < tol]

                # Use INNER bounds - max of mins and min of maxs
                # This ensures the base fits within the walls, not extends beyond
                x_min = left[:, 0].max() if len(left) > 0 else x_min_global
                x_max = right[:, 0].min() if len(right) > 0 else x_max_global
                y_min = bottom[:, 1].max() if len(bottom) > 0 else y_min_global
                y_max = top[:, 1].min() if len(top) > 0 else y_max_global
            else:
                x_min, x_max = V_coords[:, 0].min(), V_coords[:, 0].max()
                y_min, y_max = V_coords[:, 1].min(), V_coords[:, 1].max()

            logger.info(
                f"    Base bounds: X={x_min:.2f} to {x_max:.2f}, Y={y_min:.2f} to {y_max:.2f}"
            )

            base_corners = np.array(
                [
                    [x_min, y_min, z_bottom],
                    [x_max, y_min, z_bottom],
                    [x_max, y_max, z_bottom],
                    [x_min, y_max, z_bottom],
                ],
                dtype=np.float32,
            )

            base_start = num_verts + len(wall_vertices)
            base_triangles = np.array(
                [
                    [base_start + 0, base_start + 1, base_start + 2],
                    [base_start + 0, base_start + 2, base_start + 3],
                ],
                dtype=np.int64,
            )

            # Combine all
            # Combine all vertices and triangles
            final_v = np.vstack([V_coords_elevated, wall_vertices, base_corners])
            final_t = np.vstack([triangles, wall_triangles, base_triangles])

            logger.info(f"    Elevated terrain by {config.base_mm}mm")
            logger.info(
                f"    Created {len(wall_triangles)} wall triangles, {len(base_triangles)} base triangles"
            )
            logger.info(f"    Total: {len(final_v)} vertices, {len(final_t)} triangles")

            logger.info(
                f"  Final mesh: {len(final_v):,} vertices, {len(final_t):,} triangles"
            )

            final_stl = indexed_to_stl(final_v, final_t)

            del V_coords, triangles, final_v, final_t
            gc.collect()
        else:
            logger.warning("  No terrain triangles generated")
            final_stl = np.zeros(0, dtype=STL_DTYPE)

        # Write final mesh to file
        writer = StreamingSTLWriter(output_file)
        writer.write_triangles(final_stl)
        final_count = writer.get_count()
        writer.close()

        os.replace(output_file + ".tmp", output_file)

        del Z_global
        gc.collect()
        log_memory("After write")
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"Output: {output_file} ({file_size_mb:.2f} MB)")
        logger.info("Complete!")
        return

    finally:
        # Cleanup temp files
        for temp_path in temp_files:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Convert GeoTIFF terrain to STL (low memory)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_folder", help="Folder containing .tif files")
    parser.add_argument("output_file", help="Output STL path")
    parser.add_argument(
        "-r", "--resolution", type=float, default=1.0, help="Sampling resolution"
    )
    parser.add_argument("-W", "--width", type=float, default=10.0, help="Width (cm)")
    parser.add_argument("-L", "--length", type=float, default=10.0, help="Length (cm)")
    parser.add_argument("-H", "--height", type=float, default=3.0, help="Height (cm)")
    parser.add_argument(
        "-B", "--base", type=float, default=0.0, help="Base (cm, 0 = no base)"
    )
    parser.add_argument(
        "-e",
        "--error-threshold",
        type=float,
        default=0.05,
        help="QEM error threshold (meters)",
    )
    parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        default=5,
        help="Maximum refinement depth (higher = more detail, slower)",
    )
    parser.add_argument(
        "-w",
        "--window-size",
        type=int,
        default=512,
        help="Attention window size for streaming RTIN",
    )
    parser.add_argument(
        "-o",
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap ratio between windows (0.0-1.0)",
    )

    args = parser.parse_args()

    create_native_stl(
        input_folder=args.input_folder,
        output_file=args.output_file,
        resolution=args.resolution,
        w_cm=args.width,
        l_cm=args.length,
        h_cm=args.height,
        b_cm=args.base,
        error_threshold_m=args.error_threshold,
        max_depth=args.max_depth,
        window_size=args.window_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
