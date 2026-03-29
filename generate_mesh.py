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
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

STL_DTYPE = np.dtype([
    ('normals', '<f4', (3,)),
    ('v1', '<f4', (3,)),
    ('v2', '<f4', (3,)),
    ('v3', '<f4', (3,)),
    ('attr', '<u2')
])

STL_TRIANGLE_SIZE = STL_DTYPE.itemsize  # 50 bytes per triangle


# =============================================================================
# STL Streaming Writer
# =============================================================================


class StreamingSTLWriter:
    """Write STL triangles directly to disk without holding in memory."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.temp_path = filepath + '.tmp'
        self.triangle_count = 0
        self.file: Optional[BinaryIO] = None

    def __enter__(self):
        self.file = open(self.temp_path, 'wb')
        # Write placeholder header (will update later)
        self.file.write(b"Binary STL - Streaming Writer".ljust(80, b'\x00'))
        self.file.write(struct.pack('<I', 0))  # Placeholder count
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

        if exc_type is None:
            # Update triangle count in header
            with open(self.temp_path, 'r+b') as f:
                f.seek(80)
                f.write(struct.pack('<I', self.triangle_count))

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

        data = np.memmap(self.temp_path, dtype=STL_DTYPE, mode='r+',
                         offset=84, shape=(self.triangle_count,))

        for start in range(0, self.triangle_count, chunk_size):
            end = min(start + chunk_size, self.triangle_count)
            chunk = data[start:end]

            edge1 = chunk['v2'] - chunk['v1']
            edge2 = chunk['v3'] - chunk['v1']
            normals = np.cross(edge1, edge2)
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            lengths[lengths == 0] = 1.0
            chunk['normals'] = normals / lengths

        data.flush()
        del data

    def write_triangles(self, triangles: np.ndarray):
        """Write triangles to file."""
        if len(triangles) == 0:
            return
        if self.file is not None:
            self.file.write(triangles.tobytes())
        self.triangle_count += len(triangles)

    def get_count(self) -> int:
        return self.triangle_count


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
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for fp in tqdm(tif_files, desc="Scanning bounds", unit="file"):
        try:
            with rasterio.open(fp) as src:
                b = src.bounds
                min_x, min_y = min(min_x, b.left), min(min_y, b.bottom)
                max_x, max_y = max(max_x, b.right), max(max_y, b.top)
        except Exception as e:
            logger.warning(f"Could not read {fp}: {e}")

    if min_x == float('inf'):
        raise ValueError("No valid tiles found")

    return GeoBounds(min_x, min_y, max_x, max_y)


def stitch_tiles_lowmem(tif_files: list, bounds: GeoBounds, resolution: float) -> tuple:
    """
    Stitch tiles using memory-mapped array.
    Returns (Z_memmap, width, height, temp_path)
    """
    global_width = int(np.ceil(bounds.width / resolution)) + 2
    global_height = int(np.ceil(bounds.height / resolution)) + 2

    logger.info(f"Canvas: {global_width} x {global_height} ({global_width * global_height * 4 / 1e6:.1f} MB on disk)")

    # Create memory-mapped file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.elevation.mmap')
    temp_path = temp_file.name
    temp_file.close()

    Z_global = np.memmap(temp_path, dtype=np.float32, mode='w+',
                         shape=(global_height, global_width))
    Z_global[:] = np.nan

    # Process tiles one at a time
    for fp in tqdm(tif_files, desc="Stitching tiles", unit="file"):
        try:
            with rasterio.open(fp) as src:
                tile_w = int(np.ceil((src.bounds.right - src.bounds.left) / resolution))
                tile_h = int(np.ceil((src.bounds.top - src.bounds.bottom) / resolution))

                if tile_w <= 0 or tile_h <= 0:
                    continue

                # Read tile data
                data = src.read(1, out_shape=(tile_h, tile_w), resampling=Resampling.bilinear)

                col_start = int(np.round((src.bounds.left - bounds.min_x) / resolution)) + 1
                row_start = int(np.round((bounds.max_y - src.bounds.top) / resolution)) + 1

                row_end = min(row_start + tile_h, global_height)
                col_end = min(col_start + tile_w, global_width)

                actual_h = row_end - row_start
                actual_w = col_end - col_start

                if actual_h <= 0 or actual_w <= 0:
                    continue

                data_slice = data[:actual_h, :actual_w]
                valid_mask = data_slice > -100

                Z_global[row_start:row_end, col_start:col_end][valid_mask] = data_slice[valid_mask]

                # Flush to disk periodically
                Z_global.flush()

                del data, data_slice, valid_mask

        except Exception as e:
            logger.warning(f"Error processing {fp}: {e}")

    gc.collect()

    return Z_global, global_width, global_height, temp_path


def normalize_elevation_inplace(Z: np.ndarray, config: MeshConfig):
    """Normalize elevation in-place to minimize memory."""
    # Find min/max in chunks to avoid loading all into RAM
    z_min, z_max = np.inf, -np.inf
    chunk_rows = 1000

    for r in range(0, Z.shape[0], chunk_rows):
        chunk = Z[r:r + chunk_rows]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 0:
            z_min = min(z_min, valid.min())
            z_max = max(z_max, valid.max())

    z_range = z_max - z_min if (z_max - z_min) > 0 else 1.0
    logger.info(f"Elevation range: {z_min:.2f} to {z_max:.2f}")

    config.z_range_input = z_range

    # Normalize in chunks
    for r in range(0, Z.shape[0], chunk_rows):
        chunk = Z[r:r + chunk_rows]
        valid = ~np.isnan(chunk)
        chunk[valid] = ((chunk[valid] - z_min) / z_range) * (config.height_mm - config.base_mm) + config.base_mm

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
                neighbors = [row[c-1], row[c+1], prev_row[c], next_row[c]]
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
                block = Z_old[dr:dr + h - 2, dc:dc + w - 2]
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

    if hasattr(Z, 'flush'):
        Z.flush()


def add_skirt_inplace(Z: np.ndarray):
    """Add skirt in-place using row-by-row processing."""
    h, w = Z.shape

    # Process row by row to minimize memory
    for r in range(h):
        row = Z[r]
        valid = ~np.isnan(row)

        skirt = np.zeros(w, dtype=bool)

        # Left neighbor
        if r > 0:
            skirt |= ~np.isnan(Z[r - 1])
        # Right neighbor
        if r < h - 1:
            skirt |= ~np.isnan(Z[r + 1])
        # Up neighbor
        skirt[:-1] |= ~np.isnan(row[1:])
        # Down neighbor
        skirt[1:] |= ~np.isnan(row[:-1])

        # Apply skirt where not already valid
        skirt &= ~valid
        row[skirt] = 0.0

    if hasattr(Z, 'flush'):
        Z.flush()


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
        coords[k]     = ax
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
def _rtin_compute_errors(coords, num_triangles, num_parent_triangles,
                         terrain_flat, grid_size, errors, h, w):
    """Compute RTIN approximation errors (Numba JIT, bottom-up)."""
    # Force full resolution on the chunk boundaries and padding
    for r in range(grid_size):
        for c in range(grid_size):
            if r >= h or c >= w or r == 0 or r == h - 1 or c == 0 or c == w - 1:
                errors[r * grid_size + c] = 1e30 # infinity representation

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
            left_idx  = ((ay + cy) >> 1) * grid_size + ((ax + cx) >> 1)
            right_idx = ((by + cy) >> 1) * grid_size + ((bx + cx) >> 1)
            e_l = errors[left_idx]
            e_r = errors[right_idx]
            if e_l > errors[mid_idx]: errors[mid_idx] = e_l
            if e_r > errors[mid_idx]: errors[mid_idx] = e_r

    return errors


@njit(cache=True, boundscheck=True)
def _rtin_count(ax, ay, bx, by, cx, cy,
                errors, grid_size, max_error, indices, counts):
    """Pass 1: Count vertices/triangles and index them."""
    mx = (ax + bx) >> 1
    my = (ay + by) >> 1

    if (abs(ax - cx) + abs(ay - cy) > 1 and
            errors[my * grid_size + mx] > max_error):
        _rtin_count(cx, cy, ax, ay, mx, my, errors, grid_size, max_error, indices, counts)
        _rtin_count(bx, by, cx, cy, mx, my, errors, grid_size, max_error, indices, counts)
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
def _rtin_extract(ax, ay, bx, by, cx, cy,
                  errors, grid_size, max_error, indices, vertices, triangles, counts):
    """Pass 2: Extract distinct vertices and triangles."""
    mx = (ax + bx) >> 1
    my = (ay + by) >> 1

    if (abs(ax - cx) + abs(ay - cy) > 1 and
            errors[my * grid_size + mx] > max_error):
        _rtin_extract(cx, cy, ax, ay, mx, my, errors, grid_size, max_error, indices, vertices, triangles, counts)
        _rtin_extract(bx, by, cx, cy, mx, my, errors, grid_size, max_error, indices, vertices, triangles, counts)
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



def _merge_and_deduplicate_indexed_meshes(parts, X, Y, Z, h, w):
    """
    Merge multiple index-mesh parts (e.g. top, bottom, sides) and deduplicate by coordinates.
    """
    if not parts:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    
    total_verts = sum(len(p[0]) for p in parts)
    total_tris = sum(len(p[1]) for p in parts)
    
    full_coords = np.empty((total_verts, 3), dtype=np.float32)
    temp_t = np.empty((total_tris, 3), dtype=np.int32)
    
    v_offset = 0
    t_offset = 0
    for grid_v, t, *opt_z in parts:
        nv = len(grid_v)
        nt = len(t)
        if nt == 0: continue
        
        gv_y, gv_x = grid_v[:, 1], grid_v[:, 0]
        # Clipping for padding vertices
        gy_c = np.minimum(gv_y, h - 1)
        gx_c = np.minimum(gv_x, w - 1)
        
        full_coords[v_offset:v_offset+nv, 0] = X[gy_c, gx_c]
        full_coords[v_offset:v_offset+nv, 1] = Y[gy_c, gx_c]
        if opt_z:
            full_coords[v_offset:v_offset+nv, 2] = opt_z[0]
        else:
            full_coords[v_offset:v_offset+nv, 2] = Z[gy_c, gx_c]
            
        temp_t[t_offset:t_offset+nt] = t + v_offset
        v_offset += nv
        t_offset += nt
    
    # Final robust coordinate-based deduplication
    unique_v, inv_idx = np.unique(full_coords[:v_offset], axis=0, return_inverse=True)
    out_t = inv_idx[temp_t[:t_offset]].astype(np.int32)
    
    # Fix 2: Remove degenerate triangles created by coordinate deduplication
    t0, t1, t2 = out_t[:, 0], out_t[:, 1], out_t[:, 2]
    non_degenerate = (t0 != t1) & (t1 != t2) & (t0 != t2)
    out_t = out_t[non_degenerate]
    
    return unique_v, out_t


def generate_chunk_triangles(Z: np.ndarray, X: np.ndarray, Y: np.ndarray,
                               valid: np.ndarray,
                               global_offset: tuple,
                               global_shape: tuple,
                               planar_tol: float = 0.0,
                               z_bottom: float = 0.0,
                               is_global_boundary: tuple = (True, True, True, True)):
    """Generate all triangles for a chunk (top, bottom, sides) as indexed mesh.

    Args:
        is_global_boundary: (top, bottom, left, right) flags indicating
            whether this chunk touches the outermost global boundary.
            Only global-boundary vertices will be marked for locking.
    """
    parts = []
    h, w = Z.shape
    quad_valid = (valid[:-1, :-1] & valid[:-1, 1:] &
                  valid[1:, :-1] & valid[1:, 1:])

    # Top surface (RTIN adaptive directly yields indexed)
    top_grid, top_t = _generate_top_triangles_rtin(X, Y, Z, valid, planar_tol, return_grid=True)
    if len(top_t) > 0:
        parts.append((top_grid, top_t))

    # Bottom surface (grid-indexed)
    bot_grid, bot_t = _generate_bottom_triangles_indexed(quad_valid)
    if len(bot_t) > 0:
        parts.append((bot_grid, bot_t, np.full(len(bot_grid), z_bottom, dtype=np.float32)))

    # Side walls (grid-indexed)
    side_grid, side_t, side_z = _generate_side_triangles_indexed(Z, valid, global_offset, global_shape, z_bottom)
    if len(side_t) > 0:
        parts.append((side_grid, side_t, side_z))

    return _merge_and_deduplicate_indexed_meshes(parts, X, Y, Z, h, w)


def _generate_top_triangles_rtin(X, Y, Z, valid, planar_tol=0.0, return_grid=False):
    """
    Generate top surface triangles using a 2-pass RTIN (Martini) algorithm.
    """
    h, w = Z.shape
    grid_size = _next_power_of_two_plus_one(max(h, w))
    
    valid_pad = np.zeros((grid_size, grid_size), dtype=bool)
    valid_pad[:h, :w] = valid
    
    z_pad = np.zeros((grid_size, grid_size), dtype=np.float32)
    z_pad[:h, :w] = np.where(np.isnan(Z), 0.0, Z)

    # Correct Numba-accelerated error computation using precomputed Martini data
    coords, num_tri, num_parent = _get_martini(grid_size)
    errors = np.zeros(grid_size * grid_size, dtype=np.float32)
    _rtin_compute_errors(coords, num_tri, num_parent, z_pad.ravel(), grid_size, errors, h, w)

    # Pass 1: Count vertices/triangles and index them
    indices = np.zeros(grid_size * grid_size, dtype=np.int32)
    counts = np.zeros(2, dtype=np.int32)  # [num_vertices, num_triangles]

    max_coord = grid_size - 1
    # Entry CCW
    _rtin_count(0, 0, max_coord, 0, max_coord, max_coord,
                errors, grid_size, planar_tol, indices, counts)
    _rtin_count(max_coord, max_coord, 0, max_coord, 0, 0,
                errors, grid_size, planar_tol, indices, counts)

    num_verts = counts[0]
    num_tris = counts[1]

    if num_tris == 0:
        if return_grid:
            return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.int32)
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    vertices_grid = np.empty((num_verts, 2), dtype=np.int32)
    triangles = np.empty((num_tris, 3), dtype=np.int32)

    counts[1] = 0
    _rtin_extract(0, 0, max_coord, 0, max_coord, max_coord,
                  errors, grid_size, planar_tol, indices, vertices_grid, triangles, counts)
    _rtin_extract(max_coord, max_coord, 0, max_coord, 0, 0,
                  errors, grid_size, planar_tol, indices, vertices_grid, triangles, counts)

    # Fix 1: Filter degenerate triangles (RTIN can emit zero-area tris
    # when leaf vertices share grid positions)
    t0, t1, t2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    non_degenerate = (t0 != t1) & (t1 != t2) & (t0 != t2)
    triangles = triangles[non_degenerate]

    if len(triangles) == 0:
        if return_grid:
            return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.int32)
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    # Filter invalid triangles (those that touch padding or invalid pixels)
    ax, ay = vertices_grid[triangles[:, 0], 0], vertices_grid[triangles[:, 0], 1]
    bx, by = vertices_grid[triangles[:, 1], 0], vertices_grid[triangles[:, 1], 1]
    cx, cy = vertices_grid[triangles[:, 2], 0], vertices_grid[triangles[:, 2], 1]
    
    # Strictly inside real data bounds AND valid
    mask = (ax < w) & (ay < h) & (bx < w) & (by < h) & (cx < w) & (cy < h)
    valid_indices = valid[ay[mask], ax[mask]] & valid[by[mask], bx[mask]] & valid[cy[mask], cx[mask]]
    triangles = triangles[mask][valid_indices]

    if len(triangles) == 0:
        if return_grid:
            return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.int32)
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    if return_grid:
        return vertices_grid, triangles

    # Map to coordinates
    vy, vx = vertices_grid[:, 1], vertices_grid[:, 0]
    vy_c, vx_c = np.minimum(vy, h-1), np.minimum(vx, w-1)
    V_coords = np.column_stack([X[vy_c, vx_c], Y[vy_c, vx_c], Z[vy_c, vx_c]])
    return V_coords, triangles

def _generate_dense_top_indexed(valid):
    """Indexed dense top generator."""
    h, w = valid.shape
    if h < 2 or w < 2:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.int32)
    quad_valid = (valid[:-1, :-1] & valid[:-1, 1:] &
                  valid[1:, :-1] & valid[1:, 1:])
    ri, ci = np.where(quad_valid)
    n = len(ri)
    if n == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.int32)
    
    # 4 vertices per quad normally, but we deduplicate later. 
    # For now build naive indices.
    vg = np.empty((n * 4, 2), dtype=np.int32)
    vg[0::4, 0] = ci;     vg[0::4, 1] = ri
    vg[1::4, 0] = ci+1;   vg[1::4, 1] = ri
    vg[2::4, 0] = ci;     vg[2::4, 1] = ri+1
    vg[3::4, 0] = ci+1;   vg[3::4, 1] = ri+1
    
    t = np.empty((n * 2, 3), dtype=np.int32)
    offsets = np.arange(0, n*4, 4, dtype=np.int32)
    # Tri 1: (c,r) -> (c+1,r) -> (c+1,r+1) CCW
    t[0::2, 0] = offsets      # (c,r)
    t[0::2, 1] = offsets + 1  # (c+1,r)
    t[0::2, 2] = offsets + 3  # (c+1,r+1)
    # Tri 2: (c,r) -> (c+1,r+1) -> (c,r+1) CCW
    t[1::2, 0] = offsets      # (c,r)
    t[1::2, 1] = offsets + 3  # (c+1,r+1)
    t[1::2, 2] = offsets + 2  # (c,r+1)
    
    return vg, t


def _generate_bottom_triangles_indexed(quad_valid):
    """Grid-indexed bottom generator using strip-merging logic."""
    h_q, w_q = quad_valid.shape
    if h_q == 0 or w_q == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.int32)
    
    strips = []
    for r in range(h_q):
        row = quad_valid[r]
        c = 0
        while c < w_q:
            if not row[c]: c += 1; continue
            c_start = c
            while c < w_q and row[c]: c += 1
            strips.append((r, c_start, c))
    if not strips:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.int32)
    
    sa = np.array(strips, dtype=np.int32)
    rs, css, ces = sa[:, 0], sa[:, 1], sa[:, 2]
    n = len(sa)
    
    vg = np.empty((n * 4, 2), dtype=np.int32)
    vg[0::4, 0] = css;   vg[0::4, 1] = rs
    vg[1::4, 0] = ces;   vg[1::4, 1] = rs
    vg[2::4, 0] = css;   vg[2::4, 1] = rs+1
    vg[3::4, 0] = ces;   vg[3::4, 1] = rs+1
    
    t = np.empty((n * 2, 3), dtype=np.int32)
    offsets = np.arange(0, n*4, 4, dtype=np.int32)
    # View from below CCW: (ces,rs) -> (css,rs) -> (css,rs+1)
    t[0::2, 0] = offsets + 1 # (ces,rs)
    t[0::2, 1] = offsets     # (css,rs)
    t[0::2, 2] = offsets + 2 # (css,rs+1)
    # (ces,rs) -> (css,rs+1) -> (ces,rs+1)
    t[1::2, 0] = offsets + 1 # (ces,rs)
    t[1::2, 1] = offsets + 2 # (css,rs+1)
    t[1::2, 2] = offsets + 3 # (ces,rs+1)
    
    return vg, t


def _generate_side_triangles_indexed(Z, valid, global_offset, global_shape, z_bottom):
    """Grid-indexed side generator."""
    h, w = valid.shape
    row_start, col_start = global_offset
    total_h, total_w = global_shape
    
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
        t_list.append([[v_offset+0, v_offset+1, v_offset+2], 
                       [v_offset+2, v_offset+1, v_offset+3]])
        v_offset += 4

    # Top/Bottom walls
    for r in range(h):
        for direction in ('top', 'bottom'):
            is_boundary = (row_start + r == 0) if direction == 'top' else (row_start + r == total_h - 1)
            check_row = r - 1 if direction == 'top' else r + 1
            c = 0
            while c < w:
                if not valid[r, c]: c += 1; continue
                on_bound = is_boundary or (0 <= check_row < h and not valid[check_row, c])
                if not on_bound or c + 1 >= w or not valid[r, c+1]: c += 1; continue
                # check run
                c_start = c
                c += 1
                while c < w and valid[r, c]:
                    if not (is_boundary or (0 <= check_row < h and not valid[check_row, c])): break
                    c += 1
                c_end = min(c, w - 1)
                # Fix 3: Skip degenerate single-point wall segments
                if c_start == c_end:
                    continue
                # orientation: we need the top edge of the wall to be reverse of the top surface edge
                if direction == 'top':
                    append_wall(r, c_start, r, c_end) 
                else: 
                    append_wall(r, c_end, r, c_start)

    # Left/Right walls
    for c in range(w):
        for direction in ('left', 'right'):
            is_boundary = (col_start + c == 0) if direction == 'left' else (col_start + c == total_w - 1)
            check_col = c - 1 if direction == 'left' else c + 1
            r = 0
            while r < h:
                if not valid[r, c]: r += 1; continue
                on_bound = is_boundary or (0 <= check_col < w and not valid[r, check_col])
                if not on_bound or r + 1 >= h or not valid[r+1, c]: r += 1; continue
                r_start = r
                r += 1
                while r < h and valid[r, c]:
                    if not (is_boundary or (0 <= check_col < w and not valid[r, check_col])): break
                    r += 1
                r_end = min(r, h - 1)
                # Fix 3: Skip degenerate single-point wall segments
                if r_start == r_end:
                    continue
                if direction == 'left':
                    append_wall(r_end, c, r_start, c)
                else:
                    append_wall(r_start, c, r_end, c)

    if not vg_list:
        return (np.empty((0, 2), dtype=np.int32), 
                np.empty((0, 3), dtype=np.int32), 
                np.empty(0, dtype=np.float32))

    return (np.concatenate(vg_list), 
            np.concatenate(t_list), 
            np.concatenate(z_list).astype(np.float32))






def _stream_spatial_chunks(Z: np.ndarray, bounds: GeoBounds, config: MeshConfig,
                            resolution: float, chunk_size: int):
    """
    Generator: yield successive (V_coords, triangles, pbar) from the elevation grid.
    Extracted so both streaming write paths can share the same spatial loop.
    """
    h, w = Z.shape
    scale_xy = min(config.width_mm / bounds.width, config.length_mm / bounds.height)

    total_chunks = ((h - 1) // chunk_size + 1) * ((w - 1) // chunk_size + 1)
    pbar = tqdm(total=total_chunks, unit="chunk")

    for row_start in range(0, h - 1, chunk_size):
        for col_start in range(0, w - 1, chunk_size):
            row_end = min(row_start + chunk_size + 1, h)
            col_end = min(col_start + chunk_size + 1, w)

            z_chunk = np.array(Z[row_start:row_end, col_start:col_end])
            chunk_h, chunk_w = z_chunk.shape

            if chunk_h < 2 or chunk_w < 2:
                pbar.update(1)
                continue

            valid_chunk = ~np.isnan(z_chunk)
            if not np.any(valid_chunk):
                pbar.update(1)
                continue

            cols = np.arange(col_start, col_end)
            rows = np.arange(row_start, row_end)
            cc, rr = np.meshgrid(cols, rows)

            x_chunk = ((bounds.min_x + (cc - 1) * resolution) - bounds.center_x) * scale_xy
            y_chunk = ((bounds.max_y - (rr - 1) * resolution) - bounds.center_y) * scale_xy
            z_bottom = config.base_mm

            # Determine which sides of this chunk are global boundaries
            is_global_boundary = (
                row_start == 0,           # top
                row_end >= h,             # bottom
                col_start == 0,           # left
                col_end >= w,             # right
            )

            v_coords, triangles = generate_chunk_triangles(
                z_chunk, x_chunk, y_chunk, valid_chunk,
                (row_start, col_start), (h, w), config.planar_tol, z_bottom,
                is_global_boundary)

            del z_chunk, x_chunk, y_chunk, valid_chunk

            if len(triangles) > 0:
                yield v_coords, triangles, pbar
            else:
                pbar.update(1)

    pbar.close()
    gc.collect()


def stream_generate_to_writer(Z: np.ndarray, bounds: GeoBounds, config: MeshConfig,
                              resolution: float, writer: StreamingSTLWriter,
                              chunk_size: int = 256) -> int:
    """
    Generate mesh triangles and write directly to output — no intermediate buffer.
    Returns total number of triangles written.
    """
    pbar = None
    from decimation import indexed_to_stl
    try:
        for V_coords, triangles, pbar in _stream_spatial_chunks(Z, bounds, config, resolution, chunk_size):
            pbar.set_description("Generating mesh")
            triangles_stl = indexed_to_stl(V_coords, triangles)
            writer.write_triangles(triangles_stl)
            del triangles, V_coords, triangles_stl
            pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        gc.collect()
    return writer.get_count()


def stream_generate_and_decimate(
        Z: np.ndarray, bounds: GeoBounds, config: MeshConfig,
        resolution: float, chunk_ratio: float, writer: StreamingSTLWriter,
        chunk_size: int = 256) -> int:
    """
    Generate + decimate each chunk inline and write directly to output.
    No intermediate disk buffer needed — RAM footprint is one chunk at a time.

    chunk_ratio is the per-chunk face reduction target (e.g. 0.05 = keep 5%).
    Only global-boundary vertices are locked; inter-chunk boundaries are free.
    """
    from decimation import decimate_mesh, indexed_to_stl

    pbar = None
    total_in = 0
    total_out = 0

    try:
        for V_coords, triangles, pbar in _stream_spatial_chunks(Z, bounds, config, resolution, chunk_size):
            total_in += len(triangles)

            if chunk_ratio < 1.0 and len(triangles) >= 100:
                target_chunk = max(4, int(len(triangles) * chunk_ratio))
                pbar.set_description(f"Gen+decimate ({total_out:,}→)")
                try:
                    triangles_stl = decimate_mesh(
                        V_coords, triangles, target_chunk,
                        lock_boundaries=True)
                except Exception as e:
                    logger.warning(f"Chunk decimation failed ({e}); writing raw chunk.")
                    triangles_stl = indexed_to_stl(V_coords, triangles)
            else:
                pbar.set_description("Generating mesh")
                triangles_stl = indexed_to_stl(V_coords, triangles)

            writer.write_triangles(triangles_stl)
            total_out += len(triangles_stl)
            del triangles, V_coords, triangles_stl
            pbar.update(1)

    finally:
        if pbar is not None:
            pbar.close()
        gc.collect()

    logger.info(f"Inline decimation: {total_in:,} → {total_out:,} triangles")
    return total_out


def global_decimate_stl(filepath: str, target_faces: int, chunk_faces: int = 500_000):
    """Decimate the final STL file in memory-mapped chunks.

    Loads the full mesh, builds a corner table, and runs the QEM decimation
    loop.  Because the mesh is already on disk, we only need RAM for the
    corner-table arrays (~40 bytes/tri + ~12 bytes/vert).

    Args:
        filepath:     Path to the binary STL file (modified in-place).
        target_faces: Target triangle count.
        chunk_faces:  Unused (reserved for future streaming decimation).
    """
    from decimation import stl_to_mesh, decimate_mesh, mesh_to_stl, build_corner_table, build_quadrics, decimate_loop

    data = np.memmap(filepath, dtype=STL_DTYPE, mode='r', offset=84)
    n_tris = len(data)
    if n_tris <= target_faces:
        logger.info(f"Global decimation: already at {n_tris:,} ≤ {target_faces:,} target")
        del data
        return

    logger.info(f"Global decimation: {n_tris:,} → {target_faces:,} target")
    logger.info(f"  Loading mesh ({n_tris * 50 / 1e6:.1f} MB STL → vertex/face arrays)...")
    log_memory("Before global decimate")

    # Convert to indexed mesh
    V_coords, triangles = stl_to_mesh(np.array(data))  # copy off memmap
    del data
    gc.collect()

    logger.info(f"  Indexed mesh: {len(V_coords):,} verts, {len(triangles):,} faces")
    log_memory("After indexing")

    # Build corner table and run decimation
    num_verts = len(V_coords)
    V, O, V2C = build_corner_table(triangles, num_verts)

    Q = np.zeros((num_verts, 10), dtype=np.float64)
    build_quadrics(V, V_coords.astype(np.float64), Q)

    log_memory("After corner table")
    logger.info("  Running QEM decimation (this may take a while)...")

    final_faces = decimate_loop(V, O, V2C, V_coords, Q, target_faces,
                                k_choices=8, lock_boundaries=True)
    logger.info(f"  Decimation complete: {n_tris:,} → {final_faces:,} faces")

    # Write back
    stl_out = mesh_to_stl(V, V_coords)
    del V, O, V2C, Q, V_coords, triangles
    gc.collect()

    # Overwrite file
    with open(filepath, 'wb') as f:
        f.write(b"Binary STL - Decimated".ljust(80, b'\x00'))
        f.write(struct.pack('<I', len(stl_out)))
        f.write(stl_out.tobytes())

    del stl_out
    gc.collect()
    log_memory("After global decimate")
    logger.info(f"  Wrote {final_faces:,} triangles to {filepath}")





# =============================================================================
# Main Pipeline
# =============================================================================

def estimate_triangle_count(Z: np.ndarray, chunk_size: int = 1000,
                            multiplier: float = 2.5) -> int:
    """
    Estimate total output triangles for ratio calculation.
    Multiplier accounts for top + bottom surfaces + side walls.
    Reduced from 3.0 to 2.5 since RTIN with tolerance merges flat regions.
    """
    h, w = Z.shape
    total_valid = 0

    for r in range(0, h, chunk_size):
        chunk = Z[r:min(r + chunk_size, h)]
        total_valid += np.sum(~np.isnan(chunk))

    return int(total_valid * multiplier)


def create_native_stl(input_folder: str, output_file: str, resolution: float,
                      w_cm: float, l_cm: float, h_cm: float, b_cm: float,
                      smoothing: int = 0,
                      planar_tol_m: float = 0.05,
                      target_faces: int = 0, chunk_size: int = 256):
    """Main entry point: streaming terrain-to-STL pipeline with low memory usage."""
    log_memory("Start")

    # Find input files
    tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    if not tif_files:
        tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tiff")))

    if not tif_files:
        logger.error(f"No .tif files found in '{input_folder}'")
        return

    config = MeshConfig(w_cm, l_cm, h_cm, b_cm, target_faces, resolution)
    config.smoothing = smoothing

    logger.info(f"Found {len(tif_files)} tiles")
    logger.info(f"Output: {config.width_mm:.1f} x {config.length_mm:.1f} x {config.height_mm:.1f} mm")

    temp_files = []  # Track temp files for cleanup

    try:
        # Phase 1: Scan bounds
        logger.info("Phase 1: Scanning bounds...")
        bounds = scan_tile_bounds(tif_files)
        logger.info(f"Extent: {bounds.width:.2f} x {bounds.height:.2f}")
        log_memory("After bounds scan")

        # Phase 2: Stitch tiles (memory-mapped)
        logger.info("Phase 2: Stitching tiles...")
        Z_global, global_width, global_height, z_temp_path = stitch_tiles_lowmem(
            tif_files, bounds, resolution)
        temp_files.append(z_temp_path)
        log_memory("After stitching")

        # Check valid data
        valid_count = 0
        for r in range(0, global_height, 1000):
            chunk = Z_global[r:min(r + 1000, global_height)]
            valid_count += np.sum(~np.isnan(chunk))

        if valid_count == 0:
            logger.error("No valid elevation data")
            return
        logger.info(f"Valid pixels: {valid_count:,}")

        # Phase 3: Normalize (in-place)
        logger.info("Phase 3: Normalizing elevation...")
        normalize_elevation_inplace(Z_global, config)
        log_memory("After normalization")

        # Scale planar tolerance from input meters to model units (mm)
        z_scale = (config.height_mm - config.base_mm) / config.z_range_input
        config.planar_tol = planar_tol_m * z_scale
        logger.info(f"Planar tolerance: {planar_tol_m:.3f}m -> {config.planar_tol:.4f} model units")

        # Phase 4: Add skirt (in-place)
        logger.info("Phase 4: Adding skirt...")
        add_skirt_inplace(Z_global)
        log_memory("After skirt")

        # Phase 4.5: Repair (always fills small holes)
        repair_elevation_inplace(Z_global, config.smoothing)

        # Phase 5+6: Generate (+ optionally decimate) in a single streaming pass.
        # No intermediate disk buffer is created — RAM footprint is one spatial
        # chunk at a time, so disk quota is never an issue.
        estimated_triangles = estimate_triangle_count(Z_global)
        logger.info(f"Estimated triangles: ~{estimated_triangles:,}")

        if target_faces > 0 and estimated_triangles > target_faces:
            # Per-chunk inline decimation with some headroom
            chunk_ratio = min((target_faces / estimated_triangles) * 1.2, 1.0)
            logger.info(
                f"Phase 5+6: Streaming generate + inline decimation "
                f"(per-chunk ratio={chunk_ratio:.4f}, "
                f"target ~{int(estimated_triangles * chunk_ratio):,} triangles)")
            with StreamingSTLWriter(output_file) as writer:
                actual_written = stream_generate_and_decimate(
                    Z_global, bounds, config, resolution,
                    chunk_ratio, writer, chunk_size)
        else:
            logger.info("Phase 5+6: Streaming generate + write (no decimation)...")
            with StreamingSTLWriter(output_file) as writer:
                actual_written = stream_generate_to_writer(
                    Z_global, bounds, config, resolution, writer, chunk_size)

        log_memory("After write")

        # Phase 7: Optional global decimation pass if still above target
        if target_faces > 0 and actual_written > target_faces:
            logger.info(f"Phase 7: Global decimation pass ({actual_written:,} → {target_faces:,})...")
            global_decimate_stl(output_file, target_faces)

        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"Output: {output_file} ({file_size_mb:.2f} MB)")
        logger.info("Complete!")

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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_folder", help="Folder containing .tif files")
    parser.add_argument("output_file", help="Output STL path")
    parser.add_argument("-r", "--resolution", type=float, default=1.0,
                        help="Sampling resolution")
    parser.add_argument("-W", "--width", type=float, default=10.0,
                        help="Width (cm)")
    parser.add_argument("-L", "--length", type=float, default=10.0,
                        help="Length (cm)")
    parser.add_argument("-H", "--height", type=float, default=3.0,
                        help="Height (cm)")
    parser.add_argument("-B", "--base", type=float, default=0.5,
                        help="Base (cm)")
    parser.add_argument("-s", "--smoothing", type=int, default=0,
                        help="Smoothing iterations (repair phase)")
    parser.add_argument("-p", "--planar-tolerance", type=float, default=0.05,
                        help="RTIN error tolerance (meters)")
    parser.add_argument("-f", "--faces", type=int, default=0,
                        help="Target faces (0 = no decimation)")
    parser.add_argument("-c", "--chunk-size", type=int, default=512,
                        help="Processing chunk size (smaller = less RAM, larger = better decimation)")

    args = parser.parse_args()

    create_native_stl(
        input_folder=args.input_folder,
        output_file=args.output_file,
        resolution=args.resolution,
        w_cm=args.width,
        l_cm=args.length,
        h_cm=args.height,
        b_cm=args.base,
        smoothing=args.smoothing,
        planar_tol_m=args.planar_tolerance,
        target_faces=args.faces,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
