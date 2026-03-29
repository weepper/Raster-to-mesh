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

    # Phase 2: Smoothing (Box blur)
    # To do this correctly in-place without "forward-leaking" values,
    # we'd need a full copy. Instead, we'll use a 3-row sliding buffer.
    for _ in range(smoothing_it):
        prev_row_orig = Z[0].copy()
        curr_row_orig = Z[1].copy()

        for r in range(1, h - 1):
            next_row_orig = Z[r + 1].copy()
            curr_row = Z[r]

            # Vectorized 3x3 local mean (approximate)
            # Only smooth pixels that aren't on the absolute edge
            for c in range(1, w - 1):
                if np.isnan(curr_row_orig[c]):
                    continue

                # Quick 3x3 check
                val_sum = 0.0
                val_count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == -1:
                            v = prev_row_orig[c+dc]
                        elif dr == 0:
                            v = curr_row_orig[c+dc]
                        else:
                            v = next_row_orig[c+dc]

                        if not np.isnan(v):
                            val_sum += v
                            val_count += 1

                if val_count > 0:
                    curr_row[c] = val_sum / val_count

            prev_row_orig = curr_row_orig
            curr_row_orig = next_row_orig

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
                         terrain_flat, grid_size, errors):
    """Compute RTIN approximation errors (Numba JIT, bottom-up)."""
    for i in range(num_triangles - 1, -1, -1):
        k = i * 4
        ax = coords[k]
        ay = coords[k + 1]
        bx = coords[k + 2]
        by = coords[k + 3]
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        cx = mx + my - ay
        cy = my + ax - mx

        mid_idx = my * grid_size + mx

        # Interpolation error at the hypotenuse midpoint
        interpolated = (terrain_flat[ay * grid_size + ax] +
                        terrain_flat[by * grid_size + bx]) * 0.5
        mid_error = abs(interpolated - terrain_flat[mid_idx])
        if mid_error > errors[mid_idx]:
            errors[mid_idx] = mid_error

        if i < num_parent_triangles:
            # Propagate child errors upward
            left_idx  = ((ay + cy) >> 1) * grid_size + ((ax + cx) >> 1)
            right_idx = ((by + cy) >> 1) * grid_size + ((bx + cx) >> 1)
            if errors[left_idx] > errors[mid_idx]:
                errors[mid_idx] = errors[left_idx]
            if errors[right_idx] > errors[mid_idx]:
                errors[mid_idx] = errors[right_idx]

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



def _merge_indexed_meshes(parts):
    if not parts:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    total_verts = sum(len(p[0]) for p in parts)
    total_tris = sum(len(p[1]) for p in parts)
    
    out_v = np.empty((total_verts, 3), dtype=np.float32)
    out_t = np.empty((total_tris, 3), dtype=np.int32)
    
    v_offset = 0
    t_offset = 0
    for v, t in parts:
        nv = len(v)
        nt = len(t)
        if nt == 0: continue
        
        out_v[v_offset:v_offset+nv] = v
        out_t[t_offset:t_offset+nt] = t + v_offset
        
        v_offset += nv
        t_offset += nt
        
    return out_v, out_t

def generate_chunk_triangles(Z: np.ndarray, X: np.ndarray, Y: np.ndarray,
                               valid: np.ndarray,
                               global_offset: tuple,
                               global_shape: tuple,
                               planar_tol: float = 0.0):
    """Generate all triangles for a chunk (top, bottom, sides) as indexed mesh."""
    from decimation import stl_to_mesh
    parts = []

    quad_valid = (valid[:-1, :-1] & valid[:-1, 1:] &
                  valid[1:, :-1] & valid[1:, 1:])

    # Top surface (RTIN adaptive directly yields indexed)
    top_v, top_t = _generate_top_triangles_rtin(X, Y, Z, valid, planar_tol)
    if len(top_t) > 0:
        parts.append((top_v, top_t))

    # Bottom surface (strip-merged -> STL -> convert)
    bottom = _generate_bottom_triangles_stripped(X, Y, quad_valid)
    if len(bottom) > 0:
        bv, bt = stl_to_mesh(bottom)
        parts.append((bv, bt))

    # Side walls (strip-merged -> STL -> convert) 
    sides = _generate_side_triangles_stripped(X, Y, Z, valid, global_offset, global_shape)
    if len(sides) > 0:
        sv, st = stl_to_mesh(sides)
        parts.append((sv, st))

    return _merge_indexed_meshes(parts)


def _generate_top_triangles_rtin(X, Y, Z, valid, planar_tol=0.0):
    """
    Generate top surface triangles using RTIN (Martini algorithm).

    Adaptively tessellates the heightmap: flat regions get large triangles,
    detailed regions get dense triangles.  No T-junctions by construction.
    Falls back to dense grid if planar_tol is 0 or chunk is too small.
    """
    h, w = Z.shape
    if h < 2 or w < 2:
        return np.array([], dtype=STL_DTYPE)

    # Fast path: no adaptive meshing requested
    if planar_tol <= 0:
        return _generate_dense_top(X, Y, Z, valid)

    # Pad chunk to next 2^k + 1  (usually already 257 = 2^8 + 1)
    grid_size = _next_power_of_two_plus_one(max(h, w))

    # Too small for RTIN to help
    if grid_size < 5:
        return _generate_dense_top(X, Y, Z, valid)

    # Build padded terrain (NaN → 0 for error computation)
    terrain = np.zeros((grid_size, grid_size), dtype=np.float32)
    terrain[:h, :w] = np.where(np.isnan(Z), 0.0, Z)
    terrain_flat = terrain.ravel()

    # Mark invalid pixels: we'll force full resolution near NaN
    valid_pad = np.zeros((grid_size, grid_size), dtype=np.bool_)
    valid_pad[:h, :w] = valid

    # Get precomputed Martini data for this grid size
    coords, num_tri, num_parent = _get_martini(grid_size)

    # Phase 1: Initialize errors array with np.inf on boundaries to force full resolution
    errors = np.zeros(grid_size * grid_size, dtype=np.float32)
    errors_2d = errors.reshape(grid_size, grid_size)
    errors_2d[0, :] = np.inf
    errors_2d[:, 0] = np.inf
    if h < grid_size:
        errors_2d[h - 1, :] = np.inf     # real bottom edge
        errors_2d[h:, :] = np.inf         # padding
    else:
        errors_2d[-1, :] = np.inf
    if w < grid_size:
        errors_2d[:, w - 1] = np.inf      # real right edge
        errors_2d[:, w:] = np.inf          # padding
    else:
        errors_2d[:, -1] = np.inf

    invalid_pad = ~valid_pad
    if np.any(invalid_pad[:h, :w]):
        from scipy.ndimage import binary_dilation
        try:
            dilated = binary_dilation(invalid_pad, iterations=2)
            errors_2d[dilated] = np.inf
        except ImportError:
            for r in range(h):
                for c in range(w):
                    if invalid_pad[r, c]:
                        r0, r1 = max(0, r - 2), min(grid_size, r + 3)
                        c0, c1 = max(0, c - 2), min(grid_size, c + 3)
                        errors_2d[r0:r1, c0:c1] = np.inf

    # Phase 2: Compute RTIN errors bottom-up, propagating inf toward root
    _rtin_compute_errors(coords, num_tri, num_parent,
                         terrain_flat, grid_size, errors)

    # Phase 3: Extract mesh at given error threshold
    # Max possible triangles = 2 * (grid_size-1)^2 (full resolution)
    max_coord = grid_size - 1
    indices = np.zeros(grid_size * grid_size, dtype=np.int32)
    counts = np.zeros(2, dtype=np.int32)  # [num_vertices, num_triangles]

    _rtin_count(0, 0, max_coord, max_coord, max_coord, 0,
                errors, grid_size, planar_tol, indices, counts)
    _rtin_count(max_coord, max_coord, 0, 0, 0, max_coord,
                errors, grid_size, planar_tol, indices, counts)

    num_verts = counts[0]
    num_tris = counts[1]

    if num_tris == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)

    vertices_grid = np.empty((num_verts, 2), dtype=np.int32)
    triangles = np.empty((num_tris, 3), dtype=np.int32)

    counts[1] = 0
    _rtin_extract(0, 0, max_coord, max_coord, max_coord, 0,
                  errors, grid_size, planar_tol, indices, vertices_grid, triangles, counts)
    _rtin_extract(max_coord, max_coord, 0, 0, 0, max_coord,
                  errors, grid_size, planar_tol, indices, vertices_grid, triangles, counts)

    tax, tay = vertices_grid[triangles[:, 0], 0], vertices_grid[triangles[:, 0], 1]
    tbx, tby = vertices_grid[triangles[:, 1], 0], vertices_grid[triangles[:, 1], 1]
    tcx, tcy = vertices_grid[triangles[:, 2], 0], vertices_grid[triangles[:, 2], 1]

    in_bounds = ((tax < w) & (tay < h) &
                 (tbx < w) & (tby < h) &
                 (tcx < w) & (tcy < h))
    valid_tris = (in_bounds &
                  valid_pad[tay, tax] &
                  valid_pad[tby, tbx] &
                  valid_pad[tcy, tcx])
    
    triangles = triangles[valid_tris]

    if len(triangles) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)

    vy = vertices_grid[:, 1]
    vx = vertices_grid[:, 0]

    # Clip indices to actual chunk size for unreferenced padding vertices
    vy_clipped = np.minimum(vy, h - 1)
    vx_clipped = np.minimum(vx, w - 1)

    V_coords = np.column_stack([
         X[vy_clipped, vx_clipped],
         Y[vy_clipped, vx_clipped],
         Z[vy_clipped, vx_clipped]
    ])

    return V_coords, triangles


def _generate_dense_top(X, Y, Z, valid):
    """Fallback: standard 2-tri-per-quad top surface (no adaptive merging)."""
    h, w = Z.shape
    if h < 2 or w < 2:
        return np.array([], dtype=STL_DTYPE)

    quad_valid = (valid[:-1, :-1] & valid[:-1, 1:] &
                  valid[1:, :-1] & valid[1:, 1:])

    ri, ci = np.where(quad_valid)
    n = len(ri)
    if n == 0:
        return np.array([], dtype=STL_DTYPE)

    # Normals are left zeroed; the final recompute_normals() pass handles them.
    triangles = np.zeros(n * 2, dtype=STL_DTYPE)
    x00, y00, z00 = X[ri, ci],     Y[ri, ci],     Z[ri, ci]
    x01, y01, z01 = X[ri, ci+1],   Y[ri, ci+1],   Z[ri, ci+1]
    x10, y10, z10 = X[ri+1, ci],   Y[ri+1, ci],   Z[ri+1, ci]
    x11, y11, z11 = X[ri+1, ci+1], Y[ri+1, ci+1], Z[ri+1, ci+1]

    triangles['v1'][::2]  = np.column_stack([x00, y00, z00])
    triangles['v2'][::2]  = np.column_stack([x10, y10, z10])
    triangles['v3'][::2]  = np.column_stack([x01, y01, z01])
    triangles['v1'][1::2] = np.column_stack([x01, y01, z01])
    triangles['v2'][1::2] = np.column_stack([x10, y10, z10])
    triangles['v3'][1::2] = np.column_stack([x11, y11, z11])

    return triangles


# ---- bottom surface (row-strip merged) -------------------------------------

def _generate_bottom_triangles_stripped(X, Y, quad_valid,
                                        z_bottom: float = 0.0) -> np.ndarray:
    """
    Generate bottom surface at constant z_bottom using row-strip merging.

    Since the bottom is perfectly planar, consecutive valid quads in each
    row are merged into a single rectangular strip (2 triangles) instead
    of the naive 2-per-quad.  Typical reduction: 80-95 % on the bottom.
    Normals are left zeroed; the final recompute_normals() pass handles them.
    """
    h_q, w_q = quad_valid.shape   # (h-1, w-1)
    if h_q == 0 or w_q == 0:
        return np.array([], dtype=STL_DTYPE)

    strips = []   # list of (row, col_start, col_end) — end is exclusive

    for r in range(h_q):
        row = quad_valid[r]
        c = 0
        while c < w_q:
            if not row[c]:
                c += 1
                continue
            c_run_start = c
            while c < w_q and row[c]:
                c += 1
            strips.append((r, c_run_start, c))

    if not strips:
        return np.array([], dtype=STL_DTYPE)

    # Vectorized: build strip arrays and fill all triangles at once
    sa = np.array(strips, dtype=np.int32)  # (n, 3): row, col_start, col_end
    rs  = sa[:, 0]
    css = sa[:, 1]
    ces = sa[:, 2]
    n = len(sa)

    triangles = np.zeros(n * 2, dtype=STL_DTYPE)

    x0 = X[rs,     css];  y0 = Y[rs,     css]
    x1 = X[rs,     ces];  y1 = Y[rs,     ces]
    x2 = X[rs + 1, css];  y2 = Y[rs + 1, css]
    x3 = X[rs + 1, ces];  y3 = Y[rs + 1, ces]

    zb = np.full(n, z_bottom, dtype=np.float32)

    triangles['v1'][::2]  = np.column_stack([x0, y0, zb])
    triangles['v2'][::2]  = np.column_stack([x1, y1, zb])
    triangles['v3'][::2]  = np.column_stack([x2, y2, zb])
    triangles['v1'][1::2] = np.column_stack([x1, y1, zb])
    triangles['v2'][1::2] = np.column_stack([x3, y3, zb])
    triangles['v3'][1::2] = np.column_stack([x2, y2, zb])

    return triangles


# ---- side walls (strip-merged) ---------------------------------------------

def _generate_side_triangles_stripped(X, Y, Z, valid,
                                      global_offset, global_shape,
                                      z_bottom: float = 0.0) -> np.ndarray:
    """
    Generate side wall triangles at global boundaries with strip merging.

    Consecutive boundary edges along the same direction are merged into
    a single wall strip (2 triangles) instead of 2-per-edge.
    """
    h, w = Z.shape
    row_start, col_start = global_offset
    total_h, total_w = global_shape

    # Collect oriented boundary edges as (x1,y1,z1, x2,y2,z2) segments.
    # Groups: left, right, top, bottom — we merge within each group per row/col.
    walls = []

    # --- Horizontal boundary rows (top / bottom walls) ---
    for r in range(h):
        for direction in ('top', 'bottom'):
            if direction == 'top':
                is_boundary_row = (row_start + r == 0)
                check_row = r - 1
            else:
                is_boundary_row = (row_start + r == total_h - 1)
                check_row = r + 1

            # Scan for runs of boundary edges along columns
            c = 0
            while c < w:
                if not valid[r, c]:
                    c += 1
                    continue

                # Is this pixel on the boundary in this direction?
                on_boundary = False
                if direction == 'top':
                    on_boundary = is_boundary_row or (check_row >= 0 and not valid[check_row, c])
                else:
                    on_boundary = is_boundary_row or (check_row < h and not valid[check_row, c])

                if not on_boundary or c + 1 >= w or not valid[r, c + 1]:
                    c += 1
                    continue

                # Start of a run of boundary edges
                c_run_start = c
                c += 1
                while c < w and valid[r, c]:
                    # Check this pixel is also on the boundary
                    if direction == 'top':
                        still_boundary = is_boundary_row or (check_row >= 0 and not valid[check_row, c])
                    else:
                        still_boundary = is_boundary_row or (check_row < h and not valid[check_row, c])
                    if not still_boundary:
                        break
                    c += 1

                # Emit a wall strip from c_run_start to c (clamped to last column)
                c_strip_end = min(c, w - 1)
                if c_strip_end <= c_run_start:
                    continue
                if direction == 'top':
                    walls.append(_wall_strip(
                        X[r, c_run_start], Y[r, c_run_start], Z[r, c_run_start],
                        X[r, c_strip_end], Y[r, c_strip_end], Z[r, c_strip_end],
                        z_bottom))
                else:
                    walls.append(_wall_strip(
                        X[r, c_strip_end], Y[r, c_strip_end], Z[r, c_strip_end],
                        X[r, c_run_start], Y[r, c_run_start], Z[r, c_run_start],
                        z_bottom))

    # --- Vertical boundary columns (left / right walls) ---
    for c in range(w):
        for direction in ('left', 'right'):
            if direction == 'left':
                is_boundary_col = (col_start + c == 0)
                check_col = c - 1
            else:
                is_boundary_col = (col_start + c == total_w - 1)
                check_col = c + 1

            r = 0
            while r < h:
                if not valid[r, c]:
                    r += 1
                    continue

                on_boundary = False
                if direction == 'left':
                    on_boundary = is_boundary_col or (check_col >= 0 and not valid[r, check_col])
                else:
                    on_boundary = is_boundary_col or (check_col < w and not valid[r, check_col])

                if not on_boundary or r + 1 >= h or not valid[r + 1, c]:
                    r += 1
                    continue

                r_run_start = r
                r += 1
                while r < h and valid[r, c]:
                    if direction == 'left':
                        still_boundary = is_boundary_col or (check_col >= 0 and not valid[r, check_col])
                    else:
                        still_boundary = is_boundary_col or (check_col < w and not valid[r, check_col])
                    if not still_boundary:
                        break
                    r += 1

                r_strip_end = min(r, h - 1)
                if r_strip_end <= r_run_start:
                    continue
                if direction == 'left':
                    walls.append(_wall_strip(
                        X[r_strip_end, c], Y[r_strip_end, c], Z[r_strip_end, c],
                        X[r_run_start, c], Y[r_run_start, c], Z[r_run_start, c],
                        z_bottom))
                else:
                    walls.append(_wall_strip(
                        X[r_run_start, c], Y[r_run_start, c], Z[r_run_start, c],
                        X[r_strip_end, c], Y[r_strip_end, c], Z[r_strip_end, c],
                        z_bottom))

    if walls:
        return np.concatenate(walls)
    return np.array([], dtype=STL_DTYPE)


def _wall_strip(x1, y1, z1, x2, y2, z2, z_bottom) -> np.ndarray:
    """Create a wall strip (2 triangles) between two top-edge points.

    Normals are left zeroed; the final recompute_normals() pass handles them.
    """
    tris = np.zeros(2, dtype=STL_DTYPE)

    tris['v1'][0] = [x1, y1, z1]
    tris['v2'][0] = [x1, y1, z_bottom]
    tris['v3'][0] = [x2, y2, z2]

    tris['v1'][1] = [x2, y2, z2]
    tris['v2'][1] = [x1, y1, z_bottom]
    tris['v3'][1] = [x2, y2, z_bottom]

    return tris



def _stream_spatial_chunks(Z: np.ndarray, bounds: GeoBounds, config: MeshConfig,
                            resolution: float, chunk_size: int):
    """
    Generator: yield successive (triangles, pbar) from the elevation grid.
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

            V_coords, triangles = generate_chunk_triangles(z_chunk, x_chunk, y_chunk, valid_chunk,
                                               (row_start, col_start), (h, w), config.planar_tol)

            del z_chunk, x_chunk, y_chunk, valid_chunk

            if len(triangles) > 0:
                yield V_coords, triangles, pbar
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
    Boundary vertices are locked to avoid seams between adjacent chunks.
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
                    triangles_stl = decimate_mesh(V_coords, triangles, target_chunk, lock_boundaries=True)
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





# =============================================================================
# Main Pipeline
# =============================================================================

def estimate_triangle_count(Z: np.ndarray, chunk_size: int = 1000,
                            multiplier: float = 4.5) -> int:
    """
    Estimate total output triangles for ratio calculation.
    Multiplier of 4.5× accounts for top + bottom surfaces + side walls.
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
            chunk_ratio = min((target_faces / estimated_triangles) * 1.1, 1.0)
            logger.info(
                f"Phase 5+6: Streaming generate + inline decimation "
                f"(per-chunk ratio={chunk_ratio:.4f}, "
                f"target ~{int(estimated_triangles * chunk_ratio):,} triangles)")
            with StreamingSTLWriter(output_file) as writer:
                stream_generate_and_decimate(
                    Z_global, bounds, config, resolution,
                    chunk_ratio, writer, chunk_size)
        else:
            logger.info("Phase 5+6: Streaming generate + write (no decimation)...")
            with StreamingSTLWriter(output_file) as writer:
                stream_generate_to_writer(
                    Z_global, bounds, config, resolution, writer, chunk_size)

        log_memory("After write")

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
    parser.add_argument("-c", "--chunk-size", type=int, default=256,
                        help="Processing chunk size (smaller = less RAM)")

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
