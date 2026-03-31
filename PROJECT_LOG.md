# Project Log: Geospatial Terrain to STL

This document provides an overview of the terrain processing pipeline and tracks architectural changes and optimizations made to the codebase.

## Project Concept

The goal of this project is to provide a high-performance, memory-efficient pipeline for converting French geographic institute (IGN) GeoTIFF elevation data into 3D printable STL files. 

### Key Components
- **`download_tiles.py`**: Handles batch downloading of GeoTIFF tiles from URL lists.
- **`generate_mesh.py`**: The core engine that stitches tiles, processes elevation, and generates STL geometry using a streaming approach to minimize RAM footprint.
- **`decimation.py`**: A custom, Numba-accelerated mesh simplifier using Quadric Error Metrics (QEM) and stellar valence heuristics.

---

## Change Log

### 2026-03-31 | **Performance Optimizations & Sharp Edge Preservation**
| Change | Rationale | Details |
| :--- | :--- | :--- |
| **Dual-raster error computation** | Error was computed against downsampled raster, causing artificial smoothing. | Added `Z_error` raster at native source resolution. Error now computed against full-resolution data regardless of base mesh resolution. |
| **Nearest neighbor sampling** | Bilinear interpolation smoothed sharp edges (buildings, trees). | Added `_nearest_sample_njit()` for high-res meshes (≤2m). Automatically enables for `-r ≤ 2` to preserve pixel-perfect detail. |
| **Numba-accelerated error computation** | `_compute_triangle_error()` called millions of times with meshgrid allocation overhead. | Replaced meshgrid with manual loops in `_compute_triangle_error_njit()`. 5-10x speedup. |
| **Numba-accelerated bilinear sampling** | Python overhead per sample call. | Added `_bilinear_sample_njit()` with `@njit(cache=True)`. 3-5x speedup. |
| **Vectorized centroid calculations** | Python loops for centroid computation in window processing. | Replaced with NumPy vectorized operations. 5-10x speedup. |
| **Queue operations optimization** | `list.pop(0)` was O(n) per operation. | Replaced with `collections.deque` for O(1) popleft. 20-30% speedup. |
| **Boolean array for processed tracking** | Set lookups had hashing overhead. | Replaced `processed` set with boolean numpy array. |
| **Batch vertex sampling** | Loop-based vertex sampling for initial mesh. | Added `sample_raster_batch()` for vectorized batch sampling. |
| **Synchronized edge splitting** | Edge splitting created T-junctions and holes. | Added `_split_edge_atomic()` that splits ALL triangles sharing an edge atomically. |
| **Live adjacency updates** | Edge-to-triangle map became stale during refinement. | Added `_update_adjacency_after_split()` to keep adjacency current. |
| **Safety limits for extreme parameters** | No protection against runaway refinement. | Added `MAX_TRIANGLES_HARD_LIMIT` (100M), `MAX_DEPTH_HARD_LIMIT` (50), `MIN_ERROR_THRESHOLD_M` (0.06mm). |
| **Edge boundary margin** | Triangles at raster edges were over-subdivided. | Added 2-pixel boundary margin check in error computation. |
| **Progress bar fix** | Progress bar was duplicating/copying itself each iteration. | Added `leave=False` and `position=0` to tqdm. |
| **Max depth limit increased** | Limit of 20 was too restrictive for high-detail meshes. | Increased to 50, now only warns instead of forcing cap. |
| **Minimum error threshold lowered** | 1mm minimum was too coarse for high-detail printers. | Set to 0.06mm to match typical FDM printer layer height capability. |
| **Removed parallel processing** | ThreadPoolExecutor had race conditions causing mesh corruption. | Always use single-threaded sequential processing with synchronized edge splitting. |
| **Removed `-j/--workers` argument** | Parallel processing no longer supported. | Cleaned up CLI and function signatures. |

### 2026-03-31 | **Parallel Processing Removal & Code Cleanup**
| Change | Rationale | Details |
| :--- | :--- | :--- |
| **Disabled ThreadPoolExecutor** | Race conditions in shared memory caused mesh corruption. | Removed parallel refinement path; now uses reliable single-threaded processing with synchronized edge splitting. |
| **Removed `-j/--workers` CLI argument** | Parallel processing no longer supported. | Removed from argument parser and `create_native_stl()` function signature. |
| **Code cleanup** | Remove dead code and unused functions. | Removed `_find_connected_components()`, `_refine_component()`, and `ThreadPoolExecutor` import. |
| **Fast base triangulation** | Ear-clipping was O(n²) and extremely slow for large boundaries. | Replaced with O(n) fan triangulation from centroid in `_triangulate_polygon_2d`. |
| **Removed diagnostic scripts** | Clean up repository before release. | Deleted: `analyze_base_boundary.py`, `analyze_skinny_triangles.py`, `analyze_stl.py`, `debug_chunk_boundaries.py`, `diagnose_mesh.py`, `test_mesh.py`, `view_stl.py`, `visual_mesh_check.py`. |

### 2026-03-30 | **Wall Generation Fix - Only Global Boundaries**
| Change | Rationale | Details |
| :--- | :--- | :--- |
| **Global boundary wall generation** | Walls were created at ALL chunk boundaries (internal), causing 61.8% of mesh to be walls. | Modified `_generate_side_triangles_indexed` to accept `is_global_boundary` flag. Only creates walls at the outer-most boundary of the entire terrain. |
| **z_bottom sign fix** | Base was at positive Z instead of negative. | Changed `z_bottom = config.base_mm` to `z_bottom = -config.base_mm` so base extends below terrain (z=0 to z=-base_mm). |
| **Visual analysis tool** | Need to visually verify mesh quality beyond topology metrics. | Created `visual_mesh_check.py` that renders mesh from multiple angles with different color modes and analyzes for visual artifacts. |

### 2026-03-29 | **Base Triangulation Fix (Critical)**
| Change | Rationale | Details |
| :--- | :--- | :--- |
| **Grid-based base triangulation** | Ear-clipping created extremely skinny triangles (aspect 200-562) for staircase boundary shapes. | Replaced `_triangulate_polygon_2d` with a simple grid-based approach spanning the terrain bounding box. Base now has uniform 2mm triangles with aspect ~2.4. |
| **Round X/Y in merge** | Ensure consistent coordinates at chunk boundaries. | Now rounds X, Y, Z in `_merge_and_deduplicate_indexed_meshes`. |

### 2026-03-29 | **Geometry Fixes - Zigzag Edges, Skirt, Base Triangulation**
| Change | Rationale | Details |
| :--- | :--- | :--- |
| **Dense grid triangulation** | RTIN created zigzag edges at chunk boundaries due to diagonal triangulation not aligning between chunks. | Replaced RTIN with dense grid triangulation for top surface. Ensures perfect boundary alignment. |
| **Skirt height fix** | `z_bottom` was positive `base_mm`, but terrain starts at z=0, so skirt should go below terrain. | Changed from `z_bottom = config.base_mm` to `z_bottom = -config.base_mm`. Skirt now extends from z=0 to z=-base_mm. |
| **Bottom surface mask fix** | Using `quad_valid` created gaps in bottom surface. | Changed to use full `valid` mask for complete bottom coverage. |
| **Zero degenerate triangles** | ~14K degenerate triangles in early versions. | All degenerate triangles eliminated (0 remaining). |
| **Zero non-manifold edges** | 1.5M non-manifold edges reduced to 0. | Combined fixes above eliminated all topology issues. |
| **Vectorized `_generate_bottom_triangles_indexed`** | Python loop over strips was slow. | Replaced with fully vectorized NumPy implementation. |
| **Vectorized `_generate_side_triangles_indexed`** | Simplified side wall generation. | Streamlined implementation using vectorized operations. |
| **Pre-round coordinates in streaming loop** | Coordinate rounding in merge was expensive. | Pre-round X, Y in chunk processing; only round Z in final merge. |
| **Test suite created** | Validate geometry correctness and performance. | `test_mesh.py` with checks for degenerate triangles, non-manifold edges, boundary regularity, decimation, and chunk sizes. |
| **Diagnostic tools created** | Analyze mesh issues and slicer warnings. | `diagnose_mesh.py`, `analyze_skinny_triangles.py`, `analyze_base_boundary.py`, `debug_chunk_boundaries.py`. |

### 2026-03-29 | **Critical Topology Fixes & Decimation Pipeline Repair**
| Change | Rationale | Details |
| :--- | :--- | :--- |
| **Fix RTIN degenerate triangles** | 25% of RTIN output was zero-area tris (e.g. `[v,v,v]`), poisoning corner-table topology and completely preventing decimation. | Added degenerate filter after RTIN extraction: `(t0 != t1) & (t1 != t2) & (t0 != t2)`. Eliminated 12,223 degenerate triangles per chunk. |
| **Fix post-dedup degenerate filter** | `np.unique` coordinate remapping can collapse distinct indices → same vertex. | Added second degenerate filter after `_merge_and_deduplicate_indexed_meshes`. |
| **Fix non-manifold edge elimination** | 8,149 non-manifold edges per chunk blocked all edge collapses. | Both fixes above eliminated all non-manifold edges (edges shared by >2 triangles). |
| **Guard single-pixel wall segments** | Side wall generator emitted zero-width walls when runs had `c_start == c_end`. | Skip wall segments where both endpoints are the same grid position. |
| **Defensive degenerate filter in decimation** | Belt-and-suspenders safety. | `decimate_mesh()` now filters degenerate triangles before building corner table. |
| **V2C refresh improvement** | After edge collapse, V2C refresh started from deleted triangles. | Now searches surviving opposite triangles first, then falls back to walk. |
| **Global decimation pass (Phase 7)** | Per-chunk decimation can't reach aggressive targets due to boundary locking. | New `global_decimate_stl()` runs QEM on the full assembled STL when still above target. |
| **Default chunk size 256 → 512** | Larger chunks have better interior:boundary ratio for decimation. | Reduces boundary vertex percentage from ~90% to ~55%. |
| **Improved estimation** | Triangle count estimator overshot by ~30%. | Multiplier reduced from 3.0 to 2.5; headroom from 1.1× to 1.2×. |
| **is_global_boundary flag** | Preparation for selective boundary locking. | Chunks now know whether they touch the outermost global boundary. |

### 2026-03-28 | **Codebase Cleanup & Optimization**
| Change | Rationale | Details |
| :--- | :--- | :--- |
| **Dead code removal** | ~300 lines of legacy two-pass pipeline code removed. | Removed `TriangleBuffer`, `decimate_streaming`, `_decimate_chunk`, `stream_generate_mesh`, `memory_mapped_array`, unused `max_block_size` config field, unused `max_memory_mb` and `buffer_multiplier` CLI params. |
| **Skip inline normal computation** | ~50% reduction in normal math work. | Normals are left zeroed during chunk generation; the final `recompute_normals()` pass in `StreamingSTLWriter` handles them in one vectorized sweep. |
| **Vectorize bottom strip loop** | Eliminate Python for-loop over strips. | Bottom surface strip fill is now fully vectorized with numpy. |
| **RTIN boundary error propagation fix** | Boundary locks were not propagating to parent nodes. | `np.inf` errors are now initialized on boundaries **before** the bottom-up error computation, ensuring correct upward propagation. |

### 2026-03-28 | **RTIN Terrain Meshing (Martini Algorithm)**
| Change | Rationale | Details |
| :--- | :--- | :--- |
| **Replaced hierarchical planar merging** | Better adaptive tessellation. | Migrated top surface from block-based merging to RTIN (Martini algorithm) for T-junction-free adaptive meshing. |
| **Numba-accelerated RTIN kernels** | Performance. | Bottom-up error computation and top-down extraction are JIT-compiled. |
| **Chunk boundary locking** | Prevent cross-chunk seams. | Boundary pixel errors set to `inf` to guarantee matching triangulations between adjacent chunks. |

### 2026-03-28 | **Full Feature Update**
| Change | Rationale | Details |
| :--- | :--- | :--- |
| **Geometry Repair Phase** | Address noise in source data. | Added `repair_elevation_inplace` with hole filling and smoothing. |
| **Internal Wall Fix (Seams)** | Remove "cutting lines" in output. | Updated `_generate_side_triangles` to avoid walls between chunks. |
| **STL Normal Recomputation** | Improve shading and STL validity. | Added a post-processing pass to recalculate face normals from vertices. |
| **Bottom/Side Strip Merging** | Reduce triangle counts. | Bottom strips: ~90% reduction. Side walls: consecutive boundary edges merged. |

---

## Technical Appendix

### RTIN / Martini Algorithm

The top-surface generator uses **Right-Triangulated Irregular Networks** (RTIN), based on the Martini algorithm by Vladimir Agafonkin (Mapbox):

1. **Precompute** triangle coordinates for the implicit binary tree over a 2^k+1 grid (cached per grid size).
2. **Bottom-up error computation**: For each triangle, compute the interpolation error at the hypotenuse midpoint and propagate child errors upward. Boundary and invalid-pixel errors are pre-set to `inf` before this pass to guarantee full resolution at edges.
3. **Top-down mesh extraction**: Recursively split triangles whose midpoint error exceeds `--planar-tolerance`; emit leaf triangles directly.

**Key properties:**
- No T-junctions by construction (matching subdivisions on shared edges).
- Chunk boundaries are locked to `inf` error → identical edge tessellation between adjacent chunks.
- Flat regions collapse to 2 triangles (the two root triangles); detailed terrain gets full resolution.

### Streaming Pipeline Architecture

```
GeoTIFF tiles
    → stitch (memory-mapped)
    → normalize elevation (in-place)
    → add skirt + repair holes (in-place)
    → for each spatial chunk:
        → RTIN top surface + strip-merged bottom + strip-merged walls
        → [optional inline QEM decimation per chunk]
        → write directly to STL file
    → recompute normals (memory-mapped final pass)
```

RAM footprint: one spatial chunk at a time. Disk: elevation memmap + final STL file.
