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
