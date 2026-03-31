# 🗺️ Raster-to-Mesh: High-Performance Terrain STL Generator

A memory-efficient, Python-powered pipeline for converting **French Geographic Institute (IGN)** GeoTIFF elevation data into 3D printable STL files.

---

## 🚀 Key Features

- **Streaming Architecture**: Process massive datasets with a minimal RAM footprint. The pipeline operates on spatial chunks, ensuring only a fraction of the data is in memory at once.
- **Adaptive Tessellation (RTIN)**: Utilizes the **Martini (Right-Triangulated Irregular Networks)** algorithm for T-junction-free, adaptive meshing of top surfaces.
- **Dual-Raster Error Computation**: Error is computed against full-resolution source data while the base mesh uses the user-specified resolution. This prevents artificial smoothing of sharp features.
- **Nearest Neighbor Sampling**: For high-resolution meshes (≤2m), nearest neighbor sampling preserves sharp edges like buildings and trees instead of bilinear smoothing.
- **Numba-Accelerated Kernels**: Core mathematical operations, including RTIN extraction and QEM decimation, are JIT-compiled for near-C performance.
- **Custom QEM Decimation**: Sophisticated mesh simplification using **Quadric Error Metrics** and stellar valence heuristics to achieve aggressive triangle reduction without losing detail.
- **Seamless Merging**: Automatic stitch-and-repair for multi-tile inputs, handling holes and noise in source data.
- **Side Wall & Base Generation**: Out-of-the-box support for generating manifold 3D printable "blocks" with clean sides and flat bases.
- **Synchronized Edge Splitting**: When triangles are refined, all triangles sharing an edge are split atomically, preventing holes and T-junctions in the mesh.

---

## 🛠️ Technical Workflow

The pipeline operates in 7 distinct phases to ensure geometry integrity and performance:

1.  **Stitch**: Memory-map and combine multiple GeoTIFF tiles into a unified elevation grid. Creates both a downsampled raster (for base mesh) and a full-resolution raster (for error computation).
2.  **Repair**: In-place elevation normalization, hole-filling, and smoothing.
3.  **RTIN Top Surface**: Adaptive triangulation using Martini-style hierarchical subdivision.
4.  **Base & Walls**: Vectorized generation of side-walls and unified bottom surface strips.
5.  **Chunk Merging**: Deduplication of vertices across chunk boundaries to ensure manifold connectivity.
6.  **Global QEM Decimation**: Aggressive simplification pass to hit target triangle counts.
7.  **Normal Recomputation**: Vectorized face normal calculation for accurate shading and valid STL headers.

---

## 📦 Installation

This project requires **Python 3.11+**.

```bash
# Clone the repository
git clone https://github.com/weepper/Raster-to-mesh.git
cd Raster-to-mesh

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy numba rasterio requests tqdm scipy
```

---

## 🖱️ Usage

### 1. Download IGN Tiles
Prepare a text file (`dalles.txt`) containing the URLs of the tiles you wish to download.

```bash
python download_tiles.py -i dalles.txt -d input_folder -w 10
```

### 2. Generate STL Mesh
Convert a folder of GeoTIFFs into a 3D printable STL.

```bash
python generate_mesh.py input_folder output.stl \
    -r 1.0 \
    -W 10 \
    -L 10 \
    -H 3 \
    -B 0.5 \
    -e 0.05 \
    -d 5
```

### Key CLI Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `-r, --resolution` | `1.0` | Grid resolution in meters. ≤2m enables nearest neighbor for sharp features. |
| `-W, --width` | `10.0` | Physical output width in cm. |
| `-L, --length` | `10.0` | Physical output length in cm. |
| `-H, --height` | `3.0` | Physical output height in cm. |
| `-B, --base` | `0.0` | Thickness of the solid base under the terrain (cm, 0 = no base). |
| `-e, --error-threshold` | `0.05` | RTIN subdivision error threshold in meters (smaller = more detail). Minimum: 0.00006m (0.06mm). |
| `-d, --max-depth` | `5` | Maximum refinement depth. Higher = more detail. Recommended: ≤15. |
| `-w, --window-size` | `512` | Attention window size for streaming RTIN. |
| `-o, --overlap` | `0.5` | Overlap ratio between windows (0.0-1.0). |

### Examples

**Standard terrain mesh (smooth hills):**
```bash
python generate_mesh.py input_folder output.stl \
    -r 5 -W 15 -L 15 -H 3 -B 0.5 -e 0.05 -d 10
```

**High-detail mesh with buildings and trees:**
```bash
python generate_mesh.py input_folder output.stl \
    -r 1 -W 15 -L 15 -H 3 -B 0.5 -e 0.01 -d 15
```

**Ultra-high precision mesh (0.06mm layer height printers):**
```bash
python generate_mesh.py input_folder output.stl \
    -r 1 -W 17.5 -L 17.5 -H 3 -B 0.5 -e 0.00006 -d 20
```

### Sampling Modes

The pipeline automatically selects the optimal sampling mode based on resolution:

- **Resolution ≤ 2m**: Uses **nearest neighbor** sampling to preserve sharp edges (buildings, trees, cliffs)
- **Resolution > 2m**: Uses **bilinear interpolation** for smooth terrain surfaces

This ensures that high-detail meshes retain every pixel from the source data, while coarse meshes benefit from smooth interpolation.

---

## 📚 Credits & Architecture

- **Martini Algorithm**: Based on the work of **Vladimir Agafonkin** for fast, T-junction-free terrain triangulation.
- **IGN Data**: Optimized for RGE ALTI® and other high-resolution French geographic datasets.
- **Numba**: JIT compilation used extensively for high-speed geospatial processing.

---

## 📝 Project Log
Detailed architectural changes and optimization history can be found in [PROJECT_LOG.md](PROJECT_LOG.md).
