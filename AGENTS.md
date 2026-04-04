# AGENTS.md - Development Guidelines

Geospatial data processing project for converting GeoTIFF elevation data to 3D printable STL meshes.

## Commands

```bash
# Virtual environment
source venv/bin/activate

# Download tiles
python download_tiles.py -i dalles.txt -d output_dir -w 10

# Generate mesh
python generate_mesh.py input_folder output.stl -r 1.0 -W 10 -L 10 -H 3 -B 0.5

# Testing
python -m py_compile generate_mesh.py decimation.py download_tiles.py
python -c "import generate_mesh; import decimation; import download_tiles"
python generate_mesh.py St_Lary_Soulan_small/ test.stl -r 1.0 -W 10 -L 10 -H 3 -B 0.5

# Linting
black *.py
ruff check *.py
mypy *.py --ignore-missing-imports
```

## Code Style

### Python Version
Target Python 3.11+ compatibility.

### Formatting
- **Indentation**: 4 spaces (no tabs)
- **Line length**: ~100 characters
- **No trailing whitespace**
- **Blank lines**: Two between top-level definitions, one between methods

### Imports (Grouped)
```python
# 1. Standard library
import os
import re
import argparse
from dataclasses import dataclass
from collections import defaultdict

# 2. Third-party
import numpy as np
from numba import njit
import rasterio
from tqdm import tqdm
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `StreamingSTLWriter` |
| Functions/Variables | snake_case | `scan_tile_bounds` |
| Constants | UPPER_SNAKE_CASE | `STL_DTYPE` |
| Private methods | _leading_underscore | `_recompute_normals` |
| Type variables | PascalCase | `T`, `Config` |

### Type Hints
Use for all function signatures:
```python
def process_data(filepath: str, config: MeshConfig) -> np.ndarray:
def calculate(vertices: np.ndarray, indices: np.ndarray) -> float:
```

### Docstrings
Required for all public functions:
```python
def scan_tile_bounds(tif_files: list) -> GeoBounds:
    """Scan tiles for bounds without loading data."""
```

### Error Handling
- Use specific exception types
- Let exceptions propagate for unrecoverable errors
- Log warnings for non-critical issues:
```python
try:
    with rasterio.open(fp) as src:
        data = src.read(1)
except rasterio.RasterioIOError as e:
    logger.warning(f"Could not read {fp}: {e}")
```

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

logger.info("Processing complete")
logger.warning("Skipping invalid tile")
```

Use `print()` only in CLI `__main__` blocks.

### Memory Efficiency
- Use `np.memmap` for large arrays
- Process in chunks when possible
- Call `gc.collect()` after releasing large objects
- Use generators for large dataset iteration

### Numba JIT
Use for performance-critical kernels:
```python
@njit(inline="always")
def next_c(c):
    return 3 * (c // 3) + ((c + 1) % 3)
```

### Dataclasses for Config
```python
@dataclass
class MeshConfig:
    width_cm: float
    length_cm: float
    height_cm: float
    base_cm: float
    target_faces: int
    resolution: float
```

### CLI Arguments
```python
parser = argparse.ArgumentParser(
    description="Description",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-r", "--resolution", type=float, default=1.0)
```

## Project Structure

```
/home/kakou/Documents/python/
├── AGENTS.md              # This file
├── PROJECT_LOG.md         # Architecture notes
├── README.md              # User documentation
├── download_tiles.py      # GeoTIFF downloader
├── generate_mesh.py       # Main STL generator (RTIN + streaming)
├── decimation.py          # QEM mesh decimation (Numba)
├── venv/                  # Virtual environment
├── St_Lary_Soulan/        # Full sample data
├── St_Lary_Soulan_small/  # Small sample data
└── *.txt                  # URL lists
```

## Key Dependencies
- `numpy` - Array processing
- `numba` - JIT compilation
- `rasterio` - GeoTIFF reading
- `requests` - HTTP downloads
- `tqdm` - Progress bars
- `scipy` (optional) - Binary dilation

## Common Issues

- **Memory errors**: Reduce `--window-size` or increase `--overlap`
- **Missing tiles**: Check URL list format
- **Stale Numba cache**: Delete `__pycache__/` if JIT segfaults
- **Integer overflow**: Avoid extreme `-d` values (>30) without testing
