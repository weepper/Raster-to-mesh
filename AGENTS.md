# AGENTS.md - Development Guidelines

This repository contains Python scripts for geospatial data processing:
- `download_tiles.py` - Downloads GeoTIFF files from IGN (French geographic institute)
- `generate_mesh.py` - Converts GeoTIFF terrain data to STL 3D mesh files
- `decimation.py` - Custom Numba-accelerated QEM mesh decimation

## Project Overview

The project processes elevation data (GeoTIFF format) to generate 3D printable STL meshes with memory-efficient streaming algorithms.

### Key Dependencies
- Python 3.11+
- numpy - Array processing
- numba - JIT compilation for RTIN and decimation kernels
- rasterio - GeoTIFF reading
- requests - HTTP downloads
- tqdm - Progress bars
- scipy (optional) - Binary dilation for NaN boundary handling

### Virtual Environment
```bash
# Activate the project's virtual environment
source venv/bin/activate

# Or use pyenv
pyenv local 3.11.9
```

## Commands

### Running Scripts

```bash
# Download tiles (from URL list file)
python download_tiles.py -i dalles.txt -d output_dir -w 10

# Generate mesh from GeoTIFF folder
python generate_mesh.py input_folder output.stl -r 1.0 -W 10 -L 10 -H 3 -B 0.5 -f 500000

# Run with specific Python
/home/kakou/Documents/python/venv/bin/python generate_mesh.py [args]
```

### Testing
There is no formal test suite. Test functionality manually:
```bash
# Quick syntax check
python -m py_compile script.py

# Check imports resolve correctly
python -c "import module_name"

# Run with verbose output
python -v script.py
```

### Linting (if configured)
```bash
# Black formatting check
black --check *.py

# Ruff linting
ruff check *.py

# mypy type checking
mypy *.py --ignore-missing-imports
```

## Code Style Guidelines

### General Style
- **Python version**: Target Python 3.11+ compatibility
- **Indentation**: 4 spaces (no tabs)
- **Line length**: ~100 characters max
- **No trailing whitespace**
- **Blank lines**: Two between top-level definitions, one between functions

### Imports
Standard library first, then third-party, then local:
```python
import os
import re
import argparse
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from tqdm import tqdm
```

### Type Hints
Use type hints for function signatures:
```python
def process_data(filepath: str, config: Config) -> np.ndarray:
def extract_links(filepath: str) -> list[str]:
def calculate(vertices: np.ndarray, indices: np.ndarray) -> float:
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `StreamingSTLWriter`, `MeshConfig`)
- Functions/variables: `snake_case` (e.g., `scan_tile_bounds`, `temp_path`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `STL_DTYPE`, `DEFAULT_CHUNK_SIZE`)
- Private methods: `_leading_underscore`
- Type variables: `PascalCase`

### Dataclasses for Configuration
Use `@dataclass` for configuration and data containers:
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

### Docstrings
Use docstrings for all public functions:
```python
def scan_tile_bounds(tif_files: list) -> GeoBounds:
    """Scan tiles for bounds without loading data."""
    ...
```

### Error Handling
- Use specific exception types
- Let exceptions propagate for unrecoverable errors
- Catch and log warnings for non-critical issues:
```python
try:
    with rasterio.open(fp) as src:
        data = src.read(1)
except rasterio.RasterioIOError as e:
    logger.warning(f"Could not read {fp}: {e}")
```

### Logging
Use `logging` module for application logging:
```python
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

logger.info("Processing phase complete")
logger.warning("Skipping invalid tile")
logger.error("Fatal error occurred")
```

Use `print()` only for CLI scripts in `__main__` blocks.

### Context Managers
Always use context managers for resource cleanup:
```python
with open(filepath, 'wb') as f:
    f.write(data)
    
with StreamingSTLWriter(output_file) as writer:
    writer.write_triangles(triangles)
```

### Memory Efficiency
- Use `np.memmap` for large arrays that don't fit in RAM
- Process data in chunks when possible
- Explicitly call `gc.collect()` after releasing large objects
- Use generators for iteration over large datasets

### Progress Indicators
Use `tqdm` for long-running operations:
```python
from tqdm import tqdm

for item in tqdm(data, desc="Processing", unit="item"):
    process(item)
```

### Array Processing
- Use numpy vectorized operations over Python loops
- Use structured dtypes for binary data (STL format):
```python
STL_DTYPE = np.dtype([
    ('normals', '<f4', (3,)),
    ('v1', '<f4', (3,)),
    ('v2', '<f4', (3,)),
    ('v3', '<f4', (3,)),
    ('attr', '<u2')
])
```

### CLI Arguments
Use `argparse` with `ArgumentDefaultsHelpFormatter`:
```python
parser = argparse.ArgumentParser(
    description="Description",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-r", "--resolution", type=float, default=1.0)
```

## File Structure

```
/home/kakou/Documents/python/
├── AGENTS.md             # This file
├── PROJECT_LOG.md        # Architecture notes and change log
├── download_tiles.py     # GeoTIFF downloader
├── generate_mesh.py      # GeoTIFF to STL converter (RTIN + streaming)
├── decimation.py         # Custom QEM mesh decimation (Numba)
├── venv/                 # Virtual environment
├── vllm_rocm/            # vllm installation (not project-related)
├── paru/                 # Arch Linux AUR helper (not project-related)
├── St_Lary_Soulan/       # Full sample input data
├── St_Lary_Soulan_small/ # Small sample input data
└── *.txt                 # URL lists for downloads
```

## Development Workflow

1. Activate virtual environment: `source venv/bin/activate`
2. Make changes to Python files
3. Test syntax: `python -m py_compile file.py`
4. Run with sample data
5. Verify output files are correct

## Common Issues

- **Memory errors**: Reduce `--chunk-size` parameter (smaller = less RAM per chunk)
- **Missing tiles**: Check URL list file format
- **Decimation fails**: Ensure numba is installed: `pip install numba`
- **Stale Numba cache**: Delete `__pycache__/` if JIT segfaults after signature changes
