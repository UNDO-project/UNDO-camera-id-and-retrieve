# CCTV Identification and Research | (cIDaR)

A multi-stage pipeline for scraping CCTV camera product data from vendor sites,
building a structured dataset, validating it, and identifying
cameras in real-world images using a YOLOv8-based detector and a catalog index.

## Quickstart

This section shows the minimal set of commands to go from a fresh checkout
to running camera identification on a single image.

```bash
# 1. Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install core dependencies (includes console scripts)
uv sync

# 3. Install identification dependencies (YOLOv8 + CLIP)
uv add ultralytics open-clip-torch torch

# 4. Configure YOLOv8 weights
cp .env-sample .env
# then edit .env and set CAMERA_DETECTOR_WEIGHTS to your .pt file, or
# place it at model_weights/yolov8_camera.pt to use the default.

# 5. Scrape cameras (Stage 1)
# Scrape Axis cameras (default)
cidar-scrape

# Or scrape HikVision cameras
cidar-scrape --vendor hikvision

# Or scrape all vendors
cidar-scrape --vendor all

# 6. Build dataset (Stage 2)
cidar-build

# 7. Build catalog embeddings (Stage 4 prep)
uv run python -c "from src.identification.index import build_catalog_embeddings; build_catalog_embeddings()"

# 8. Run identification on an image (Stage 4)
cidar-identify \
  --image path/to/photo.jpg \
  --top-k 5 \
  --min-similarity 0.3
```

## Stages

The project is organized into sequential stages:

1. **Stage 1 – Scrape**
   - Entry point: `cidar-scrape` (console script from `src/scraping/main.py`)
   - Scrapes CCTV camera products from supported vendors:
     - **Axis Communications**: Network cameras (categories, series, products)
     - **HikVision**: Network cameras, PTZ cameras, and Explosion-Proof series
   - Downloads product images and datasheet PDFs.
   - Writes a verification manifest (`output/verification_manifest.json`) describing
     what was scraped.

2. **Stage 2 – Build dataset**
   - Entry point: `cidar-build` (console script from `src/building/main.py`)
   - Reads the manifest and filesystem (`data/images`, `data/pdfs`).
   - Builds a tabular dataset (`output/products.parquet`) of `CameraRecord` rows
     (see `src/models/camera.py`).
   - **New features:**
     - **Append mode**: Merge new data with existing datasets (`--append`)
     - **Dataset versioning**: Track dataset evolution over time (`--version-mode auto`)
     - **Merge strategies**: Handle duplicates with update/skip/error strategies

3. **Stage 3 – Validate dataset**
   - Entry point: `cidar-validate` (console script from `src/validation/cli.py`)
   - Performs multi-layer validation:
     - schema / structural checks,
     - file integrity (images / PDFs),
     - data quality & statistics,
     - comparison against the manifest.

4. **Stage 4 – Camera Identification & Retrieval**
   - Entry point: `cidar-identify` (console script from `src/identification/cli.py`)
   - Package: `src/identification/`
   - Goal: given an input image, detect cameras (YOLOv8), crop them, and retrieve
     the most likely catalog matches from `products.parquet`.
   - Current components:
     - `src/models/identification.py`: Pydantic models for
       `BoundingBox`, `CameraDetection`, `CameraMatch`, `RetrievalResult`.
     - `src/identification/detector.py`: `Detector` wrapper around a YOLOv8
       model (Ultralytics) that returns `CameraDetection` objects.
     - `src/identification/catalog.py`: helpers for loading `CameraRecord` and
       iterating over reference image files.
     - `src/identification/embeddings.py`: CLIP-based image embeddings with
       CUDA/MPS/CPU support.
     - `src/identification/index.py`: offline catalog embedding builder and
       in-memory cosine-similarity index (`CatalogIndex`).
     - `src/identification/service.py`: `IdentificationService` orchestration
       of detection, cropping, embedding, and retrieval.
     - `src/identification/cli.py`: CLI wrapper around `IdentificationService`.

## Setup

This project targets Python 3.12 and uses `uv` for environment and dependency
management.

1. Create and activate a virtual environment:

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Add the YOLOv8 runtime for the detector (once):

   ```bash
   uv add ultralytics
   ```

## Environment configuration (.env)

Environment variables are loaded via `python-dotenv`. A sample configuration is
provided in `.env-sample`.

1. Copy the sample file:

   ```bash
   cp .env-sample .env
   ```

2. Edit `.env` and set the path to your trained YOLOv8 camera detector weights:

   ```bash
   # Path to YOLOv8 camera detector weights (.pt file)
   CAMERA_DETECTOR_WEIGHTS=/absolute/or/project/relative/path/to/model_weights/yolov8_camera.pt
   ```

If `CAMERA_DETECTOR_WEIGHTS` is not set, the code will fall back to the default
project-relative path `model_weights/yolov8_camera.pt`. The `model_weights/` directory is
created automatically at runtime.

## Running the pipeline

### Stage 1 – Scrape cameras

Scrape Axis Communications cameras (default):
```bash
cidar-scrape
```

Or scrape HikVision cameras:
```bash
cidar-scrape --vendor hikvision
```

Or scrape all vendors:
```bash
cidar-scrape --vendor all
```

This will populate `data/images`, `data/pdfs`, and generate
`output/verification_manifest.json`.

**Supported vendors:**
- `axis`: Axis Communications (Network cameras)
- `hikvision`: HikVision (Network cameras, PTZ cameras, Explosion-Proof series)
- `all`: Scrape all supported vendors sequentially

### Stage 2 – Build parquet dataset

**Basic usage:**
```bash
cidar-build
```

This reads the manifest and filesystem and produces `output/products.parquet`.

**Advanced options:**

**Append mode** - Merge new data with existing dataset:
```bash
# Append with update strategy (overwrites duplicates)
cidar-build --append --merge-strategy update

# Append with skip strategy (keeps original data)
cidar-build --append --merge-strategy skip

# Append with error strategy (fails on duplicates)
cidar-build --append --merge-strategy error
```

**Dataset versioning** - Track dataset evolution over time:
```bash
# Build with auto-versioning (creates products_v1.parquet)
cidar-build --version-mode auto

# Append and create new version
cidar-build --append --version-mode auto

# List all dataset versions
cidar-build --list-versions

# Clean up old versions (keep last 3)
cidar-build --cleanup-versions 3
```

**Version structure:**
```
output/
├── products.parquet               # Symlink → products_v3.parquet
├── products_v1.parquet            # Version 1 snapshot
├── products_v2.parquet            # Version 2 snapshot
├── products_v3.parquet            # Version 3 snapshot (current)
├── verification_manifest.json     # Symlink → verification_manifest_v3.json
├── verification_manifest_v1.json  # Version 1 manifest
├── verification_manifest_v2.json  # Version 2 manifest
├── verification_manifest_v3.json  # Version 3 manifest
└── dataset_metadata.json          # Version history and metadata
```

### Stage 3 – Validate dataset

**Basic usage:**
```bash
# Validate current dataset (symlink)
cidar-validate
```

This runs the validator CLI, which loads the parquet dataset and manifest,
performs checks, and prints a validation report.

**Version-aware validation:**
```bash
# Validate specific version
cidar-validate --version 2

# Validate version with verbose output
cidar-validate --version 1 --verbose
```

Version-aware validation reports include:
- Version number and timestamp
- Record count and manifest hash
- Append mode details (if applicable)
- Parent version tracking

### Stage 4 – Camera Identification & Retrieval

The identification system builds on the scraped dataset and catalog
embeddings to identify cameras in arbitrary input images.

Key entry points:

- `build_catalog_embeddings` in `src/identification/index.py`:
  - Builds `output/catalog_embeddings.npz` from `output/products.parquet`.
- `IdentificationService` in `src/identification/service.py`:
  - Combines YOLOv8 detection, cropping, CLIP embeddings, and catalog search.
- CLI: `cidar-identify` (console script)

Example CLI usage:

```bash
# Build catalog embeddings once (after scraping and dataset build)
python -c "from src.identification.index import build_catalog_embeddings; build_catalog_embeddings()"

# Run identification on a single image
cidar-identify \
  --image path/to/photo.jpg \
  --top-k 5 \
  --min-similarity 0.3
```

## API Server

The project includes a FastAPI-based REST API that wraps the identification service,
enabling remote access for frontend applications and integrations.

### Starting the API server

**Basic usage:**
```bash
cidar-api
```

The API server will start on `http://localhost:8000` by default.

**With custom configuration:**
```bash
# Development mode with auto-reload
CIDAR_API_RELOAD=true cidar-api

# Custom host and port
CIDAR_API_HOST=0.0.0.0 CIDAR_API_PORT=9000 cidar-api

# Custom CORS origins
CIDAR_API_CORS_ORIGINS='["http://localhost:3000","https://myapp.com"]' cidar-api
```

### Available endpoints

**Health & Status:**
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/health/ready` - Readiness check (catalog loaded, service ready)

**Identification:**
- `POST /api/v1/identify` - Upload an image and identify cameras
  - Parameters: `top_k` (default: 5), `min_similarity` (default: 0.3)
  - Request: `multipart/form-data` with image file
  - Response: Detection results with matched cameras

**Catalog:**
- `GET /api/v1/catalog/stats` - Get catalog statistics
- `POST /api/v1/catalog/reload` - Reload catalog embeddings (admin)

**Documentation:**
- `GET /` - API information
- `GET /docs` - Interactive Swagger UI documentation
- `GET /redoc` - ReDoc documentation

### Example API usage

**Using curl:**
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Identify cameras in an image
curl -X POST "http://localhost:8000/api/v1/identify?top_k=5&min_similarity=0.3" \
  -F "image=@path/to/photo.jpg"

# Get catalog statistics
curl http://localhost:8000/api/v1/catalog/stats
```

**Using Python requests:**
```python
import requests

# Identify cameras
with open("photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/identify",
        files={"image": f},
        params={"top_k": 5, "min_similarity": 0.3}
    )
    results = response.json()
    print(f"Found {results['detections_count']} cameras")
```

**Using JavaScript fetch:**
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

const response = await fetch('http://localhost:8000/api/v1/identify?top_k=5', {
  method: 'POST',
  body: formData
});

const results = await response.json();
console.log(`Found ${results.detections_count} cameras`);
```

### API configuration

Configure the API server using environment variables with the `CIDAR_API_` prefix:

| Variable | Description | Default |
|----------|-------------|---------|
| `CIDAR_API_HOST` | Host to bind to | `0.0.0.0` |
| `CIDAR_API_PORT` | Port to listen on | `8000` |
| `CIDAR_API_RELOAD` | Enable auto-reload (development) | `false` |
| `CIDAR_API_WORKERS` | Number of worker processes | `1` |
| `CIDAR_API_CORS_ORIGINS` | Allowed CORS origins (JSON array) | `["http://localhost:3000", "http://localhost:5173"]` |
| `CIDAR_API_LOG_LEVEL` | Logging level | `INFO` |
| `CIDAR_API_MAX_UPLOAD_SIZE_MB` | Maximum upload size in MB | `10` |

### Interactive API documentation

Once the API server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive documentation where you can test all endpoints directly from your browser.

## Building the documentation

The project uses Sphinx to generate HTML documentation from docstrings and
reStructuredText files.

### Build documentation

From the project root:

```bash
cd docs
make html
```

The generated documentation will be in `docs/_build/html/index.html`. Open this
file in your browser to view the docs.

### Clean build

To remove all build artifacts and rebuild from scratch:

```bash
cd docs
make clean
make html
```

This is useful when:
- You've made structural changes to the documentation
- You want to ensure there are no stale artifacts
- You're troubleshooting build warnings or errors

### Verify build quality

To check for warnings or errors during the build:

```bash
cd docs
make clean && make html 2>&1 | grep -i "warning\|error"
```

A successful build should produce no warnings or errors.
