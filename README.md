# Camera Identification and Research | (cIDaR)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/python-3.12+-green)



A multi-stage pipeline for scraping CCTV camera product data from vendor sites,
building a structured dataset, validating it, and identifying cameras in real-world
images using a YOLOv8-based detector and CLIP-based catalog matching. Includes a
REST API with real-time WebSocket video streaming and batch video processing capabilities.

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

# 9. (Optional) Start the API server for remote access
cidar-api
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
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
     - **Manifest reconstruction**: Rebuild manifest from cache without rescraping (`cidar-rebuild-manifest`)

3. **Stage 3 – Validate dataset**
   - Entry point: `cidar-validate` (console script from `src/validation/cli.py`)
   - Performs multi-layer validation:
     - schema / structural checks,
     - file integrity (images / PDFs),
     - data quality & statistics,
     - comparison against the manifest.

4. **Stage 4 – Camera Identification & Retrieval**
   - Entry point: `cidar-identify` (console script from `src/identification/cli.py`)
   - API entry point: `cidar-api` (FastAPI server from `src/api/main.py`)
   - Package: `src/identification/`, `src/api/`
   - Goal: given an input image or video, detect cameras (YOLOv8), crop them, and retrieve
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
     - `src/api/`: FastAPI REST API with WebSocket video streaming, batch video processing,
       and catalog browsing endpoints.

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

#### Manifest Reconstruction (Recovery Tool)

If the verification manifest is missing or corrupted but you still have the downloaded data and cache, you can reconstruct the manifest without rescraping:

```bash
# Reconstruct manifest from filesystem and cache
cidar-rebuild-manifest

# Force overwrite existing manifest
cidar-rebuild-manifest --force
```

**When to use:**
- Manifest file (`verification_manifest.json`) was deleted or corrupted
- Scraping completed but manifest creation failed
- Need to regenerate manifest from existing data

**How it works:**
1. Scans filesystem structure in `data/images/` and `data/pdfs/`
2. Loads product metadata from `output/download_cache.db`
3. Matches downloaded files with cached metadata
4. Reconstructs the manifest with file counts and statistics

**Example output:**
```
INFO: Found 2 categories with 15 series and 1517 products
INFO: Loaded 1678 products from cache
INFO: Built manifest with 1517 products, 3663 images, 1402 PDFs
SUCCESS: Reconstructed manifest saved to output/verification_manifest.json
```

After reconstruction, proceed with the normal build:
```bash
cidar-build
```

**Note:** This tool prevents having to rescrape all data when only the manifest is missing, saving significant time and network bandwidth.

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

#### ⚠️ Important: Catalog Embeddings

**Embeddings are NOT automatically built by `cidar-build`**. You must build them manually as a separate step using the Python function below.

**When to rebuild embeddings:**
- ✅ After running `cidar-build` for the first time
- ✅ After appending new cameras via `cidar-build --append`
- ✅ After restoring or replacing `output/products.parquet`
- ✅ When the API returns "missing record data" errors

**Why embeddings are separate:**
- Computing CLIP embeddings is expensive (~40 seconds for 1500 cameras)
- Only needed for identification, not for scraping/building
- Allows rebuilding catalog metadata without recomputing embeddings

**What happens if embeddings are out of sync:**
If `catalog_embeddings.npz` contains camera IDs that don't exist in `products.parquet`, the identification API will fail with "missing record data" errors. Always rebuild embeddings after modifying the catalog.

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

The project includes a FastAPI-based REST API that provides:
- **Image identification**: Upload images for camera detection and catalog matching
- **Real-time video streaming**: WebSocket endpoint for live video frame processing
- **Video batch processing**: Upload videos for background processing with progress tracking
- **Catalog browsing**: Search, filter, and paginate through the camera catalog
- **Static file serving**: Direct access to camera images and PDF datasheets

This enables frontend applications, mobile apps, and third-party integrations to access the camera identification system remotely.

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

**Video Streaming (WebSocket):**
- `WS /api/v1/ws/video-stream` - Real-time video frame processing
  - Query params: `mode` (detect_only|full), `target_fps` (1-30)
  - Accepts JPEG frames, returns annotated frames + JSON metadata
  - Two modes: detection-only or full identification with catalog matching

**Video Upload (Batch Processing):**
- `POST /api/v1/process-video` - Upload video for background processing
  - Accepts: MP4, AVI, MOV, WebM (up to 500MB, 30 min duration)
  - Returns task ID for tracking progress
- `GET /api/v1/video-status/{task_id}` - Get processing task status
- `GET /api/v1/video-download/{task_id}` - Download processed video
- `DELETE /api/v1/video-task/{task_id}` - Delete task and files

**Catalog:**
- `GET /api/v1/catalog/stats` - Get catalog statistics
- `GET /api/v1/catalog/cameras` - List cameras with pagination and filters
  - Query params: `page`, `limit`, `vendor`, `category`, `series`, `search`
- `GET /api/v1/catalog/cameras/{camera_id}` - Get camera details by ID
- `GET /api/v1/catalog/facets` - Get available filter options with counts
- `POST /api/v1/catalog/reload` - Reload catalog embeddings (admin)

**Static Files:**
- `GET /api/v1/images/{vendor}/{series}/{filename}` - Serve camera images (CORS enabled)
- `GET /api/v1/datasheets/{vendor}/{series}/{filename}` - Serve PDF datasheets (CORS enabled)

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

### Real-Time Video Streaming (WebSocket)

The API supports real-time video frame processing via WebSocket for live camera feeds or video playback. This is ideal for browser-based applications that need low-latency camera detection.

**Features:**
- **Two processing modes:**
  - `detect_only`: Fast detection with bounding boxes only
  - `full`: Detection + catalog matching with similarity scores
- **Configurable FPS**: Target processing rate (1-30 FPS)
- **Binary protocol**: Send JPEG frames, receive annotated frames + JSON metadata
- **Automatic frame skipping**: Maintains target FPS under load

**JavaScript example:**
```javascript
// Connect to WebSocket endpoint
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/video-stream?mode=detect_only&target_fps=15');
ws.binaryType = 'arraybuffer';

// Handle connection open
ws.onopen = () => {
  console.log('WebSocket connected');
};

// Send video frames (from canvas, video element, or MediaStream)
function sendFrame(videoElement) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0);

  canvas.toBlob((blob) => {
    ws.send(blob);
  }, 'image/jpeg', 0.85);
}

// Receive results
ws.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    // Annotated frame (binary JPEG)
    const blob = new Blob([event.data], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    document.getElementById('output').src = url;
  } else {
    // Metadata (JSON)
    const metadata = JSON.parse(event.data);
    console.log(`Frame ${metadata.frame_number}: ${metadata.detections.length} cameras detected`);
    console.log(`Processing time: ${metadata.processing_time_ms}ms, FPS: ${metadata.fps_actual}`);

    // Access detection data
    metadata.detections.forEach(det => {
      console.log(`  Camera: confidence=${det.confidence}, bbox=[${det.bbox}]`);
      if (det.matches) {
        det.matches.forEach(match => {
          console.log(`    Match: ${match.model_name} (${match.similarity})`);
        });
      }
    });
  }
};

// Handle errors and disconnection
ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket disconnected');
};
```

**Python example (using websockets library):**
```python
import asyncio
import websockets
import cv2
import json

async def stream_video():
    uri = "ws://localhost:8000/api/v1/ws/video-stream?mode=full&target_fps=10"

    async with websockets.connect(uri) as ws:
        cap = cv2.VideoCapture("video.mp4")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()

            # Send frame
            await ws.send(frame_bytes)

            # Receive annotated frame (binary)
            annotated_bytes = await ws.recv()

            # Receive metadata (text/JSON)
            metadata_json = await ws.recv()
            metadata = json.loads(metadata_json)

            print(f"Frame {metadata['frame_number']}: {len(metadata['detections'])} cameras")

        cap.release()

asyncio.run(stream_video())
```

### Video Upload & Batch Processing

For processing pre-recorded videos, the API provides asynchronous batch processing with task tracking. Upload a video file and retrieve the processed result when ready.

**Features:**
- Supports MP4, AVI, MOV, WebM formats
- Configurable processing options (FPS, detection mode)
- Background processing with progress tracking
- Automatic cleanup after 24 hours

**Upload a video for processing:**
```bash
# Basic upload
curl -X POST "http://localhost:8000/api/v1/process-video" \
  -F "file=@surveillance_footage.mp4"

# With custom options
curl -X POST "http://localhost:8000/api/v1/process-video" \
  -F "file=@surveillance_footage.mp4" \
  -F 'options={"target_fps": 10, "identification_mode": "full", "output_format": "mp4"}'

# Response:
{
  "task_id": "abc-123-def-456",
  "status": "queued",
  "filename": "surveillance_footage.mp4",
  "file_size_mb": 45.2,
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Check processing status:**
```bash
curl "http://localhost:8000/api/v1/video-status/abc-123-def-456"

# Response (in progress):
{
  "task_id": "abc-123-def-456",
  "status": "processing",
  "progress_percent": 67.5,
  "frames_processed": 1350,
  "frames_total": 2000,
  "current_fps": 12.3,
  "elapsed_seconds": 110.5,
  "output_url": null,
  "filename": "surveillance_footage.mp4",
  "options": {
    "target_fps": 10,
    "identification_mode": "full",
    "output_format": "mp4"
  }
}

# Response (completed):
{
  "task_id": "abc-123-def-456",
  "status": "completed",
  "progress_percent": 100.0,
  "frames_processed": 2000,
  "frames_total": 2000,
  "output_url": "/api/v1/video-download/abc-123-def-456",
  ...
}
```

**Download processed video:**
```bash
curl -O "http://localhost:8000/api/v1/video-download/abc-123-def-456"
# Downloads: surveillance_footage_processed.mp4
```

**Delete task and files:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/video-task/abc-123-def-456"
```

**Python example:**
```python
import requests
import time

# Upload video
with open("video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/process-video",
        files={"file": f},
        data={"options": '{"target_fps": 15, "identification_mode": "detect_only"}'}
    )
    task = response.json()
    task_id = task["task_id"]
    print(f"Task created: {task_id}")

# Poll for completion
while True:
    response = requests.get(f"http://localhost:8000/api/v1/video-status/{task_id}")
    status = response.json()

    print(f"Progress: {status['progress_percent']:.1f}% ({status['frames_processed']}/{status['frames_total']} frames)")

    if status["status"] == "completed":
        print("Processing complete!")
        break
    elif status["status"] == "failed":
        print(f"Processing failed: {status['error']}")
        break

    time.sleep(2)

# Download result
response = requests.get(f"http://localhost:8000/api/v1/video-download/{task_id}")
with open("output_processed.mp4", "wb") as f:
    f.write(response.content)
print("Downloaded processed video")
```

### Catalog Browsing & Search

The API provides rich catalog endpoints for building frontend UIs with filtering, pagination, and search.

**List cameras with filters:**
```bash
# Get first page (20 cameras)
curl "http://localhost:8000/api/v1/catalog/cameras?page=1&limit=20"

# Filter by vendor
curl "http://localhost:8000/api/v1/catalog/cameras?vendor=Axis&page=1"

# Filter by category and series
curl "http://localhost:8000/api/v1/catalog/cameras?category=Network%20cameras&series=AXIS%20M30&page=1"

# Search by model name
curl "http://localhost:8000/api/v1/catalog/cameras?search=PTZ&page=1"

# Response:
{
  "data": [
    {
      "id": "axis-m3057-plr-mk-ii",
      "model_name": "AXIS M3057-PLRVE Mk II",
      "display_name": "AXIS M3057-PLRVE Mk II Network Camera",
      "source": "Axis",
      "category": "Network cameras",
      "series": "AXIS M30 Series",
      "image_urls": [
        "/api/v1/images/Axis/AXIS_M30_Series/axis-m3057-plr-mk-ii_1.jpg"
      ],
      "datasheet_url": "/api/v1/datasheets/Axis/AXIS_M30_Series/axis-m3057-plr-mk-ii.pdf"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 1517,
    "total_pages": 76
  },
  "facets": {
    "vendors": [{"value": "Axis", "count": 892}, {"value": "HikVision", "count": 625}],
    "categories": [{"value": "Network cameras", "count": 1245}, ...],
    "series": [{"value": "AXIS M30 Series", "count": 45}, ...]
  }
}
```

**Get camera details:**
```bash
curl "http://localhost:8000/api/v1/catalog/cameras/axis-m3057-plr-mk-ii"

# Response includes full specifications, all images, datasheet, etc.
```

**Get filter facets:**
```bash
curl "http://localhost:8000/api/v1/catalog/facets"

# Returns available vendors, categories, and series with counts
```

### API configuration

Configure the API server using environment variables with the `CIDAR_API_`, `CIDAR_VIDEO_`, and `CIDAR_STREAM_` prefixes:

**API Server Settings** (`CIDAR_API_` prefix):

| Variable | Description | Default |
|----------|-------------|---------|
| `CIDAR_API_HOST` | Host to bind to | `0.0.0.0` |
| `CIDAR_API_PORT` | Port to listen on | `8000` |
| `CIDAR_API_RELOAD` | Enable auto-reload (development) | `false` |
| `CIDAR_API_WORKERS` | Number of worker processes | `1` |
| `CIDAR_API_CORS_ORIGINS` | Allowed CORS origins (JSON array) | `["http://localhost:3000", "http://localhost:5173"]` |
| `CIDAR_API_LOG_LEVEL` | Logging level | `INFO` |
| `CIDAR_API_MAX_UPLOAD_SIZE_MB` | Max image upload size in MB | `10` |

**Video Upload Settings** (`CIDAR_VIDEO_` prefix):

| Variable | Description | Default |
|----------|-------------|---------|
| `CIDAR_VIDEO_MAX_UPLOAD_SIZE_MB` | Max video upload size in MB | `500` |
| `CIDAR_VIDEO_MAX_DURATION_MINUTES` | Max video duration in minutes | `30` |
| `CIDAR_VIDEO_DEFAULT_TARGET_FPS` | Default processing FPS | `15` |
| `CIDAR_VIDEO_TASK_CLEANUP_HOURS` | Hours before task cleanup | `24` |

**Video Streaming Settings** (`CIDAR_STREAM_` prefix):

| Variable | Description | Default |
|----------|-------------|---------|
| `CIDAR_STREAM_MAX_CONNECTIONS` | Max concurrent WebSocket connections | `5` |
| `CIDAR_STREAM_MAX_FPS` | Maximum allowed FPS | `30` |
| `CIDAR_STREAM_DEFAULT_FPS` | Default FPS if not specified | `15` |
| `CIDAR_STREAM_FRAME_TIMEOUT_SECONDS` | Frame processing timeout | `5.0` |
| `CIDAR_STREAM_INPUT_BUFFER_SIZE` | Input frame buffer size | `10` |
| `CIDAR_STREAM_JPEG_QUALITY` | JPEG encoding quality (1-100) | `85` |

### Interactive API documentation

Once the API server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive documentation where you can test all endpoints directly from your browser.

## Docker Deployment

The API can be deployed using Docker for simplified setup and consistent environments across development and production.

### Prerequisites

Before building the Docker image, ensure these files exist:

```bash
# Required files for the API to function
ls -lh model_weights/yolov8_camera.pt    # YOLO detector weights
ls -lh output/products.parquet            # Camera catalog dataset
ls -lh output/catalog_embeddings.npz      # Precomputed CLIP embeddings
```

If any are missing, run the pipeline stages first:

```bash
# Scrape and build dataset
cidar-scrape && cidar-build

# Build catalog embeddings
python -c "from src.identification.index import build_catalog_embeddings; build_catalog_embeddings()"
```

### Quick Start with Docker Compose

The simplest way to run the API in Docker:

```bash
# Build the Docker image (~5-10 minutes first time)
docker build -t cidar-api:latest .

# Start the container
docker-compose up -d

# Watch logs (CLIP model downloads on first startup, ~60 seconds)
docker-compose logs -f cidar-api

# Test the API
curl http://localhost:8000/api/v1/health

# Access interactive docs
open http://localhost:8000/docs
```

### Docker Image Details

**Architecture:**
- Multi-stage build using Python 3.12-slim base image
- Uses `uv` package manager for fast, reproducible builds
- Non-root user (UID 1000) for security
- CPU-only PyTorch (no GPU dependencies)

**Image size:**
- Base image: ~2.1 GB (includes PyTorch and dependencies)
- CLIP models: +350 MB (downloaded at first startup, cached thereafter)

**Startup behavior:**
- **First startup**: ~60 seconds (CLIP model downloads from HuggingFace)
- **Subsequent startups**: ~5 seconds (models load from cache)

### Volume Mounts

The docker-compose.yml configures three volume mounts:

```yaml
volumes:
  - ./model_weights:/app/model_weights:ro    # YOLO weights (read-only)
  - ./output:/app/output:rw                  # Catalog + saved crops (read-write)
  - ./data:/app/data:ro                      # Reference images (read-only)
```

### Configuration

Configure the containerized API using environment variables in `docker-compose.yml`:

```yaml
environment:
  - CIDAR_API_HOST=0.0.0.0
  - CIDAR_API_PORT=8000
  - CIDAR_API_LOG_LEVEL=INFO
  - CIDAR_API_CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
  - CIDAR_API_MAX_UPLOAD_SIZE_MB=10
```

Or create a `.env` file in the project root (docker-compose automatically loads it).

### Resource Limits

The default docker-compose.yml sets resource limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
    reservations:
      cpus: '2'
      memory: 2G
```

Adjust these based on your system's capabilities and expected load.

### Health Checks

The container includes automatic health monitoring:

```bash
# Check container health status
docker-compose ps

# Container is healthy when this returns {"status":"healthy"}
curl http://localhost:8000/api/v1/health

# Check detailed readiness status
curl http://localhost:8000/api/v1/health/ready
```

### Common Docker Commands

```bash
# Start container in foreground (see logs)
docker-compose up

# Start container in background
docker-compose up -d

# View logs
docker-compose logs -f cidar-api

# Stop container
docker-compose down

# Rebuild image after code changes
docker-compose build

# Restart container
docker-compose restart cidar-api

# Execute commands inside container
docker-compose exec cidar-api bash

# Check resource usage
docker stats cidar-api
```

### Development Mode

For development with hot-reload, uncomment the source volume mount in `docker-compose.yml`:

```yaml
volumes:
  - ./src:/app/src:ro  # Uncomment this line

environment:
  - CIDAR_API_RELOAD=true  # Enable auto-reload
```

Then rebuild and restart:

```bash
docker-compose up --build
```

### Production Deployment

For production deployments:

1. **Use a specific version tag** instead of `latest`:
   ```bash
   docker build -t cidar-api:v1.0.0 .
   ```

2. **Disable development features** in docker-compose.yml:
   ```yaml
   environment:
     - CIDAR_API_RELOAD=false
     - CIDAR_API_LOG_LEVEL=INFO
   ```

3. **Set appropriate resource limits** based on expected load

4. **Configure CORS** for your production domains:
   ```yaml
   environment:
     - CIDAR_API_CORS_ORIGINS=["https://yourdomain.com"]
   ```

5. **Consider using Docker secrets** for sensitive configuration

## Testing and Code Quality

The project uses pytest for testing, ruff for linting/formatting, and pytest-cov for code coverage tracking.

### Running Tests

**Basic test run:**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_identification_service.py

# Run tests matching a pattern
pytest -k "test_detector"

# Run with verbose output
pytest -v
```

### Code Coverage

**Run tests with coverage:**
```bash
# Terminal report with missing lines
pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Both terminal and HTML reports
pytest --cov=src --cov-report=html --cov-report=term-missing

# JSON report (for tooling/CI)
pytest --cov=src --cov-report=json
```

**View HTML coverage report:**
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

**Coverage configuration:**

Coverage settings are configured in `pyproject.toml`:
- Source: `src/` directory
- Omitted: `tests/`, `__init__.py`, `conftest.py`
- Reports: Terminal (with missing lines) and HTML
- HTML output: `htmlcov/` directory

### Linting and Formatting

**Lint with ruff:**
```bash
# Check all files
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/
```

**Format with ruff:**
```bash
# Format all files
ruff format src/ tests/

# Check formatting without making changes
ruff format --check src/ tests/
```

### Pre-commit Hooks

The project uses pre-commit hooks to automatically run checks before commits.

**Run hooks manually:**
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run all hooks on staged files
pre-commit run

# Run specific hook
pre-commit run ruff-format --all-files
```

**Install hooks (optional - runs automatically on git commit):**
```bash
pre-commit install
```

### Coverage Targets

Current focus areas for improving coverage:
1. **API layer** (`src/api/`) - Write integration tests for FastAPI endpoints
2. **Manifest reconstruction** (`src/building/manifest_reconstruction.py`) - Test filesystem scanning and cache loading
3. **CLI entry points** - Test command-line interfaces
4. **Scrapers** - Add tests for vendor-specific extraction logic

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
