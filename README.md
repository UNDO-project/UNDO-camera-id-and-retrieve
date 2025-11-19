# cctv-scrapers

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

# 2. Install core dependencies
uv sync

# 3. Install identification dependencies (YOLOv8 + CLIP)
uv add ultralytics open-clip-torch torch

# 4. Configure YOLOv8 weights
cp .env-sample .env
# then edit .env and set CAMERA_DETECTOR_WEIGHTS to your .pt file, or
# place it at model_weights/yolov8_camera.pt to use the default.

# 5. Scrape Axis cameras (Stage 1)
python scrape.py

# 6. Build dataset (Stage 2)
python build_dataset.py

# 7. Build catalog embeddings (Stage 4 prep)
uv run python -c "from src.identification.index import build_catalog_embeddings; build_catalog_embeddings()"

# 8. Run identification on an image (Stage 4)
uv run python -m src.identification.cli \
  --image path/to/photo.jpg \
  --top-k 5 \
  --min-similarity 0.3
```

## Stages

The project is organized into sequential stages:

1. **Stage 1 – Scrape**
   - Entry points: `scrape.py`, `main.py`.
   - Scrapes Axis Communications network cameras (categories, series, products).
   - Downloads product images and datasheet PDFs.
   - Writes a verification manifest (`output/verification_manifest.json`) describing
     what was scraped.

2. **Stage 2 – Build dataset**
   - Entry point: `build_dataset.py`.
   - Reads the manifest and filesystem (`data/images`, `data/pdfs`).
   - Builds a tabular dataset (`output/products.parquet`) of `CameraRecord` rows
     (see `src/models/camera.py`).

3. **Stage 3 – Validate dataset**
   - Entry point: `validate_dataset.py` (CLI in `src/validation/cli.py`).
   - Performs multi-layer validation:
     - schema / structural checks,
     - file integrity (images / PDFs),
     - data quality & statistics,
     - comparison against the manifest.

4. **Stage 4 – Camera Identification & Retrieval**
   - New package: `src/identification/`.
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

### Stage 1 – Scrape Axis cameras

```bash
python scrape.py
```

This will populate `data/images`, `data/pdfs`, and generate
`output/verification_manifest.json`.

### Stage 2 – Build parquet dataset

```bash
python build_dataset.py
```

This reads the manifest and filesystem and produces `output/products.parquet`.

### Stage 3 – Validate dataset

```bash
python validate_dataset.py
```

This runs the validator CLI, which loads the parquet dataset and manifest,
performs checks, and prints a validation report.

### Stage 4 – Camera Identification & Retrieval

The identification system builds on the scraped dataset and catalog
embeddings to identify cameras in arbitrary input images.

Key entry points:

- `build_catalog_embeddings` in `src/identification/index.py`:
  - Builds `output/catalog_embeddings.npz` from `output/products.parquet`.
- `IdentificationService` in `src/identification/service.py`:
  - Combines YOLOv8 detection, cropping, CLIP embeddings, and catalog search.
- CLI in `src/identification/cli.py`:
  - Run via `python -m src.identification.cli --image path/to/photo.jpg`.

Example CLI usage:

```bash
# Build catalog embeddings once (after scraping and dataset build)
python -m src.identification.index  # or call build_catalog_embeddings from a small script

# Run identification on a single image
python -m src.identification.cli \
  --image path/to/photo.jpg \
  --top-k 5 \
  --min-similarity 0.3
```
