# cctv-scrapers

A multi-stage pipeline for scraping CCTV camera product data from vendor sites,
building a structured dataset, validating it, and (in progress) identifying
cameras in real-world images using a YOLOv8-based detector and a catalog index.

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

4. **Stage 4 – Camera Identification & Retrieval (WIP)**
   - New package: `src/identification/`.
   - Goal: given an input image, detect cameras (YOLOv8), crop them, and retrieve
     the most likely catalog matches from `products.parquet`.
   - Current components:
     - `src/models/identification.py`: Pydantic models for
       `BoundingBox`, `CameraDetection`, `CameraMatch`, `RetrievalResult`.
     - `src/identification/detector.py`: `Detector` wrapper around a YOLOv8
       model (Ultralytics) that returns `CameraDetection` objects.
     - `src/identification/embeddings.py`, `src/identification/index.py`,
       `src/identification/service.py`, `src/identification/cli.py`:
       skeleton modules that will be filled in as the identification system
       is implemented.

The README will be updated as Stage 4 evolves (embeddings, index building,
retrieval service, and CLI).

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

### Stage 4 – Camera Identification & Retrieval (WIP)

The identification system is under active development.

Currently available:

- `Detector` in `src/identification/detector.py`:
  - Loads YOLOv8 weights using `src.config.get_yolo_camera_weights_path()`.
  - Exposes `detect_from_path(image_path)` which returns a list of
    `CameraDetection` objects (bounding box, confidence, label, class ID).

Planned components (not yet implemented):

- Image embedding model and catalog index over `image_files` from
  `output/products.parquet`.
- High-level `IdentificationService` that combines detection, embeddings, and
  nearest-neighbor search to produce `RetrievalResult` objects.
- CLI entry point (e.g. `identify.py`) for running identification on one or more
  input images.

As these pieces are implemented, this README will be extended with concrete
usage examples and commands.
