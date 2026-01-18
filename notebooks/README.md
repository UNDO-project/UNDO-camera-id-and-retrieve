# CCTV Scrapers Notebooks

This directory contains Jupyter notebooks for interactive demonstrations and experiments with the CCTV scrapers pipeline.

## ⚠️ Required Setup Before Running Notebooks

**IMPORTANT:** All notebooks require setup before they can run. You must copy data files to `notebooks/data/` or notebooks will fail with `FileNotFoundError`.

### Prerequisites

1. **You must have already run the pipeline** (Stages 1-2) to generate the required files:
   - `output/products.parquet` - Dataset file
   - `output/verification_manifest.json` - Manifest file

2. **If you don't have these files**, generate them first:
   ```bash
   # Stage 1: Scrape vendor data
   cidar-scrape

   # Stage 2: Build dataset
   cidar-build
   ```


### Setup Steps

**Step 1: Create the notebooks data directory**
```bash
mkdir -p notebooks/data
```

**Step 2: Copy required files**
```bash
# Copy dataset file
cp output/products.parquet notebooks/data/

# Copy manifest file
cp output/verification_manifest.json notebooks/data/
```

**Step 3: Verify setup was successful**
```bash
# Check that files exist
ls -lh notebooks/data/

# Or use these verification commands
test -f notebooks/data/products.parquet && echo "✓ Dataset ready" || echo "✗ Missing dataset - see setup instructions"
test -f notebooks/data/verification_manifest.json && echo "✓ Manifest ready" || echo "✗ Missing manifest - see setup instructions"
```

### Why This Setup Is Required

- **Safety:** Notebooks work on copies in `notebooks/data/` to prevent modifying your production data in `output/`
- **Isolation:** Each notebook creates, modifies, and deletes test files without affecting your main dataset
- **Experimentation:** You can safely test append modes, versioning, and validation without risk

### Troubleshooting

**Problem:** `FileNotFoundError: 'notebooks/data/products.parquet'`
**Solution:** Follow setup steps above to copy files to `notebooks/data/`

**Problem:** Files don't exist in `output/` directory
**Solution:** Run the pipeline first (see Prerequisites above)

**Problem:** Notebooks fail even after copying files
**Solution:** Verify you're running Jupyter from the project root, not from `notebooks/` directory

---

## Available Notebooks

### dataset_append_demo.ipynb (Phase 1)

**Purpose**: Interactive demonstration of Phase 1 dataset append functionality using real production data

**Dataset**: Uses copies of actual production dataset (`output/products.parquet` → `notebooks/data/products.parquet`)

**Features Demonstrated**:
- Inspecting the real dataset (2 Axis camera records)
- Appending with different merge strategies (update, skip)
- Merge statistics tracking (added/updated/skipped counts)
- Overwrite vs append behavior
- Data integrity verification
- Dataset growth analysis (2 → ~1,361 records after append)

**Usage**:
```bash
# From project root, launch Jupyter
jupyter notebook

# Then navigate to notebooks/dataset_append_demo.ipynb in the browser
# Or directly open it:
jupyter notebook notebooks/dataset_append_demo.ipynb
```

**Key Sections**:
1. Inspect current dataset (2 records)
2. Create backup before modifications
3. Append with UPDATE strategy (overwrites duplicates)
4. Verify merge statistics
5. Verify data integrity (original records preserved)
6. Compare before/after datasets
7. Append with SKIP strategy (keeps originals)
8. Test overwrite mode (complete replacement)
9. Analyze dataset growth by source/category
10. Restore original dataset

### dataset_versioning_demo.ipynb (Phase 2)


**Purpose**: Interactive demonstration of Phase 2 dataset versioning functionality using real production data

**Dataset**: Uses copies of actual production dataset (`output/products.parquet` → `notebooks/data/products.parquet`)

**Features Demonstrated**:
- Auto-versioning mode (automatic version increments)
- Creating multiple versions (v1, v2, v3)
- Version metadata tracking (timestamps, record counts, hashes)
- Manifest versioning with SHA-256 hashing
- Symlink management (automatic updates)
- Listing version history
- Manual cleanup of old versions
- Append mode integration with versioning
- Parent version tracking

**Usage**:
```bash
# Launch Jupyter from project root
jupyter notebook notebooks/dataset_versioning_demo.ipynb

# Or use JupyterLab
jupyter lab notebooks/dataset_versioning_demo.ipynb
```

**Key Sections**:
1. Create version 1 with auto-versioning
2. Verify version 1 creation and metadata
3. Create version 2 with append mode
4. Compare versions 1 and 2
5. Create version 3 (overwrite mode)
6. List all versions
7. Verify symlinks
8. Test manual cleanup
9. Verify files after cleanup
10. Verify manifest hashing
11. Cleanup test files

### validation_integration_demo.ipynb (Phase 3)

**Purpose**: Interactive demonstration of Phase 3 validation integration with dataset versioning using real production data

**Dataset**: Uses copies of actual production dataset (`output/products.parquet` → `notebooks/data/products.parquet`)

**Features Demonstrated**:
- Creating multiple dataset versions
- Validating current dataset (symlink)
- Validating specific versions
- Version-aware validation reports
- Validating versions created with append mode
- Comparing validation results across versions
- Historical validation (older versions remain accessible)

**Usage**:
```bash
# Launch Jupyter from project root
jupyter notebook notebooks/validation_integration_demo.ipynb

# Or use JupyterLab
jupyter lab notebooks/validation_integration_demo.ipynb
```

**Key Sections**:
1. Create version 1
2. Validate version 1 with version info
3. Create version 2 with append
4. Validate version 2 (shows append metadata)
5. Create version 3 (overwrite mode)
6. Validate current dataset (symlink, no version info)
7. List all versions
8. Validate older version
9. Compare validation results across versions
10. Verify symlinks point to current version
11. Cleanup test files

### manifest_reconstruction_demo.ipynb (Phase 4)

**Purpose**: Interactive demonstration of manifest reconstruction from cache database and filesystem

**Dataset**: Uses copies of actual production dataset and requires `output/download_cache.db` and `data/images/`

**Features Demonstrated**:
- Simulating manifest loss (backup and delete)
- Reconstructing manifest from cache + filesystem
- Comparing reconstructed vs original manifest
- Validating dataset with reconstructed manifest
- CLI command usage (`cidar-reconstruct-manifest`)
- Edge case handling (missing cache/data)

**Usage**:
```bash
# Launch Jupyter from project root
jupyter notebook notebooks/manifest_reconstruction_demo.ipynb

# Or use JupyterLab
jupyter lab notebooks/manifest_reconstruction_demo.ipynb
```

**Prerequisites** (in addition to standard setup):
- `output/download_cache.db` - Cache database from scraping
- `data/images/` - Filesystem structure from scraping

**Key Sections**:
1. Setup paths and verify prerequisites
2. Inspect original manifest
3. Create backup and simulate manifest loss
4. Reconstruct manifest from cache + filesystem
5. Compare original vs reconstructed
6. Validate dataset with reconstructed manifest
7. Restore original manifest
8. CLI command demonstration
9. Test edge cases
10. Cleanup test files

## Running Notebooks

### Prerequisites

Ensure you have Jupyter installed:
```bash
# Using uv
uv add jupyter notebook

# Or using pip
pip install jupyter notebook
```

### Launch Jupyter

**IMPORTANT:** You **must** run Jupyter from the **project root** directory, not from `notebooks/`:

```bash
# Navigate to project root first
cd /path/to/cctv-scrapers

# Verify you're in the project root (should see src/, notebooks/, data/, etc.)
ls

# Launch Jupyter from project root
jupyter notebook

# In the browser, navigate to notebooks/ and open a notebook
```

**Why project root?** The notebooks access both:
- Test data in `notebooks/data/` (copies for safe testing)
- Production data in `data/` and `output/` (for building datasets)

Running from project root ensures all paths resolve correctly.

## Data Directory

**`notebooks/data/`** (gitignored):
- Contains copies of production datasets for safe testing
- Files are copied from `output/` directory
- All notebook operations work on these copies, never touching production data
  - To refresh data: `cp output/products.parquet notebooks/data/` and `cp output/verification_manifest.json notebooks/data/`

## Notes

- Notebooks use copies of real production data in `notebooks/data/` (gitignored)
- Safe to experiment - all operations work on copies, not production datasets
- Demonstrates real-world usage scenarios with actual data
- Includes backup/restore steps for repeatable testing
- For unit tests with synthetic data, see `tests/test_dataset_append.py`

## Contributing

When adding new notebooks:
1. Place them in this directory
2. Use clear section headers and markdown explanations
3. Include cleanup cells to remove temporary data
4. Update this README with a description
5. Ensure notebooks can run independently