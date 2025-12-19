# CCTV Scrapers Notebooks

This directory contains Jupyter notebooks for interactive demonstrations and experiments with the CCTV scrapers pipeline.

## Available Notebooks

### dataset_append_demo.ipynb

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
# Launch Jupyter from project root
jupyter notebook notebooks/dataset_append_demo.ipynb

# Or use JupyterLab
jupyter lab notebooks/dataset_append_demo.ipynb
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

From the project root:
```bash
jupyter notebook
# Navigate to notebooks/ directory in the browser
```

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