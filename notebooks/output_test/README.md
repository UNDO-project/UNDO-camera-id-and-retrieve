# Notebook Test Output Directory

This directory is for testing and experimentation with notebooks.

**Do NOT use `../output/` in notebooks!** Always use this directory instead.

## Usage in Notebooks

```python
# Instead of:
# output_dir = Path("../output")

# Use:
output_dir = Path("./output_test")

# For building datasets:
from src.building.builder import DatasetBuilder
builder = DatasetBuilder(
    output_path="notebooks/output_test/products_test.parquet"
)
```

## Why?

The `output/` directory is the **production** directory used by:
- CLI commands (`cidar-build`, `cidar-validate`, `cidar-identify`)
- API server (`cidar-api`)

Notebooks should never modify production files.

## What's in this directory?

- `products_test.parquet` - Test datasets from notebooks
- `test_embeddings.npz` - Test embeddings
- Any other experimental outputs

## Cleaning

This directory can be safely deleted/cleaned at any time:

```bash
rm -rf notebooks/output_test/*
```
