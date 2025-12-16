#!/bin/bash
PYTHONPATH="${PYTHONPATH}:$(realpath "./src")"
export PYTHONPATH

# Navigate to the script's directory (project root)
cd "$(dirname "$0")" || exit

echo "Running dataset creation tests"
pytest tests/test_dataset_creation.py
echo "Done..."
echo "==============================================="

echo "Running detector tests"
pytest tests/test_detector.py
echo "Done..."
echo "==============================================="

echo "Running download cache tests"
pytest tests/test_download_cache.py
echo "Done..."
echo "==============================================="

echo "Running extraction tests"
pytest tests/test_extraction.py
echo "Done..."
echo "==============================================="

echo "Running identification service tests"
pytest tests/test_identification_service.py
echo "Done..."
echo "==============================================="

echo "Running Hikvision extraction tests"
pytest tests/test_hikvision_extraction.py
echo "Done..."
echo "==============================================="