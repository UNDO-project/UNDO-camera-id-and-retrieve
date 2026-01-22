#!/bin/bash
PYTHONPATH="${PYTHONPATH}:$(realpath "./src")"
export PYTHONPATH

# Navigate to the script's directory (project root)
cd "$(dirname "$0")" || exit

echo "=========================================="
echo "CCTV Scrapers Test Pipeline"
echo "=========================================="
echo ""

echo "Running Axis extraction tests"
pytest tests/test_axis_extraction.py -v
echo "Done..."
echo "==============================================="

echo "Running Hikvision extraction tests"
pytest tests/test_hikvision_extraction.py -v
echo "Done..."
echo "==============================================="

echo "Running configuration tests"
pytest tests/test_config.py -v
echo "Done..."
echo "==============================================="

echo "Running dataset creation tests"
pytest tests/test_dataset_creation.py -v
echo "Done..."
echo "==============================================="

echo "Running dataset append tests"
pytest tests/test_dataset_append.py -v
echo "Done..."
echo "==============================================="

echo "Running dataset versioning tests"
pytest tests/test_dataset_versioning.py -v
echo "Done..."
echo "==============================================="

echo "Running validation integration tests"
pytest tests/test_validation_integration.py -v
echo "Done..."
echo "==============================================="

echo "Running download cache tests"
pytest tests/test_download_cache.py -v
echo "Done..."
echo "==============================================="

echo "Running detector tests"
pytest tests/test_detector.py -v
echo "Done..."
echo "==============================================="

echo "Running identification service tests"
pytest tests/test_identification_service.py -v
echo "Done..."
echo "==============================================="

echo "Running catalog service tests"
pytest tests/test_catalog_service.py -v
echo "Done..."
echo "==============================================="

echo "Running catalog and index tests"
pytest tests/test_catalog_index.py -v
echo "Done..."
echo "==============================================="

echo "Running CLI identify tests"
pytest tests/test_cli_identify.py -v
echo "Done..."
echo "==============================================="

echo "Running manifest reconstruction tests"
pytest tests/test_manifest_reconstruction.py -v
echo "Done..."
echo "==============================================="

echo "Running API routes tests"
pytest tests/test_api_routes.py -v
echo "Done..."
echo "==============================================="

echo "Running API middleware tests"
pytest tests/test_api_middleware.py -v
echo "Done..."
echo "==============================================="

echo ""
echo "=========================================="
echo "All individual tests completed!"
echo "=========================================="
echo ""
echo "Running full test suite with coverage..."
echo ""

pytest --cov=src --cov-report=term-missing --cov-report=html

echo ""
echo "=========================================="
echo "Test Pipeline Complete!"
echo "=========================================="
echo ""
echo "Coverage report saved to: htmlcov/index.html"
echo "Run 'open htmlcov/index.html' to view the HTML report"
echo ""