#!/bin/bash
# Simple Docker build - NO DVC REQUIRED
# All files are already local!

set -e  # Exit on error

echo "========================================="
echo "Building Docker Image (DVC-free)"
echo "========================================="
echo ""

# Check if required files exist
echo "1. Checking files..."
./check_required_files.sh || {
    echo ""
    echo "ERROR: Missing files! Run:"
    echo "  - For dataset: dvc pull data/final_data/20251115_dataset_crp.csv.dvc -r gcsremote"
    echo "  - For reference data: python scripts/prepare_reference_data.py"
    exit 1
}

echo ""
echo "2. Building Docker image..."
docker-compose -f docker-compose.monitoring.yml build api

echo ""
echo "3. Starting services..."
docker-compose -f docker-compose.monitoring.yml up -d

echo ""
echo "========================================="
echo "✓ Build Complete!"
echo "========================================="
echo ""
echo "Services:"
echo "  API:          http://localhost:8000"
echo "  Prometheus:   http://localhost:9090"
echo "  Grafana:      http://localhost:3000"
echo "  Alertmanager: http://localhost:9093"
echo ""
echo "Test drift detection:"
echo "  python scripts/check_drift.py"
echo ""
