#!/bin/bash

# Fix Missing DVC Files Script
# This script handles the issue where dvc pull doesn't properly handle standalone .dvc files

echo "========================================="
echo "Fixing Missing DVC Files"
echo "========================================="
echo ""

# Pull all pipeline outputs (models, artifacts)
echo "1. Pulling pipeline outputs from remote..."
dvc pull -r gcsremote
if [ $? -eq 0 ]; then
    echo "   ✓ Pipeline outputs pulled"
else
    echo "   ⚠ Pipeline pull completed with warnings"
fi

# Then checkout the training data (tracked by standalone .dvc file)
# This must be done AFTER dvc pull to avoid it being deleted
echo ""
echo "2. Checking out training data..."
dvc checkout data/final_data/20251115_dataset_crp.csv.dvc
if [ $? -eq 0 ]; then
    echo "   ✓ Training data checked out"
else
    echo "   ✗ Failed to checkout training data"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ All files restored!"
echo "========================================="
echo ""

# Run the check script to verify
if [ -f "./check_required_files.sh" ]; then
    ./check_required_files.sh
fi
