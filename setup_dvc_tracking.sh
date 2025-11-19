#!/bin/bash
# Setup DVC tracking for models

echo "Setting up DVC tracking for model files..."

# Add models to DVC
echo "1. Adding models to DVC..."
dvc add models/model_ann.keras
dvc add models/model_gru.keras
dvc add models/model_lstm.keras
dvc add models/model_transformer.keras
dvc add models/model_ensemble.json

# Add ensemble artifacts
echo "2. Adding ensemble artifacts to DVC..."
dvc add artifacts/ensemble

echo "3. Pushing to GCS remote..."
dvc push -r gcsremote

echo ""
echo "4. Git commit DVC files..."
git add models/*.dvc artifacts/*.dvc .gitignore
git status

echo ""
echo "✓ Done! Now you can commit:"
echo "  git commit -m 'Track models and artifacts with DVC'"
echo ""
echo "To pull later:"
echo "  dvc pull -r gcsremote"
