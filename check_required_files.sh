#!/bin/bash
# Check all required files for the project

echo "========================================="
echo "Checking Required Files"
echo "========================================="
echo ""

MISSING=0

# Check models
echo "1. Models:"
for model in model_ann.keras model_gru.keras model_lstm.keras model_transformer.keras; do
    if [ -f "models/$model" ]; then
        size=$(ls -lh "models/$model" | awk '{print $5}')
        echo "   ✓ $model ($size)"
    else
        echo "   ✗ MISSING: $model"
        MISSING=1
    fi
done

echo ""
echo "2. Ensemble Artifacts:"
for file in ensemble_meta.json scaler_X.pkl scaler_y.pkl; do
    if [ -f "artifacts/ensemble/$file" ]; then
        size=$(ls -lh "artifacts/ensemble/$file" | awk '{print $5}')
        echo "   ✓ $file ($size)"
    else
        echo "   ✗ MISSING: $file"
        MISSING=1
    fi
done

echo ""
echo "3. Drift Detection:"
if [ -f "artifacts/drift_detection/reference_data.csv" ]; then
    size=$(ls -lh "artifacts/drift_detection/reference_data.csv" | awk '{print $5}')
    echo "   ✓ reference_data.csv ($size)"
else
    echo "   ✗ MISSING: reference_data.csv"
    MISSING=1
fi

echo ""
echo "4. Training Data:"
if [ -f "data/final_data/20251115_dataset_crp.csv" ]; then
    size=$(ls -lh "data/final_data/20251115_dataset_crp.csv" | awk '{print $5}')
    echo "   ✓ 20251115_dataset_crp.csv ($size)"
else
    echo "   ✗ MISSING: 20251115_dataset_crp.csv"
    MISSING=1
fi

echo ""
echo "========================================="
if [ $MISSING -eq 0 ]; then
    echo "✓ ALL FILES PRESENT - Ready to build!"
else
    echo "✗ SOME FILES MISSING - See above"
    echo ""
    echo "To fix:"
    echo "  1. Pull from DVC: dvc pull -r gcsremote"
    echo "  2. Or run: ./fix_missing_files.sh"
fi
echo "========================================="

exit $MISSING
