#!/bin/bash
# Local Docker build script
# Pulls models from DVC, then builds the Docker image

set -e  # Exit on error

echo "===================="
echo "Docker Build Script"
echo "===================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="${IMAGE_NAME:-time-series-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${BLUE}Step 1: Pulling models from DVC remote...${NC}"
if ! dvc pull models/model_ann.keras.dvc \
             models/model_gru.keras.dvc \
             models/model_lstm.keras.dvc \
             models/model_transformer.keras.dvc \
             models/model_ensemble.json.dvc \
             artifacts/ensemble.dvc -r gcsremote; then
    echo -e "${RED}ERROR: Failed to pull models from DVC${NC}"
    echo "Make sure you have:"
    echo "  1. DVC configured with gcsremote"
    echo "  2. Access to GCS bucket"
    echo "  3. Models pushed to remote (dvc push)"
    exit 1
fi

echo -e "${GREEN}✓ Models pulled successfully${NC}"

echo -e "${BLUE}Step 2: Verifying required files...${NC}"
required_files=(
    "models/model_ann.keras"
    "models/model_gru.keras"
    "models/model_lstm.keras"
    "models/model_transformer.keras"
    "artifacts/ensemble/scaler_X.pkl"
    "artifacts/ensemble/scaler_y.pkl"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}✗ Missing: $file${NC}"
        missing_files=$((missing_files + 1))
    else
        echo -e "${GREEN}✓ Found: $file${NC}"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo -e "${RED}ERROR: $missing_files required file(s) missing${NC}"
    exit 1
fi

echo -e "${BLUE}Step 3: Building Docker image...${NC}"
if docker build -t "$FULL_IMAGE" .; then
    echo -e "${GREEN}✓ Docker image built successfully: $FULL_IMAGE${NC}"
else
    echo -e "${RED}ERROR: Docker build failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}===================="
echo "Build completed!"
echo "===================="
echo ""
echo "Image: $FULL_IMAGE"
echo ""
echo "To run locally:"
echo "  docker run -p 8000:8000 $FULL_IMAGE"
echo ""
echo "To test the API:"
echo "  curl http://localhost:8000"
echo ""
