# Docker Deployment Guide

This guide explains how to build and deploy the Time Series Prediction API as a Docker container.

## Overview

The API serves ensemble predictions from trained models (ANN, GRU, LSTM, Transformer) via a FastAPI endpoint.

**Base URL**: `http://localhost:8000` (local) or Cloud Run URL (production)

**Endpoints**:
- `GET /` - Health check
- `POST /predict` - Get ensemble prediction

---

## Local Build & Run

### Prerequisites

1. **DVC configured** with access to GCS remote
2. **Docker installed** (Docker Desktop or Docker Engine)
3. **Models trained and pushed** to DVC remote

### Build the Docker Image

```bash
# Pull models from DVC and build image
./build_docker.sh

# Or with custom image name/tag:
IMAGE_NAME=my-api IMAGE_TAG=v1.0 ./build_docker.sh
```

The script will:
1. Pull all required models from DVC remote
2. Verify all files exist
3. Build the Docker image

### Run the Container

```bash
# Run on port 8000
docker run -p 8000:8000 time-series-api:latest

# Run in background
docker run -d -p 8000:8000 --name ts-api time-series-api:latest

# View logs
docker logs ts-api

# Stop container
docker stop ts-api
```

### Test the API

```bash
# Health check
curl http://localhost:8000

# Example prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      [0.1, 0.2, 0.3, ...],  # 20 timesteps
      [0.2, 0.3, 0.4, ...],  # each with your feature vector
      ...
    ]
  }'
```

**Expected response**:
```json
{
  "prediction": 42156.78,
  "components": {
    "ANN": 41200.50,
    "GRU": 42500.25,
    "LSTM": 42300.10,
    "Transformer": 42600.35
  }
}
```

---

## GitHub Actions Automatic Build

The workflow `.github/workflows/docker-build.yml` automatically:

1. **Triggers** after successful model training
2. **Pulls** trained models from DVC remote
3. **Builds** Docker image
4. **Pushes** to Google Container Registry (GCR)

### Required GitHub Secrets

Configure these in your repository settings (`Settings → Secrets and variables → Actions`):

| Secret | Description | Example |
|--------|-------------|---------|
| `GCP_SERVICE_ACCOUNT_KEY` | GCP service account JSON key | `{"type": "service_account", ...}` |
| `GCP_PROJECT_ID` | Your GCP project ID | `mlflow-testpj` |
| `GCS_BUCKET_NAME` | GCS bucket for DVC storage | `time-series-dvc-storage` |

### Container Registry Setup

Before the workflow can push images, create the Artifact Registry repository:

```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west3"

# Enable Artifact Registry API
gcloud services enable artifactregistry.googleapis.com --project=$PROJECT_ID

# Create repository
gcloud artifacts repositories create time-series \
  --repository-format=docker \
  --location=$REGION \
  --description="Time series prediction API images" \
  --project=$PROJECT_ID
```

### Manual Trigger

You can also trigger the Docker build manually:

```bash
# Via GitHub UI: Actions → Build & Push Docker Image → Run workflow

# Via gh CLI:
gh workflow run docker-build.yml
```

---

## Cloud Run Deployment (Optional)

To enable automatic deployment to Cloud Run, uncomment the deployment steps in `.github/workflows/docker-build.yml` (lines 109-121).

### Setup Cloud Run

```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com --project=$PROJECT_ID

# Deploy manually (first time)
gcloud run deploy time-series-api \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/time-series/time-series-api:latest \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=3
```

After deployment, Cloud Run provides a public URL:
```
https://time-series-api-xxxxx-ew.a.run.app
```

### Cloud Run Configuration

- **Memory**: 2GB (models are large)
- **CPU**: 2 vCPUs
- **Min instances**: 0 (scales to zero when idle)
- **Max instances**: 3 (limits concurrent requests)
- **Authentication**: Public (remove `--allow-unauthenticated` for private API)

---

## Image Architecture

The Docker image includes:

```
/app/
├── service/
│   └── app.py          # FastAPI application
├── models/
│   ├── model_ann.keras
│   ├── model_gru.keras
│   ├── model_lstm.keras
│   └── model_transformer.keras
└── artifacts/
    └── ensemble/
        ├── scaler_X.pkl
        └── scaler_y.pkl
```

**Base image**: `python:3.11-slim`
**Size**: ~1.5GB (includes TensorFlow)
**Health check**: Every 30s on `GET /`

---

## Troubleshooting

### Build fails with "DVC pull failed"

**Issue**: Models not found in DVC remote

**Solutions**:
1. Verify models are pushed: `dvc push -r gcsremote`
2. Check GCP credentials: `gcloud auth list`
3. Verify DVC remote: `dvc remote list -v`

### Container fails to start

**Issue**: Missing models or dependencies

**Solutions**:
1. Check logs: `docker logs <container-id>`
2. Verify models exist: `docker run <image> ls -lh /app/models`
3. Test locally first before deploying

### Prediction returns errors

**Issue**: Input data format mismatch

**Solutions**:
1. Verify sequence length = 20 timesteps
2. Check feature count matches training data
3. Ensure numeric values (floats)

### GitHub Actions workflow fails

**Issue**: Secrets not configured or GCR permissions

**Solutions**:
1. Verify all secrets are set
2. Check service account has `roles/artifactregistry.writer`
3. Ensure Artifact Registry repository exists

---

## Development

### Local Testing Without Docker

```bash
# Install dependencies
pip install -r requirements.txt
pip install uvicorn fastapi

# Pull models
dvc pull -r gcsremote

# Run FastAPI dev server
uvicorn service.app:app --reload --port 8000
```

### Modifying the API

1. Edit `service/app.py`
2. Test locally
3. Rebuild Docker image: `./build_docker.sh`
4. Push changes to GitHub (triggers automatic build)

---

## Performance Notes

- **Cold start**: ~3-5 seconds (model loading)
- **Warm requests**: ~50-200ms per prediction
- **Memory usage**: ~1.5GB (loaded models)
- **Concurrent requests**: Limited by memory (recommend max 10-20)

---

## Security Considerations

### Production Deployment

For production use:

1. **Add authentication** (API keys, OAuth, etc.)
2. **Rate limiting** (prevent abuse)
3. **Input validation** (sanitize requests)
4. **HTTPS only** (Cloud Run provides this)
5. **Private API** (remove `--allow-unauthenticated`)

### Example: Add API Key Authentication

```python
# In service/app.py
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(request: SequenceRequest):
    # ... existing code
```

Then set `API_KEY` environment variable in Cloud Run.

---

## Cost Estimation (Cloud Run)

**Pricing factors**:
- CPU time: $0.00002400/vCPU-second
- Memory: $0.00000250/GiB-second
- Requests: $0.40/million

**Example**: 10,000 requests/month (avg 1s response)
- CPU: 10,000 × 2 vCPU × 1s × $0.000024 = $0.48
- Memory: 10,000 × 2 GB × 1s × $0.0000025 = $0.05
- Requests: 10,000 × $0.40/1M = $0.004

**Total**: ~$0.53/month

With min-instances=0, you only pay when the service is used.

---

## Next Steps

1. ✅ Build and test locally
2. ✅ Verify GitHub Actions workflow
3. ⬜ Set up monitoring (Cloud Monitoring, Prometheus)
4. ⬜ Add authentication for production
5. ⬜ Implement request logging
6. ⬜ Set up alerts for errors/high latency

---

Questions? Check the main [README](README) or open an issue.
