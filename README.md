# Crypto Price Prediction API

An ML ensemble system predicting 5-day cryptocurrency price changes using ANN, LSTM, GRU, and Transformer models.

## Features

- **4-Model Ensemble**: Combines predictions from multiple deep learning architectures
- **Real-time API**: FastAPI with <100ms latency per model
- **Drift Detection**: Monitors data/prediction drift using Evidently AI
- **MLOps Ready**: MLflow tracking, Prometheus metrics, Cloud Run deployment

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn service.app:app --host 0.0.0.0 --port 8000

# Run tests
pytest
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Single prediction (20×68 sequence) |
| GET | `/predict/batch` | Batch predictions for all tickers |
| GET | `/drift/status` | Drift detection status |
| GET | `/metrics` | Prometheus metrics |

## Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sequence": [[...], [...], ...]}'  # 20 timesteps × 68 features
```

## Architecture

```
Input (20 days × 68 features)
         ↓
    ┌────┴────┐
    │ Scaling │
    └────┬────┘
         ↓
 ┌───┬───┼───┬───┐
 ANN GRU LSTM Transformer
 └───┴───┼───┴───┘
         ↓
         ↓
   5-day % Change
```

## Tech Stack

- **API**: FastAPI, Uvicorn
- **ML**: TensorFlow/Keras, scikit-learn
- **MLOps**: MLflow, Evidently, Prometheus
- **Cloud**: GCP Cloud Run, Cloud SQL, GCS

## Deployment

```bash
# Manual deployment via GitHub Actions
gh workflow run deploy-prediction-api.yml
```

**Production URL**: `https://prediction-api-101264457040.europe-west3.run.app`

## License

MIT
