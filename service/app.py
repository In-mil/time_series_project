from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
import time
import uuid
import logging
import os
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from . import drift_detector

# MLflow for loading models from registry
import mlflow
import mlflow.keras

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Time Series Prediction API",
    description="ML models API for cryptocurrency price change predictions (% change in 5 days)",
    version="2.0.0"
)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['model']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)
PREDICTION_VALUE = Gauge(
    'last_prediction_value',
    'Last prediction value',
    ['model']
)
INPUT_VALIDATION_ERRORS = Counter(
    'input_validation_errors_total',
    'Total number of input validation errors'
)

# Initialize Prometheus instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

# Instrument the app
instrumentator.instrument(app)

# Add metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

LOOK_BACK = 20

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://mlflow-server-101264457040.europe-west3.run.app"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Model names in MLflow Registry (latest version)
MODEL_ANN_URI = os.getenv("MODEL_ANN_URI", "models:/Model_ANN/latest")
MODEL_GRU_URI = os.getenv("MODEL_GRU_URI", "models:/Model_GRU/latest")
MODEL_LSTM_URI = os.getenv("MODEL_LSTM_URI", "models:/Model_LSTM/latest")
MODEL_TRF_URI = os.getenv("MODEL_TRF_URI", "models:/Model_Transformer/latest")

# Load models from MLflow Registry at startup
logger.info(f"Loading models from MLflow Registry: {MLFLOW_TRACKING_URI}")
model_ann = mlflow.keras.load_model(MODEL_ANN_URI)
logger.info(f"Loaded ANN model: {MODEL_ANN_URI}")
model_gru = mlflow.keras.load_model(MODEL_GRU_URI)
logger.info(f"Loaded GRU model: {MODEL_GRU_URI}")
model_lstm = mlflow.keras.load_model(MODEL_LSTM_URI)
logger.info(f"Loaded LSTM model: {MODEL_LSTM_URI}")
model_trf = mlflow.keras.load_model(MODEL_TRF_URI)
logger.info(f"Loaded Transformer model: {MODEL_TRF_URI}")

# Load scalers from MLflow (stored with ANN model)
from mlflow.tracking import MlflowClient
client = MlflowClient()
ann_versions = client.get_latest_versions("Model_ANN")
if ann_versions:
    ann_run_id = ann_versions[0].run_id
    scalers_path = mlflow.artifacts.download_artifacts(run_id=ann_run_id, artifact_path="scalers")
    scaler_X = joblib.load(Path(scalers_path) / "scaler_X.pkl")
    scaler_y = joblib.load(Path(scalers_path) / "scaler_y.pkl")
    logger.info(f"Loaded scalers from MLflow run: {ann_run_id}")
else:
    raise RuntimeError("Model_ANN not found in MLflow Registry - scalers unavailable")


class SequenceRequest(BaseModel):
    # sequence[timestep][feature_index]
    sequence: List[List[float]] = Field(
        ..., description="Zeitreihenfenster: Liste von Zeitpunkten, jeder mit Feature-Vektor"
    )


class PredictionResponse(BaseModel):
    predictions: dict = Field(..., description="Individual model predictions (% price change in 5 days)")
    unit: str = Field(default="percent_change_5d", description="Unit of prediction values")


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup"""
    logger.info("Starting application")

    # Initialize drift detection
    reference_data_path = REPO_ROOT / "artifacts" / "drift_detection" / "reference_data.csv"
    if reference_data_path.exists():
        # Load reference data to get feature names
        import pandas as pd
        ref_df = pd.read_csv(reference_data_path, nrows=1)
        # Exclude metadata columns
        non_feature_cols = ['ticker', 'date']
        feature_names = [col for col in ref_df.columns if col not in non_feature_cols]

        drift_detector.initialize_drift_detector(
            reference_data_path=reference_data_path,
            window_size=1000,
            drift_threshold=0.3,
            feature_names=feature_names
        )
        logger.info(f"Drift detector initialized successfully with {len(feature_names)} features")
    else:
        logger.warning(f"Reference data not found at {reference_data_path}, drift detection disabled")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown"""
    logger.info("Shutting down application")


@app.get("/")
def home():
    return {"status": "ok"}


@app.get("/drift/status")
def get_drift_status():
    """Get current drift detection status"""
    detector = drift_detector.get_drift_detector()
    if not detector:
        return {"status": "disabled", "message": "Drift detection not initialized"}
    return detector.get_status()


@app.post("/drift/check")
def trigger_drift_check():
    """Manually trigger drift detection check"""
    detector = drift_detector.get_drift_detector()
    if not detector:
        return {"status": "disabled", "message": "Drift detection not initialized"}

    results = detector.check_drift()
    return results


@app.get("/drift/report")
def get_drift_report():
    """Generate detailed drift report (JSON)"""
    detector = drift_detector.get_drift_detector()
    if not detector:
        return {"status": "disabled", "message": "Drift detection not initialized"}

    # Get current status as detailed report
    status = detector.get_status()

    if status['current_window_size'] < 10:
        return {"status": "insufficient_data", "message": "Not enough data for drift report"}

    # Trigger drift check to get latest results
    drift_results = detector.check_drift()

    return {
        "status": "success",
        "report": {
            "overview": status,
            "drift_analysis": drift_results,
            "timestamp": datetime.now().isoformat(),
            "recommendation": (
                "Consider retraining the model"
                if drift_results.get('dataset_drift', False)
                else "No action required"
            )
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: SequenceRequest):
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received prediction request")

    try:
        seq = np.array(request.sequence, dtype=float)  # Shape (T, n_features)
        logger.info(f"[{request_id}] Input shape: {seq.shape}")

        # Validierung der Länge
        if seq.shape[0] != LOOK_BACK:
            INPUT_VALIDATION_ERRORS.inc()
            logger.warning(f"[{request_id}] Invalid sequence length: {seq.shape[0]}, expected {LOOK_BACK}")
            raise HTTPException(
                status_code=400,
                detail=f"Expected sequence length {LOOK_BACK}, got {seq.shape[0]}"
            )

        # Check for NaN or Inf values
        if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
            INPUT_VALIDATION_ERRORS.inc()
            logger.warning(f"[{request_id}] Input contains NaN or Inf values")
            raise HTTPException(
                status_code=400,
                detail="Input sequence contains NaN or Inf values"
            )

        # Skalierung wie im Training: scaler_X wurde auf (n_samples, n_features) gefittet,
        # wir geben ihm hier (LOOK_BACK, n_features)
        logger.info(f"[{request_id}] Scaling input sequence")
        seq_scaled = scaler_X.transform(seq)  # (LOOK_BACK, n_features)

        # ANN bekommt den letzten Zeitschritt (wie ein „normaler" Sample)
        x_last_scaled = seq_scaled[-1].reshape(1, -1)  # (1, n_features)

        # Track latencies in ms
        latencies = {}

        logger.info(f"[{request_id}] Running ANN prediction")
        start_time = time.time()
        pred_ann_scaled = model_ann.predict(x_last_scaled, verbose=0)[0][0]
        latency_ann = (time.time() - start_time) * 1000
        latencies['ann'] = latency_ann
        PREDICTION_LATENCY.labels(model='ann').observe(latency_ann / 1000)
        PREDICTION_COUNTER.labels(model='ann').inc()
        logger.info(f"[{request_id}] ANN prediction: {pred_ann_scaled:.4f} (latency: {latency_ann:.2f}ms)")

        # RNN-Modelle bekommen die gesamte Sequenz
        seq_scaled_rnn = seq_scaled.reshape(1, LOOK_BACK, -1)  # (1, 20, n_features)

        logger.info(f"[{request_id}] Running GRU prediction")
        start_time = time.time()
        pred_gru_scaled = model_gru.predict(seq_scaled_rnn, verbose=0)[0][0]
        latency_gru = (time.time() - start_time) * 1000
        latencies['gru'] = latency_gru
        PREDICTION_LATENCY.labels(model='gru').observe(latency_gru / 1000)
        PREDICTION_COUNTER.labels(model='gru').inc()
        logger.info(f"[{request_id}] GRU prediction: {pred_gru_scaled:.4f} (latency: {latency_gru:.2f}ms)")

        logger.info(f"[{request_id}] Running LSTM prediction")
        start_time = time.time()
        pred_lstm_scaled = model_lstm.predict(seq_scaled_rnn, verbose=0)[0][0]
        latency_lstm = (time.time() - start_time) * 1000
        latencies['lstm'] = latency_lstm
        PREDICTION_LATENCY.labels(model='lstm').observe(latency_lstm / 1000)
        PREDICTION_COUNTER.labels(model='lstm').inc()
        logger.info(f"[{request_id}] LSTM prediction: {pred_lstm_scaled:.4f} (latency: {latency_lstm:.2f}ms)")

        logger.info(f"[{request_id}] Running Transformer prediction")
        start_time = time.time()
        pred_trf_scaled = model_trf.predict(seq_scaled_rnn, verbose=0)[0][0]
        latency_transformer = (time.time() - start_time) * 1000
        latencies['transformer'] = latency_transformer
        PREDICTION_LATENCY.labels(model='transformer').observe(latency_transformer / 1000)
        PREDICTION_COUNTER.labels(model='transformer').inc()
        logger.info(f"[{request_id}] Transformer prediction: {pred_trf_scaled:.4f} (latency: {latency_transformer:.2f}ms)")

        # Ensemble im skalierten Raum
        preds_scaled = np.array([
            pred_ann_scaled,
            pred_gru_scaled,
            pred_lstm_scaled,
            pred_trf_scaled,
        ])
        ensemble_scaled = preds_scaled.mean()
        logger.info(f"[{request_id}] Ensemble prediction (scaled): {ensemble_scaled:.4f}")

        # Zurück in Original-Skala
        logger.info(f"[{request_id}] Inverse transforming predictions to original scale")
        ensemble_original = scaler_y.inverse_transform([[ensemble_scaled]])[0][0]
        ann_original = scaler_y.inverse_transform([[pred_ann_scaled]])[0][0]
        gru_original = scaler_y.inverse_transform([[pred_gru_scaled]])[0][0]
        lstm_original = scaler_y.inverse_transform([[pred_lstm_scaled]])[0][0]
        trf_original = scaler_y.inverse_transform([[pred_trf_scaled]])[0][0]

        # Update prediction value gauges
        PREDICTION_VALUE.labels(model='ensemble').set(ensemble_original)
        PREDICTION_VALUE.labels(model='ann').set(ann_original)
        PREDICTION_VALUE.labels(model='gru').set(gru_original)
        PREDICTION_VALUE.labels(model='lstm').set(lstm_original)
        PREDICTION_VALUE.labels(model='transformer').set(trf_original)
        PREDICTION_COUNTER.labels(model='ensemble').inc()

        # Track prediction for drift detection
        detector = drift_detector.get_drift_detector()
        if detector:
            try:
                detector.add_prediction(
                    features=seq_scaled,  # Use scaled features
                    prediction=float(ensemble_original),
                    metadata={'request_id': request_id, 'timestamp': time.time()}
                )
            except Exception as e:
                logger.warning(f"[{request_id}] Drift tracking failed: {e}")

        logger.info(f"[{request_id}] Prediction completed successfully")
        return PredictionResponse(
            predictions={
                "ANN": float(ann_original),
                "GRU": float(gru_original),
                "LSTM": float(lstm_original),
                "Transformer": float(trf_original),
            },
            unit="percent_change_5d"
        )

    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except ValueError as e:
        # Input validation errors
        logger.error(f"[{request_id}] Validation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch all other errors
        logger.error(f"[{request_id}] Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )


# Columns to drop for batch predictions (same as training)
BATCH_COLS_TO_DROP = [
    'Unnamed: 0', 'ticker', 'date',
    'future_5_close_higher_than_today', 'future_10_close_higher_than_today',
    'future_5_close_lower_than_today', 'future_10_close_lower_than_today',
    'higher_close_today_vs_future_5_close', 'higher_close_today_vs_future_10_close',
    'lower_close_today_vs_future_5_close', 'lower_close_today_vs_future_10_close'
]

# Data path for batch predictions
DATA_PATH = REPO_ROOT / "data" / "final_data" / "20251115_dataset_crp.csv"


class BatchPredictionResponse(BaseModel):
    date: str = Field(..., description="Prediction date")
    predictions: List[Dict] = Field(..., description="Predictions for all tickers")
    count: int = Field(..., description="Number of predictions")


@app.get("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(ticker: Optional[str] = None):
    """
    Batch prediction for all cryptocurrencies using latest available data.

    - Returns 5-day price change predictions for all tickers
    - Sorted by predicted change (highest first)
    - Optionally filter by ticker: /predict/batch?ticker=btcusd
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Batch prediction request")

    try:
        # Load data
        if not DATA_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Data file not found: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)
        logger.info(f"[{request_id}] Loaded {len(df)} rows")

        # Filter to latest date
        latest_date = df['date'].max()
        df = df[df['date'] == latest_date].copy()
        logger.info(f"[{request_id}] Filtered to {latest_date}: {len(df)} rows")

        # Filter by ticker if specified
        if ticker:
            df = df[df['ticker'] == ticker.lower()].copy()
            if len(df) == 0:
                raise HTTPException(status_code=404, detail=f"Ticker not found: {ticker}")
            logger.info(f"[{request_id}] Filtered to ticker {ticker}: {len(df)} rows")

        # Prepare features
        cols_to_drop = [c for c in BATCH_COLS_TO_DROP if c in df.columns]
        X = df.drop(cols_to_drop, axis='columns')

        # Scale features
        X_scaled = scaler_X.transform(X)

        # Run ANN predictions (best performing model)
        start_time = time.time()
        y_pred_scaled = model_ann.predict(X_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
        latency = (time.time() - start_time) * 1000
        logger.info(f"[{request_id}] ANN predictions completed in {latency:.2f}ms")

        # Build results
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            results.append({
                "ticker": row['ticker'],
                "predicted_5d_change": round(float(y_pred[i]), 2)
            })

        # Sort by predicted change (highest first)
        results.sort(key=lambda x: x['predicted_5d_change'], reverse=True)

        PREDICTION_COUNTER.labels(model='ann_batch').inc(len(results))

        logger.info(f"[{request_id}] Returning {len(results)} predictions")
        return BatchPredictionResponse(
            date=latest_date,
            predictions=results,
            count=len(results)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))