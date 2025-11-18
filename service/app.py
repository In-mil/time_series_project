from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
import time
import uuid
import logging
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from . import database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Time Series Prediction API",
    description="Ensemble model API for cryptocurrency price predictions",
    version="1.0.0"
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

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "ensemble"

LOOK_BACK = 20  

# Artefakte laden
scaler_X = joblib.load(ARTIFACTS_DIR / "scaler_X.pkl")
scaler_y = joblib.load(ARTIFACTS_DIR / "scaler_y.pkl")

# Basis-Modelle laden
model_ann = load_model(MODELS_DIR / "model_ann.keras")
model_gru = load_model(MODELS_DIR / "model_gru.keras")
model_lstm = load_model(MODELS_DIR / "model_lstm.keras")
model_trf = load_model(MODELS_DIR / "model_transformer.keras")


class SequenceRequest(BaseModel):
    # sequence[timestep][feature_index]
    sequence: List[List[float]] = Field(
        ..., description="Zeitreihenfenster: Liste von Zeitpunkten, jeder mit Feature-Vektor"
    )


class EnsembleResponse(BaseModel):
    prediction: float
    components: dict


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup"""
    logger.info("Starting application, initializing database connection pool")
    if database.DB_ENABLED:
        database.get_pool()  # Initialize pool at startup


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown"""
    logger.info("Shutting down application, closing database connection pool")
    database.close_pool()


@app.get("/")
def home():
    return {"status": "ok", "database_enabled": database.DB_ENABLED}


@app.get("/analytics/recent")
def get_recent_predictions(limit: int = 100):
    """Get recent predictions from database"""
    predictions = database.get_recent_predictions(limit=limit)
    return {"predictions": predictions, "count": len(predictions)}


@app.get("/analytics/performance")
def get_performance_metrics():
    """Get aggregated model performance metrics"""
    metrics = database.get_model_performance()
    if metrics is None:
        return {"error": "No data available or database disabled"}
    return metrics


@app.post("/analytics/update/{prediction_id}")
def update_prediction_actual(prediction_id: int, actual_value: float):
    """Update the actual value for a prediction (for drift detection)"""
    success = database.update_actual_value(prediction_id, actual_value)
    return {"success": success, "prediction_id": prediction_id}


@app.post("/predict", response_model=EnsembleResponse)
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

        # Log to database (async, non-blocking)
        predictions = {
            'ann': float(ann_original),
            'gru': float(gru_original),
            'lstm': float(lstm_original),
            'transformer': float(trf_original),
            'ensemble': float(ensemble_original)
        }

        try:
            database.log_prediction(
                input_sequence=request.sequence,
                predictions=predictions,
                latencies=latencies,
                request_id=request_id,
                model_version="1.0.0"
            )
        except Exception as e:
            # Log error but don't fail the prediction
            logger.warning(f"[{request_id}] Database logging failed: {e}")

        logger.info(f"[{request_id}] Prediction completed successfully. Ensemble: {ensemble_original:.4f}")
        return EnsembleResponse(
            prediction=float(ensemble_original),
            components={
                "ANN": float(ann_original),
                "GRU": float(gru_original),
                "LSTM": float(lstm_original),
                "Transformer": float(trf_original),
            },
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