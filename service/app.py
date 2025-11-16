from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
import time
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator

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


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/predict", response_model=EnsembleResponse)
def predict(request: SequenceRequest):
    seq = np.array(request.sequence, dtype=float)  # Shape (T, n_features)

    # Validierung der Länge
    if seq.shape[0] != LOOK_BACK:
        INPUT_VALIDATION_ERRORS.inc()
        raise ValueError(f"Expected sequence length {LOOK_BACK}, got {seq.shape[0]}")

    # Skalierung wie im Training: scaler_X wurde auf (n_samples, n_features) gefittet,
    # wir geben ihm hier (LOOK_BACK, n_features)
    seq_scaled = scaler_X.transform(seq)  # (LOOK_BACK, n_features)

    # ANN bekommt den letzten Zeitschritt (wie ein „normaler" Sample)
    x_last_scaled = seq_scaled[-1].reshape(1, -1)  # (1, n_features)

    start_time = time.time()
    pred_ann_scaled = model_ann.predict(x_last_scaled, verbose=0)[0][0]
    PREDICTION_LATENCY.labels(model='ann').observe(time.time() - start_time)
    PREDICTION_COUNTER.labels(model='ann').inc()

    # RNN-Modelle bekommen die gesamte Sequenz
    seq_scaled_rnn = seq_scaled.reshape(1, LOOK_BACK, -1)  # (1, 20, n_features)

    start_time = time.time()
    pred_gru_scaled = model_gru.predict(seq_scaled_rnn, verbose=0)[0][0]
    PREDICTION_LATENCY.labels(model='gru').observe(time.time() - start_time)
    PREDICTION_COUNTER.labels(model='gru').inc()

    start_time = time.time()
    pred_lstm_scaled = model_lstm.predict(seq_scaled_rnn, verbose=0)[0][0]
    PREDICTION_LATENCY.labels(model='lstm').observe(time.time() - start_time)
    PREDICTION_COUNTER.labels(model='lstm').inc()

    start_time = time.time()
    pred_trf_scaled = model_trf.predict(seq_scaled_rnn, verbose=0)[0][0]
    PREDICTION_LATENCY.labels(model='transformer').observe(time.time() - start_time)
    PREDICTION_COUNTER.labels(model='transformer').inc()

    # Ensemble im skalierten Raum
    preds_scaled = np.array([
        pred_ann_scaled,
        pred_gru_scaled,
        pred_lstm_scaled,
        pred_trf_scaled,
    ])
    ensemble_scaled = preds_scaled.mean()

    # Zurück in Original-Skala
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

    return EnsembleResponse(
        prediction=float(ensemble_original),
        components={
            "ANN": float(ann_original),
            "GRU": float(gru_original),
            "LSTM": float(lstm_original),
            "Transformer": float(trf_original),
        },
    )