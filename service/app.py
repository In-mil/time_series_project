from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

app = FastAPI()

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
        raise ValueError(f"Expected sequence length {LOOK_BACK}, got {seq.shape[0]}")

    # Skalierung wie im Training: scaler_X wurde auf (n_samples, n_features) gefittet,
    # wir geben ihm hier (LOOK_BACK, n_features)
    seq_scaled = scaler_X.transform(seq)  # (LOOK_BACK, n_features)

    # ANN bekommt den letzten Zeitschritt (wie ein „normaler“ Sample)
    x_last_scaled = seq_scaled[-1].reshape(1, -1)  # (1, n_features)
    pred_ann_scaled = model_ann.predict(x_last_scaled, verbose=0)[0][0]

    # RNN-Modelle bekommen die gesamte Sequenz
    seq_scaled_rnn = seq_scaled.reshape(1, LOOK_BACK, -1)  # (1, 20, n_features)

    pred_gru_scaled = model_gru.predict(seq_scaled_rnn, verbose=0)[0][0]
    pred_lstm_scaled = model_lstm.predict(seq_scaled_rnn, verbose=0)[0][0]
    pred_trf_scaled = model_trf.predict(seq_scaled_rnn, verbose=0)[0][0]

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

    return EnsembleResponse(
        prediction=float(ensemble_original),
        components={
            "ANN": float(ann_original),
            "GRU": float(gru_original),
            "LSTM": float(lstm_original),
            "Transformer": float(trf_original),
        },
    )