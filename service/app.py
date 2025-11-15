from fastapi import FastAPI
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

app = FastAPI()

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "ensemble"

# Load ensemble artifacts
scaler_X = joblib.load(ARTIFACTS_DIR / "scaler_X.pkl")
scaler_y = joblib.load(ARTIFACTS_DIR / "scaler_y.pkl")

# Load base models
model_ann = load_model(MODELS_DIR / "model_ann.keras")
model_gru = load_model(MODELS_DIR / "model_gru.keras")
model_lstm = load_model(MODELS_DIR / "model_lstm.keras")
model_trf = load_model(MODELS_DIR / "model_transformer.keras")

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: list):
    x = np.array(features).reshape(1, -1)
    x_scaled = scaler_X.transform(x)

    pred_ann = model_ann.predict(x_scaled)[0][0]
    pred_gru = model_gru.predict(x_scaled.reshape(1, 1, -1))[0][0]
    pred_lstm = model_lstm.predict(x_scaled.reshape(1, 1, -1))[0][0]
    pred_trf = model_trf.predict(x_scaled.reshape(1, 1, -1))[0][0]

    ensemble_pred = np.mean([pred_ann, pred_gru, pred_lstm, pred_trf])

    pred_original = scaler_y.inverse_transform([[ensemble_pred]])[0][0]

    return {"prediction": float(pred_original)}