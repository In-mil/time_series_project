#!/usr/bin/env python3
"""
Upload Models to GCS and Register in MLflow

Uploads trained models and artifacts to GCS for CI/CD deployment,
and registers models in MLflow Model Registry.

Usage:
    python scripts/upload_models_to_gcs.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
from google.cloud import storage
import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model


GCS_BUCKET = 'time-series-mlflow-data'
MLFLOW_TRACKING_URI = 'https://mlflow-server-101264457040.europe-west3.run.app'
REPO_ROOT = Path(__file__).resolve().parents[1]

# Model name mapping for registry
MODEL_REGISTRY_NAMES = {
    'model_ann.keras': 'ANN_Crypto_Predictor',
    'model_gru.keras': 'GRU_Crypto_Predictor',
    'model_lstm.keras': 'LSTM_Crypto_Predictor',
    'model_transformer.keras': 'Transformer_Crypto_Predictor',
}


def upload_to_gcs():
    """Upload models and artifacts to GCS."""
    print(f"Uploading to gs://{GCS_BUCKET}/artifacts/latest/")

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    uploaded = []

    # Upload all .keras model files
    models_dir = REPO_ROOT / 'models'
    for model_file in models_dir.glob('*.keras'):
        blob = bucket.blob(f'artifacts/latest/{model_file.name}')
        blob.upload_from_filename(model_file)
        uploaded.append(f'models/{model_file.name}')
        print(f"  Uploaded: {model_file.name}")

    # Upload scalers
    artifacts_dir = REPO_ROOT / 'artifacts' / 'ensemble'
    for scaler_file in artifacts_dir.glob('*.pkl'):
        blob = bucket.blob(f'artifacts/latest/{scaler_file.name}')
        blob.upload_from_filename(scaler_file)
        uploaded.append(f'artifacts/ensemble/{scaler_file.name}')
        print(f"  Uploaded: {scaler_file.name}")

    # Upload drift detection reference data
    drift_dir = REPO_ROOT / 'artifacts' / 'drift_detection'
    if drift_dir.exists():
        for drift_file in drift_dir.glob('*'):
            if drift_file.is_file():
                blob = bucket.blob(f'artifacts/latest/drift_detection/{drift_file.name}')
                blob.upload_from_filename(drift_file)
                uploaded.append(f'artifacts/drift_detection/{drift_file.name}')
                print(f"  Uploaded: drift_detection/{drift_file.name}")

    # Upload data file for batch predictions
    data_file = REPO_ROOT / 'data' / 'final_data' / '20251115_dataset_crp.csv'
    if data_file.exists():
        blob = bucket.blob('data/final_data/20251115_dataset_crp.csv')
        blob.upload_from_filename(data_file)
        uploaded.append('data/final_data/20251115_dataset_crp.csv')
        print(f"  Uploaded: 20251115_dataset_crp.csv")

    print(f"\nDone! Uploaded {len(uploaded)} files to GCS.")
    return uploaded


def register_models_to_mlflow():
    """Register all models to MLflow Model Registry."""
    print(f"\nRegistering models to MLflow...")
    print(f"  Tracking URI: {MLFLOW_TRACKING_URI}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("model_registry")

    models_dir = REPO_ROOT / 'models'
    registered = []

    for model_file in models_dir.glob('*.keras'):
        registry_name = MODEL_REGISTRY_NAMES.get(model_file.name)
        if not registry_name:
            print(f"  Skipping {model_file.name} (no registry name defined)")
            continue

        print(f"  Registering: {model_file.name} -> {registry_name}")

        try:
            # Load the model
            model = load_model(model_file)

            # Log and register in MLflow
            with mlflow.start_run(run_name=f"{registry_name}_registration"):
                mlflow.keras.log_model(
                    model,
                    "model",
                    registered_model_name=registry_name
                )
                registered.append(registry_name)
                print(f"    Registered: {registry_name}")

        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nRegistered {len(registered)} models to MLflow Model Registry.")
    return registered


def main():
    print("Upload Models to GCS & Register in MLflow")
    print("=" * 50)

    # Check for .keras files
    models_dir = REPO_ROOT / 'models'
    keras_files = list(models_dir.glob('*.keras'))

    if not keras_files:
        print("ERROR: No .keras files found in models/")
        print("Train models first before uploading.")
        return

    print(f"Found {len(keras_files)} model files:")
    for f in keras_files:
        print(f"  - {f.name}")
    print()

    # Upload to GCS
    upload_to_gcs()

    # Register to MLflow
    register_models_to_mlflow()


if __name__ == "__main__":
    main()
