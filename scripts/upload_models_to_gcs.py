#!/usr/bin/env python3
"""
Upload Models to GCS

Uploads trained models and artifacts to GCS for CI/CD deployment.

Usage:
    python scripts/upload_models_to_gcs.py
"""

from pathlib import Path
from google.cloud import storage


GCS_BUCKET = 'time-series-mlflow-data'
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def main():
    print("Upload Models to GCS")
    print("=" * 40)

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

    upload_to_gcs()


if __name__ == "__main__":
    main()
