#!/usr/bin/env python3
"""
ANN Prediction Script

Predicts 5-day price change for all cryptocurrencies using the latest available data.

Usage:
    python scripts/predict_ann.py                    # Predict locally
    python scripts/predict_ann.py --gcs              # Save directly to GCS (no local file)
    python scripts/predict_ann.py --mlflow --gcs     # MLflow model + GCS output
    python scripts/predict_ann.py --all --gcs        # All dates to GCS
    python scripts/predict_ann.py --ticker btcusd    # Specific ticker
"""

import argparse
import os
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Disable TensorFlow verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# MLflow for Model Registry
import mlflow
import mlflow.keras

# GCS for cloud storage
from google.cloud import storage
from datetime import datetime

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / 'data' / 'final_data' / '20251115_dataset_crp.csv'
MODEL_PATH = REPO_ROOT / 'models' / 'model_ann.keras'
SCALER_X_PATH = REPO_ROOT / 'artifacts' / 'ensemble' / 'scaler_X.pkl'
SCALER_Y_PATH = REPO_ROOT / 'artifacts' / 'ensemble' / 'scaler_y.pkl'

# MLflow settings
MLFLOW_TRACKING_URI = 'https://mlflow-server-101264457040.europe-west3.run.app'
MLFLOW_MODEL_NAME = 'Model_ANN'

# GCS settings
GCS_BUCKET = 'time-series-mlflow-data'
GCS_PREDICTIONS_PATH = 'predictions'

# Columns to drop from features (same as training)
COLS_TO_DROP = [
    'Unnamed: 0',
    'ticker',
    'date',
    'future_5_close_higher_than_today',
    'future_10_close_higher_than_today',
    'future_5_close_lower_than_today',
    'future_10_close_lower_than_today',
    'higher_close_today_vs_future_5_close',
    'higher_close_today_vs_future_10_close',
    'lower_close_today_vs_future_5_close',
    'lower_close_today_vs_future_10_close'
]

TARGET_COL = 'future_5_close_higher_than_today'


def load_artifacts(use_mlflow: bool = False, model_version: str = "latest"):
    """Load model and scalers.

    Args:
        use_mlflow: If True, load model from MLflow Model Registry
        model_version: Version to load - "latest", version number, or alias like "@champion"
    """
    print("Loading model and scalers...")

    # Load scalers (always from local files)
    if not SCALER_X_PATH.exists():
        raise FileNotFoundError(f"Scaler X not found: {SCALER_X_PATH}")
    if not SCALER_Y_PATH.exists():
        raise FileNotFoundError(f"Scaler Y not found: {SCALER_Y_PATH}")

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    print(f"  Scalers loaded")

    # Load model
    if use_mlflow:
        print(f"  Loading from MLflow Model Registry...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Build model URI
        if model_version.startswith("@"):
            # Alias like "@champion"
            model_uri = f"models:/{MLFLOW_MODEL_NAME}{model_version}"
        elif model_version == "latest":
            # Latest version
            model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
        else:
            # Specific version number
            model_uri = f"models:/{MLFLOW_MODEL_NAME}/{model_version}"

        print(f"  Model URI: {model_uri}")
        model = mlflow.keras.load_model(model_uri)
        print(f"  Model loaded from MLflow: {MLFLOW_MODEL_NAME} ({model_version})")
    else:
        # Load from local file
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print(f"  Model loaded: {MODEL_PATH.name}")

    return model, scaler_X, scaler_y


def load_data(data_path: Path) -> pd.DataFrame:
    """Load dataset."""
    print(f"Loading data from {data_path}...")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    return df


def prepare_features(df: pd.DataFrame, scaler_X) -> np.ndarray:
    """Prepare and scale features."""
    print("Preparing features...")

    # Drop non-feature columns
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    X = df.drop(cols_to_drop, axis='columns')

    print(f"  Feature columns: {X.shape[1]}")
    print(f"  Sample features: {list(X.columns[:5])}...")

    # Scale features
    X_scaled = scaler_X.transform(X)

    return X_scaled


def run_predictions(model, X_scaled: np.ndarray, scaler_y) -> np.ndarray:
    """Run model predictions and inverse transform."""
    print("Running predictions...")

    # Predict (scaled)
    y_pred_scaled = model.predict(X_scaled, verbose=0)

    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

    print(f"  Predictions completed: {len(y_pred):,} samples")
    print(f"  Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"  Mean prediction: {y_pred.mean():.2f}")

    return y_pred


def save_results(df: pd.DataFrame, predictions: np.ndarray, to_gcs: bool = False):
    """Save predictions to CSV (locally or directly to GCS)."""

    # Create results DataFrame
    results = pd.DataFrame({
        'ticker': df['ticker'],
        'date': df['date'],
        'actual': df[TARGET_COL] if TARGET_COL in df.columns else np.nan,
        'predicted': predictions,
    })

    # Add error if actual values exist
    if TARGET_COL in df.columns:
        results['error'] = results['actual'] - results['predicted']
        results['abs_error'] = results['error'].abs()

        mae = results['abs_error'].mean()
        print(f"  MAE: {mae:.4f}")

    if to_gcs:
        # Upload directly to GCS (no local file)
        gcs_uri = upload_to_gcs(results)
        print(f"  Saved {len(results):,} predictions to GCS")
        return results, gcs_uri
    else:
        # Save locally
        output_path = REPO_ROOT / 'predictions_ann.csv'
        results.to_csv(output_path, index=False)
        print(f"  Saved {len(results):,} predictions to {output_path}")
        return results, str(output_path)


def upload_to_gcs(results_df: pd.DataFrame, bucket_name: str = GCS_BUCKET):
    """Upload predictions DataFrame directly to GCS (no local file)."""
    print(f"Uploading to GCS bucket: {bucket_name}...")

    # Create timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gcs_filename = f"{GCS_PREDICTIONS_PATH}/predictions_ann_{timestamp}.csv"

    # Convert DataFrame to CSV string
    csv_data = results_df.to_csv(index=False)

    # Upload directly to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_filename)
    blob.upload_from_string(csv_data, content_type='text/csv')

    gcs_uri = f"gs://{bucket_name}/{gcs_filename}"
    print(f"  Uploaded to: {gcs_uri}")

    return gcs_uri


def main():
    parser = argparse.ArgumentParser(description='Predict 5-day crypto price change')
    parser.add_argument(
        '--data-path',
        type=Path,
        default=DEFAULT_DATA_PATH,
        help='Path to input dataset'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Predict for all dates (default: latest date only)'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default=None,
        help='Filter for specific ticker (e.g., btcusd)'
    )
    parser.add_argument(
        '--mlflow',
        action='store_true',
        help='Load model from MLflow Model Registry instead of local file'
    )
    parser.add_argument(
        '--model-version',
        type=str,
        default='latest',
        help='Model version to load: "latest", version number, or alias like "@champion"'
    )
    parser.add_argument(
        '--gcs',
        action='store_true',
        help='Save predictions directly to GCS (default: save locally)'
    )
    args = parser.parse_args()

    print("ANN Prediction Script")
    print("=" * 10)

    # Load artifacts
    model, scaler_X, scaler_y = load_artifacts(
        use_mlflow=args.mlflow,
        model_version=args.model_version
    )

    # Load data
    df = load_data(args.data_path)

    # Filter for latest date only (unless --all flag)
    if not args.all:
        latest_date = df['date'].max()
        df = df[df['date'] == latest_date].copy()
        print(f"  Filtered to latest date: {latest_date}")
        print(f"  Rows after filter: {len(df):,}")

    # Filter for specific ticker
    if args.ticker:
        df = df[df['ticker'] == args.ticker.lower()].copy()
        print(f"  Filtered to ticker: {args.ticker}")
        print(f"  Rows after filter: {len(df):,}")

    if len(df) == 0:
        print("ERROR: No data after filtering!")
        return

    # Prepare features
    X_scaled = prepare_features(df, scaler_X)

    # Run predictions
    predictions = run_predictions(model, X_scaled, scaler_y)

    # Save results (to GCS or locally)
    results, output_location = save_results(df, predictions, to_gcs=args.gcs)

    # Show predictions (sorted by predicted change)
    print("\nPredictions (sorted by expected 5-day change):")
    results_sorted = results.sort_values('predicted', ascending=False)
    print(results_sorted[['ticker', 'date', 'predicted']].to_string(index=False))

    # Calculate and display metrics (only if actual values exist)
    if TARGET_COL in df.columns:
        actual = results['actual']
        predicted = results['predicted']

        print("\nMODEL METRICS")
        print("=" * 10)

        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print("=" * 10)

    print("\nI'm done!")
    print(f"Results saved to: {output_location}")

if __name__ == "__main__":
    main()
