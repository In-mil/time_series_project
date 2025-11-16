#!/usr/bin/env python3
"""Evaluate all trained models on the test set."""

import json
import random
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "final_data" / "20251115_dataset_crp.csv"
MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "ensemble"
REPORT_DIR = REPO_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "metrics.json"

# Create reports directory if needed
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Target columns to exclude from features
TARGET_COLUMNS = [
    "future_5_close_higher_than_today",
    "future_10_close_higher_than_today",
    "future_5_close_lower_than_today",
    "future_10_close_lower_than_today",
    "higher_close_today_vs_future_5_close",
    "higher_close_today_vs_future_10_close",
    "lower_close_today_vs_future_5_close",
    "lower_close_today_vs_future_10_close",
]


def create_sequences(
    X_data: np.ndarray,
    y_data: np.ndarray,
    df: pd.DataFrame,
    look_back: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding-window sequences for each cryptocurrency.

    Args:
        X_data: Scaled feature array
        y_data: Scaled target array
        df: Original dataframe (for ticker and date)
        look_back: Number of time steps to look back

    Returns:
        Tuple of (X_sequences, y_sequences, original_indices)
    """
    X_sequences = []
    y_sequences = []
    original_indices = []

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_indices = df[mask].index.values
        crypto_X = X_data[mask]
        crypto_y = y_data[mask]
        crypto_dates = df.loc[mask, "date"].values

        # Sort by date
        order = np.argsort(crypto_dates)
        crypto_X = crypto_X[order]
        crypto_y = crypto_y[order]
        ticker_indices = ticker_indices[order]

        # Create sequences
        for i in range(len(crypto_X) - look_back):
            X_sequences.append(crypto_X[i : i + look_back])
            y_sequences.append(crypto_y[i + look_back])
            original_indices.append(ticker_indices[i + look_back])

    return (
        np.array(X_sequences),
        np.array(y_sequences),
        np.array(original_indices)
    )


def load_and_prepare_data() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler
]:
    """Load data and prepare train/test splits with scaling.

    Returns:
        Tuple of (X_test_scaled, y_test_scaled, X_test_seq, y_test_seq,
                  X_test_df, df_test, scaler_X, scaler_y)
    """
    # Error handling for file operations
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Split data
    split_date_train = "2024-07-01"
    split_date_test = "2024-10-01"

    df_train = df[df["date"] < split_date_train].copy()
    df_test = df[df["date"] >= split_date_test].copy()

    print(f"\nData Split:")
    print(f"  Training set: {len(df_train):,} samples")
    print(f"  Test set: {len(df_test):,} samples")

    # Prepare features and targets
    feature_cols = [col for col in df_train.columns if col not in TARGET_COLUMNS]

    X_train = df_train[feature_cols].copy()
    y_train = df_train["future_5_close_higher_than_today"].values

    X_test = df_test[feature_cols].copy()
    y_test = df_test["future_5_close_higher_than_today"].values

    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Columns to scale (exclude ticker and date)
    cols_to_scale = [col for col in feature_cols if col not in ["ticker", "date"]]

    # Fit on train, transform test
    scaler_X.fit(X_train[cols_to_scale])
    scaler_y.fit(y_train.reshape(-1, 1))

    X_test_scaled = scaler_X.transform(X_test[cols_to_scale])
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print(f"\nScaled test data:")
    print(f"  Features shape: {X_test_scaled.shape}")
    print(f"  Target shape: {y_test_scaled.shape}")

    # Create sequences for RNN models
    print("\nCreating sequences for RNN models...")
    X_test_seq, y_test_seq, test_indices_seq = create_sequences(
        X_test_scaled, y_test_scaled, X_test, look_back=20
    )

    print(f"  Sequence shape: {X_test_seq.shape}")
    print(f"  ({X_test_seq.shape[0]} sequences, {X_test_seq.shape[1]} timesteps, {X_test_seq.shape[2]} features)")

    return (
        X_test_scaled, y_test_scaled, X_test_seq, y_test_seq,
        X_test, df_test, scaler_X, scaler_y
    )


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    y_test_original: np.ndarray,
    scaler_y: StandardScaler,
    model_name: str
) -> Dict[str, float]:
    """Evaluate a single model.

    Args:
        model: Loaded Keras model
        X_test: Test features (2D for ANN, 3D for RNN)
        y_test_scaled: Scaled test targets
        y_test_original: Original scale test targets
        scaler_y: Fitted target scaler
        model_name: Name of the model

    Returns:
        Dictionary with MAE and MSE metrics
    """
    print(f"\nEvaluating {model_name}...")

    # Get predictions (scaled)
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform to original scale
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Calculate metrics on both scales
    mae_scaled = mean_absolute_error(y_test_scaled, y_pred_scaled)
    mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
    mae_original = mean_absolute_error(y_test_original, y_pred_original)
    mse_original = mean_squared_error(y_test_original, y_pred_original)

    print(f"  MAE (original): {mae_original:.4f}")
    print(f"  MSE (original): {mse_original:.4f}")

    return {
        "MAE_scaled": float(mae_scaled),
        "MSE_scaled": float(mse_scaled),
        "MAE_original": float(mae_original),
        "MSE_original": float(mse_original)
    }


def evaluate_ensemble(
    models: Dict[str, tf.keras.Model],
    X_test_ann: np.ndarray,
    X_test_seq: np.ndarray,
    y_test_scaled: np.ndarray,
    y_test_original: np.ndarray,
    test_indices_ann: np.ndarray,
    test_indices_seq: np.ndarray,
    scaler_y: StandardScaler
) -> Dict[str, float]:
    """Evaluate ensemble model.

    Args:
        models: Dictionary of loaded models
        X_test_ann: Test features for ANN (2D)
        X_test_seq: Test features for RNN models (3D)
        y_test_scaled: Scaled test targets (for sequences)
        y_test_original: Original scale test targets (for sequences)
        test_indices_ann: Indices for ANN test data
        test_indices_seq: Indices for sequence test data
        scaler_y: Fitted target scaler

    Returns:
        Dictionary with ensemble metrics
    """
    print("\nEvaluating ENSEMBLE...")

    all_predictions = []
    model_names = []

    # ANN predictions
    if "ANN" in models and models["ANN"] is not None:
        pred_ann = models["ANN"].predict(X_test_ann, verbose=0)
        # Align to sequence indices
        ann_positions = []
        for seq_idx in test_indices_seq:
            position = np.where(test_indices_ann == seq_idx)[0]
            if len(position) > 0:
                ann_positions.append(position[0])
        pred_ann_aligned = pred_ann[ann_positions]
        all_predictions.append(pred_ann_aligned)
        model_names.append("ANN")

    # RNN predictions
    for name in ["GRU", "LSTM", "Transformer"]:
        if name in models and models[name] is not None:
            pred = models[name].predict(X_test_seq, verbose=0)
            all_predictions.append(pred)
            model_names.append(name)

    if len(all_predictions) == 0:
        print("  No models available for ensemble!")
        return {}

    # Average predictions
    stacked_predictions = np.stack(all_predictions, axis=0)
    ensemble_pred_scaled = np.mean(stacked_predictions, axis=0)

    print(f"  Averaged {len(all_predictions)} models: {model_names}")

    # Inverse transform
    ensemble_pred_original = scaler_y.inverse_transform(
        ensemble_pred_scaled.reshape(-1, 1)
    ).ravel()

    # Calculate metrics
    mae_scaled = mean_absolute_error(y_test_scaled, ensemble_pred_scaled)
    mse_scaled = mean_squared_error(y_test_scaled, ensemble_pred_scaled)
    mae_original = mean_absolute_error(y_test_original, ensemble_pred_original)
    mse_original = mean_squared_error(y_test_original, ensemble_pred_original)

    print(f"  MAE (original): {mae_original:.4f}")
    print(f"  MSE (original): {mse_original:.4f}")

    return {
        "MAE_scaled": float(mae_scaled),
        "MSE_scaled": float(mse_scaled),
        "MAE_original": float(mae_original),
        "MSE_original": float(mse_original),
        "base_models": model_names
    }


def main():
    """Run model evaluation pipeline."""
    print("=" * 60)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 60)

    # Load and prepare data
    (X_test_scaled, y_test_scaled_ann, X_test_seq, y_test_seq,
     X_test_df, df_test, scaler_X, scaler_y) = load_and_prepare_data()

    # Get indices for alignment
    test_indices_ann = X_test_df.index.values
    _, _, test_indices_seq = create_sequences(
        X_test_scaled, y_test_scaled_ann, X_test_df, look_back=20
    )

    # Get original scale targets
    y_test_original_ann = df_test["future_5_close_higher_than_today"].values
    y_test_original_seq = df_test.loc[test_indices_seq, "future_5_close_higher_than_today"].values

    # Load models
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    models = {}
    model_files = {
        "ANN": "model_ann.keras",
        "GRU": "model_gru.keras",
        "LSTM": "model_lstm.keras",
        "Transformer": "model_transformer.keras"
    }

    for name, filename in model_files.items():
        model_path = MODELS_DIR / filename
        try:
            models[name] = tf.keras.models.load_model(model_path)
            print(f"  ✓ {name} loaded")
        except Exception as e:
            print(f"  ✗ {name} not found: {e}")
            models[name] = None

    # Evaluate models
    print("\n" + "=" * 60)
    print("EVALUATING MODELS")
    print("=" * 60)

    results = {}

    # ANN (uses 2D data)
    if models["ANN"] is not None:
        results["ANN"] = evaluate_model(
            models["ANN"], X_test_scaled, y_test_scaled_ann,
            y_test_original_ann, scaler_y, "ANN"
        )

    # RNN models (use 3D sequences)
    for name in ["GRU", "LSTM", "Transformer"]:
        if models[name] is not None:
            results[name] = evaluate_model(
                models[name], X_test_seq, y_test_seq,
                y_test_original_seq, scaler_y, name
            )

    # Ensemble
    if any(m is not None for m in models.values()):
        results["ENSEMBLE"] = evaluate_ensemble(
            models, X_test_scaled, X_test_seq, y_test_seq,
            y_test_original_seq, test_indices_ann, test_indices_seq,
            scaler_y
        )

    # Save results
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nMetrics saved to: {REPORT_PATH}")
    print("\nFinal Results (Original Scale):")
    for model_name, metrics in results.items():
        if "MAE_original" in metrics:
            print(f"  {model_name:<15} MAE: {metrics['MAE_original']:.4f}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
