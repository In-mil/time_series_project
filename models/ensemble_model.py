#!/usr/bin/env python3
"""Ensemble model script ported from notebooks/2_Modeling_Ensemble.ipynb.

This script combines predictions from ANN, GRU, LSTM, and Transformer models
to create an ensemble prediction for crypto price movements.
"""
import mlflow
import os

# Use environment variable if set, otherwise default to local server
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.set_experiment("ensemble_experiment")

import joblib
from pathlib import Path
import datetime
import random
from typing import List, Tuple, Optional, Dict
import json


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Project paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "final_data" / "20251115_dataset_crp.csv"
MODELS_DIR = REPO_ROOT / "models"

ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "ensemble"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# Random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Create timestamped output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
FIG_DIR = REPO_ROOT / "figures" / f"ensemble_run_{timestamp}"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split_data(
    data_path: Path,
    train_split: str = "2024-07-01",
    val_split: str = "2024-10-01",
    test_split: str = "2024-10-01"  # Fixed: removed 3-month gap
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data and split into train/validation/test sets.

    Args:
        data_path: Path to the CSV file
        train_split: Date to split training data
        val_split: Date to split validation data
        test_split: Date to split test data

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Error handling for file operations
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    df_train = df[df['date'] < train_split].copy()
    df_val = df[(df['date'] >= train_split) & (df['date'] < val_split)].copy()
    df_test = df[df['date'] >= test_split].copy()

    print(f"\nData Split:")
    print(f"  Training set: {len(df_train):,} samples")
    print(f"  Validation set: {len(df_val):,} samples")
    print(f"  Test set: {len(df_test):,} samples")

    return df_train, df_val, df_test


def prepare_features_and_targets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "future_5_close_higher_than_today"
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str]]:
    """Prepare features and target variables.

    Args:
        df_train: Training dataframe
        df_val: Validation dataframe
        df_test: Test dataframe
        target_col: Name of the target column

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, feature_columns)
    """
    # Target columns to exclude
    target_columns = [
        'future_5_close_higher_than_today',
        'future_10_close_higher_than_today',
        'future_5_close_lower_than_today',
        'future_10_close_lower_than_today',
        'higher_close_today_vs_future_5_close',
        'higher_close_today_vs_future_10_close',
        'lower_close_today_vs_future_5_close',
        'lower_close_today_vs_future_10_close'
    ]

    # Get feature columns
    feature_columns = [col for col in df_train.columns if col not in target_columns]

    # Split features and target
    X_train = df_train[feature_columns].copy()
    y_train = df_train[target_col].values

    X_val = df_val[feature_columns].copy()
    y_val = df_val[target_col].values

    X_test = df_test[feature_columns].copy()
    y_test = df_test[target_col].values

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns


def scale_data(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Scale features and targets.

    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Data to scale

    Returns:
        Tuple of scaled data and fitted scalers
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Columns to scale (exclude ticker and date)
    columns_to_scale = [col for col in X_train.columns if col not in ['ticker', 'date']]

    # Fit and transform
    X_train_scaled = scaler_X.fit_transform(X_train[columns_to_scale])
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    X_val_scaled = scaler_X.transform(X_val[columns_to_scale])
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    X_test_scaled = scaler_X.transform(X_test[columns_to_scale])
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print(f"\nScaled Data:")
    print(f"  Features shape: {X_train_scaled.shape}")
    print(f"  Target shape: {y_train_scaled.shape}")

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, \
           X_test_scaled, y_test_scaled, scaler_X, scaler_y


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    tickers: pd.Series,
    dates: pd.Series,
    indices: pd.Index,
    look_back: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences for RNN models (GRU, LSTM, Transformer).

    Args:
        X: Feature array
        y: Target array
        tickers: Ticker symbols
        dates: Date column
        indices: Original dataframe indices
        look_back: Number of time steps to look back

    Returns:
        Tuple of (X_sequences, y_sequences, sequence_indices)
    """
    X_sequences = []
    y_sequences = []
    sequence_indices = []

    for ticker in tickers.unique():
        is_ticker = (tickers == ticker).values
        ticker_features = X[is_ticker]
        ticker_targets = y[is_ticker]
        ticker_dates = dates[is_ticker].values
        ticker_rows = indices[is_ticker].values

        # Sort by date
        date_order = np.argsort(ticker_dates)
        ticker_features = ticker_features[date_order]
        ticker_targets = ticker_targets[date_order]
        ticker_rows = ticker_rows[date_order]

        # Create sequences
        num_sequences = len(ticker_features) - look_back
        for i in range(num_sequences):
            sequence = ticker_features[i : i + look_back]
            target = ticker_targets[i + look_back]
            row = ticker_rows[i + look_back]

            X_sequences.append(sequence)
            y_sequences.append(target)
            sequence_indices.append(row)

    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)
    seq_indices = np.array(sequence_indices)

    print(f"\nSequence Data:")
    print(f"  Shape: {X_seq.shape}")
    print(f"  Number of sequences: {len(X_seq):,}")

    return X_seq, y_seq, seq_indices


def load_models(models_dir: Path) -> Dict[str, Optional[tf.keras.Model]]:
    """Load all saved models.

    Args:
        models_dir: Directory containing saved models

    Returns:
        Dictionary mapping model names to loaded models (None if not found)
    """
    models = {}
    model_files = {
        'ANN': 'model_ann.keras',
        'GRU': 'model_gru.keras',
        'LSTM': 'model_lstm.keras',
        'Transformer': 'model_transformer.keras'
    }

    print("\nLoading Models:")
    for name, filename in model_files.items():
        model_path = models_dir / filename
        try:
            models[name] = load_model(model_path)
            print(f"  ✓ {name} model loaded from {filename}")
        except Exception as e:
            print(f"  ✗ {name} model not found: {e}")
            models[name] = None

    return models


def align_ann_predictions(
    pred_ann: np.ndarray,
    test_indices_ann: np.ndarray,
    test_indices_seq: np.ndarray
) -> np.ndarray:
    """Align ANN predictions to match sequence model predictions.

    Args:
        pred_ann: ANN predictions
        test_indices_ann: Indices for ANN test data
        test_indices_seq: Indices for sequence test data

    Returns:
        Aligned ANN predictions
    """
    ann_positions = []
    for seq_idx in test_indices_seq:
        position = np.where(test_indices_ann == seq_idx)[0]
        if len(position) > 0:
            ann_positions.append(position[0])

    pred_ann_aligned = pred_ann[ann_positions]
    print(f"\nANN predictions aligned: {pred_ann_aligned.shape}")

    return pred_ann_aligned


def create_ensemble_predictions(
    models: Dict[str, Optional[tf.keras.Model]],
    X_test_ann: np.ndarray,
    X_test_seq: np.ndarray,
    test_indices_ann: np.ndarray,
    test_indices_seq: np.ndarray
) -> Tuple[Optional[np.ndarray], List[str], Dict[str, np.ndarray]]:
    """Generate ensemble predictions from all models.

    Args:
        models: Dictionary of loaded models
        X_test_ann: Test data for ANN
        X_test_seq: Test data for sequence models
        test_indices_ann: Indices for ANN test data
        test_indices_seq: Indices for sequence test data

    Returns:
        Tuple of (ensemble_predictions, model_names, individual_predictions)
    """
    all_predictions = []
    model_names = []
    individual_preds = {}

    print("\nGenerating Predictions:")

    # ANN predictions
    if models['ANN'] is not None:
        pred_ann = models['ANN'].predict(X_test_ann, verbose=0)
        pred_ann_aligned = align_ann_predictions(pred_ann, test_indices_ann, test_indices_seq)
        all_predictions.append(pred_ann_aligned)
        model_names.append('ANN')
        individual_preds['ANN'] = pred_ann_aligned
        print(f"  ✓ ANN predictions: {pred_ann.shape}")

    # GRU predictions
    if models['GRU'] is not None:
        pred_gru = models['GRU'].predict(X_test_seq, verbose=0)
        all_predictions.append(pred_gru)
        model_names.append('GRU')
        individual_preds['GRU'] = pred_gru
        print(f"  ✓ GRU predictions: {pred_gru.shape}")

    # LSTM predictions
    if models['LSTM'] is not None:
        pred_lstm = models['LSTM'].predict(X_test_seq, verbose=0)
        all_predictions.append(pred_lstm)
        model_names.append('LSTM')
        individual_preds['LSTM'] = pred_lstm
        print(f"  ✓ LSTM predictions: {pred_lstm.shape}")

    # Transformer predictions
    if models['Transformer'] is not None:
        pred_transformer = models['Transformer'].predict(X_test_seq, verbose=0)
        all_predictions.append(pred_transformer)
        model_names.append('Transformer')
        individual_preds['Transformer'] = pred_transformer
        print(f"  ✓ Transformer predictions: {pred_transformer.shape}")

    # Create ensemble
    if len(all_predictions) > 0:
        stacked_predictions = np.stack(all_predictions, axis=0)
        ensemble_pred = np.mean(stacked_predictions, axis=0)
        print(f"\n  ✓ Ensemble predictions: {ensemble_pred.shape}")
        print(f"    Created by averaging {len(all_predictions)} models: {model_names}")
    else:
        print("\n  ✗ No models available for ensemble!")
        ensemble_pred = None

    return ensemble_pred, model_names, individual_preds


def evaluate_predictions(
    y_true_scaled: np.ndarray,
    predictions_scaled: Dict[str, np.ndarray],
    ensemble_pred_scaled: Optional[np.ndarray],
    scaler_y: StandardScaler,
    y_true_original: np.ndarray
) -> Dict[str, float]:
    """Evaluate predictions and calculate metrics.

    Args:
        y_true_scaled: True values (scaled)
        predictions_scaled: Individual model predictions (scaled)
        ensemble_pred_scaled: Ensemble predictions (scaled)
        scaler_y: Target scaler for inverse transform
        y_true_original: True values (original scale)

    Returns:
        Dictionary of MAE values for each model
    """
    mae_values = {}

    print("\n" + "="*50)
    print("MODEL PERFORMANCE (Original Scale)")
    print("="*50)

    # Evaluate individual models
    for name, pred_scaled in predictions_scaled.items():
        pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        mae = mean_absolute_error(y_true_original, pred_original)
        mae_values[name] = mae
        print(f"{name:<15} MAE: {mae:.4f}")

    # Evaluate ensemble
    if ensemble_pred_scaled is not None:
        ensemble_pred_original = scaler_y.inverse_transform(
            ensemble_pred_scaled.reshape(-1, 1)
        ).ravel()
        mae = mean_absolute_error(y_true_original, ensemble_pred_original)
        mse = mean_squared_error(y_true_original, ensemble_pred_original)
        mae_values['ENSEMBLE'] = mae

        print(f"\n{'ENSEMBLE':<15} MAE: {mae:.4f}")
        print(f"{'ENSEMBLE':<15} MSE: {mse:.4f}")
        print("="*50)

    return mae_values


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path
) -> None:
    """Plot predicted vs actual values.

    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values (Scaled)')
    plt.ylabel('Predicted Values (Scaled)')
    plt.title('Ensemble: Predicted vs Actual (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path
) -> None:
    """Plot residuals analysis.

    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save the figure
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals histogram
    axes[0].hist(residuals, bins=50, edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residual (Actual - Predicted)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Ensemble: Distribution of Residuals')
    axes[0].grid(True, alpha=0.3)

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Ensemble: Residuals vs Predicted')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Saved: {output_path.name}")
    print(f"\n  Residual Statistics:")
    print(f"    Mean: {residuals.mean():.4f}")
    print(f"    Std:  {residuals.std():.4f}")
    print(f"    Min:  {residuals.min():.4f}")
    print(f"    Max:  {residuals.max():.4f}")


def plot_model_comparison(
    mae_values: Dict[str, float],
    output_path: Path
) -> None:
    """Plot comparison of all models.

    Args:
        mae_values: Dictionary of MAE values for each model
        output_path: Path to save the figure
    """
    labels = list(mae_values.keys())
    values = list(mae_values.values())

    # Color ensemble bar differently
    colors = ['skyblue'] * (len(labels) - 1) + ['gold'] if 'ENSEMBLE' in labels else ['skyblue'] * len(labels)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')

    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path.name}")


def main() -> None:
    """Run the ensemble modeling pipeline."""
    print("="*50)
    print("ENSEMBLE MODEL PIPELINE")
    print("="*50)

    # Load and split data
    df_train, df_val, df_test = load_and_split_data(DATA_PATH)

    # Prepare features and targets
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = \
        prepare_features_and_targets(df_train, df_val, df_test)

    # Scale data
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, \
    X_test_scaled, y_test_scaled, scaler_X, scaler_y = \
        scale_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # Prepare data for ANN (no sequences)
    X_test_ann = X_test_scaled
    y_test_ann = y_test_scaled
    test_indices_ann = X_test.index.values

    # Create sequences for RNN models
    X_test_seq, y_test_seq, test_indices_seq = create_sequences(
        X_test_scaled,
        y_test_scaled,
        X_test['ticker'],
        X_test['date'],
        X_test.index,
        look_back=20
    )

    # Load models
    models = load_models(MODELS_DIR)

    # Generate predictions
    ensemble_pred_scaled, model_names, individual_preds = create_ensemble_predictions(
        models,
        X_test_ann,
        X_test_seq,
        test_indices_ann,
        test_indices_seq
    )

    if ensemble_pred_scaled is None:
        print("\nERROR: No models available for ensemble. Exiting.")
        return

    # Get original scale values for evaluation
    y_test_original = df_test.loc[test_indices_seq, 'future_5_close_higher_than_today'].values

    # Evaluate predictions
    mae_values = evaluate_predictions(
        y_test_seq,
        individual_preds,
        ensemble_pred_scaled,
        scaler_y,
        y_test_original
    )

    # Generate visualizations
    print("\nGenerating Visualizations:")

    plot_predictions_vs_actual(
        y_test_seq,
        ensemble_pred_scaled,
        FIG_DIR / "ensemble_predictions_vs_actual.png"
    )

    ensemble_pred_original = scaler_y.inverse_transform(
        ensemble_pred_scaled.reshape(-1, 1)
    ).ravel()

    plot_residuals(
        y_test_original,
        ensemble_pred_original,
        FIG_DIR / "ensemble_residuals.png"
    )

    plot_model_comparison(
        mae_values,
        FIG_DIR / "model_comparison.png"
    )

    # Save ensemble artifacts
    scaler_X_path = ARTIFACTS_DIR / "scaler_X.pkl"
    scaler_y_path = ARTIFACTS_DIR / "scaler_y.pkl"
    ensemble_meta_path = ARTIFACTS_DIR / "ensemble_meta.json"

    # Save scaler
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

    # Metadaten für das Ensemble speichern
    ensemble_meta = {
        "base_models": model_names,  # z.B. ["ANN", "GRU", "LSTM", "Transformer"]
        "look_back": 20,
        "target_column": "future_5_close_higher_than_today",
        "created_at": datetime.datetime.now().isoformat()

    }
    with open(ensemble_meta_path, "w") as f:
        json.dump(ensemble_meta, f, indent=2)

    # Save ensemble configuration as "model" for DVC tracking
    MODEL_PATH = REPO_ROOT / "models" / "model_ensemble.json"
    ensemble_config = {
        "model_type": "ensemble",
        "base_models": [f"models/model_{name.lower()}.keras" for name in model_names],
        "averaging_method": "mean",
        "look_back": 20,
        "target_column": "future_5_close_higher_than_today",
        "created_at": datetime.datetime.now().isoformat(),
        "performance": {
            model_name: float(mae_val) for model_name, mae_val in mae_values.items()
        }
    }
    with open(MODEL_PATH, "w") as f:
        json.dump(ensemble_config, f, indent=2)

    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")
    print(f"Ensemble configuration saved to: {MODEL_PATH}")
    print(f"\nAll outputs saved to: {FIG_DIR}")

    # MLflow Tracking
    with mlflow.start_run(run_name=f"ensemble_{timestamp}"):
        mlflow.log_param("look_back", 20)
        mlflow.log_param("base_models", model_names)
        mlflow.log_param("target_column", "future_5_close_higher_than_today")

        # Log metrics
        for model_name, mae in mae_values.items():
            mlflow.log_metric(f"{model_name}_MAE", mae)

        # Log artifacts (plots + scalers)
        mlflow.log_artifacts(str(FIG_DIR), artifact_path="figures")
        mlflow.log_artifacts(str(ARTIFACTS_DIR), artifact_path="artifacts")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()