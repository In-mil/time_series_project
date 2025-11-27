#!/usr/bin/env python3
"""
GRU Model for Crypto Price Prediction

Target: Predicting whether the crypto price will be higher in 5 days than today
Method: Gated Recurrent Unit (GRU) - Recurrent Neural Network with TensorFlow/Keras

Usage:
    python model_gru.py --data-path /path/to/data.csv
    python model_gru.py --epochs 50 --batch-size 128
"""

import argparse
import datetime
import os
import random
from pathlib import Path

# Disable TensorFlow verbose logging (must be before TF import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import mlflow

# Default seed for reproducibility
SEED = 42


def set_seeds(seed: int = SEED):
    """Set random seeds for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and validate data from CSV."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)


def split_data(df: pd.DataFrame, split_date_train: str, split_date_val: str, split_date_test: str):
    """Split data into train, validation, and test sets based on dates."""
    df_train = df[df['date'] < split_date_train].copy()
    df_val = df[(df['date'] >= split_date_train) & (df['date'] < split_date_val)].copy()
    df_test = df[df['date'] >= split_date_test].copy()

    print(f"\nTraining set: {len(df_train)} samples")
    print(f"Validation set: {len(df_val)} samples")
    print(f"Test set: {len(df_test)} samples")
    print(f"Train up to date: {df_train['date'].max()}")
    print(f"Validation from date: {df_val['date'].min()} to {df_val['date'].max()}")
    print(f"Test from date: {df_test['date'].min()}")

    return df_train, df_val, df_test


def prepare_features(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, target_col: str):
    """Prepare features and targets, apply scaling."""
    # Target columns to exclude
    target_cols = [
        'future_5_close_higher_than_today', 'future_10_close_higher_than_today',
        'future_5_close_lower_than_today', 'future_10_close_lower_than_today',
        'higher_close_today_vs_future_5_close', 'higher_close_today_vs_future_10_close',
        'lower_close_today_vs_future_5_close', 'lower_close_today_vs_future_10_close'
    ]

    # Features: exclude targets but KEEP 'ticker' and 'date' for sequence creation
    feature_cols = [col for col in df_train.columns if col not in target_cols]

    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_col].values

    X_val = df_val[feature_cols].copy()
    y_val = df_val[target_col].values

    X_test = df_test[feature_cols].copy()
    y_test = df_test[target_col].values

    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Columns to scale (exclude ticker and date)
    cols_to_scale = [col for col in X_train.columns if col not in ['ticker', 'date']]

    # Fit on training data only (avoid leakage)
    X_train_scaled = scaler_X.fit_transform(X_train[cols_to_scale])
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Transform validation and test data
    X_val_scaled = scaler_X.transform(X_val[cols_to_scale])
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    X_test_scaled = scaler_X.transform(X_test[cols_to_scale])
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
            X_test_scaled, y_test_scaled, X_train, X_val, X_test, df_test, scaler_y)


def create_sequences(X_data: np.ndarray, y_data: np.ndarray, df: pd.DataFrame, look_back: int = 20):
    """Create sliding window sequences for each crypto separately.

    Returns sequences AND original indices for alignment.
    """
    X_sequences = []
    y_sequences = []
    original_indices = []

    # Process each cryptocurrency separately
    for ticker in df['ticker'].unique():
        # Get data for this crypto only
        mask = df['ticker'] == ticker
        ticker_indices = df[mask].index.values
        crypto_X = X_data[mask]
        crypto_y = y_data[mask]
        crypto_dates = df.loc[mask, 'date'].values

        # Sort by date (important for time series!)
        order = np.argsort(crypto_dates)
        crypto_X = crypto_X[order]
        crypto_y = crypto_y[order]
        ticker_indices = ticker_indices[order]

        # Create sliding windows
        for i in range(len(crypto_X) - look_back):
            X_sequences.append(crypto_X[i:i + look_back])
            y_sequences.append(crypto_y[i + look_back])
            original_indices.append(ticker_indices[i + look_back])

    return np.array(X_sequences), np.array(y_sequences), np.array(original_indices)


def build_model(look_back: int, n_features: int, gru_units: int, gru_dropout: float,
                use_second_gru: bool, learning_rate: float, l2_reg: float) -> tf.keras.Model:
    """Build and compile the GRU model."""
    layers = [tf.keras.layers.Input(shape=(look_back, n_features))]

    if use_second_gru:
        layers.extend([
            tf.keras.layers.GRU(
                gru_units,
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name='gru_layer_1'
            ),
            tf.keras.layers.Dropout(gru_dropout, name='dropout_1'),
            tf.keras.layers.GRU(
                gru_units // 2,
                return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name='gru_layer_2'
            ),
            tf.keras.layers.Dropout(gru_dropout, name='dropout_2'),
        ])
    else:
        layers.extend([
            tf.keras.layers.GRU(
                gru_units,
                return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name='gru_layer'
            ),
            tf.keras.layers.Dropout(gru_dropout, name='dropout'),
        ])

    layers.append(tf.keras.layers.Dense(1, activation='linear', name='output'))

    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=['mae', 'mse']
    )

    print("\nGRU architecture:")
    model.summary()

    return model


def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: int, batch_size: int,
                patience: int) -> tf.keras.callbacks.History:
    """Train the model with early stopping."""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    return history


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test_seq: np.ndarray,
                   test_indices: np.ndarray, df_test: pd.DataFrame, target_col: str,
                   scaler_y: StandardScaler, history: tf.keras.callbacks.History):
    """Evaluate model and return metrics."""
    # Training metrics from last epoch
    mae_train = history.history['mae'][-1]
    mse_train = history.history['mse'][-1]
    val_mae = history.history['val_mae'][-1]
    val_mse = history.history['val_mse'][-1]

    print("\nTrain Set Performance:")
    print(f"   MAE:  {mae_train:.4f}")
    print(f"   MSE:  {mse_train:.4f}")

    print("\nValidation Set Performance:")
    print(f"   MAE:  {val_mae:.4f}")
    print(f"   MSE:  {val_mse:.4f}")

    # Test predictions
    y_pred_scaled = model.predict(X_test)

    # Metrics on scaled data
    test_mae_scaled = mean_absolute_error(y_test_seq, y_pred_scaled)
    test_mse_scaled = mean_squared_error(y_test_seq, y_pred_scaled)

    print("\nTest Set Performance (Scaled):")
    print(f"   MAE:  {test_mae_scaled:.4f}")
    print(f"   MSE:  {test_mse_scaled:.4f}")

    # Inverse transform to original scale
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_original = df_test.loc[test_indices, target_col].values

    # Metrics on original scale
    test_mae_original = mean_absolute_error(y_test_original, y_pred_original)
    test_mse_original = mean_squared_error(y_test_original, y_pred_original)

    print("\nTest Set Performance (Original Scale):")
    print(f"   MAE:  {test_mae_original:.4f}")
    print(f"   MSE:  {test_mse_original:.4f}")

    return {
        'train_mae': mae_train,
        'train_mse': mse_train,
        'val_mae': val_mae,
        'val_mse': val_mse,
        'test_mae_scaled': test_mae_scaled,
        'test_mse_scaled': test_mse_scaled,
        'test_mae_original': test_mae_original,
        'test_mse_original': test_mse_original,
    }


def log_to_mlflow(params: dict, metrics: dict, mlflow_uri: str, experiment_name: str):
    """Log parameters and metrics to MLflow."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"gru_{timestamp}"):
        for key, value in params.items():
            mlflow.log_param(key, value)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        print(f"\nMLflow run logged with name: gru_{timestamp}")


def main():
    parser = argparse.ArgumentParser(description='Train GRU model for crypto price prediction')

    # Data arguments
    parser.add_argument(
        '--data-path',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'final_data' / '20251115_dataset_crp.csv',
        help='Path to the dataset CSV file'
    )
    parser.add_argument('--split-date-train', default='2024-07-01', help='Training split date')
    parser.add_argument('--split-date-val', default='2024-10-01', help='Validation split date')
    parser.add_argument('--split-date-test', default='2024-10-01', help='Test split date')
    parser.add_argument('--target-col', default='future_5_close_higher_than_today', help='Target column')

    # Model architecture arguments
    parser.add_argument('--gru-units', type=int, default=64, help='GRU units in first layer')
    parser.add_argument('--gru-dropout', type=float, default=0.3, help='Dropout rate for GRU layers')
    parser.add_argument('--no-second-gru', action='store_true', help='Disable second GRU layer')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--l2-reg', type=float, default=0.0001, help='L2 regularization strength')
    parser.add_argument('--look-back', type=int, default=20, help='Number of past days for sequences')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    # MLflow arguments
    parser.add_argument('--mlflow-uri', default='http://127.0.0.1:5001', help='MLflow tracking URI')
    parser.add_argument('--experiment-name', default='gru_experiment', help='MLflow experiment name')

    args = parser.parse_args()

    # Set seeds
    set_seeds(args.seed)

    # Load data
    print("Loading data...")
    df = load_data(args.data_path)

    # Split data
    df_train, df_val, df_test = split_data(
        df, args.split_date_train, args.split_date_val, args.split_date_test
    )

    # Prepare features
    (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
     X_test_scaled, y_test_scaled, X_train_df, X_val_df, X_test_df,
     df_test, scaler_y) = prepare_features(df_train, df_val, df_test, args.target_col)

    # Create sequences
    print("\nCreating sequences...")
    X_train_seq, y_train_seq, _ = create_sequences(
        X_train_scaled, y_train_scaled, X_train_df, args.look_back
    )
    X_val_seq, y_val_seq, _ = create_sequences(
        X_val_scaled, y_val_scaled, X_val_df, args.look_back
    )
    X_test_seq, y_test_seq, test_indices = create_sequences(
        X_test_scaled, y_test_scaled, X_test_df, args.look_back
    )

    n_features = X_train_seq.shape[2]
    print(f"Shape: {X_train_seq.shape}")
    print(f"   {X_train_seq.shape[0]} sequences, {args.look_back} days look-back, {n_features} features")

    # Build model
    model = build_model(
        look_back=args.look_back,
        n_features=n_features,
        gru_units=args.gru_units,
        gru_dropout=args.gru_dropout,
        use_second_gru=not args.no_second_gru,
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg
    )

    # Train model
    history = train_model(
        model, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        args.epochs, args.batch_size, args.patience
    )

    # Evaluate model
    metrics = evaluate_model(
        model, X_test_seq, y_test_seq, test_indices, df_test,
        args.target_col, scaler_y, history
    )

    # Log to MLflow
    params = {
        'gru_units': args.gru_units,
        'gru_dropout': args.gru_dropout,
        'use_second_gru': not args.no_second_gru,
        'learning_rate': args.learning_rate,
        'L2_regularization': args.l2_reg,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'look_back': args.look_back,
    }
    log_to_mlflow(params, metrics, args.mlflow_uri, args.experiment_name)


if __name__ == "__main__":
    main()
