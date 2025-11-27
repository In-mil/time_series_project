#!/usr/bin/env python3
"""
LSTM Model for Crypto Price Prediction

Target: Predicting whether the crypto price will be higher in 5 days than today
Method: Long Short-Term Memory (LSTM) neural network with TensorFlow/Keras

Usage:
    python model_lstm.py --data-path /path/to/data.csv
    python model_lstm.py --epochs 50 --batch-size 128
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


def warmup_tensorflow():
    """Pre-initialize TensorFlow to speed up first epoch."""
    print("Warming up TensorFlow...")
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(10,))
    ])
    dummy_model.predict(np.zeros((1, 10)), verbose=0)
    del dummy_model
    print("TensorFlow ready.")


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


def build_model(look_back: int, n_features: int, lstm_units: int, lstm_dropout: float,
                use_second_lstm: bool, learning_rate: float, l2_reg: float) -> tf.keras.Model:
    """Build and compile the LSTM model."""
    layers = [tf.keras.layers.Input(shape=(look_back, n_features))]

    if use_second_lstm:
        layers.extend([
            tf.keras.layers.LSTM(
                lstm_units,
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name='lstm_layer_1'
            ),
            tf.keras.layers.Dropout(lstm_dropout, name='dropout_1'),
            tf.keras.layers.LSTM(
                lstm_units // 2,
                return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name='lstm_layer_2'
            ),
            tf.keras.layers.Dropout(lstm_dropout, name='dropout_2'),
        ])
    else:
        layers.extend([
            tf.keras.layers.LSTM(
                lstm_units,
                return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name='lstm_layer'
            ),
            tf.keras.layers.Dropout(lstm_dropout, name='dropout'),
        ])

    layers.append(tf.keras.layers.Dense(1, activation='linear', name='output'))

    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=['mae', 'mse']
    )

    print("\nLSTM architecture:")
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


def evaluate_model(model: tf.keras.Model,
                   X_train_seq: np.ndarray, X_val_seq: np.ndarray, X_test_seq: np.ndarray,
                   train_indices: np.ndarray, val_indices: np.ndarray, test_indices: np.ndarray,
                   df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                   target_col: str, scaler_y: StandardScaler):
    """Evaluate model and return metrics on original scale."""
    # Train predictions
    y_train_pred = scaler_y.inverse_transform(
        model.predict(X_train_seq, verbose=0).reshape(-1, 1)
    ).ravel()
    y_train_original = df_train.loc[train_indices, target_col].values
    train_mae = mean_absolute_error(y_train_original, y_train_pred)
    train_mse = mean_squared_error(y_train_original, y_train_pred)

    print("\nTrain Set Performance (Original Scale):")
    print(f"   MAE:  {train_mae:.4f}")
    print(f"   MSE:  {train_mse:.4f}")

    # Validation predictions
    y_val_pred = scaler_y.inverse_transform(
        model.predict(X_val_seq, verbose=0).reshape(-1, 1)
    ).ravel()
    y_val_original = df_val.loc[val_indices, target_col].values
    val_mae = mean_absolute_error(y_val_original, y_val_pred)
    val_mse = mean_squared_error(y_val_original, y_val_pred)

    print("\nValidation Set Performance (Original Scale):")
    print(f"   MAE:  {val_mae:.4f}")
    print(f"   MSE:  {val_mse:.4f}")

    # Test predictions
    y_test_pred = scaler_y.inverse_transform(
        model.predict(X_test_seq, verbose=0).reshape(-1, 1)
    ).ravel()
    y_test_original = df_test.loc[test_indices, target_col].values
    test_mae = mean_absolute_error(y_test_original, y_test_pred)
    test_mse = mean_squared_error(y_test_original, y_test_pred)

    print("\nTest Set Performance (Original Scale):")
    print(f"   MAE:  {test_mae:.4f}")
    print(f"   MSE:  {test_mse:.4f}")

    return {
        'train_mae': train_mae,
        'train_mse': train_mse,
        'val_mae': val_mae,
        'val_mse': val_mse,
        'test_mae': test_mae,
        'test_mse': test_mse,
    }


def log_to_mlflow(params: dict, metrics: dict, mlflow_uri: str, experiment_name: str):
    """Log parameters and metrics to MLflow."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"lstm_{timestamp}"):
        for key, value in params.items():
            mlflow.log_param(key, value)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        print(f"\nMLflow run logged with name: lstm_{timestamp}")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for crypto price prediction')

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
    parser.add_argument('--lstm-units', type=int, default=64, help='LSTM units in first layer')
    parser.add_argument('--lstm-dropout', type=float, default=0.3, help='Dropout rate for LSTM layers')
    parser.add_argument('--no-second-lstm', action='store_true', help='Disable second LSTM layer')
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
    parser.add_argument('--experiment-name', default='lstm_experiment', help='MLflow experiment name')

    args = parser.parse_args()

    # Set seeds
    set_seeds(args.seed)

    # Warmup TensorFlow
    warmup_tensorflow()

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
    X_train_seq, y_train_seq, train_indices = create_sequences(
        X_train_scaled, y_train_scaled, X_train_df, args.look_back
    )
    X_val_seq, y_val_seq, val_indices = create_sequences(
        X_val_scaled, y_val_scaled, X_val_df, args.look_back
    )
    X_test_seq, _, test_indices = create_sequences(
        X_test_scaled, y_test_scaled, X_test_df, args.look_back
    )

    n_features = X_train_seq.shape[2]
    print(f"Shape: {X_train_seq.shape}")
    print(f"   {X_train_seq.shape[0]} sequences, {args.look_back} days look-back, {n_features} features")

    # Build model
    model = build_model(
        look_back=args.look_back,
        n_features=n_features,
        lstm_units=args.lstm_units,
        lstm_dropout=args.lstm_dropout,
        use_second_lstm=not args.no_second_lstm,
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg
    )

    # Train model
    train_model(
        model, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        args.epochs, args.batch_size, args.patience
    )

    # Evaluate model
    metrics = evaluate_model(
        model, X_train_seq, X_val_seq, X_test_seq,
        train_indices, val_indices, test_indices,
        df_train, df_val, df_test, args.target_col, scaler_y
    )

    # Log to MLflow
    params = {
        'lstm_units': args.lstm_units,
        'lstm_dropout': args.lstm_dropout,
        'use_second_lstm': not args.no_second_lstm,
        'learning_rate': args.learning_rate,
        'L2_regularization': args.l2_reg,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'look_back': args.look_back,
    }
    log_to_mlflow(params, metrics, args.mlflow_uri, args.experiment_name)


if __name__ == "__main__":
    main()
