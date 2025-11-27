#!/usr/bin/env python3
"""
Transformer Model for Crypto Price Prediction

Target: Predicting whether the crypto price will be higher in 5 days than today
Method: Transformer - Self-Attention Neural Network with TensorFlow/Keras

Usage:
    python model_transformer.py --data-path /path/to/data.csv
    python model_transformer.py --epochs 50 --batch-size 128
"""

import argparse
import datetime
import os
import random
from pathlib import Path

# Disable TensorFlow verbose logging (must be before TF import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# IMPORTANT: TensorFlow must be imported BEFORE pandas to avoid hangs on Mac
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D
)
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Note: mlflow is imported lazily in log_to_mlflow() to avoid TensorFlow conflicts

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
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(1)
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


def create_positional_encoding(look_back: int, n_features: int) -> tf.Tensor:
    """Create positional encoding for transformer."""
    position_numbers = np.arange(look_back)[:, np.newaxis]
    even_numbers = np.arange(0, n_features, 2)
    frequencies = np.exp(even_numbers * -(np.log(10000.0) / n_features))

    position_encoding_table = np.zeros((look_back, n_features))
    position_encoding_table[:, 0::2] = np.sin(position_numbers * frequencies)
    position_encoding_table[:, 1::2] = np.cos(position_numbers * frequencies)

    return tf.constant(position_encoding_table, dtype=tf.float32)


def build_model(look_back: int, n_features: int, head_size: int, num_heads: int,
                ff_dim: int, dropout_rate: float, mlp_dropout: float,
                learning_rate: float) -> tf.keras.Model:
    """Build and compile the Transformer model."""
    # Create positional encoding
    positional_encoding = create_positional_encoding(look_back, n_features)

    # Input layer
    inputs = Input(shape=(look_back, n_features))

    # Add positional encoding
    x = inputs + positional_encoding

    # Transformer Block 1
    attention_1 = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout_rate
    )(x, x)
    attention_1 = Dropout(dropout_rate)(attention_1)
    attention_1 = x + attention_1
    attention_1 = LayerNormalization(epsilon=1e-6)(attention_1)

    ffn_1 = Dense(ff_dim, activation="relu")(attention_1)
    ffn_1 = Dropout(dropout_rate)(ffn_1)
    ffn_1 = Dense(n_features)(ffn_1)
    ffn_1 = Dropout(dropout_rate)(ffn_1)
    ffn_1 = attention_1 + ffn_1
    block_1_output = LayerNormalization(epsilon=1e-6)(ffn_1)

    # Transformer Block 2
    attention_2 = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout_rate
    )(block_1_output, block_1_output)
    attention_2 = Dropout(dropout_rate)(attention_2)
    attention_2 = block_1_output + attention_2
    attention_2 = LayerNormalization(epsilon=1e-6)(attention_2)

    ffn_2 = Dense(ff_dim, activation="relu")(attention_2)
    ffn_2 = Dropout(dropout_rate)(ffn_2)
    ffn_2 = Dense(n_features)(ffn_2)
    ffn_2 = Dropout(dropout_rate)(ffn_2)
    ffn_2 = attention_2 + ffn_2
    block_2_output = LayerNormalization(epsilon=1e-6)(ffn_2)

    # Pool across time dimension
    pooled = GlobalAveragePooling1D()(block_2_output)

    # Final prediction layers
    prediction = Dense(64, activation="relu")(pooled)
    prediction = Dropout(mlp_dropout)(prediction)
    outputs = Dense(1, activation="linear")(prediction)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mae", "mse"]
    )

    print("\nTransformer architecture:")
    model.summary()

    return model


def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: int, batch_size: int,
                patience: int) -> tf.keras.callbacks.History:
    """Train the model with early stopping and learning rate reduction."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
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


def log_to_mlflow(params: dict, metrics: dict, model: tf.keras.Model,
                  mlflow_uri: str, experiment_name: str):
    """Log parameters, metrics, and model to MLflow."""
    import mlflow
    mlflow.autolog(disable=True)
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"transformer_{timestamp}"):
        for key, value in params.items():
            mlflow.log_param(key, value)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model to GCS (using legacy method for server compatibility)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.keras"
            model.save(model_path)
            mlflow.log_artifact(model_path, "model")
        print(f"\nMLflow run logged with name: transformer_{timestamp}")
        print("Model artifact saved to GCS")


def main():
    parser = argparse.ArgumentParser(description='Train Transformer model for crypto price prediction')

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
    parser.add_argument('--head-size', type=int, default=128, help='Attention head size')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--ff-dim', type=int, default=128, help='Feed-forward dimension')
    parser.add_argument('--dropout-rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--mlp-dropout', type=float, default=0.3, help='MLP dropout rate')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--look-back', type=int, default=20, help='Number of past days for sequences')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    # MLflow arguments
    parser.add_argument('--mlflow-uri', default='https://mlflow-server-101264457040.europe-west3.run.app', help='MLflow tracking URI')
    parser.add_argument('--experiment-name', default='transformer_experiment', help='MLflow experiment name')

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
        head_size=args.head_size,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout_rate=args.dropout_rate,
        mlp_dropout=args.mlp_dropout,
        learning_rate=args.learning_rate
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
        'head_size': args.head_size,
        'num_heads': args.num_heads,
        'ff_dim': args.ff_dim,
        'dropout_rate': args.dropout_rate,
        'mlp_dropout': args.mlp_dropout,
        'num_blocks': 2,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'patience': args.patience,
        'look_back': args.look_back,
    }
    log_to_mlflow(params, metrics, model, args.mlflow_uri, args.experiment_name)


if __name__ == "__main__":
    main()
