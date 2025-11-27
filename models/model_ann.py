#!/usr/bin/env python3
"""
ANN Model for Crypto Price Prediction

Target: Predicting whether the crypto price will be higher in 5 days than today
Method: Artificial Neural Network (ANN) - Feedforward Neural Network with TensorFlow/Keras

Usage:
    python model_ann.py --data-path /path/to/data.csv
    python model_ann.py --epochs 50 --batch-size 256
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
    # Create dummy model and run inference to trigger JIT compilation
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
    # Columns to drop from features
    x_cols_to_drop = [
        'ticker', 'date', 'future_5_close_higher_than_today', 'future_10_close_higher_than_today',
        'future_5_close_lower_than_today', 'future_10_close_lower_than_today',
        'higher_close_today_vs_future_5_close', 'higher_close_today_vs_future_10_close',
        'lower_close_today_vs_future_5_close', 'lower_close_today_vs_future_10_close'
    ]

    # Prepare features and targets
    X_train = df_train.drop(x_cols_to_drop, axis='columns')
    y_train = df_train[target_col].to_numpy()

    X_val = df_val.drop(x_cols_to_drop, axis='columns')
    y_val = df_val[target_col].to_numpy()

    X_test = df_test.drop(x_cols_to_drop, axis='columns')
    y_test = df_test[target_col].to_numpy()

    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit on training data only (avoid leakage)
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Transform validation and test data
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    X_test_scaled = scaler_X.transform(X_test)

    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
            X_test_scaled, y_train, y_val, y_test, scaler_y)


def build_model(input_shape: int, layer_1_nodes: int, layer_1_dropout: float,
                layer_2_nodes: int, layer_2_dropout: float, use_layer_2: bool,
                learning_rate: float, l2_reg: float) -> tf.keras.Model:
    """Build and compile the ANN model."""
    layers = [
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(
            layer_1_nodes,
            activation='tanh',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='hidden_layer_1'
        ),
        tf.keras.layers.Dropout(layer_1_dropout, name='dropout_1'),
    ]

    if use_layer_2:
        layers.extend([
            tf.keras.layers.Dense(
                layer_2_nodes,
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name='hidden_layer_2'
            ),
            tf.keras.layers.Dropout(layer_2_dropout, name='dropout_2'),
        ])

    layers.append(tf.keras.layers.Dense(1, activation='linear', name='output'))

    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=['mae', 'mse']
    )

    print("\nANN architecture:")
    model.summary()

    return model


def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: int, batch_size: int,
                patience: int) -> tf.keras.callbacks.History:
    """Train the model with early stopping."""
    early_stopping = tf.keras.callbacks.EarlyStopping(
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


def evaluate_model(model: tf.keras.Model, X_train: np.ndarray, X_val: np.ndarray,
                   X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray,
                   y_test: np.ndarray, scaler_y: StandardScaler):
    """Evaluate model and return metrics on original scale."""
    # Train predictions
    y_train_pred = scaler_y.inverse_transform(
        model.predict(X_train, verbose=0).reshape(-1, 1)
    ).ravel()
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)

    print("\nTrain Set Performance (Original Scale):")
    print(f"   MAE:  {train_mae:.4f}")
    print(f"   MSE:  {train_mse:.4f}")

    # Validation predictions
    y_val_pred = scaler_y.inverse_transform(
        model.predict(X_val, verbose=0).reshape(-1, 1)
    ).ravel()
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    print("\nValidation Set Performance (Original Scale):")
    print(f"   MAE:  {val_mae:.4f}")
    print(f"   MSE:  {val_mse:.4f}")

    # Test predictions
    y_test_pred = scaler_y.inverse_transform(
        model.predict(X_test, verbose=0).reshape(-1, 1)
    ).ravel()
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

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

    with mlflow.start_run(run_name=f"ann_{timestamp}"):
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model to GCS (using legacy method for server compatibility)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.keras"
            model.save(model_path)
            mlflow.log_artifact(model_path, "model")
        print(f"\nMLflow run logged with name: ann_{timestamp}")
        print("Model artifact saved to GCS")


def main():
    parser = argparse.ArgumentParser(description='Train ANN model for crypto price prediction')

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
    parser.add_argument('--layer-1-nodes', type=int, default=128, help='Nodes in first hidden layer')
    parser.add_argument('--layer-1-dropout', type=float, default=0.2, help='Dropout rate for first layer')
    parser.add_argument('--layer-2-nodes', type=int, default=64, help='Nodes in second hidden layer')
    parser.add_argument('--layer-2-dropout', type=float, default=0.2, help='Dropout rate for second layer')
    parser.add_argument('--no-layer-2', action='store_true', help='Disable second hidden layer')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--l2-reg', type=float, default=0.0001, help='L2 regularization strength')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    # MLflow arguments
    parser.add_argument('--mlflow-uri', default='https://mlflow-server-101264457040.europe-west3.run.app', help='MLflow tracking URI')
    parser.add_argument('--experiment-name', default='ann_experiment', help='MLflow experiment name')

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
    (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled,
     y_train, y_val, y_test, scaler_y) = prepare_features(df_train, df_val, df_test, args.target_col)

    # Build model
    model = build_model(
        input_shape=X_train_scaled.shape[1],
        layer_1_nodes=args.layer_1_nodes,
        layer_1_dropout=args.layer_1_dropout,
        layer_2_nodes=args.layer_2_nodes,
        layer_2_dropout=args.layer_2_dropout,
        use_layer_2=not args.no_layer_2,
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg
    )

    # Train model
    history = train_model(
        model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
        args.epochs, args.batch_size, args.patience
    )

    # Evaluate model
    metrics = evaluate_model(model, X_train_scaled, X_val_scaled, X_test_scaled,
                            y_train, y_val, y_test, scaler_y)

    # Log to MLflow
    params = {
        'layer_1_nodes': args.layer_1_nodes,
        'layer_2_nodes': args.layer_2_nodes,
        'dropout_1': args.layer_1_dropout,
        'dropout_2': args.layer_2_dropout,
        'learning_rate': args.learning_rate,
        'L2_regularization': args.l2_reg,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
    }
    log_to_mlflow(params, metrics, model, args.mlflow_uri, args.experiment_name)


if __name__ == "__main__":
    main()
