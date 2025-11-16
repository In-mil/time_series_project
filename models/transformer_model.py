#!/usr/bin/env python3
"""Transformer experiment ported from notebooks/2_Modeling_Transformer.ipynb."""
from pathlib import Path
import datetime
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention

import mlflow
import os

# Use environment variable if set, otherwise default to local server
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.set_experiment("transformer_experiment")


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "final_data" / "20251115_dataset_crp.csv"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
FIG_DIR = REPO_ROOT / "figures" / f"transformer_run_{timestamp}"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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


def set_seed(seed: int = 42) -> None:
    """Set pseudo-random seeds for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_sequences(X_data: np.ndarray, y_data: np.ndarray, df: pd.DataFrame, look_back: int = 20):
    """Create sliding windows for each crypto separately."""
    X_sequences = []
    y_sequences = []
    row_indices = []

    for crypto in df["ticker"].unique():
        mask = df["ticker"] == crypto
        crypto_feat = X_data[mask]
        crypto_target = y_data[mask]
        crypto_dates = df.loc[mask, "date"].values
        crypto_rows = df[mask].index.values

        order = np.argsort(crypto_dates)
        crypto_feat = crypto_feat[order]
        crypto_target = crypto_target[order]
        crypto_rows = crypto_rows[order]

        for i in range(len(crypto_feat) - look_back):
            X_sequences.append(crypto_feat[i : i + look_back])
            y_sequences.append(crypto_target[i + look_back])
            row_indices.append(crypto_rows[i + look_back])

    return np.array(X_sequences), np.array(y_sequences), np.array(row_indices)


def positional_encoding(num_steps: int, depth: int) -> tf.Tensor:
    """Create sinusoidal positional encodings."""
    positions = np.arange(num_steps)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]
    angle_rates = 1 / (10000 ** ((2 * (depths // 2)) / depth))
    angle_rads = positions * angle_rates
    encoding = np.zeros((num_steps, depth))
    encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(encoding, dtype=tf.float32)


def build_transformer(look_back: int, n_features: int):
    """Construct the transformer encoder with two attention blocks."""
    inputs = Input(shape=(look_back, n_features), name="transformer_input")
    pos_encoding = positional_encoding(look_back, n_features)
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)
    x = tf.keras.layers.Add()([inputs, pos_encoding])

    head_size = 128
    num_heads = 4
    ff_dim = 128
    dropout_rate = 0.2
    mlp_dropout = 0.3

    for block in range(2):
        attention = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout_rate)(
            x, x
        )
        attention = Dropout(dropout_rate)(attention)
        attention = LayerNormalization(epsilon=1e-6)(x + attention)

        ff = Dense(ff_dim, activation="relu")(attention)
        ff = Dropout(dropout_rate)(ff)
        ff = Dense(n_features)(ff)
        ff = Dropout(dropout_rate)(ff)
        x = LayerNormalization(epsilon=1e-6)(attention + ff)

    pooled = GlobalAveragePooling1D()(x)
    pooled = Dense(64, activation="relu")(pooled)
    pooled = Dropout(mlp_dropout)(pooled)
    outputs = Dense(1, activation="linear")(pooled)

    model = Model(inputs=inputs, outputs=outputs, name="transformer_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="mae",
        metrics=["mae", "mse"],
    )
    model.summary()
    return model


def main() -> None:
    """Run the transformer modeling pipeline."""
    set_seed()

    # Error handling for file operations
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df_trans = pd.read_csv(DATA_PATH)
    print(f"Loaded {df_trans.shape[0]:,} rows, {df_trans.shape[1]} columns")
    print(f"Cryptocurrencies: {df_trans['ticker'].nunique()}")

    split_date_train = "2024-07-01"
    split_date_val = "2024-10-01"
    split_date_test = "2024-10-01"  # Fixed: removed 3-month gap

    df_train = df_trans[df_trans["date"] < split_date_train].copy()
    df_val = df_trans[(df_trans["date"] >= split_date_train) & (df_trans["date"] < split_date_val)].copy()
    df_test = df_trans[df_trans["date"] >= split_date_test].copy()

    X_train = df_train[[col for col in df_train.columns if col not in TARGET_COLUMNS]].copy()
    y_train = df_train["future_5_close_higher_than_today"].values
    X_val = df_val[[col for col in df_val.columns if col not in TARGET_COLUMNS]].copy()
    y_val = df_val["future_5_close_higher_than_today"].values
    X_test = df_test[[col for col in df_test.columns if col not in TARGET_COLUMNS]].copy()
    y_test = df_test["future_5_close_higher_than_today"].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    cols_to_scale = [col for col in X_train.columns if col not in ["ticker", "date"]]
    X_train_scaled = scaler_X.fit_transform(X_train[cols_to_scale])
    X_val_scaled = scaler_X.transform(X_val[cols_to_scale])
    X_test_scaled = scaler_X.transform(X_test[cols_to_scale])

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print(f"Features: {X_train_scaled.shape}, Target: {y_train_scaled.shape}")

    look_back = 20
    X_train_seq, y_train_seq, train_indices = create_sequences(X_train_scaled, y_train_scaled, X_train, look_back)
    X_val_seq, y_val_seq, val_indices = create_sequences(X_val_scaled, y_val_scaled, X_val, look_back)
    X_test_seq, y_test_seq, test_indices = create_sequences(X_test_scaled, y_test_scaled, X_test, look_back)

    n_features = X_train_seq.shape[2]
    print(f"Shape after creating sequences: {X_train_seq.shape}")

    model_transformer = build_transformer(look_back, n_features)

    epochs = 100
    batch_size = 128
    patience = 15
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
    )

    history = model_transformer.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    mae_train = history.history["mae"][-1]
    mse_train = history.history["mse"][-1]
    print("Train Set Performance:")
    print(f"   MAE:  {mae_train:.4f}")
    print(f"   MSE:  {mse_train:.4f}")
    val_mae = history.history["val_mae"][-1]
    val_mse = history.history["val_mse"][-1]
    print("\nValidation Set Performance:")
    print(f"   MAE:  {val_mae:.4f}")
    print(f"   MSE:  {val_mse:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history.history["mae"], label="training mae", linewidth=2)
    axes[0].plot(history.history["val_mae"], label="validation mae", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Transformer: Mean Absolute Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history["mse"], label="training mse", linewidth=2)
    axes[1].plot(history.history["val_mse"], label="validation mse", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Transformer: Mean Squared Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "history_mae_mse.png", bbox_inches="tight")
    plt.close(fig)

    y_pred_scaled = model_transformer.predict(X_test_seq)
    mae = mean_absolute_error(y_test_seq, y_pred_scaled)
    mse = mean_squared_error(y_test_seq, y_pred_scaled)
    print("Test Set Performance (Scaled Data):")
    print(f"   MAE:  {mae:.4f}")
    print(f"   MSE:  {mse:.4f}")

    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_original = df_test.loc[test_indices, "future_5_close_higher_than_today"].values
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    print("Test Set Performance (Original Scale):")
    print(f"   MAE:  {mae:.4f} percentage points")
    print(f"   MSE:  {mse:.4f}")
    print(f"\nNote: {len(y_pred_original)} predictions aligned with original data using indices")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_seq, y_pred_scaled, alpha=0.5, s=10)
    plt.plot(
        [y_test_seq.min(), y_test_seq.max()],
        [y_test_seq.min(), y_test_seq.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    plt.xlabel("Actual Values (Scaled)")
    plt.ylabel("Predicted Values (Scaled)")
    plt.title("Transformer: Predicted vs Actual (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pred_vs_actual_scaled.png", bbox_inches="tight")
    plt.close()

    residuals = y_test_original - y_pred_original
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(residuals, bins=50, edgecolor="black")
    axes[0].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residual (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Transformer: Distribution of Residuals")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(y_pred_original, residuals, alpha=0.5, s=10)
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Transformer: Residuals vs Predicted")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "residuals.png", bbox_inches="tight")
    plt.close(fig)

    print("Residual Stats:")
    print(f"   Mean: {residuals.mean():.4f}")
    print(f"   Std:  {residuals.std():.4f}")
    print(f"   Min:  {residuals.min():.4f}")
    print(f"   Max:  {residuals.max():.4f}")

    # Save model for DVC
    MODEL_PATH = REPO_ROOT / "models" / "model_transformer.keras"
    model_transformer.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # MLflow Tracking
    with mlflow.start_run(run_name=f"transformer_{timestamp}"):
        # Hyperparameters (from build_transformer function)
        mlflow.log_param("head_size", 128)
        mlflow.log_param("num_heads", 4)
        mlflow.log_param("ff_dim", 128)
        mlflow.log_param("dropout_rate", 0.2)
        mlflow.log_param("mlp_dropout", 0.3)
        mlflow.log_param("num_blocks", 2)
        mlflow.log_param("learning_rate", 0.0001)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("patience", patience)
        mlflow.log_param("look_back", look_back)

        mlflow.log_metric("train_mae", mae_train)
        mlflow.log_metric("train_mse", mse_train)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("test_mae_scaled", mean_absolute_error(y_test_seq, y_pred_scaled))
        mlflow.log_metric("test_mse_scaled", mean_squared_error(y_test_seq, y_pred_scaled))
        mlflow.log_metric("test_mae_original", mean_absolute_error(y_test_original, y_pred_original))
        mlflow.log_metric("test_mse_original", mean_squared_error(y_test_original, y_pred_original))

        mlflow.log_artifacts(str(FIG_DIR), artifact_path="figures")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")


if __name__ == "__main__":
    main()
