#!/usr/bin/env python3
"""ANN model script ported from notebooks/2_Modeling_ANN.ipynb."""

from pathlib import Path
import datetime
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow
import os

# Use environment variable if set, otherwise default to local server
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.set_experiment("ann_experiment")


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "final_data" / "20251115_dataset_crp.csv"
(REPO_ROOT / "models").mkdir(exist_ok=True)



timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
FIG_DIR = REPO_ROOT / "figures" / f"ann_run_{timestamp}"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run the ANN modeling pipeline."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # Error handling for file operations
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df_ann = pd.read_csv(DATA_PATH)

    split_date_train = "2024-07-01"
    split_date_val = "2024-10-01"
    split_date_test = "2024-10-01"  # Fixed: removed 3-month gap

    x_cols_to_drop = [
        "ticker",
        "date",
        "future_5_close_higher_than_today",
        "future_10_close_higher_than_today",
        "future_5_close_lower_than_today",
        "future_10_close_lower_than_today",
        "higher_close_today_vs_future_5_close",
        "higher_close_today_vs_future_10_close",
        "lower_close_today_vs_future_5_close",
        "lower_close_today_vs_future_10_close",
    ]

    df_train = df_ann[df_ann["date"] < split_date_train].copy()
    df_val = df_ann[(df_ann["date"] >= split_date_train) & (df_ann["date"] < split_date_val)].copy()
    df_test = df_ann[df_ann["date"] >= split_date_test].copy()

    print(f"\nTraining set: {len(df_train)} samples")
    print(f"Validation set: {len(df_val)} samples")
    print(f"Test set: {len(df_test)} samples")
    print(f"Train up to date: {df_train['date'].max()}")
    print(f"Validation from date: {df_val['date'].min()} to {df_val['date'].max()}")
    print(f"Test from date: {df_test['date'].min()}")

    X_train = df_train.drop(x_cols_to_drop, axis="columns")
    y_train = df_train["future_5_close_higher_than_today"].to_numpy()

    X_val = df_val.drop(x_cols_to_drop, axis="columns")
    y_val = df_val["future_5_close_higher_than_today"].to_numpy()

    X_test = df_test.drop(x_cols_to_drop, axis="columns")
    y_test = df_test["future_5_close_higher_than_today"].to_numpy()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    layer_1_nodes = 128
    layer_1_dropout = 0.2
    layer_2_nodes = 64
    layer_2_dropout = 0.2
    use_layer_2 = True
    learning_rate = 0.0001
    L2_regularization = 0.0001

    if use_layer_2:
        ann_model = tf.keras.Sequential(
            [
                Input(shape=(X_train_scaled.shape[1],)),
                Dense(
                    layer_1_nodes,
                    activation="tanh",
                    kernel_regularizer=tf.keras.regularizers.l2(L2_regularization),
                    name="hidden_layer_1",
                ),
                Dropout(layer_1_dropout, name="dropout_1"),
                Dense(
                    layer_2_nodes,
                    activation="tanh",
                    kernel_regularizer=tf.keras.regularizers.l2(L2_regularization),
                    name="hidden_layer_2",
                ),
                Dropout(layer_2_dropout, name="dropout_2"),
                Dense(1, activation="linear", name="output"),
            ]
        )
    else:
        ann_model = tf.keras.Sequential(
            [
                Input(shape=(X_train_scaled.shape[1],)),
                Dense(
                    layer_1_nodes,
                    activation="tanh",
                    kernel_regularizer=tf.keras.regularizers.l2(L2_regularization),
                ),
                Dropout(layer_1_dropout),
                Dense(1, activation="linear"),
            ]
        )

    print("ANN architecture:")
    ann_model.summary()

    ann_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mae", metrics=["mae", "mse"])

    epochs = 100
    batch_size = 512
    patience = 10

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True, verbose=1)

    history = ann_model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    mae_train = history.history["mae"][-1]
    mse_train = history.history["mse"][-1]
    print("Train Set Performance:")
    print(f"   mae:  {mae_train:.4f}")
    print(f"   mse:  {mse_train:.4f}")
    val_mae = history.history["val_mae"][-1]
    val_mse = history.history["val_mse"][-1]
    print("\n Validation Set Performance:")
    print(f"   mae:  {val_mae:.4f}")
    print(f"   mse:  {val_mse:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history.history["mae"], label="training mae", linewidth=2)
    axes[0].plot(history.history["val_mae"], label="validation mae", linewidth=2)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("mae")
    axes[0].set_title("ANN: Mean Absolute Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["mse"], label="training mse", linewidth=2)
    axes[1].plot(history.history["val_mse"], label="validation mse", linewidth=2)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mse")
    axes[1].set_title("ANN: Mean Squared Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "history_mae_mse.png", bbox_inches="tight")
    plt.close(fig)

    y_pred_scaled = ann_model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test_scaled, y_pred_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    print("Test Set Performance on Scaled Data:")
    print(f"   mae:  {mae:.4f}")
    print(f"   mse:  {mse:.4f}")

    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    mae = mean_absolute_error(y_test, y_pred_original)
    mse = mean_squared_error(y_test, y_pred_original)
    print("Test Set Performance on Original Scale:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   MSE:  {mse:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_scaled, y_pred_scaled, alpha=0.5, s=10)
    plt.plot(
        [y_test_scaled.min(), y_test_scaled.max()],
        [y_test_scaled.min(), y_test_scaled.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("ANN: Predicted vs Actual Values (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pred_vs_actual.png", bbox_inches="tight")
    plt.close()


    residuals = y_test - y_pred_original.flatten()
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=50, edgecolor="black")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("ANN: Distribution of Residuals")
    plt.axvline(x=0, color="r", linestyle="--", linewidth=2)

    plt.subplot(1, 2, 2)
    plt.scatter(y_pred_original, residuals, alpha=0.5, s=10)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("ANN: Residuals vs Predicted")
    plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "residuals.png", bbox_inches="tight")
    plt.close()

    # Save model for DVC
    MODEL_PATH = REPO_ROOT / "models" / "model_ann.keras"
    ann_model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # MLflow Tracking
    with mlflow.start_run(run_name=f"ann_{timestamp}"):
        mlflow.log_param("layer_1_nodes", layer_1_nodes)
        mlflow.log_param("layer_2_nodes", layer_2_nodes)
        mlflow.log_param("dropout_1", layer_1_dropout)
        mlflow.log_param("dropout_2", layer_2_dropout)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("L2_regularization", L2_regularization)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        mlflow.log_metric("train_mae", mae_train)
        mlflow.log_metric("train_mse", mse_train)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("test_mae_scaled", mae)
        mlflow.log_metric("test_mse_scaled", mse)

        mlflow.log_artifacts(str(FIG_DIR), artifact_path="figures")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")


if __name__ == "__main__":
    main()
