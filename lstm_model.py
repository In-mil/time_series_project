#!/usr/bin/env python3
"""LSTM model script copied from notebooks/2_Modeling_LSTM.ipynb."""

import random
import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

DATA_PATH = Path(__file__).resolve().parent / "data" / "final_data" / "20251115_dataset_crp.csv"
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


def create_sequences(X_data: np.ndarray, y_data: np.ndarray, df: pd.DataFrame, look_back: int = 10):
    """Create sliding-window sequences for each cryptocurrency."""
    X_sequences = []
    y_sequences = []
    original_indices = []

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_indices = df[mask].index.values
        crypto_X = X_data[mask]
        crypto_y = y_data[mask]
        crypto_dates = df.loc[mask, "date"].values

        order = np.argsort(crypto_dates)
        crypto_X = crypto_X[order]
        crypto_y = crypto_y[order]
        ticker_indices = ticker_indices[order]

        for i in range(len(crypto_X) - look_back):
            X_sequences.append(crypto_X[i : i + look_back])
            y_sequences.append(crypto_y[i + look_back])
            original_indices.append(ticker_indices[i + look_back])

    return np.array(X_sequences), np.array(y_sequences), np.array(original_indices)


def main() -> None:
    """Run the LSTM modeling experiment."""
    df_lstm = pd.read_csv(DATA_PATH)

    split_date_train = "2024-07-01"
    split_date_val = "2024-10-01"
    split_date_test = "2025-01-01"

    df_train = df_lstm[df_lstm["date"] < split_date_train].copy()
    df_val = df_lstm[(df_lstm["date"] >= split_date_train) & (df_lstm["date"] < split_date_val)].copy()
    df_test = df_lstm[df_lstm["date"] >= split_date_test].copy()

    print(f"\nTraining set: {len(df_train)} samples")
    print(f"Validation set: {len(df_val)} samples")
    print(f"Test set: {len(df_test)} samples")
    print(f"Train up to date: {df_train['date'].max()}")
    print(f"Validation from date: {df_val['date'].min()} to {df_val['date'].max()}")
    print(f"Test from date: {df_test['date'].min()}")

    feature_cols = [col for col in df_train.columns if col not in TARGET_COLUMNS]
    X_train = df_train[feature_cols].copy()
    y_train = df_train["future_5_close_higher_than_today"].values
    X_val = df_val[feature_cols].copy()
    y_val = df_val["future_5_close_higher_than_today"].values
    X_test = df_test[feature_cols].copy()
    y_test = df_test["future_5_close_higher_than_today"].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    cols_to_scale = [col for col in feature_cols if col not in ["ticker", "date"]]

    X_train_scaled_vals = scaler_X.fit_transform(X_train[cols_to_scale])
    X_val_scaled_vals = scaler_X.transform(X_val[cols_to_scale])
    X_test_scaled_vals = scaler_X.transform(X_test[cols_to_scale])

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print(f"Features shape: {X_train_scaled_vals.shape}")
    print(f"Target shape: {y_train_scaled.shape}")
    print("Note: 'ticker' and 'date' kept for sequence creation")

    look_back = 20
    print("Creating sequences...")
    X_train_seq, y_train_seq, train_indices = create_sequences(X_train_scaled_vals, y_train_scaled, X_train, look_back)
    X_val_seq, y_val_seq, val_indices = create_sequences(X_val_scaled_vals, y_val_scaled, X_val, look_back)
    X_test_seq, y_test_seq, test_indices = create_sequences(X_test_scaled_vals, y_test_scaled, X_test, look_back)

    n_features = X_train_seq.shape[2]
    print(f"\nShape: {X_train_seq.shape}")
    print(f"   {X_train_seq.shape[0]} sequences")
    print(f"   {look_back} days look-back")
    print(f"   {n_features} features")
    print("   Indices tracked for alignment")

    lstm_units = 64
    lstm_dropout = 0.3
    use_second_lstm = True
    learning_rate = 0.0001
    L2_regularization = 0.0001

    if use_second_lstm:
        model_lstm = Sequential(
            [
                Input(shape=(look_back, n_features)),
                LSTM(
                    lstm_units,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(L2_regularization),
                    name="lstm_layer_1",
                ),
                Dropout(lstm_dropout, name="dropout_1"),
                LSTM(
                    lstm_units // 2,
                    return_sequences=False,
                    kernel_regularizer=tf.keras.regularizers.l2(L2_regularization),
                    name="lstm_layer_2",
                ),
                Dropout(lstm_dropout, name="dropout_2"),
                Dense(1, activation="linear", name="output"),
            ]
        )
    else:
        model_lstm = Sequential(
            [
                Input(shape=(look_back, n_features)),
                LSTM(
                    lstm_units,
                    return_sequences=False,
                    kernel_regularizer=tf.keras.regularizers.l2(L2_regularization),
                    name="lstm_layer",
                ),
                Dropout(lstm_dropout, name="dropout"),
                Dense(1, activation="linear", name="output"),
            ]
        )

    model_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mae", metrics=["mae", "mse"])
    print("LSTM Model Architecture:")
    model_lstm.summary()

    epochs = 100
    batch_size = 128
    patience = 10

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )

    print("Training LSTM model...")
    history = model_lstm.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
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
    axes[0].set_title("LSTM: Mean Absolute Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history["mse"], label="training mse", linewidth=2)
    axes[1].plot(history.history["val_mse"], label="validation mse", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("LSTM: Mean Squared Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    y_pred_scaled = model_lstm.predict(X_test_seq)
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
    print(f"   MAE:  {mae:.4f}")
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
    plt.title("LSTM: Predicted vs Actual (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    residuals = y_test_original - y_pred_original
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(residuals, bins=50, edgecolor="black")
    axes[0].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residual (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("LSTM: Distribution of Residuals")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(y_pred_original, residuals, alpha=0.5, s=10)
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("LSTM: Residuals vs Predicted")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Residual Stats:")
    print(f"   Mean: {residuals.mean():.4f}")
    print(f"   Std:  {residuals.std():.4f}")
    print(f"   Min:  {residuals.min():.4f}")
    print(f"   Max:  {residuals.max():.4f}")


if __name__ == "__main__":
    main()
