#!/usr/bin/env python3
"""Reusable model experiments for the time-series dataset."""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "final_data" / "20251115_dataset_crp.csv"
TARGET_COLUMN = "future_5_close_higher_than_today"
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
META_COLUMNS = ["ticker", "date"]
TABULAR_DROP_COLUMNS = TARGET_COLUMNS + META_COLUMNS
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingParams:
    epochs: int
    batch_size: int
    patience: int
    monitor: str = "val_loss"
    reduce_lr: bool = False


@dataclass(frozen=True)
class ANNParams:
    layer_1_nodes: int = 128
    layer_1_dropout: float = 0.2
    layer_2_nodes: int = 64
    layer_2_dropout: float = 0.2
    use_layer_2: bool = True
    learning_rate: float = 0.0001
    l2_reg: float = 0.0001


@dataclass(frozen=True)
class RNNParams:
    units: int = 64
    dropout: float = 0.3
    use_second_layer: bool = True
    learning_rate: float = 0.0001
    l2_reg: float = 0.0001


@dataclass(frozen=True)
class TransformerParams:
    head_size: int = 128
    num_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.2
    mlp_units: int = 64
    mlp_dropout: float = 0.3
    learning_rate: float = 0.0001


@dataclass(frozen=True)
class TabularSets:
    X_train_scaled: np.ndarray
    y_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    y_val_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_test_scaled: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler_X: StandardScaler
    scaler_y: StandardScaler


@dataclass(frozen=True)
class SequenceSets:
    X_train_seq: np.ndarray
    y_train_seq: np.ndarray
    X_val_seq: np.ndarray
    y_val_seq: np.ndarray
    X_test_seq: np.ndarray
    y_test_seq: np.ndarray
    scaler_X: StandardScaler
    scaler_y: StandardScaler
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    n_features: int


def set_seed(seed: int) -> None:
    """Fix pseudo-random generators for reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the dataset and ensure the date column is a datetime."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def split_time_series(
    df: pd.DataFrame, train_cut: str, val_cut: str, test_cut: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data in time order to avoid leaking the future."""
    df_train = df[df["date"] < train_cut].copy()
    df_val = df[(df["date"] >= train_cut) & (df["date"] < val_cut)].copy()
    df_test = df[df["date"] >= test_cut].copy()
    return df_train, df_val, df_test


def log_split_info(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> None:
    """Log counts and date ranges for each split."""
    LOGGER.info("Training set: %d samples (up to %s)", len(df_train), df_train["date"].max())
    LOGGER.info(
        "Validation set: %d samples (from %s to %s)",
        len(df_val),
        df_val["date"].min(),
        df_val["date"].max(),
    )
    LOGGER.info("Test set: %d samples (from %s)", len(df_test), df_test["date"].min())


def _drop_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Drop columns if they exist to avoid KeyError."""
    existing = [col for col in columns if col in frame.columns]
    return frame.drop(columns=existing, axis="columns")


def prepare_tabular_sets(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> TabularSets:
    """Prepare tabular features that do not require sequencing."""
    X_train = _drop_columns(df_train, TABULAR_DROP_COLUMNS)
    X_val = _drop_columns(df_val, TABULAR_DROP_COLUMNS)
    X_test = _drop_columns(df_test, TABULAR_DROP_COLUMNS)

    y_train = df_train[TARGET_COLUMN].to_numpy()
    y_val = df_val[TARGET_COLUMN].to_numpy()
    y_test = df_test[TARGET_COLUMN].to_numpy()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    LOGGER.info("Tabular matrices: X_train %s, y_train %s", X_train_scaled.shape, y_train_scaled.shape)
    return TabularSets(
        X_train_scaled=X_train_scaled,
        y_train_scaled=y_train_scaled,
        X_val_scaled=X_val_scaled,
        y_val_scaled=y_val_scaled,
        X_test_scaled=X_test_scaled,
        y_test_scaled=y_test_scaled,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )


def create_sequences(
    X_data: np.ndarray, y_data: np.ndarray, df: pd.DataFrame, look_back: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows for each crypto individually."""
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

    return (
        np.asarray(X_sequences),
        np.asarray(y_sequences),
        np.asarray(original_indices),
    )


def prepare_sequence_sets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    look_back: int,
) -> SequenceSets:
    """Prepare features for recurrent and attention-based models."""
    feature_columns = [col for col in df_train.columns if col not in TARGET_COLUMNS]
    X_train = df_train[feature_columns].copy()
    X_val = df_val[feature_columns].copy()
    X_test = df_test[feature_columns].copy()

    y_train = df_train[TARGET_COLUMN].to_numpy()
    y_val = df_val[TARGET_COLUMN].to_numpy()
    y_test = df_test[TARGET_COLUMN].to_numpy()

    cols_to_scale = [col for col in feature_columns if col not in META_COLUMNS]
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled_vals = scaler_X.fit_transform(X_train[cols_to_scale])
    X_val_scaled_vals = scaler_X.transform(X_val[cols_to_scale])
    X_test_scaled_vals = scaler_X.transform(X_test[cols_to_scale])

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    LOGGER.info("Sequence scaling shapes: %s", X_train_scaled_vals.shape)

    X_train_seq, y_train_seq, train_indices = create_sequences(
        X_train_scaled_vals, y_train_scaled, X_train, look_back
    )
    X_val_seq, y_val_seq, val_indices = create_sequences(
        X_val_scaled_vals, y_val_scaled, X_val, look_back
    )
    X_test_seq, y_test_seq, test_indices = create_sequences(
        X_test_scaled_vals, y_test_scaled, X_test, look_back
    )

    if X_train_seq.size == 0:
        raise ValueError("Lookback window is too large for the available training data.")

    return SequenceSets(
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
        X_val_seq=X_val_seq,
        y_val_seq=y_val_seq,
        X_test_seq=X_test_seq,
        y_test_seq=y_test_seq,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        n_features=X_train_seq.shape[2],
    )


def configure_callbacks(params: TrainingParams) -> list[tf.keras.callbacks.Callback]:
    """Create early stopping (and optional LR reducer) callbacks."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=params.monitor,
            patience=params.patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    if params.reduce_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=params.monitor,
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
            )
        )
    return callbacks


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: TrainingParams,
) -> tf.keras.callbacks.History:
    """Train the model with the requested training parameters."""
    callbacks = configure_callbacks(params)
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=params.epochs,
        batch_size=params.batch_size,
        callbacks=callbacks,
        verbose=1,
    )


def log_history_summary(history: tf.keras.callbacks.History, label: str) -> None:
    """Log metrics saved in the history object for the final epoch."""
    train_mae = history.history["mae"][-1]
    train_mse = history.history["mse"][-1]
    val_mae = history.history["val_mae"][-1]
    val_mse = history.history["val_mse"][-1]
    LOGGER.info("[%s] last epoch -> train MAE: %.4f, MSE: %.4f", label, train_mae, train_mse)
    LOGGER.info("[%s] last epoch -> val   MAE: %.4f, MSE: %.4f", label, val_mae, val_mse)


def build_ann_model(input_dim: int, params: ANNParams) -> tf.keras.Model:
    """Create the feed-forward baseline model."""
    regularizer = tf.keras.regularizers.l2(params.l2_reg)
    model = tf.keras.Sequential(name="ann_model")
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    model.add(
        tf.keras.layers.Dense(
            params.layer_1_nodes,
            activation="tanh",
            kernel_regularizer=regularizer,
            name="hidden_room_1",
        )
    )
    model.add(tf.keras.layers.Dropout(params.layer_1_dropout, name="dropout_1"))

    if params.use_layer_2:
        model.add(
            tf.keras.layers.Dense(
                params.layer_2_nodes,
                activation="tanh",
                kernel_regularizer=regularizer,
                name="hidden_room_2",
            )
        )
        model.add(tf.keras.layers.Dropout(params.layer_2_dropout, name="dropout_2"))

    model.add(tf.keras.layers.Dense(1, activation="linear", name="output"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss="mae",
        metrics=["mae", "mse"],
    )
    LOGGER.info("ANN summary:")
    model.summary(print_fn=LOGGER.info)
    return model


def build_rnn_model(model_type: str, look_back: int, n_features: int, params: RNNParams) -> tf.keras.Model:
    """Reuse the same construction for GRU and LSTM with configurable stacking."""
    layer_cls = tf.keras.layers.GRU if model_type == "gru" else tf.keras.layers.LSTM
    regularizer = tf.keras.regularizers.l2(params.l2_reg)
    model = tf.keras.Sequential(name=f"{model_type}_model")
    model.add(tf.keras.layers.Input(shape=(look_back, n_features)))

    if params.use_second_layer:
        model.add(
            layer_cls(
                params.units,
                return_sequences=True,
                kernel_regularizer=regularizer,
                name=f"{model_type}_layer_1",
            )
        )
        model.add(tf.keras.layers.Dropout(params.dropout, name="dropout_1"))
        model.add(
            layer_cls(
                params.units // 2,
                return_sequences=False,
                kernel_regularizer=regularizer,
                name=f"{model_type}_layer_2",
            )
        )
        model.add(tf.keras.layers.Dropout(params.dropout, name="dropout_2"))
    else:
        model.add(
            layer_cls(
                params.units,
                return_sequences=False,
                kernel_regularizer=regularizer,
                name=f"{model_type}_layer",
            )
        )
        model.add(tf.keras.layers.Dropout(params.dropout, name="dropout"))

    model.add(tf.keras.layers.Dense(1, activation="linear", name="output"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss="mae",
        metrics=["mae", "mse"],
    )
    LOGGER.info("%s summary:", model_type.upper())
    model.summary(print_fn=LOGGER.info)
    return model


def positional_encoding(length: int, depth: int) -> tf.Tensor:
    """Build the sinusoidal encoding that gives each timestep a unique signature."""
    positions = np.arange(length)[:, np.newaxis]
    depth_indices = np.arange(depth)[np.newaxis, :]
    angle_rates = 1 / (10000 ** ((2 * (depth_indices // 2)) / depth))
    angle_rads = positions * angle_rates
    encoding = np.zeros((length, depth))
    encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(encoding, dtype=tf.float32)


def build_transformer_model(
    look_back: int, n_features: int, params: TransformerParams
) -> tf.keras.Model:
    """Construct a small transformer encoder followed by a pooling head."""
    inputs = tf.keras.layers.Input(shape=(look_back, n_features), name="transformer_input")
    pos_encoding = positional_encoding(look_back, n_features)
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)
    x = tf.keras.layers.Add()([inputs, pos_encoding])

    for block in range(2):
        attention = tf.keras.layers.MultiHeadAttention(
            key_dim=params.head_size,
            num_heads=params.num_heads,
            dropout=params.dropout,
            name=f"attention_{block + 1}",
        )(x, x)
        attention = tf.keras.layers.Dropout(params.dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
            x + attention
        )

        ffn = tf.keras.layers.Dense(params.ff_dim, activation="relu")(attention)
        ffn = tf.keras.layers.Dropout(params.dropout)(ffn)
        ffn = tf.keras.layers.Dense(n_features)(ffn)
        ffn = tf.keras.layers.Dropout(params.dropout)(ffn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + ffn)

    pooled = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Dense(params.mlp_units, activation="relu")(pooled)
    pooled = tf.keras.layers.Dropout(params.mlp_dropout)(pooled)
    outputs = tf.keras.layers.Dense(1, activation="linear")(pooled)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss="mae",
        metrics=["mae", "mse"],
    )
    LOGGER.info("Transformer summary:")
    model.summary(print_fn=LOGGER.info)
    return model


def plot_training_history(history: tf.keras.callbacks.History, label: str) -> None:
    """Show MAE and MSE curves for train/validation splits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history.history["mae"], label="train MAE", linewidth=2)
    axes[0].plot(history.history["val_mae"], label="val MAE", linewidth=2)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MAE")
    axes[0].set_title(f"{label}: Mean Absolute Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["mse"], label="train MSE", linewidth=2)
    axes[1].plot(history.history["val_mse"], label="val MSE", linewidth=2)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title(f"{label}: Mean Squared Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_prediction_scatter(y_true, y_pred, title: str) -> None:
    """Visualize predicted vs actual (scaled space)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}: Scaled Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_residuals(y_true, y_pred, title: str) -> None:
    """Show the residual distribution and heteroskedasticity check."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=50, edgecolor="black")
    axes[0].axvline(0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residual (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{title}: Residual Distribution")

    axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1].axhline(0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"{title}: Residuals vs Predicted")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def evaluate_tabular(
    model: tf.keras.Model, data: TabularSets, plot: bool, label: str
) -> None:
    """Evaluate the ANN output on scaled and original scales."""
    y_pred_scaled = model.predict(data.X_test_scaled).ravel()
    mae_scaled = mean_absolute_error(data.y_test_scaled, y_pred_scaled)
    mse_scaled = mean_squared_error(data.y_test_scaled, y_pred_scaled)
    LOGGER.info("[%s] scaled test -> MAE %.4f, MSE %.4f", label, mae_scaled, mse_scaled)

    y_pred_original = data.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    mae_orig = mean_absolute_error(data.y_test, y_pred_original)
    mse_orig = mean_squared_error(data.y_test, y_pred_original)
    LOGGER.info("[%s] original test -> MAE %.4f, MSE %.4f", label, mae_orig, mse_orig)

    if plot:
        plot_prediction_scatter(data.y_test_scaled, y_pred_scaled, f"{label} predictions")
        plot_residuals(data.y_test, y_pred_original, f"{label} residuals")


def evaluate_sequence_model(
    model: tf.keras.Model,
    seq_sets: SequenceSets,
    df_test: pd.DataFrame,
    plot: bool,
    label: str,
) -> None:
    """Evaluate recurrent and transformer models and align predictions with the original rows."""
    y_pred_scaled = model.predict(seq_sets.X_test_seq).ravel()
    mae_scaled = mean_absolute_error(seq_sets.y_test_seq, y_pred_scaled)
    mse_scaled = mean_squared_error(seq_sets.y_test_seq, y_pred_scaled)
    LOGGER.info("[%s] scaled test -> MAE %.4f, MSE %.4f", label, mae_scaled, mse_scaled)

    y_pred_original = seq_sets.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_original = df_test.loc[seq_sets.test_indices, TARGET_COLUMN].values

    mae_orig = mean_absolute_error(y_test_original, y_pred_original)
    mse_orig = mean_squared_error(y_test_original, y_pred_original)
    LOGGER.info("[%s] original test -> MAE %.4f, MSE %.4f", label, mae_orig, mse_orig)
    LOGGER.info("[%s] aligned %d predictions with original indices", label, len(y_pred_original))
    LOGGER.info(
        "[%s] residual stats -> mean %.4f, std %.4f, min %.4f, max %.4f",
        label,
        (y_test_original - y_pred_original).mean(),
        (y_test_original - y_pred_original).std(),
        (y_test_original - y_pred_original).min(),
        (y_test_original - y_pred_original).max(),
    )

    if plot:
        plot_prediction_scatter(seq_sets.y_test_seq, y_pred_scaled, f"{label} predictions")
        plot_residuals(y_test_original, y_pred_original, f"{label} residuals")


def main() -> None:
    """Entry point that exposes CLI selection of models."""
    parser = argparse.ArgumentParser(description="Run one of the modeling experiments.")
    parser.add_argument(
        "--model",
        choices=["ann", "gru", "lstm", "transformer"],
        required=True,
        help="Select which architecture to train.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the cleaned CSV file.",
    )
    parser.add_argument(
        "--look-back",
        type=int,
        default=20,
        help="How many past timesteps the recurrent/attention models should see.",
    )
    parser.add_argument("--plot", action="store_true", help="Show training and evaluation plots.")
    parser.add_argument("--seed", type=int, default=101, help="Fix random seeds for reproducibility.")
    parser.add_argument(
        "--split-train", default="2024-07-01", help="Cutoff for the training split."
    )
    parser.add_argument(
        "--split-val", default="2024-10-01", help="Start of the validation split."
    )
    parser.add_argument("--split-test", default="2025-01-01", help="Start of the test split.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    tf.get_logger().setLevel("ERROR")
    set_seed(args.seed)

    df = load_dataset(args.data_path.expanduser())
    df_train, df_val, df_test = split_time_series(
        df, args.split_train, args.split_val, args.split_test
    )
    log_split_info(df_train, df_val, df_test)

    if args.model == "ann":
        data = prepare_tabular_sets(df_train, df_val, df_test)
        model = build_ann_model(data.X_train_scaled.shape[1], ANNParams())
        history = train_model(
            model,
            data.X_train_scaled,
            data.y_train_scaled,
            data.X_val_scaled,
            data.y_val_scaled,
            TrainingParams(epochs=100, batch_size=512, patience=10),
        )
        log_history_summary(history, "ANN")
        if args.plot:
            plot_training_history(history, "ANN")
        evaluate_tabular(model, data, args.plot, "ANN")
    else:
        seq_sets = prepare_sequence_sets(df_train, df_val, df_test, args.look_back)
        if args.model == "transformer":
            model = build_transformer_model(args.look_back, seq_sets.n_features, TransformerParams())
            training_params = TrainingParams(epochs=100, batch_size=128, patience=15, reduce_lr=True)
        else:
            model = build_rnn_model(args.model, args.look_back, seq_sets.n_features, RNNParams())
            training_params = TrainingParams(epochs=100, batch_size=128, patience=10)

        history = train_model(
            model,
            seq_sets.X_train_seq,
            seq_sets.y_train_seq,
            seq_sets.X_val_seq,
            seq_sets.y_val_seq,
            training_params,
        )
        label = args.model.upper()
        log_history_summary(history, label)
        if args.plot:
            plot_training_history(history, label)
        evaluate_sequence_model(model, seq_sets, df_test, args.plot, label)


if __name__ == "__main__":
    main()
