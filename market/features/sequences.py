"""
market/features/sequences.py — Convert DataFrames into fixed-length MLX arrays.

Handles padding/truncation to MAX_SEQ_LEN, normalization,
train/val split by market ticker, and dtype conversion to mlx.core.float32.
"""

from typing import NamedTuple

import mlx.core as mx
import numpy as np
import pandas as pd

from market.features.extract import get_feature_columns, get_label_column

# Must match prepare.py constant
MAX_SEQ_LEN = 512

# Train/val split ratio (by unique market ticker)
TRAIN_RATIO = 0.85


class MarketSequences(NamedTuple):
    """Container for processed market sequences."""

    train_features: mx.array  # (n_train_seqs, MAX_SEQ_LEN, n_features)
    train_labels: mx.array  # (n_train_seqs,) — outcome 0 or 1
    val_features: mx.array  # (n_val_seqs, MAX_SEQ_LEN, n_features)
    val_labels: mx.array  # (n_val_seqs,)
    feature_names: list[str]


def build_sequences(df: pd.DataFrame) -> MarketSequences:
    """
    Convert a feature DataFrame into padded, normalized MLX arrays.

    Args:
        df: DataFrame from extract.py with columns:
            ticker, bucket_time, yes_price, volume, spread,
            volatility, time_to_resolution, outcome, source

    Returns:
        MarketSequences with train/val splits by market ticker.
    """
    feature_cols = get_feature_columns()
    label_col = get_label_column()

    # Split by market ticker (no data leakage)
    tickers = df["ticker"].unique()
    np.random.seed(42)
    np.random.shuffle(tickers)
    split_idx = int(len(tickers) * TRAIN_RATIO)
    train_tickers = set(tickers[:split_idx])
    val_tickers = set(tickers[split_idx:])

    train_df = df[df["ticker"].isin(train_tickers)]
    val_df = df[df["ticker"].isin(val_tickers)]

    train_features, train_labels = _df_to_sequences(train_df, feature_cols, label_col)
    val_features, val_labels = _df_to_sequences(val_df, feature_cols, label_col)

    # Normalize using train statistics only (prevent val leakage)
    mean = train_features.mean(axis=(0, 1), keepdims=True)
    std = train_features.std(axis=(0, 1), keepdims=True) + 1e-8
    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std

    return MarketSequences(
        train_features=mx.array(train_features, dtype=mx.float32),
        train_labels=mx.array(train_labels, dtype=mx.float32),
        val_features=mx.array(val_features, dtype=mx.float32),
        val_labels=mx.array(val_labels, dtype=mx.float32),
        feature_names=feature_cols,
    )


def _df_to_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group by ticker, pad/truncate each market's time series to MAX_SEQ_LEN.

    Returns:
        features: (n_markets, MAX_SEQ_LEN, n_features) float32
        labels: (n_markets,) float32 — one outcome per market
    """
    grouped = df.groupby("ticker")
    sequences = []
    labels = []

    for ticker, group in grouped:
        group = group.sort_values("bucket_time")
        vals = group[feature_cols].values.astype(np.float32)

        # Pad (left) or truncate (right-most window)
        if len(vals) >= MAX_SEQ_LEN:
            vals = vals[-MAX_SEQ_LEN:]
        else:
            pad_width = MAX_SEQ_LEN - len(vals)
            vals = np.pad(vals, ((pad_width, 0), (0, 0)), mode="constant")

        sequences.append(vals)

        # One label per market (outcome of resolution)
        outcome = group[label_col].iloc[-1]
        labels.append(float(outcome))

    if not sequences:
        n_features = len(feature_cols)
        return np.zeros((0, MAX_SEQ_LEN, n_features), dtype=np.float32), np.zeros(
            0, dtype=np.float32
        )

    return np.stack(sequences), np.array(labels, dtype=np.float32)
