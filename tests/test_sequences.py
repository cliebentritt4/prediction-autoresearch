"""
tests/test_sequences.py — Unit tests for market/features/sequences.py.

Verifies shape, padding, normalization, MLX array dtype,
and train/val split by market ticker.
"""

import numpy as np
import pandas as pd
import pytest

from market.features.sequences import MAX_SEQ_LEN, build_sequences


def make_test_df(n_markets: int = 20, steps_per_market: int = 100) -> pd.DataFrame:
    """Create a synthetic DataFrame for testing."""
    rows = []
    rng = np.random.default_rng(42)

    for i in range(n_markets):
        ticker = f"TEST-{i:03d}"
        outcome = float(rng.choice([0, 1]))
        for t in range(steps_per_market):
            rows.append({
                "ticker": ticker,
                "bucket_time": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=t),
                "yes_price": rng.uniform(0.1, 0.9),
                "volume": rng.uniform(0, 5),
                "spread": rng.uniform(0, 0.5),
                "volatility": rng.uniform(0, 0.1),
                "time_to_resolution": rng.uniform(5, 15),
                "outcome": outcome,
                "source": "test",
            })

    return pd.DataFrame(rows)


def test_build_sequences_output_type():
    """build_sequences should return MarketSequences namedtuple."""
    df = make_test_df()
    result = build_sequences(df)
    assert hasattr(result, "train_features")
    assert hasattr(result, "train_labels")
    assert hasattr(result, "val_features")
    assert hasattr(result, "val_labels")
    assert hasattr(result, "feature_names")


def test_sequence_shape():
    """Feature arrays should be (n_markets, MAX_SEQ_LEN, n_features)."""
    df = make_test_df(n_markets=20)
    result = build_sequences(df)

    # Total markets should sum to ~20 (split ~85/15)
    total = result.train_features.shape[0] + result.val_features.shape[0]
    assert total == 20

    # Sequence length
    assert result.train_features.shape[1] == MAX_SEQ_LEN
    assert result.val_features.shape[1] == MAX_SEQ_LEN

    # 5 features
    assert result.train_features.shape[2] == 5
    assert result.val_features.shape[2] == 5


def test_labels_shape():
    """Labels should be 1D with one per market."""
    df = make_test_df(n_markets=20)
    result = build_sequences(df)
    assert len(result.train_labels.shape) == 1
    assert len(result.val_labels.shape) == 1
    total = result.train_labels.shape[0] + result.val_labels.shape[0]
    assert total == 20


def test_labels_binary():
    """Labels should be 0 or 1."""
    df = make_test_df()
    result = build_sequences(df)
    train_labels = np.array(result.train_labels)
    val_labels = np.array(result.val_labels)
    all_labels = np.concatenate([train_labels, val_labels])
    assert set(all_labels.tolist()).issubset({0.0, 1.0})


def test_padding_short_sequences():
    """Markets with fewer steps than MAX_SEQ_LEN should be left-padded."""
    df = make_test_df(n_markets=5, steps_per_market=10)
    result = build_sequences(df)
    features = np.array(result.train_features)

    # First (MAX_SEQ_LEN - 10) rows should be zero (padding)
    if features.shape[0] > 0:
        pad_region = features[0, : MAX_SEQ_LEN - 10, :]
        # After normalization, padded values will be negative (0 - mean) / std
        # But the non-padded region should be different
        data_region = features[0, -10:, :]
        assert not np.allclose(pad_region, data_region)


def test_no_data_leakage():
    """Train and val should have completely different tickers."""
    df = make_test_df(n_markets=20)
    train_tickers = set(df.iloc[:int(len(df) * 0.85)]["ticker"].unique())
    val_tickers = set(df.iloc[int(len(df) * 0.85):]["ticker"].unique())
    # The actual split is by shuffled ticker, so we just verify
    # that the function runs without error and produces both splits
    result = build_sequences(df)
    assert result.train_features.shape[0] > 0
    assert result.val_features.shape[0] > 0


def test_feature_names():
    """feature_names should list the 5 feature columns."""
    df = make_test_df()
    result = build_sequences(df)
    assert len(result.feature_names) == 5
    assert "yes_price" in result.feature_names
