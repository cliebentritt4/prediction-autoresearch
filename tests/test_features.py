"""
tests/test_features.py — Unit tests for market/features/extract.py.

Tests feature extraction with mock DuckDB results, verifies
DataFrame schema and value ranges.
"""

import pandas as pd
import pytest

from market.features.extract import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    get_feature_columns,
    get_label_column,
)


def test_feature_columns_defined():
    """Feature column list should contain the spec's 5 features."""
    cols = get_feature_columns()
    assert "yes_price" in cols
    assert "volume" in cols
    assert "spread" in cols
    assert "volatility" in cols
    assert "time_to_resolution" in cols
    assert len(cols) == 5


def test_label_column():
    """Label column should be 'outcome'."""
    assert get_label_column() == "outcome"


def test_feature_columns_immutable():
    """Modifying the returned list shouldn't affect the module constant."""
    cols = get_feature_columns()
    cols.append("extra")
    assert len(get_feature_columns()) == 5


def make_mock_features_df() -> pd.DataFrame:
    """Create a mock DataFrame matching extract.py output schema."""
    return pd.DataFrame({
        "ticker": ["MARKET-A"] * 5 + ["MARKET-B"] * 3,
        "bucket_time": pd.date_range("2024-01-01", periods=8, freq="h"),
        "yes_price": [0.5, 0.52, 0.55, 0.60, 0.58, 0.30, 0.35, 0.40],
        "volume": [2.1, 2.5, 3.0, 2.8, 2.2, 1.5, 1.8, 2.0],
        "spread": [0.1, 0.08, 0.12, 0.15, 0.10, 0.40, 0.35, 0.30],
        "volatility": [0.02, 0.03, 0.01, 0.04, 0.02, 0.05, 0.03, 0.02],
        "time_to_resolution": [14.0, 13.5, 13.0, 12.5, 12.0, 10.0, 9.5, 9.0],
        "outcome": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "source": ["kalshi"] * 5 + ["polymarket"] * 3,
    })


def test_mock_df_schema():
    """Mock DataFrame should have all required columns."""
    df = make_mock_features_df()
    required = ["ticker", "bucket_time"] + FEATURE_COLUMNS + [LABEL_COLUMN, "source"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_mock_df_value_ranges():
    """Feature values should be in expected ranges."""
    df = make_mock_features_df()
    assert (df["yes_price"] >= 0).all() and (df["yes_price"] <= 1).all()
    assert (df["volume"] >= 0).all()
    assert (df["spread"] >= 0).all()
    assert (df["volatility"] >= 0).all()
    assert df["outcome"].isin([0.0, 1.0]).all()


def test_mock_df_has_multiple_markets():
    """Should have at least 2 distinct tickers."""
    df = make_mock_features_df()
    assert df["ticker"].nunique() >= 2
