"""
market/features/extract.py — DuckDB queries on parquet → pandas DataFrames.

The key bridge between raw market data and the ML training loop.
Extracts per-market time series features: price, volume, spread,
volatility, time-to-resolution, and resolution outcome.
"""

import duckdb
import pandas as pd

from market.config import (
    DATA_DIR,
    KALSHI_MARKETS_GLOB,
    KALSHI_TRADES_GLOB,
    POLYMARKET_MARKETS_GLOB,
    POLYMARKET_TRADES_GLOB,
)

# Feature columns served to sequences.py
FEATURE_COLUMNS = [
    "yes_price",
    "volume",
    "spread",
    "volatility",
    "time_to_resolution",
]

LABEL_COLUMN = "outcome"


def get_feature_columns() -> list[str]:
    """Return the list of feature column names."""
    return list(FEATURE_COLUMNS)


def get_label_column() -> str:
    """Return the label column name."""
    return LABEL_COLUMN


def extract_market_features(source: str = "all") -> pd.DataFrame:
    """
    Extract market features from parquet files using DuckDB.

    Args:
        source: "kalshi", "polymarket", or "all"

    Returns:
        DataFrame with columns:
            ticker, bucket_time, yes_price, volume, spread,
            volatility, time_to_resolution, outcome, source
    """
    con = duckdb.connect(":memory:")
    frames = []

    if source in ("kalshi", "all"):
        df = _extract_kalshi(con)
        if len(df) > 0:
            frames.append(df)

    if source in ("polymarket", "all"):
        df = _extract_polymarket(con)
        if len(df) > 0:
            frames.append(df)

    con.close()

    if not frames:
        return pd.DataFrame(
            columns=["ticker", "bucket_time"] + FEATURE_COLUMNS + [LABEL_COLUMN, "source"]
        )

    return pd.concat(frames, ignore_index=True)


def get_resolved_features(source: str = "all") -> pd.DataFrame:
    """
    Extract features for resolved markets only (outcome is not null).
    Used for training data where we need ground truth labels.
    """
    df = extract_market_features(source)
    return df.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)


def _extract_kalshi(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract features from Kalshi parquet files."""
    query = f"""
    WITH markets AS (
        SELECT ticker, title, category, status, result,
               close_time, open_time
        FROM read_parquet('{KALSHI_MARKETS_GLOB}')
    ),
    trades AS (
        SELECT ticker, yes_price, count AS volume,
               created_time AS trade_time
        FROM read_parquet('{KALSHI_TRADES_GLOB}')
    ),
    bucketed AS (
        SELECT
            t.ticker,
            time_bucket(INTERVAL '1 hour', t.trade_time) AS bucket_time,
            AVG(t.yes_price) / 100.0 AS yes_price,
            LN(SUM(t.volume) + 1) AS volume,
            AVG(ABS(t.yes_price - 50.0)) / 50.0 AS spread,
            STDDEV(t.yes_price) / 100.0 AS volatility,
            LN(EXTRACT(EPOCH FROM (MIN(m.close_time) - time_bucket(INTERVAL '1 hour', t.trade_time))) + 1) AS time_to_resolution,
            CASE WHEN MIN(m.result) = 'yes' THEN 1.0
                 WHEN MIN(m.result) = 'no' THEN 0.0
                 ELSE NULL END AS outcome
        FROM trades t
        JOIN markets m ON t.ticker = m.ticker
        GROUP BY t.ticker, time_bucket(INTERVAL '1 hour', t.trade_time)
    )
    SELECT *, 'kalshi' AS source
    FROM bucketed
    ORDER BY ticker, bucket_time
    """
    try:
        return con.execute(query).fetchdf()
    except Exception:
        return pd.DataFrame()


def _extract_polymarket(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract features from Polymarket parquet files."""
    query = f"""
    WITH markets AS (
        SELECT ticker, title, category, status, result,
               close_time, open_time
        FROM read_parquet('{POLYMARKET_MARKETS_GLOB}')
    ),
    trades AS (
        SELECT ticker, yes_price, count AS volume,
               created_time AS trade_time
        FROM read_parquet('{POLYMARKET_TRADES_GLOB}')
    ),
    bucketed AS (
        SELECT
            t.ticker,
            time_bucket(INTERVAL '1 hour', t.trade_time) AS bucket_time,
            AVG(t.yes_price) / 100.0 AS yes_price,
            LN(SUM(t.volume) + 1) AS volume,
            AVG(ABS(t.yes_price - 50.0)) / 50.0 AS spread,
            STDDEV(t.yes_price) / 100.0 AS volatility,
            LN(EXTRACT(EPOCH FROM (MIN(m.close_time) - time_bucket(INTERVAL '1 hour', t.trade_time))) + 1) AS time_to_resolution,
            CASE WHEN MIN(m.result) = 'yes' THEN 1.0
                 WHEN MIN(m.result) = 'no' THEN 0.0
                 ELSE NULL END AS outcome
        FROM trades t
        JOIN markets m ON t.ticker = m.ticker
        GROUP BY t.ticker, time_bucket(INTERVAL '1 hour', t.trade_time)
    )
    SELECT *, 'polymarket' AS source
    FROM bucketed
    ORDER BY ticker, bucket_time
    """
    try:
        return con.execute(query).fetchdf()
    except Exception:
        return pd.DataFrame()
