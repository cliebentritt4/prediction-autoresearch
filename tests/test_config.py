"""
tests/test_config.py — Tests that config paths resolve correctly.
"""

import os


def test_config_imports():
    """market.config should import without error."""
    from market.config import DATA_DIR, CACHE_DIR, OUTPUT_DIR

    assert DATA_DIR is not None
    assert CACHE_DIR is not None
    assert OUTPUT_DIR is not None


def test_data_dir_is_absolute():
    """DATA_DIR should resolve to an absolute path."""
    from market.config import DATA_DIR

    assert os.path.isabs(DATA_DIR)


def test_cache_dir_in_home():
    """CACHE_DIR should be under ~/.cache/."""
    from market.config import CACHE_DIR

    assert ".cache" in CACHE_DIR


def test_parquet_globs_reference_data_dir():
    """Parquet glob patterns should reference the data directory."""
    from market.config import (
        DATA_DIR,
        KALSHI_MARKETS_GLOB,
        KALSHI_TRADES_GLOB,
        POLYMARKET_MARKETS_GLOB,
        POLYMARKET_TRADES_GLOB,
    )

    assert DATA_DIR in KALSHI_MARKETS_GLOB or "data" in KALSHI_MARKETS_GLOB
    assert DATA_DIR in KALSHI_TRADES_GLOB or "data" in KALSHI_TRADES_GLOB
    assert DATA_DIR in POLYMARKET_MARKETS_GLOB or "data" in POLYMARKET_MARKETS_GLOB
    assert DATA_DIR in POLYMARKET_TRADES_GLOB or "data" in POLYMARKET_TRADES_GLOB


def test_env_loading():
    """dotenv loading should not crash even without a .env file."""
    # Re-importing should work fine
    from market import config

    assert hasattr(config, "DATA_DIR")
