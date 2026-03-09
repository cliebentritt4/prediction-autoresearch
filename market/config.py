"""
market/config.py — Paths, constants, and environment variable loading.

Centralizes all path resolution and configuration for the market data pipeline.
Loads .env for API keys (Kalshi, Polymarket RPC, HuggingFace).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------

DATA_DIR = str(_PROJECT_ROOT / "data")
CACHE_DIR = os.path.expanduser("~/.cache/autoresearch/market")
OUTPUT_DIR = str(_PROJECT_ROOT / "output")

# ---------------------------------------------------------------------------
# Parquet glob patterns (used by extract.py DuckDB queries)
# ---------------------------------------------------------------------------

KALSHI_MARKETS_GLOB = str(Path(DATA_DIR) / "kalshi" / "markets" / "*.parquet")
KALSHI_TRADES_GLOB = str(Path(DATA_DIR) / "kalshi" / "trades" / "*.parquet")
POLYMARKET_MARKETS_GLOB = str(Path(DATA_DIR) / "polymarket" / "markets" / "*.parquet")
POLYMARKET_TRADES_GLOB = str(Path(DATA_DIR) / "polymarket" / "trades" / "*.parquet")

# ---------------------------------------------------------------------------
# API keys (loaded from .env)
# ---------------------------------------------------------------------------

KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_API_KEY_FILE = os.getenv(
    "KALSHI_API_KEY_FILE", str(_PROJECT_ROOT / "kalshi.key")
)
KALSHI_API_BASE_URL = os.getenv("KALSHI_API_BASE_URL", "https://api.kalshi.com")
POLYMARKET_RPC_URL = os.getenv("POLYMARKET_RPC_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default time bucket for feature aggregation
DEFAULT_TIME_BUCKET = "1 hour"

# Minimum trades per market to include in feature extraction
MIN_TRADES_PER_MARKET = 20
