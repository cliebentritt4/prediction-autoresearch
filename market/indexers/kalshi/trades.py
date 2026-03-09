"""
market/indexers/kalshi/trades.py — Kalshi trade history collector.

Fetches trade history from the Kalshi API with RSA-PSS request signing
and cursor-based pagination. Writes to parquet.
"""

import time
import base64
import requests
import pandas as pd
from tqdm import tqdm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

from market.indexers.base import Indexer
from market.config import (
    KALSHI_API_KEY_ID,
    KALSHI_API_KEY_FILE,
    KALSHI_API_BASE_URL,
    DATA_DIR,
)

# Reuse signing utilities from markets module
from market.indexers.kalshi.markets import _load_private_key, _kalshi_headers


class KalshiTradesIndexer(Indexer):
    """Indexer for Kalshi trade history."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = f"{DATA_DIR}/kalshi/trades"
        super().__init__(output_dir)
        self.key_id = KALSHI_API_KEY_ID
        self.base_url = KALSHI_API_BASE_URL
        self.private_key = _load_private_key(KALSHI_API_KEY_FILE)

    def run(self) -> None:
        """Fetch all Kalshi trades with pagination and save to parquet."""
        progress = self.load_progress()
        cursor = progress.get("cursor", None)
        all_trades = []

        print("Fetching Kalshi trades...")
        with tqdm(desc="Trades") as pbar:
            while True:
                path = "/trade-api/v2/markets/trades"
                params = {"limit": 1000}
                if cursor:
                    params["cursor"] = cursor

                headers = _kalshi_headers(
                    self.private_key, self.key_id, "GET", path
                )

                response = requests.get(
                    f"{self.base_url}{path}",
                    headers=headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                trades = data.get("trades", [])
                if not trades:
                    break

                for t in trades:
                    all_trades.append({
                        "ticker": t.get("ticker"),
                        "yes_price": t.get("yes_price", 0),
                        "no_price": t.get("no_price", 0),
                        "count": t.get("count", 0),
                        "taker_side": t.get("taker_side", ""),
                        "created_time": pd.to_datetime(t.get("created_time")),
                    })
                    pbar.update(1)

                cursor = data.get("cursor")
                if not cursor:
                    break

                self.save_progress({"cursor": cursor})

        if all_trades:
            df = pd.DataFrame(all_trades)
            out = self.save_parquet(df, "kalshi_trades.parquet")
            print(f"Saved {len(df)} trades to {out}")
        else:
            print("No trades fetched.")

        self.save_progress({"cursor": None, "complete": True})


if __name__ == "__main__":
    KalshiTradesIndexer().run()
