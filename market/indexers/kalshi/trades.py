"""
market/indexers/kalshi/trades.py — Kalshi trade history collector.

Fetches trade history from the Kalshi API with pagination and
incremental progress. Writes to parquet.
"""

import requests
import pandas as pd
from tqdm import tqdm

from market.indexers.base import Indexer
from market.config import KALSHI_API_KEY, DATA_DIR


class KalshiTradesIndexer(Indexer):
    """Indexer for Kalshi trade history."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = f"{DATA_DIR}/kalshi/trades"
        super().__init__(output_dir)
        self.api_key = KALSHI_API_KEY
        self.base_url = "https://api.kalshi.com/trade-api/v2"

    def run(self) -> None:
        """Fetch all Kalshi trades with pagination and save to parquet."""
        progress = self.load_progress()
        cursor = progress.get("cursor", None)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        all_trades = []

        print("Fetching Kalshi trades...")
        with tqdm(desc="Trades") as pbar:
            while True:
                params = {"limit": 1000}
                if cursor:
                    params["cursor"] = cursor

                response = requests.get(
                    f"{self.base_url}/trades",
                    headers=headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                trades = data.get("trades", [])
                if not trades:
                    break

                for trade in trades:
                    all_trades.append({
                        "ticker": trade.get("ticker"),
                        "yes_price": trade.get("yes_price", 0),
                        "no_price": trade.get("no_price", 0),
                        "count": trade.get("count", 0),
                        "taker_side": trade.get("taker_side", ""),
                        "created_time": pd.to_datetime(trade.get("created_time")),
                    })
                    pbar.update(1)

                cursor = data.get("cursor")
                if not cursor:
                    break

                self.save_progress({"cursor": cursor})

        if all_trades:
            df = pd.DataFrame(all_trades)
            path = self.save_parquet(df, "kalshi_trades.parquet")
            print(f"Saved {len(df)} trades to {path}")
        else:
            print("No trades fetched.")

        self.save_progress({"cursor": None, "complete": True})


if __name__ == "__main__":
    KalshiTradesIndexer().run()
