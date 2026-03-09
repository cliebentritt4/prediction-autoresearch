"""
market/indexers/kalshi/trades.py — Kalshi trade history collector.

Fetches trade history from the Kalshi public API with cursor-based
pagination. No authentication required for public trade data.
"""

import requests
import pandas as pd
from tqdm import tqdm

from market.indexers.base import Indexer
from market.config import KALSHI_API_BASE_URL, DATA_DIR


class KalshiTradesIndexer(Indexer):
    """Indexer for Kalshi trade history."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = f"{DATA_DIR}/kalshi/trades"
        super().__init__(output_dir)
        self.base_url = KALSHI_API_BASE_URL

    def run(self) -> None:
        """Fetch all Kalshi trades with pagination and save to parquet."""
        progress = self.load_progress()
        cursor = progress.get("cursor", None)
        all_trades = []

        print("Fetching Kalshi trades...")
        with tqdm(desc="Trades") as pbar:
            while True:
                params = {"limit": 1000}
                if cursor:
                    params["cursor"] = cursor

                response = requests.get(
                    f"{self.base_url}/trade-api/v2/markets/trades",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                trades = data.get("trades", [])
                if not trades:
                    break

                for t in trades:
                    yes_price = t.get("yes_price", t.get("price", 0))
                    all_trades.append({
                        "ticker": t.get("ticker", ""),
                        "yes_price": float(yes_price),
                        "no_price": float(100 - yes_price),
                        "count": int(t.get("count", t.get("volume", 1))),
                        "taker_side": t.get("taker_side", t.get("side", "")),
                        "created_time": pd.to_datetime(t.get("created_time")),
                    })
                    pbar.update(1)

                cursor = data.get("cursor")
                if not cursor:
                    break

                self.save_progress({"cursor": cursor, "count": len(all_trades)})

        if all_trades:
            df = pd.DataFrame(all_trades)
            out = self.save_parquet(df, "kalshi_trades.parquet")
            print(f"Saved {len(df)} trades to {out}")
        else:
            print("No trades fetched.")

        self.save_progress({"cursor": None, "complete": True, "total": len(all_trades)})


if __name__ == "__main__":
    KalshiTradesIndexer().run()
