"""
market/indexers/kalshi/markets.py — Kalshi market metadata collector.

Fetches market metadata from the Kalshi API with pagination,
writes to parquet. Ported from prediction-market-analysis.
"""

import requests
import pandas as pd
from tqdm import tqdm

from market.indexers.base import Indexer
from market.config import KALSHI_API_KEY, DATA_DIR


class KalshiMarketsIndexer(Indexer):
    """Indexer for Kalshi market metadata."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = f"{DATA_DIR}/kalshi/markets"
        super().__init__(output_dir)
        self.api_key = KALSHI_API_KEY
        self.base_url = "https://api.kalshi.com/trade-api/v2"

    def run(self) -> None:
        """Fetch all Kalshi markets with pagination and save to parquet."""
        progress = self.load_progress()
        cursor = progress.get("cursor", None)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        all_markets = []

        print("Fetching Kalshi markets...")
        with tqdm(desc="Markets") as pbar:
            while True:
                params = {"limit": 200}
                if cursor:
                    params["cursor"] = cursor

                response = requests.get(
                    f"{self.base_url}/markets",
                    headers=headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                markets = data.get("markets", [])
                if not markets:
                    break

                for market in markets:
                    all_markets.append({
                        "ticker": market.get("ticker"),
                        "title": market.get("title"),
                        "category": market.get("category", ""),
                        "status": market.get("status"),
                        "yes_price": market.get("yes_price", 0),
                        "no_price": market.get("no_price", 0),
                        "volume": market.get("volume", 0),
                        "open_time": pd.to_datetime(market.get("open_time")),
                        "close_time": pd.to_datetime(market.get("close_time")),
                        "result": market.get("result"),
                    })
                    pbar.update(1)

                cursor = data.get("cursor")
                if not cursor:
                    break

                self.save_progress({"cursor": cursor})

        if all_markets:
            df = pd.DataFrame(all_markets)
            path = self.save_parquet(df, "kalshi_markets.parquet")
            print(f"Saved {len(df)} markets to {path}")
        else:
            print("No markets fetched.")

        self.save_progress({"cursor": None, "complete": True})


if __name__ == "__main__":
    KalshiMarketsIndexer().run()
