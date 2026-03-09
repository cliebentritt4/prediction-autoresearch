"""
market/indexers/kalshi/markets.py — Kalshi market metadata collector.

Fetches market metadata from the Kalshi public API with cursor-based
pagination. No authentication required for market data endpoints.
"""

import requests
import pandas as pd
from tqdm import tqdm

from market.indexers.base import Indexer
from market.config import KALSHI_API_BASE_URL, DATA_DIR


class KalshiMarketsIndexer(Indexer):
    """Indexer for Kalshi market metadata."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = f"{DATA_DIR}/kalshi/markets"
        super().__init__(output_dir)
        self.base_url = KALSHI_API_BASE_URL

    def run(self) -> None:
        """Fetch all Kalshi markets with pagination and save to parquet."""
        progress = self.load_progress()
        cursor = progress.get("cursor", None)
        all_markets = []

        print("Fetching Kalshi markets...")
        with tqdm(desc="Markets") as pbar:
            while True:
                params = {"limit": 200}
                if cursor:
                    params["cursor"] = cursor

                response = requests.get(
                    f"{self.base_url}/trade-api/v2/markets",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                markets = data.get("markets", [])
                if not markets:
                    break

                for m in markets:
                    # Map API fields to our parquet schema
                    # yes_bid is the current yes price in cents
                    yes_price = m.get("yes_bid", m.get("last_price", 0))
                    no_price = m.get("no_bid", 100 - yes_price)

                    all_markets.append({
                        "ticker": m.get("ticker", ""),
                        "title": m.get("title", ""),
                        "category": m.get("event_ticker", "").split("-")[0] if m.get("event_ticker") else "",
                        "status": m.get("status", ""),
                        "yes_price": float(yes_price),
                        "no_price": float(no_price),
                        "volume": int(m.get("volume", 0)),
                        "open_time": pd.to_datetime(m.get("open_time")),
                        "close_time": pd.to_datetime(m.get("close_time", m.get("expiration_time"))),
                        "result": m.get("result") if m.get("result") else None,
                    })
                    pbar.update(1)

                cursor = data.get("cursor")
                if not cursor:
                    break

                # Save progress every page
                self.save_progress({"cursor": cursor, "count": len(all_markets)})

        if all_markets:
            df = pd.DataFrame(all_markets)
            out = self.save_parquet(df, "kalshi_markets.parquet")
            print(f"Saved {len(df)} markets to {out}")
        else:
            print("No markets fetched.")

        self.save_progress({"cursor": None, "complete": True, "total": len(all_markets)})


if __name__ == "__main__":
    KalshiMarketsIndexer().run()
