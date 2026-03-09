"""
market/indexers/polymarket/markets.py — Polymarket market metadata collector.

Fetches market metadata from the Polymarket CLOB API with pagination,
writes to parquet. Ported from prediction-market-analysis.
"""

import requests
import pandas as pd
from tqdm import tqdm

from market.indexers.base import Indexer
from market.config import DATA_DIR


class PolymarketMarketsIndexer(Indexer):
    """Indexer for Polymarket market metadata."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = f"{DATA_DIR}/polymarket/markets"
        super().__init__(output_dir)
        self.base_url = "https://clob.polymarket.com"

    def run(self) -> None:
        """Fetch all Polymarket markets with pagination and save to parquet."""
        progress = self.load_progress()
        next_cursor = progress.get("next_cursor", None)

        all_markets = []

        print("Fetching Polymarket markets...")
        with tqdm(desc="Markets") as pbar:
            while True:
                params = {"limit": 100}
                if next_cursor:
                    params["next_cursor"] = next_cursor

                response = requests.get(
                    f"{self.base_url}/markets",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                markets = data if isinstance(data, list) else data.get("data", [])
                if not markets:
                    break

                for market in markets:
                    # Polymarket uses condition_id or question_id as ticker
                    ticker = market.get("condition_id", market.get("question_id", ""))
                    tokens = market.get("tokens", [{}])
                    yes_token = tokens[0] if tokens else {}

                    all_markets.append({
                        "ticker": ticker,
                        "title": market.get("question", ""),
                        "category": market.get("category", ""),
                        "status": "resolved" if market.get("resolved") else "open",
                        "yes_price": float(yes_token.get("price", 0)) * 100,
                        "no_price": (1 - float(yes_token.get("price", 0))) * 100,
                        "volume": int(market.get("volume", 0)),
                        "open_time": pd.to_datetime(market.get("created_at")),
                        "close_time": pd.to_datetime(market.get("end_date_iso")),
                        "result": market.get("outcome", None),
                    })
                    pbar.update(1)

                next_cursor = data.get("next_cursor") if isinstance(data, dict) else None
                if not next_cursor:
                    break

                self.save_progress({"next_cursor": next_cursor})

        if all_markets:
            df = pd.DataFrame(all_markets)
            path = self.save_parquet(df, "polymarket_markets.parquet")
            print(f"Saved {len(df)} markets to {path}")
        else:
            print("No markets fetched.")

        self.save_progress({"next_cursor": None, "complete": True})


if __name__ == "__main__":
    PolymarketMarketsIndexer().run()
