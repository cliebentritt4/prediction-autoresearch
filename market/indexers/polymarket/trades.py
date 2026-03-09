"""
market/indexers/polymarket/trades.py — Polymarket on-chain trade collector.

Fetches trade history from the Polymarket CLOB API with pagination.
For deeper on-chain data, this would use Polygon RPC — currently
uses the REST API as a starting point. Writes to parquet.
"""

import requests
import pandas as pd
from tqdm import tqdm

from market.indexers.base import Indexer
from market.config import POLYMARKET_RPC_URL, DATA_DIR


class PolymarketTradesIndexer(Indexer):
    """Indexer for Polymarket trade history."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = f"{DATA_DIR}/polymarket/trades"
        super().__init__(output_dir)
        self.base_url = "https://clob.polymarket.com"
        self.rpc_url = POLYMARKET_RPC_URL

    def run(self) -> None:
        """Fetch Polymarket trades with pagination and save to parquet."""
        progress = self.load_progress()
        next_cursor = progress.get("next_cursor", None)

        all_trades = []

        print("Fetching Polymarket trades...")
        with tqdm(desc="Trades") as pbar:
            while True:
                params = {"limit": 500}
                if next_cursor:
                    params["next_cursor"] = next_cursor

                response = requests.get(
                    f"{self.base_url}/trades",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                trades = data if isinstance(data, list) else data.get("data", [])
                if not trades:
                    break

                for trade in trades:
                    # Map Polymarket trade fields to our schema
                    price = float(trade.get("price", 0))
                    all_trades.append({
                        "ticker": trade.get("asset_id", trade.get("condition_id", "")),
                        "yes_price": price * 100,
                        "no_price": (1 - price) * 100,
                        "count": int(trade.get("size", trade.get("amount", 1))),
                        "taker_side": trade.get("side", "buy"),
                        "created_time": pd.to_datetime(
                            trade.get("created_at", trade.get("timestamp"))
                        ),
                    })
                    pbar.update(1)

                next_cursor = data.get("next_cursor") if isinstance(data, dict) else None
                if not next_cursor:
                    break

                self.save_progress({"next_cursor": next_cursor})

        if all_trades:
            df = pd.DataFrame(all_trades)
            path = self.save_parquet(df, "polymarket_trades.parquet")
            print(f"Saved {len(df)} trades to {path}")
        else:
            print("No trades fetched.")

        self.save_progress({"next_cursor": None, "complete": True})


if __name__ == "__main__":
    PolymarketTradesIndexer().run()
