"""
market/indexers/kalshi/trades.py — Kalshi trade history collector.

Fetches trade history from the Kalshi public API with cursor-based
pagination, rate limiting (20 req/sec Basic tier), batch saves, and resume.
"""

import time
import requests
import pandas as pd
from tqdm import tqdm

from market.indexers.base import Indexer
from market.config import KALSHI_API_BASE_URL, DATA_DIR

REQUEST_DELAY = 0.1  # 10 req/sec (Basic tier allows 20)
BATCH_SAVE_SIZE = 100_000  # Trades are smaller rows, save less often
RATE_LIMIT_WAIT = 60
MAX_RETRIES = 5


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
        batch_num = progress.get("batch_num", 0)
        total_saved = progress.get("total_saved", 0)
        current_batch = []

        if cursor:
            print(
                f"Resuming from checkpoint (batch {batch_num}, {total_saved} already saved)..."
            )

        print("Fetching Kalshi trades (rate limited to ~10 req/sec)...")
        with tqdm(desc="Trades", initial=total_saved) as pbar:
            while True:
                params = {"limit": 1000}
                if cursor:
                    params["cursor"] = cursor

                time.sleep(REQUEST_DELAY)

                response = None
                for attempt in range(MAX_RETRIES):
                    try:
                        response = requests.get(
                            f"{self.base_url}/trade-api/v2/markets/trades",
                            params=params,
                        )
                        if response.status_code == 429:
                            wait = RATE_LIMIT_WAIT * (attempt + 1)
                            print(
                                f"\nRate limited. Waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})..."
                            )
                            time.sleep(wait)
                            continue
                        response.raise_for_status()
                        break
                    except requests.exceptions.RequestException as e:
                        if attempt == MAX_RETRIES - 1:
                            print(f"\nFailed after {MAX_RETRIES} retries: {e}")
                            if current_batch:
                                self._save_batch(current_batch, batch_num)
                                total_saved += len(current_batch)
                                batch_num += 1
                            self.save_progress(
                                {
                                    "cursor": cursor,
                                    "batch_num": batch_num,
                                    "total_saved": total_saved,
                                }
                            )
                            print(
                                f"Progress saved. Run again to resume. Total: {total_saved}"
                            )
                            return
                        wait = RATE_LIMIT_WAIT
                        print(f"\nError: {e}. Retrying in {wait}s...")
                        time.sleep(wait)

                if response is None:
                    break

                data = response.json()
                trades = data.get("trades", [])
                if not trades:
                    break

                for t in trades:
                    yes_price = t.get("yes_price", t.get("price", 0))
                    current_batch.append(
                        {
                            "ticker": t.get("ticker", ""),
                            "yes_price": float(yes_price),
                            "no_price": float(100 - yes_price),
                            "count": int(t.get("count", t.get("volume", 1))),
                            "taker_side": t.get("taker_side", t.get("side", "")),
                            "created_time": pd.to_datetime(t.get("created_time")),
                        }
                    )
                    pbar.update(1)

                if len(current_batch) >= BATCH_SAVE_SIZE:
                    self._save_batch(current_batch, batch_num)
                    total_saved += len(current_batch)
                    batch_num += 1
                    current_batch = []

                cursor = data.get("cursor")
                self.save_progress(
                    {
                        "cursor": cursor,
                        "batch_num": batch_num,
                        "total_saved": total_saved,
                    }
                )

                if not cursor:
                    break

        if current_batch:
            self._save_batch(current_batch, batch_num)
            total_saved += len(current_batch)
            batch_num += 1

        self.save_progress(
            {
                "cursor": None,
                "complete": True,
                "total_saved": total_saved,
                "batch_num": batch_num,
            }
        )
        print(f"\nDone. Total trades saved: {total_saved} across {batch_num} file(s).")

    def _save_batch(self, records: list, batch_num: int):
        """Save a batch of records to a numbered parquet file."""
        df = pd.DataFrame(records)
        filename = f"kalshi_trades_{batch_num:04d}.parquet"
        path = self.save_parquet(df, filename)
        print(f"\n  Saved batch {batch_num}: {len(records)} trades → {path}")


if __name__ == "__main__":
    KalshiTradesIndexer().run()
