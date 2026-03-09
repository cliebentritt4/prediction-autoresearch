"""
market/indexers/kalshi/markets.py — Kalshi market metadata collector.

Fetches market metadata from the Kalshi public API with cursor-based
pagination, rate limiting (20 req/sec Basic tier), batch saves, and resume.
"""

import time
import requests
import pandas as pd
from tqdm import tqdm

from market.indexers.base import Indexer
from market.config import KALSHI_API_BASE_URL, DATA_DIR

# Kalshi Basic tier: 20 reads/sec. Stay under at 10 req/sec for safety.
REQUEST_DELAY = 0.1  # 100ms between requests = 10 req/sec
# Save to parquet every N markets
BATCH_SAVE_SIZE = 50_000
# Wait this many seconds on 429 before retrying
RATE_LIMIT_WAIT = 60
MAX_RETRIES = 5


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
        batch_num = progress.get("batch_num", 0)
        total_saved = progress.get("total_saved", 0)
        current_batch = []

        if cursor:
            print(f"Resuming from checkpoint (batch {batch_num}, {total_saved} already saved)...")

        print("Fetching Kalshi markets (rate limited to ~10 req/sec)...")
        with tqdm(desc="Markets", initial=total_saved) as pbar:
            while True:
                params = {"limit": 200}
                if cursor:
                    params["cursor"] = cursor

                # Rate limit: wait between requests
                time.sleep(REQUEST_DELAY)

                # Request with retry on rate limit
                response = None
                for attempt in range(MAX_RETRIES):
                    try:
                        response = requests.get(
                            f"{self.base_url}/trade-api/v2/markets",
                            params=params,
                        )
                        if response.status_code == 429:
                            wait = RATE_LIMIT_WAIT * (attempt + 1)
                            print(f"\nRate limited. Waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
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
                            self.save_progress({
                                "cursor": cursor,
                                "batch_num": batch_num,
                                "total_saved": total_saved,
                            })
                            print(f"Progress saved. Run again to resume. Total: {total_saved}")
                            return
                        wait = RATE_LIMIT_WAIT
                        print(f"\nError: {e}. Retrying in {wait}s...")
                        time.sleep(wait)

                if response is None:
                    break

                data = response.json()
                markets = data.get("markets", [])
                if not markets:
                    break

                for m in markets:
                    yes_price = m.get("yes_bid", m.get("last_price", 0))
                    no_price = m.get("no_bid", 100 - yes_price)

                    current_batch.append({
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

                # Save batch if big enough
                if len(current_batch) >= BATCH_SAVE_SIZE:
                    self._save_batch(current_batch, batch_num)
                    total_saved += len(current_batch)
                    batch_num += 1
                    current_batch = []

                cursor = data.get("cursor")
                self.save_progress({
                    "cursor": cursor,
                    "batch_num": batch_num,
                    "total_saved": total_saved,
                })

                if not cursor:
                    break

        # Save remaining
        if current_batch:
            self._save_batch(current_batch, batch_num)
            total_saved += len(current_batch)
            batch_num += 1

        self.save_progress({
            "cursor": None,
            "complete": True,
            "total_saved": total_saved,
            "batch_num": batch_num,
        })
        print(f"\nDone. Total markets saved: {total_saved} across {batch_num} file(s).")

    def _save_batch(self, records: list, batch_num: int):
        """Save a batch of records to a numbered parquet file."""
        df = pd.DataFrame(records)
        filename = f"kalshi_markets_{batch_num:04d}.parquet"
        path = self.save_parquet(df, filename)
        print(f"\n  Saved batch {batch_num}: {len(records)} markets → {path}")


if __name__ == "__main__":
    KalshiMarketsIndexer().run()
