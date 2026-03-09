"""
market/indexers/polymarket/markets.py — Polymarket market metadata collector.

Fetches market metadata from the Polymarket CLOB API with cursor-based
pagination, batch saves, and rate limiting.

Rate limits (from docs):
  - CLOB general: 9000 req / 10s
  - We use ~5 req/sec to be conservative
"""

import time
import requests
import pandas as pd
from tqdm import tqdm

from market.indexers.base import Indexer
from market.config import DATA_DIR

REQUEST_DELAY = 0.2  # 5 req/sec (CLOB allows 900/s)
BATCH_SAVE_SIZE = 50_000
RATE_LIMIT_WAIT = 30
MAX_RETRIES = 5
CLOB_BASE_URL = "https://clob.polymarket.com"


class PolymarketMarketsIndexer(Indexer):
    """Indexer for Polymarket market metadata."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = f"{DATA_DIR}/polymarket/markets"
        super().__init__(output_dir)

    def run(self) -> None:
        """Fetch all Polymarket markets with pagination and save to parquet."""
        progress = self.load_progress()
        next_cursor = progress.get("next_cursor", None)
        batch_num = progress.get("batch_num", 0)
        total_saved = progress.get("total_saved", 0)
        current_batch = []

        if next_cursor:
            print(
                f"Resuming from checkpoint (batch {batch_num}, {total_saved} already saved)..."
            )

        print("Fetching Polymarket markets...")
        with tqdm(desc="Markets", initial=total_saved) as pbar:
            while True:
                params = {"limit": 100}
                if next_cursor:
                    params["next_cursor"] = next_cursor

                time.sleep(REQUEST_DELAY)

                response = None
                for attempt in range(MAX_RETRIES):
                    try:
                        response = requests.get(
                            f"{CLOB_BASE_URL}/markets",
                            params=params,
                        )
                        if response.status_code == 429:
                            wait = RATE_LIMIT_WAIT * (attempt + 1)
                            print(f"\nRate limited. Waiting {wait}s...")
                            time.sleep(wait)
                            continue
                        if response.status_code == 400:
                            # Bad request usually means invalid cursor = end of data
                            print(f"\nReached end of pagination (400 response).")
                            response = None
                            break
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
                                    "next_cursor": next_cursor,
                                    "batch_num": batch_num,
                                    "total_saved": total_saved,
                                }
                            )
                            print(
                                f"Progress saved. Run again to resume. Total: {total_saved}"
                            )
                            return
                        time.sleep(RATE_LIMIT_WAIT)

                if response is None:
                    break

                try:
                    data = response.json()
                except Exception:
                    print(f"\nFailed to parse JSON response. Stopping.")
                    break

                # Polymarket returns either a list or {"data": [...], "next_cursor": "..."}
                if isinstance(data, list):
                    markets = data
                    next_cursor = None
                elif isinstance(data, dict):
                    markets = data.get("data", data.get("markets", []))
                    next_cursor = data.get("next_cursor", None)
                else:
                    break

                if not markets:
                    break

                for m in markets:
                    # Map Polymarket fields to our schema
                    tokens = m.get("tokens", [{}])
                    yes_token = tokens[0] if tokens else {}
                    price = float(yes_token.get("price", 0))

                    ticker = m.get(
                        "condition_id", m.get("question_id", m.get("id", ""))
                    )

                    current_batch.append(
                        {
                            "ticker": str(ticker),
                            "title": m.get("question", m.get("title", "")),
                            "category": m.get("category", ""),
                            "status": "resolved" if m.get("resolved") else "active",
                            "yes_price": price * 100,
                            "no_price": (1 - price) * 100,
                            "volume": int(
                                float(m.get("volume", m.get("volumeNum", 0)))
                            ),
                            "open_time": pd.to_datetime(
                                m.get("created_at", m.get("startDate"))
                            ),
                            "close_time": pd.to_datetime(
                                m.get("end_date_iso", m.get("endDate"))
                            ),
                            "result": m.get("outcome", None),
                        }
                    )
                    pbar.update(1)

                if len(current_batch) >= BATCH_SAVE_SIZE:
                    self._save_batch(current_batch, batch_num)
                    total_saved += len(current_batch)
                    batch_num += 1
                    current_batch = []

                self.save_progress(
                    {
                        "next_cursor": next_cursor,
                        "batch_num": batch_num,
                        "total_saved": total_saved,
                    }
                )

                # End of pagination
                if not next_cursor or next_cursor == "LTE=":
                    break

        # Save remaining
        if current_batch:
            self._save_batch(current_batch, batch_num)
            total_saved += len(current_batch)
            batch_num += 1

        self.save_progress(
            {
                "next_cursor": None,
                "complete": True,
                "total_saved": total_saved,
                "batch_num": batch_num,
            }
        )
        print(f"\nDone. Total markets saved: {total_saved} across {batch_num} file(s).")

    def _save_batch(self, records: list, batch_num: int):
        df = pd.DataFrame(records)
        filename = f"polymarket_markets_{batch_num:04d}.parquet"
        path = self.save_parquet(df, filename)
        print(f"\n  Saved batch {batch_num}: {len(records)} markets → {path}")


if __name__ == "__main__":
    PolymarketMarketsIndexer().run()
