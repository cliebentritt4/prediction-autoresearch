"""
market/indexers/polymarket/trades.py — Polymarket trade history collector.
Uses the public Data API (https://data-api.polymarket.com/trades)
to fetch trades per market conditionId. This bypasses the 10K offset
cap by querying each market individually.
Also fetches price history from the CLOB API
(https://clob.polymarket.com/prices-history) for pre-bucketed time series.
Requires: Polymarket markets already collected (run markets indexer first).
"""
import time
import requests
import pandas as pd
import duckdb
from pathlib import Path
from tqdm import tqdm
from market.indexers.base import Indexer
from market.config import DATA_DIR

DATA_API_BASE = "https://data-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
REQUEST_DELAY = 0.1  # ~10 req/sec
BATCH_SAVE_SIZE = 100_000
RATE_LIMIT_WAIT = 30
MAX_RETRIES = 3


class PolymarketTradesIndexer(Indexer):
    """Indexer for Polymarket trade history via Data API."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = f"{DATA_DIR}/polymarket/trades"
        super().__init__(output_dir)

    def _get_market_ids(self) -> list[str]:
        """Load unique market conditionIds/tickers from collected parquet data."""
        markets_dir = Path(DATA_DIR) / "polymarket" / "markets"
        if not markets_dir.exists():
            raise FileNotFoundError(
                "No Polymarket market data found. Run markets indexer first."
            )

        con = duckdb.connect(":memory:")
        df = con.execute(f"""
            SELECT DISTINCT ticker
            FROM read_parquet('{markets_dir}/*.parquet')
            WHERE ticker IS NOT NULL AND ticker != ''
        """).fetchdf()
        con.close()
        return df["ticker"].tolist()

    def _fetch_trades_for_market(self, condition_id: str) -> list[dict]:
        """Fetch all trades for a single market from the Data API."""
        all_trades = []
        offset = 0

        while True:
            time.sleep(REQUEST_DELAY)

            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.get(
                        f"{DATA_API_BASE}/trades",
                        params={
                            "market": condition_id,
                            "limit": 10000,
                            "offset": offset,
                        },
                    )
                    if response.status_code == 429:
                        time.sleep(RATE_LIMIT_WAIT * (attempt + 1))
                        continue
                    if response.status_code in (401, 404):
                        return all_trades
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException:
                    if attempt == MAX_RETRIES - 1:
                        return all_trades
                    time.sleep(RATE_LIMIT_WAIT)
            else:
                break

            trades = response.json()
            if not trades:
                break

            for t in trades:
                price = float(t.get("price", 0))
                all_trades.append({
                    "ticker": condition_id,
                    "yes_price": price * 100,
                    "no_price": (1 - price) * 100,
                    "count": int(t.get("size", 1)),
                    "taker_side": t.get("side", "").lower(),
                    "created_time": pd.to_datetime(
                        t.get("timestamp"), unit="s", utc=True
                    ) if t.get("timestamp") else None,
                })

            if len(trades) < 10000:
                break
            offset += len(trades)

        return all_trades

    def run(self) -> None:
        """Fetch trades for all markets and save to parquet."""
        progress = self.load_progress()
        completed_markets = set(progress.get("completed_markets", []))
        batch_num = progress.get("batch_num", 0)
        total_saved = progress.get("total_saved", 0)
        current_batch = []

        market_ids = self._get_market_ids()
        remaining = [m for m in market_ids if m not in completed_markets]

        print(f"Fetching trades for {len(remaining)} markets "
              f"({len(completed_markets)} already done)...")

        with tqdm(total=len(market_ids), initial=len(completed_markets),
                  desc="Markets processed") as pbar:
            for condition_id in remaining:
                trades = self._fetch_trades_for_market(condition_id)
                current_batch.extend(trades)
                completed_markets.add(condition_id)
                pbar.update(1)

                # Save batch if big enough
                if len(current_batch) >= BATCH_SAVE_SIZE:
                    self._save_batch(current_batch, batch_num)
                    total_saved += len(current_batch)
                    batch_num += 1
                    current_batch = []
                    self.save_progress({
                        "completed_markets": list(completed_markets),
                        "batch_num": batch_num,
                        "total_saved": total_saved,
                    })

                # Save progress periodically (every 100 markets)
                if len(completed_markets) % 100 == 0:
                    self.save_progress({
                        "completed_markets": list(completed_markets),
                        "batch_num": batch_num,
                        "total_saved": total_saved + len(current_batch),
                    })

        # Save remaining
        if current_batch:
            self._save_batch(current_batch, batch_num)
            total_saved += len(current_batch)
            batch_num += 1

        self.save_progress({
            "complete": True,
            "completed_markets": list(completed_markets),
            "total_saved": total_saved,
            "batch_num": batch_num,
        })
        print(f"\nDone. Total trades: {total_saved} across {batch_num} file(s).")

    def _save_batch(self, records: list, batch_num: int):
        df = pd.DataFrame(records)
        filename = f"polymarket_trades_{batch_num:04d}.parquet"
        path = self.save_parquet(df, filename)
        print(f"\n  Saved batch {batch_num}: {len(records)} trades → {path}")


if __name__ == "__main__":
    PolymarketTradesIndexer().run()
