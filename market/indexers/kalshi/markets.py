"""
market/indexers/kalshi/markets.py — Kalshi market metadata collector.

Fetches market metadata from the Kalshi API with RSA-PSS request signing
and cursor-based pagination. Writes to parquet.
"""

import time
import base64
import requests
import pandas as pd
from tqdm import tqdm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

from market.indexers.base import Indexer
from market.config import (
    KALSHI_API_KEY_ID,
    KALSHI_API_KEY_FILE,
    KALSHI_API_BASE_URL,
    DATA_DIR,
)


def _load_private_key(key_file: str):
    """Load RSA private key from PEM file."""
    with open(key_file, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )


def _sign_request(private_key, timestamp_ms: str, method: str, path: str) -> str:
    """
    Sign a request per Kalshi API spec.
    Signs: timestamp_ms + method + path (without query params)
    """
    path_clean = path.split("?")[0]
    message = (timestamp_ms + method + path_clean).encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def _kalshi_headers(private_key, key_id: str, method: str, path: str) -> dict:
    """Build signed headers for a Kalshi API request."""
    timestamp_ms = str(int(time.time() * 1000))
    signature = _sign_request(private_key, timestamp_ms, method, path)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json",
    }


class KalshiMarketsIndexer(Indexer):
    """Indexer for Kalshi market metadata."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = f"{DATA_DIR}/kalshi/markets"
        super().__init__(output_dir)
        self.key_id = KALSHI_API_KEY_ID
        self.base_url = KALSHI_API_BASE_URL
        self.private_key = _load_private_key(KALSHI_API_KEY_FILE)

    def run(self) -> None:
        """Fetch all Kalshi markets with pagination and save to parquet."""
        progress = self.load_progress()
        cursor = progress.get("cursor", None)
        all_markets = []

        print("Fetching Kalshi markets...")
        with tqdm(desc="Markets") as pbar:
            while True:
                path = "/trade-api/v2/markets"
                params = {"limit": 200}
                if cursor:
                    params["cursor"] = cursor

                headers = _kalshi_headers(
                    self.private_key, self.key_id, "GET", path
                )

                response = requests.get(
                    f"{self.base_url}{path}",
                    headers=headers,
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                markets = data.get("markets", [])
                if not markets:
                    break

                for m in markets:
                    all_markets.append({
                        "ticker": m.get("ticker"),
                        "title": m.get("title"),
                        "category": m.get("category", ""),
                        "status": m.get("status"),
                        "yes_price": m.get("yes_price", 0),
                        "no_price": m.get("no_price", 0),
                        "volume": m.get("volume", 0),
                        "open_time": pd.to_datetime(m.get("open_time")),
                        "close_time": pd.to_datetime(m.get("close_time")),
                        "result": m.get("result"),
                    })
                    pbar.update(1)

                cursor = data.get("cursor")
                if not cursor:
                    break

                self.save_progress({"cursor": cursor})

        if all_markets:
            df = pd.DataFrame(all_markets)
            out = self.save_parquet(df, "kalshi_markets.parquet")
            print(f"Saved {len(df)} markets to {out}")
        else:
            print("No markets fetched.")

        self.save_progress({"cursor": None, "complete": True})


if __name__ == "__main__":
    KalshiMarketsIndexer().run()
