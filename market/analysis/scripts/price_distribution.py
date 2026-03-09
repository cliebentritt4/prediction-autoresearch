"""
market/analysis/scripts/price_distribution.py — Price distribution histogram.

Analyzes the distribution of yes_price across all markets,
broken down by platform (Kalshi vs Polymarket).
"""

import duckdb
import matplotlib.pyplot as plt
import numpy as np

from market.analysis.base import Analysis
from market.config import KALSHI_TRADES_GLOB, POLYMARKET_TRADES_GLOB


class PriceDistributionAnalysis(Analysis):
    """Histogram of yes_price distribution across platforms."""

    name = "price_distribution"

    def run(self) -> None:
        with self.progress():
            con = duckdb.connect(":memory:")

            kalshi_prices = self._load_prices(con, KALSHI_TRADES_GLOB, "kalshi")
            poly_prices = self._load_prices(con, POLYMARKET_TRADES_GLOB, "polymarket")
            con.close()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

            if len(kalshi_prices) > 0:
                axes[0].hist(kalshi_prices, bins=50, color="#2196F3", alpha=0.8, edgecolor="white")
            axes[0].set_title("Kalshi — Yes Price Distribution")
            axes[0].set_xlabel("Yes Price (cents)")
            axes[0].set_ylabel("Trade Count")

            if len(poly_prices) > 0:
                axes[1].hist(poly_prices, bins=50, color="#FF9800", alpha=0.8, edgecolor="white")
            axes[1].set_title("Polymarket — Yes Price Distribution")
            axes[1].set_xlabel("Yes Price (cents)")

            fig.suptitle("Price Distribution Across Platforms", fontsize=14)
            plt.tight_layout()
            self.save(fig, "price_distribution.png")
            plt.close(fig)

    def _load_prices(self, con, glob_pattern: str, label: str) -> np.ndarray:
        try:
            df = con.execute(
                f"SELECT yes_price FROM read_parquet('{glob_pattern}')"
            ).fetchdf()
            return df["yes_price"].dropna().values
        except Exception:
            print(f"  Warning: no data for {label}")
            return np.array([])


if __name__ == "__main__":
    PriceDistributionAnalysis().run()
