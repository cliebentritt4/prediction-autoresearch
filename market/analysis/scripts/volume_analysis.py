"""
market/analysis/scripts/volume_analysis.py — Volume over time analysis.

Plots daily trade volume across markets to visualize activity trends.
"""

import duckdb
import matplotlib.pyplot as plt

from market.analysis.base import Analysis
from market.config import KALSHI_TRADES_GLOB, POLYMARKET_TRADES_GLOB


class VolumeAnalysis(Analysis):
    """Daily trade volume over time by platform."""

    name = "volume_analysis"

    def run(self) -> None:
        with self.progress():
            con = duckdb.connect(":memory:")

            fig, ax = plt.subplots(figsize=(14, 5))

            for glob_pattern, label, color in [
                (KALSHI_TRADES_GLOB, "Kalshi", "#2196F3"),
                (POLYMARKET_TRADES_GLOB, "Polymarket", "#FF9800"),
            ]:
                try:
                    df = con.execute(f"""
                        SELECT
                            DATE_TRUNC('day', created_time) AS day,
                            SUM(count) AS daily_volume
                        FROM read_parquet('{glob_pattern}')
                        GROUP BY day
                        ORDER BY day
                    """).fetchdf()

                    if len(df) > 0:
                        ax.plot(
                            df["day"],
                            df["daily_volume"],
                            label=label,
                            color=color,
                            alpha=0.8,
                            linewidth=1.2,
                        )
                except Exception:
                    print(f"  Warning: no data for {label}")

            con.close()

            ax.set_title("Daily Trade Volume Over Time", fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Trade Volume")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            self.save(fig, "volume_analysis.png")
            plt.close(fig)


if __name__ == "__main__":
    VolumeAnalysis().run()
