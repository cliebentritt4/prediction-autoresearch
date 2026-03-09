"""
market/indexers/base.py — Abstract Indexer base class.

All data collection indexers inherit from this. Provides:
- Parquet write helpers
- Progress checkpoint save/load (for incremental indexing)
- Output directory management
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class Indexer(ABC):
    """Abstract base for data collection indexers."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._progress_file = self.output_dir / ".progress.json"

    @abstractmethod
    def run(self) -> None:
        """Execute the indexing job. Subclasses implement this."""
        ...

    def save_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        """Write a DataFrame to a parquet file in output_dir."""
        path = self.output_dir / filename
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def load_progress(self) -> dict:
        """Load checkpoint state for incremental indexing."""
        if self._progress_file.exists():
            return json.loads(self._progress_file.read_text())
        return {}

    def save_progress(self, state: dict) -> None:
        """Save checkpoint state."""
        self._progress_file.write_text(json.dumps(state, default=str))
